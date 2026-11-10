"""Run the official swegemma harness on a fixed task subset against any OpenAI-compatible server.

The agent config keeps the competition alias (gemma-4-31b-it-qat-w4a16-ct) so it compiles exactly like a
submission; the model registry routes that alias to whatever model the server is actually serving.
"""
import argparse
import asyncio
import json
import os
import random
import shutil
import threading
import time
import urllib.request
from pathlib import Path

ALIAS = 'gemma-4-31b-it-qat-w4a16-ct'
SMOKE_SEED = 20260930
SMOKE_QUOTA = {'fastapi/fastapi': 8, 'Textualize/rich': 6, 'psf/requests': 2}

AGENT_YAML = f"""name: swe_agent
model: {ALIAS}
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
"""

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--base-url', default='http://127.0.0.1:9/v1')
p.add_argument('--model-id', default='none')
p.add_argument('--api-key', default='EMPTY')
p.add_argument('--tag', required=True)
p.add_argument('--out', required=True)
p.add_argument('--sampling', default='temperature: 0.2\n', help='contents of configs/sampling.yaml')
p.add_argument('--tasks', choices=['canary', 'smoke', 'all', 'file'], default='smoke')
p.add_argument('--task-ids-file', default=None, help='with --tasks file: JSON list, or dict + --task-ids-key')
p.add_argument('--task-ids-key', default=None)
p.add_argument('--stop-token-ids', default='',
               help='comma-separated token ids sent as extra_body.stop_token_ids (gpt-oss: 200012,200002 = <|call|>,<|return|>)')
p.add_argument('--drop-thinking-budget', action='store_true',
               help='keep thinking on but never send thinking_token_budget (servers without --reasoning-config)')
p.add_argument('--order-seed', type=int, default=None, help='shuffle task order (partial passes cover a random subset)')
p.add_argument('--order-ids-file', default=None,
               help='JSON list of instance ids: run tasks in this order (after --order-seed shuffling, listed ids first)')
p.add_argument('--control', choices=['none', 'gold', 'patches'], default=None,
               help='no model: verify an empty patch (none), the reference patch (gold), or saved patches')
p.add_argument('--patch-dir', default=None, help='with --control patches: dir of <instance_id>.patch files')
p.add_argument('--extra-wheels', default=None, help='dir of extra test wheels merged with the competition wheels')
p.add_argument('--py313-kit', default=None,
               help='dir with py313.tar.gz (+ wheels313/): run every task sandbox on Python 3.13 with only the '
                    'competition wheels + test extras + official shims, isolated from the host (like Dockerfile.public)')
p.add_argument('--find-links', action='append', default=[], help='extra wheel dirs for --py313-kit per-task installs')
p.add_argument('--time-verify', action='store_true',
               help='log the wall time of every verification pytest run to <out>/verify_times.jsonl')
p.add_argument('--uv-task-env', action='store_true',
               help='PR-Gym tasks (SWE-rebench rows with python/requirements/install): rebuild each sandbox venv with '
                    'uv for the task\'s Python version + frozen requirements + install command (needs internet)')
p.add_argument('--official-env', action='store_true',
               help='with --py313-kit: mirror the official Docker scorer instead of per-task pinned installs: every '
                    'competition wheel (newest per package, swegemma\'s own index) on the path, large wheels only '
                    'when the workspace imports them, `pip install --no-deps -e /workspace`, fast-path workspace paths')
p.add_argument('--repo-wheels', default=None,
               help='with --official-env: root of per-repository wheelhouses (<root>/<instance_id prefix>/*.whl, as '
                    'debmalya84/repograph-verified-swe-tasks-wheels); a task whose prefix has a directory there uses '
                    'it as its /wheels instead of the competition wheels')
p.add_argument('--raw-sandbox-paths', action='store_true',
               help='do NOT map /tmp/swegemma_sandbox_*/workspace back to /workspace in command output')
p.add_argument('--isolate', action='store_true',
               help='sandbox venv without host site-packages (closer to the official clean image)')
p.add_argument('--concurrency', type=int, default=8)
p.add_argument('--budget-s', type=float, required=True, help='hard wall-clock cap for the whole eval')
p.add_argument('--max-time-minutes', type=float, default=15)
p.add_argument('--max-tool-calls', type=int, default=60)
p.add_argument('--max-turns', type=int, default=100)
p.add_argument('--timeout-seconds', type=int, default=120)
p.add_argument('--bundle', default=None,
               help='run this submission bundle (agent.yaml root) instead of the official single-agent prompt; '
                    'per-task caps come from its eval_config.yaml unless --caps-from-args')
p.add_argument('--caps-from-args', action='store_true', help='with --bundle: use the --max-* / --timeout-* args')
p.add_argument('--compaction-interval', type=int, default=5,
               help='ADK events compaction every N invocations (5 = scripts/inference.py, the scorer); 0 = off')
p.add_argument('--time-scale', type=float, default=1.0,
               help='with --bundle caps: multiply max_time_minutes (concurrent proxy runs decode slower than '
                    'the sequential scorer)')
p.add_argument('--override-sampling', action='store_true',
               help="with --bundle: replace the bundle's configs/sampling.yaml with --sampling (e.g. teacher models)")
a = p.parse_args()

os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
DATA, OUT = Path(a.data), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f'[eval {time.time() - T0:7.1f}s] {msg}', flush=True)


# ---- fixed smoke subset (identical on every machine) ----
tasks = [json.loads(line) for line in open(DATA / 'tasks.jsonl')]
by_repo = {}
for t in sorted(tasks, key=lambda t: t['instance_id']):
    by_repo.setdefault(t['repo'], []).append(t)
rng = random.Random(SMOKE_SEED)
smoke_ids = [t['instance_id'] for repo, n in SMOKE_QUOTA.items()
             for t in rng.sample(by_repo.get(repo, []), min(n, len(by_repo.get(repo, []))))]
CANARY_IDS = ['rich_2943', 'rich_3043']  # small, verifiable, solved by the base model
if a.tasks == 'file':
    _ids = json.load(open(a.task_ids_file))
    run_ids = sorted(_ids[a.task_ids_key] if isinstance(_ids, dict) else _ids)
else:
    run_ids = {'canary': CANARY_IDS, 'smoke': smoke_ids}.get(a.tasks) or [t['instance_id'] for t in tasks]
tasks_path = DATA / 'tasks.jsonl'
if a.order_seed is not None:
    order = sorted(tasks, key=lambda t: t['instance_id'])
    random.Random(a.order_seed).shuffle(order)
    tasks_path = OUT / 'tasks_order.jsonl'
    tasks_path.write_text(''.join(json.dumps(t) + '\n' for t in order))
if a.order_ids_file:  # e.g. least-solved tasks first, so a pass cut by the deadline still covers the useful ones
    _base = [json.loads(line) for line in open(tasks_path)]
    _rank = {iid: k for k, iid in enumerate(json.load(open(a.order_ids_file)))}
    _base.sort(key=lambda t: _rank.get(t['instance_id'], len(_rank)))  # stable: unlisted keep their order, last
    tasks_path = OUT / 'tasks_order.jsonl'
    tasks_path.write_text(''.join(json.dumps(t) + '\n' for t in _base))
log(f'{a.tasks} tasks ({len(run_ids)}), order_seed={a.order_seed}, control={a.control}')

# ---- policy dir: official prompt, single agent (clean trajectories), no adapters ----
policy = OUT / 'policy'
shutil.rmtree(policy, ignore_errors=True)
if a.bundle:
    import yaml as _yaml
    shutil.copytree(a.bundle, policy)
    if a.override_sampling:
        (policy / 'configs' / 'sampling.yaml').write_text(a.sampling)
    if not a.caps_from_args:
        _ev = (_yaml.safe_load((policy / 'eval_config.yaml').read_text()) or {}).get('evaluation', {})
        a.timeout_seconds = int(_ev.get('timeout_seconds', a.timeout_seconds))
        a.max_tool_calls = int(_ev.get('max_tool_calls', a.max_tool_calls))
        a.max_turns = int(_ev.get('max_turns', a.max_turns))
        a.max_time_minutes = float(_ev.get('max_time_minutes', a.max_time_minutes)) * a.time_scale
    log(f'bundle {a.bundle}: caps {a.max_time_minutes:.1f} min / {a.max_tool_calls} tools / {a.max_turns} turns '
        f'/ {a.timeout_seconds} s (time scale {a.time_scale})')
else:
    (policy / 'prompts').mkdir(parents=True)
    (policy / 'configs').mkdir()
    shutil.copy(DATA / 'sample_submission' / 'prompts' / 'system.md', policy / 'prompts' / 'system.md')
    (policy / 'agent.yaml').write_text(AGENT_YAML)
    (policy / 'configs' / 'sampling.yaml').write_text(a.sampling)
(policy / 'eval_config.yaml').write_text(
    'evaluation:\n'
    f'  timeout_seconds: {a.timeout_seconds}\n'
    f'  max_tool_calls: {a.max_tool_calls}\n'
    f'  max_time_minutes: {a.max_time_minutes}\n'
    f'  max_turns: {a.max_turns}\n'
)

# ---- sandbox environment ----
wheels_dir = DATA / 'wheels'
if a.extra_wheels:
    wheels_dir = Path('/tmp/g4dev_wheels')
    wheels_dir.mkdir(parents=True, exist_ok=True)
    for src in [*sorted((DATA / 'wheels').glob('*.whl')), *sorted(Path(a.extra_wheels).glob('*.whl'))]:
        try:  # base and adapter harnesses start together and race on the same links
            os.symlink(src, wheels_dir / src.name)
        except FileExistsError:
            pass
    log(f'wheels: {len(list(wheels_dir.glob("*.whl")))} (competition + extra from {a.extra_wheels})')
    # The subprocess sandbox never injects wheels (that path is Docker-only): test deps come from the host
    # interpreter's site-packages. Install only extra test wheels whose distribution is not already present.
    import importlib.metadata as _md
    import subprocess as _sp
    import sys as _sys
    missing = []
    for w in sorted(Path(a.extra_wheels).glob('*.whl')):
        dist = w.name.split('-')[0]
        try:
            _md.version(dist)
        except _md.PackageNotFoundError:
            missing.append(str(w))
    if missing:
        r = _sp.run([_sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--no-index', *missing],
                    capture_output=True, text=True)
        log(f'installed {len(missing)} missing test wheels into host env (rc={r.returncode}): '
            f'{[Path(m).name.split("-")[0] for m in missing]} {r.stderr[-500:]}')

# ---- Python 3.13 sandbox (mirrors the official python:3.13-slim image + /wheels) ----
PY313 = None
if a.py313_kit:
    import glob as _glob
    import shutil as _shutil
    import subprocess as _sp
    import tarfile as _tarfile
    from packaging.utils import parse_wheel_filename
    kit = Path(a.py313_kit)
    py_root = Path('/tmp/py313')
    if not py_root.exists():
        with _tarfile.open(sorted(kit.glob('py313.tar.gz*'))[0]) as tf:  # stored as .bin so Kaggle won't unpack it
            tf.extractall(py_root)
    PY313 = sorted(_glob.glob(str(py_root / '*/bin/python3.13')))[0]
    # Per-task environments: each task's 3.13 venv installs the snapshot itself (-e) plus its own test
    # requirements, resolved offline from a wheelhouse built per pin-group (tasks span years of pins).
    find_links = [*a.find_links, str(DATA / 'wheels'), *sorted({str(w.parent) for w in kit.rglob('*.whl')})]
    shims = [DATA / 'docker' / n for n in ('imp.py', 'telnetlib.py') if (DATA / 'docker' / n).exists()]
    a.isolate = True  # no _host_env.pth: nothing from the Kaggle host leaks into task sandboxes

    if a.official_env:
        # Mirror the official Docker scorer. container_setup._ensure_container_site_packages unpacks every
        # competition wheel (deduplicated to the newest py3.13-compatible version per package) below 20 MB into
        # site-packages, and larger wheels only when /workspace imports them (transitively); sandbox/setup.py then
        # runs `pip install --no-deps -e /workspace` and the fast path. No requirement file or version pin is read.
        # The subprocess sandbox skips all of this (install_test_dependencies returns early), so do it here with
        # swegemma's own index and tar builder.
        import re as _re
        import threading as _threading
        from swegemma.harness.container_setup import _build_unpacked_wheels_tar, _inspect_wheels_dir
        _OFF = Path('/tmp/g4dev_official_sp')
        _OFF_LOCK = _threading.Lock()
        _base_w, _large_w, _mod2pkg, _pkgreq = _inspect_wheels_dir(DATA / 'wheels')
        _IDX = {'comp': (DATA / 'wheels', _base_w, _large_w, _mod2pkg, _pkgreq)}
        _RW_PREFIX, _RW_HEAD = {}, {}  # repo -> instance-id prefix; base_commit -> prefix
        if a.repo_wheels:
            for _line in open(DATA / 'tasks.jsonl'):
                if _line.strip():
                    _t = json.loads(_line)
                    _RW_PREFIX[_t['repo']] = _RW_HEAD[_t['base_commit']] = _t['instance_id'].rsplit('_', 1)[0]

        def _wheel_index(ws, repo):
            """(tag, wheels dir, base, large, mod2pkg, pkgreq) for this task: its repo's wheelhouse if it has one."""
            if a.repo_wheels:
                pre = _RW_PREFIX.get(repo)
                if pre is None:
                    head = _sp.run(['git', '-C', str(ws), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
                    pre = _RW_HEAD.get(head.strip())
                d = Path(a.repo_wheels) / pre if pre else None
                if d is not None and d.is_dir():
                    with _OFF_LOCK:
                        if pre not in _IDX:
                            _IDX[pre] = (d, *_inspect_wheels_dir(d))
                    return (pre, *_IDX[pre])
            return ('comp', *_IDX['comp'])

        def _unpacked(wheels, name):
            d = _OFF / name
            with _OFF_LOCK:
                if not (d / '.g4dev_done').exists():
                    tar = _build_unpacked_wheels_tar(wheels, f'g4dev_{name}', f'.g4dev_{name}')
                    d.mkdir(parents=True, exist_ok=True)
                    with _tarfile.open(tar) as tf:
                        tf.extractall(d)
                    (d / '.g4dev_done').write_text('ok\n')
                    tar.unlink(missing_ok=True)  # the extracted dir is the cache; the tar only doubled the disk use
            return d

        # Per-repo caches (--repo-wheels) are deleted once every sandbox that uses them is gone (sandboxes are removed
        # on stop). Without this the repograph CPU controls filled /tmp at task 141 of 221 (EROFS on every later task).
        _REFS = {}  # repo cache dir name -> site-packages dirs of the task venvs whose .pth points at it

        def _evict_unused():
            with _OFF_LOCK:
                for name, sps in list(_REFS.items()):
                    if not any(Path(x).exists() for x in sps):
                        shutil.rmtree(_OFF / name, ignore_errors=True)
                        _REFS.pop(name)

        _unpacked(_base_w, 'base')
        log(f'official env: {len(_base_w)} base wheels unpacked, {len(_large_w)} large wheels on demand '
            f'({sorted(_large_w)})')

        def _official_task_env(ws, py, repo):
            tag, wdir, base_w, large_w, mod2pkg, pkgreq = _wheel_index(ws, repo)
            pfx = '' if tag == 'comp' else tag + '_'
            mods = set()
            for f in ws.rglob('*.py'):
                if '.git' in f.parts:
                    continue
                try:
                    txt = f.read_text(errors='ignore')
                except OSError:
                    continue
                mods.update(m.group(1).lower() for m in _re.finditer(r'^\s*(?:import|from)\s+([a-zA-Z0-9_]+)', txt, _re.M))
            if repo:
                mods.add(repo.rsplit('/', 1)[-1].lower())
            needed, queue = set(), []
            for mod in mods:
                for pk in mod2pkg.get(mod, ()):
                    if pk not in needed:
                        needed.add(pk)
                        queue.append(pk)
            while queue:
                cur = queue.pop()
                for dep in pkgreq.get(cur, ()):
                    if dep not in needed:
                        needed.add(dep)
                        queue.append(dep)
                for cand in large_w:
                    if cand.startswith(cur + '_') and cand not in needed:
                        needed.add(cand)
                        queue.append(cand)
            large = sorted(p for p in needed if p in large_w)
            sp = next(Path(py).parent.parent.glob('lib/python3.13/site-packages'))
            if pfx:
                with _OFF_LOCK:
                    for nm in [pfx + 'base'] + [pfx + 'whl_' + p for p in large]:
                        _REFS.setdefault(nm, set()).add(str(sp))
                _evict_unused()
            dirs = [_unpacked(base_w, pfx + 'base')] + [_unpacked([large_w[p]], pfx + 'whl_' + p) for p in large]
            # addsitedir (not a bare path) so .pth files shipped inside wheels are honored, as in site-packages
            (sp / '00_g4dev_official.pth').write_text(
                ''.join(f'import site; site.addsitedir({str(d)!r})\n' for d in dirs))
            # Dockerfile.public pip-installs these into the image (PyPI); take them from the offline wheels
            fl = [x for d in find_links for x in ('--find-links', d)]
            r1 = _sp.run([py, '-m', 'pip', 'install', '-q', '--no-index', *fl, 'pytest', 'pytest-timeout==2.1.0',
                          'typer', 'pdm-backend', 'setuptools', 'wheel', 'poetry-core', 'hatchling', 'flit-core',
                          'editables'], capture_output=True, text=True, timeout=600)
            # sandbox/setup.py install_editable_package (`|| true`)
            r2 = _sp.run([py, '-m', 'pip', 'install', '-q', '--no-index', f'--find-links={wdir}',
                          '--no-build-isolation', '--no-deps', '-e', str(ws)], capture_output=True, text=True,
                         timeout=600)
            # sandbox/setup.py execute_fast_path: workspace and its top-level directories on sys.path
            wsr = ws.resolve()
            paths = [str(wsr)] + [str(s) for s in sorted(wsr.iterdir()) if s.is_dir() and not s.name.startswith('.')]
            (sp / 'workspace_paths.pth').write_text('\n'.join(paths) + '\n')
            with open(OUT / 'env_setup.jsonl', 'a') as fh:
                fh.write(json.dumps({'workspace': str(ws), 'repo': repo, 'wheels': tag, 'n_base': len(base_w), 'large': large,
                                     'tmp_free_gb': round(shutil.disk_usage('/tmp').free / 2**30, 1), 'repo_caches': len(_REFS),
                                     'tools_rc': r1.returncode, 'editable_rc': r2.returncode,
                                     'editable_err': (r2.stderr or '')[-300:]}) + '\n')

    class _Py313Venv:
        """Drop-in for the `venv` module inside swegemma.sandbox.subprocess: builds task venvs with Python 3.13."""

        @staticmethod
        def create(env_dir, system_site_packages=False, clear=False, symlinks=False, with_pip=False, **_):
            cmd = [PY313, '-m', 'venv', *(['--clear'] if clear else []),
                   '--symlinks' if symlinks else '--copies', str(env_dir)]
            _sp.run(cmd, check=True, capture_output=True, timeout=300)
            sp = next(Path(env_dir).glob('lib/python3.13/site-packages'))
            for shim in shims:  # the official image copies these into site-packages
                _shutil.copy(shim, sp / shim.name)


    import re as _re2
    import tomllib as _tomllib
    _SKIP_REQ = ('doc', 'translation', 'github')

    def _req_name(req):
        m = _re2.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', req)
        return m.group(1) if m else ''

    def _expand_req_file(rf, seen):
        rf = Path(rf).resolve()
        if rf in seen or not rf.exists():
            return []
        seen.add(rf)
        out = []
        for raw in rf.read_text(errors='replace').splitlines():
            st = raw.split(' #', 1)[0].strip()
            if not st or st.startswith('#'):
                continue
            if st.startswith(('-r', '--requirement')):
                nested = rf.parent / (st.split(None, 1)[1] if ' ' in st else st[2:]).strip()
                if any(x in nested.name for x in _SKIP_REQ) and 'test' not in nested.name:
                    continue  # docs-only include; docs-*tests* files carry test deps (fastapi: httpx)
                out += _expand_req_file(nested, seen)
            elif st.startswith(('-e', '--editable')):
                tgt = st.split(None, 1)[1] if ' ' in st else st[2:]
                ex = _re2.findall(r'\[([^\]]*)\]', tgt)
                if ex:
                    out.append(('extras', [e.strip() for e in ex[0].split(',') if e.strip()]))
            elif not st.startswith('-'):
                out.append(st)
        return out

    def _task_requirements(ws):
        items, extras, seen = [], [], set()
        for rf in sorted(ws.glob('requirements*.txt')):
            if any(x in rf.name for x in _SKIP_REQ):
                continue
            for it in _expand_req_file(rf, seen):
                (extras.extend(it[1]) if isinstance(it, tuple) else items.append(it))
        pp = ws / 'pyproject.toml'
        if pp.exists():
            try:
                _pj = _tomllib.loads(pp.read_text())
            except Exception:  # noqa: BLE001
                _pj = {}
            opt = _pj.get('project', {}).get('optional-dependencies', {})
            groups = _pj.get('dependency-groups', {})  # PEP 735: newer snapshots keep test deps here (no req files)

            def _group(name, depth=0):
                out = []
                for it in groups.get(name, []) if depth < 5 else []:
                    if isinstance(it, str):
                        out.append(it)
                    elif isinstance(it, dict) and it.get('include-group'):
                        out += _group(it['include-group'], depth + 1)
                return out
            for e in dict.fromkeys(extras):
                items += opt.get(e, [])
            for g in ('tests', 'test', 'testing', 'dev'):
                items += _group(g)
            if not items:  # no requirement files and no groups: fall back to the project's test-ish extras
                for e in ('all', 'standard', 'test', 'tests', 'testing', 'dev'):
                    items += opt.get(e, [])
        out, names = [], set()
        for q in items:
            n = (_req_name(q) or q).lower().replace('_', '-')
            if n not in names:
                names.add(n)
                out.append(q)
        return out

    import swegemma.harness.container_setup as _cs
    _orig_install = _cs.install_test_dependencies

    def _install_task_env(docker, container_id, repo='', *, fast_path=True, config=None):
        paths = getattr(docker, '_sandboxes', {}).get(container_id)
        if not paths:
            return _orig_install(docker, container_id, repo, fast_path=fast_path, config=config)
        ws, py = paths['workspace'], str(paths['venv'] / 'bin' / 'python3')
        if a.official_env:
            _official_task_env(ws, py, repo)
            if _sp.run([py, '-c', 'import pytest'], capture_output=True).returncode != 0:
                raise RuntimeError('pytest missing in official-env task venv')
            return _orig_install(docker, container_id, repo, fast_path=fast_path, config=config)
        fl = [x for d in find_links for x in ('--find-links', d)]
        pip = [py, '-m', 'pip', 'install', '-q', '--no-index', *fl, '--cache-dir', '/tmp/pipcache']
        # Requirement files with nested -r includes inlined and `-e .[extras]` expanded from pyproject (controls v5 bug:
        # the old line-by-line fallback skipped -r lines, so FastAPI never got httpx from requirements-docs-tests.txt).
        reqs = _task_requirements(ws)
        dropped, bulk_ok = [], False
        for _ in range(10):  # bulk install; drop each pin pip reports as unavailable offline and retry
            r = _sp.run([*pip, '-e', str(ws), *reqs], capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                bulk_ok = True
                break
            m = _re2.search(r'(?:satisfies the requirement|matching distribution found for) ([A-Za-z0-9][A-Za-z0-9._-]*)',
                            r.stdout + r.stderr)
            bad = [q for q in reqs if m and _req_name(q).lower().replace('_', '-') ==
                   m.group(1).lower().replace('_', '-')]
            if not bad:
                break
            dropped += bad
            reqs = [q for q in reqs if q not in bad]
        if not bulk_ok:  # conflicts: project first, then each requirement alone
            _sp.run([*pip, '-e', str(ws)], capture_output=True, text=True, timeout=600)
            for q in reqs:
                if _sp.run([*pip, q], capture_output=True, text=True, timeout=300).returncode != 0:
                    dropped.append(q)
        for q in dropped:  # an available version beats a missing module (pins are often stale dev tools)
            n = _req_name(q)
            if n and ';' not in q:
                _sp.run([*pip, n], capture_output=True, text=True, timeout=300)
        with open(OUT / 'env_setup.jsonl', 'a') as fh:
            fh.write(json.dumps({'workspace': str(ws), 'repo': repo, 'n_reqs': len(reqs), 'bulk_ok': bulk_ok,
                                 'dropped': dropped}) + '\n')
        r = _sp.run([*pip, 'pytest', 'pytest-timeout'], capture_output=True, text=True, timeout=300)
        if _sp.run([py, '-c', 'import pytest'], capture_output=True).returncode != 0:
            # fail loudly: a sandbox without pytest fails every test silently (controls v5: 0/129)
            raise RuntimeError(f'pytest missing in task venv; pip said: {(r.stdout + r.stderr)[-1500:]}')
        return _orig_install(docker, container_id, repo, fast_path=fast_path, config=config)

    # callers bind the function by name at import time, so patch every module that holds a reference
    import importlib
    for _name in ('swegemma.harness.container_setup', 'swegemma.harness', 'swegemma.evaluate',
                  'swegemma.harness.agent_runner', 'swegemma.harness.verification',
                  'swegemma.harness.sample_verification'):
        _mod = importlib.import_module(_name)
        if getattr(_mod, 'install_test_dependencies', None) is _orig_install:
            _mod.install_test_dependencies = _install_task_env
    import swegemma.sandbox.subprocess as _sbx
    _sbx.venv = _Py313Venv
    log(f'task sandboxes use {PY313}')

# ---- PR-Gym task environments (SWE-rebench rows): per-task uv venv, identified by the snapshot's HEAD commit ----
if a.uv_task_env:
    import re as _re3
    import subprocess as _sp3
    import shutil as _sh3
    import swegemma.harness.container_setup as _cs3
    _UV = os.environ.get('G4DEV_UV') or _sh3.which('uv') or 'uv'
    _ROWS = {}
    for _line in open(DATA / 'tasks.jsonl'):
        if _line.strip():
            _t = json.loads(_line)
            _ROWS[_t['base_commit']] = _t
    _orig_install3 = _cs3.install_test_dependencies
    a.isolate = True

    def _rc3(cmd, timeout, **kw):
        try:
            return _sp3.run(cmd, capture_output=True, text=True, timeout=timeout, **kw).returncode
        except _sp3.TimeoutExpired:
            return 124

    def _uv_task_env(docker, container_id, repo='', *, fast_path=True, config=None):
        paths = getattr(docker, '_sandboxes', {}).get(container_id)
        if not paths:
            return _orig_install3(docker, container_id, repo, fast_path=fast_path, config=config)
        ws, venv = Path(paths['workspace']), Path(paths['venv'])
        head = _sp3.run(['git', '-C', str(ws), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        t = _ROWS.get(head)
        if t is None:  # FreshSWE-style snapshots: one synthetic commit whose subject is `base <base_commit>`
            _subj = _sp3.run(['git', '-C', str(ws), 'log', '-1', '--format=%s'], capture_output=True, text=True).stdout
            _m = _re3.match(r'base ([0-9a-f]{40})', _subj.strip())
            if _m and _m.group(1) in _ROWS:
                head = _m.group(1)
                t = _ROWS[head]
        rec = {'workspace': str(ws), 'head': head, 'task': t and t['instance_id']}
        # The harness calls this hook once before the agent phase and once before verification, on the same sandbox:
        # build the env once (the project is installed editable, so later source edits are picked up).
        marker = venv / '.g4dev_env'
        if t and marker.exists() and marker.read_text() == head:
            rec['reused'] = True
            t = None
        if t:
            py = str(venv / 'bin' / 'python')
            uvpip = [_UV, 'pip', 'install', '-q', '--python', py]
            rec['venv_rc'] = _rc3([_UV, 'venv', '-q', '--clear', '--python', str(t.get('python') or '3.11'), str(venv)], 600)
            _t_env = time.time()
            # Frozen requirements from SWE-rebench always contain lines uv cannot parse: `-e git+...#egg=<project>` (the
            # project itself; installed below) and conda build paths (`pkg @ file:///croot/...`). Unsanitized, the bulk
            # install exits 2 and the old fallback ran one uv process per line (~50 per task): fine on an idle CPU box,
            # >350 s per task on the loaded TPU VM (gen-tpu-rb1 gate: 29/852 tasks in 30 min).
            lines = []
            for _l in (t.get('requirements') or '').splitlines():
                _s = _l.strip()
                if not _s or _s.startswith(('#', '-')):
                    continue
                if ' @ ' in _s and 'file://' in _s:
                    _s = _s.split(' @ ', 1)[0].strip()  # conda-built package: take the released one
                lines.append(_s)
            req = ws.parent / 'tmp' / 'g4dev_requirements.txt'
            req.parent.mkdir(parents=True, exist_ok=True)
            req.write_text('\n'.join(lines) + '\n')
            rec['frozen_rc'] = _rc3([*uvpip, '-r', str(req)], 600, cwd=ws)
            if rec['frozen_rc'] != 0:  # conflicting pins: the freeze is complete, so skip resolution
                rec['frozen_nodeps_rc'] = _rc3([*uvpip, '--no-deps', '-r', str(req)], 420, cwd=ws)
                if rec['frozen_nodeps_rc'] != 0:  # last resort, bounded: one line at a time for at most 4 minutes
                    _end, _bad = time.time() + 240, 0
                    for _s in lines:
                        if time.time() > _end:
                            break
                        _bad += _rc3([*uvpip, '--no-deps', _s], 60, cwd=ws) != 0
                    rec['line_fail'] = _bad
            inst = _re3.sub(r'\bpython -m pip install\b|\bpip3? install\b', ' '.join(uvpip),
                            str(t.get('install') or 'pip install -e .'))
            rec['install_rc'] = _rc3(inst, 900, shell=True, cwd=ws)
            _rc3([*uvpip, 'pytest', 'pytest-timeout'], 300, cwd=ws)
            rec['env_s'] = round(time.time() - _t_env, 1)
            if rec['venv_rc'] == 0 and rec['install_rc'] == 0:
                marker.write_text(head)
        with open(OUT / 'env_setup.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        return _orig_install3(docker, container_id, repo, fast_path=fast_path, config=config)

    import importlib as _il3
    for _name in ('swegemma.harness.container_setup', 'swegemma.harness', 'swegemma.evaluate',
                  'swegemma.harness.agent_runner', 'swegemma.harness.verification',
                  'swegemma.harness.sample_verification'):
        _mod = _il3.import_module(_name)
        if getattr(_mod, 'install_test_dependencies', None) is _orig_install3:
            _mod.install_test_dependencies = _uv_task_env
    log(f'uv task envs for {len(_ROWS)} tasks')

# ---- verification timing: the scorer runs this pytest with the bundle's timeout_seconds, so its duration matters ----
if a.time_verify:
    try:
        import inspect as _insp4
        import subprocess as _sp4
        import swegemma.sandbox.subprocess as _sbm4
        _HEAD2ID = {}
        for _line in open(DATA / 'tasks.jsonl'):
            if _line.strip():
                _t = json.loads(_line)
                _HEAD2ID[_t.get('base_commit')] = _t['instance_id']

        def _wrap_exec(cls):
            orig = cls.exec

            def exec_(self, sandbox_id, command, *args, **kw):
                t0, res = time.time(), None
                try:
                    res = orig(self, sandbox_id, command, *args, **kw)
                    return res
                finally:
                    try:
                        if '--junitxml' in str(command):
                            ws = (getattr(self, '_sandboxes', {}).get(sandbox_id) or {}).get('workspace')
                            head = _sp4.run(['git', '-C', str(ws), 'rev-parse', 'HEAD'], capture_output=True,
                                            text=True).stdout.strip() if ws else ''
                            with open(OUT / 'verify_times.jsonl', 'a') as fh:
                                fh.write(json.dumps({'task': _HEAD2ID.get(head), 'head': head,
                                                     'secs': round(time.time() - t0, 2),
                                                     'exit_code': getattr(res, 'exit_code', None),
                                                     'timeout': kw.get('timeout')}) + '\n')
                    except Exception:  # noqa: BLE001
                        pass
            cls.exec = exec_

        _n4 = 0
        for _c in list(vars(_sbm4).values()):
            if _insp4.isclass(_c) and _c.__module__ == _sbm4.__name__ and 'exec' in vars(_c):
                _wrap_exec(_c)
                _n4 += 1
        log(f'verify timing installed on {_n4} sandbox class(es)')
    except Exception as _e4:  # noqa: BLE001
        log(f'verify timing NOT installed: {type(_e4).__name__}: {_e4}')

# ---- model registry: competition alias -> served model ----
import litellm
from adk_submission import discover_adapters
from adk_submission.registry import ModelRegistry
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.sandbox.subprocess import SubprocessManager

STOP_IDS = [int(x) for x in a.stop_token_ids.split(',') if x.strip()]
if a.drop_thinking_budget or STOP_IDS:
    # (1) The harness merges thinking_config.thinking_budget=4096 into every agent and sends it as
    #     extra_body.thinking_token_budget, which vLLM rejects (400) unless the server has --reasoning-config.
    # (2) vLLM 0.19.1 chat completions builds stop_token_ids from the request only, dropping the harmony stop
    #     tokens it adds to its defaults for gpt-oss, so generation never stops at <|call|> (runaway tool calls).
    import adk_submission.builders.llm as _llm_builder
    _orig_apply = _llm_builder.apply_thinking_config_to_model

    def _apply_request_fixes(model, gen_config):
        out = _orig_apply(model, gen_config)
        args = getattr(out, '_additional_args', None)
        if not isinstance(args, dict):
            return out
        if out is model:  # never mutate the shared base model
            import copy
            out = copy.copy(model)
            args = out._additional_args = copy.deepcopy(args)
        extra = dict(args.get('extra_body') or {})
        if a.drop_thinking_budget:
            extra.pop('thinking_token_budget', None)
        if STOP_IDS:
            extra['stop_token_ids'] = STOP_IDS
        args['extra_body'] = extra
        return out
    _llm_builder.apply_thinking_config_to_model = _apply_request_fixes

if not a.raw_sandbox_paths:
    # The subprocess sandbox rewrites /workspace and /tmp in commands but not in their output, so agents see
    # /tmp/swegemma_sandbox_<id>/workspace/... paths the official Docker scorer never shows (and get confused).
    _sm_exec = SubprocessManager.exec

    def _exec_official_paths(self, sandbox_id, command, *, timeout=None):
        res = _sm_exec(self, sandbox_id, command, timeout=timeout)
        paths = getattr(self, '_sandboxes', {}).get(sandbox_id)
        if paths and res is not None:
            mapping = ((str(paths['venv'] / 'bin'), '/usr/local/bin'), (str(paths['workspace']), '/workspace'),
                       (str(paths['tmp']), '/tmp'))
            for attr in ('stdout', 'stderr'):
                text = getattr(res, attr, None)
                if isinstance(text, str) and text:
                    for host, official in mapping:
                        text = text.replace(host, official)
                    setattr(res, attr, text)
        return res
    SubprocessManager.exec = _exec_official_paths

if a.isolate:
    _sm_init = SubprocessManager.__init__

    def _isolated_init(self, *args, **kwargs):
        kwargs['system_site_packages'] = False
        _sm_init(self, *args, **kwargs)
    SubprocessManager.__init__ = _isolated_init

litellm.drop_params = True
try:
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    install_litellm_reasoning_patch()
except Exception as e:  # noqa: BLE001
    log(f'reasoning patch not installed: {e}')

if a.control in ('gold', 'patches'):
    from swegemma.evaluate import SessionTrace

    async def _fixed_patch(self, task, *args, **kwargs):
        if a.control == 'gold':
            return task.patch, None, SessionTrace()
        f = Path(a.patch_dir) / f'{task.instance_id}.patch'
        return (f.read_text() if f.exists() else ''), None, SessionTrace()
    Evaluator._run_agent_sandbox = _fixed_patch

models = ModelRegistry()
models.register(ALIAS, LiteLlm(model=f'openai/{a.model_id}', api_base=a.base_url, api_key=a.api_key))

# ---- server metrics sampler ----
METRIC_KEYS = ('vllm:generation_tokens_total', 'vllm:prompt_tokens_total', 'vllm:num_requests_running',
               'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')
stop = threading.Event()


def sample_metrics():
    url = a.base_url.rsplit('/v1', 1)[0] + '/metrics'
    with open(OUT / 'server_metrics.jsonl', 'w') as f:
        while not stop.is_set():
            row = {'t': round(time.time() - T0, 1)}
            try:
                body = urllib.request.urlopen(url, timeout=5).read().decode()
                for line in body.splitlines():
                    for k in METRIC_KEYS:
                        if line.startswith(k):
                            row[k] = row.get(k, 0.0) + float(line.rsplit(' ', 1)[1])
            except Exception as e:  # noqa: BLE001
                row['err'] = str(e)[:80]
            f.write(json.dumps(row) + '\n')
            f.flush()
            stop.wait(15)


threading.Thread(target=sample_metrics, daemon=True).start()

limits, gen_constraints = build_submission_limits()
cfg = EvalConfig(
    tasks_path=tasks_path,
    snapshots_dir=DATA / 'snapshots',
    results_dir=OUT / 'results',
    submission_dir=policy,
    models=models,
    sandbox='subprocess',
    timeout_seconds=a.timeout_seconds,
    max_time_minutes=a.max_time_minutes,
    max_tool_calls=a.max_tool_calls,
    max_turns=a.max_turns,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=discover_adapters(str(policy), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS),
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=a.compaction_interval, overlap_size=2, token_threshold=14336, event_retention_size=5,
    ) if a.compaction_interval > 0 else None,
    graph_dir=str(DATA / 'graphs'),
    embeddings_dir=str(DATA / 'embeddings'),
    wheels_dir=wheels_dir,
    task_ids=run_ids,
    skip_agent_patch=(a.control == 'none'),
    concurrency=a.concurrency,
    display_mode='quiet',
    verbose=False,
)


async def main():
    return await asyncio.wait_for(Evaluator(cfg).run(), timeout=a.budget_s)


status = 'complete'
try:
    asyncio.run(main())
except asyncio.TimeoutError:
    status = 'budget_exhausted'
except Exception as e:  # noqa: BLE001
    status = f'error: {type(e).__name__}: {e}'
stop.set()
eval_secs = time.time() - T0

# ---- summary ----
rows = []
jl = OUT / 'results' / 'task_results.jsonl'
if jl.exists():
    rows = [json.loads(line) for line in jl.read_text().splitlines() if line.strip()]
resolved = [r.get('instance_id') for r in rows if r.get('resolved') or r.get('score') == 1.0]
metrics = [json.loads(line) for line in open(OUT / 'server_metrics.jsonl')] if (OUT / 'server_metrics.jsonl').exists() else []
gen = [m.get('vllm:generation_tokens_total') for m in metrics if 'vllm:generation_tokens_total' in m]
summary = {
    'tag': a.tag,
    'model_id': a.model_id,
    'status': status,
    'eval_secs': round(eval_secs, 1),
    'n_tasks': len(run_ids),
    'task_set': a.tasks,
    'order_seed': a.order_seed,
    'control': a.control,
    'extra_wheels': a.extra_wheels,
    'isolate': a.isolate,
    'py313': PY313,
    'official_env': a.official_env,
    'repo_wheels': a.repo_wheels,
    'uv_task_env': a.uv_task_env,
    'official_paths': not a.raw_sandbox_paths,
    'n_finished': len(rows),
    'n_resolved': len(resolved),
    'n_errors': sum(1 for r in rows if r.get('error')),
    'n_llm_calls': sum(int(r.get('total_llm_calls') or 0) for r in rows),
    'n_tool_calls': sum(int(r.get('tool_calls') or 0) for r in rows),
    'tool_calls_per_llm_call': round(sum(int(r.get('tool_calls') or 0) for r in rows)
                                     / max(1, sum(int(r.get('total_llm_calls') or 0) for r in rows)), 2),
    'healthy': any(int(r.get('total_llm_calls') or 0) >= 3 for r in rows),
    'errors_sample': sorted({str(r.get('error'))[:200] for r in rows if r.get('error')})[:5],
    'drop_thinking_budget': a.drop_thinking_budget,
    'stop_token_ids': STOP_IDS,
    'resolved_ids': resolved,
    'generated_tokens': (gen[-1] - gen[0]) if len(gen) > 1 else None,
    'gen_tok_per_s': round((gen[-1] - gen[0]) / max(eval_secs, 1), 1) if len(gen) > 1 else None,
    'finished_per_hour': round(len(rows) / max(eval_secs, 1) * 3600, 1),
    'run_ids': run_ids,
    'concurrency': a.concurrency,
    'sampling': (policy / 'configs' / 'sampling.yaml').read_text() if a.bundle else a.sampling,
    'bundle': a.bundle and Path(a.bundle).name,
    'caps': {'max_time_minutes': a.max_time_minutes, 'max_tool_calls': a.max_tool_calls,
             'max_turns': a.max_turns, 'timeout_seconds': a.timeout_seconds},
    'row_keys': sorted(rows[0].keys()) if rows else [],
}
(OUT / 'summary_smoke.json').write_text(json.dumps(summary, indent=2))
log(json.dumps({k: v for k, v in summary.items() if k not in ('run_ids', 'row_keys')}))
