"""Smoke run on TPU v5e-8: serve Qwen3.8-27B (bf16) with vllm-tpu, run the official harness on 16 tasks.

Serving reuses rahim3's published launcher (kaggle-tpu-lab, vendored below) with the public tunnel disabled;
the server only listens on 127.0.0.1 inside this session. The harness runs in its own Python 3.12 venv,
because swegemma needs >=3.12 and the sandbox test wheels are cp312/cp313.
Outputs land in /kaggle/working/smoke/<tag>/ + report.json (+ vllm.log from the launcher).
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
SERVE_PY = None  # @embed vendor/serve_qwen38.py
BUNDLES = None  # @embed_dir ../bundles/gen-v2 ../bundles/sub-v4 ../bundles/sub-v5b

T0 = time.time()
# gen-tpu-rb2 lost 21 min: uv's 30 s HTTP timeout broke the runtime download (llvmlite) and the launcher fell back to pip
os.environ.setdefault('UV_HTTP_TIMEOUT', '300')
DEADLINE_S = 95 * 60  # @define
EVAL_CAP_S = 30 * 60  # @define
PASSES = False  # @define (True: repeated passes over all tasks until the deadline)
SERVER_WAIT_S = 90 * 60  # counted from start; the env gate runs while the server loads
MIN_TRUSTED = 60  # @define (env gate: abort if fewer tasks are verifiable)
TEXT_ONLY = True  # @define (False matches the env dataset's cached XLA graphs for 131072 ctx / 16 seqs: ~6 min start)
CONC = 8  # @define (rollout concurrency; the server holds max_num_seqs=16)
TEMPS = [0.6]  # @define (temperature of pass k = TEMPS[k % len(TEMPS)])
SOLVED = {}  # @define (instance_id -> earlier teacher solves; passes run least-solved first)
TASKSET = 'competition'  # @define ('rebench': PR-Gym tasks from the g4dev-rebench-bank* kernel outputs, per-task uv envs)
RB_MAX_TASKS = 700  # @define (rebench: tasks sent to the env gate; newest first, one per repo before seconds)
GEN_BUNDLE = 'gen-v2'  # @define (bundle whose prompt drives the rollouts)
MAX_TOOL_CALLS = 60  # @define (rollout tool-call cap; training keeps <=28-call trajectories only)
FAST_START = False  # @define (launcher fast_start: skip the precompile, load graphs from the env dataset's cache on demand)
EARLY_SERVER = False  # @define (rebench: launch the server first and skip the env mini-gates; the bank is already
#                       harness-verified on CPU and the canary checks env + server end to end)
PRIOR_GLOB = ''  # @define (task_results.jsonl of earlier rollouts: their resolved tasks are skipped, unresolved go last)
PRIOR = {}  # filled from PRIOR_GLOB (rebench)
API_KEY = 'sk-smoke-local-only'
SERVED = 'qwen3.8-27b'
WORK = Path('/kaggle/working')
OUT = WORK / 'smoke'
OUT.mkdir(parents=True, exist_ok=True)
HVENV = '/tmp/hvenv'
REPORT = {'machine': 'tpu-v5e-8', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}


def save():
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    save()
    print(f'[{(time.time() - T0) / 60:5.1f} min] {name} {kv}', flush=True)


def find(pattern_tail, depth=6):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def healthy():
    try:
        req = urllib.request.Request('http://127.0.0.1:8000/v1/models', headers={'Authorization': f'Bearer {API_KEY}'})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


server = None
try:
    rc, accel = sh(['bash', '-c', 'ls /dev/accel* /dev/vfio 2>/dev/null | head; nproc; free -g | head -2; df -h /tmp | tail -1'])
    stage('env', python=sys.version.split()[0], host=accel)
    DATA = find('tasks.jsonl').parent
    WH = find('swegemma-*.whl').parent
    if TASKSET == 'rebench':
        import shutil
        rows, seen = [], set()
        for f in sorted(glob.glob('/kaggle/input/**/rebench/tasks_verified.jsonl', recursive=True)):
            snaps = Path(f).parent / 'snapshots'
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                src = snaps / f'{r["instance_id"].replace("/", "__")}.tgz'
                if r['instance_id'] in seen or not src.exists():
                    continue
                seen.add(r['instance_id'])
                r['_snap'] = str(src)
                rows.append(r)
        # Only tasks the official Evaluator already verified on a Kaggle CPU kernel (g4dev-rebench-harness-bank-cpu:
        # gold passes, empty patch fails). gen-tpu-rb1 tried to redo that gate here for all 852 tasks and never finished.
        cpu_ok = set()
        for f in glob.glob('/kaggle/input/**/rebench_harness/harness_trusted_ids.json', recursive=True):
            cpu_ok |= set(json.loads(Path(f).read_text()))
        stage('cpu_trusted', ids=len(cpu_ok), in_bank=sum(1 for r in rows if r['instance_id'] in cpu_ok))
        rows = [r for r in rows if r['instance_id'] in cpu_ok]
        PRIOR = {}  # earlier rollouts: instance_id -> resolved (any run)
        for f in (sorted(glob.glob(PRIOR_GLOB, recursive=True)) if PRIOR_GLOB else []):
            for line in open(f):
                if line.strip():
                    x = json.loads(line)
                    PRIOR[x['instance_id']] = PRIOR.get(x['instance_id'], False) or bool(x.get('resolved'))
        if PRIOR:
            stage('prior_rollouts', tasks=len(PRIOR), resolved=sum(PRIOR.values()),
                  skipped=sum(1 for r in rows if PRIOR.get(r['instance_id'])))
            rows = [r for r in rows if not PRIOR.get(r['instance_id'])]
        # never tried first, then earlier failures; newest first; first task of every repo before any second (diversity)
        rows.sort(key=lambda r: str(r.get('created_at') or ''), reverse=True)
        first, rest, repos = [], [], set()
        for r in rows:
            (rest if r['repo'] in repos else first).append(r)
            repos.add(r['repo'])
        picked = sorted(first + rest, key=lambda r: r['instance_id'] in PRIOR)[:RB_MAX_TASKS]
        rb = WORK / 'rbdata'
        (rb / 'snapshots').mkdir(parents=True, exist_ok=True)
        for d in ('wheels', 'graphs', 'embeddings'):
            (rb / d).mkdir(exist_ok=True)
        shutil.copytree(DATA / 'sample_submission', rb / 'sample_submission', dirs_exist_ok=True)
        for r in picked:
            dst = rb / 'snapshots' / f'{r["instance_id"]}.tgz'
            if not dst.exists():
                os.symlink(r.pop('_snap'), dst)
        (rb / 'tasks.jsonl').write_text(''.join(json.dumps({k: v for k, v in r.items() if k != '_snap'}) + '\n'
                                                for r in picked))
        stage('rebench_data', verified_rows=len(rows), repos=len(repos), picked=len(picked),
              newest=picked[0].get('created_at') if picked else None, oldest=picked[-1].get('created_at') if picked else None)
        if len(picked) < MIN_TRUSTED:
            raise RuntimeError(f'rebench bank not mounted or too small: {len(picked)} tasks')
        DATA = rb
    stage('data', data=str(DATA), wheelhouse=str(WH),
          weights=str(find('qwen3-8-27b-bf16/config.json') or find('*/qwen3-8-27b-bf16/config.json')))

    # ---- 1. launch the vllm-tpu server (background) ----
    def launch_server():
        serve_dir = WORK / 'tpu_serve'
        serve_dir.mkdir(exist_ok=True)
        src = SERVE_PY
        patches = {
            'threading.Thread(target=fetch_cloudflared, daemon=True).start()': 'pass  # public tunnel disabled',
            'for _ in range(60):  # cloudflared': 'for _ in range(0):  # cloudflared',
            '\nif CLOUDFLARED.exists():\n    tunnel = subprocess.Popen': '\nif False:\n    tunnel = subprocess.Popen',
            # Qwen's template raises when no plain user message is left (ADK compaction can drop it):
            # 18/129 tasks died with "No user query found in messages" in gen-tpu-v1 pass0. Without the raise the
            # template falls back to last_query_index = len(messages) - 1.
            """f"reasoning_effort|default('{cfg['reasoning_effort_default']}')")""":
                """f"reasoning_effort|default('{cfg['reasoning_effort_default']}')").replace(
                "{{- raise_exception('No user query found in messages.') }}", "")""",
        }
        applied = {}
        for old, new in patches.items():
            applied[old[:40]] = old in src
            src = src.replace(old, new, 1)
        if not all(applied.values()):
            raise RuntimeError(f'launcher patches not applied: {applied}')
        (serve_dir / 'serve_qwen38.py').write_text(src)
        (serve_dir / 'serve_config.json').write_text(json.dumps({
            'max_model_len': 131072, 'max_num_seqs': 16, 'text_only': TEXT_ONLY, 'fast_start': FAST_START,
            'reasoning_effort_default': 'medium', 'keepalive_min': 530, 'api_key': API_KEY,
            'served_model_name': SERVED,
        }))
        serve_log = open(OUT / 'serve_stdout.log', 'w')
        proc = subprocess.Popen([sys.executable, 'serve_qwen38.py'], cwd=serve_dir,
                                stdout=serve_log, stderr=subprocess.STDOUT)
        stage('server_launched', tunnel_patches=applied)
        return proc

    if TASKSET != 'rebench' or EARLY_SERVER:
        server = launch_server()


    # ---- 2. meanwhile: harness venv (Python 3.12) ----
    # Task sandboxes see this venv's site-packages (swegemma writes them into _host_env.pth), so it must also hold
    # pytest and the repos' runtime/test dependencies. On GPU the Kaggle image provides them; here nothing does
    # (gen-tpu-v1 ran 129 tasks without pytest: 0 resolved). The venv has no pip, so everything goes through uv.
    extra_whl = find('dirty_equals-*.whl')
    TEST_REQS = ['pytest', 'pytest-timeout', 'fastapi', 'starlette', 'pydantic', 'httpx', 'httpcore', 'h11', 'anyio',
                 'sniffio', 'requests', 'urllib3', 'idna', 'certifi', 'charset-normalizer', 'rich', 'pygments',
                 'markdown-it-py', 'mdurl', 'email-validator', 'python-multipart', 'orjson', 'ujson', 'pyyaml',
                 'jinja2', 'sqlalchemy', 'sqlmodel', 'uvicorn', 'typer', 'trio', 'inline-snapshot', 'dirty-equals',
                 'attrs', 'pysocks', 'brotlicffi', 'pytest-httpbin', 'httpbin', 'flask', 'werkzeug', 'trustme',
                 'pytest-cov', 'coverage', 'strawberry-graphql', 'passlib', 'pyjwt', 'pwdlib', 'annotated-doc',
                 'asttokens', 'executing', 'tokenize-rt', 'toml', 'tomlkit', 'zope-interface']
    if extra_whl:  # everything the GPU env needed on top of the Kaggle image, by distribution name
        TEST_REQS += sorted({w.name.split('-')[0].replace('_', '-').lower() for w in extra_whl.parent.glob('*.whl')})
    TEST_REQS = sorted(set(TEST_REQS))
    HV_STATE = {}

    def build_harness():
        uv_pip = [sys.executable, '-m', 'uv', 'pip', 'install', '--python', f'{HVENV}/bin/python']
        links = ['--find-links', str(DATA / 'wheels'), *(['--find-links', str(extra_whl.parent)] if extra_whl else [])]
        steps = [
            [sys.executable, '-m', 'pip', 'install', '-q', 'uv'],
            [sys.executable, '-m', 'uv', 'venv', HVENV, '--python', '3.12', '-q'],
            [*uv_pip, '--torch-backend=cpu',
             *[str(next(WH.glob(p))) for p in ('swegemma-*.whl', 'adk_submission-*.whl', 'adk_eval_core-*.whl',
                                                'google_adk-*.whl', 'google_genai-*.whl')]],
        ]
        for cmd in steps:
            rc, out = sh(cmd)
            if rc != 0:
                HV_STATE['error'] = f'{" ".join(cmd[:8])} -> {out}'
                stage('harness_venv_failed', cmd=' '.join(cmd[:8]), out=out)
                return
        if TASKSET == 'rebench':  # task envs are built per task with uv; the harness venv needs no test deps
            import shutil
            os.environ['G4DEV_UV'] = shutil.which('uv') or str(Path(sys.executable).parent / 'uv')
            rc, out = sh([os.environ['G4DEV_UV'], '--version'])
            stage('uv', path=os.environ['G4DEV_UV'], rc=rc, version=out.strip()[:60])
            vers = sorted({str(r.get('python') or '3.11') for r in picked})
            rc, out = sh([os.environ['G4DEV_UV'], 'python', 'install', *vers], timeout=900)
            stage('uv_pythons', versions=vers, rc=rc, out=out.strip()[-300:])
            rc, out = sh([f'{HVENV}/bin/python', '-c', 'import swegemma, adk_submission, google.adk, litellm; print("ok")'])
            if rc != 0:
                HV_STATE['error'] = f'import check failed: {out}'
            stage('harness_venv_ready', import_check=out.strip()[-300:], rc=rc)
            return
        # test deps: one resolve; if some pair conflicts, install one by one (the env gate below decides what is usable)
        rc, out = sh([*uv_pip, *links, *TEST_REQS])
        failed = []
        if rc != 0:
            for req in TEST_REQS:
                if sh([*uv_pip, *links, req])[0] != 0:
                    failed.append(req)
        stage('test_deps_installed', joint_rc=rc, failed=failed, out=out[-600:] if rc else '')
        rc, out = sh([f'{HVENV}/bin/python', '-c',
                      'import swegemma, adk_submission, google.adk, litellm, pytest, starlette, pygments; print("ok")'])
        if rc != 0:
            HV_STATE['error'] = f'import check failed: {out}'
        stage('harness_venv_ready', import_check=out.strip()[-300:], rc=rc, n_test_reqs=len(TEST_REQS))

    hv = threading.Thread(target=build_harness)
    hv.start()
    hv.join()
    if HV_STATE.get('error') or not Path(f'{HVENV}/bin/python').exists():
        raise RuntimeError(f'harness venv unusable: {HV_STATE.get("error", "missing")[-2000:]}')
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    extra_args = ['--extra-wheels', str(extra_whl.parent)] if extra_whl else []
    if TASKSET == 'rebench':
        extra_args = ['--uv-task-env']

    # ---- 3. env gate (CPU only, overlaps the server's weight loading): which tasks can this env verify? ----
    # gold patch must pass and the empty patch must fail; rollouts run only on those tasks.
    def gold_probe(name, ids, conc, budget_s):
        d = OUT / name
        (OUT / f'{name}_ids.json').write_text(json.dumps(ids))
        try:
            subprocess.run([f'{HVENV}/bin/python', str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', name,
                            '--out', str(d), '--tasks', 'file', '--task-ids-file', str(OUT / f'{name}_ids.json'),
                            '--control', 'gold', '--concurrency', str(conc), '--budget-s', str(budget_s), *extra_args],
                           timeout=budget_s + 300)
        except subprocess.TimeoutExpired:
            pass
        f, e = d / 'results' / 'task_results.jsonl', d / 'env_setup.jsonl'
        res = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        env = [json.loads(x) for x in e.read_text().splitlines() if x.strip()] if e.exists() else []
        env_s = sorted(r['env_s'] for r in env if 'env_s' in r)
        out = {'n': len(ids), 'finished': len(res), 'resolved': sum(1 for r in res if r.get('resolved')),
               'env_median_s': env_s[len(env_s) // 2] if env_s else None, 'env_max_s': env_s[-1] if env_s else None,
               'bulk_ok': sum(1 for r in env if r.get('frozen_rc') == 0), 'reused': sum(1 for r in env if r.get('reused')),
               'errors': sorted({str(r.get('error'))[:160] for r in res if r.get('error')})[:4]}
        stage(name, **out)
        return out

    gate_dir = OUT / 'gate'
    gate_conc = str(min(32, max(4, (os.cpu_count() or 8) // 2)))
    gate = {}
    if TASKSET == 'rebench':
        # Mini-gate BEFORE the server starts (no contention, a few minutes of quota): 16 CPU-verified tasks must still
        # verify here and their envs must build quickly. Then the same probe once more while the server loads
        # (informational: does server start-up slow env builds?).
        all_ids = [r['instance_id'] for r in picked]
        if not EARLY_SERVER:
            g = gold_probe('minigate', all_ids[:16], 8, 12 * 60)
            if g['resolved'] < 10 or (g['env_median_s'] or 1e9) > 180:
                raise RuntimeError(f'mini-gate failed: {g}; not starting the server')
            server = launch_server()
            gold_probe('gate_under_load', all_ids[16:32], 8, 8 * 60)
        gate = {'gold': {i: {'resolved': True} for i in all_ids}, 'none': {i: {'resolved': False} for i in all_ids}}
    for control in (() if TASKSET == 'rebench' else ('gold', 'none')):
        subprocess.run([f'{HVENV}/bin/python', str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', control,
                        '--out', str(gate_dir / control), '--tasks', 'all', '--control', control,
                        '--concurrency', gate_conc, '--budget-s', str(30 * 60), *extra_args], timeout=40 * 60)
        f = gate_dir / control / 'results' / 'task_results.jsonl'
        gate[control] = {r['instance_id']: r for r in map(json.loads, f.read_text().splitlines())} if f.exists() else {}
    trusted = sorted(i for i, r in gate['gold'].items()
                     if r.get('resolved') and gate['none'].get(i, {}).get('resolved') is False)
    (OUT / 'trusted_ids.json').write_text(json.dumps(trusted))
    by_repo = {}
    for i in trusted:
        by_repo[i.rsplit('_', 1)[0]] = by_repo.get(i.rsplit('_', 1)[0], 0) + 1
    if TASKSET == 'rebench':
        by_repo = {'repos': len(by_repo)}
    stage('env_gate', trusted=len(trusted), by_repo=by_repo, gold_n=len(gate['gold']),
          free_pass=sorted(i for i, r in gate['none'].items() if r.get('resolved')))
    if len(trusted) < MIN_TRUSTED:
        raise RuntimeError(f'env gate: only {len(trusted)} tasks verifiable (< {MIN_TRUSTED}); not starting rollouts')

    # ---- 4. wait for the server ----
    while not healthy():
        if server.poll() is not None:
            raise RuntimeError(f'serve script exited early with code {server.returncode}')
        if time.time() - T0 > SERVER_WAIT_S:
            raise TimeoutError('server not healthy within the wait budget')
        time.sleep(15)
    stage('server_up', startup_min=round((time.time() - T0) / 60, 1))

    # ---- 5. rollouts on gate-trusted tasks: canary first, then passes ----
    gen_bundle = WORK / 'bundles' / GEN_BUNDLE
    for rel, text in BUNDLES[GEN_BUNDLE].items():
        (gen_bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (gen_bundle / rel).write_text(text)
    common = [f'{HVENV}/bin/python', str(WORK / 'eval_harness.py'), '--data', str(DATA),
              '--base-url', 'http://127.0.0.1:8000/v1', '--model-id', SERVED, '--api-key', API_KEY,
              # trace rules v2: discipline prompt (single agent), scorer-like tool/turn caps; the minute cap stays
              # loose because the teacher runs at a different speed (filter traces by length afterwards)
              '--bundle', str(gen_bundle), '--override-sampling', '--caps-from-args',
              # vllm-tpu is launched without --reasoning-config: thinking_token_budget would 400 every request
              '--drop-thinking-budget', '--max-tool-calls', str(MAX_TOOL_CALLS), '--max-turns', str(2 * MAX_TOOL_CALLS),
              '--timeout-seconds', '120',
              *extra_args]

    def summary_of(d):
        f = d / 'summary_smoke.json'
        out = json.loads(f.read_text()) if f.exists() else None
        if out:
            out.pop('run_ids', None)
        return out

    def sampling(k):
        return ['--sampling', f'temperature: {TEMPS[k % len(TEMPS)]}\ntop_p: 0.95\nmax_output_tokens: 16384\n']

    canary_dir = OUT / SERVED / 'canary'
    canary_sel = ['--tasks', 'canary']
    if TASKSET == 'rebench':  # the fixed canary ids are competition tasks; use two gate-trusted tasks instead
        (OUT / 'canary_ids.json').write_text(json.dumps(trusted[:2]))
        canary_sel = ['--tasks', 'file', '--task-ids-file', str(OUT / 'canary_ids.json')]
    rc = subprocess.run([*common, *sampling(0), '--tag', SERVED, '--out', str(canary_dir), *canary_sel,
                         '--concurrency', '2', '--max-time-minutes', '6', '--budget-s', str(15 * 60)],
                        timeout=15 * 60 + 600).returncode
    canary = summary_of(canary_dir)
    stage('canary', returncode=rc, summary=canary)
    if not (canary and canary.get('healthy')):
        raise RuntimeError(f'canary unhealthy: {canary and canary.get("errors_sample")}')

    if not PASSES:
        budget = min(EVAL_CAP_S, DEADLINE_S - (time.time() - T0) - 5 * 60)
        out_dir = OUT / SERVED / 'smoke'
        stage('eval_start', budget_min=round(budget / 60, 1))
        rc = subprocess.run([*common, *sampling(0), '--tag', SERVED, '--out', str(out_dir), '--tasks', 'smoke',
                             '--concurrency', '8', '--max-time-minutes', '20', '--budget-s', str(int(budget))],
                            timeout=budget + 600).returncode
        stage('eval_done', returncode=rc, summary=summary_of(out_dir))
    k = 0
    while PASSES and DEADLINE_S - (time.time() - T0) > 30 * 60:
        budget = DEADLINE_S - (time.time() - T0) - 10 * 60
        out_dir = OUT / SERVED / f'pass{k}'
        stage('pass_start', k=k, budget_min=round(budget / 60, 1))
        # least-solved first (earlier teacher passes + this run's earlier passes), random within ties
        import random as _random
        done = dict(SOLVED)
        for j in range(k):
            prev = summary_of(OUT / SERVED / f'pass{j}') or {}
            for iid in prev.get('resolved_ids', []):
                done[iid] = done.get(iid, 0) + 1
        rng = _random.Random(3000 + k)
        order = sorted(trusted, key=lambda i: (done.get(i, 0), i in PRIOR, rng.random()))
        (OUT / f'order_pass{k}.json').write_text(json.dumps(order))
        stage('pass_order', k=k, temperature=TEMPS[k % len(TEMPS)], first=order[:5],
              unsolved=sum(1 for i in order if not done.get(i)))
        proc = subprocess.Popen([*common, *sampling(k), '--tag', SERVED, '--out', str(out_dir), '--tasks', 'file',
                                 '--task-ids-file', str(OUT / 'trusted_ids.json'),
                                 '--order-ids-file', str(OUT / f'order_pass{k}.json'),
                                 '--concurrency', str(CONC), '--max-time-minutes', '20',
                                 '--budget-s', str(int(budget))])
        # throughput watchdog: a pass that crawls must not burn the session (the CLI cannot stop a running kernel)
        t_pass, res_f, n_done, stalled, last_note = time.time(), out_dir / 'results' / 'task_results.jsonl', 0, False, 0
        while proc.poll() is None:
            time.sleep(60)
            el = time.time() - t_pass
            n_done = sum(1 for _ in open(res_f)) if res_f.exists() else 0
            if el - last_note >= 30 * 60:
                last_note = el
                stage('pass_progress', k=k, min=round(el / 60), finished=n_done)
            stalled = el > 45 * 60 and n_done < 12
            if stalled or el > budget + 900:
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except Exception:  # noqa: BLE001
                    proc.kill()
                break
        rc = proc.returncode
        summ = summary_of(out_dir)
        e_f = out_dir / 'env_setup.jsonl'
        env_s = sorted(r['env_s'] for r in map(json.loads, e_f.read_text().splitlines()) if 'env_s' in r) if e_f.exists() else []
        stage('pass_done', k=k, returncode=rc, summary=summ, finished=n_done,
              env_median_s=env_s[len(env_s) // 2] if env_s else None, env_p90_s=env_s[int(len(env_s) * 0.9)] if env_s else None)
        if stalled:
            raise RuntimeError(f'pass {k}: only {n_done} tasks finished in 45 min; stopping to save quota')
        if summ and summ.get('n_finished', 0) >= 20 and not summ.get('healthy'):
            raise RuntimeError('pass unhealthy; stopping to save quota')
        k += 1
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-3000:])
finally:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:  # noqa: BLE001
            server.kill()
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    save()
    print(json.dumps(REPORT, indent=2, default=str)[-4000:], flush=True)
