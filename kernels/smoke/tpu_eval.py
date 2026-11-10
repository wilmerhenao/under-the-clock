"""Agent evals on a Kaggle TPU v5e-8 instead of the GPU (user 10-28, B144).

Gemma 4 31B is served by vllm-tpu from the bf16 QAT checkpoint (gemma-4-31b-it-qat-q4_0-unquantized, the weights our
adapters are trained on), with each arm's LoRA merged into the weights on the host (W += alpha/r * B @ A, the same
mapping as lora_tpu.merge_into_hf); the arms run one after the other on the same server settings. The agent side is
unchanged from the GPU evals: eval_harness.py, the submission bundle (its sampling and eval_config caps, time x
EVAL_TIME_SCALE), the repograph unseen-repo tasks trusted by the CPU controls, scorer-style py3.13 task envs.
Serving flags follow the competition's VllmServer (gemma4 tool and reasoning parsers, reasoning config, thinking on)
and sauravbellam's TPU smoke test (vllm-tpu 0.31.0, tp 8, max_num_seqs 8, 4,096 batched tokens, multimodal off).
Known differences from the scorer: bf16 instead of w4a16 weights, TPU speed, and vllm-tpu does not enforce
thinking_token_budget (B46: only ~3% of completion tokens lie beyond 2,048 thinking tokens).

MODE 'canary': the first arm only, request probes (plain / tool call / harness request shape / seed / single-stream
and 8-stream decode speed), then EVAL_MAX_TASKS tasks; the report gives the per-call latency fit (seconds = a + b x
completion tokens) to set EVAL_TIME_SCALE against the GPU evals (16.7 s + 0.139 s/token at 8 concurrent, time x2, B142).
MODE 'eval': every arm on the same tasks. Outputs: /kaggle/working/tpu_eval/{log.jsonl, report.json, <arm>/...},
/kaggle/working/xla_cache (mount this kernel's output in the next run to skip the TPU compile).
"""
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODE = 'canary'  # @define
DEADLINE_S = 4800  # @define (whole notebook; the server is stopped and the report written before it)
ARMS = [{'name': 'g11a', 'adapter': '**/gemma-4-31b-agent-lora-short-thought-trainer/adapter_out'}]  # @define
VLLM_PINS = ['vllm-tpu==0.31.0']  # @define
TORCH_PINS = ['torch==2.10.0', 'torchvision==0.25.0']  # @define (CPU wheels; tpu-inference 0.31.0 pins torchvision 0.25.0)
SERVE = {'max_model_len': 32768, 'hbm_util': 0.90, 'max_num_seqs': 8, 'max_num_batched_tokens': 4096}  # @define
SERVER_START_MIN = 50  # @define
EVAL_BUNDLE = 'sub-v9c'  # @define
EVAL_REPOGRAPH = 'priv_eval'  # @define
REPOGRAPH_EXCLUDE = []  # @define
EVAL_MAX_TASKS = 8  # @define
EVAL_SEED = 0  # @define
EVAL_CONC = 8  # @define
EVAL_TIME_SCALE = 2.0  # @define
EVAL_ARM_MIN = 25  # @define (wall-clock budget per arm)
EVAL_CAPS = {}  # @define (non-empty: lift the bundle's limits, e.g. {'max_time_minutes': 60, 'max_tool_calls': 100, 'max_turns': 200,
               # 'timeout_seconds': 240}; prompt unchanged; the long-budget run behind the solved-within-T curve, B150)
ENV_ARGS = ['--py313-kit', '@kit', '--find-links', '@wh313', '--official-env', '--repo-wheels', '@rw']  # @define
EVAL_PY = None  # @embed eval_harness.py
BUNDLES = None  # @embed_dir ../bundles/sub-v9c ../bundles/sub-v9g ../bundles/sub-v9T1 ../bundles/sub-v9s

T0 = time.time()
WORKING = Path('/kaggle/working')
OUT = WORKING / 'tpu_eval'
OUT.mkdir(parents=True, exist_ok=True)
XLA_CACHE = WORKING / 'xla_cache'
XLA_CACHE.mkdir(parents=True, exist_ok=True)
SCRATCH = Path('/tmp/tpu_eval')
SCRATCH.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'log.jsonl', 'a')
REPORT = {'mode': MODE, 'arms': {}, 'stages': []}
SERVER = {'proc': None}


def stage(name, **kv):
    kv = {'stage': name, 't_min': round((time.time() - T0) / 60, 2), **kv}
    LOG.write(json.dumps(kv, default=str) + '\n')
    LOG.flush()
    REPORT['stages'].append(kv)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=1, default=str))
    print(json.dumps(kv, default=str)[:4000], flush=True)


def left_s():
    return DEADLINE_S - (time.time() - T0)


def sh(cmd, timeout=3600, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return Path(hits[0]) if hits else None


SKIP_NAMES = ('torch', 'torchvision', 'torchaudio', 'vllm', 'triton', 'cupy-cuda12x', 'flashinfer-python', 'xformers',
              'bitsandbytes', 'xgrammar', 'deep-gemm', 'flash-attn', 'apache-tvm-ffi')


def wheel_name(fn):
    return re.sub(r'[-_.]+', '-', Path(fn).name.split('-')[0]).lower()


def harness_requirements(whls, skip):
    """Requires-Dist of the harness wheels that apply on this Python, minus the GPU stack and the wheels themselves."""
    import zipfile
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    have = {wheel_name(w) for w in whls}
    reqs = {}
    for w in whls:
        with zipfile.ZipFile(w) as z:
            meta = next((n for n in z.namelist() if n.endswith('.dist-info/METADATA')), None)
            lines = z.read(meta).decode(errors='replace').splitlines() if meta else []
        for line in lines:
            if not line.startswith('Requires-Dist:'):
                continue
            spec = line.split(':', 1)[1].strip()
            if Requirement:
                try:
                    r = Requirement(spec)
                except Exception:  # noqa: BLE001
                    continue
                env = {'python_version': '3.12', 'python_full_version': '3.12.0', 'sys_platform': 'linux',
                       'platform_system': 'Linux', 'platform_machine': 'x86_64', 'extra': ''}
                if r.marker and not r.marker.evaluate(env):
                    continue
                name = re.sub(r'[-_.]+', '-', r.name).lower()
                spec = f'{r.name}{"[" + ",".join(sorted(r.extras)) + "]" if r.extras else ""}{r.specifier}'
            else:
                if 'extra ==' in spec:
                    continue
                spec = spec.split(';')[0].strip()
                name = re.sub(r'[-_.]+', '-', re.split(r'[\s\[<>=!~(]', spec, 1)[0]).lower()
            if name in have or skip.match(name) or name in SKIP_NAMES:
                continue
            reqs.setdefault(name, spec)
    return sorted(reqs.values())


def http(path, body=None, timeout=900):
    req = urllib.request.Request('http://127.0.0.1:8000' + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {'error': e.read().decode(errors='replace')[:800]}
    except Exception as e:  # noqa: BLE001
        return None, {'error': f'{type(e).__name__}: {e}'[:800]}


def stop_server():
    p = SERVER['proc']
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
    SERVER['proc'] = None


def watchdog():  # the notebook must end cleanly before DEADLINE_S whatever a subprocess does
    while True:
        time.sleep(30)
        if left_s() < 120:
            stage('deadline_watchdog', note='stopping the server and ending the session')
            stop_server()
            os._exit(0)


threading.Thread(target=watchdog, daemon=True).start()

MERGE_PY = r'''
import json, sys, shutil, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
src, adir, dst = map(Path, sys.argv[1:4])
t0 = time.time()
cfg = json.loads((adir / 'adapter_config.json').read_text())
scale = float(cfg['lora_alpha']) / float(cfg['r'])
peft = load_file(str(adir / 'adapter_model.safetensors'))
PEFT_PREFIX, HF_PREFIX = 'base_model.model.model.language_model.layers.', 'model.language_model.layers.'
deltas = {}
for k, A in peft.items():
    if k.endswith('.lora_A.weight'):
        deltas[HF_PREFIX + k[len(PEFT_PREFIX):].replace('.lora_A.weight', '.weight')] = (A, peft[k.replace('lora_A', 'lora_B')])
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix != '.safetensors':
        shutil.copy2(f, dst / f.name)
done, maxrel = 0, 0.0
for shard in sorted(src.glob('*.safetensors')):
    sd = load_file(str(shard))
    for key in list(sd):
        if key in deltas:
            A, B = deltas[key]
            w = sd[key]
            d = scale * (B.float() @ A.float())
            assert d.shape == w.shape, (key, tuple(d.shape), tuple(w.shape))
            maxrel = max(maxrel, float(d.abs().max() / w.float().abs().max()))
            sd[key] = (w.float() + d).to(w.dtype)
            done += 1
    save_file(sd, str(dst / shard.name), metadata={'format': 'pt'})
print(json.dumps({'merged': done, 'expected': len(deltas), 'scale': scale, 'max_rel_delta': maxrel,
                  's': round(time.time() - t0, 1)}))
assert done == len(deltas), f'merged {done} of {len(deltas)} LoRA modules: HF key names differ'
'''


def repograph_data(comp):
    """As train_gpu.repograph_data: competition-layout dir over the repograph split, graphs under the harness's names,
    stand-in embeddings, and the CPU-trusted ids minus excluded repos (seeded subset of EVAL_MAX_TASKS)."""
    import csv
    import hashlib
    import numpy as np
    rg = Path(sorted(glob.glob(f'/kaggle/input/**/repograph-verified-swe-tasks/{EVAL_REPOGRAPH}/tasks.jsonl',
                               recursive=True))[0]).parent
    data = SCRATCH / 'rgdata'
    for d in ('snapshots', 'graphs', 'embeddings'):
        (data / d).mkdir(parents=True, exist_ok=True)
    for d in ('wheels', 'docker', 'sandbox', 'sample_submission'):
        if not (data / d).exists() and (comp / d).exists():
            os.symlink(comp / d, data / d)
    rows = [json.loads(x) for x in (rg / 'tasks.jsonl').read_text().splitlines() if x.strip()]
    trusted, rg_ids = None, {r['instance_id'] for r in rows}
    for f in sorted(glob.glob('/kaggle/input/**/controls/controls.csv', recursive=True)):
        with open(f) as fh:
            got = {r['instance_id']: r['label_trusted'] == 'True' for r in csv.DictReader(fh)}
        if got and set(got) <= rg_ids:
            trusted = (trusted or set()) | {i for i, ok in got.items() if ok}
    if trusted is None:
        raise RuntimeError('no repograph controls.csv mounted (g4dev-repograph-controls-cpu output)')
    excl = {x.lower() for x in REPOGRAPH_EXCLUDE}

    def vec(text, dim=64):
        v = np.zeros(dim, np.float32)
        for tok in re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])', text):
            h = int(hashlib.md5(tok.lower().encode()).hexdigest()[:8], 16)
            v[h % dim] += 1.0 if (h >> 12) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v
    for r in rows:
        os.symlink(rg / 'snapshots' / f"{r['instance_id']}.tgz", data / 'snapshots' / f"{r['instance_id']}.tgz")
        short, sha = r['repo'].split('/')[-1], r['base_commit']
        dst = data / 'graphs' / f'{short}_{sha}.json'
        src = next((p for p in (rg / 'graphs' / f"{r['instance_id'].rsplit('_', 1)[0]}_{sha}.json",
                                rg / 'graphs' / f'{short}_{sha}.json') if p.exists()), None)
        if src is None or dst.exists():
            continue
        os.symlink(src, dst)
        g = json.loads(src.read_text())
        keys = {}
        for n in g.get('nodes', []):
            nid = str(n.get('id') or n.get('name') or '')
            if nid and '/' not in nid:
                keys[nid] = vec(nid + ' ' + str(n.get('text') or '')[:600])
        if keys:
            np.savez_compressed(data / 'embeddings' / f'{short}_{sha}.npz', **keys)
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    ids = sorted(r['instance_id'] for r in rows if r['instance_id'] in trusted and r['repo'].lower() not in excl)
    if EVAL_MAX_TASKS and len(ids) > EVAL_MAX_TASKS:
        ids = sorted(random.Random(EVAL_SEED).sample(ids, EVAL_MAX_TASKS))
    stage('repograph_data', tasks=len(rows), trusted=len(trusted), eval_ids=len(ids), excluded_repos=len(excl))
    return data, ids


def latency_fit(run_dir):
    """Per agent call: seconds since the previous observation and completion tokens (trace steps[], B142)."""
    import collections
    import numpy as np
    pts, tool_s = [], []
    tools, tasks_using = collections.Counter(), collections.Counter()  # tool use per arm (does the model follow a rule?)
    for tf in glob.glob(f'{run_dir}/results/traces/trace_*.json'):
        try:
            t = json.load(open(tf))
        except Exception:  # noqa: BLE001
            continue
        used = {c.get('function_name') for s in (t.get('steps') or []) for c in (s.get('tool_calls') or []) if isinstance(c, dict)}
        tools.update(c.get('function_name') for s in (t.get('steps') or []) for c in (s.get('tool_calls') or []) if isinstance(c, dict))
        tasks_using.update(used)
        last = None
        for s in t.get('steps') or []:
            ex = s.get('extra') if isinstance(s.get('extra'), dict) else {}
            se = ex.get('elapsed_s')
            if s.get('source') == 'agent':
                tc = (s.get('tool_calls') or [{}])[0] or {}
                tcall = (tc.get('extra') or {}).get('elapsed_s', se) if isinstance(tc.get('extra'), dict) else se
                ob = s.get('observation') if isinstance(s.get('observation'), dict) else {}
                tobs = (ob.get('extra') or {}).get('elapsed_s') if isinstance(ob.get('extra'), dict) else None
                if isinstance(tcall, (int, float)) and isinstance(last, (int, float)):
                    pts.append((max(0.0, tcall - last), ((s.get('metrics') or {}).get('completion_tokens') or 0)))
                if isinstance(tobs, (int, float)) and isinstance(tcall, (int, float)):
                    tool_s.append(max(0.0, tobs - tcall))
                    last = tobs
                elif isinstance(tcall, (int, float)):
                    last = tcall
            elif isinstance(se, (int, float)):
                last = se
    if len(pts) < 4:
        return {'calls': len(pts), 'tools': dict(tools), 'tasks_using': dict(tasks_using)}
    X = np.array([[1.0, c] for _, c in pts])
    y = np.array([s for s, _ in pts])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    c = sorted(p[1] for p in pts)
    return {'calls': len(pts), 'intercept_s': round(a, 2), 's_per_token': round(b, 4),
            'tok_per_s_marginal': round(1 / b, 1) if b > 0 else None, 'llm_s_p50': round(sorted(y)[len(y) // 2], 1),
            'completion_p50': c[len(c) // 2], 'tool_s_total': round(sum(tool_s), 1),
            'vs_gpu_eval': 'GPU 8-concurrent eval: 16.7 s + 0.139 s/token (B142)', 'tools': dict(tools),
            'tasks_using': dict(tasks_using)}


try:
    for src in sorted(glob.glob('/kaggle/input/**/xla_cache', recursive=True)):  # an earlier run's compiled TPU programs
        if Path(src).resolve() != XLA_CACHE.resolve():
            shutil.copytree(src, XLA_CACHE, dirs_exist_ok=True)
    stage('env', python=sys.version.split()[0], cpus=os.cpu_count(), xla_cache_files=sum(1 for _ in XLA_CACHE.rglob('*') if _.is_file()), tmp_free_gb=round(shutil.disk_usage('/tmp').free / 2**30),
          mem_gb=round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30), arms=ARMS, mode=MODE)
    os.environ['UV_HTTP_TIMEOUT'] = '300'
    rc, out = sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    if rc:
        raise RuntimeError(f'uv: {out}')

    # ---- serve env (vllm-tpu, CPU torch) in a thread; harness env meanwhile ----
    SV, HV = SCRATCH / 'serve_venv', SCRATCH / 'harness_venv'
    SPY, HPY = SV / 'bin/python', HV / 'bin/python'
    serve_env = {}

    def install_serve():
        t = time.time()
        r1 = sh(['uv', 'venv', '-p', '3.12', str(SV)])
        r2 = sh(['uv', 'pip', 'install', '-p', str(SPY), *TORCH_PINS, '--index-url', 'https://download.pytorch.org/whl/cpu'],
                timeout=2400)
        r3 = sh(['uv', 'pip', 'install', '-p', str(SPY), *VLLM_PINS, *TORCH_PINS, '--extra-index-url',
                 'https://download.pytorch.org/whl/cpu', '--index-strategy', 'unsafe-best-match'], timeout=2400)
        r4 = sh([str(SPY), '-c', 'import jax, vllm, safetensors, torch; print(vllm.__version__, jax.__version__, len(jax.devices()), jax.devices()[0].platform)'])
        serve_env.update(rc=[r1[0], r2[0], r3[0], r4[0]], check=r4[1][-400:], err=(r3[1] if r3[0] else '')[-1500:],
                         s=round(time.time() - t))
    th = threading.Thread(target=install_serve)
    th.start()

    wh = find('swegemma-*.whl').parent
    skip = re.compile(r'^(torch|torchvision|torchaudio|vllm|nvidia|triton|cuda|cupy|flashinfer|xformers|bitsandbytes|'
                      r'cutlass|apache_tvm|tvm|xgrammar|deep_gemm|flash_attn)', re.I)
    whls = sorted(str(w) for w in wh.glob('*.whl') if not skip.match(w.name))
    r1 = sh(['uv', 'venv', '-p', '3.12', str(HV)])
    # The competition wheels expect the GPU image's site-packages (pydantic, httpx, fastapi, opentelemetry, ...).
    # Resolve their declared requirements from PyPI, minus the GPU stack (an override with a never-true marker
    # removes a package from uv's resolution), then lay the wheels themselves over with --no-deps.
    reqs = harness_requirements(whls, skip) + ['google-cloud-storage']  # imported by google.adk.skills, undeclared
    (SCRATCH / 'harness_reqs.txt').write_text('\n'.join(reqs) + '\n')
    (SCRATCH / 'harness_override.txt').write_text(''.join(f"{n}; sys_platform == 'never'\n" for n in SKIP_NAMES))
    r3 = sh(['uv', 'pip', 'install', '-p', str(HPY), '-r', str(SCRATCH / 'harness_reqs.txt'), '--find-links', str(wh),
             '--override', str(SCRATCH / 'harness_override.txt')], timeout=2400) if reqs else (0, '')
    r2 = sh(['uv', 'pip', 'install', '-p', str(HPY), '--no-deps', '--reinstall', *whls], timeout=2400)
    fixed = []
    for _ in range(12):  # whatever the TPU image lacks comes from PyPI (internet is on for this notebook)
        rc, out = sh([str(HPY), '-c', 'import swegemma, adk_submission, google.adk, litellm, yaml, numpy, packaging; '
                      'from google.adk.models.lite_llm import LiteLlm; import swegemma.evaluate, swegemma.harness.container_setup'])
        if rc == 0:
            break
        m = re.findall(r"No module named '([A-Za-z0-9_\.]+)'", out)
        if not m:
            raise RuntimeError(f'harness env import failed: {out[-1500:]}')
        full = m[-1]
        mod = full.split('.')[0]
        # dotted names first: google.adk imports google.cloud.storage without declaring it (G17 session, B161)
        dotted = {'google.cloud': 'google-cloud-storage', 'google.cloud.storage': 'google-cloud-storage',
                  'google.cloud.aiplatform': 'google-cloud-aiplatform', 'google.auth': 'google-auth',
                  'google.protobuf': 'protobuf', 'google.api_core': 'google-api-core'}
        pkg = dotted.get(full) or dotted.get('.'.join(full.split('.')[:2])) or {
            'yaml': 'pyyaml', 'google': 'google-genai', 'dateutil': 'python-dateutil', 'PIL': 'pillow',
            'opentelemetry': 'opentelemetry-api opentelemetry-sdk'}.get(mod, mod)
        if pkg in fixed:
            raise RuntimeError(f'harness env: installing {pkg} did not provide {mod}: {out[-1200:]}')
        fixed.append(pkg)
        sh(['uv', 'pip', 'install', '-p', str(HPY), *pkg.split()])
    else:
        raise RuntimeError(f'harness env still missing modules after {fixed}')
    stage('harness_env', wheels=len(whls), reqs=len(reqs), rc=[r1[0], r3[0], r2[0]], pypi_fixes=fixed,
          resolve_err=(r3[1] if r3[0] else '')[-800:])

    comp = (find('competitions/gemma-4-developer-agent/tasks.jsonl') or find('tasks.jsonl')).parent
    kit, wh313 = find('py313.tar.gz*'), sorted(glob.glob('/kaggle/input/**/g4dev-py313-wheels/**/pytest_timeout-*.whl', recursive=True))
    rw = sorted(glob.glob('/kaggle/input/**/repograph-verified-swe-tasks-wheels', recursive=True))
    sub = {'@kit': str(kit.parent) if kit else None, '@wh313': str(Path(wh313[0]).parent) if wh313 else None,
           '@rw': rw[0] if rw else None}
    env_args = [sub.get(x, x) if x.startswith('@') else x for x in ENV_ARGS]
    if None in env_args:
        raise RuntimeError(f'unresolved env args {ENV_ARGS} -> {env_args}')
    data, ids = repograph_data(comp)
    (SCRATCH / 'eval_ids.json').write_text(json.dumps(ids))
    (SCRATCH / 'eval_harness.py').write_text(EVAL_PY)
    def write_bundle(name):
        d = SCRATCH / 'bundles' / name
        for rel, text in BUNDLES[name].items():
            if rel == 'prompts/system.md' and EVAL_TIME_SCALE != 1:
                # clock rules quote seconds of the scorer's budget: scale them with the eval's time limit (B152)
                text = re.sub(r'(`time_seconds_remaining` is below )(\d+)',
                              lambda m: m.group(1) + str(int(int(m.group(2)) * EVAL_TIME_SCALE)), text)
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text)
        return d
    bdirs = {name: write_bundle(name) for name in {EVAL_BUNDLE, *[a.get('bundle') or EVAL_BUNDLE for a in ARMS]}}
    bdir = bdirs[EVAL_BUNDLE]
    caps_args = []
    if EVAL_CAPS:  # same prompt and sampling; only the enforced limits change (eval_config.yaml and the harness args agree)
        for d in bdirs.values():
            (d / 'eval_config.yaml').write_text('evaluation:\n' + ''.join(f'  {k}: {v}\n' for k, v in EVAL_CAPS.items()))
        caps_args = ['--caps-from-args', '--max-time-minutes', str(EVAL_CAPS['max_time_minutes']),
                     '--max-tool-calls', str(EVAL_CAPS['max_tool_calls']), '--max-turns', str(EVAL_CAPS['max_turns']),
                     '--timeout-seconds', str(EVAL_CAPS['timeout_seconds'])]

    th.join(timeout=max(60, left_s() - 1800))
    stage('serve_env', **serve_env)
    if serve_env.get('rc', [1])[-1] != 0:
        raise RuntimeError('vllm-tpu environment failed: ' + str(serve_env.get('err') or serve_env.get('check')))

    base = (find('gemma-4-31b-it-qat-q4_0-unquantized/*/config.json') or
            find('gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')).parent
    stage('inputs', base=str(base), data=str(data), eval_ids=len(ids), env_args=env_args)

    arms = ARMS[:1] if MODE == 'canary' else ARMS
    SERVED = 'gemma'  # one served name: an arm that reuses the previous arm's server (same weights) asks for the same id
    drop_budget = False
    for ai, arm in enumerate(arms):
        name = arm['name']
        model_dir = base
        if arm.get('adapter'):
            pat = arm['adapter'] if arm['adapter'].startswith('/') else f"/kaggle/input/{arm['adapter']}"  # '/...': this session's own output
            adir = Path(sorted(glob.glob(pat, recursive=True))[0])
            mdir = SCRATCH / ('merged_' + re.sub(r'[^A-Za-z0-9]+', '_', arm['adapter'])[-60:])
            if not (mdir / 'config.json').exists():
                rc, out = sh([str(SPY), '-c', MERGE_PY, str(base), str(adir), str(mdir)], timeout=3600)
                stage(f'{name}_merge', rc=rc, adapter=str(adir), out=out[-800:])
                if rc:
                    raise RuntimeError(f'merge failed for {name}')
            model_dir = mdir
        # ---- serve (unless the previous arm left the same weights up) ----
        reuse = SERVER['proc'] is not None and SERVER['proc'].poll() is None
        slog = OUT / f'{name}_vllm.log'
        cmd = [str(SV / 'bin/vllm'), 'serve', str(model_dir), '--served-model-name', SERVED, '--host', '127.0.0.1',
               '--port', '8000', '--tensor-parallel-size', '8', '--max-model-len', str(SERVE['max_model_len']),
               '--gpu-memory-utilization', str(SERVE['hbm_util']), '--max-num-seqs', str(SERVE['max_num_seqs']),
               '--max-num-batched-tokens', str(SERVE['max_num_batched_tokens']),
               '--enable-auto-tool-choice', '--tool-call-parser', 'gemma4', '--reasoning-parser', 'gemma4',
               '--reasoning-config', '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}',
               '--default-chat-template-kwargs', '{"enable_thinking": true}',
               '--limit-mm-per-prompt', '{"image": 0, "audio": 0, "video": 0}']
        t = time.time()
        if not reuse:
            SERVER['proc'] = subprocess.Popen(cmd, stdout=open(slog, 'w'), stderr=subprocess.STDOUT,
                                              env={**os.environ, 'VLLM_XLA_CACHE_PATH': str(XLA_CACHE), 'VLLM_NO_USAGE_STATS': '1'})
        else:
            slog.write_text('server reused from the previous arm\n')
        up = False
        while time.time() - t < SERVER_START_MIN * 60 and left_s() > 900:
            if SERVER['proc'].poll() is not None:
                break
            st, body = http('/v1/models', timeout=10)
            if st == 200:
                up = True
                break
            time.sleep(15)
        txt = slog.read_text(errors='replace')
        stage(f'{name}_server', up=up, reused=reuse, start_min=round((time.time() - t) / 60, 1),
              xla_cache_files=sum(1 for _ in XLA_CACHE.rglob('*') if _.is_file()),
              facts=re.findall(r'[^\n]*(?:MODEL_IMPL|KV cache|Maximum concurrency|model_impl|Loading weights took)[^\n]{0,160}', txt)[-6:],
              tail=None if up else txt.splitlines()[-30:])
        if not up:
            raise RuntimeError(f'vllm-tpu did not come up for {name}')

        # ---- request probes (first arm) ----
        if ai == 0:
            tool = {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file.', 'parameters': {
                'type': 'object', 'properties': {'filepath': {'type': 'string'}}, 'required': ['filepath']}}}
            msgs = [{'role': 'user', 'content': 'Read the file src/app/routing.py using the tool.'}]
            count = [{'role': 'user', 'content': 'Count from 1 to 2000, separated by spaces.'}]
            probes = {
                'plain': {'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 16, 'temperature': 0},
                'tool_call': {'messages': msgs, 'tools': [tool], 'tool_choice': 'auto', 'max_tokens': 512, 'temperature': 0},
                'harness_shape': {'messages': msgs, 'tools': [tool], 'max_tokens': 512, 'temperature': 0.2,
                                  'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 128},
                'seeded': {'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 16, 'temperature': 0.2, 'seed': 1},
                'decode_1': {'messages': count, 'max_tokens': 512, 'temperature': 0, 'ignore_eos': True,
                             'chat_template_kwargs': {'enable_thinking': False}},
                'decode_2': {'messages': count, 'max_tokens': 512, 'temperature': 0, 'ignore_eos': True,
                             'chat_template_kwargs': {'enable_thinking': False}},
            }
            res = {}
            for pn, body in probes.items():
                t1 = time.time()
                st, r = http('/v1/chat/completions', {'model': SERVED, **body})
                msg = (r.get('choices') or [{}])[0].get('message', {}) if st == 200 else {}
                res[pn] = {'status': st, 's': round(time.time() - t1, 1), 'tool_calls': [c['function'] for c in msg.get('tool_calls') or []],
                           'reasoning_chars': len(msg.get('reasoning_content') or msg.get('reasoning') or ''),
                           'content': (msg.get('content') or '')[:160], 'tokens': (r.get('usage') or {}).get('completion_tokens'),
                           'error': str(r.get('error', ''))[:300]}
            out8 = []

            def one(i):
                t1 = time.time()
                st, r = http('/v1/chat/completions', {'model': SERVED, **probes['decode_2']})
                out8.append(((r.get('usage') or {}).get('completion_tokens') or 0, time.time() - t1, st))
            t8 = time.time()
            ths = [threading.Thread(target=one, args=(i,)) for i in range(8)]
            [x.start() for x in ths]
            [x.join(timeout=900) for x in ths]
            w8 = time.time() - t8
            d2 = res['decode_2']
            drop_budget = res['harness_shape']['status'] == 400
            REPORT['probes'] = res
            stage('probes', **{k: {kk: v[kk] for kk in ('status', 's', 'tool_calls', 'tokens', 'error')} for k, v in res.items()},
                  single_stream_tok_s=round(d2['tokens'] / d2['s'], 1) if d2['status'] == 200 and d2['s'] else None,
                  eight_stream_tok_s=round(sum(x[0] for x in out8) / w8, 1) if w8 else None,
                  eight_stream_status=sorted({x[2] for x in out8}), drop_thinking_budget=drop_budget)
            if not any(c.get('name') == 'read_file' for c in res['tool_call']['tool_calls']):
                raise RuntimeError('no parsed tool call: the gemma4 tool parser does not work on this vllm-tpu build')

        # ---- the agent on the eval tasks ----
        budget_s = min(EVAL_ARM_MIN * 60, left_s() - 600 - (len(arms) - ai - 1) * EVAL_ARM_MIN * 60)
        rdir = OUT / name
        t = time.time()
        p = subprocess.run([str(HPY), str(SCRATCH / 'eval_harness.py'), '--data', str(data),
                            '--base-url', 'http://127.0.0.1:8000/v1', '--model-id', SERVED, '--tag', name, '--out', str(rdir),
                            '--tasks', 'file', '--task-ids-file', str(SCRATCH / 'eval_ids.json'),
                            '--concurrency', str(EVAL_CONC), '--budget-s', str(int(budget_s)),
                            '--bundle', str(bdirs[arm.get('bundle') or EVAL_BUNDLE]), '--time-scale', str(EVAL_TIME_SCALE),
                            *(['--drop-thinking-budget'] if drop_budget else []), *caps_args, *env_args],
                           capture_output=True, text=True, timeout=budget_s + 900)
        f = rdir / 'summary_smoke.json'
        sm = json.loads(f.read_text()) if f.exists() else {}
        res_arm = {k: sm.get(k) for k in ('status', 'healthy', 'n_tasks', 'n_finished', 'n_resolved', 'n_errors',
                                          'n_llm_calls', 'resolved_ids')}
        rows = [json.loads(x) for x in open(rdir / 'results' / 'task_results.jsonl') if x.strip()] \
            if (rdir / 'results' / 'task_results.jsonl').exists() else []
        res_arm['tasks'] = [{k: r.get(k) for k in ('instance_id', 'resolved', 'duration_seconds', 'tool_calls',
                                                   'total_llm_calls', 'error')} for r in rows]
        res_arm.update(wall_min=round((time.time() - t) / 60, 1), rc=p.returncode, latency=latency_fit(rdir),
                       stderr_tail=(p.stderr or '')[-1200:] if p.returncode else None)
        REPORT['arms'][name] = res_arm
        stage(f'{name}_eval', **res_arm)
        nxt = arms[ai + 1] if ai + 1 < len(arms) else None
        if not (nxt and nxt.get('adapter') == arm.get('adapter')):
            stop_server()  # the next arm serves different weights (or this was the last arm)
            if arm.get('adapter'):
                shutil.rmtree(model_dir, ignore_errors=True)
    stage('done')
except Exception as e:  # noqa: BLE001
    import traceback
    stage('fatal', error=f'{type(e).__name__}: {e}'[-3000:], tb=traceback.format_exc()[-3000:])
finally:
    stop_server()
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=1, default=str))
