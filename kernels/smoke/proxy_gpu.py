"""LB-proxy run on 4xL4 (offline): the scoring model (gemma-4-31b-it-qat-w4a16-ct) served like the scorer
(adk_submission VllmServer defaults: gemma4 parsers + reasoning config, 32k context, no full-ISL reservation).

1. env gate (CPU, overlaps weight loading): gold patch must pass and the empty patch must fail -> trusted ids
2. speed probe: per-stream decode tok/s at 1/2/4/8 concurrent streams (sets the time scale; T10 data)
3. canary: sub-v2 on 2 small tasks must show real agent activity
4. paired A/B: bundles A and B run side by side on the trusted tasks in the same shuffled order, each with half
   the slots; time caps scaled by the measured slowdown (the scorer runs tasks one at a time)
5. leftover time: single-agent Gemma traces with the gen-v2 bundle (trace rules v2)
Outputs: /kaggle/working/proxy/<tag>/ (results, traces, patches, test logs) + report.json + ab_summary.json.
"""
import glob
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
BUNDLES = None  # @embed_dir ../bundles/sub-v1 ../bundles/sub-v2 ../bundles/gen-v2 ../bundles/sub-v2-think

T0 = time.time()
DEADLINE_S = int(2.5 * 3600)  # @define (stops itself; L4x4 appears to bill quota at 2x)
AB = ['sub-v2', 'sub-v1', 'gen-v2', 'sub-v2-think']  # @define (paired arms, same task order; first = canary)
SLOTS = 12  # @define (total concurrent tasks; split evenly across the arms)
MIN_TRUSTED = 60  # @define (env gate: abort if fewer tasks are verifiable)
WORK = Path('/kaggle/working')
OUT = WORK / 'proxy'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'machine': 'gpu-4xL4', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}


def save():
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    save()
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:3000]}', flush=True)


def left():
    return DEADLINE_S - (time.time() - T0)


def find(pattern_tail, depth=6):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def results(out_dir):
    f = out_dir / 'results' / 'task_results.jsonl'
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
    return {r['instance_id']: r for r in rows}


def summary_of(out_dir):
    f = out_dir / 'summary_smoke.json'
    s = json.loads(f.read_text()) if f.exists() else None
    if s:
        s.pop('run_ids', None)
    return s


for k, v in {
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1500', 'VLLM_NO_USAGE_STATS': '1', 'OTEL_SDK_DISABLED': 'true',
    # proxy-gpu-v1 died in sampler warm-up (CUDA OOM) without this; gen-gpu-v1 served Gemma fine with it + util 0.92
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ[k] = v

server = None
try:
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    stage('env', python=sys.version.split()[0], gpus=gpus, cpus=os.cpu_count())
    if len(gpus) < 4:
        raise SystemExit(f'expected 4 GPUs, got {len(gpus)}: stopping to save quota')

    # ---- wheelhouse (official procedure) ----
    wh = find('swegemma-*.whl').parent
    for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
        try:
            os.unlink(pth)
        except OSError:
            pass
    tmp_whl = Path('/tmp/wheelhouse')
    tmp_whl.mkdir(parents=True, exist_ok=True)
    for w in wh.glob('*.whl'):
        if 'cutlass' in w.name.lower():
            continue
        name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
        if not (tmp_whl / name).exists():
            os.symlink(w, tmp_whl / name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                    *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
    importlib.invalidate_caches()
    DATA = find('tasks.jsonl').parent
    extra_whl = find('dirty_equals-*.whl')
    extra_args = ['--extra-wheels', str(extra_whl.parent)] if extra_whl else []
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    for bname, files in BUNDLES.items():
        for rel, text in files.items():
            (WORK / 'bundles' / bname / rel).parent.mkdir(parents=True, exist_ok=True)
            (WORK / 'bundles' / bname / rel).write_text(text)
    gemma = (find('gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*/config.json')
             or find('gemma-4-31b-it-qat-w4a16-ct/*/config.json'))
    if gemma is None:
        raise RuntimeError('gemma-4-31b-it-qat-w4a16-ct weights not mounted')
    stage('ready', data=str(DATA), wheelhouse=str(wh), extra=str(extra_whl and extra_whl.parent),
          bundles=sorted(BUNDLES), model=str(gemma.parent))

    # ---- 1. env gate in a thread while the server loads ----
    gate = {}

    def run_gate():
        conc = '8'  # modest: vLLM's compile/warm-up is CPU-heavy and runs at the same time
        for control in ('gold', 'none'):
            d = OUT / 'gate' / control
            subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', control,
                            '--out', str(d), '--tasks', 'all', '--control', control, '--concurrency', conc,
                            '--budget-s', str(30 * 60), *extra_args], timeout=40 * 60,
                           stdout=open(OUT / f'gate_{control}.log', 'w'), stderr=subprocess.STDOUT)
            gate[control] = results(d)

    gate_thread = threading.Thread(target=run_gate)
    gate_thread.start()

    from adk_submission import VllmConfig, VllmServer
    t = time.time()
    server = VllmServer(VllmConfig(model=str(gemma.parent), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
                                   reasoning_parser='gemma4', max_model_len=32768, enable_lora=False,
                                   gpu_memory_utilization=0.92,
                                   tensor_parallel_size=4, startup_timeout=1500))
    server.start()
    model_id = str(getattr(server, 'effective_model_path', None) or server.config.model)
    stage('server_up', startup_s=round(time.time() - t, 1), cmd=' '.join(server.build_cmd()[3:]))

    gate_thread.join()
    trusted = sorted(i for i, r in gate.get('gold', {}).items()
                     if r.get('resolved') and gate.get('none', {}).get(i, {}).get('resolved') is False)
    (OUT / 'trusted_ids.json').write_text(json.dumps(trusted))
    by_repo = {}
    for i in trusted:
        by_repo[i.rsplit('_', 1)[0]] = by_repo.get(i.rsplit('_', 1)[0], 0) + 1
    stage('env_gate', trusted=len(trusted), by_repo=by_repo, gold_n=len(gate.get('gold', {})),
          free_pass=sorted(i for i, r in gate.get('none', {}).items() if r.get('resolved')))
    if len(trusted) < MIN_TRUSTED:
        raise RuntimeError(f'env gate: only {len(trusted)} tasks verifiable (< {MIN_TRUSTED})')

    # ---- 2. speed probe: per-stream decode speed vs concurrency ----
    def one_request(_):
        body = json.dumps({'model': model_id, 'max_tokens': 256, 'temperature': 0.0, 'ignore_eos': True,
                           'messages': [{'role': 'user', 'content': 'Write a long Python tutorial about decorators.'}],
                           'chat_template_kwargs': {'enable_thinking': False}}).encode()
        req = urllib.request.Request(server.base_url + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        t1 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            n = json.loads(r.read())['usage']['completion_tokens']
        return n / (time.time() - t1)

    one_request(0)  # warm-up: the first request after start-up is slow and skewed the v2 probe
    probe = {}
    for c in sorted({1, 2, 4, 8, SLOTS}):
        with ThreadPoolExecutor(c) as ex:
            rates = list(ex.map(one_request, range(c)))
        probe[c] = round(sum(rates) / len(rates), 2)
    slowdown = max(probe[1], probe[2]) / probe[SLOTS] if probe.get(SLOTS) else 1.5
    time_scale = round(max(1.0, slowdown), 2)
    stage('speed_probe', tok_per_s_per_stream=probe, slowdown_at_slots=round(slowdown, 2), time_scale=time_scale)

    def run_bundle(bname, out_dir, budget_s, conc, extra):
        cmd = [sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--base-url', server.base_url,
               '--model-id', model_id, '--tag', bname, '--out', str(out_dir), '--bundle', str(WORK / 'bundles' / bname),
               '--time-scale', str(time_scale), '--concurrency', str(conc), '--budget-s', str(int(budget_s)),
               *extra_args, *extra]
        rc = subprocess.run(cmd, timeout=budget_s + 900).returncode
        return rc, summary_of(out_dir)

    # ---- 3. canary ----
    rc, canary = run_bundle(AB[0], OUT / 'canary', 15 * 60, 2, ['--tasks', 'canary'])
    stage('canary', returncode=rc, summary=canary)
    # running out of turns/time/tool calls is a normal task outcome; only crashes (bad requests, unknown tools,
    # sandbox failures) mean the setup is broken (v2 aborted a healthy run on a turn-budget 'error')
    crash = [e for e in (canary or {}).get('errors_sample', []) if 'exceeded' not in e.lower()]
    if not (canary and canary.get('healthy') and not crash):
        raise RuntimeError(f'canary failed: {canary and {k: canary.get(k) for k in ("n_resolved", "n_errors", "errors_sample")}}')

    # ---- 4. paired A/B on trusted tasks ----
    budget = left() - 12 * 60
    stage('ab_start', bundles=AB, budget_min=round(budget / 60, 1), slots_each=SLOTS // len(AB))
    with ThreadPoolExecutor(len(AB)) as ex:
        futs = {b: ex.submit(run_bundle, b, OUT / b, budget, SLOTS // len(AB),
                             ['--tasks', 'file', '--task-ids-file', str(OUT / 'trusted_ids.json'),
                              '--order-seed', '3001']) for b in AB}
        ab = {b: f.result() for b, f in futs.items()}
    res = {b: results(OUT / b) for b in AB}
    common = sorted(set.intersection(*(set(r) for r in res.values())))
    paired = {b: sum(1 for i in common if res[b][i].get('resolved')) for b in AB}
    only = {b: sorted(i for i in common if res[b][i].get('resolved')
                      and not any(res[o][i].get('resolved') for o in AB if o != b)) for b in AB}
    ab_summary = {'common_tasks': len(common), 'resolved_on_common': paired, 'only_this_bundle': only,
                  'time_scale': time_scale, 'per_bundle': {b: s for b, (rc, s) in ab.items()}}
    (OUT / 'ab_summary.json').write_text(json.dumps(ab_summary, indent=2, default=str))
    stage('ab_done', common=len(common), resolved_on_common=paired,
          n_only={b: len(v) for b, v in only.items()})

    # ---- 5. leftover time: single-agent Gemma traces (gen-v2) ----
    k = 0
    while left() > 25 * 60:
        budget = left() - 10 * 60
        rc, s = run_bundle('gen-v2', OUT / f'gen-v2_pass{k}', budget, SLOTS,
                           ['--tasks', 'file', '--task-ids-file', str(OUT / 'trusted_ids.json'),
                            '--order-seed', str(4000 + k)])
        stage('gen_pass_done', k=k, returncode=rc, summary=s)
        k += 1
except SystemExit as e:
    stage('stopped', reason=str(e))
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-4000:], tb=traceback.format_exc()[-3000:])
finally:
    if server is not None:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    save()
    print(json.dumps(REPORT, indent=2, default=str)[-4000:], flush=True)
