"""Smoke run on 4xL4: serve a teacher model with the wheelhouse vLLM, run the official harness on 16 tasks.

Runs offline (this competition disables internet on L4x4). Teacher order: gpt-oss-120b (MXFP4, 65 GB) -> fallback gemma-4-31b-it-qat-w4a16-ct (the competition model),
so a failed teacher start still yields a usable on-policy smoke run.
Outputs land in /kaggle/working/smoke/<tag>/ (results, traces, server metrics, summary) + report.json.
"""
import glob
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py

T0 = time.time()
DEADLINE_S = 80 * 60          # whole session, including install + model load
EVAL_CAP_S = 30 * 60          # rollout phase
WORK = Path('/kaggle/working')
OUT = WORK / 'smoke'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'machine': 'gpu-4xL4', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}


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


for k, v in {
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'VLLM_ENGINE_READY_TIMEOUT_S': '1500',
    'VLLM_NO_USAGE_STATS': '1', 'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ[k] = v

try:
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    stage('env', python=sys.version.split()[0], gpus=gpus, cpus=os.cpu_count())
    if len(gpus) < 4:
        raise SystemExit(f'expected 4 GPUs, got {len(gpus)}: stopping to save quota')

    # ---- wheelhouse install (same procedure as the official getting-started notebook) ----
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
    stage('wheelhouse_installed', wheelhouse=str(wh))

    # thoughts must survive between tool calls (vLLM reads `reasoning`, ADK sends `reasoning_content`; thread 744354)
    import vllm
    cu = Path(vllm.__file__).parent / 'entrypoints' / 'chat_utils.py'
    src = cu.read_text()
    old = 'reasoning = message.get("reasoning")'
    if old in src and 'reasoning_content' not in src.split(old, 1)[1][:80]:
        cu.write_text(src.replace(old, 'reasoning = message.get("reasoning") or message.get("reasoning_content")', 1))
        stage('vllm_reasoning_patch', applied=True)
    else:
        stage('vllm_reasoning_patch', applied=False, note='pattern not found or already patched')

    # gpt-oss (harmony) loads tiktoken encodings at startup; this competition's L4x4 sessions are offline
    tk = find('o200k_base.tiktoken')
    if tk:
        tk_dir = Path('/tmp/tiktoken')
        tk_dir.mkdir(exist_ok=True)
        for f in tk.parent.iterdir():
            if f.is_file():
                (tk_dir / f.name).write_bytes(f.read_bytes())
        for var in ('TIKTOKEN_ENCODINGS_BASE', 'TIKTOKEN_RS_CACHE_DIR', 'TIKTOKEN_CACHE_DIR'):
            os.environ[var] = str(tk_dir)
    stage('tiktoken', dir=str(tk.parent) if tk else None)

    DATA = find('tasks.jsonl').parent
    graphs_empty = sum(1 for f in (DATA / 'graphs').glob('*') if f.stat().st_size == 0)
    stage('data', data=str(DATA), empty_graph_files=graphs_empty)

    from adk_submission import VllmConfig, VllmServer

    candidates = [
        {
            'tag': 'gpt-oss-120b',
            'path': find('gpt-oss-120b/*/default/*/config.json') or find('gpt-oss-120b/*/*/*/config.json'),
            'cfg': dict(tool_call_parser='openai', reasoning_parser='openai_gptoss', max_model_len=65536),
            'sampling': 'temperature: 1.0\ntop_p: 1.0\nmax_output_tokens: 16384\n',
        },
        {
            'tag': 'gemma-4-31b-qat',
            'path': find('gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*/config.json')
                    or find('gemma-4-31b-it-qat-w4a16-ct/*/config.json'),
            'cfg': dict(tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768),
            'sampling': ('temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 16384\n'
                         'thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n'),
        },
    ]

    for c in candidates:
        if c['path'] is None:
            stage('skip_model', tag=c['tag'], reason='weights not mounted')
            continue
        remaining = DEADLINE_S - (time.time() - T0)
        if remaining < 25 * 60:
            stage('skip_model', tag=c['tag'], reason=f'only {remaining / 60:.0f} min left')
            continue
        model_dir = c['path'].parent
        server = None
        try:
            t_start = time.time()
            server = VllmServer(VllmConfig(
                model=str(model_dir), port=8000, host='127.0.0.1', dtype='auto',
                gpu_memory_utilization=0.92, enable_auto_tool_choice=True, enable_lora=False,
                tensor_parallel_size=4, startup_timeout=int(min(1500, remaining - 20 * 60)), **c['cfg'],
            ))
            server.start()
            stage('server_up', tag=c['tag'], startup_s=round(time.time() - t_start, 1), base_url=server.base_url)
        except Exception as e:  # noqa: BLE001
            stage('server_failed', tag=c['tag'], error=f'{type(e).__name__}: {e}'[-4000:])
            try:
                server and server.stop()
            except Exception:  # noqa: BLE001
                pass
            continue

        budget = min(EVAL_CAP_S, DEADLINE_S - (time.time() - T0) - 5 * 60)
        (WORK / 'eval_harness.py').write_text(EVAL_PY)
        out_dir = OUT / c['tag']
        cmd = [sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--base-url', server.base_url,
               '--model-id', str(getattr(server, 'effective_model_path', None) or server.config.model), '--tag', c['tag'], '--out', str(out_dir),
               '--sampling', c['sampling'], '--concurrency', '8', '--budget-s', str(int(budget))]
        stage('eval_start', tag=c['tag'], budget_min=round(budget / 60, 1))
        rc = subprocess.run(cmd, timeout=budget + 600).returncode
        summary = json.loads((out_dir / 'summary_smoke.json').read_text()) if (out_dir / 'summary_smoke.json').exists() else None
        stage('eval_done', tag=c['tag'], returncode=rc, summary=summary)
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
        break
except SystemExit as e:
    stage('stopped', reason=str(e))
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    save()
    print(json.dumps(REPORT, indent=2, default=str)[-4000:], flush=True)
