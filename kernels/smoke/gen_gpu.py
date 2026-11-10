"""Generation run on 4xL4 or 2xL4 (offline): rollouts on all 129 public tasks until the session deadline.

Model ladder: gpt-oss-120b (teacher) with progressively leaner memory settings, then gpt-oss-20b
(13 GB, fits easily), then gemma-4-31b-it-qat-w4a16-ct (on-policy) as the guaranteed fallback. The first model that starts runs repeated passes over all
tasks, each pass in a fresh shuffled order (a pass cut by the deadline still covers a random subset).
gpt-oss note (run v1): weights load fine (MXFP4 -> Marlin on sm89) but at 0.92 util with default batching
(8192 batched tokens, cudagraphs up to 1024) only 0.17 GiB was left for KV cache.
Test deps missing from the Kaggle host (inline-snapshot, dirty-equals, pytest-httpbin, ...) come from
verracodeguacas/g4dev-test-wheels via eval_harness --extra-wheels.
Outputs: /kaggle/working/gen/<tag>/pass<k>/ (results, traces, patches, test logs, server metrics) + report.json.
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
DEADLINE_S = 11.25 * 3600  # @define (Kaggle GPU sessions stop at 12 h; leave room to flush outputs)
MIN_PASS_S = 25 * 60
WORK = Path('/kaggle/working')
OUT = WORK / 'gen'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'machine': 'gpu-4xL4', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}

BUDGETS = ['--concurrency', '16', '--max-tool-calls', '100', '--max-time-minutes', '20',
           '--max-turns', '150', '--timeout-seconds', '120']
GEMMA_SAMPLING = ('temperature: 0.6\ntop_p: 0.95\nmax_output_tokens: 16384\n'
                  'thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n')
GPTOSS_SAMPLING = 'temperature: 1.0\ntop_p: 1.0\nmax_output_tokens: 8192\n'
QWEN_SAMPLING = 'temperature: 0.6\ntop_p: 0.95\ntop_k: 20\nmax_output_tokens: 8192\n'
LADDER = []  # @define (model names to try, in order; empty = all)


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


for k, v in {
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'VLLM_ENGINE_READY_TIMEOUT_S': '1500',
    'VLLM_NO_USAGE_STATS': '1', 'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ[k] = v


def start_server(model_dir, gpu_memory_utilization=0.92, tensor_parallel_size=None, **cfg):
    from adk_submission import VllmConfig, VllmServer
    server = VllmServer(VllmConfig(
        model=str(model_dir), port=8000, host='127.0.0.1', dtype='auto',
        gpu_memory_utilization=gpu_memory_utilization,
        enable_auto_tool_choice=True, enable_lora=False, tensor_parallel_size=tensor_parallel_size or TP,
        startup_timeout=1500, **cfg,
    ))
    server.start()
    return server


def server_alive(server):
    import urllib.request
    try:
        with urllib.request.urlopen(server.base_url + '/models', timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def run_eval(server, tag, out_dir, sampling, budget_s, extra):
    cmd = [sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--base-url', server.base_url,
           '--model-id', str(getattr(server, 'effective_model_path', None) or server.config.model),
           '--tag', tag, '--out', str(out_dir), '--sampling', sampling, '--budget-s', str(int(budget_s)),
           *(['--extra-wheels', str(EXTRA_WHEELS)] if EXTRA_WHEELS else []), *extra]
    rc = subprocess.run(cmd, timeout=budget_s + 900).returncode
    f = out_dir / 'summary_smoke.json'
    summary = json.loads(f.read_text()) if f.exists() else None
    if summary:
        summary.pop('run_ids', None)
    return rc, summary


server = None
try:
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    stage('env', python=sys.version.split()[0], gpus=gpus, cpus=os.cpu_count())
    TP = 4 if len(gpus) >= 4 else 2 if len(gpus) >= 2 else 0
    if TP == 0:
        raise SystemExit(f'expected >=2 GPUs, got {len(gpus)}: stopping to save quota')

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
    stage('wheelhouse_installed', wheelhouse=str(wh))

    import vllm
    cu = Path(vllm.__file__).parent / 'entrypoints' / 'chat_utils.py'
    src = cu.read_text()
    old = 'reasoning = message.get("reasoning")'
    if old in src and 'reasoning_content' not in src.split(old, 1)[1][:80]:
        cu.write_text(src.replace(old, 'reasoning = message.get("reasoning") or message.get("reasoning_content")', 1))
        stage('vllm_reasoning_patch', applied=True)
    else:
        stage('vllm_reasoning_patch', applied=False)

    tk = find('o200k_base.tiktoken')
    if tk:
        tk_dir = Path('/tmp/tiktoken')
        tk_dir.mkdir(exist_ok=True)
        for f in tk.parent.iterdir():
            if f.is_file():
                (tk_dir / f.name).write_bytes(f.read_bytes())
        for var in ('TIKTOKEN_ENCODINGS_BASE', 'TIKTOKEN_RS_CACHE_DIR', 'TIKTOKEN_CACHE_DIR'):
            os.environ[var] = str(tk_dir)
    DATA = find('tasks.jsonl').parent
    extra_whl = find('dirty_equals-*.whl')
    EXTRA_WHEELS = extra_whl.parent if extra_whl else None
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    stage('ready', data=str(DATA), tiktoken=str(tk.parent) if tk else None, extra_wheels=str(EXTRA_WHEELS))

    gptoss = find('gpt-oss-120b/*/default/*/config.json') or find('gpt-oss-120b/*/*/*/config.json')
    gptoss20 = find('gpt-oss-20b/*/default/*/config.json') or find('gpt-oss-20b/*/*/*/config.json')
    qwen38 = find('qwen3-8-27b-fp8-official/*/*/*/config.json')
    gemma = (find('gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*/config.json')
             or find('gemma-4-31b-it-qat-w4a16-ct/*/config.json'))
    lean = ['--max-num-seqs', '16', '--max-num-batched-tokens', '2048', '--max-cudagraph-capture-size', '32']
    # gpt-oss request fixes: no thinking_token_budget (server has no --reasoning-config; v2 run: every request 400'd)
    # and explicit harmony stop tokens <|call|>,<|return|> (vLLM 0.19.1 drops its own defaults; v4 canary: 49-87
    # blind tool calls in one turn -> context overflow)
    no_budget = ['--drop-thinking-budget', '--stop-token-ids', '200012,200002']
    ladder = [
        # Qwen3.8-27B (arch Qwen3_5ForConditionalGeneration, in vLLM 0.19.1's registry); XML tool calls via
        # qwen3_coder; text-only (no vision memory); no --reasoning-config -> drop the harness thinking budget
        # Qwen3.8's chat template defaults reasoning_effort to 'xhigh' (very long thinking every turn); use medium.
        # Rung A: 2 replicas x TP=2 (31 GB FP8 fits on 2 L4s; avoids 4-way all-reduce over PCIe). Rung B: TP=4.
        ('qwen3.8-27b-fp8', qwen38, QWEN_SAMPLING, ['--drop-thinking-budget', '--concurrency', '24'],
         dict(tool_call_parser='qwen3_coder', reasoning_parser='qwen3', max_model_len=65536,
              gpu_memory_utilization=0.92, tensor_parallel_size=2,
              default_chat_template_kwargs={'reasoning_effort': 'medium'},
              extra_args=['--data-parallel-size', '2', '--max-num-seqs', '16', '--language-model-only'])),
        ('qwen3.8-27b-fp8', qwen38, QWEN_SAMPLING, ['--drop-thinking-budget', '--concurrency', '24'],
         dict(tool_call_parser='qwen3_coder', reasoning_parser='qwen3', max_model_len=65536,
              gpu_memory_utilization=0.92, default_chat_template_kwargs={'reasoning_effort': 'medium'},
              extra_args=['--max-num-seqs', '32', '--language-model-only'])),
        ('gpt-oss-120b', gptoss, GPTOSS_SAMPLING, no_budget,
         dict(tool_call_parser='openai', reasoning_parser='openai_gptoss', max_model_len=65536,
              gpu_memory_utilization=0.96, extra_args=lean)),
        ('gpt-oss-120b', gptoss, GPTOSS_SAMPLING, no_budget,
         dict(tool_call_parser='openai', reasoning_parser='openai_gptoss', max_model_len=65536,
              gpu_memory_utilization=0.97, kv_cache_dtype='fp8', extra_args=[*lean, '--enforce-eager'])),
        ('gpt-oss-20b', gptoss20, GPTOSS_SAMPLING, no_budget,
         dict(tool_call_parser='openai', reasoning_parser='openai_gptoss', max_model_len=65536,
              gpu_memory_utilization=0.92, extra_args=['--max-num-seqs', '32'])),
        ('gemma-4-31b-qat', gemma, GEMMA_SAMPLING, [],
         dict(tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768)),
    ]
    if LADDER:
        ladder = [r for r in ladder if r[0] in LADDER]
    ladder_path = {name: (pth.parent if pth else None) for name, pth, *_ in ladder}
    rung_cfg = None
    if TP < 4:  # 2xL4 (48 GB): gpt-oss-120b (65 GB of weights) cannot fit
        ladder = [r for r in ladder if r[0] != 'gpt-oss-120b']
    stage('ladder', tp=TP, models=[r[0] for r in ladder])
    tag = None
    for i, (cand, path, sampling, eval_flags, cfg) in enumerate(ladder):
        if path is None:
            stage('skip', model=cand, reason='weights not mounted')
            continue
        t = time.time()
        try:
            server = start_server(path.parent, **cfg)
            stage('server_up', model=cand, rung=i, startup_s=round(time.time() - t, 1),
                  cfg={k: v for k, v in cfg.items() if k != 'extra_args'}, extra_args=cfg.get('extra_args'))
            # canary: 2 small solvable tasks must show real multi-turn agent activity before we spend hours
            rc, canary = run_eval(server, cand, OUT / cand / f'canary_rung{i}', sampling, 15 * 60,
                                  ['--tasks', 'canary', *[f for f in eval_flags if f not in ('--concurrency', '24')],
                                   '--concurrency', '2', '--max-tool-calls', '100',
                                   '--max-time-minutes', '10', '--max-turns', '150', '--timeout-seconds', '120'])
            stage('canary', model=cand, rung=i, returncode=rc, summary=canary)
            if not (canary and canary.get('healthy') and canary.get('n_resolved', 0) >= 1
                    and canary.get('tool_calls_per_llm_call', 99) <= 3):
                raise RuntimeError(f'canary failed (need >=3 LLM calls, >=1 solved, <=3 tool calls per LLM call): '
                                   f'{canary and {k: canary.get(k) for k in ("n_resolved", "n_llm_calls", "tool_calls_per_llm_call", "errors_sample")}}')
            tag = cand
            rung_cfg = cfg
            break
        except Exception as e:  # noqa: BLE001
            stage('rung_failed', model=cand, rung=i, after_s=round(time.time() - t, 1),
                  error=f'{type(e).__name__}: {e}'[-6000:])
            try:
                server and server.stop()
            except Exception:  # noqa: BLE001
                pass
            server = None
    if server is None:
        raise RuntimeError('no model could be served')

    k = 0
    restarts = 0
    while left() > MIN_PASS_S:
        if not server_alive(server):
            restarts += 1
            stage('server_dead', model=tag, k=k, restarts=restarts)
            if restarts > 2:
                raise RuntimeError('server keeps dying; stopping to save quota')
            try:
                server.stop()
            except Exception:  # noqa: BLE001
                pass
            server = start_server(ladder_path[tag], **rung_cfg)
            stage('server_restarted', model=tag)
        budget = left() - 10 * 60
        stage('pass_start', model=tag, k=k, budget_min=round(budget / 60, 1))
        rc, summary = run_eval(server, tag, OUT / tag / f'pass{k}', sampling, budget,
                               ['--tasks', 'all', '--order-seed', str(1000 + k), *BUDGETS, *eval_flags])
        stage('pass_done', model=tag, k=k, returncode=rc, summary=summary)
        if summary and summary.get('n_finished', 0) >= 20 and not summary.get('healthy'):
            raise RuntimeError('pass unhealthy (no task made >=3 LLM calls); stopping to save quota')
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
