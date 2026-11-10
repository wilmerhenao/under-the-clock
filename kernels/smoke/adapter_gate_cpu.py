"""Adapter submission dry run on CPU: does sub-v5b with a LoRA adapter (adapters/<name>/ + `adapter:` on the root agent)
pass the official checks, and do the agent's requests reach the adapter?

1. Rank-16 LoRA weights with the real Gemma 4 31B shapes for train_lora.py's targets (q/k/v/o/gate/up/down of the
   language model): shapes from the safetensors headers of the unquantized QAT checkpoint (no weights loaded); PEFT key
   names; zeros. adapter_config.json exactly as train_lora.py writes it (template from a local run).
2. Bundle = sub-v5b + `adapter: <name>` + adapters/<name>/{adapter_config.json, adapter_model.safetensors, README.md};
   zipped like the notebook does, then unzipped again (the scorer unpacks submission.zip).
3. Official static checks on the unzipped dir: adk_submission validate_directory (size, file count, extensions, root
   config), discover_adapters (name, inferred rank, size), declared model.
4. The two canary tasks through the official Evaluator against a fake OpenAI server that logs the requested model id;
   control run without the `adapter:` line.
Pass = checks clean, every request of the adapter run names the adapter, the control names the base model.
Outputs: /kaggle/working/adapter_gate/report.json
"""
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
BUNDLES = None  # @embed_dir ../bundles/sub-v5b ../bundles/sub-v8 ../bundles/sub-v9
REAL_ADAPTER_GLOB = ''  # @define (a trained adapter's adapter_model.safetensors: gate it and inspect its weights instead of a zero adapter)
BASE_SAMPLE_LAYERS = [0, 30, 59]  # @define (layers whose base weights are read to put the adapter's change in proportion)
GATE_BUNDLE = 'sub-v5b'  # @define (the bundle under test)
ADAPTER = 'g4dev_sft_v1'  # @define
RANK = 16  # @define
ADAPTER_CONFIG = {
    "alora_invocation_tokens": None, "alpha_pattern": {}, "arrow_config": None, "auto_mapping": None,
    "base_model_name_or_path": "google/gemma-4-31b-it-qat-w4a16-ct", "bias": "none", "corda_config": None,
    "ensure_weight_tying": False, "eva_config": None, "exclude_modules": None, "fan_in_fan_out": False,
    "inference_mode": True, "init_lora_weights": True, "layer_replication": None, "layers_pattern": None,
    "layers_to_transform": None, "loftq_config": {}, "lora_alpha": 32, "lora_bias": False, "lora_dropout": 0.05,
    "lora_ga_config": None, "megatron_config": None, "megatron_core": "megatron.core", "modules_to_save": None,
    "peft_type": "LORA", "peft_version": "0.19.1", "qalora_group_size": 16, "r": 16, "rank_pattern": {}, "revision": None,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "target_parameters": None, "task_type": "CAUSAL_LM", "trainable_token_indices": None, "use_bdlora": None,
    "use_dora": False, "use_qalora": False, "use_rslora": False}
TARGET_RX = re.compile(r'language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$')
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'adapter_gate'
OUT.mkdir(parents=True, exist_ok=True)
PORT, MODEL = 8012, 'fake-gemma'
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
STATE = {'run': None, 'seen': {}, 'log': []}
LOCK = threading.Lock()


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:2500]}', flush=True)


def find(pattern_tail, depth=7):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def text_of(m):
    c = m.get('content')
    if isinstance(c, list):
        return ' '.join(str(p.get('text', '')) if isinstance(p, dict) else str(p) for p in c)
    return str(c or '')


class Fake(BaseHTTPRequestHandler):
    """First request of a conversation: one run_command; second: submit_patch; then a final text."""
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _send(self, body, ctype='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            self._send(json.dumps({'object': 'list', 'data': [{'id': MODEL, 'object': 'model'},
                                                               {'id': ADAPTER, 'object': 'model'}]}).encode())
        else:
            self._send(b'', 'text/plain')

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        msgs, tools = req.get('messages') or [], req.get('tools') or []
        names = [t.get('function', {}).get('name') for t in tools]
        users = [text_of(m) for m in msgs if m.get('role') == 'user']
        key = hashlib.md5(((users[0][:300] if users else '') + ','.join(map(str, names))).encode()).hexdigest()
        with LOCK:
            n = STATE['seen'].get(key, 0)
            STATE['seen'][key] = n + 1
            sys_txt = str((msgs[0] or {}).get('content') if msgs else '')
            usr_txt = next((str(m_.get('content')) for m_ in msgs if m_.get('role') == 'user'), '')
            budget = re.search(r'Tool calls allowance: (\d+) calls', usr_txt)
            tmin = re.search(r'Time allowance: ([\d.]+) minutes', usr_txt)
            STATE['log'].append({'run': STATE['run'], 'model': req.get('model'), 'turn': n, 'stream': bool(req.get('stream')),
                                 'diff_line': 'git diff' in sys_txt and 'For context:' in sys_txt,
                                 'calls_budget': budget.group(1) if budget else None, 'time_budget': tmin.group(1) if tmin else None})
        call = None
        if n == 0 and 'run_command' in names:
            call = {'id': f'call_{key[:8]}', 'type': 'function', 'function': {
                'name': 'run_command', 'arguments': json.dumps({'command': "printf '\\n' >> README.md && git status --short | head -3"})}}
        elif n == 1 and 'submit_patch' in names:
            call = {'id': f'call_{key[:8]}s', 'type': 'function', 'function': {'name': 'submit_patch', 'arguments': '{}'}}
        content = None if call else 'done'
        base = {'id': 'chatcmpl-agate', 'created': int(time.time()), 'model': req.get('model') or MODEL}
        finish = 'tool_calls' if call else 'stop'
        usage = {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}
        if req.get('stream'):
            delta = {'role': 'assistant', 'content': content}
            if call:
                delta['tool_calls'] = [{'index': 0, **call}]
            chunks = [{**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                      {**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                       'usage': usage}]
            self._send((''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n').encode(), 'text/event-stream')
        else:
            msg = {'role': 'assistant', 'content': content}
            if call:
                msg['tool_calls'] = [call]
            self._send(json.dumps({**base, 'object': 'chat.completion', 'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}],
                                   'usage': usage}).encode())


def inspect_adapter(path, cfg, ckpt_files):
    """Trained LoRA weights: finite?, which modules moved (||dW||_F = alpha/r * sqrt(tr((B^T B)(A A^T)))), by module
    type and by layer, and ||dW|| / ||W0|| on a sample of layers (base weights read one tensor at a time)."""
    import torch
    from safetensors import safe_open
    scale = cfg.get('lora_alpha', 2 * RANK) / cfg.get('r', RANK)
    pairs, bad, n_params = {}, [], 0
    with safe_open(str(path), framework='pt') as f:
        for k in f.keys():
            t = f.get_tensor(k).float()
            n_params += t.numel()
            if not torch.isfinite(t).all():
                bad.append(k)
            m = re.match(r'base_model\.model\.(.*)\.lora_([AB])\.weight$', k)
            if m:
                pairs.setdefault(m.group(1), {})[m.group(2)] = t
    rows = []
    for mod, ab in pairs.items():
        if 'A' not in ab or 'B' not in ab:
            continue
        A, B = ab['A'], ab['B']
        dw = scale * float(torch.sqrt(torch.clamp(torch.trace((B.T @ B) @ (A @ A.T)), min=0)))
        lay = re.search(r'\.layers\.(\d+)\.', mod)
        rows.append({'module': mod, 'kind': mod.rsplit('.', 1)[-1], 'layer': int(lay.group(1)) if lay else -1,
                     'A': round(float(A.norm()), 4), 'B': round(float(B.norm()), 6), 'dW': dw})
    by_kind, by_layer = {}, {}
    for r in rows:
        by_kind.setdefault(r['kind'], []).append(r['dW'])
        by_layer.setdefault(r['layer'], []).append(r['dW'])
    rel = []
    want = {r['module'] for r in rows if r['layer'] in BASE_SAMPLE_LAYERS}
    for fpath in ckpt_files:
        with safe_open(str(fpath), framework='pt') as f:
            for k in f.keys():
                mod = k[:-len('.weight')] if k.endswith('.weight') else None
                if mod in want:
                    w0 = float(f.get_tensor(k).float().norm())
                    r = next(x for x in rows if x['module'] == mod)
                    rel.append({'module': mod, 'w0': round(w0, 2), 'dW': round(r['dW'], 5), 'rel': r['dW'] / max(w0, 1e-9)})
    rel.sort(key=lambda x: -x['rel'])
    return {'tensors': len(list(pairs)) * 2, 'params': n_params, 'non_finite': bad[:5], 'modules': len(rows),
            'untrained_B': sum(1 for r in rows if r['B'] == 0), 'scale': scale,
            'dW_by_kind': {k: round(sum(v) / len(v), 5) for k, v in sorted(by_kind.items())},
            'dW_by_layer': {k: round(sum(v) / len(v), 5) for k, v in sorted(by_layer.items())},
            'top_modules': sorted(rows, key=lambda r: -r['dW'])[:10],
            'relative_change_sample': {'n': len(rel), 'median': sorted(x['rel'] for x in rel)[len(rel) // 2] if rel else None,
                                       'max': rel[:5]}}


def write_lora(path, shapes):
    """PEFT-named zero LoRA tensors in safetensors format (bf16), header padded to 8 bytes."""
    header, offset, order = {'__metadata__': {'format': 'pt'}}, 0, []
    for base_name, (out_f, in_f) in sorted(shapes.items()):
        stem = 'base_model.model.' + base_name[:-len('.weight')]
        for suffix, shape in (('lora_A.weight', [RANK, in_f]), ('lora_B.weight', [out_f, RANK])):
            nbytes = shape[0] * shape[1] * 2
            header[f'{stem}.{suffix}'] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
            order.append(nbytes)
            offset += nbytes
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(raw)))
        fh.write(raw)
        chunk = b'\x00' * (1 << 20)
        for nbytes in order:
            while nbytes > 0:
                k = min(nbytes, len(chunk))
                fh.write(chunk[:k])
                nbytes -= k
    return offset


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    DATA = find('tasks.jsonl').parent
    (WORK / 'eval_harness.py').write_text(EVAL_PY)

    # 1. LoRA shapes from the checkpoint headers
    ckpt = [Path(p) for p in sorted(glob.glob('/kaggle/input/**/*unquantized*/**/*.safetensors', recursive=True))]
    shapes = {}
    for p in ckpt:
        with open(p, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name != '__metadata__' and TARGET_RX.search(name):
                shapes[name] = tuple(meta['shape'])
    by_kind = {}
    for name, shp in shapes.items():
        by_kind.setdefault(name.rsplit('.', 2)[-2], set()).add(shp)
    stage('shapes', checkpoint_files=len(ckpt), modules=len(shapes), by_kind={k: sorted(v) for k, v in by_kind.items()},
          example=next(iter(sorted(shapes)), None))
    if not shapes:
        raise RuntimeError('no target modules found in the checkpoint headers')

    # 2. bundle with the adapter
    bundle = OUT / 'bundle'
    shutil.rmtree(bundle, ignore_errors=True)
    for rel, text in BUNDLES[GATE_BUNDLE].items():
        if rel == 'SOURCE.txt':
            continue
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_text(text)
    agent_yaml = (bundle / 'agent.yaml').read_text()
    if not re.search(r'^model:.*$', agent_yaml, re.M):
        raise RuntimeError('agent.yaml has no model line')
    (bundle / 'agent.yaml').write_text(re.sub(r'^(model:.*)$', rf'\1\nadapter: {ADAPTER}', agent_yaml, count=1, flags=re.M))
    adir = bundle / 'adapters' / ADAPTER
    adir.mkdir(parents=True)
    real = sorted(glob.glob(REAL_ADAPTER_GLOB, recursive=True)) if REAL_ADAPTER_GLOB else []
    if REAL_ADAPTER_GLOB and not real:
        raise RuntimeError(f'no trained adapter under {REAL_ADAPTER_GLOB}')
    if real:
        src = Path(real[0])
        shutil.copy(src, adir / 'adapter_model.safetensors')
        shutil.copy(src.parent / 'adapter_config.json', adir / 'adapter_config.json')
        (adir / 'README.md').write_text('LoRA adapter for gemma-4-31b-it-qat-w4a16-ct (trained, under inspection).\n')
        nbytes = (adir / 'adapter_model.safetensors').stat().st_size
        stage('real_adapter', source=str(src), bytes=nbytes, config=json.loads((adir / 'adapter_config.json').read_text()))
        stage('inspect', **inspect_adapter(adir / 'adapter_model.safetensors', json.loads((adir / 'adapter_config.json').read_text()), ckpt))
    else:
        (adir / 'adapter_config.json').write_text(json.dumps({**ADAPTER_CONFIG, 'r': RANK, 'lora_alpha': 2 * RANK}, indent=2))
        (adir / 'README.md').write_text('LoRA adapter for gemma-4-31b-it-qat-w4a16-ct (dry run: zero weights).\n')
        nbytes = write_lora(adir / 'adapter_model.safetensors', shapes)
    zpath = OUT / 'submission.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(bundle).as_posix(),
                        compress_type=zipfile.ZIP_STORED if f.suffix == '.safetensors' else zipfile.ZIP_DEFLATED)
    unz = OUT / 'unzipped'
    shutil.rmtree(unz, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(unz)
        names = z.namelist()
    stage('bundle', adapter_bytes=nbytes, zip_bytes=zpath.stat().st_size, zip_entries=names,
          agent_yaml=(unz / 'agent.yaml').read_text()[:400])

    # 3. official static checks
    from adk_submission.discovery import discover_adapters, validate_directory
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, build_submission_limits
    limits, _gen = build_submission_limits()
    checks = {}
    try:
        sd = validate_directory(unz, limits)
        checks['validate_directory'] = {'ok': True, **{k: (str(v)[:300]) for k, v in vars(sd).items()}} if hasattr(sd, '__dict__') else {'ok': True}
    except Exception as e:  # noqa: BLE001
        checks['validate_directory'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:600]}
    try:
        man = discover_adapters(unz, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        checks['discover_adapters'] = {'ok': ADAPTER in man.adapters,
                                       'adapters': {k: {'rank': v.rank, 'size_bytes': v.size_bytes, 'format': v.format,
                                                        'path': str(v.path)} for k, v in man.adapters.items()}}
    except Exception as e:  # noqa: BLE001
        checks['discover_adapters'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:600]}
    try:
        from swegemma.models.discovery import validate_single_declared_model
        checks['declared_model'] = {'ok': True, 'model': str(validate_single_declared_model(unz))}
    except Exception as e:  # noqa: BLE001
        checks['declared_model'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:600]}
    checks['limits'] = {k: getattr(limits, k, None) for k in ('max_total_size_bytes', 'max_files', 'adapter_extensions')}
    stage('static_checks', **checks)

    # 4. Evaluator: adapter run, then control without the adapter line
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    control = OUT / 'control_bundle'
    shutil.rmtree(control, ignore_errors=True)
    shutil.copytree(unz, control, ignore=shutil.ignore_patterns('adapters'))
    (control / 'agent.yaml').write_text(re.sub(rf'^adapter: {ADAPTER}\n', '', (control / 'agent.yaml').read_text(), flags=re.M))
    runs = {}
    for run, bdir in (('adapter', unz), ('control', control)):
        with LOCK:
            STATE.update(run=run, seen={})
        t1 = time.time()
        p = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', run,
                            '--out', str(OUT / run), '--tasks', 'canary', '--bundle', str(bdir),
                            '--base-url', f'http://127.0.0.1:{PORT}/v1', '--model-id', MODEL, '--api-key', 'x',
                            '--concurrency', '1', '--budget-s', '900'], capture_output=True, text=True, timeout=1500)
        f = OUT / run / 'results' / 'task_results.jsonl'
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        with LOCK:
            models = sorted({str(c['model']) for c in STATE['log'] if c['run'] == run})
            n_req = sum(1 for c in STATE['log'] if c['run'] == run)
            seen_ = [c for c in STATE['log'] if c['run'] == run]
        prompt_checks = {'diff_line_in_system': sorted({c['diff_line'] for c in seen_}),
                         'task_calls_budget': sorted({str(c['calls_budget']) for c in seen_}),
                         'task_time_budget': sorted({str(c['time_budget']) for c in seen_})}
        runs[run] = {'returncode': p.returncode, 'secs': round(time.time() - t1, 1), 'requests': n_req, 'models': models,
                     'prompt_checks': prompt_checks,
                     'tasks': [{k: r.get(k) for k in ('instance_id', 'error', 'tool_calls', 'agent_patch_size')} for r in rows],
                     'log_tail': (p.stdout + p.stderr)[-1200:] if p.returncode or not rows else ''}
        stage(f'run_{run}', **runs[run])
    verdict = {
        'static_ok': all(checks[k].get('ok') for k in ('validate_directory', 'discover_adapters', 'declared_model')),
        'adapter_run_routes_to_adapter': runs['adapter']['requests'] > 0 and runs['adapter']['models'] == [ADAPTER],
        'control_routes_to_base': runs['control']['requests'] > 0 and runs['control']['models'] == [MODEL],
        'adapter_run_tasks': len(runs['adapter']['tasks']),
        'adapter_run_errors': [t['error'] for t in runs['adapter']['tasks'] if t.get('error')][:3]}
    verdict['pass'] = bool(verdict['static_ok'] and verdict['adapter_run_routes_to_adapter'] and verdict['control_routes_to_base']
                           and verdict['adapter_run_tasks'] == 2)
    stage('verdict', **verdict)
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-3000:], tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    for d in ('bundle', 'unzipped', 'control_bundle', 'submission.zip'):  # keep the output small: the zip stays
        shutil.rmtree(OUT / d, ignore_errors=True)
