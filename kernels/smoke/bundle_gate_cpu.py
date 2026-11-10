"""Bundle gate on CPU: does each candidate bundle load and run end to end in the official harness?

No GPU and no real model. A scripted fake OpenAI-compatible server answers every agent the same way: first one
`run_command` tool call (it appends a newline to README.md), then `submit_patch` if the agent has that tool, then a
final text. Each bundle runs on the two canary tasks through the official Evaluator, so a
bundle that fails to compile (bad include, unknown tool or skill, wrong sub-agent wiring, bad eval_config) fails here and
not in a daily submission or a GPU session. Also recorded per bundle and agent: tools declared, system-prompt size,
request size, and whether text from the first agent reaches the second one (`output_key` -> `{key?}`).
Outputs: /kaggle/working/gate/report.json.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
BUNDLES = None  # @embed_dir ../bundles/sub-v5b ../bundles/sub-v6 ../bundles/public-hsiaosuan-013 ../bundles/sub-v4 ../bundles/sub-v1
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'gate'
OUT.mkdir(parents=True, exist_ok=True)
PORT, MODEL, MARKER = 8011, 'fake-gemma', 'PLAN_MARKER_7781'
DUMP_BUNDLES = ('sub-v5b', 'sub-v6')
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': [], 'bundles': {}}
STATE = {'bundle': None, 'seen': {}, 'log': []}
LOCK = threading.Lock()


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:2500]}', flush=True)


def find(pattern_tail, depth=6):
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


def arg_name(tools):
    props = next(t for t in tools if t.get('function', {}).get('name') == 'run_command')['function'] \
        .get('parameters', {}).get('properties', {})
    return 'command' if 'command' in props or not props else next(iter(props))


class Fake(BaseHTTPRequestHandler):
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
            self._send(json.dumps({'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]}).encode())
        else:
            self._send(b'', 'text/plain')

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        msgs, tools = req.get('messages') or [], req.get('tools') or []
        names = [t.get('function', {}).get('name') for t in tools]
        system = ' '.join(text_of(m) for m in msgs if m.get('role') in ('system', 'developer'))
        users = [text_of(m) for m in msgs if m.get('role') == 'user']
        key = hashlib.md5((system[:400] + '|' + ','.join(map(str, names)) + '|' + (users[0][:300] if users else '')).encode()).hexdigest()
        with LOCK:
            n = STATE['seen'].get(key, 0)
            STATE['seen'][key] = n + 1
            agent_no = list(STATE['seen']).index(key)
        everything = system + ' ' + ' '.join(text_of(m) for m in msgs)
        call = None
        if n == 0 and 'run_command' in names:
            props = next(t for t in tools if t.get('function', {}).get('name') == 'run_command')['function'] \
                .get('parameters', {}).get('properties', {})
            arg = 'command' if 'command' in props or not props else next(iter(props))
            call = {'id': f'call_{key[:8]}', 'type': 'function',
                    'function': {'name': 'run_command',
                                 'arguments': json.dumps({arg: "echo gate-ok && printf '\\n' >> README.md && git status --short | head -5"})}}
        elif n == 1 and 'read_file' in names:
            call = {'id': f'call_{key[:8]}r', 'type': 'function',
                    'function': {'name': 'read_file', 'arguments': json.dumps({'filepath': 'README.md', 'start_line': 1, 'end_line': 5})}}
        elif n == 2 and 'edit_file' in names:
            call = {'id': f'call_{key[:8]}e', 'type': 'function',
                    'function': {'name': 'edit_file', 'arguments': json.dumps({'filepath': 'README.md', 'old_string': 'NO_SUCH_TEXT_caf\u00e9', 'new_string': 'x'})}}
        elif n == 3 and 'run_command' in names:
            call = {'id': f'call_{key[:8]}f', 'type': 'function',
                    'function': {'name': 'run_command', 'arguments': json.dumps({arg_name(tools): "python3 -c \"print('caf\\u00e9'); raise SystemExit(3)\""})}}
        elif n == 4 and 'submit_patch' in names:
            call = {'id': f'call_{key[:8]}s', 'type': 'function', 'function': {'name': 'submit_patch', 'arguments': '{}'}}
        content = None if call else f'{MARKER} agent#{agent_no}: edit the file named in the issue; assert the new behaviour.'
        with LOCK:
            STATE['n_req'] = STATE.get('n_req', 0) + 1
            if STATE['bundle'] in DUMP_BUNDLES and STATE['n_req'] <= 14:  # full request bodies of the first task
                d_ = OUT / 'transcript'
                d_.mkdir(exist_ok=True)
                (d_ / f"{STATE['bundle']}_{STATE['n_req']:02d}.json").write_text(json.dumps(req, indent=1))
            STATE['log'].append({'bundle': STATE['bundle'], 'agent_key': key[:8], 'turn': n, 'tools': names,
                                 'system_chars': len(system), 'request_chars': len(json.dumps(req)),
                                 'tools_chars': len(json.dumps(tools)), 'n_messages': len(msgs),
                                 'stream': bool(req.get('stream')), 'marker_in_request': MARKER in everything,
                                 'marker_in_system': MARKER in system,
                                 'action': call['function']['name'] if call else 'text',
                                 'user_chars': sum(len(u) for u in users),
                                 'extra': {k: v for k, v in req.items() if k not in ('messages', 'tools', 'model')}})
        base = {'id': 'chatcmpl-gate', 'created': int(time.time()), 'model': req.get('model') or MODEL}
        finish = 'tool_calls' if call else 'stop'
        if req.get('stream'):
            delta = {'role': 'assistant', 'content': content}
            if call:
                delta['tool_calls'] = [{'index': 0, **call}]
            chunks = [{**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                      {**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                       'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}]
            body = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n'
            self._send(body.encode(), 'text/event-stream')
        else:
            msg = {'role': 'assistant', 'content': content,
                   'reasoning_content': f'THOUGHT_MARKER_{n}: I should look at the code before editing. caf\u00e9'}
            if call:
                msg['tool_calls'] = [call]
            self._send(json.dumps({**base, 'object': 'chat.completion',
                                   'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}],
                                   'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}).encode())


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    # harness wheels (same install rule as controls_cpu: only wheels this interpreter supports)
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    DATA = find('tasks.jsonl').parent
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    stage('ready', wheels=len(ok), python=sys.version.split()[0], data=str(DATA))

    server = ThreadingHTTPServer(('127.0.0.1', PORT), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    for name, files in BUNDLES.items():
        bdir = WORK / 'bundles' / name
        for rel, text in files.items():
            if rel == 'SOURCE.txt':
                continue
            (bdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (bdir / rel).write_text(text)
        rec = REPORT['bundles'][name] = {'files': sorted(f for f in files if f != 'SOURCE.txt'),
                                         'eval_config': files.get('eval_config.yaml')}
        # static checks the public notebooks use before zipping
        try:
            from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES
            bad = [f for f in rec['files'] if Path(f).suffix.lower() not in set(ALLOWED_SUBMISSION_EXTENSIONS)]
            rec['static'] = {'bad_extensions': bad, 'bytes': sum(len(files[f].encode()) for f in rec['files']),
                             'max_bytes': MAX_SUBMISSION_SIZE_BYTES}
        except Exception as e:  # noqa: BLE001
            rec['static'] = {'error': f'{type(e).__name__}: {e}'[:300]}
        try:
            from swegemma.models.discovery import validate_single_declared_model
            rec['declared_model'] = str(validate_single_declared_model(bdir))
        except Exception as e:  # noqa: BLE001
            rec['declared_model_error'] = f'{type(e).__name__}: {e}'[:400]
        with LOCK:
            STATE.update(bundle=name, seen={}, n_req=0)
        out_dir = OUT / name
        t1 = time.time()
        p = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', name,
                            '--out', str(out_dir), '--tasks', 'canary', '--bundle', str(bdir),
                            '--base-url', f'http://127.0.0.1:{PORT}/v1', '--model-id', MODEL, '--api-key', 'x',
                            '--concurrency', '1', '--budget-s', '900'], capture_output=True, text=True, timeout=1500)
        rec.update(returncode=p.returncode, secs=round(time.time() - t1, 1), log_tail=(p.stdout + p.stderr)[-1500:])
        f = out_dir / 'results' / 'task_results.jsonl'
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        with LOCK:
            calls = [c for c in STATE['log'] if c['bundle'] == name]
        agents = {}
        for c in calls:
            a_ = agents.setdefault(c['agent_key'], {'requests': 0, 'tools': c['tools'], 'system_chars': c['system_chars'],
                                                     'tools_chars': c['tools_chars'], 'first_request_chars': c['request_chars'],
                                                     'marker_in_first_request': c['marker_in_request'],
                                                     'marker_in_system': c['marker_in_system'], 'stream': c['stream'],
                                                     'user_chars': c['user_chars'], 'extra': c['extra'], 'actions': []})
            a_['requests'] += 1
            a_['actions'].append(c['action'])
        rec.update(tasks=[{k: r.get(k) for k in ('instance_id', 'resolved', 'error', 'tool_calls', 'total_llm_calls',
                                                 'duration_seconds', 'agent_patch_size', 'test_exit_code')} for r in rows],
                   llm_requests=len(calls), agents=list(agents.values()))
        n_agents_per_task = len(agents) / max(1, len(rows))
        rec['verdict'] = {
            'ran_both_tasks': len(rows) == 2,
            'no_harness_error': all(not r.get('error') or 'Required test node' in str(r.get('error'))
                                    or 'submit_patch' in str(r.get('error')) for r in rows) and bool(rows),
            'patch_reached_verification': all((r.get('agent_patch_size') or 0) > 0 for r in rows) and bool(rows),
            'tools_executed': all((r.get('tool_calls') or 0) >= 1 for r in rows) and bool(rows),
            'agents_per_task': n_agents_per_task,
            'handoff_text_reaches_later_agent': any(a_['marker_in_first_request'] for a_ in list(agents.values())[1:])
            if n_agents_per_task > 1 else None}
        rec['pass'] = bool(rec['verdict']['ran_both_tasks'] and rec['verdict']['no_harness_error']
                           and rec['verdict']['tools_executed'])
        stage(f'bundle_{name}', **{k: v for k, v in rec.items() if k not in ('files', 'log_tail')},
              log_tail=rec['log_tail'][-500:] if not rec['pass'] else '')
    stage('summary', **{n: {'pass': r.get('pass'), **(r.get('verdict') or {}),
                            'system_chars': [a_['system_chars'] for a_ in r.get('agents', [])],
                            'tools_chars': [a_['tools_chars'] for a_ in r.get('agents', [])],
                            'n_tools': [len(a_['tools']) for a_ in r.get('agents', [])]}
                        for n, r in REPORT['bundles'].items()})
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-3000:], tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
