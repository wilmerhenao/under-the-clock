"""Replay execution-verified SWE-Zero trajectories through the official Evaluator (CPU, scripted model, no GPU).

`g4dev-public-patches-cpu` applied the final patches of public SWE-Zero trajectories (Qwen3-Coder-480B, OpenHands tools,
no code execution: shell commands are read-only) to our task bank in the competition's own harness: 565 runs of <=30 calls
resolve 330 bank tasks. Here each of those trajectories is converted to the harness's tools and replayed: a fake
OpenAI-compatible server answers the agent with the teacher's next call (and the teacher's reasoning as the thought), the
official Evaluator executes every call for real in the task's uv environment with the sub-v5b bundle and its budgets, and
Phase 2 verification decides `resolved`. Resolved replays are the training data: the scorer's exact request/observation
format, real observations, a real label, and tasks from repositories outside the competition's four.

Conversion: think -> reasoning of the next call; str_replace_editor view -> read_file in <=150-line chunks (whole-file views
of more than 450 lines -> a class/def outline via grep; directory views -> find), str_replace -> edit_file, create ->
write_file, insert/undo_edit -> trajectory dropped; execute_bash -> run_command (repo path rewritten to /workspace);
finish -> submit_patch; task_tracker dropped.
Outputs: /kaggle/working/zero_replay/{sft_zero_v1.jsonl, scripts.jsonl, results.jsonl, stats.json, report.json, traces/}.
"""
import collections
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
BUILD_SFT_PY = None  # @embed ../train/build_sft.py
TOOLS_JSON = None  # @embed templates/harness_tools_request.json
BUNDLES = None  # @embed_dir ../bundles/sub-v5b
DATASET = 'nvidia/SWE-Zero-openhands-trajectories'  # @define
MAX_TEACHER_CALLS = 30  # @define
MAX_HARNESS_CALLS = 28  # @define (the bundle's max_tool_calls; longer conversions are not replayed)
MAX_TRAJ = 0  # @define (0 = every verified trajectory)
SMOKE_N = 24  # @define (first slice of pass 0; the run stops if none of them resolves)
REASONING_CAP = 1500  # @define (characters of teacher reasoning per call)
DEADLINE_S = 9 * 3600  # @define
SKIP_GLOB = ''  # @define (results.jsonl of earlier replays mounted as kernel sources: their trajectories are skipped)
CAPTURE_N = 0  # @define (save every request body and scripted answer of the first N matched tasks: zero_replay/captured/)
PARITY = False  # @define (render each captured request as vLLM 0.19.1 does and diff it against the training text)
REAL_USAGE = False  # @define (report real prompt token counts, so the harness's events compaction triggers as in scoring)
PREFER_LONG = False  # @define (with MAX_TRAJ: keep the longest conversions, the ones that cross the compaction threshold)
SUMMARIZE = False  # @define (answer the harness's summary requests with compaction_sim.summarize, as a trained model would)
INJECT_DIFF = False  # @define (with SUMMARIZE: the first call after each compaction is `cd /workspace && git diff`, then the script resumes)
COMPACT_SIM_PY = None  # @embed ../train/compaction_sim.py
CS = None
if SUMMARIZE:
    _ns = {'__name__': 'compaction_sim'}
    exec(compile(COMPACT_SIM_PY, 'compaction_sim.py', 'exec'), _ns)
    CS = types.SimpleNamespace(**_ns)
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'zero_replay'
(OUT / 'traces').mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/zr')
TMP.mkdir(parents=True, exist_ok=True)
HUB = 'https://huggingface.co'
PORT, MODEL = 8013, 'fake-gemma'
FINAL_TEXT = 'The fix is implemented and the patch is submitted.'
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
LOCK = threading.Lock()
STATE = {'scripts': {}, 'snips': {}, 'key2iid': {}, 'turn': collections.Counter(), 'tools': None,
         'unmatched': 0, 'requests': 0, 'beyond_script': 0}


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


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-zero-replay'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep((40 if '429' in str(e) else 4) * (k + 1))


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-zero-replay'})
            with urllib.request.urlopen(req, timeout=900) as r, open(dest, 'wb') as fh:
                shutil.copyfileobj(r, fh, 1 << 22)
            with open(dest, 'rb') as fh:  # a truncated download has no parquet footer
                fh.seek(-4, 2)
                if fh.read(4) == b'PAR1':
                    return dest.stat().st_size
            raise RuntimeError('no parquet footer (truncated download)')
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def norm(s):
    return re.sub(r'\s+', ' ', s or '').strip()


def text_of(m):
    c = m.get('content') if isinstance(m, dict) else m
    if isinstance(c, list):
        return '\n'.join(str(p.get('text', '')) if isinstance(p, dict) else str(p) for p in c)
    return str(c or '')


# ---------------------------------------------------------------- teacher trajectory -> harness script
class Reject(Exception):
    pass


def to_script(traj):
    """OpenHands messages -> [(reasoning, tool, args)] in the harness's tools, ending with submit_patch."""
    msgs = json.loads(traj) if isinstance(traj, str) else list(traj)
    user = next((m for m in msgs if m.get('role') == 'user'), None)
    mm = re.search(r'<uploaded_files>\s*(/workspace/[^\s<]+)', text_of(user)) if user else None
    if not mm:
        raise Reject('no_root')
    root = mm.group(1).rstrip('/')

    def rel(p):
        p = (p or '').strip()
        if p.rstrip('/') in (root, '/workspace'):
            return ''
        if p.startswith(root + '/'):
            return p[len(root) + 1:]
        return None if p.startswith('/') or p.startswith('~') else p

    steps, pending, saw_file = [], [], False

    def reason():
        t = '\n\n'.join(x.strip() for x in pending if x and x.strip()).replace(root, '/workspace')
        pending.clear()
        if len(t) > REASONING_CAP:
            cut = t[:REASONING_CAP]
            t = cut[:max(cut.rfind('. '), cut.rfind('\n'), REASONING_CAP // 2) + 1].rstrip()
        return t

    def read_chunks(r, a_, b_):
        k = a_
        while k <= b_:
            e_ = min(b_, k + 149)
            steps.append((reason(), 'read_file', {'filepath': r, 'start_line': k, 'end_line': e_}))
            k = e_ + 1

    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        if text_of(m).strip():
            pending.append(text_of(m))
        tcs = m.get('tool_calls') or []
        results = []
        j = i + 1
        while j < len(msgs) and msgs[j].get('role') == 'tool':
            results.append(text_of(msgs[j]))
            j += 1
        for k_, tc in enumerate(tcs):
            fn = tc.get('function') or {}
            name = fn.get('name')
            try:
                args = json.loads(fn['arguments']) if isinstance(fn.get('arguments'), str) else (fn.get('arguments') or {})
            except Exception:  # noqa: BLE001
                raise Reject('bad_arguments') from None
            obs = results[k_] if k_ < len(results) else ''
            if name == 'think':
                pending.append(str(args.get('thought', '')))
            elif name == 'task_tracker':
                continue
            elif name == 'finish':
                steps.append((reason(), 'submit_patch', {}))
                return steps, root
            elif name == 'execute_bash':
                cmd = str(args.get('command', '')).strip()
                if not cmd or cmd in ('C-c', 'C-d', 'C-z'):
                    continue
                steps.append((reason(), 'run_command', {'command': cmd.replace(root, '/workspace')}))
            elif name == 'str_replace_editor':
                sub, r = args.get('command'), rel(args.get('path'))
                if r is None:
                    raise Reject('path_outside_workspace')
                if sub == 'view':
                    if obs.startswith("Here's the files and directories") or (r == '' and not saw_file):
                        if r == '' and not saw_file:
                            continue  # the task message already shows the tree
                        steps.append((reason(), 'run_command',
                                      {'command': f'find {r or "."} -maxdepth 2 -not -path "*/.*" | sort | head -100'}))
                        continue
                    saw_file = True
                    shown = [int(x) for x in re.findall(r'^\s*(\d+)\t', obs, re.M)]
                    vr = args.get('view_range')
                    if isinstance(vr, list) and len(vr) == 2:
                        a_, b_ = int(vr[0]), int(vr[1])
                        if b_ == -1:
                            b_ = max(shown) if shown else a_ + 149
                        read_chunks(r, max(1, a_), min(b_, max(1, a_) + 449))
                    elif shown and max(shown) <= 450:
                        read_chunks(r, 1, max(shown))
                    elif shown:
                        steps.append((reason(), 'run_command',
                                      {'command': f'grep -n "^\\s*\\(class\\|def\\|async def\\) " {r} | head -150'}))
                    else:
                        steps.append((reason(), 'read_file', {'filepath': r}))
                elif sub == 'str_replace':
                    steps.append((reason(), 'edit_file', {'filepath': r, 'old_string': args.get('old_str') or '',
                                                          'new_string': args.get('new_str') or ''}))
                elif sub == 'create':
                    steps.append((reason(), 'write_file', {'filepath': r, 'content': args.get('file_text') or ''}))
                else:
                    raise Reject(f'editor_{sub}')
            else:
                raise Reject(f'tool_{name}')
    steps.append((reason(), 'submit_patch', {}))
    return steps, root


# ---------------------------------------------------------------- scripted model
def vllm_messages(msgs):
    """What vLLM 0.19.1's chat_utils changes before templating: string content, assistant `reasoning` kept (adk mirrors
    reasoning_content into it), tool-call arguments parsed from JSON strings into dicts (empty -> {})."""
    out = []
    for m in msgs:
        m2 = {k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name', 'reasoning')}
        if isinstance(m2.get('content'), list):
            m2['content'] = ''.join(str(p_.get('text', '')) for p_ in m2['content'] if isinstance(p_, dict))
        if m2.get('role') == 'assistant' and m2.get('tool_calls'):
            calls_ = []
            for tc in m2['tool_calls']:
                f_ = dict(tc.get('function') or {})
                a_ = f_.get('arguments')
                f_['arguments'] = json.loads(a_) if isinstance(a_, str) and a_.strip() else (a_ or {})
                calls_.append({**tc, 'function': f_})
            m2['tool_calls'] = calls_
        out.append(m2)
    return out


def render_request(req):
    kw = dict(req.get('chat_template_kwargs') or {})
    kw.setdefault('enable_thinking', True)
    return tok.apply_chat_template(vllm_messages(req.get('messages') or []), tools=req.get('tools') or None,
                                   tokenize=False, add_generation_prompt=True, **kw)


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
        msgs = req.get('messages') or []
        users = [text_of(m) for m in msgs if m.get('role') == 'user']
        first = norm(users[0]) if users else ''
        key = hashlib.md5(first[:3000].encode()).hexdigest()
        step = None
        with LOCK:
            STATE['requests'] += 1
            if STATE['tools'] is None and req.get('tools'):
                STATE['tools'] = req['tools']
            iid = STATE['key2iid'].get(key) if req.get('tools') else None  # tool-less = compaction/summary call
            if iid is None and req.get('tools'):
                scored = [(sum(s_ in first for s_ in snips), i) for i, snips in STATE['snips'].items()]
                best = max(scored, default=(0, None))
                iid = best[1] if best[0] > 0 else None  # the task whose problem-statement windows match most
                STATE['key2iid'][key] = iid
            sum_iid = None
            if not req.get('tools') and SUMMARIZE:  # whose summary request (for capture only; no turn is consumed)
                scored = [(sum(s_ in first for s_ in snips), i) for i, snips in STATE['snips'].items()]
                best = max(scored, default=(0, None))
                sum_iid = best[1] if best[0] > 0 else None
                STATE['summaries'] = STATE.get('summaries', 0) + 1
            if sum_iid is not None and INJECT_DIFF:
                STATE.setdefault('pending_diff', set()).add(sum_iid)
            injected = False
            if iid is None and sum_iid is None:
                STATE['unmatched'] += 1
            elif iid is None:
                pass  # a summary request: answered below, no script step consumed
            elif INJECT_DIFF and iid in STATE.get('pending_diff', set()):
                injected = True
                STATE['pending_diff'].discard(iid)  # recovery call after a compaction; the script position is kept
                STATE['diffs'] = STATE.get('diffs', 0) + 1
                step = ('My earlier steps were summarized. Let me check the edits I already made.', 'run_command',
                        {'command': 'cd /workspace && git diff'})
            else:
                n = STATE['turn'][iid]
                STATE['turn'][iid] = n + 1
                script = STATE['scripts'].get(iid) or []
                if n < len(script):
                    step = script[n]
                else:
                    STATE['beyond_script'] += 1
        call = None
        reasoning = ''
        if step is not None:
            reasoning, name, args = step
            call = {'id': f'call_{key[:6]}_{STATE["turn"][iid]}' + ('_diff' if injected else ''), 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}
        content = None if call else (FINAL_TEXT if req.get('tools') else
                                     'Summary of the earlier steps (replay placeholder): files were read and edited as above.')
        if not req.get('tools') and SUMMARIZE:
            prompt = text_of((req.get('messages') or [{}])[0])
            body = prompt[len(CS.PREFIX):] if prompt.startswith(CS.PREFIX) else prompt
            content = CS.summarize(re.split(r'\\n(?=(?:user|model|swe_agent): )', body))
        base = {'id': 'chatcmpl-replay', 'created': int(time.time()), 'model': req.get('model') or MODEL}
        finish = 'tool_calls' if call else 'stop'
        usage = {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}
        if REAL_USAGE:
            try:
                n_prompt = len(tok(render_request(req), add_special_tokens=False)['input_ids'])
                usage = {'prompt_tokens': n_prompt, 'completion_tokens': 50, 'total_tokens': n_prompt + 50}
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE['usage_errors'] = STATE.get('usage_errors', 0) + 1
                    STATE['usage_error'] = f'{type(e).__name__}: {e}'[:200]
        if CAPTURE_N:
            with LOCK:
                cap = STATE.setdefault('captured', {})
                who = iid if iid is not None else (sum_iid if sum_iid is not None else '_unmatched')
                if who in cap or (iid is not None and len([k for k in cap if k != '_unmatched']) < CAPTURE_N) or \
                        (iid is None and len(cap.get('_unmatched', [])) < 40):
                    cap.setdefault(who, []).append({'turn': STATE['turn'][iid] - 1 if iid is not None else None,
                                                    'request': req, 'answer': {'reasoning': reasoning, 'call': call,
                                                                               'content': content}, 'usage': usage})
        if req.get('stream'):
            delta = {'role': 'assistant', 'content': content}
            if reasoning:
                delta['reasoning_content'] = reasoning
            if call:
                delta['tool_calls'] = [{'index': 0, **call}]
            chunks = [{**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                      {**base, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                       'usage': usage}]
            self._send((''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n').encode(), 'text/event-stream')
        else:
            msg = {'role': 'assistant', 'content': content}
            if reasoning:
                msg['reasoning_content'] = reasoning
            if call:
                msg['tool_calls'] = [call]
            self._send(json.dumps({**base, 'object': 'chat.completion',
                                   'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}], 'usage': usage}).encode())


def pct(v):
    v = sorted(v)
    return {f'p{q}': v[min(len(v) - 1, int(len(v) * q / 100))] for q in (10, 50, 90)} if v else {}


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    # 1. harness wheels + uv (same install rule as public_patches_cpu)
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    import pyarrow.parquet as pq

    # 2. data dir from every mounted bank
    comp = find('competitions/gemma-4-developer-agent/tasks.jsonl') or find('gemma-4-developer-agent/tasks.jsonl')
    data = WORK / 'rbdata'
    (data / 'snapshots').mkdir(parents=True, exist_ok=True)
    for d in ('wheels', 'graphs', 'embeddings'):
        (data / d).mkdir(exist_ok=True)
    shutil.copytree(comp.parent / 'sample_submission', data / 'sample_submission', dirs_exist_ok=True)
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
            os.symlink(src, data / 'snapshots' / f'{r["instance_id"]}.tgz')
            rows.append(r)
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    problem = {r['instance_id']: norm(r.get('problem_statement')) for r in rows}
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    bdir = WORK / 'bundles' / 'sub-v5b'
    for rel_, text in BUNDLES['sub-v5b'].items():
        if rel_ != 'SOURCE.txt':
            (bdir / rel_).parent.mkdir(parents=True, exist_ok=True)
            (bdir / rel_).write_text(text)
    # bank tasks have no code graphs: the replay agent declares only the six workspace tools (the scripts never call the
    # graph tools); training text is rendered with the deployment bundle's full tool schema (TOOLS_JSON)
    agent_yaml = bdir / 'agent.yaml'
    agent_yaml.write_text(''.join(l for l in agent_yaml.read_text().splitlines(keepends=True)
                                  if l.strip() not in ('- get_code_neighbors', '- search_similar_code', '- get_code_subgraph')))

    # 3. which trajectories: execution-verified on our bank, <= MAX_TEACHER_CALLS teacher calls
    vf = sorted(glob.glob('/kaggle/input/**/public_patches/verified.jsonl', recursive=True))
    if not vf:
        raise RuntimeError('public_patches/verified.jsonl not mounted')
    wanted = {}
    for f in vf:
        for line in open(f):
            x = json.loads(line)
            if x.get('dataset') == DATASET and x.get('resolved') and (x.get('tool_calls') or 99) <= MAX_TEACHER_CALLS \
                    and x['instance_id'] in seen and x.get('traj_id'):
                wanted[str(x['traj_id'])] = x
    skipped = set()
    for f in (sorted(glob.glob(SKIP_GLOB, recursive=True)) if SKIP_GLOB else []):
        skipped |= {json.loads(line)['traj_id'] for line in open(f) if line.strip()}
    wanted = {k: v for k, v in wanted.items() if k not in skipped}
    if not wanted:
        raise RuntimeError('no verified trajectories of this dataset on the mounted bank')
    stage('data_ready', bank=len(rows), skipped_already_replayed=len(skipped), verified_short=len(wanted), tasks=len({x['instance_id'] for x in wanted.values()}),
          verified_files=vf)

    # 4. fetch those trajectories from the Hub (only the rows we need)
    tree = get(f'{HUB}/api/datasets/{DATASET}/tree/refs%2Fconvert%2Fparquet?recursive=true')
    files = sorted((t['path'], t.get('size', 0)) for t in tree if t.get('path', '').endswith('.parquet'))
    trajs, got_mb, shards_read = {}, 0, 0
    for path, size in files:
        if len(trajs) >= len(wanted) or time.time() - T0 > 0.3 * DEADLINE_S:
            break
        dest = TMP / 'shard.parquet'
        download(f'{HUB}/datasets/{DATASET}/resolve/refs%2Fconvert%2Fparquet/{urllib.parse.quote(path)}', dest)
        got_mb += size / 1e6
        shards_read += 1
        ids = set(map(str, pq.read_table(dest, columns=['trajectory_id']).column(0).to_pylist())) & set(wanted)
        if ids:
            for b in pq.read_table(dest, columns=['trajectory_id', 'instance_id', 'trajectory'],
                                   filters=[('trajectory_id', 'in', sorted(ids))]).to_batches(max_chunksize=50):
                for r in b.to_pylist():
                    trajs[str(r['trajectory_id'])] = r['trajectory']
        dest.unlink(missing_ok=True)
    stage('trajectories', found=len(trajs), wanted=len(wanted), shards_read=shards_read, of=len(files), mb=round(got_mb))

    # 5. convert
    scripts, conv_rej = [], collections.Counter()
    for tid, traj in trajs.items():
        x = wanted[tid]
        try:
            steps, root = to_script(traj)
        except Reject as e:
            conv_rej[str(e)] += 1
            continue
        except Exception as e:  # noqa: BLE001
            conv_rej[f'error:{type(e).__name__}'] += 1
            continue
        n_calls = sum(1 for s in steps if s[1] != 'submit_patch')
        if n_calls > MAX_HARNESS_CALLS:
            conv_rej['over_budget'] += 1
            continue
        scripts.append({'traj_id': tid, 'instance_id': x['instance_id'], 'teacher_calls': x.get('tool_calls'),
                        'harness_calls': n_calls, 'root': root, 'steps': steps})
    if MAX_TRAJ:
        scripts = (sorted(scripts, key=lambda s: (-s['harness_calls'], s['traj_id'])) if PREFER_LONG else scripts)[:MAX_TRAJ]
    with open(OUT / 'scripts.jsonl', 'w') as fh:
        for s in scripts:
            fh.write(json.dumps(s) + '\n')
    stage('converted', scripts=len(scripts), tasks=len({s['instance_id'] for s in scripts}), rejects=dict(conv_rej),
          harness_calls=pct([s['harness_calls'] for s in scripts]),
          tools=dict(collections.Counter(st[1] for s in scripts for st in s['steps'])))
    if not scripts:
        raise RuntimeError('nothing to replay')

    # 6. tokenizer for rendering
    import transformers
    from transformers import AutoTokenizer
    tok_dir = None
    for c in sorted(glob.glob('/kaggle/input/**/tokenizer.json', recursive=True)):
        if 'w4a16' in c or tok_dir is None:
            tok_dir = Path(c).parent
    try:
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    except Exception as e:  # noqa: BLE001
        stage('tokenizer_fallback', error=f'{type(e).__name__}: {e}'[:300])
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_dir / 'tokenizer.json'))
    jinja = tok_dir / 'chat_template.jinja'
    if jinja.exists():
        tok.chat_template = jinja.read_text()
    elif not getattr(tok, 'chat_template', None):
        tok.chat_template = json.loads((tok_dir / 'tokenizer_config.json').read_text()).get('chat_template')
    ns = {'__name__': 'build_sft'}
    exec(compile(BUILD_SFT_PY, 'build_sft.py', 'exec'), ns)
    trace_to_messages, trainable_spans = ns['trace_to_messages'], ns['trainable_spans']
    stage('ready', transformers=transformers.__version__, tokenizer=str(tok_dir))

    # 7. scripted model + replay, one trajectory per task per pass (pass j = each task's j-th shortest conversion)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    by_task = collections.defaultdict(list)
    for s in sorted(scripts, key=lambda s: (s['harness_calls'], s['traj_id'])):
        by_task[s['instance_id']].append(s)
    passes = []
    for j in range(max(len(v) for v in by_task.values())):
        batch = [v[j] for v in by_task.values() if len(v) > j]
        if j == 0 and SMOKE_N and len(batch) > SMOKE_N:
            passes += [('p0smoke', batch[:SMOKE_N]), ('p0', batch[SMOKE_N:])]
        else:
            passes.append((f'p{j}', batch))
    sft = open(OUT / 'sft_zero_v1.jsonl', 'w')
    results = open(OUT / 'results.jsonl', 'w')
    totals = collections.Counter()
    tokens, calls = [], []
    for tag, batch in passes:
        left = DEADLINE_S - (time.time() - T0)
        if left < 25 * 60:
            stage('deadline', skipped_pass=tag, trajectories=len(batch))
            break
        with LOCK:
            STATE['scripts'] = {s['instance_id']: s['steps'] for s in batch}
            STATE['snips'] = {}
            for s in batch:
                p = problem.get(s['instance_id'], '')
                STATE['snips'][s['instance_id']] = [p[o:o + 90] for o in (0, 150, 400) if len(p) >= o + 60] or [p]
            STATE['key2iid'], STATE['turn'] = {}, collections.Counter()
            STATE['unmatched'] = STATE['beyond_script'] = STATE['summaries'] = 0
        ids_file = WORK / f'ids_{tag}.json'
        ids_file.write_text(json.dumps(sorted(s['instance_id'] for s in batch)))
        out_dir = WORK / 'runs' / tag
        budget = int(min(3 * 3600, left - 15 * 60))
        t1 = time.time()
        p = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data), '--tag', tag,
                            '--out', str(out_dir), '--tasks', 'file', '--task-ids-file', str(ids_file),
                            '--bundle', str(bdir), '--base-url', f'http://127.0.0.1:{PORT}/v1', '--model-id', MODEL,
                            '--api-key', 'x', '--uv-task-env', '--concurrency', '4', '--budget-s', str(budget)],
                           capture_output=True, text=True)
        f = out_dir / 'results' / 'task_results.jsonl'
        res = {r['instance_id']: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}
        n_res = 0
        for s in batch:
            r = res.get(s['instance_id']) or {}
            row = {'traj_id': s['traj_id'], 'instance_id': s['instance_id'], 'pass': tag, 'teacher_calls': s['teacher_calls'],
                   'harness_calls': s['harness_calls'], 'finished': bool(r), 'resolved': bool(r.get('resolved')),
                   'tool_calls': r.get('tool_calls') or r.get('n_tool_calls'), 'error': str(r.get('error') or '')[:200] or None}
            tf = out_dir / 'results' / 'traces' / f'trace_{s["instance_id"]}.json'
            if row['resolved'] and tf.exists():
                try:
                    trace = json.load(open(tf))
                    msgs = trace_to_messages(trace)
                    tools = json.loads(TOOLS_JSON)
                    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=True)
                    n_tok = len(tok(text, add_special_tokens=False)['input_ids'])
                    spans = trainable_spans(text)
                    sft.write(json.dumps({'instance_id': s['instance_id'], 'traj_id': s['traj_id'],
                                          'source': f'swe-zero-replay/{DATASET}', 'n_tokens': n_tok,
                                          'n_trainable_chars': sum(e - b for b, e in spans), 'n_chars': len(text),
                                          'text': text, 'spans': spans, 'messages': msgs}) + '\n')
                    sft.flush()
                    shutil.copy(tf, OUT / 'traces' / f'{s["traj_id"]}.json')
                    tokens.append(n_tok)
                    calls.append(s['harness_calls'])
                    n_res += 1
                except Exception as e:  # noqa: BLE001
                    row['render_error'] = f'{type(e).__name__}: {e}'[:200]
            results.write(json.dumps(row) + '\n')
            results.flush()
            totals['replayed'] += 1
            totals['finished'] += row['finished']
            totals['resolved'] += row['resolved']
        if CAPTURE_N:
            cap_dir = OUT / 'captured'
            cap_dir.mkdir(exist_ok=True)
            with LOCK:
                captured = dict(STATE.get('captured', {}))
            for who, reqs in captured.items():
                (cap_dir / f'{who}.json').write_text(json.dumps(reqs, indent=1, ensure_ascii=False))
            if PARITY:
                par = []
                for who, reqs in captured.items():
                    tf = out_dir / 'results' / 'traces' / f'trace_{who}.json'
                    if who == '_unmatched' or not tf.exists():
                        continue
                    try:
                        trace = json.load(open(tf))
                        msgs = trace_to_messages(trace)
                        req_tools = next((r_['request'].get('tools') for r_ in reqs if r_['request'].get('tools')), None)
                        train_text = tok.apply_chat_template(msgs, tools=req_tools, tokenize=False, enable_thinking=True)
                        (cap_dir / f'{who}.train.txt').write_text(train_text)
                        for r_ in reqs:
                            prompt = render_request(r_['request'])
                            row = {'instance_id': who, 'turn': r_['turn'], 'n_messages': len(r_['request'].get('messages') or []),
                                   'prompt_chars': len(prompt), 'prompt_tokens': r_['usage'].get('prompt_tokens'),
                                   'roles': ''.join((m_.get('role') or '?')[0] for m_ in r_['request'].get('messages') or []),
                                   'match': train_text.startswith(prompt)}
                            if not row['match']:
                                i_ = next((k_ for k_, (x_, y_) in enumerate(zip(prompt, train_text)) if x_ != y_),
                                          min(len(prompt), len(train_text)))
                                row.update(diverge_at=i_, prompt_ctx=prompt[max(0, i_ - 300):i_ + 300],
                                           train_ctx=train_text[max(0, i_ - 300):i_ + 300])
                            par.append(row)
                    except Exception as e:  # noqa: BLE001
                        par.append({'instance_id': who, 'error': f'{type(e).__name__}: {e}'[:300]})
                with open(OUT / 'parity.jsonl', 'a') as fh_:
                    for row in par:
                        fh_.write(json.dumps(row, ensure_ascii=False) + '\n')
                same_tools = None
                try:
                    any_req = next(r_['request'] for reqs in captured.values() for r_ in reqs if r_['request'].get('tools'))
                    deploy = {t_['function']['name']: t_ for t_ in json.loads(TOOLS_JSON)}
                    live = {t_['function']['name']: t_ for t_ in any_req['tools']}
                    same_tools = {'live_only': sorted(set(live) - set(deploy)), 'deploy_only': sorted(set(deploy) - set(live)),
                                  'differ': sorted(k_ for k_ in set(live) & set(deploy) if live[k_] != deploy[k_])}
                except Exception as e:  # noqa: BLE001
                    same_tools = {'error': f'{type(e).__name__}: {e}'[:200]}
                stage('parity', tag=tag, requests=len([r for r in par if 'match' in r]),
                      matched=sum(1 for r in par if r.get('match')),
                      first_mismatches=[{k_: r[k_] for k_ in ('instance_id', 'turn', 'n_messages', 'roles', 'diverge_at')}
                                        for r in par if r.get('match') is False][:12],
                      unmatched_requests=len(captured.get('_unmatched', [])), usage_errors=STATE.get('usage_errors', 0),
                      tools_vs_deployment=same_tools)
        shutil.rmtree(out_dir / 'results' / 'test_outputs', ignore_errors=True)
        stage('pass', tag=tag, trajectories=len(batch), finished=sum(1 for s in batch if s['instance_id'] in res),
              resolved_and_rendered=n_res, minutes=round((time.time() - t1) / 60, 1), returncode=p.returncode,
              requests=STATE['requests'], unmatched=STATE['unmatched'], beyond_script=STATE['beyond_script'],
              summaries=STATE.get('summaries', 0), diffs=STATE.get('diffs', 0),
              log_tail=(p.stdout + p.stderr)[-600:] if p.returncode else '')
        if tag == 'p0smoke' and totals['resolved'] == 0:
            stage('stop', reason='no trajectory of the smoke slice resolved on replay; inspect before spending more')
            break
    sft.close()
    results.close()
    stats = {'dataset': DATASET, 'verified_short': len(wanted), 'found': len(trajs), 'converted': len(scripts),
             'conversion_rejects': dict(conv_rej), **totals,
             'resolved_rate': round(totals['resolved'] / max(1, totals['replayed']), 3),
             'tokens': pct(tokens), 'tokens_total': sum(tokens), 'harness_calls': pct(calls),
             'minutes': round((time.time() - T0) / 60, 1)}
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1))
    stage('done', **stats)
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-3000:], tb=traceback.format_exc()[-3000:])
