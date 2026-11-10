"""Convert public SWE-smith trajectories into training text in the competition harness's own format.

For every successful, short SWE-smith trajectory (Claude / GPT-4o with SWE-agent tools):
  1. check out the task's real repository state (SWE-smith keeps one GitHub mirror per repo, one branch per task);
  2. map the teacher's calls onto the competition's tools and run them through the harness's OWN tool functions
     (swegemma.tools.*) on that checkout, so file reads, edits, diffs, truncation and budget warnings are exactly what
     the scorer would return. Shell commands that only read or move files are executed for real; commands that need the
     task's Python environment (python, pytest, ...) are not executed: their output is the teacher's, wrapped by the
     harness's own formatter;
  3. keep the trajectory only if it fits the bundle's budget, edits early, ends with a non-empty patch, and (by default)
     if the replayed working tree reproduces the teacher's final patch;
  4. render it as the exact request the scorer would send (system text, task message, tool schemas captured from the
     harness) with the served model's chat template; only model-generated spans are trained.
Output: /kaggle/working/sft/{sft_v1.jsonl, sft_v1_nomatch.jsonl, meta.jsonl, stats.json, samples/}.
"""
import collections
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
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

BUILD_SFT_PY = None  # @embed ../train/build_sft.py
SYSTEM_TEXT = None  # @embed templates/sub-v5b_system.txt
USER_TEMPLATE = None  # @embed templates/sub-v5b_user.txt
TOOLS_JSON = None  # @embed templates/harness_tools_request.json
MAX_TRAJ = 0  # @define (0 = every candidate)
MAX_CALLS = 28  # @define (the bundle's max_tool_calls; submit_patch does not count)
MAX_FIRST_EDIT = 12  # @define (first source edit must come by this counted call)
MAX_TOKENS = 24000  # @define (whole conversation, Gemma tokens)
REASONING_CAP = 1200  # @define (characters of teacher reasoning kept per turn)
WORKERS = 4  # @define
DEADLINE_S = 6 * 3600  # @define
SPLIT_GLOB = 'tool-*.parquet'  # @define
R2E_REF = 'parent'  # @define (R2E-Gym checkout: 'parent' = commit^ or 'commit'; from match-scout's revision check)
SOURCE_KIND = 'swesmith'  # @define ('swesmith': SWE-smith rows on swesmith mirror branches; 'swegym': SWE-Gym SFT rows on upstream GitHub at base_commit, tasks from the match-scout mapping)
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'sft'
(OUT / 'samples').mkdir(parents=True, exist_ok=True)
REPOS = Path('/tmp/repos')
REPOS.mkdir(parents=True, exist_ok=True)
FINAL_TEXT = 'The fix is implemented and the patch is submitted.'
SAFE = {'grep', 'egrep', 'fgrep', 'find', 'ls', 'cat', 'head', 'tail', 'wc', 'sed', 'awk', 'git', 'rm', 'mv', 'cp',
        'mkdir', 'touch', 'echo', 'diff', 'sort', 'uniq', 'cut', 'tr', 'cd', 'pwd', 'file', 'stat', 'true', 'nl'}
GIT_OK = {'status', 'diff', 'log', 'show', 'checkout', 'stash', 'restore', 'grep', 'ls-files', 'branch', 'rev-parse',
          'blame', 'add', 'reset'}
LOCK = threading.Lock()
REJECT = collections.Counter()
STATS = collections.Counter()


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def find(pattern_tail, depth=7):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def sh(cmd, cwd=None, timeout=180):
    try:
        r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout,
                           errors='replace')
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'timeout'


# ---------------------------------------------------------------- harness tools on a local checkout
class Res(SimpleNamespace):
    pass


class LocalDocker:
    """The three sandbox operations the harness tools use, on a local directory that plays /workspace."""

    def __init__(self, root):
        self.root = str(root)
        self.canned = None

    def _in(self, s):
        return s.replace('/workspace', self.root)

    def _out(self, s):
        return s.replace(self.root, '/workspace')

    def exec(self, container_id, command, timeout=None, **kw):
        if self.canned is not None:
            out, code = self.canned
            self.canned = None
            return Res(stdout=out, stderr='', exit_code=code, timed_out=False)
        try:
            r = subprocess.run(['bash', '-c', self._in(command)], cwd=self.root, capture_output=True, text=True,
                               timeout=min(int(timeout or 60), 60), errors='replace',
                               env={**os.environ, 'GIT_PAGER': 'cat', 'PAGER': 'cat'})
            return Res(stdout=self._out(r.stdout), stderr=self._out(r.stderr), exit_code=r.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            return Res(stdout='', stderr='', exit_code=-1, timed_out=True)

    def copy_from(self, container_id, src, local_file):
        p = Path(self._in(str(src)))
        if p.is_file():
            shutil.copy(p, local_file)

    def copy_to(self, container_id, local_file, dest_dir):
        d = Path(self._in(str(dest_dir)))
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, d / Path(local_file).name)


class Ctx:
    """The parts of SwegemmaContext the tools read (same budget rule as the real check_budget)."""

    def __init__(self, docker, max_calls):
        self.docker, self.container_id, self.tool_lock = docker, 'local', None
        self.harness = SimpleNamespace(max_file_lines=150, max_file_chars=10000, max_stdout_chars=5000,
                                       command_timeout_seconds=240)
        self.budget = SimpleNamespace(tool_calls=max_calls, time_minutes=None, turns=None, cost_usd=None, total_tokens=None)
        self.tool_calls_used = self.llm_calls_used = 0
        self.submitted_patch, self.patch_submitted = None, False
        self.remaining_time_seconds, self.agent_elapsed_seconds, self.agent_start_time = None, 0.0, None

    def check_budget(self, *, count_tool_call=True):
        if count_tool_call and self.tool_calls_used >= self.budget.tool_calls:
            return json.dumps({'status': 'error', 'error_type': 'BudgetExceeded',
                               'error_message': f'Tool call budget exhausted ({self.budget.tool_calls} calls)'},
                              ensure_ascii=False)
        if count_tool_call:
            self.tool_calls_used += 1
        return None


def safe_shell(cmd):
    """True if every piece of the command only reads or moves files inside the workspace (run it for real)."""
    if any(x in cmd for x in ('$(', '`', '<(', 'sudo', 'curl', 'wget', '&\n')) or cmd.rstrip().endswith('&'):
        return False
    for seg in re.split(r'&&|\|\||;|\||\n', cmd):
        words = seg.strip().split()
        while words and re.fullmatch(r'[A-Za-z_]\w*=\S*', words[0]):
            words = words[1:]
        if not words:
            continue
        w = words[0]
        if w not in SAFE:
            return False
        if w == 'git' and (len(words) < 2 or words[1] not in GIT_OK):
            return False
        if w in ('rm', 'mv', 'cp', 'cd', 'mkdir', 'touch'):
            for a_ in words[1:]:
                if a_.startswith('-'):
                    continue
                if '..' in a_ or (a_.startswith('/') and not a_.startswith('/workspace')) or a_.startswith('~'):
                    return False
    return True


def exit_guess(out):
    tail = out[-1500:]
    return 1 if re.search(r'Traceback \(most recent call last\)|^\S*(Error|Exception): |command not found|'
                          r'No such file or directory|\bFAILED\b|=+ .*\b(failed|error)s?\b.* in [\d.]+s|^E +\w',
                          tail, re.M) else 0


def rel_path(p):
    p = (p or '').strip()
    if p in ('/testbed', '/testbed/'):
        return ''
    if p.startswith('/testbed/'):
        return p[len('/testbed/'):]
    return None if p.startswith('/') or p.startswith('~') else p


def obs_text(m):
    c = m.get('content')
    if isinstance(c, list):
        c = '\n'.join(str(x.get('text', '')) if isinstance(x, dict) else str(x) for x in c)
    c = (c or '').replace('\r\n', '\n')
    return c[len('OBSERVATION:\n'):] if c.startswith('OBSERVATION:\n') else c


CALL_RE = re.compile(r'<function=([\w.-]+)>(.*?)</function>', re.S)
PARAM_RE = re.compile(r'<parameter=([\w.-]+)>(.*?)</parameter>', re.S)


def text_calls(msgs):
    """SWE-agent XML-format turns (`<function=..><parameter=..>` in the text, results as `OBSERVATION:` user turns) ->
    the tool-call form the rest of the converter reads. Parameter values are stripped, as SWE-agent's XML parser does.
    Trajectories that already use tool calls are returned unchanged."""
    if any(isinstance(m, dict) and m.get('tool_calls') for m in msgs):
        return msgs
    out, k = [], 0
    for m in msgs:
        c = m.get('content')
        c = c if isinstance(c, str) else '\n'.join(str(x.get('text', '')) for x in (c or []) if isinstance(x, dict))
        if m.get('role') == 'assistant' and CALL_RE.search(c or ''):
            calls = []
            for fm in CALL_RE.finditer(c):
                k += 1
                params = {a: v.strip() for a, v in PARAM_RE.findall(fm.group(2))}
                if isinstance(params.get('view_range'), str):  # '[235, 253]' -> [235, 253]
                    try:
                        params['view_range'] = json.loads(params['view_range'])
                    except ValueError:
                        params.pop('view_range')
                calls.append({'id': f'call_{k}', 'type': 'function', 'function': {'name': fm.group(1), 'arguments': json.dumps(params)}})
            out.append({'role': 'assistant', 'content': c[:CALL_RE.search(c).start()].rstrip(), 'tool_calls': calls})
        elif m.get('role') == 'user' and out and (c or '').startswith('OBSERVATION:'):
            out.append({'role': 'tool', 'content': c})
        elif m.get('role') == 'user' and out and ((c or '').startswith('EXECUTION RESULT of [') or (c or '').startswith('Execution output of [')):
            out.append({'role': 'tool', 'content': c.split('\n', 1)[1] if '\n' in c else ''})
        else:
            out.append(m)
    return out


def normalize_swegym(msgs):
    """OpenHands text-format SWE-Gym trajectory -> the SWE-smith conventions the converter reads: the uploaded repo root
    becomes /testbed, execute_bash -> bash, finish -> submit, view_range 'None' dropped."""
    user = next((m for m in msgs if m.get('role') == 'user'), {})
    mr = re.search(r'<uploaded_files>\s*(/workspace/[^\s<]+)', user.get('content') or '')
    root = mr.group(1).rstrip('/') if mr else None
    if root:
        msgs = [{**m, 'content': (m.get('content') or '').replace(root, '/testbed')} for m in msgs]
    out = text_calls(msgs)
    for m in out:
        for tc in m.get('tool_calls') or []:
            fn = tc['function']
            fn['name'] = {'execute_bash': 'bash', 'finish': 'submit'}.get(fn['name'], fn['name'])
            args = json.loads(fn['arguments'])
            if args.get('view_range') in ('None', None, ''):
                args.pop('view_range', None)
            fn['arguments'] = json.dumps(args)
    return out


R2E_REPOS = {'pandas': 'pandas-dev/pandas', 'numpy': 'numpy/numpy', 'pillow': 'python-pillow/Pillow',
             'aiohttp': 'aio-libs/aiohttp', 'tornado': 'tornadoweb/tornado', 'coveragepy': 'nedbat/coveragepy',
             'datalad': 'datalad/datalad', 'pyramid': 'Pylons/pyramid', 'orange3': 'biolab/orange3', 'scrapy': 'scrapy/scrapy',
             'sympy': 'sympy/sympy', 'matplotlib': 'matplotlib/matplotlib', 'moto': 'getmoto/moto'}


def normalize_r2egym(msgs):
    """R2E-Gym text-format trajectory (/testbed paths) -> SWE-smith conventions: file_editor -> str_replace_editor,
    execute_bash(cmd) -> bash(command), search -> a real grep with per-file match counts, finish -> submit."""
    import shlex
    out = text_calls(msgs)
    for m in out:
        for tc in m.get('tool_calls') or []:
            fn = tc['function']
            args = json.loads(fn['arguments'])
            if fn['name'] == 'file_editor':
                fn['name'] = 'str_replace_editor'
                args.pop('concise', None)
            elif fn['name'] == 'execute_bash':
                fn['name'], args = 'bash', {'command': args.get('cmd') or args.get('command') or ''}
            elif fn['name'] == 'search':
                term, where = args.get('search_term') or '', args.get('path') or '/testbed'
                fn['name'] = 'bash'
                args = {'command': f"grep -rnc --include='*.py' -- {shlex.quote(term)} {shlex.quote(where)} | grep -v ':0$' | head -50"}
            elif fn['name'] == 'finish':
                fn['name'], args = 'submit', {}
            if args.get('view_range') in ('None', None, ''):
                args.pop('view_range', None)
            fn['arguments'] = json.dumps(args)
    return out


class Reject(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.detail = detail


def replay(row, root, tools):
    """Teacher trajectory -> list of (reasoning, tool name, arguments, observation JSON) using the real harness tools."""
    read_file, write_file, edit_file, run_command, submit_patch = tools
    msgs = text_calls(json.loads(row['messages']))
    docker = LocalDocker(root)
    ctx = Ctx(docker, MAX_CALLS)
    steps, first_edit, cmd_count, n_real, n_canned = [], None, collections.Counter(), 0, 0
    calls = [(i, m) for i, m in enumerate(msgs) if m.get('role') == 'assistant' and m.get('tool_calls')]
    if not calls:
        raise Reject('no_calls')
    last_submit = max((i for i, m in calls if m['tool_calls'][0]['function']['name'] == 'submit'), default=None)
    if last_submit is None:
        raise Reject('no_submit')

    def emit(reasoning, name, args, raw):
        docker.canned = None
        parsed = json.loads(raw)
        if parsed.get('error_type') == 'BudgetExceeded':
            raise Reject('over_budget')
        steps.append((reasoning, name, args, json.dumps(parsed, ensure_ascii=False)))
        return parsed

    for i, m in calls:
        if len(m['tool_calls']) != 1:
            raise Reject('multi_call')
        fn = m['tool_calls'][0]['function']
        name = fn['name']
        try:
            args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else (fn['arguments'] or {})
        except Exception:  # noqa: BLE001
            raise Reject('bad_arguments') from None
        think = (m.get('thought') or '')
        if not think:
            c = m.get('content')
            think = c if isinstance(c, str) else ' '.join(str(x.get('text', '')) for x in (c or []) if isinstance(x, dict))
        think = think.strip().replace('/testbed', '/workspace')
        if len(think) > REASONING_CAP:
            cut = think[:REASONING_CAP]
            think = cut[:max(cut.rfind('. '), cut.rfind('\n'), REASONING_CAP // 2) + 1].rstrip()
        teacher = obs_text(msgs[i + 1]) if i + 1 < len(msgs) and msgs[i + 1].get('role') == 'tool' else ''
        if name == 'submit':
            if i != last_submit:
                continue  # SWE-agent's first submit only prints a review reminder
            r = emit(think, 'submit_patch', {}, submit_patch(ctx))
            if r.get('status') != 'ok' or not r.get('patch_size'):
                raise Reject('empty_patch')
            break
        if name == 'bash':
            cmd = re.sub(r'^\s*cd /workspace/?\s*(&&|;)\s*', '', str(args.get('command', '')).replace('/testbed', '/workspace')).strip()
            if not cmd:
                raise Reject('empty_command')
            cmd_count[cmd] += 1
            if cmd_count[cmd] > 2:
                raise Reject('repeated_command')
            if safe_shell(cmd):
                n_real += 1
                r = emit(think, 'run_command', {'command': cmd}, run_command(ctx, cmd))
                if re.search(r'\bsed\s+(-\w+\s+)*-i\b|>>?\s*[\w./-]+', cmd) and r.get('status') == 'ok' \
                        and not re.search(r'>\s*/dev/null', cmd):
                    cmd_count.clear()
                    if first_edit is None:
                        first_edit = ctx.tool_calls_used
            else:
                n_canned += 1
                out = teacher.replace('/testbed', '/workspace')
                docker.canned = (out, exit_guess(out))
                emit(think, 'run_command', {'command': cmd}, run_command(ctx, cmd))
        elif name == 'str_replace_editor':
            sub, rel = args.get('command'), rel_path(args.get('path'))
            if rel is None:
                raise Reject('path_outside_workspace')
            target = Path(root) / rel
            if sub == 'view':
                if target.is_dir():
                    if rel == '' and not steps:
                        continue  # the task message already shows the tree
                    cmd = f'find {rel or "."} -maxdepth 2 -not -path "*/.*" | sort | head -100'
                    emit(think, 'run_command', {'command': cmd}, run_command(ctx, cmd))
                    continue
                total = len(target.read_text(errors='replace').splitlines()) if target.is_file() else 0
                vr = args.get('view_range')
                if vr and isinstance(vr, list) and len(vr) == 2:
                    a_, b_ = int(vr[0]), int(vr[1])
                    b_ = total if b_ == -1 else b_
                elif 'too large to display entirely' in teacher or total > 450:
                    cmd = f'grep -n "^\\s*\\(class\\|def\\|async def\\) " {rel} | head -150'
                    emit(think, 'run_command', {'command': cmd}, run_command(ctx, cmd))
                    continue
                else:
                    a_, b_ = 1, total
                if b_ - a_ + 1 > 450:
                    raise Reject('long_view')
                teacher_saw = 'result of running `cat -n`' in teacher or teacher.startswith('Invalid `view_range`') \
                    or 'Here is a condensed view for file' in teacher
                if not vr and total <= 150:
                    r = emit(think, 'read_file', {'filepath': rel}, read_file(ctx, rel))
                    if (r.get('status') == 'ok') != teacher_saw:
                        raise Reject('view_desync', {'rel': rel, 'ours': r.get('status'), 'err': str(r.get('error_message', ''))[:300],
                                                     'teacher': teacher[:400], 'total': total})
                    continue
                k = a_
                while True:
                    e_ = min(b_, k + 149)
                    r = emit(think if k == a_ else '', 'read_file', {'filepath': rel, 'start_line': k, 'end_line': e_},
                             read_file(ctx, rel, k, e_))
                    if k == a_ and (r.get('status') == 'ok') != teacher_saw:
                        raise Reject('view_desync', {'rel': rel, 'range': [a_, b_], 'ours': r.get('status'),
                                                     'err': str(r.get('error_message', ''))[:300], 'teacher': teacher[:400], 'total': total})
                    k = e_ + 1
                    if k > b_:
                        break
            elif sub == 'str_replace':
                old, new = (args.get('old_str', '') or '').replace('/testbed', '/workspace'), (args.get('new_str', '') or '').replace('/testbed', '/workspace')
                r = emit(think, 'edit_file', {'filepath': rel, 'old_string': old, 'new_string': new},
                         edit_file(ctx, rel, old, new))
                if (r.get('status') == 'ok') != ('has been edited' in teacher):
                    raise Reject('edit_desync', {'rel': rel, 'ours': r.get('status'), 'err': str(r.get('error_message', ''))[:300],
                                                 'teacher': teacher[:400]})
                if r.get('status') == 'ok':
                    cmd_count.clear()
                if r.get('status') == 'ok' and first_edit is None and not re.search(r'(^|/)(tests?/|test_|reproduce|repro)', rel):
                    first_edit = ctx.tool_calls_used
            elif sub == 'create':
                text_ = (args.get('file_text', '') or '').replace('/testbed', '/workspace')
                r = emit(think, 'write_file', {'filepath': rel, 'content': text_}, write_file(ctx, rel, text_))
                if (r.get('status') == 'ok') != ('created successfully' in teacher or 'File created at' in teacher):
                    raise Reject('create_desync', {'rel': rel, 'ours': r.get('status'), 'err': str(r.get('error_message', ''))[:300],
                                                   'teacher': teacher[:400]})
                if r.get('status') == 'ok':
                    cmd_count.clear()
            else:
                raise Reject(f'editor_{sub}')
        else:
            raise Reject('unknown_tool')
    if not steps or steps[-1][1] != 'submit_patch':
        raise Reject('no_final_submit')
    if first_edit is None:
        raise Reject('no_source_edit')
    if first_edit > MAX_FIRST_EDIT:
        raise Reject('late_edit')
    return steps, {'calls': ctx.tool_calls_used, 'first_edit': first_edit, 'real_shell': n_real, 'canned_shell': n_canned,
                   'patch': ctx.submitted_patch or ''}


def diff_body(d):
    """Changed lines of the source files in a diff (scratch files and headers ignored)."""
    out, keep = [], True
    for ln in (d or '').splitlines():
        if ln.startswith('diff --git'):
            keep = not re.search(r'(^| b/)(reproduce|repro|test_\w*|debug\w*|check\w*|verify\w*)[^/]*\.py$', ln.split(' b/')[-1])
            continue
        if keep and ln[:1] in '+-' and not ln.startswith(('+++', '---')):
            out.append(ln.rstrip())
    return sorted(out)


# ---------------------------------------------------------------- main
try:
    # harness code (the same wheels the scorer uses; only those this interpreter supports)
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    from swegemma.tools.execution import run_command, submit_patch
    from swegemma.tools.workspace import edit_file, read_file, write_file
    TOOLS = (read_file, write_file, edit_file, run_command, submit_patch)
    import transformers
    from transformers import AutoTokenizer
    tok_dir = None
    for c in sorted(glob.glob('/kaggle/input/**/tokenizer.json', recursive=True)):
        if 'w4a16' in c or tok_dir is None:
            tok_dir = Path(c).parent
    try:
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    except Exception as e:  # noqa: BLE001
        log(f'AutoTokenizer failed ({type(e).__name__}: {e}); using the tokenizer file directly')
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_dir / 'tokenizer.json'))
    jinja = tok_dir / 'chat_template.jinja'
    if jinja.exists():
        tok.chat_template = jinja.read_text()
    elif not getattr(tok, 'chat_template', None):
        cfg = json.loads((tok_dir / 'tokenizer_config.json').read_text())
        tok.chat_template = cfg.get('chat_template')
    if not tok.chat_template:
        raise RuntimeError(f'no chat template found in {tok_dir}: {sorted(p.name for p in tok_dir.iterdir())}')
    tools_schema = json.loads(TOOLS_JSON)
    ns = {'__name__': 'build_sft'}
    exec(compile(BUILD_SFT_PY, 'build_sft.py', 'exec'), ns)
    trainable_spans = ns['trainable_spans']
    log(f'ready: transformers {transformers.__version__}, tokenizer {tok_dir}, chat template {"file" if jinja.exists() else "config"}')

    # candidates. The dataset's `patch` and `resolved` columns are not always in the row of the trajectory they describe
    # (2026-10-05 smoke: a dask trajectory's `patch` edits webargs). Re-align them: a row's label comes from the row whose
    # patch's added lines appear in this trajectory's own editor calls.
    import hashlib
    import pyarrow.parquet as pq
    if SOURCE_KIND in ('swegym', 'r2egym'):
        mp = sorted(glob.glob('/kaggle/input/**/match_scout/mapping.jsonl', recursive=True))
        if not mp:
            raise RuntimeError('match_scout/mapping.jsonl not mounted')
        src, normalize = {'swegym': ('swe-gym-openhands-sft-trajectories', normalize_swegym),
                          'r2egym': ('r2egym-sft-trajectories', normalize_r2egym)}[SOURCE_KIND]
        task_of = {}
        for f_ in mp:
            for line in open(f_):
                x = json.loads(line)
                if x['source'] == src and x['matched'] and len(x['commits']) == 1 and x.get('repo'):
                    task_of[x['row']] = x  # same file order as match_scout_cpu.py
        files = sorted(glob.glob(f'/kaggle/input/**/{src}/**/*.parquet', recursive=True)) or \
            sorted(glob.glob(f'/kaggle/input/{src}/*.parquet'))
        log(f'{len(files)} parquet files, {len(task_of)} rows with one task commit')
        pre, cands, row_no = collections.Counter(), [], 0
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=['messages']):
                for r in b.to_pylist():
                    row_no += 1
                    pre['rows'] += 1
                    x = task_of.get(row_no - 1)
                    if x is None:
                        pre['no_task'] += 1
                        continue
                    msgs = normalize(r['messages'] if isinstance(r['messages'], list) else json.loads(r['messages']))
                    n_calls = sum(1 for m in msgs if m.get('role') == 'assistant' and m.get('tool_calls'))
                    subs = collections.Counter(json.loads(tc['function']['arguments']).get('command') for m in msgs
                                               for tc in (m.get('tool_calls') or []) if tc['function']['name'] == 'str_replace_editor')
                    if n_calls > MAX_CALLS + 6:
                        pre['too_many_calls'] += 1
                    elif subs['insert'] or subs['undo_edit']:
                        pre['insert_or_undo'] += 1
                    elif not subs['str_replace']:
                        pre['no_edit'] += 1
                    else:  # the SFT rows are successful runs; no reference patch in the row
                        slug = x['repo'] if '/' in x['repo'] else R2E_REPOS.get(x['repo'].lower())
                        if not slug:
                            pre['unknown_repo'] += 1
                            continue
                        commit = x['commits'][0]
                        iid = x['task_keys'][0] if SOURCE_KIND == 'swegym' else f"{x['repo']}__{commit[:12]}"
                        if SOURCE_KIND == 'r2egym' and R2E_REF == 'parent':
                            commit += '^'
                        cands.append({'instance_id': iid, 'traj_id': f'{SOURCE_KIND}-{row_no - 1}', 'model': f'{SOURCE_KIND}-sft',
                                      'repo_key': slug, 'commit': commit, 'messages': json.dumps(msgs),
                                      'label_resolved': True, 'label_patch': '', 'patch_col': '', 'align': SOURCE_KIND})
        ALIGN = {'mapping': mp[0], 'rows_with_one_commit': len(task_of)}
    else:
        files = sorted(glob.glob(f'/kaggle/input/**/{SPLIT_GLOB}', recursive=True))
        log(f'{len(files)} parquet files')
        scratch_rx = re.compile(r'(^| b/)(reproduce|repro|test_\w*|debug\w*|check\w*|verify\w*)[^/]*\.py$')

        def sig_lines(text):
            return {hashlib.md5(x.strip().encode()).hexdigest()[:12] for x in (text or '').splitlines() if len(x.strip()) >= 8}

        def patch_sig(d):
            added, keep = [], True
            for ln in (d or '').splitlines():
                if ln.startswith('diff --git'):
                    keep = not scratch_rx.search(ln.split(' b/')[-1])
                elif keep and ln.startswith('+') and not ln.startswith('+++'):
                    added.append(ln[1:])
            return sig_lines('\n'.join(added))

        rows, pre = [], collections.Counter()
        for f in files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=256, columns=['messages', 'instance_id', 'resolved', 'model', 'traj_id', 'patch']):
                for r in b.to_pylist():
                    pre['rows'] += 1
                    try:
                        msgs = text_calls(json.loads(r['messages']))
                    except Exception:  # noqa: BLE001
                        pre['bad_json'] += 1
                        msgs = []
                    esig, subs, n_calls = set(), collections.Counter(), 0
                    for m in msgs:
                        if m.get('role') != 'assistant' or not m.get('tool_calls'):
                            continue
                        n_calls += 1
                        for tc in m['tool_calls']:
                            fn = tc.get('function') or {}
                            try:
                                args = json.loads(fn['arguments']) if isinstance(fn.get('arguments'), str) else (fn.get('arguments') or {})
                            except Exception:  # noqa: BLE001
                                continue
                            if fn.get('name') == 'str_replace_editor' and isinstance(args, dict):
                                subs[args.get('command')] += 1
                                if args.get('command') in ('str_replace', 'create', 'insert'):
                                    esig |= sig_lines(args.get('new_str') or args.get('file_text') or '')
                    why = None
                    if n_calls > MAX_CALLS + 6:
                        why = 'too_many_calls'
                    elif subs['insert'] or subs['undo_edit']:
                        why = 'insert_or_undo'
                    elif not subs['str_replace']:
                        why = 'no_edit'
                    elif 'There are bug(s) in this codebase, likely located in the following file' in r['messages'][:6000]:
                        why = 'templated_statement'
                    rows.append({'instance_id': r['instance_id'], 'traj_id': r['traj_id'], 'model': r['model'],
                                 'resolved_col': bool(r['resolved']), 'patch_col': r['patch'] or '', 'psig': patch_sig(r['patch']),
                                 'esig': esig, 'why': why, 'messages': r['messages'] if why is None else None})
        # inverted index of patch lines; lines that occur in many patches carry no identity
        df = collections.Counter(h for r in rows for h in r['psig'])
        index = collections.defaultdict(list)
        for k, r in enumerate(rows):
            r['psig'] = {h for h in r['psig'] if df[h] <= 50}
            for h in r['psig']:
                index[h].append(k)
        align, offsets = collections.Counter(), collections.Counter()
        for i, r in enumerate(rows):
            votes = collections.Counter(k for h in r['esig'] for k in index.get(h, ()))
            scored = sorted(((v / len(rows[k]['psig']), v, k) for k, v in votes.items() if v >= 2), reverse=True)
            r['label_resolved'], r['label_patch'], r['align'] = None, '', 'none'
            if scored and scored[0][0] >= 0.6:
                top = scored[0][0]
                ties = [k for sc, _, k in scored if sc >= top - 1e-9]
                k = i if i in ties else ties[0]
                labels = {rows[t]['resolved_col'] for t in ties}
                r['align'] = 'self' if k == i else ('same_instance' if rows[k]['instance_id'] == r['instance_id'] else 'other_instance')
                if len(labels) == 1:
                    r['label_resolved'], r['label_patch'] = rows[k]['resolved_col'], rows[k]['patch_col']
                else:
                    r['align'] += '_ambiguous'
                if k != i:
                    offsets[k - i] += 1
            align[r['align']] += 1
        agree = collections.Counter((r['resolved_col'], r['label_resolved']) for r in rows)
        ALIGN = {'align': dict(align), 'offsets_top': [[o, n] for o, n in offsets.most_common(10)],
                 'resolved_col_vs_label': {f'{a}->{b}': n for (a, b), n in agree.items()}}
        log(f'alignment: {json.dumps(ALIGN)}')
        cands = []
        for r in rows:
            if r['why']:
                pre[r['why']] += 1
            elif r['label_resolved'] is None:
                pre['no_label'] += 1
            elif not r['label_resolved']:
                pre['unresolved'] += 1
            else:
                cands.append(r)
        del rows, index
    log(f'prefilter: {dict(pre)} -> {len(cands)} candidates')
    random.Random(11).shuffle(cands)
    if MAX_TRAJ:
        cands = cands[:MAX_TRAJ]
    by_repo = collections.defaultdict(list)
    for r in cands:
        by_repo[r.get('repo_key') or '.'.join(r['instance_id'].split('.')[:2])].append(r)
    log(f'{len(cands)} trajectories over {len(by_repo)} repos')

    out_ok = open(OUT / 'sft_v1.jsonl', 'w')
    out_nm = open(OUT / 'sft_v1_nomatch.jsonl', 'w')
    out_meta = open(OUT / 'meta.jsonl', 'w')
    dbg = open(OUT / 'rejects_debug.jsonl', 'w')
    (OUT / 'tools.json').write_text(json.dumps(tools_schema, indent=1))
    samples_left = [6]

    def do_repo(mirror, rows):
        path = REPOS / mirror.replace('/', '__')
        if SOURCE_KIND in ('swegym', 'r2egym'):  # upstream repo, blobless: every commit, file contents fetched on checkout
            rc, _, err = sh(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', f'https://github.com/{mirror}.git', str(path)],
                            timeout=1800)
        else:
            rc, _, err = sh(['git', 'clone', '-q', '--depth', '1', f'https://github.com/swesmith/{mirror}.git', str(path)], timeout=900)
        if rc != 0:
            with LOCK:
                REJECT['clone_failed'] += len(rows)
            log(f'clone failed {mirror}: {err[-160:]}')
            return
        repo = mirror if SOURCE_KIND in ('swegym', 'r2egym') else mirror.split('.')[0].replace('__', '/', 1)
        for row in rows:
            if time.time() - T0 > DEADLINE_S:
                with LOCK:
                    REJECT['deadline'] += 1
                continue
            iid = row['instance_id']
            try:
                if SOURCE_KIND in ('swegym', 'r2egym'):
                    rc, _, err = sh(['git', '-C', str(path), 'checkout', '-q', '-f', row['commit']], timeout=900)
                    if rc != 0:
                        raise Reject('checkout_failed', {'commit': row['commit'], 'err': err[-200:]})
                    sh(['git', '-C', str(path), 'clean', '-fdxq'])
                else:
                    rc, _, err = sh(['git', '-C', str(path), 'fetch', '-q', '--depth', '2', 'origin', iid], timeout=300)
                    if rc != 0:
                        raise Reject('fetch_failed')
                    sh(['git', '-C', str(path), 'checkout', '-q', '-f', 'FETCH_HEAD'])
                    sh(['git', '-C', str(path), 'clean', '-fdxq'])
                    viewed, created = [], set()
                    for m_ in text_calls(json.loads(row['messages'])):
                        for tc in (m_.get('tool_calls') or []) if m_.get('role') == 'assistant' else []:
                            fn_ = tc.get('function') or {}
                            try:
                                a_ = json.loads(fn_['arguments']) if isinstance(fn_.get('arguments'), str) else (fn_.get('arguments') or {})
                            except Exception:  # noqa: BLE001
                                continue
                            if fn_.get('name') == 'str_replace_editor' and isinstance(a_, dict):
                                rp = rel_path(a_.get('path'))
                                if rp and a_.get('command') == 'create':
                                    created.add(rp)
                                elif rp and a_.get('command') == 'view' and rp not in created and rp not in viewed:
                                    viewed.append(rp)
                    restored = [rp for rp in viewed if not (path / rp).exists()
                                and sh(['git', '-C', str(path), 'cat-file', '-e', f'FETCH_HEAD~1:{rp}'])[0] == 0]
                    for rp in restored:
                        sh(['git', '-C', str(path), 'checkout', 'FETCH_HEAD~1', '--', rp])
                    if restored:
                        sh(['git', '-C', str(path), '-c', 'user.name=x', '-c', 'user.email=x@x', 'commit', '-q', '-m', 'restore viewed files'])
                        with LOCK:
                            STATS['restored_files'] += len(restored)
                            STATS['restored_trajs'] += 1
                _, tree, _ = sh('find . -maxdepth 3 -not -path "./.git/*" -not -name "*.pyc" -not -name "__pycache__" | sort | head -150',
                                cwd=str(path))
                with LOCK:
                    if not STATS['tree_logged']:
                        STATS['tree_logged'] = 1
                        log(f'tree sample ({mirror}): {tree.splitlines()[:8]}')
                msgs0 = json.loads(row['messages'])
                first_user = next(m for m in msgs0 if m.get('role') == 'user')
                ut = first_user['content'] if isinstance(first_user['content'], str) else \
                    '\n'.join(x.get('text', '') for x in first_user['content'] if isinstance(x, dict))
                mm = re.search(r'<(pr_description|github_issue)>\n?(.*?)\n?</\1>', ut, re.S)
                if not mm or len(mm.group(2).strip()) < 40:
                    raise Reject('no_problem_statement')
                steps, info = replay(row, path, TOOLS)
                touched = re.findall(r'^diff --git a/(\S+)', info['patch'], re.M)
                if any(re.search(r'(^|/)tests?/|(^|/)conftest\.py$', f) for f in touched):
                    raise Reject('edits_tests', {'files': touched})
                match = diff_body(info['patch']) == diff_body(row['label_patch']) if row['label_patch'] else None
                conv = [{'role': 'system', 'content': SYSTEM_TEXT},
                        {'role': 'user', 'content': USER_TEMPLATE.replace('{REPO}', repo).replace('{PROBLEM}', mm.group(2).strip())
                         .replace('{TREE}', tree.strip())}]
                for k, (think, name, args, obs) in enumerate(steps):
                    msg = {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': f'call_{k}', 'type': 'function', 'function': {'name': name, 'arguments': args}}]}
                    if think:
                        msg['reasoning'] = think
                    conv.append(msg)
                    conv.append({'role': 'tool', 'tool_call_id': f'call_{k}', 'name': name, 'content': obs})
                conv.append({'role': 'assistant', 'content': FINAL_TEXT})
                text = tok.apply_chat_template(conv, tools=tools_schema, tokenize=False, enable_thinking=True)
                n_tokens = len(tok(text, add_special_tokens=False)['input_ids'])
                if n_tokens > MAX_TOKENS:
                    raise Reject('too_many_tokens')
                spans = trainable_spans(text)
                ex = {'instance_id': iid, 'source': f'swe-smith/{row.get("model")}', 'n_tokens': n_tokens,
                      'n_trainable_chars': sum(e - s for s, e in spans), 'n_chars': len(text), 'text': text, 'spans': spans,
                      'messages': conv, 'align': row['align'], 'patch_match': match}
                meta = {'instance_id': iid, 'traj_id': row.get('traj_id'), 'model': row.get('model'), 'n_tokens': n_tokens,
                        'calls': info['calls'], 'first_edit': info['first_edit'], 'real_shell': info['real_shell'],
                        'canned_shell': info['canned_shell'], 'patch_match': match, 'align': row['align'], 'steps': len(steps),
                        'tools': dict(collections.Counter(s[1] for s in steps))}
                with LOCK:
                    (out_ok if match or match is None else out_nm).write(json.dumps(ex) + '\n')
                    out_meta.write(json.dumps(meta) + '\n')
                    STATS['kept_match' if match or match is None else 'kept_nomatch'] += 1
                    STATS['tokens'] += n_tokens
                    if samples_left[0] > 0:
                        samples_left[0] -= 1
                        (OUT / 'samples' / f'{iid}.txt').write_text(text)
                        (OUT / 'samples' / f'{iid}.meta.json').write_text(json.dumps(
                            {**meta, 'our_patch': info['patch'][:6000], 'teacher_patch': row['label_patch'][:6000], 'patch_col': row['patch_col'][:3000]}, indent=1))
                    n = STATS['kept_match'] + STATS['kept_nomatch']
                    if n % 100 == 0:
                        out_ok.flush(), out_nm.flush(), out_meta.flush()
                        log(f'kept {n} (match {STATS["kept_match"]}); rejects {dict(REJECT.most_common(8))}')
            except Reject as e:
                with LOCK:
                    REJECT[str(e)] += 1
                    if e.detail is not None and REJECT[str(e)] <= 25:
                        dbg.write(json.dumps({'instance_id': iid, 'traj_id': row.get('traj_id'), 'reason': str(e), **e.detail}) + '\n')
                        dbg.flush()
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    REJECT[f'error:{type(e).__name__}'] += 1
                    if REJECT[f'error:{type(e).__name__}'] <= 3:
                        log(f'{iid}: {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}')
        shutil.rmtree(path, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda kv: do_repo(*kv), sorted(by_repo.items(), key=lambda kv: -len(kv[1]))))
    for fh in (out_ok, out_nm, out_meta, dbg):
        fh.close()
    metas = [json.loads(x) for x in open(OUT / 'meta.jsonl')]
    good = [m for m in metas if m['patch_match']]

    def pct(v):
        v = sorted(v)
        return {f'p{q}': v[min(len(v) - 1, int(len(v) * q / 100))] for q in (10, 50, 90)} if v else {}
    stats = {'candidates': len(cands), 'repos': len(by_repo), 'kept_patch_match': len(good), 'kept_no_match': len(metas) - len(good),
             'rejects': dict(REJECT.most_common()), 'prefilter': dict(pre),
             'restored': {'files': STATS['restored_files'], 'trajectories': STATS['restored_trajs']}, 'alignment': ALIGN, 'kept_by_align': dict(collections.Counter(m['align'] for m in metas)),
             'tokens_total_match': sum(m['n_tokens'] for m in good), 'tokens_total_all': sum(m['n_tokens'] for m in metas),
             'tokens': pct([m['n_tokens'] for m in metas]), 'calls': pct([m['calls'] for m in metas]),
             'first_edit': pct([m['first_edit'] for m in metas]),
             'canned_shell_share': round(sum(m['canned_shell'] for m in metas) / max(1, sum(m['canned_shell'] + m['real_shell'] for m in metas)), 3),
             'tools': dict(sum((collections.Counter(m['tools']) for m in metas), collections.Counter())),
             'models': dict(collections.Counter(m['model'] for m in metas)), 'minutes': round((time.time() - T0) / 60, 1),
             'settings': {'MAX_CALLS': MAX_CALLS, 'MAX_FIRST_EDIT': MAX_FIRST_EDIT, 'MAX_TOKENS': MAX_TOKENS,
                          'REASONING_CAP': REASONING_CAP, 'bundle': 'sub-v5b'}}
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1))
    log('STATS ' + json.dumps(stats))
except Exception as e:  # noqa: BLE001
    log(f'FATAL {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}')
    (OUT / 'stats.json').write_text(json.dumps({'fatal': f'{type(e).__name__}: {e}', 'rejects': dict(REJECT)}, indent=1))
