"""Assemble the LoRA training mix from the converted teacher datasets (CPU, no internet).

Inputs (mounted): the rendered conversations of each source (SOURCES: name -> glob of a jsonl with `text`, `spans`,
`messages`, `n_tokens`, `instance_id`), plus the served model's tokenizer.
Steps: drop rows from the competition's repositories and their close relatives (EXCLUDE_REPOS), rows whose trained text
still carries the teacher's markup (call/observation tags, teacher tool names), edits of real test files, more than
MAX_CALLS tool calls or MAX_TOKENS tokens, exact duplicates. Fix and re-render (only when re-rendering untouched rows of
that source reproduces their text exactly): scratch scripts the teacher created at the repository root under test names
(`test_x.py`, `x_test.py`; our prompt forbids creating test files, and the harness discards them) are renamed
(`repro_x.py`, `x_repro.py`) everywhere after the task message; R2E `[ISSUE]` markers are stripped. At most PER_INSTANCE
conversations per task and REPO_CAP per repository; hold out whole repositories (~VAL_SHARE of the rows) for validation
loss. Train rows are ordered so that every prefix has about the SHARES mix of sources: `train_lora.py --max-examples N`
then trains a balanced first N. Token and trainable-token counts use the training script's own labelling.
COMPACT: the scorer compacts above 14,336 prompt tokens (ADK 1.36.1, `kernels/train/compaction_sim.py`); each
conversation becomes what the scorer shows: segments (system + task, or system + `For context` summary + the retained
calls; earlier-trained retained turns are not trained again) and the summary requests themselves (task message and the
agent's text in, our deterministic summary out). NORMALIZE_SYSTEM: every system message = the deployment prompt (with
ADK's identity suffix). PARITY_GLOB: before anything else, the simulator and the token counts are checked against the
captured requests of `g4dev-format-parity-cpu` (exact token counts, trigger step, summary prompt, rendered post-compaction
request); any mismatch stops the job.
Outputs: /kaggle/working/mix/{train.jsonl, val.jsonl, stats.json}; with COMPACT one line per segment / summary request.
"""
import collections
import glob
import hashlib
import json
import math
import random
import re
import time
import traceback
import types
from pathlib import Path

BUILD_SFT_PY = None  # @embed ../train/build_sft.py
COMPACT_SIM_PY = None  # @embed ../train/compaction_sim.py
SYSTEM_TEXT = None  # @embed templates/sub-v5b_system.txt
COMPACT = False  # @define
NORMALIZE_SYSTEM = False  # @define
PARITY_GLOB = ''  # @define (e.g. '/kaggle/input/**/zero_replay/captured/*.json')
REFINE = 64  # token counts this close to the threshold are recomputed from the exact request render
THOUGHT_TURNS_ONLY = False  # @define (train only the model turns that open with a thought, as render_traces_cpu does for on-policy rows)
SUMMARY_KEEP = 1.0  # @define (with COMPACT: keep the summary-request rows of this share of compacted conversations; segments are always kept)
SUMMARY_SOURCE = 'template'  # @define ('gemma': compaction summaries written by Gemma 4 31B, looked up by prompt hash in
# GEMMA_SUMMARIES_GLOB; a missing one goes to mix/pending.jsonl and its conversation-chain waits for the next round, B39)
GEMMA_SUMMARIES_GLOB = '/kaggle/input/**/g4dev-mix8-summaries/*.jsonl'  # @define (rows {chain, hash, summary})
CHAINS = 1  # @define (with 'gemma': independent summary chains per compacted conversation; chain k uses sample k at every
# compaction. train/val.jsonl = chain 0; train/val_chains.jsonl = chains 1.. of the compacted conversations)
SOURCES = {  # name -> glob; the part before '-' is the family that SHARES weights
    'swesmith-tool': '/kaggle/input/**/swe-smith-trajectories-transformed/sft_v1.jsonl',
    'swesmith-xml': '/kaggle/input/**/swe-smith-trajectories-transformed/xml_sft_v1.jsonl',
    'swegym': '/kaggle/input/**/swe-gym-trajectories-transformed/sft_v1.jsonl',
    'r2egym': '/kaggle/input/**/r2egym-trajectories-transformed/sft_v1.jsonl',
    'swezero': '/kaggle/input/**/swe-zero-harness-replays/sft_zero_v1.jsonl',
}
EXTRA_SOURCES = {}  # @define (added to SOURCES, e.g. {'swezero-r2': '/kaggle/input/**/zero_replay/sft_zero_v1.jsonl'})
SOURCES.update(EXTRA_SOURCES)
DROP_FAMILIES = []  # @define (source families left out entirely, e.g. teachers whose thoughts are one-line narration: B27)
SOURCES = {k: v for k, v in SOURCES.items() if k.split('-')[0] not in DROP_FAMILIES}
SHARES = {'swezero': 0.25, 'swesmith': 0.40, 'r2egym': 0.25, 'swegym': 0.10}  # @define (by source family)
EXCLUDE_REPOS = ['fastapi/fastapi', 'textualize/rich', 'encode/httpx', 'psf/requests', 'encode/starlette']  # @define
EXCLUDE_SUBSTR = ['fastapi', 'starlette', 'httpx']  # @define
MAX_CALLS = 28  # @define
ALLOW_NO_SUBMIT = []  # @define (source families whose resolved rows may end without submit_patch: the harness extracts the patch itself, README 8.1)
MAX_TOKENS = 24000  # @define
PER_INSTANCE = 2  # @define
PAIR_MODE = ''  # @define (DPO pairs, chosen before compaction so unpaired runs cost no summaries; '+'-joined kinds:
               # 'fail' (B127): a task with a resolved and an unresolved run -> the resolved run with the fewest tool
               #   calls vs the unresolved run closest to it in length;
               # 'speed' (user 10-25, B130): a task with only resolved runs -> fewest vs most tool calls, if the gap
               #   is >= SPEED_GAP calls and >= SPEED_REL of the slower run, and the faster run still verifies
               #   (a pytest/python command after its last edit) whenever the slower one did)
SPEED_GAP = 3  # @define
SPEED_REL = 0.25  # @define
# serving-time model of a run (B142, 7,980 calls of the unseen-repo evals): every model call costs ~LAT_FIXED_S plus
# LAT_PER_TOKEN_S per generated token; each paired row carries pair_cost (calls, turns, generated tokens, est_s) so the
# trainer can set a per-pair preference strength from the time gap (clock-calibrated IPO, B154)
LAT_FIXED_S = 16.7  # @define
LAT_PER_TOKEN_S = 0.139  # @define
REPO_CAP = 250  # @define
VAL_SHARE = 0.04  # @define
SEED = 20261006  # @define
RENDER_CHECK_N = 40  # @define (re-render this many rows per source from `messages` and compare with `text`)
TOOLS_JSON = None  # @embed templates/harness_tools_request.json
T0 = time.time()
OUT = Path('/kaggle/working/mix')
OUT.mkdir(parents=True, exist_ok=True)
R2E_REPOS = {'pandas': 'pandas-dev/pandas', 'numpy': 'numpy/numpy', 'pillow': 'python-pillow/pillow', 'aiohttp': 'aio-libs/aiohttp',
             'tornado': 'tornadoweb/tornado', 'coveragepy': 'nedbat/coveragepy', 'datalad': 'datalad/datalad',
             'pyramid': 'pylons/pyramid', 'orange3': 'biolab/orange3', 'scrapy': 'scrapy/scrapy', 'sympy': 'sympy/sympy',
             'matplotlib': 'matplotlib/matplotlib', 'moto': 'getmoto/moto'}
SCRATCH_RX = re.compile(r'^(?:/workspace/)?(test_[^/]*\.py|[^/]*_test\.py)$')
TEST_RX = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$|pytest\.ini$)')
MARKUP_RX = re.compile(r'OBSERVATION:|<function_results>|<function=|<parameter=|</function>|EXECUTION RESULT of \[|'
                       r'Execution output of \[|\b(str_replace_editor|file_editor|execute_bash)\b')


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def pct(v):
    v = sorted(v)
    return {f'p{q}': v[min(len(v) - 1, int(len(v) * q / 100))] for q in (10, 50, 90)} if v else {}


def family(src):
    return src.split('-')[0]


def repo_of(src, iid):
    if src == 'r2egym':
        return R2E_REPOS.get(iid.split('__')[0], iid.split('__')[0])
    head = iid.split('.')[0] if src.startswith('swesmith') else re.sub(r'-\d+$', '', iid)
    return head.replace('__', '/', 1).lower()


def args_of(tc):
    a = tc['function']['arguments']
    return json.loads(a) if isinstance(a, str) else (a or {})


def scratch_renames(msgs):
    """{old: new} for root-level test-named scratch scripts the conversation creates; None if it edits a real test file."""
    created = set()
    for m in msgs:
        for tc in (m.get('tool_calls') or []):
            name, path = tc['function']['name'], str(args_of(tc).get('filepath', ''))
            if name not in ('edit_file', 'write_file') or not TEST_RX.search(path):
                continue
            mm = SCRATCH_RX.match(path)
            if mm and name == 'write_file':
                created.add(mm.group(1))
            elif not mm or mm.group(1) not in created:
                return None
    blob, renames = json.dumps(msgs), {}
    for old in sorted(created):
        stem = old[:-3]
        for new in ((f'repro_{stem[5:]}.py' if stem.startswith('test_') else f'{stem[:-5]}_repro.py'), f'scratch_{len(renames)}.py'):
            if new not in blob and not TEST_RX.search(new):
                renames[old] = new
                break
        else:
            return None
    return renames


def substitute(obj, subs):
    if isinstance(obj, str):
        for rx, new in subs:
            obj = rx.sub(new, obj)
        return obj
    if isinstance(obj, list):
        return [substitute(x, subs) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, subs) for k, v in obj.items()}
    return obj


def fix_messages(msgs, renames):
    """Renamed scratch scripts after the task message; R2E [ISSUE] markers stripped from the task message."""
    out = [dict(m) for m in msgs]
    first_user = next(i for i, m in enumerate(out) if m['role'] == 'user')
    if isinstance(out[first_user].get('content'), str):
        out[first_user]['content'] = re.sub(r'\[ISSUE\]\s*|\s*\[/ISSUE\]', '', out[first_user]['content'])
    subs = [(re.compile(r'(?<![\w-])' + re.escape(old) + r'(?!\w)'), new) for old, new in renames.items()]
    return out[:first_user + 1] + [substitute(m, subs) for m in out[first_user + 1:]]


try:
    from transformers import AutoTokenizer
    import bisect
    tok_dir = None
    for c in sorted(glob.glob('/kaggle/input/**/tokenizer.json', recursive=True)):
        if 'w4a16' in c or tok_dir is None:
            tok_dir = Path(c).parent
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if (tok_dir / 'chat_template.jinja').exists():
        tok.chat_template = (tok_dir / 'chat_template.jinja').read_text()
    ns = {'__name__': 'build_sft'}
    exec(compile(BUILD_SFT_PY, 'build_sft.py', 'exec'), ns)
    trainable_spans = ns['trainable_spans']
    tools = json.loads(TOOLS_JSON)

    cs = None
    if COMPACT or NORMALIZE_SYSTEM:
        ns_c = {'__name__': 'compaction_sim'}
        exec(compile(COMPACT_SIM_PY, 'compaction_sim.py', 'exec'), ns_c)
        cs = types.SimpleNamespace(**ns_c)
        SYSTEM_MSG = {'role': 'system', 'content': SYSTEM_TEXT}
        PLACEHOLDER = 'Summary of the earlier steps (replay placeholder): files were read and edited as above.'
        GEMMA, PENDING, DROP = {}, {}, set()
        if SUMMARY_SOURCE == 'gemma':
            for f_ in sorted(glob.glob(GEMMA_SUMMARIES_GLOB, recursive=True)):
                for line_ in open(f_):
                    x_ = json.loads(line_) if line_.strip() else {}
                    if x_.get('hash') and x_.get('drop'):  # no usable summary (e.g. Gemma role-plays the agent, B41)
                        DROP.add((int(x_['chain']), x_['hash']))
                    elif x_.get('hash') and isinstance(x_.get('summary'), str) and x_['summary'].strip():
                        GEMMA[(int(x_['chain']), x_['hash'])] = x_['summary']
            log(f'gemma summaries loaded: {len(GEMMA)}, dropped prompts: {len(DROP)} from {GEMMA_SUMMARIES_GLOB}')
        refine = {'refined': 0, 'diff': collections.Counter()}

    def content_format(template):
        """vLLM 0.19.1 `_detect_content_format`: 'openai' (content parts handed to the template) if the template loops
        over a message's content, else 'string' (text parts joined with a newline before templating)."""
        import jinja2.nodes as jn
        from transformers.utils.chat_template_utils import _compile_jinja_template
        root = _compile_jinja_template(template).environment.parse(template)

        def is_var(n, v):
            return isinstance(n, jn.Name) and n.ctx == 'load' and n.name == v

        def is_attr(n, v, key):
            if isinstance(n, jn.Getitem):
                return is_var(n.node, v) and isinstance(n.arg, jn.Const) and n.arg.value == key
            if isinstance(n, jn.Getattr):
                return is_var(n.node, v) and n.attr == key
            return False

        def is_ve(n, v, key=None):
            if isinstance(n, jn.Filter):
                return n.node is not None and is_ve(n.node, v, key)
            if isinstance(n, jn.Test):
                return is_ve(n.node, v, key)
            if isinstance(n, jn.Getitem) and isinstance(n.arg, jn.Slice):
                return is_ve(n.node, v, key)
            return is_attr(n, v, key) if key else is_var(n, v)

        names, queue = ['messages'], ['messages']
        while queue:
            v = queue.pop(0)
            for a_ in root.find_all(jn.Assign):
                if is_ve(a_.node, v) and isinstance(a_.target, jn.Name) and a_.target.name not in names:
                    names.append(a_.target.name)
                    queue.append(a_.target.name)
        msg_vars = [lp.target.name for lp in root.find_all(jn.For)
                    if isinstance(lp.target, jn.Name) and any(is_ve(lp.iter, v) for v in names)]
        return 'openai' if any(is_ve(lp.iter, v, 'content') for lp in root.find_all(jn.For) for v in msg_vars) else 'string'

    CF = content_format(tok.chat_template) if (COMPACT or NORMALIZE_SYSTEM) else None
    if CF:
        log(f'vLLM content format of the chat template: {CF}')

    def as_served(msgs):
        """List content as vLLM hands it to the template ('string' format: text parts joined with a newline)."""
        if CF != 'string':
            return msgs
        return [{**m, 'content': '\n'.join(p_.get('text', '') for p_ in m['content'] if p_.get('type', 'text') == 'text')}
                if isinstance(m.get('content'), list) else m for m in msgs]

    def render(msgs, tools_=None, gen=False):
        return tok.apply_chat_template(as_served(msgs), tools=tools_, tokenize=False, enable_thinking=True,
                                       add_generation_prompt=gen)

    def n_tok(text):
        return len(tok(text, add_special_tokens=False)['input_ids'])

    GEN_TOOL = 0
    if cs is not None:  # tokens the generation prompt adds after a tool result (template-dependent)
        probe = [{'role': 'user', 'content': 'x'},
                 {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c0', 'type': 'function', 'function': {
                     'name': 'run_command', 'arguments': {'command': 'ls'}}}]},
                 {'role': 'tool', 'tool_call_id': 'c0', 'content': 'ok'}]
        t0_, t1_ = render(probe, tools), render(probe, tools, gen=True)
        if not t1_.startswith(t0_):
            raise RuntimeError(f'generation prompt after a tool result is not a suffix: {t0_[-80:]!r} / {t1_[-80:]!r}')
        GEN_TOOL = n_tok(t1_) - n_tok(t0_)
        log(f'generation prompt after a tool result: {t1_[len(t0_):]!r} = {GEN_TOOL} tokens')

    def make_counts_fn(tools_, refine_on=True):
        """Prompt tokens of the request for each assistant message: from one render's offsets (+ the forced thought
        opener after a tool result); counts within REFINE of the threshold are recomputed from the exact request."""
        def counts_fn(msgs):
            text = render(msgs, tools_)
            spans = trainable_spans(text)
            a_idx = cs.response_starts(msgs)
            if len(spans) != len(a_idx):
                return None
            starts = [x for x, _ in tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']]
            out = []
            for (c0, _), i in zip(spans, a_idx):
                n = bisect.bisect_left(starts, c0) + (GEN_TOOL if text.endswith('<tool_response|>', 0, c0) else 0)
                if refine_on and abs(n - cs.THRESHOLD) <= REFINE:
                    n2 = n_tok(render(msgs[:i], tools_, gen=True))
                    refine['refined'] += 1
                    refine['diff'][n2 - n] += 1
                    n = n2
                out.append(n)
            return out
        return counts_fn

    def vllm_like(msgs):
        """Captured request messages as vLLM hands them to the template: tool-call arguments decoded, list content kept."""
        out = []
        for m in msgs:
            m2 = {k_: v_ for k_, v_ in m.items() if k_ in ('role', 'content', 'tool_calls', 'tool_call_id', 'name', 'reasoning')}
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

    def parity_check():
        """The simulator against the captured scorer-path requests of g4dev-format-parity-cpu."""
        found = sorted(glob.glob(PARITY_GLOB, recursive=True))
        un_f = [f for f in found if f.endswith('_unmatched.json')]
        files = [f for f in found if not f.endswith('_unmatched.json')]
        if len(un_f) != 1 or not files:
            raise RuntimeError(f'parity captures not found under {PARITY_GLOB}: {found[:5]}')
        un = json.load(open(un_f[0]))
        summ = [x for x in un if not x['request'].get('tools')]
        post = [x for x in un if x['request'].get('tools')]
        rep = {'tasks': 0, 'requests': 0, 'count_exact': 0, 'offset_diff': collections.Counter(), 'trigger_ok': 0,
               'prompt_ok': 0, 'post_msgs_ok': 0, 'post_render_ok': 0, 'post_count_ok': 0, 'failures': []}
        for f in files:
            d = json.load(open(f))
            last = d[-1]['request']
            tools_ = last.get('tools')
            rep['tasks'] += 1
            for x in d:  # 1. exact request token counts = the counts the replay reported
                rep['requests'] += 1
                rep['count_exact'] += n_tok(render(vllm_like(x['request']['messages']), x['request'].get('tools'), gen=True)) \
                    == (x.get('usage') or {}).get('prompt_tokens')
            pc = [x['request'] for x in post if x['request']['messages'][0] == last['messages'][0]
                  and json.dumps(x['request']['messages'][2:6]) == json.dumps(last['messages'][-4:])]
            if not pc:
                rep['failures'].append(f'{f}: no post-compaction request')
                continue
            pmsgs = vllm_like(pc[0]['messages'])
            conv = vllm_like(last['messages']) + pmsgs[6:8] + [{'role': 'assistant', 'content': '(next)'}]
            exact = [n_tok(render(conv[:i], tools_, gen=True)) for i, m in enumerate(conv) if m['role'] == 'assistant']
            offs = make_counts_fn(tools_, refine_on=False)(conv)
            for a_, b_ in zip(exact, offs or []):  # 2. offset counts vs exact
                rep['offset_diff'][a_ - b_] += 1
            parts = cs.simulate(conv[0], conv[1:], make_counts_fn(tools_), lambda lines: PLACEHOLDER)
            if not parts:
                rep['failures'].append(f'{f}: simulate returned nothing')
                continue
            seg = [x for x in parts if x[0] == 'segment']
            sp = [x for x in parts if x[0] == 'summary']
            n_a0 = sum(m['role'] == 'assistant' for m in seg[0][1])
            ok_trig = len(sp) == 1 and n_a0 == len(d) and len(seg) == 2  # 3. compaction right after the last captured call
            task = conv[1]['content'] if isinstance(conv[1]['content'], str) else json.dumps(conv[1]['content'])
            target = next((x['request']['messages'][0]['content'] for x in summ if task[:300] in x['request']['messages'][0]['content']), None)
            ok_prompt = bool(sp) and sp[0][1] == target  # 4. summary request text
            ok_msgs = len(seg) == 2 and json.dumps(seg[1][1][:8], sort_keys=True) == json.dumps(pmsgs[:8], sort_keys=True)
            ok_render = len(seg) == 2 and render(seg[1][1][:8], tools_, gen=True) == render(pmsgs, tools_, gen=True)
            ok_count = len(seg) == 2 and n_tok(render(seg[1][1][:8], tools_, gen=True)) == (
                next(x for x in post if x['request'] is pc[0]).get('usage') or {}).get('prompt_tokens')
            rep['trigger_ok'] += ok_trig
            rep['prompt_ok'] += ok_prompt
            rep['post_msgs_ok'] += ok_msgs
            rep['post_render_ok'] += ok_render
            rep['post_count_ok'] += ok_count
            if not (ok_trig and ok_prompt and ok_msgs and ok_render and ok_count):  # count: informational
                rep['failures'].append(f'{Path(f).name}: trigger {ok_trig} prompt {ok_prompt} msgs {ok_msgs} render {ok_render} count {ok_count}')
        rep['offset_diff'] = dict(rep['offset_diff'])
        rep['content_format'] = CF
        rep['passed'] = (rep['tasks'] > 0 and rep['count_exact'] == rep['requests'] and
                         rep['trigger_ok'] == rep['prompt_ok'] == rep['post_msgs_ok'] == rep['post_render_ok'] == rep['tasks'])
        return rep

    def rerender_rows(rows_in, render_check_, rej_):
        """System prompt normalized; with COMPACT each conversation becomes the scorer's segments + summary requests."""
        info = {'parity': None, 'by_source': {}}
        if COMPACT and PARITY_GLOB:
            info['parity'] = parity_check()
            log(f'parity: {json.dumps(info["parity"])[:1200]}')
            if not info['parity']['passed']:
                raise RuntimeError('compaction simulator does not reproduce the captured requests: ' + json.dumps(info['parity'])[:800])
        counts_fn = make_counts_fn(tools)
        kept, n_sum, sum_chars = [], [0], []

        def _render_parts(parts, r, st, count=True):
            """-> (rendered lines, reject reason). count: add summary rows to the summary stats (chain 0 only)."""
            out, bad = [], None
            for kind, a_, b_ in parts:
                if kind == 'segment':
                    text = render(a_, tools)
                    spans = [list(x) for x in trainable_spans(text)]
                    if len(spans) != len(cs.response_starts(a_)):
                        bad = 'span_mismatch'
                        break
                    spans = spans[b_:]
                    if THOUGHT_TURNS_ONLY:
                        spans = [sp for sp in spans if text.startswith('<|channel>thought', sp[0])]
                else:
                    if SUMMARY_KEEP < 1.0 and random.Random(f'{SEED}:{r["ref"]}').random() >= SUMMARY_KEEP:
                        st['summary_dropped'] += count
                        continue
                    text = render([{'role': 'user', 'content': a_}, {'role': 'assistant', 'content': b_}])
                    spans = [list(x) for x in trainable_spans(text)]
                    if count:
                        n_sum[0] += 1
                        sum_chars.append(len(b_))
                nt, nl = label_counts(text, spans)
                if nt > MAX_TOKENS:
                    bad = 'too_many_tokens'
                    break
                out.append({'kind': kind, 'text': text, 'spans': spans, 'n_tokens': nt, 'n_trainable': nl})
            return out, bad
        for r in rows_in:
            src = r['source']
            st = info['by_source'].setdefault(src, collections.Counter())
            msgs = r.pop('_msgs')
            chk = render_check_.get(src) or {}
            if chk.get('mismatched') or not chk.get('checked'):
                rej_[(src, 'rerender_unverified')] += 1
                continue
            if not msgs or msgs[0]['role'] != 'system':
                rej_[(src, 'no_system_message')] += 1
                continue
            sys_m = SYSTEM_MSG if NORMALIZE_SYSTEM else msgs[0]
            st['system_changed'] += msgs[0].get('content') != sys_m['content']
            chain_parts = {}
            if COMPACT and SUMMARY_SOURCE == 'gemma':
                def run_chain(k, ref=r['ref']):
                    state = {'missing': False, 'dropped': False}

                    def gemma_fn(lines):
                        prompt = cs.prompt_of(lines)
                        h = hashlib.sha1(prompt.encode()).hexdigest()
                        got = None if state['missing'] else GEMMA.get((k, h))
                        if got is not None:
                            return got
                        if not state['missing'] and (k, h) in DROP:  # unusable: this chain is left out, not pending
                            state['missing'] = state['dropped'] = True
                        if not state['missing']:  # later prompts of this chain depend on this summary: next round
                            e_ = PENDING.setdefault(h, {'hash': h, 'prompt': prompt, 'chains': [], 'refs': []})
                            if k not in e_['chains']:
                                e_['chains'].append(k)
                            if ref not in e_['refs']:
                                e_['refs'].append(ref)
                            state['missing'] = True
                        return cs.summarize(lines)  # placeholder only: an incomplete chain is not written
                    parts_k = cs.simulate(sys_m, msgs[1:], counts_fn, gemma_fn)
                    st['chain_dropped'] += state['dropped']
                    return parts_k, state['missing']
                parts, missing = run_chain(0)
                if parts and any(x_[0] == 'summary' for x_ in parts):
                    for k_ in range(1, CHAINS):
                        p_k, miss_k = run_chain(k_)
                        if p_k and not miss_k:
                            chain_parts[k_] = p_k
                        st['chain_pending'] += bool(miss_k)
                if missing:
                    rej_[(src, 'summary_pending')] += 1
                    continue
            elif COMPACT:
                parts = cs.simulate(sys_m, msgs[1:], counts_fn, cs.summarize)
            else:
                parts = [('segment', [sys_m] + msgs[1:], 0)]
            if not parts:
                rej_[(src, 'simulation_failed')] += 1
                continue
            out, bad = _render_parts(parts, r, st)
            chain_out = {}
            for k_, p_k in chain_parts.items():
                o_k, b_k = _render_parts(p_k, r, st, count=False)
                if not b_k:
                    chain_out[k_] = o_k
            if bad:
                rej_[(src, bad)] += 1
                continue
            r.pop('text', None)
            r.pop('spans', None)
            r['parts'] = out
            if chain_out:
                r['chain_parts'] = chain_out
            # from the simulation, not the written lines: SUMMARY_KEEP drops summary lines (mix7's stats under-counted
            # compactions this way: 331 reported, 889 conversations actually compact, 18r)
            r['compactions'] = sum(x_[0] == 'summary' for x_ in parts)
            r['n_tokens'] = sum(p_['n_tokens'] for p_ in out)
            r['n_trainable'] = sum(p_['n_trainable'] for p_ in out)
            st['kept'] += 1
            st['compacted'] += r['compactions'] > 0
            st['compactions'] += r['compactions']
            st['chains_complete'] += len(chain_out)
            kept.append(r)
        info['by_source'] = {k_: dict(v_) for k_, v_ in info['by_source'].items()}
        info['summaries'] = {'n': n_sum[0], 'chars': pct(sum_chars)}
        if SUMMARY_SOURCE == 'gemma':
            info['gemma'] = {'loaded': len(GEMMA), 'pending_prompts': len(PENDING),
                             'pending_requests': sum(len(e_['chains']) for e_ in PENDING.values())}
            with open(OUT / 'pending.jsonl', 'w') as fh_:
                for e_ in PENDING.values():
                    fh_.write(json.dumps(e_) + '\n')
        if cs is not None and COMPACT:
            info['refine'] = {'refined': refine['refined'], 'diff': dict(refine['diff'])}
        return kept, info

    rng = random.Random(SEED)
    rej = collections.Counter()
    rows, inputs, render_check = [], {}, {}
    seen_text = set()
    for src, pattern in SOURCES.items():
        files = sorted(glob.glob(pattern, recursive=True))
        if len(files) != 1:
            raise RuntimeError(f'{src}: expected one file for {pattern}, found {files}')
        inputs[src] = {'path': files[0], 'bytes': Path(files[0]).stat().st_size, 'rows': 0, 'kept': 0}
        checked = mismatched = 0
        pending = []
        for k, line in enumerate(open(files[0])):
            e = json.loads(line)
            inputs[src]['rows'] += 1
            iid = str(e['instance_id'])
            repo = repo_of(src, iid)
            msgs = e['messages']
            calls = [tc for m in msgs for tc in (m.get('tool_calls') or [])]
            trained = ''.join(e['text'][b:x] for b, x in e['spans'])
            task = next((m['content'] for m in msgs if m['role'] == 'user'), '')
            task = task if isinstance(task, str) else json.dumps(task)
            renames = scratch_renames(msgs)
            reason = None
            if repo in EXCLUDE_REPOS or any(s_ in repo for s_ in EXCLUDE_SUBSTR):
                reason = 'excluded_repo'
            elif MARKUP_RX.search(trained):
                reason = 'teacher_markup'
            elif renames is None:
                reason = 'edits_test_files'
            elif len(calls) > MAX_CALLS:
                reason = 'over_budget'
            elif not calls:
                reason = 'no_calls'
            elif calls[-1]['function']['name'] != 'submit_patch' and family(src) not in ALLOW_NO_SUBMIT:
                reason = 'no_final_submit'
            if reason is None:
                h = hashlib.md5(e['text'].encode()).hexdigest()
                if h in seen_text:
                    reason = 'duplicate'
                seen_text.add(h)
            if reason:
                rej[(src, reason)] += 1
                continue
            row = {'instance_id': iid, 'source': src, 'repo': repo, 'ref': f'{src}:{k}', 'text': e['text'],
                   'spans': [list(x) for x in e['spans']], 'calls': len(calls), 'fixed': [],
                   'resolved': e.get('resolved', True)}
            if COMPACT or NORMALIZE_SYSTEM:
                row['_msgs'] = msgs
            if renames or '[ISSUE]' in task or '[/ISSUE]' in task:
                row['fixed'] = (['scratch_renamed'] if renames else []) + (['issue_markers'] if '[ISSUE]' in task else [])
                pending.append((row, fix_messages(msgs, renames)))
                continue
            if checked < RENDER_CHECK_N and rng.random() < 0.2:
                checked += 1
                again = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=True)
                mismatched += again != e['text']
            rows.append(row)
        render_check[src] = {'checked': checked, 'mismatched': mismatched, 'to_fix': len(pending)}
        for row, fixed_msgs in pending:  # re-render only if this source's untouched rows re-render byte-identically
            if mismatched or not checked:
                rej[(src, 'fix_needs_rerender')] += 1
                continue
            row['text'] = tok.apply_chat_template(fixed_msgs, tools=tools, tokenize=False, enable_thinking=True)
            row['spans'] = [list(x) for x in trainable_spans(row['text'])]
            if COMPACT or NORMALIZE_SYSTEM:
                row['_msgs'] = fixed_msgs
            leftover = [tc for m in fixed_msgs for tc in (m.get('tool_calls') or []) if tc['function']['name'] in ('edit_file', 'write_file')
                        and TEST_RX.search(str(args_of(tc).get('filepath', '')))]
            if leftover:
                rej[(src, 'rename_incomplete')] += 1
                continue
            rows.append(row)
        log(f'{src}: {inputs[src]["rows"]} rows, kept so far {sum(1 for r in rows if r["source"] == src)}; render check {render_check[src]}')

    def label_counts(text, spans):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = [s for s, _ in spans]
        n_lab = 0
        for c0, _ in enc['offset_mapping']:
            j = bisect.bisect_right(starts, c0) - 1
            n_lab += j >= 0 and spans[j][0] <= c0 < spans[j][1]
        return len(enc['input_ids']), n_lab

    def verifies_after_last_edit(msgs):
        calls_ = [tc for m in msgs for tc in (m.get('tool_calls') or [])]
        last_edit = max((j for j, tc in enumerate(calls_) if tc['function']['name'] in ('edit_file', 'write_file')), default=-1)
        return any(tc['function']['name'] == 'run_command' and re.search(r'\bpytest\b|\bpython3?\b', str(args_of(tc).get('command', '')))
                   for tc in calls_[last_edit + 1:])

    def run_cost(r):
        """Generated tokens (thoughts, text, tool-call arguments of every assistant turn) and the serving-time
        estimate LAT_FIXED_S x turns + LAT_PER_TOKEN_S x tokens of the whole run (before compaction)."""
        turns, gen = 0, []
        for m in r.get('_msgs') or []:
            if m.get('role') != 'assistant':
                continue
            turns += 1
            gen.append(str(m.get('reasoning_content') or m.get('reasoning') or ''))
            gen.append(str(m.get('content') or ''))
            gen += [json.dumps(args_of(tc)) for tc in (m.get('tool_calls') or [])]
        n_gen = len(tok('\n'.join(gen), add_special_tokens=False)['input_ids']) if gen else 0
        return {'calls': r['calls'], 'turns': turns, 'gen_tokens': n_gen,
                'est_s': round(LAT_FIXED_S * turns + LAT_PER_TOKEN_S * n_gen, 1)}

    if PAIR_MODE:  # rows keep pair_kind / pair_side; dpo_pairs.py tags them for the trainer
        kinds = set(PAIR_MODE.split('+'))
        by_task = collections.defaultdict(lambda: {True: [], False: []})
        for r in rows:
            by_task[(r['repo'], r['instance_id'])][bool(r['resolved'])].append(r)
        paired, n_kind = [], collections.Counter()
        for key, sides in by_task.items():
            ok_, bad_ = sorted(sides[True], key=lambda r: (r['calls'], len(r['text']), r['ref'])), sides[False]
            pick = []
            if ok_ and bad_ and 'fail' in kinds:  # the outcome signal is the stronger one: a task gives one pair only
                c = ok_[0]
                x = min(bad_, key=lambda r: (abs(len(r['text']) - len(c['text'])), r['ref']))
                pick = [(c, 'fail', 'c'), (x, 'fail', 'r')]
            elif len(ok_) >= 2 and not bad_ and 'speed' in kinds:
                c, x = ok_[0], ok_[-1]
                gap = x['calls'] - c['calls']
                if gap < SPEED_GAP or gap < SPEED_REL * x['calls']:
                    rej[(c['source'], 'speed_gap_small')] += 1
                elif verifies_after_last_edit(x.get('_msgs') or []) and not verifies_after_last_edit(c.get('_msgs') or []):
                    rej[(c['source'], 'speed_skips_verification')] += 1
                else:
                    pick = [(c, 'speed', 'c'), (x, 'speed', 'r')]
            for r, kind, side in pick:
                r['pair_kind'], r['pair_side'] = kind, side
                r['pair_cost'] = run_cost(r)
                paired.append(r)
            n_kind[pick[0][1] if pick else 'none'] += 1
            for r in sides[True] + sides[False]:
                if not any(r is p_[0] for p_ in pick):
                    rej[(r['source'], 'pair_unused')] += 1
        rows = paired
        log(f'pair mode {PAIR_MODE}: tasks by pair kind {dict(n_kind)}')

    compaction = {}
    if COMPACT or NORMALIZE_SYSTEM:
        rows, compaction = rerender_rows(rows, render_check, rej)
        log(f'compaction stage: {json.dumps(compaction)[:1500]}')

    # exact token counts with train_lora.py's labelling (offsets inside spans -> trained)
    for r in rows:
        if 'parts' not in r:
            r['n_tokens'], r['n_trainable'] = label_counts(r['text'], r['spans'])
    over = [r for r in rows if r['n_tokens'] > MAX_TOKENS and 'parts' not in r]
    for r in over:
        rej[(r['source'], 'too_many_tokens')] += 1
    rows = [r for r in rows if 'parts' in r or r['n_tokens'] <= MAX_TOKENS]
    log(f'tokenized {len(rows)} rows')

    # at most PER_INSTANCE conversations per task (shortest first), at most REPO_CAP per repository (random)
    by_iid = collections.defaultdict(list)
    for r in rows:
        by_iid[(r['repo'], r['instance_id'])].append(r)
    capped = []
    for key, rs in by_iid.items():
        rs.sort(key=lambda r: (r['n_tokens'], r['ref']))
        capped += rs[:PER_INSTANCE]
        for r in rs[PER_INSTANCE:]:
            rej[(r['source'], 'per_instance_cap')] += 1
    by_repo = collections.defaultdict(list)
    for r in capped:
        by_repo[r['repo']].append(r)
    rows = []
    for repo, rs in sorted(by_repo.items()):
        rng.shuffle(rs)
        rows += rs[:REPO_CAP]
        for r in rs[REPO_CAP:]:
            rej[(r['source'], 'repo_cap')] += 1
    by_repo = collections.Counter(r['repo'] for r in rows)

    # validation: whole repositories (each under 3% of the rows), chosen per source family (smallest first) until
    # that family has VAL_SHARE of its rows held out, so the validation sample is not one family's
    order = sorted(by_repo, key=lambda x: hashlib.md5(f'{SEED}:{x}'.encode()).hexdigest())
    fam_rows = collections.Counter(family(r['source']) for r in rows)
    fam_repo = collections.Counter((family(r['source']), r['repo']) for r in rows)
    val_repos = []
    for fam in sorted(fam_rows, key=lambda f: fam_rows[f]):
        for repo in order:
            if sum(fam_repo[(fam, v)] for v in val_repos) >= VAL_SHARE * fam_rows[fam]:
                break
            if repo not in val_repos and fam_repo[(fam, repo)] and by_repo[repo] <= 0.03 * len(rows):
                val_repos.append(repo)
    val = [r for r in rows if r['repo'] in val_repos]
    train = [r for r in rows if r['repo'] not in val_repos]

    # prefix-balanced order: weighted sampling without replacement (Efraimidis-Spirakis, key = log(u) / w in log space:
    # u ** (1 / w) underflows to 0 for small w and the stable sort then keeps the input order), w = family share / size
    fam_n = collections.Counter(family(r['source']) for r in train)
    for r in train:
        w = SHARES.get(family(r['source']), 0.05) / fam_n[family(r['source'])]
        r['_key'] = math.log(1.0 - rng.random()) / w
    train.sort(key=lambda r: -r['_key'])
    for r in train:
        r.pop('_key')
    def lines_of(r):
        if 'parts' not in r:
            return [r]
        base = {k_: v_ for k_, v_ in r.items() if k_ not in ('parts', 'chain_parts')}
        return [{**base, 'part': j, **p_} for j, p_ in enumerate(r['parts'])]

    def chain_lines_of(r):
        base = {k_: v_ for k_, v_ in r.items() if k_ not in ('parts', 'chain_parts')}
        return [{**base, 'chain': k_, 'part': j, **p_} for k_, ps in sorted((r.get('chain_parts') or {}).items())
                for j, p_ in enumerate(ps)]

    for name, part in (('train', train), ('val', val)):
        with open(OUT / f'{name}.jsonl', 'w') as fh:
            for r in part:
                for line in lines_of(r):
                    fh.write(json.dumps(line) + '\n')
        if SUMMARY_SOURCE == 'gemma' and CHAINS > 1:  # same rows and order, the other summary chains (B41)
            with open(OUT / f'{name}_chains.jsonl', 'w') as fh:
                for r in part:
                    for line in chain_lines_of(r):
                        fh.write(json.dumps(line) + '\n')

    def summary(part):
        lines = [x for r in part for x in lines_of(r)]
        return {'rows': len(part), 'lines': len(lines), 'lines_by_kind': dict(collections.Counter(x.get('kind', 'full') for x in lines)),
                'line_tokens': pct([x['n_tokens'] for x in lines]),
                'line_tokens_max': max([x['n_tokens'] for x in lines] or [0]),
                'trainable_by_kind': dict(collections.Counter({k_: sum(x['n_trainable'] for x in lines if x.get('kind', 'full') == k_)
                                                               for k_ in {x.get('kind', 'full') for x in lines}})),
                'compactions': dict(collections.Counter(r.get('compactions', 0) for r in part)),
                'tokens': sum(r['n_tokens'] for r in part), 'trainable': sum(r['n_trainable'] for r in part),
                'by_source': dict(collections.Counter(r['source'] for r in part)), 'repos': len({r['repo'] for r in part}),
                'n_tokens': pct([r['n_tokens'] for r in part]), 'n_trainable': pct([r['n_trainable'] for r in part]),
                'calls': pct([r['calls'] for r in part]),
                'fixed': dict(collections.Counter(f for r in part for f in r['fixed']))}
    prefixes = {}
    train_lines = [x for r in train for x in lines_of(r)]
    for n in (250, 500, 1000, 2000, 4000, 6000):
        if n <= len(train_lines):
            p = train_lines[:n]
            prefixes[n] = {'tokens': sum(r['n_tokens'] for r in p), 'trainable': sum(r['n_trainable'] for r in p),
                           'by_family': dict(collections.Counter(family(r['source']) for r in p)),
                           'top_repos': collections.Counter(r['repo'] for r in p).most_common(5)}
    rej_tab = collections.defaultdict(dict)
    for (src, reason), n in sorted(rej.items()):
        rej_tab[src][reason] = n
    stats = {'inputs': inputs, 'render_check': render_check, 'compaction': compaction, 'rejects': rej_tab,
             'train': summary(train), 'val': summary(val),
             'chains': {name: {'rows_with_chains': sum(1 for r in part if r.get('chain_parts')),
                               'chain_lines': sum(len(chain_lines_of(r)) for r in part)} for name, part in (('train', train), ('val', val))},
             'val_repos': val_repos, 'top_repos': by_repo.most_common(15), 'prefixes': prefixes,
             'settings': {'SHARES': SHARES, 'DROP_FAMILIES': DROP_FAMILIES, 'EXCLUDE_REPOS': EXCLUDE_REPOS, 'EXCLUDE_SUBSTR': EXCLUDE_SUBSTR, 'MAX_CALLS': MAX_CALLS,
                          'MAX_TOKENS': MAX_TOKENS, 'PER_INSTANCE': PER_INSTANCE, 'REPO_CAP': REPO_CAP, 'VAL_SHARE': VAL_SHARE,
                          'ALLOW_NO_SUBMIT': ALLOW_NO_SUBMIT, 'SUMMARY_KEEP': SUMMARY_KEEP, 'THOUGHT_TURNS_ONLY': THOUGHT_TURNS_ONLY, 'SUMMARY_SOURCE': SUMMARY_SOURCE, 'CHAINS': CHAINS, 'COMPACT': COMPACT, 'NORMALIZE_SYSTEM': NORMALIZE_SYSTEM, 'PARITY_GLOB': PARITY_GLOB,
                          'SEED': SEED, 'tokenizer': str(tok_dir)},
             'minutes': round((time.time() - T0) / 60, 1)}
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1))
    log('STATS ' + json.dumps(stats))
except Exception as e:  # noqa: BLE001
    log(f'FATAL {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}')
    (OUT / 'stats.json').write_text(json.dumps({'fatal': f'{type(e).__name__}: {e}'}, indent=1))
