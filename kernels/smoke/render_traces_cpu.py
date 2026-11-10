"""Render resolved harness traces (any mounted GPU/TPU run) into SFT examples in the deployment bundle's format.

Input: `results/task_results.jsonl` + `results/traces/trace_<id>.json` of mounted kernel outputs (glob RUN_GLOB).
Each resolved run within MAX_CALLS tool calls becomes one example: `build_sft.trace_to_messages`, the system prompt
replaced by the deployment bundle's (SYSTEM_TEXT; the original is kept in `orig_system`), rendered with the served
model's chat template and the deployment tool schema (TOOLS_JSON). Dropped: over MAX_TOKENS, writes to test files.
Qwen teacher fixes (CLEAN_MARKUP, SPLIT_PARALLEL): a tool call's tail sometimes stays inside the thinking (`...</parameter>
</function></tool_call>` then the real thought): only the text after the last `</tool_call>` is kept; turns with several
calls become one call per turn, results interleaved, the thought on the first (the Gemma template drops the thought of a
multi-call turn). Rows whose trained text still has tool markup are dropped.
DROP_NUDGES: the harness's continuation nudge is a user message, and the Gemma template drops every thought before the
last user message (rb2: 631 of 2,185 thoughts lost in the 71 nudged rows). The nudge and the text-only answer that
triggered it are removed, so the trajectory stays one tool-calling turn with all its thoughts.
UNPACK_GLOB: tarballs to extract first (daily AI-quota outputs, `<day>-s<k>.tar.gz` holding `<phase>/results/...`);
their results are rendered with RUN_GLOB's. THOUGHT_TURNS_ONLY: keep only trainable spans (model turns) that open with a
thought; turns that act without thinking stay context (B55: the proxy's Gemma thinks on ~40% of turns, the scorer's on
every turn). TRUNC_TOKENS: cut a longer row after the last complete model turn that fits (the trainer drops lines over
its probed length; early turns keep the localisation work).
Outputs: /kaggle/working/render/{sft.jsonl, stats.json}.
"""
import collections
import glob
import json
import re
import time
import traceback
from pathlib import Path

BUILD_SFT_PY = None  # @embed ../train/build_sft.py
SYSTEM_TEXT = None  # @embed templates/sub-v5b_system.txt
TOOLS_JSON = None  # @embed templates/harness_tools_request.json
RUN_GLOB = '/kaggle/input/**/results/task_results.jsonl'  # @define
SOURCE = 'harness-run'  # @define
MAX_CALLS = 28  # @define
MAX_TOKENS = 24000  # @define
CLEAN_MARKUP = False  # @define
SPLIT_PARALLEL = False  # @define
DROP_NUDGES = False  # @define
KEEP_SYSTEM = False  # @define (True: keep each trace's own system prompt, e.g. on-policy runs made with the deployment bundle)
UNPACK_GLOB = ''  # @define (e.g. '/kaggle/input/**/*.tar.gz')
THOUGHT_TURNS_ONLY = False  # @define
TRUNC_TOKENS = 0  # @define (0: off)
KEEP_UNRESOLVED = False  # @define (True: also render failed runs, tagged resolved=false, for DPO pairs, B127)
NUDGE_PREFIX = 'Please continue your work using the available tools'
T0 = time.time()
OUT = Path('/kaggle/working/render')
OUT.mkdir(parents=True, exist_ok=True)
TEST_RX = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$|pytest\.ini$)')
TOOL_MARKUP_RX = re.compile(r'</?tool_call>|<function=[^>]*>|</function>|<parameter=[^>]*>|</parameter>')


def clean_markup(text):
    """Keep what follows the last `</tool_call>` when a call's tail leaked into the text; '' if markup remains."""
    if not isinstance(text, str) or not TOOL_MARKUP_RX.search(text):
        return text
    tail = text.rsplit('</tool_call>', 1)[1] if '</tool_call>' in text else ''
    return '' if TOOL_MARKUP_RX.search(tail) else tail.strip()


def split_parallel(msgs):
    """One tool call per assistant turn, each followed by its own result; None if a result cannot be matched."""
    out, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get('tool_calls') or []
        if m['role'] != 'assistant' or len(calls) < 2:
            out.append(m)
            i += 1
            continue
        results, j = {}, i + 1
        while j < len(msgs) and msgs[j]['role'] == 'tool':
            results[msgs[j].get('tool_call_id')] = msgs[j]
            j += 1
        if any(c['id'] not in results for c in calls):
            return None
        for k, c in enumerate(calls):
            a = {'role': 'assistant', 'content': '', 'tool_calls': [c]}
            if k == 0 and m.get('reasoning'):
                a['reasoning'] = m['reasoning']
            out += [a, results[c['id']]]
        i = j
    return out


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def pct(v):
    v = sorted(v)
    return {f'p{q}': v[min(len(v) - 1, int(len(v) * q / 100))] for q in (10, 50, 90)} if v else {}


try:
    run_globs = [RUN_GLOB]
    if UNPACK_GLOB:
        import tarfile
        for tgz in sorted(glob.glob(UNPACK_GLOB, recursive=True)):
            dest = Path('/tmp/unpack') / Path(tgz).name.split('.')[0]
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz) as tf:
                tf.extractall(dest)
            log(f'unpacked {tgz} -> {dest}')
        run_globs.append('/tmp/unpack/**/results/task_results.jsonl')
    from transformers import AutoTokenizer
    tok_dir = None
    for c in sorted(glob.glob('/kaggle/input/**/tokenizer.json', recursive=True)):
        if 'w4a16' in c or tok_dir is None:
            tok_dir = Path(c).parent
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if (tok_dir / 'chat_template.jinja').exists():
        tok.chat_template = (tok_dir / 'chat_template.jinja').read_text()
    ns = {'__name__': 'build_sft'}
    exec(compile(BUILD_SFT_PY, 'build_sft.py', 'exec'), ns)
    trace_to_messages, trainable_spans = ns['trace_to_messages'], ns['trainable_spans']
    tools = json.loads(TOOLS_JSON)
    rej, tokens, calls = collections.Counter(), [], []
    fixed, thoughts = collections.Counter(), collections.Counter()
    run_files = sorted({f for g in run_globs for f in glob.glob(g, recursive=True)})
    with open(OUT / 'sft.jsonl', 'w') as out:
        for rf in run_files:
            rdir = Path(rf).parent
            for line in open(rf):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get('resolved') and not KEEP_UNRESOLVED:
                    rej['unresolved'] += 1
                    continue
                if (r.get('tool_calls') or 0) > MAX_CALLS:
                    rej['over_budget'] += 1
                    continue
                tf = rdir / 'traces' / f'trace_{r["instance_id"]}.json'
                if not tf.exists():
                    rej['no_trace'] += 1
                    continue
                msgs = trace_to_messages(json.load(open(tf)))
                if not msgs or msgs[0]['role'] != 'system':
                    rej['no_system'] += 1
                    continue
                if CLEAN_MARKUP:
                    for m in msgs:
                        if m['role'] == 'assistant':
                            for key in ('reasoning', 'content'):
                                if m.get(key):
                                    cleaned = clean_markup(m[key])
                                    fixed['markup_cleaned'] += cleaned != m[key]
                                    m[key] = cleaned
                            if not m.get('reasoning'):
                                m.pop('reasoning', None)
                if DROP_NUDGES:
                    first_user = next((k for k, m in enumerate(msgs) if m['role'] == 'user'), None)
                    keep = []
                    for k, m in enumerate(msgs):
                        if m['role'] == 'user' and k != first_user and str(m.get('content', '')).startswith(NUDGE_PREFIX):
                            fixed['nudges_dropped'] += 1
                            if keep and keep[-1]['role'] == 'assistant' and not keep[-1].get('tool_calls'):
                                keep.pop()
                            continue
                        keep.append(m)
                    msgs = keep
                    if sum(1 for m in msgs if m['role'] == 'user') > 1:
                        rej['other_user_turns'] += 1
                        continue
                if SPLIT_PARALLEL:
                    n_multi = sum(1 for m in msgs if len(m.get('tool_calls') or []) > 1)
                    msgs = split_parallel(msgs)
                    if msgs is None:
                        rej['split_unmatched'] += 1
                        continue
                    fixed['parallel_split'] += n_multi
                writes = [json.loads(c['function']['arguments']) if isinstance(c['function']['arguments'], str) else c['function']['arguments']
                          for m in msgs for c in (m.get('tool_calls') or []) if c['function']['name'] in ('edit_file', 'write_file')]
                if any(TEST_RX.search(str(w.get('filepath', ''))) for w in writes):
                    rej['writes_test_files'] += 1
                    continue
                orig = msgs[0]['content']
                if not KEEP_SYSTEM:
                    msgs[0] = {'role': 'system', 'content': SYSTEM_TEXT}
                text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=True)
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                n_tok = len(enc['input_ids'])
                if n_tok > MAX_TOKENS:
                    rej['too_many_tokens'] += 1
                    continue
                spans = trainable_spans(text)
                if TRUNC_TOKENS and n_tok > TRUNC_TOKENS:
                    cut_char = enc['offset_mapping'][TRUNC_TOKENS - 1][1]
                    keep_sp = [sp for sp in spans if sp[1] <= cut_char]
                    if not keep_sp:
                        rej['trunc_empty'] += 1
                        continue
                    text = text[:keep_sp[-1][1]]
                    spans = keep_sp
                    n_tok = len(tok(text, add_special_tokens=False)['input_ids'])
                    fixed['truncated'] += 1
                if THOUGHT_TURNS_ONLY:
                    thoughts['spans_all'] += len(spans)
                    spans = [sp for sp in spans if text.startswith('<|channel>thought', sp[0])]
                    thoughts['spans_kept'] += len(spans)
                    if not spans:
                        rej['no_thought_turns'] += 1
                        continue
                trained = ''.join(text[b:e] for b, e in spans)
                if CLEAN_MARKUP and TOOL_MARKUP_RX.search(trained):
                    rej['markup_left'] += 1
                    continue
                n_reason = sum(1 for m in msgs if m['role'] == 'assistant' and (m.get('reasoning') or '').strip())
                thoughts['with_reasoning'] += n_reason
                thoughts['rendered'] += text.count('<|channel>thought')
                thoughts['trainable_chars'] += sum(e - b for b, e in spans)
                out.write(json.dumps({'instance_id': r['instance_id'], 'source': f'{SOURCE}/{rdir.parent.name}', 'n_tokens': n_tok,
                                      'resolved': bool(r.get('resolved')), 'run': str(rdir),
                                      'tool_calls': r.get('tool_calls'), 'n_trainable_chars': sum(e - b for b, e in spans),
                                      'n_chars': len(text), 'text': text, 'spans': spans, 'messages': msgs,
                                      'orig_system': orig}) + '\n')
                tokens.append(n_tok)
                calls.append(r.get('tool_calls') or 0)
                fixed['resolved' if r.get('resolved') else 'unresolved_kept'] += 1
    stats = {'kept': len(tokens), 'rejects': dict(rej), 'fixed': dict(fixed), 'thoughts': dict(thoughts),
             'tokens': pct(tokens), 'tokens_total': sum(tokens),
             'tool_calls': pct(calls), 'runs': run_files, 'minutes': round((time.time() - T0) / 60, 1)}
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1))
    log('STATS ' + json.dumps(stats))
except Exception as e:  # noqa: BLE001
    log(f'FATAL {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}')
    (OUT / 'stats.json').write_text(json.dumps({'fatal': f'{type(e).__name__}: {e}'}, indent=1))
