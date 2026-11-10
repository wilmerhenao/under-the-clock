#!/usr/bin/env python3
"""G17 training rows from the teacher's answers (B157): step-level preference pairs at Gemma's own states.

For every state the teacher rated slow, wasteful or wrong, two rows share the state's rendered context (`prefix`, exactly
what Gemma saw) and differ only in the final step:
  chosen   = the teacher's step rendered in Gemma's own format: <|channel>thought\\n{plan}<channel|><|tool_call>call:...
  rejected = Gemma's actual step (its thought and call, as it generated them).
Only that final step is trained (`spans`), thought included (`train_on: all` on these rows): the decision lives in the
plan, so a call trained without its plan would only teach the model to follow whatever plan it wrote. Sources
'dpo:<state>~teacher:c|r' make them DPO pairs in the TPU trainer; pairs.json carries each step's clock cost
(16.7 s + 0.139 s x tokens, B142) for the record.
A teacher step is dropped if its call is invalid: unknown tool or arguments, an edit whose old_string the agent never saw
(read results, earlier edits, command output), a test file edited, a bare pytest, or a plan outside 80-1,200 characters.
usage: python3 kernels/analysis/g17_build.py <states.jsonl> <out dir> --teacher <teacher.jsonl> [...] --slug g4dev-g17-data
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g17_states import STEP, parse_args, render_args  # noqa: E402

TOOLS = {'run_command': ({'command'}, {'command'}), 'read_file': ({'filepath'}, {'filepath', 'start_line', 'end_line'}),
         'edit_file': ({'filepath', 'old_string', 'new_string'}, {'filepath', 'old_string', 'new_string', 'allow_multiple'}),
         'write_file': ({'filepath', 'content'}, {'filepath', 'content'}), 'submit_patch': (set(), set()),
         'get_status': (set(), set())}
INTS = {'start_line', 'end_line'}
TEST_FILE = re.compile(r'(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)')
BARE_PYTEST = re.compile(r'(^|[;&|]\s*)(python3? -m )?pytest\s*($|-q\s*$|-x\s*$|[;&|])')
LAT_FIXED_S, LAT_PER_TOKEN_S, CHARS_PER_TOKEN = 16.7, 0.139, 3.6


def visible(prefix):
    """Everything the agent could have seen: decoded tool results and its own earlier arguments."""
    out = []
    for m in STEP.finditer(prefix):
        args = parse_args(m.group('args'))
        out += [v for v in args.values() if isinstance(v, str)]
        if m.group('resp'):
            try:
                d = json.loads(m.group('resp'))
            except ValueError:
                out.append(m.group('resp'))
                continue
            out += [v for v in (d.values() if isinstance(d, dict) else [d]) if isinstance(v, str)]
    return '\n'.join(out)


def check(ans, prefix):
    tool, args, plan = ans.get('tool'), ans.get('args'), (ans.get('plan') or '').strip()
    if tool not in TOOLS:
        return None, 'bad_tool'
    args = dict(args or {})
    need, allowed = TOOLS[tool]
    if not need <= set(args) or not set(args) <= allowed:
        return None, 'bad_args'
    for k in list(args):
        if k in INTS:
            try:
                args[k] = int(args[k])
            except (TypeError, ValueError):
                return None, 'bad_int'
        elif k == 'allow_multiple':
            args[k] = bool(args[k])
        elif not isinstance(args[k], str):
            return None, 'bad_type'
    if not 80 <= len(plan) <= 1200:
        return None, 'plan_length'
    if re.search(r'\bthe agent\b|\bthe teacher\b', plan, re.I):
        return None, 'plan_voice'
    if tool in ('edit_file', 'write_file') and TEST_FILE.search(args['filepath']):
        return None, 'test_file'
    if tool == 'edit_file':
        if not args['old_string'] or args['old_string'] == args['new_string']:
            return None, 'edit_noop'
        if args['old_string'] not in visible(prefix):
            return None, 'edit_unseen'
    if tool == 'run_command' and (not args['command'].strip() or BARE_PYTEST.search(args['command'].strip())):
        return None, 'bare_pytest'
    if any('<|' in v or '|>' in v for v in args.values() if isinstance(v, str)) or '<|' in plan:
        return None, 'special_tokens'
    return (tool, args, plan), None


def cost(step_text):
    tok = len(step_text) / CHARS_PER_TOKEN
    return {'chars': len(step_text), 'est_tokens': int(tok), 'est_s': round(LAT_FIXED_S + LAT_PER_TOKEN_S * tok, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('states')
    ap.add_argument('out')
    ap.add_argument('--teacher', action='append', required=True)
    ap.add_argument('--slug', required=True)
    ap.add_argument('--max-chars', type=int, default=60000)  # both rows of a pair must fit the 20,480-token bucket
    a = ap.parse_args()
    states = {}
    for line in open(a.states):
        s = json.loads(line)
        states[s['id']] = s
    answers = {}
    for f in a.teacher:  # later files win for the same state
        for line in open(f):
            t = json.loads(line)
            answers[t['id']] = t
    st, rows, meta = collections.Counter(), [], {}
    plan_len, rej_len = [], []
    for sid, t in sorted(answers.items()):
        s, ans = states.get(sid), t.get('answer') or {}
        if s is None:
            st['unknown_state'] += 1
            continue
        rating = ans.get('rating')
        st['rating_' + str(rating)] += 1
        st['side_' + s['pair_side'] + '_' + str(rating)] += 1
        if rating not in ('slow', 'wasteful', 'wrong'):
            continue
        ok, why = check(ans, s['prefix'])
        if not ok:
            st['dropped_' + why] += 1
            continue
        tool, args, plan = ok
        chosen = f'<|channel>thought\n{plan}<channel|><|tool_call>call:{tool}{{{render_args(args)}}}<tool_call|><|tool_response>'
        rejected = s['step']
        if chosen == rejected:
            st['dropped_identical'] += 1
            continue
        if len(s['prefix']) + max(len(chosen), len(rejected)) > a.max_chars:
            st['dropped_too_long'] += 1  # truncation would cut the trained final step
            continue
        m = STEP.match(rejected)
        same_call = bool(m) and m.group('name') == tool and parse_args(m.group('args')) == args
        st['same_call'] += same_call
        st['pairs'] += 1
        st['pairs_' + rating] += 1
        pid = f'{sid}~teacher'
        meta[pid] = {'kind': 'teacher', 'rating': rating, 'same_call': same_call, 'instance_id': s['instance_id'],
                     'c': cost(chosen), 'r': cost(rejected), 'teacher': t.get('model')}
        plan_len.append(len(plan))
        rej_len.append(len((m.group('thought') or '') if m else ''))
        for side, step in (('c', chosen), ('r', rejected)):
            text = s['prefix'] + step
            rows.append({'instance_id': s['instance_id'], 'repo': s['repo'], 'ref': s['ref'], 'part': s['part'],
                         'kind': 'segment', 'text': text, 'spans': [[len(s['prefix']), len(text)]], 'train_on': 'all',
                         'n_tokens': int(len(text) / CHARS_PER_TOKEN), 'source': f'dpo:{pid}:{side}', 'chain': 0,
                         'rating': rating})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sft.jsonl', 'w') as fo:
        for r in rows:
            fo.write(json.dumps(r) + '\n')
    (out / 'pairs.json').write_text(json.dumps(meta, indent=0))
    q = lambda x, p: sorted(x)[min(len(x) - 1, int(p * len(x)))] if x else None  # noqa: E731
    st['plan_chars_p10_50_90'] = [q(plan_len, p) for p in (0.1, 0.5, 0.9)]
    st['rejected_thought_chars_p10_50_90'] = [q(rej_len, p) for p in (0.1, 0.5, 0.9)]
    st['lines'] = len(rows)
    st['est_tokens'] = sum(r['n_tokens'] for r in rows)
    (out / 'stats.json').write_text(json.dumps(dict(st), indent=1))
    (out / 'README.md').write_text(
        '# G17 teacher step pairs (private)\n\nDecision points from Gemma 4 31B agent runs (its own rendered context), '
        "each with two continuations: the step a stronger model (Gemini, Kaggle model proxy) would take ('c') and the "
        "step Gemma took ('r'), where the teacher rated Gemma's step slow, wasteful or wrong. Only the final step is "
        'trained. Built by kernels/analysis/g17_build.py.\n\n' + json.dumps(dict(st), indent=1) + '\n')
    (out / 'dataset-metadata.json').write_text(json.dumps(
        {'title': a.slug, 'id': f'verracodeguacas/{a.slug}', 'licenses': [{'name': 'other'}]}, indent=1))
    print(json.dumps(dict(st), indent=1))


if __name__ == '__main__':
    main()
