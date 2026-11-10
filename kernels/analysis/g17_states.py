#!/usr/bin/env python3
"""G17 teacher prompts (B157): decision points from Gemma's own runs, shown to a stronger model (Gemini) as a readable
transcript, with Gemma's actual next step to rate and a request for the step the teacher would take instead.

Why on-policy states: G2b (B26-B31) imitated teacher trajectories, i.e. the teacher's states and its narration-style
thoughts, and Gemma got worse. Here every state is one Gemma actually reached (its own context, compactions included);
the teacher only labels the next step. Rows built from the answers (g17_build.py) are trained on that single step.

Input: a g10_assemble output (sft.jsonl, one rendered segment per line with `spans` = the model's trained steps, and the
pair fields of the G16 mix: pair_kind fail|speed, pair_side c|r, resolved). States are sampled mostly from the slow or
failed side ('r'), weighted toward steps that usually waste time: long thoughts, repeated calls, still exploring late,
full test-suite runs. The context (everything before the step) is kept as rendered text for training (`prefix`), and
converted to a compact transcript for the teacher.
usage: python3 kernels/analysis/g17_states.py <assembled sft.jsonl> <out dir> [--n 1000] [--seed 0]
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

STEP = re.compile(r'(?:<\|channel>thought\n(?P<thought>.*?)<channel\|>)?<\|tool_call>call:(?P<name>[a-z_]+)\{(?P<args>.*?)\}'
                  r'<tool_call\|><\|tool_response>(?:response:(?P<rname>[a-z_]+)\{value:<\|"\|>(?P<resp>.*?)<\|"\|>\}'
                  r'<tool_response\|>)?', re.S)
SYSTEM = re.compile(r'<\|turn>system\n(?:<\|think\|>\n)?(?P<sys>.*?)(?:<\|tool>|<turn\|>)', re.S)
USER = re.compile(r'<\|turn>user\n(?P<u>.*?)<turn\|>', re.S)
STR = '<|"|>'
MAX_PREFIX_CHARS = 60000      # the rendered context of a state (~16k tokens); longer contexts are not sampled
THOUGHT_SHOW = 1500           # characters of each earlier thought shown to the teacher
RESULT_SHOW = 4000            # characters of each tool result shown (read_file: 6000)
TRANSCRIPT_MAX = 42000        # older results are shortened until the transcript fits
LONG_THOUGHT = 1500
PER_RUN = 3


def parse_args(body):
    """Gemma's rendered call arguments (keys in alphabetical order, strings between <|"|> markers) -> dict."""
    out, i = {}, 0
    while i < len(body):
        j = body.index(':', i)
        key = body[i:j]
        i = j + 1
        if body.startswith(STR, i):
            k = body.index(STR, i + len(STR))
            out[key] = body[i + len(STR):k]
            i = k + len(STR)
        else:
            k = body.find(',', i)
            k = len(body) if k < 0 else k
            raw = body[i:k]
            out[key] = True if raw == 'true' else False if raw == 'false' else int(raw) if re.fullmatch(r'-?\d+', raw) else raw
            i = k
        if i < len(body) and body[i] == ',':
            i += 1
    return out


def render_args(args):
    parts = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, bool):
            parts.append(f'{k}:{"true" if v else "false"}')
        elif isinstance(v, int):
            parts.append(f'{k}:{v}')
        else:
            parts.append(f'{k}:{STR}{v}{STR}')
    return ','.join(parts)


def show_call(name, args):
    return f'{name}(' + ', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in args.items()) + ')'


def show_result(resp, name):
    if resp is None:
        return '(no result)'
    lim = 6000 if name == 'read_file' else RESULT_SHOW
    try:
        d = json.loads(resp)
    except ValueError:
        return resp[:lim]
    if not isinstance(d, dict):
        return json.dumps(d)[:lim]
    parts = []
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 80:
            parts.append(f'{k}:\n{v}')
        else:
            parts.append(f'{k}: {json.dumps(v, ensure_ascii=False)}')
    s = '\n'.join(parts)
    return s if len(s) <= lim else s[:lim // 2] + f'\n[... {len(s) - lim} characters omitted ...]\n' + s[-lim // 2:]


def transcript(prefix):
    """Readable transcript of a rendered context: rules, task (or compaction summary), then the steps so far."""
    sm = SYSTEM.search(prefix)
    rules = sm.group('sys').strip() if sm else ''
    users = [m.group('u').strip() for m in USER.finditer(prefix)]
    steps = [m for m in STEP.finditer(prefix)]
    blocks = []
    for n, m in enumerate(steps, 1):
        args = parse_args(m.group('args'))
        th = (m.group('thought') or '').strip()
        if len(th) > THOUGHT_SHOW:
            th = th[:THOUGHT_SHOW] + f' [... {len(th) - THOUGHT_SHOW} more characters of reasoning]'
        blocks.append([f'--- Step {n} ---\n' + (f'AGENT THOUGHT: {th}\n' if th else '') +
                       f'TOOL CALL: {show_call(m.group("name"), args)}\n', show_result(m.group('resp'), m.group('name'))])
    def size():
        return sum(len(a) + len(b) for a, b in blocks)
    k = 0
    while size() > TRANSCRIPT_MAX and k < max(0, len(blocks) - 4):  # shorten the oldest results first, keep the last 4
        a, b = blocks[k]
        if len(b) > 400:
            blocks[k][1] = b[:300] + f'\n[... older result shortened, {len(b) - 300} characters omitted ...]'
        k += 1
    body = ''.join(a + 'RESULT:\n' + b + '\n' for a, b in blocks)
    return rules, '\n\n'.join(users), body, len(steps)


def features(text, spans):
    """Per trained step: index, thought chars, repeated call, edits so far, full-suite test run."""
    out, seen, edits = [], set(), 0
    for k, (s, e) in enumerate(spans):
        m = STEP.match(text, s)
        if not m or '<|tool_call>' in m.group('args'):  # parallel calls (4 of 15,669 in the G16 rows): skipped
            out.append(None)
            continue
        call = m.group('name') + '{' + m.group('args') + '}'
        cmd = parse_args(m.group('args')).get('command', '') if m.group('name') == 'run_command' else ''
        before_edits = edits
        if m.group('name') in ('edit_file', 'write_file'):
            edits += 1
        prior = len(STEP.findall(text[:s]))
        out.append({'k': k, 'call_index': prior + 1, 'thought': len(m.group('thought') or ''), 'name': m.group('name'),
                    'repeat': call in seen or bool(re.search(re.escape(call), text[:s])),
                    'edits_before': before_edits + len(re.findall(r'call:(?:edit_file|write_file)\{', text[:s])),
                    'suite': bool(re.search(r'(^|[;&|]\s*)(python3? -m )?pytest\s*($|-q|-x|--|\.|;|&|\|)', cmd))})
        seen.add(call)
    return out


def weight(f):
    w = 1.0
    if f['thought'] > LONG_THOUGHT:
        w += 2
    if f['repeat']:
        w += 3
    if f['edits_before'] == 0 and f['call_index'] >= 8:
        w += 2
    if f['suite']:
        w += 2
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sft')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--c-share', type=float, default=0.15)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cands = collections.defaultdict(list)  # run key -> [(weight, state)]
    st = collections.Counter()
    for line in open(a.sft):
        r = json.loads(line)
        if r.get('chain') not in (None, 0) or not r.get('pair_side'):
            continue
        run = f"{r['instance_id']}|{r['ref']}"
        for f, (s, e) in zip(features(r['text'], r['spans']), r['spans']):
            if f is None:
                st['unparsed_step'] += 1
                continue
            if s > MAX_PREFIX_CHARS:
                st['prefix_too_long'] += 1
                continue
            cands[(r['pair_side'], run)].append((weight(f), {
                'instance_id': r['instance_id'], 'repo': r['repo'], 'ref': r['ref'], 'part': r['part'],
                'pair_kind': r.get('pair_kind'), 'pair_side': r['pair_side'], 'resolved': r.get('resolved'),
                'run_calls': r.get('calls'), 'feat': f, 'prefix': r['text'][:s], 'step': r['text'][s:e]}))
    runs = {side: [k for k in cands if k[0] == side] for side in 'cr'}
    want = {'r': round(a.n * (1 - a.c_share)), 'c': round(a.n * a.c_share)}
    picked = []
    for side in 'cr':
        keys = runs[side][:]
        rng.shuffle(keys)
        per_run = {k: sorted(cands[k], key=lambda x: -x[0] * rng.random()) for k in keys}
        while len([p for p in picked if p['pair_side'] == side]) < want[side] and any(per_run.values()):
            for k in keys:
                if per_run[k] and sum(1 for p in picked if p.get('_run') == k) < PER_RUN:
                    w, s = per_run[k].pop(0)
                    s['_run'] = k
                    picked.append(s)
                    if len([p for p in picked if p['pair_side'] == side]) >= want[side]:
                        break
                else:
                    per_run[k] = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_tok = []
    with open(out / 'states.jsonl', 'w') as fo:
        for s in picked:
            s.pop('_run', None)
            rules, task, body, n_steps = transcript(s['prefix'])
            m = STEP.match(s['step'])
            nxt = {'thought': (m.group('thought') or '').strip(), 'name': m.group('name'), 'args': parse_args(m.group('args'))}
            sid = hashlib.md5((s['ref'] + str(s['part']) + str(len(s['prefix']))).encode()).hexdigest()[:16]
            s.update(id=sid, rules=rules, task=task, transcript=body, n_steps=n_steps, next=nxt)
            n_tok.append((len(rules) + len(task) + len(body)) / 3.6)
            fo.write(json.dumps(s) + '\n')
            st['states_' + s['pair_side']] += 1
            st['states_' + s['feat']['name']] += 1
            st['long_thought'] += s['feat']['thought'] > LONG_THOUGHT
            st['repeat'] += s['feat']['repeat']
            st['late_no_edit'] += s['feat']['edits_before'] == 0 and s['feat']['call_index'] >= 8
            st['suite'] += s['feat']['suite']
    n_tok.sort()
    st['transcript_tokens_p50'] = int(n_tok[len(n_tok) // 2]) if n_tok else 0
    st['transcript_tokens_p90'] = int(n_tok[int(len(n_tok) * 0.9)]) if n_tok else 0
    st['runs_c'], st['runs_r'] = len(runs['c']), len(runs['r'])
    (out / 'states_stats.json').write_text(json.dumps(dict(st), indent=1))
    print(json.dumps(dict(st), indent=1))


if __name__ == '__main__':
    main()
