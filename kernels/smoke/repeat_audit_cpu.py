"""Do the training rows repeat identical tool calls? (CPU, no internet)

G2b's adapter looped (2026-10-07, BITACORA B28): in dev pass 1, a median 20% of its calls repeated an earlier identical
call (base 4%), and 31 of 70 runs repeated one call >= 5 times (base 0 of 53). Question: is that in the data (teachers that
re-issue the same call) or an SFT degeneration? Decision: filter looping rows, or look at training itself.
For every segment line of each mounted mix: the tool calls (`<|tool_call>call:NAME{ARGS}<tool_call|>`), the share that
repeat an earlier identical call in the same line, and lines with one call >= 3 and >= 5 times; per source.
v3: stale repeats (the same call with no edit_file / write_file since its previous occurrence): v2 showed the
teachers' repeats are mostly re-runs of a repro script after edits. v2: also up to 6 examples per source of the most-repeated call in a line (is R2E's 47% of lines with a call >= 3 times
a converter artifact, e.g. distinct views collapsed into one identical read?). Outputs: /kaggle/working/repeat/audit.json.
"""
import collections
import glob
import json
import re
from pathlib import Path

OUT = Path('/kaggle/working/repeat')
OUT.mkdir(parents=True, exist_ok=True)
MIX_GLOB = '/kaggle/input/**/mix/train.jsonl'  # @define
CALL = re.compile(r'<\|tool_call>(.*?)<tool_call\|>', re.S)

rep = {}
for f in sorted(glob.glob(MIX_GLOB, recursive=True)):
    name = f.split('/')[-3]
    per = collections.defaultdict(lambda: {'lines': 0, 'calls': 0, 'repeats': 0, 'max3': 0, 'max5': 0, 'stale': 0, 'stale3': 0, 'stale5': 0})
    examples = collections.defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        if r.get('kind', 'segment') != 'segment':
            continue
        calls = [c.strip() for c in CALL.findall(r['text'])]
        if not calls:
            continue
        c = collections.Counter(calls)
        d = per[r['source']]
        d['lines'] += 1
        d['calls'] += len(calls)
        d['repeats'] += sum(v - 1 for v in c.values())
        last_edit, seen, chain = 0, {}, collections.Counter()
        for i, call in enumerate(calls):
            if call.startswith(('call:edit_file', 'call:write_file')):
                last_edit = i + 1
                continue
            if call in seen and seen[call] >= last_edit:
                d['stale'] += 1
                chain[call] += 1
            seen[call] = i
        longest = max(chain.values()) + 1 if chain else 1
        d['stale3'] += longest >= 3
        d['stale5'] += longest >= 5
        top_call, top = c.most_common(1)[0]
        if top >= 3 and len(examples[r['source']]) < 6:
            examples[r['source']].append({'instance_id': r.get('instance_id'), 'times': top, 'call': top_call[:240]})
        d['max3'] += top >= 3
        d['max5'] += top >= 5
    rep[name] = {src: {**d, 'repeat_share': round(d['repeats'] / max(1, d['calls']), 3),
                       'lines_max3_share': round(d['max3'] / max(1, d['lines']), 3),
                       'lines_max5_share': round(d['max5'] / max(1, d['lines']), 3),
                       'stale_share': round(d['stale'] / max(1, d['calls']), 3), 'lines_stale3_share': round(d['stale3'] / max(1, d['lines']), 3),
                       'lines_stale5_share': round(d['stale5'] / max(1, d['lines']), 3), 'examples': examples[src]}
                 for src, d in sorted(per.items())}
    print(name, json.dumps(rep[name])[:2500], flush=True)
(OUT / 'audit.json').write_text(json.dumps(rep, indent=1))
