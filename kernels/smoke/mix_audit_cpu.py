"""Which training lines a ~100 tok/s, length-capped run actually sees (CPU, no internet).

Reads mix/train.jsonl of the mounted mix kernel (one line per segment / summary request, in training order). For each
length cap (8k, 12k, 16k, 20k tokens): lines kept, by kind and source family, trainable share; and how far into the
ordered lines a token budget reaches (1.5M = G2b's 240 min at ~102 tok/s, 2.6M = G3's 420 min), with the mix of what
falls inside that prefix. Note: the trainer shuffles within an epoch, so the prefix is a proxy for "what a short run
samples"; the per-cap totals are exact.
Also lists every repository (repos.json) and flags names that match a competition repository (forks, renamed orgs).
Outputs: /kaggle/working/audit/{audit.json, repos.json}.
"""
import collections
import glob
import json
from pathlib import Path

OUT = Path('/kaggle/working/audit')
OUT.mkdir(parents=True, exist_ok=True)
MIX_GLOB = '/kaggle/input/**/mix/train.jsonl'  # @define
CAPS = [8192, 12288, 16384, 20480]
BUDGETS = [1_500_000, 2_600_000]
f = sorted(glob.glob(MIX_GLOB, recursive=True))[0]
rows, repos = [], collections.Counter()
for line in open(f):
    r = json.loads(line)
    repos[r.get('repo', '')] += 1
    rows.append({'kind': r.get('kind', 'full'), 'source': r['source'], 'fam': r['source'].split('-')[0],
                 'n': r['n_tokens'], 't': r['n_trainable'], 'compactions': r.get('compactions', 0), 'part': r.get('part', 0),
                 'ref': (r['source'], r.get('ref') or r.get('instance_id'))})
# segments per conversation: with SUMMARY_KEEP < 1 a compacted conversation can have no summary line (compactions 0)
n_seg = collections.Counter(r['ref'] for r in rows if r['kind'] == 'segment')
rep = {'file': f, 'lines': len(rows), 'tokens': sum(r['n'] for r in rows), 'trainable': sum(r['t'] for r in rows), 'caps': {}}
for cap in CAPS:
    kept = [r for r in rows if r['n'] <= cap]
    c = {'lines': len(kept), 'tokens': sum(r['n'] for r in kept), 'trainable': sum(r['t'] for r in kept),
         'by_kind': dict(collections.Counter(r['kind'] for r in kept)),
         'by_family': dict(collections.Counter(r['fam'] for r in kept)),
         'post_compaction_segments': sum(1 for r in kept if r['kind'] == 'segment' and r['part'] > 0),
         'pre_compaction_segments_of_compacted': sum(1 for r in kept if r['kind'] == 'segment' and r['part'] == 0 and n_seg[r['ref']] > 1),
         'budgets': {}}
    c['trainable_share'] = round(c['trainable'] / max(1, c['tokens']), 3)
    c['summary_line_share'] = round(c['by_kind'].get('summary', 0) / max(1, c['lines']), 3)
    for b in BUDGETS:
        acc, inside = 0, []
        for r in kept:
            if acc + r['n'] > b:
                break
            acc += r['n']
            inside.append(r)
        c['budgets'][b] = {'lines': len(inside), 'trainable': sum(r['t'] for r in inside),
                           'by_kind': dict(collections.Counter(r['kind'] for r in inside)),
                           'by_family': dict(collections.Counter(r['fam'] for r in inside))}
    rep['caps'][cap] = c
# contamination check: repositories whose name matches a competition repository (forks, renamed orgs)
COMP = ('fastapi', 'rich', 'requests', 'httpx', 'starlette')
rep['repos'] = len(repos)
rep['repo_name_matches'] = {k: v for k, v in repos.items() if k.split('/')[-1].lower() in COMP or any(c in k.lower() for c in ('fastapi', 'starlette', 'httpx'))}
vf = sorted(glob.glob(MIX_GLOB.replace('train.jsonl', 'val.jsonl'), recursive=True))
if vf:
    vrepos = collections.Counter(json.loads(x).get('repo', '') for x in open(vf[0]))
    rep['val_repo_name_matches'] = {k: v for k, v in vrepos.items() if k.split('/')[-1].lower() in COMP}
(OUT / 'repos.json').write_text(json.dumps(dict(repos.most_common()), indent=1))
(OUT / 'audit.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep)[:3000])
