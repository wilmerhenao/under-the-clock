"""Paired results of the unseen-repository sessions with and without the tasks of repos found in the training task pool
(B192: omni-us/jsonargparse, tophat/syrupy in private/daily/pool_meta.json). Per session: both arms' solved counts on
the common tasks, discordant pairs, exact McNemar; then the cut's cross-session comparisons (c70 vs every full-model-
level arm) the same way.
usage: python3 kernels/analysis/eval_leak_check.py [out.json]"""
import glob
import json
import math
import sys

LEAK = ('jsonargparse_', 'syrupy_')


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    return {r['instance_id']: bool(r['resolved']) for r in (json.loads(x) for x in open(p) if x.strip())}


def pair(A, B, drop):
    ids = sorted(i for i in set(A) & set(B) if not (drop and i.startswith(LEAK)))
    a = sum(A[i] and not B[i] for i in ids)
    b = sum(B[i] and not A[i] for i in ids)
    return {'tasks': len(ids), 'solved': [sum(A[i] for i in ids), sum(B[i] for i in ids)], 'only': [a, b],
            'p': round(mcnemar(a, b), 4), 'leak_tasks_in_set': sum(i.startswith(LEAK) for i in set(A) & set(B))}


out = {'sessions': {}, 'cut_cross': {}}
arms = {}
for d in sorted(glob.glob('runs/eval-rg-*')):
    f = {t: f'{d}/train_eval/repograph_p0_{t}/results/task_results.jsonl' for t in ('base', 'adapter')}
    try:
        A, B = load(f['base']), load(f['adapter'])
    except FileNotFoundError:
        continue
    name = d.split('runs/')[1]
    arms[name] = (A, B)
    out['sessions'][name] = {'with': pair(A, B, False), 'without': pair(A, B, True)}
c70 = arms.get('eval-rg-cut70-gpu', (None, None))[0]
if c70:
    refs = {'base@sessA': arms['eval-rg-base-g11a-gpu'][0], 'G11a@sessA': arms['eval-rg-base-g11a-gpu'][1]}
    for n, (A, B) in arms.items():
        if n in ('eval-rg-g11g13-gpu', 'eval-rg-g16a-gpu', 'eval-rg-g17-gpu', 'eval-rg-g17v2a-gpu', 'eval-rg-g17v2b-gpu'):
            refs[f'G11a@{n[8:-4]}'] = A   # G11a is the base arm of these sessions (B122, B159, B161, B163/B173)
    for n, R in refs.items():
        out['cut_cross'][f'{n} vs c70'] = {'with': pair(R, c70, False), 'without': pair(R, c70, True)}
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
for k, v in {**out['sessions'], **out['cut_cross']}.items():
    w, wo = v['with'], v['without']
    print(f"{k:42s} with: {w['solved'][0]:>3} vs {w['solved'][1]:>3} of {w['tasks']:>3} ({w['only'][0]}/{w['only'][1]}) p {w['p']:<6} "
          f"| without: {wo['solved'][0]:>3} vs {wo['solved'][1]:>3} of {wo['tasks']:>3} ({wo['only'][0]}/{wo['only'][1]}) p {wo['p']}")
