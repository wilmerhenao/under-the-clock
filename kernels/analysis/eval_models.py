"""Same-session comparison of checkpoints (B192, train_gpu.py MODELS mode), as pre-registered in B192: the analysis set is,
per block, the tasks every model run finished; per model its blocks merged (stats also over all its finished tasks);
the primary test, d = c70 - (full + full2)/2 per task with an exact two-sided sign-flip test (same for c70kd); every
pair of models on the analysis set (solved, discordant tasks, exact McNemar), also per block (the blocks ran in opposite
model orders, A B C D | D C B A: a model that wins only in one order is suspect); timeouts, task time, tool calls.
usage: python3 kernels/analysis/eval_models.py <runs/eval-... dir> [out.json]
reads <dir>/train_eval/<EVAL_SET>_<model>_b<k>/results/task_results.jsonl"""
import collections
import itertools
import json
import math
import re
import statistics as stt
import sys
from pathlib import Path


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    return {r['instance_id']: r for r in (json.loads(x) for x in open(p) if x.strip())}


def timeout(r):
    return (r.get('error') or '').startswith('Agent exceeded session')


def stats(rs):
    rs = list(rs)
    dur = [r['duration_seconds'] for r in rs if r.get('duration_seconds') is not None]
    errs = collections.Counter((r.get('error') or '')[:60] for r in rs if r.get('error') and not timeout(r))
    return {'finished': len(rs), 'resolved': sum(bool(r['resolved']) for r in rs), 'timeouts': sum(map(timeout, rs)),
            'task_s_mean_median': [round(stt.mean(dur)), round(stt.median(dur))] if dur else None,
            'tool_calls_mean': round(stt.mean(r.get('tool_calls') or 0 for r in rs), 1) if rs else None,
            'other_errors': dict(errs)}


def pair(A, B, na, nb):
    common = sorted(set(A) & set(B))
    a_only = [i for i in common if A[i]['resolved'] and not B[i]['resolved']]
    b_only = [i for i in common if B[i]['resolved'] and not A[i]['resolved']]
    return {'tasks': len(common), f'resolved_{na}': sum(bool(A[i]['resolved']) for i in common),
            f'resolved_{nb}': sum(bool(B[i]['resolved']) for i in common), f'{na}_only': len(a_only),
            f'{nb}_only': len(b_only), 'mcnemar_exact_p': round(mcnemar(len(a_only), len(b_only)), 4),
            f'timeouts_{na}': sum(timeout(A[i]) for i in common), f'timeouts_{nb}': sum(timeout(B[i]) for i in common),
            f'{na}_only_ids': a_only, f'{nb}_only_ids': b_only}


def signflip_p(d):
    """Exact two-sided sign-flip test of mean(d) = 0 for per-task differences d (multiples of 0.5): the distribution
    of sum(+-d_i) under random signs by convolution over the doubled integer values."""
    v = [int(round(2 * abs(x))) for x in d if x]
    if not v:
        return 1.0
    dist = {0: 1}
    for x in v:
        nd = collections.Counter()
        for s, c in dist.items():
            nd[s + x] += c
            nd[s - x] += c
        dist = nd
    obs = abs(int(round(2 * sum(d))))
    return min(1.0, sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(v))


def primary(runs, cut='c70', fulls=('full', 'full2')):
    """B192's pre-registered primary test: per block, the tasks every model run finished; d = cut - mean(full runs)."""
    if cut not in runs or not all(f in runs for f in fulls):
        return None
    blocks = sorted(set.intersection(*(set(runs[m]) for m in runs)))
    d, n = [], 0
    for b in blocks:
        ids = set.intersection(*(set(runs[m][b]) for m in runs))
        n += len(ids)
        for i in sorted(ids):
            d.append(float(bool(runs[cut][b][i]['resolved']))
                     - sum(bool(runs[f][b][i]['resolved']) for f in fulls) / len(fulls))
    return {'analysis_tasks': n, 'cut': cut, 'fulls': list(fulls), 'mean_d': round(sum(d) / max(1, len(d)), 4),
            'sum_d': sum(d), 'nonzero': sum(1 for x in d if x), 'signflip_exact_p': round(signflip_p(d), 4)}


root = Path(sys.argv[1])
runs = {}
for d in sorted((root / 'train_eval').glob('*_b[0-9]*')):
    m = re.match(r'(.+)_(.+)_b(\d+)$', d.name)
    f = d / 'results' / 'task_results.jsonl'
    if m and f.exists():
        runs.setdefault(m.group(2), {})[int(m.group(3))] = load(f)
models = sorted(runs)
# the pre-registered analysis set (B192): per block, the tasks every model run finished
blocks_all = sorted(set.intersection(*(set(runs[m]) for m in models))) if models else []
aset = {b: set.intersection(*(set(runs[m][b]) for m in models)) for b in blocks_all}
runs_a = {m: {b: {i: runs[m][b][i] for i in aset[b]} for b in blocks_all} for m in models}
merged = {m: {k: v for blk in runs[m].values() for k, v in blk.items()} for m in models}
merged_a = {m: {k: v for blk in runs_a[m].values() for k, v in blk.items()} for m in models}
out = {'models': {m: {'all': stats(merged[m].values()),
                      **{f'block{b}': stats(runs[m][b].values()) for b in sorted(runs[m])}} for m in models},
       'analysis_set': {f'block{b}': len(aset[b]) for b in blocks_all},
       'primary': {c: primary(runs_a, cut=c) for c in models if c not in ('full', 'full2')},   # every cut vs the full runs
       'pairs': {}}
for a, b in itertools.combinations(models, 2):
    out['pairs'][f'{a} vs {b}'] = {'all': pair(merged_a[a], merged_a[b], a, b),
                                   **{f'block{k}': pair(runs_a[a][k], runs_a[b][k], a, b) for k in blocks_all}}
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('analysis set (tasks every run finished):', out['analysis_set'])
for c, p in out['primary'].items():
    if p:
        print(f"PRIMARY {c} vs mean(full, full2) on {p['analysis_tasks']} tasks: mean d {p['mean_d']:+.4f} (sum {p['sum_d']:+.1f}, "
              f"{p['nonzero']} tasks differ), exact sign-flip p {p['signflip_exact_p']}")
for m in models:
    s = out['models'][m]['all']
    print(f"{m:>10}: finished {s['finished']}, solved {s['resolved']}, timeouts {s['timeouts']}, "
          f"task s {s['task_s_mean_median']}, calls {s['tool_calls_mean']}, other errors {sum(s['other_errors'].values())}")
for k, v in out['pairs'].items():
    a, b = k.split(' vs ')
    p = v['all']
    blocks = '; '.join(f"{bk}: {v[bk][f'resolved_{a}']} vs {v[bk][f'resolved_{b}']} ({v[bk][f'{a}_only']}/{v[bk][f'{b}_only']})"
                       for bk in v if bk.startswith('block'))
    print(f"{k}: {p[f'resolved_{a}']} vs {p[f'resolved_{b}']} of {p['tasks']} ({p[f'{a}_only']}/{p[f'{b}_only']}), "
          f"p {p['mcnemar_exact_p']}; timeouts {p[f'timeouts_{a}']} vs {p[f'timeouts_{b}']} | {blocks}")
