"""Paired comparison of arms from different GPU eval sessions on the tasks they share (B177: the full model in session A
vs the cut model and the healed cut in session B; same tasks, same settings, same load). Per arm: solved, timeouts,
mean tool calls and task seconds on the common tasks; every pair of arms: discordant counts and an exact McNemar test.
usage: python3 kernels/analysis/eval_cross.py name=path/to/task_results.jsonl name2=... [...]
"""
import collections
import itertools
import json
import math
import statistics as stt
import sys


def load(p):
    return {r['instance_id']: r for r in (json.loads(x) for x in open(p) if x.strip())}


def mcnemar(b, c):  # exact two-sided
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def arm_stats(d):  # as eval_paired.py
    rs = list(d.values())
    to = sum(1 for r in rs if (r.get('error') or '').startswith('Agent exceeded session timeout'))
    errs = collections.Counter((r.get('error') or '')[:60] for r in rs if r.get('error') and not r['error'].startswith('Agent exceeded'))
    dur = [r['duration_seconds'] for r in rs]
    return {'finished': len(rs), 'resolved': sum(r['resolved'] for r in rs), 'timeouts': to,
            'task_s_mean_median': [round(stt.mean(dur)), round(stt.median(dur))] if dur else None,
            'tool_calls_mean': round(stt.mean(r['tool_calls'] for r in rs), 1) if rs else None,
            'other_errors': dict(errs)}


arms = dict(a.split('=', 1) for a in sys.argv[1:])
R = {n: load(p) for n, p in arms.items()}
common = sorted(set.intersection(*(set(d) for d in R.values())))
out = {'tasks_common': len(common), 'tasks_per_arm': {n: len(d) for n, d in R.items()},
       'arms': {n: arm_stats({i: d[i] for i in common}) for n, d in R.items()}, 'pairs': {}}
for a, b in itertools.combinations(R, 2):
    a_only = [i for i in common if R[a][i]['resolved'] and not R[b][i]['resolved']]
    b_only = [i for i in common if R[b][i]['resolved'] and not R[a][i]['resolved']]
    out['pairs'][f'{a} vs {b}'] = {f'{a}_only': len(a_only), f'{b}_only': len(b_only),
                                   'mcnemar_exact_p': round(mcnemar(len(a_only), len(b_only)), 3),
                                   f'{a}_only_ids': a_only, f'{b}_only_ids': b_only}
print(json.dumps(out, indent=1))
