"""Paired comparison of the two arms of a GPU eval (task_results.jsonl per arm), as B159/B165: solved, timeouts, discordant
pairs with an exact McNemar test; optional cross-check of the base arm against an earlier run's task_results.jsonl.
usage: python3 kernels/analysis/eval_paired.py <runs/eval-... dir> <name of base arm> <name of adapter arm> [earlier base task_results.jsonl]"""
import collections, json, math, statistics as stt, sys
from pathlib import Path

def load(p):
    return {r['instance_id']: r for r in (json.loads(x) for x in open(p) if x.strip())}

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def arm_stats(d):
    rs = list(d.values())
    to = sum(1 for r in rs if (r.get('error') or '').startswith('Agent exceeded session timeout'))
    errs = collections.Counter((r.get('error') or '')[:60] for r in rs if r.get('error') and not r['error'].startswith('Agent exceeded'))
    dur = [r['duration_seconds'] for r in rs]
    return {'finished': len(rs), 'resolved': sum(r['resolved'] for r in rs), 'timeouts': to,
            'task_s_mean_median': [round(stt.mean(dur)), round(stt.median(dur))] if dur else None,
            'tool_calls_mean': round(stt.mean(r['tool_calls'] for r in rs), 1) if rs else None,
            'other_errors': dict(errs)}

root = Path(sys.argv[1])
names = {'base': sys.argv[2], 'adapter': sys.argv[3]}
A = load(root / 'train_eval/repograph_p0_base/results/task_results.jsonl')
B = load(root / 'train_eval/repograph_p0_adapter/results/task_results.jsonl')
common = sorted(set(A) & set(B))
a_only = [i for i in common if A[i]['resolved'] and not B[i]['resolved']]
b_only = [i for i in common if B[i]['resolved'] and not A[i]['resolved']]
out = {names['base']: arm_stats(A), names['adapter']: arm_stats(B),
       'paired': {'tasks': len(common), 'resolved_' + names['base']: sum(A[i]['resolved'] for i in common),
                  'resolved_' + names['adapter']: sum(B[i]['resolved'] for i in common),
                  names['base'] + '_only': len(a_only), names['adapter'] + '_only': len(b_only),
                  'mcnemar_exact_p': round(mcnemar(len(a_only), len(b_only)), 3),
                  names['base'] + '_only_ids': a_only, names['adapter'] + '_only_ids': b_only}}
if len(sys.argv) > 4:  # cross-check: the same base arm in an earlier run
    C = load(sys.argv[4])
    cc = sorted(set(A) & set(C))
    out['crosscheck_base_vs_earlier_run'] = {'tasks': len(cc), 'resolved_now': sum(A[i]['resolved'] for i in cc),
                                            'resolved_earlier': sum(C[i]['resolved'] for i in cc),
                                            'agree': sum(A[i]['resolved'] == C[i]['resolved'] for i in cc),
                                            'now_only': sum(A[i]['resolved'] and not C[i]['resolved'] for i in cc),
                                            'earlier_only': sum(C[i]['resolved'] and not A[i]['resolved'] for i in cc)}
print(json.dumps(out, indent=1))
