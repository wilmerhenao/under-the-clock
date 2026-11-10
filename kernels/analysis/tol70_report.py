"""The tolerant-parser run (B209) against same4 on the same 56 tasks, as pre-registered.

usage: python3 kernels/analysis/tol70_report.py <tol70 run dir> [<tol70 traces dir> <same4 block-0 traces root>] [out.json]
  run dir: runs/eval-rg-tol70-gpu (task results, tolerant_rewrites_c70tol.jsonl); traces (not committed): the c70tol
  segment's traces/*.json and a directory holding same4's block-0 traces per model (c70/, full/, full2/).
Primary: c70tol vs same4's c70, solved / discordant / exact McNemar (two-sided); recovery counts only if c70tol >= c70 + 4
with discordant tasks at least 3:1; also the share of the gap to the full runs' mean closed. Mechanism (descriptive):
calls the plugin rewrote over all calls, malformed calls left in the traces, endings (clock / call cap / own stop), and
loop sessions (one call 3+ times in a row, empty re-submits apart)."""
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_next_action import ALL_PARAMS, PARAMS  # noqa: E402
import re  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SAME4 = ROOT / 'runs/eval-rg-same4-gpu/train_eval'


def results(path):
    out = {}
    for f in glob.glob(str(path)):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r['resolved'] = r['resolved'] in (True, 'True')
                r['tool_calls'] = int(r['tool_calls'])
                r['duration_seconds'] = float(r['duration_seconds'])
                out[r['instance_id']] = r
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


timeout = lambda r: 'timeout' in (r.get('error') or '').lower()  # noqa: E731
cap = lambda r: not timeout(r) and r['tool_calls'] >= 24 and 'turns budget' not in (r.get('error') or '')  # noqa: E731


def endings(R):
    t = [r for r in R.values() if timeout(r)]
    c = [r for r in R.values() if cap(r)]
    o = [r for r in R.values() if not timeout(r) and not cap(r)]
    return {'n': len(R), 'solved': sum(r['resolved'] for r in R.values()), 'clock': [len(t), sum(r['resolved'] for r in t)],
            'call_cap': [len(c), sum(r['resolved'] for r in c)], 'own_stop': [len(o), sum(r['resolved'] for r in o)],
            'tool_calls': sum(r['tool_calls'] for r in R.values())}


def swallowed(name, args):
    keys = PARAMS.get(name) or ALL_PARAMS
    for k, v in (args or {}).items():
        if isinstance(v, str) and any(re.search(r'[`"\']\s*,\s*' + re.escape(p) + r'\s*:', v) for p in keys if p != k):
            return True
    return False


def trace_stats(tdir):
    calls = bad = sessions = loops = real = 0
    for f in sorted(glob.glob(f'{tdir}/*.json')):
        d = json.load(open(f))
        sessions += 1
        last, run, best, best_key = None, 0, 0, None
        for s in (x for x in d.get('steps', []) if x.get('source') == 'agent'):
            for c in (s.get('tool_calls') or []):
                calls += 1
                bad += swallowed(c['function_name'], c.get('arguments'))
                key = c['function_name'] + json.dumps(c.get('arguments') or {}, sort_keys=True)
                run = run + 1 if key == last else 1
                if run > best:
                    best, best_key = run, key
                last = key
        if best >= 3:
            loops += 1
            real += best_key != 'submit_patch{}'
    return {'sessions': sessions, 'tool_calls': calls, 'malformed': bad, 'share': round(bad / max(1, calls), 4),
            'loop_sessions': loops, 'real_loop_sessions': real}


def main(run_dir, tdir=None, same4_tr=None, out=None):
    run_dir = Path(run_dir)
    tol = results(run_dir / 'train_eval/repograph_c70tol_b0/results/task_results.jsonl')
    ref = {m: results(SAME4 / f'repograph_{m}_b0/results/task_results.jsonl') for m in ('c70', 'full', 'full2', 'c70kd')}
    ids = sorted(set(tol) & set(ref['c70']))
    rep = {'tasks_compared': len(ids), 'endings': {'c70tol': endings({i: tol[i] for i in ids})}}
    for m, R in ref.items():
        rep['endings'][m] = endings({i: R[i] for i in ids})
        b = sum(tol[i]['resolved'] and not R[i]['resolved'] for i in ids)
        c = sum(R[i]['resolved'] and not tol[i]['resolved'] for i in ids)
        rep[f'c70tol_vs_{m}'] = {'c70tol': sum(tol[i]['resolved'] for i in ids), m: sum(R[i]['resolved'] for i in ids),
                                 'only_c70tol': b, f'only_{m}': c, 'mcnemar_exact_p': round(mcnemar(b, c), 4)}
    p = rep['c70tol_vs_c70']
    full_mean = (rep['endings']['full']['solved'] + rep['endings']['full2']['solved']) / 2
    gap = full_mean - p['c70']
    rep['primary'] = {'difference': p['c70tol'] - p['c70'], 'discordant': [p['only_c70tol'], p['only_c70']],
                      'p': p['mcnemar_exact_p'],
                      'recovery_by_rule': p['c70tol'] - p['c70'] >= 4 and p['only_c70tol'] >= 3 * max(1, p['only_c70']),
                      'gap_closed': round((p['c70tol'] - p['c70']) / gap, 2) if gap else None, 'full_mean': full_mean}
    rw = run_dir / 'train_eval/tolerant_rewrites_c70tol.jsonl'
    if rw.exists():
        ev = [json.loads(x) for x in rw.read_text().splitlines() if x.strip()]
        n = sum('tool' in e for e in ev)
        def kind(b):   # which slip the rewrite repaired (first match)
            if '`,' in b or b.rstrip().endswith('`'):
                return 'backtick'
            if '\\",' in b:
                return 'escaped quote'
            if re.search(r'\w"\s*:', b):
                return 'JSON-style key or quote'
            return 'other quote'
        rep['rewrites'] = {'registered_markers': sum(e.get('event') == 'registered' for e in ev), 'rewritten_calls': n,
                           'share_of_calls': round(n / max(1, rep['endings']['c70tol']['tool_calls']), 4),
                           'by_kind': {k: sum(kind(e['before']) == k for e in ev if 'tool' in e)
                                       for k in ('backtick', 'JSON-style key or quote', 'escaped quote', 'other quote')},
                           'by_tool': {t: sum(e.get('tool') == t for e in ev) for t in sorted({e.get('tool') for e in ev
                                                                                                if 'tool' in e})}}
    if tdir:
        rep['traces'] = {'c70tol': trace_stats(tdir)}
        if 'rewrites' in rep:
            rep['rewrites']['share_of_trace_calls'] = round(rep['rewrites']['rewritten_calls'] /
                                                            max(1, rep['traces']['c70tol']['tool_calls']), 4)
        for m in ('c70', 'full', 'full2'):
            if same4_tr and Path(same4_tr, m).exists():
                rep['traces'][m] = trace_stats(Path(same4_tr, m))
    if out:
        json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 2 else None, a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else (a[1] if len(a) == 2 else None))
