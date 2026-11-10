#!/usr/bin/env python3
"""Base vs adapter on the same tasks, from a train_gpu.py run (G2b, G3, ...): paired per-task wins and losses.

Reads <dir>/<set>_p<k>_{base,adapter}/results/task_results.jsonl (fetch with
`kout.py <kernel> <dest> 'train_eval/(report\\.json|train/train_log\\.jsonl|[^/]+/summary_smoke\\.json|[^/]+/results/task_results\\.jsonl)$'`)
and <dir>/report.json. Per pass and pooled: tasks finished by both arms, solved by each, adapter-only wins (c),
base-only wins (b), and McNemar's exact two-sided p on the discordant tasks. Also prints the training summary
(probe, validation loss, eval gate). The passes differ by order seed only; time-boxed passes may finish different
task subsets per arm, so only tasks finished by both arms are compared.
usage: paired_eval.py runs/train-g2b-gpu/train_eval [--json out.json]
"""
import glob
import json
import math
import os
import re
import sys

d = sys.argv[1]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[r['instance_id']] = bool(r.get('resolved'))
    return rows


out = {'passes': {}}
rep = json.load(open(os.path.join(d, 'report.json'))) if os.path.exists(os.path.join(d, 'report.json')) else {}
for s in rep.get('stages', []):
    if s.get('stage') in ('train_done', 'eval_gate', 'canary_done', 'stopped', 'fatal'):
        out.setdefault('stages', []).append({k: v for k, v in s.items() if k not in ('first', 'last', 'tb')})
tl = os.path.join(d, 'train', 'train_log.jsonl')
if os.path.exists(tl):
    ev = [json.loads(x) for x in open(tl) if x.startswith('{')]
    out['probe'] = [{k: e.get(k) for k in ('tokens', 'ok', 'peak_gb', 'training')} for e in ev if e.get('event') == 'probe']
    out['probe_gate'] = [e.get('verdict') for e in ev if e.get('event') == 'probe_gate']
    out['val'] = {e.get('when'): e.get('loss_tok') for e in ev if e.get('event') == 'val'}
    ex = [e for e in ev if e.get('event') == 'example']
    out['examples'] = len(ex)
    out['steps'] = max([e.get('step', 0) for e in ev if e.get('event') == 'step'] or [0])
    out['tok_per_s'] = ex[-1].get('tok_per_s') if ex else None
    out['oom_skips'] = sum(1 for e in ev if e.get('event') == 'oom_skip')
    out['nan'] = sum(1 for e in ev if e.get('event') == 'nan_loss')
pooled_b = pooled_c = pooled_n = pooled_base = pooled_ad = 0
for base_dir in sorted(glob.glob(os.path.join(d, '*_p*_base'))):
    name = re.sub(r'_base$', '', os.path.basename(base_dir))
    base = load(os.path.join(base_dir, 'results', 'task_results.jsonl'))
    ad = load(os.path.join(d, f'{name}_adapter', 'results', 'task_results.jsonl'))
    common = sorted(set(base) & set(ad))
    b = sum(1 for i in common if base[i] and not ad[i])
    c = sum(1 for i in common if ad[i] and not base[i])
    sb, sa = sum(base[i] for i in common), sum(ad[i] for i in common)
    out['passes'][name] = {'finished_base': len(base), 'finished_adapter': len(ad), 'common': len(common),
                           'solved_base': sb, 'solved_adapter': sa, 'adapter_only': c, 'base_only': b,
                           'mcnemar_p': round(mcnemar_p(b, c), 4),
                           'adapter_only_ids': [i for i in common if ad[i] and not base[i]],
                           'base_only_ids': [i for i in common if base[i] and not ad[i]]}
    pooled_b, pooled_c, pooled_n = pooled_b + b, pooled_c + c, pooled_n + len(common)
    pooled_base, pooled_ad = pooled_base + sb, pooled_ad + sa
out['pooled'] = {'task_runs': pooled_n, 'solved_base': pooled_base, 'solved_adapter': pooled_ad,
                 'adapter_only': pooled_c, 'base_only': pooled_b, 'mcnemar_p': round(mcnemar_p(pooled_b, pooled_c), 4)}
print(json.dumps(out, indent=1))
if '--json' in sys.argv:
    open(sys.argv[sys.argv.index('--json') + 1], 'w').write(json.dumps(out, indent=1) + '\n')
