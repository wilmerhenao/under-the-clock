#!/usr/bin/env python3
"""Build the task pool for the daily AI-quota on-policy runs (private dataset `g4dev-daily-pool`).

usage: daily_pool.py ROWS_DIR OUT_DIR
  ROWS_DIR: bank outputs with rebench/tasks_verified.jsonl (bank parts 1-5), harness-bank harness_trusted_ids.json
            and the Qwen TPU task_results.jsonl (kout.py downloads; see docs/status_2026-10-12.md).
  OUT_DIR:  dataset dir: pool.jsonl (task rows, priority order), eval_harness.py, bundle/ (sub-v9c),
            dataset-metadata.json. Also writes private/daily/pool_meta.json (ids, tiers, repos; kept in Git).

Eligible = harness-trusted (gold resolves and the empty patch fails through the official Evaluator: rebench-harness
v3, harness-bank, reverify4) or solved by Qwen through the Evaluator (gen-tpu-rb2/rb3), minus every task Gemma has
already attempted on the AI quota (runs/onpolicy-*), minus the public set's repos (dev/held-out stay untouched).
Priority: A = trusted and Qwen-solved, B = Qwen-solved only, C = trusted only; repos round-robin inside a tier.
Tier R (retry candidates, after the fresh tiers) = tasks Gemma attempted on the AI quota and never resolved.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rows_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = {}
for p in sorted(glob.glob(f'{rows_dir}/**/rebench/tasks_verified.jsonl', recursive=True)) + \
        sorted(glob.glob(str(ROOT / 'runs/rebench-bank[5-9]-cpu/rebench/tasks_verified.jsonl'))):  # parts 5+ kept in runs/
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            rows.setdefault(r['instance_id'], r)
trusted = set()
for p in [ROOT / 'runs/rebench-harness-cpu-v3/rebench_harness/harness_trusted_ids.json',
          *sorted(glob.glob(str(ROOT / 'runs/rebench-reverify*-cpu/rebench_harness/harness_trusted_ids.json'))),
          *glob.glob(f'{rows_dir}/g4dev-rebench-harness-bank-cpu/**/harness_trusted_ids.json', recursive=True)]:
    trusted |= set(json.load(open(p)))
qwen = set()
for p in glob.glob(f'{rows_dir}/g4dev-gen-tpu-rb*/**/pass0/results/task_results.jsonl', recursive=True):
    qwen |= {json.loads(x)['instance_id'] for x in open(p) if x.strip() and json.loads(x).get('resolved')}
gemma, gemma_solved = set(), set()
for p in glob.glob(str(ROOT / 'runs/onpolicy-*/**/task_results.jsonl'), recursive=True):
    for x in open(p):
        if x.strip():
            r = json.loads(x)
            gemma.add(r['instance_id'])
            if r.get('resolved'):
                gemma_solved.add(r['instance_id'])
public_repos = {json.loads(x)['repo'] for x in open(ROOT / 'data/tasks.jsonl') if x.strip()}

elig = sorted(i for i in (trusted | qwen) - gemma if i in rows and rows[i]['repo'] not in public_repos)
tier = {i: 'A' if i in trusted and i in qwen else 'B' if i in qwen else 'C' for i in elig}
retry = sorted(i for i in gemma - gemma_solved if i in rows and rows[i]['repo'] not in public_repos)
tier.update({i: 'R' for i in retry})
elig += retry
order = []
for t in 'ABCR':
    by_repo = {}
    for i in elig:
        if tier[i] == t:
            by_repo.setdefault(rows[i]['repo'], []).append(i)
    queues = [by_repo[r] for r in sorted(by_repo)]
    while any(queues):  # round-robin over repos: a day's slice covers many repos
        for q in queues:
            if q:
                order.append(q.pop(0))

out.mkdir(parents=True, exist_ok=True)
with open(out / 'pool.jsonl', 'w') as f:
    f.writelines(json.dumps(rows[i]) + '\n' for i in order)
shutil.copy(ROOT / 'kernels/smoke/eval_harness.py', out / 'eval_harness.py')
shutil.copytree(ROOT / 'kernels/bundles/sub-v9c', out / 'bundle', dirs_exist_ok=True)
(out / 'dataset-metadata.json').write_text(json.dumps(
    {'title': 'g4dev-daily-pool', 'id': 'verracodeguacas/g4dev-daily-pool', 'licenses': [{'name': 'other'}]}))
meta = {'order': order, 'tier': tier, 'repo': {i: rows[i]['repo'] for i in order},
        'counts': {t: sum(1 for i in order if tier[i] == t) for t in 'ABCR'},
        'excluded': {'gemma_attempted': len(gemma), 'public_repos': sorted(public_repos)}}
(ROOT / 'private/daily').mkdir(parents=True, exist_ok=True)
(ROOT / 'private/daily/pool_meta.json').write_text(json.dumps(meta, indent=0))
print(json.dumps({'pool': len(order), 'counts': meta['counts'], 'repos': len(set(meta['repo'].values())),
                  'rows_known': len(rows), 'trusted': len(trusted), 'qwen_solved': len(qwen),
                  'gemma_attempted': len(gemma)}))
