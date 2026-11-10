#!/usr/bin/env python3
"""DPO training rows (B127): pass/fail pairs on one task, tagged for the TPU trainer's DPO mode, plus optional SFT rows.

Input: a g11_data.py output (sft.jsonl) built from a PAIR_MODE mix (render KEEP_UNRESOLVED, train_mix PAIR_MODE): each
line is one segment of a run with `resolved` true/false. Only chain 0 is used (chain 1 repeats the post-compaction
segments with another summary; a pair compares two runs, not two summaries). A task becomes a pair when it has
segments from both sides of a pair; its lines get `source` 'dpo:<instance_id>~<kind>:c' or ':r' (kind 'fail': resolved
vs unresolved run; 'speed': the faster vs the slower resolved run, B130).
--sft files are added unchanged as SFT rows (weight 1 in the trainer), minus exact duplicates of chosen lines.
pairs.json: {pair id: {kind, c: cost, r: cost}} with each side's run cost (calls, turns, generated tokens, est_s) for the
trainer's clock-calibrated IPO (B154).
--speed-clock-agree drops speed pairs whose chosen run (fewer calls) is not also faster by estimated time.
usage: python3 kernels/analysis/dpo_pairs.py <pairs sft.jsonl> <out dir> --slug <dataset slug> [--sft <sft.jsonl> ...]
       [--speed-clock-agree]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('pairs')
ap.add_argument('out')
ap.add_argument('--slug', required=True)
ap.add_argument('--sft', action='append', default=[])
ap.add_argument('--speed-clock-agree', action='store_true',
                help='keep a speed pair only if its chosen run is also faster by estimated serving time (B155)')
a = ap.parse_args()

by_task = collections.defaultdict(lambda: {'c': [], 'r': []})
st = collections.Counter()
for line in open(a.pairs):
    r = json.loads(line)
    if r.get('chain') not in (None, 0):
        st['chain1_skipped'] += 1
        continue
    if 'pair_side' in r:  # PAIR_MODE mix (fail and/or speed pairs, B130)
        by_task[f"{r['instance_id']}~{r['pair_kind']}"][r['pair_side']].append(r)
    elif 'resolved' in r:
        by_task[f"{r['instance_id']}~fail"]['c' if r['resolved'] else 'r'].append(r)
    else:
        raise SystemExit(f"{r['instance_id']}: no pair fields (render with KEEP_UNRESOLVED, mix with PAIR_MODE)")

out_rows, chosen_hash, meta = [], set(), {}
for iid, sides in sorted(by_task.items()):
    if not sides['c'] or not sides['r']:
        st['one_sided_tasks'] += 1
        continue
    kind = iid.rsplit('~', 1)[1]
    cost_c, cost_r = sides['c'][0].get('pair_cost'), sides['r'][0].get('pair_cost')
    if kind == 'speed' and a.speed_clock_agree and not (cost_c and cost_r and cost_c['est_s'] < cost_r['est_s']):
        # fewer tool calls but more generated tokens: the clock model says the chosen run was not faster
        st['speed_clock_disagrees'] += 1
        continue
    st['pairs'] += 1
    st['pairs_' + kind] += 1
    # run-level cost of each side (train_mix_cpu.run_cost, B154): the trainer's clock-calibrated IPO target
    meta[iid] = {'kind': kind, 'c': cost_c, 'r': cost_r}
    for side in 'cr':
        for r in sides[side]:
            r['source'] = f'dpo:{iid}:{side}'
            out_rows.append(r)
            st[f'lines_{side}'] += 1
            st[f'tokens_{side}'] += r['n_tokens']
            if side == 'c':
                chosen_hash.add(hashlib.md5(r['text'].encode()).hexdigest())
for f in a.sft:
    for line in open(f):
        r = json.loads(line)
        if hashlib.md5(r['text'].encode()).hexdigest() in chosen_hash:
            st['sft_dup_of_chosen'] += 1
            continue
        out_rows.append(r)
        st['sft_lines'] += 1

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / 'sft.jsonl', 'w') as fo:
    for r in out_rows:
        fo.write(json.dumps(r) + '\n')
(out / 'stats.json').write_text(json.dumps(dict(st), indent=1))
(out / 'pairs.json').write_text(json.dumps(meta, indent=0))
(out / 'README.md').write_text(
    '# DPO pairs (private)\n\nPass/fail pairs of Gemma 4 31B agent runs on the same task, segmented as the competition '
    "harness shows them around context compaction. `source` 'dpo:<task>~fail:c|r' = resolved vs unresolved run, "
    "'dpo:<task>~speed:c|r' = faster vs slower resolved run; other lines are plain SFT rows. Built by "
    'kernels/analysis/dpo_pairs.py.\n\n'
    + json.dumps(dict(st), indent=1) + '\n')
(out / 'dataset-metadata.json').write_text(json.dumps(
    {'title': a.slug, 'id': f'verracodeguacas/{a.slug}', 'licenses': [{'name': 'other'}]}, indent=1))
print(json.dumps(dict(st), indent=1))
