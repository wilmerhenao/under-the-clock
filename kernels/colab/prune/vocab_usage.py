"""How much of Gemma 4's 262,144-token vocabulary does our agent write? (pruner, the output layer; B175: the tied
output layer is stored in 16 bits, 2.8 GB = 12% of the bytes read per decoded token, untouched by the channel cuts.)

CPU only. For the agent rows of g4dev-g11-data, split by task exactly as the pruner (md5(instance_id) % 10 == 0 held
out): counts of every token the agent wrote (inside the rows' `spans`) and of every token; the coverage of the k most
frequent agent-written tokens of the training split, measured on the held-out split's agent-written tokens (does a
vocabulary chosen on some tasks cover new tasks?); the special tokens the harness format needs. If a cut output layer
kept the top k rows, decoding would read (262,144 - k) x 5,376 x 2 bytes less per token.
Outputs /kaggle/working/vocab/report.json (+ top-500 token strings for inspection).
"""
import collections
import glob
import hashlib
import json
import os
import time

import numpy as np
from transformers import AutoTokenizer

T0 = time.time()
OUT = '/kaggle/working/vocab'
os.makedirs(OUT, exist_ok=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return hits[0] if hits else None


sft = find('g4dev-g11-data/sft.jsonl') or find('**/sft.jsonl')
model_dir = os.path.dirname(find('gemma-4-31b-it-qat-w4a16-ct/*/tokenizer.json') or find('**/tokenizer.json'))
tok = AutoTokenizer.from_pretrained(model_dir)
V = len(tok)
rows = [json.loads(x) for x in open(sft) if x.strip()]
held = lambda r: int(hashlib.md5(r['instance_id'].encode()).hexdigest(), 16) % 10 == 0  # noqa: E731
cnt = {(split, kind): collections.Counter() for split in ('train', 'held') for kind in ('own', 'all')}
n_rows = collections.Counter()
for r in rows:
    split = 'held' if held(r) else 'train'
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc['input_ids'])
    starts = np.array([o[0] for o in enc['offset_mapping']])
    own = np.zeros(len(ids), bool)
    for s, e in r.get('spans') or []:
        own |= (starts >= s) & (starts < e)
    cnt[(split, 'all')].update(ids.tolist())
    cnt[(split, 'own')].update(ids[own].tolist())
    n_rows[split] += 1


def coverage(ranked, target, ks):
    tot = sum(target.values())
    out = {}
    for k in ks:
        keep = set(ranked[:k])
        out[str(k)] = round(sum(c for t, c in target.items() if t in keep) / max(tot, 1), 6)
    return out


KS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
rep = {'vocab': V, 'rows': dict(n_rows), 'tokens': {f'{a}_{b}': sum(c.values()) for (a, b), c in cnt.items()},
       'distinct': {f'{a}_{b}': len(c) for (a, b), c in cnt.items()}}
for kind in ('own', 'all'):
    ranked = [t for t, _ in cnt[('train', kind)].most_common()]
    rep[f'coverage_{kind}_rank_by_train_{kind}'] = {
        'train': coverage(ranked, cnt[('train', kind)], KS),
        'held': coverage(ranked, cnt[('held', kind)], KS)}
# the realistic choice: rank by everything the agent saw or wrote on the training tasks, cover the held-out writing
ranked_all = [t for t, _ in cnt[('train', 'all')].most_common()]
rep['coverage_held_own_rank_by_train_all'] = coverage(ranked_all, cnt[('held', 'own')], KS)
special = [t for t in tok.all_special_ids] + [tok.convert_tokens_to_ids(x) for x in
                                              ('<|tool_call>', '<tool_call|>', '<|tool_response>', '<tool_response|>',
                                               '<|channel>', '<channel|>', '<|turn>', '<turn|>', '<|"|>')]
rep['format_tokens_in_train_own_top_4096'] = {tok.convert_ids_to_tokens(int(t)): (int(t) in set(
    [x for x, _ in cnt[('train', 'own')].most_common(4096)])) for t in special if t is not None and t >= 0}
held_own_unseen = [t for t in cnt[('held', 'own')] if t not in cnt[('train', 'all')]]
rep['held_own_tokens_never_seen_in_train'] = {'distinct': len(held_own_unseen),
                                              'count': int(sum(cnt[('held', 'own')][t] for t in held_own_unseen)),
                                              'examples': [tok.convert_ids_to_tokens(int(t)) for t in held_own_unseen[:40]]}
bytes_per_row = 5376 * 2
rep['output_layer_gb'] = {str(k): round(k * bytes_per_row / 1e9, 3) for k in KS + [V]}
rep['minutes'] = round((time.time() - T0) / 60, 1)
json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1)
json.dump([[tok.convert_ids_to_tokens(int(t)), c] for t, c in cnt[('train', 'own')].most_common(500)],
          open(f'{OUT}/top500_own.json', 'w'), ensure_ascii=False)
print(json.dumps({k: v for k, v in rep.items() if k != 'format_tokens_in_train_own_top_4096'}, indent=1)[:6000])
