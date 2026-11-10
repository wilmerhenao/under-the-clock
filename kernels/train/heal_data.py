"""Rows, held-out rows and the channel cut for healing a cut Gemma 4 31B (pruner part 3, train_tpu_runner.py 'heal').

The split and the held-out rows are the pruner's own (kernels/colab/prune/prune_score.py): the agent rows of
g4dev-g11-data, one task in ten held out by md5(instance_id) % 10 == 0, both lists shuffled with random.Random(SEED),
held-out = the first EVAL_ROWS held-out rows with agent-written tokens in their first EVAL_LEN tokens. Agent-written
tokens ("own") = tokens starting inside the row's `spans`. The cut is the speed test's (prune_serve.py c70):
slice_model.make_plan(scores, KEEP, multiple=MULTIPLE, group=1) on the scores' `agent_ff`, i.e. per layer the
round(KEEP * I / MULTIPLE) * MULTIPLE hidden units with the highest score (stable argsort).

Outputs (np.savez):
- rows.npz: ids, labels (POSITIONS 'all': every token is a target; 'own': only agent-written tokens; 'calls': only the
  tool calls of every agent turn: the `<|tool_call>` opener (B207: serve-v6 showed cut models dropping it), the body
  through `<tool_call|>` and the `<|tool_response>` stop marker that follows; rows with no call in their first MAX_LEN tokens
  are dropped), lens, sources (instance ids): the format train_tpu_runner.py reads. Rows are cut to MAX_LEN tokens.
- eval.npz: ids, labels (every token), own (bool per token), calls (bool per token: inside a tool call, opener
  included, any POSITIONS), lens, instance_ids. The held-out rows do not depend on POSITIONS.
- mask.npz: mask (L, I) uint8, keep (per-layer kept count), sha (sha256 of the kept indices), score_key.
usage: python heal_data.py '<json args>'  (sft, tokenizer, scores, out_dir, keep, multiple, max_len, eval_rows, eval_len,
       seed, positions, score_key)
"""
import hashlib
import json
import random
import re
import sys

import numpy as np


def cut_mask(scores, keep, multiple=512, score_key='agent_ff'):
    S = np.asarray(scores[score_key], dtype=np.float64)
    L, I = S.shape
    multiple = min(multiple, I)
    k = int(keep) if isinstance(keep, (int, np.integer)) else int(round(keep * I / multiple)) * multiple
    k = max(multiple, min(I, k))
    mask = np.zeros((L, I), np.uint8)
    h = hashlib.sha256()
    for i in range(L):
        idx = np.sort(np.argsort(-S[i], kind='stable')[:k])
        mask[i, idx] = 1
        h.update(idx.astype(np.int64).tobytes())
    return mask, k, h.hexdigest()


def split_rows(rows, seed):
    held = lambda r: int(hashlib.md5(r['instance_id'].encode()).hexdigest(), 16) % 10 == 0  # noqa: E731
    rng = random.Random(seed)
    cal = [r for r in rows if not held(r)]
    ho = [r for r in rows if held(r)]
    rng.shuffle(cal)
    rng.shuffle(ho)
    return cal, ho


TOOL_CALL = re.compile(r'<\|tool_call>(.*?<tool_call\|>)(<\|tool_response>)?', re.S)   # train_lora.py's


def call_spans(text):
    """Character spans of every tool call in the row, from its `<|tool_call>` opener (B207; train_lora.action_spans
    starts after it) through `<tool_call|>` and the `<|tool_response>` marker, in any turn."""
    return [(m.start(), m.end()) for m in TOOL_CALL.finditer(text)]


def encode(tok, r, cap):
    """(ids, own, calls) of the first `cap` tokens and the row's full length (a token belongs to a span when it starts
    inside it)."""
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc['input_ids'][:cap], np.int32)
    starts = np.array([o[0] for o in enc['offset_mapping'][:cap]])
    own = np.zeros(len(ids), bool)
    for s, e in r.get('spans') or []:
        own |= (starts >= s) & (starts < e)
    calls = np.zeros(len(ids), bool)
    for s, e in call_spans(r['text']):
        calls |= (starts >= s) & (starts < e)
    return ids, own, calls, len(enc['input_ids'])


def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a['tokenizer'])
    rows = [json.loads(x) for x in open(a['sft']) if x.strip()]
    cal, ho = split_rows(rows, a.get('seed', 0))
    ev = []
    pos = a.get('positions', 'all')
    assert pos in ('all', 'own', 'calls'), pos
    for r in ho:   # the pruner's held-out rows (same choice for every POSITIONS)
        ids, own, calls, _ = encode(tok, r, a['eval_len'])
        if own[1:].any():
            ev.append((ids, own, r['instance_id'], calls))
        if len(ev) == a['eval_rows']:
            break
    out = a['out_dir'].rstrip('/')
    np.savez(f'{out}/eval.npz', ids=np.concatenate([x[0] for x in ev]), labels=np.concatenate([x[0] for x in ev]),
             own=np.concatenate([x[1] for x in ev]), calls=np.concatenate([x[3] for x in ev]),
             lens=np.array([len(x[0]) for x in ev], np.int32), instance_ids=np.array([x[2] for x in ev]))
    tr_ids, tr_lab, tr_len, tr_src, n_own, n_cut, skipped = [], [], [], [], 0, 0, 0
    n_calls, no_calls = 0, 0
    for r in cal:
        ids, own, calls, n_full = encode(tok, r, a['max_len'])
        if len(ids) < 2:
            skipped += 1
            continue
        if pos == 'calls' and not calls[1:].any():
            no_calls += 1
            continue
        n_cut += int(n_full > a['max_len'])
        lab = ids.copy() if pos == 'all' else np.where(own if pos == 'own' else calls, ids, -100).astype(np.int32)
        tr_ids.append(ids), tr_lab.append(lab), tr_len.append(len(ids)), tr_src.append(r['instance_id'])
        n_own += int(own.sum())
        n_calls += int(calls.sum())
    np.savez(f'{out}/rows.npz', ids=np.concatenate(tr_ids), labels=np.concatenate(tr_lab),
             lens=np.array(tr_len, np.int32), sources=np.array(tr_src))
    z = np.load(a['scores'])
    mask, k, sha = cut_mask(z, a['keep'], a.get('multiple', 512), a.get('score_key', 'agent_ff'))
    np.savez(f'{out}/mask.npz', mask=mask, keep=np.int32(k), sha=np.array(sha), score_key=np.array(a.get('score_key', 'agent_ff')))
    held_ids = {r['instance_id'] for r in ho}
    rep = {'rows_total': len(rows), 'train_rows': len(tr_len), 'train_tokens': int(sum(tr_len)), 'train_own_tokens': n_own,
           'train_rows_cut_at_max_len': n_cut, 'skipped': skipped, 'positions': pos,
           'train_calls_tokens': n_calls, 'train_rows_without_calls_dropped': no_calls,
           'eval_rows': len(ev), 'eval_tokens': int(sum(len(x[0]) for x in ev)),
           'eval_own_tokens': int(sum(int(x[1].sum()) for x in ev)),  # part 1's heldout_own_tokens counts the same way
           'eval_calls_tokens': int(sum(int(x[3].sum()) for x in ev)),
           'eval_tasks_in_train': len({x[2] for x in ev} & set(tr_src)), 'train_tasks_held': len(set(tr_src) & held_ids),
           'mask_keep_per_layer': k, 'mask_units': int(mask.shape[1]), 'mask_layers': int(mask.shape[0]), 'mask_sha': sha}
    json.dump(rep, open(f'{out}/heal_data.json', 'w'), indent=1)
    print(json.dumps(rep))
    assert rep['eval_tasks_in_train'] == 0 and rep['train_tasks_held'] == 0, rep
    return rep


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
