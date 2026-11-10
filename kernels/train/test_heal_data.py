"""CPU test of heal_data.py with a stand-in tokenizer: the held-out rows are exactly the ones prune_score.py takes (same
md5 split, same shuffles, first rows with agent-written tokens), labels per POSITIONS, rows cut at max_len, and the cut
mask equals the top-k of the scores (stable argsort). usage: python3 kernels/train/test_heal_data.py"""
import hashlib, json, os, random, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heal_data as H

class Tok:  # one token per 3 characters, offsets like a fast tokenizer
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offs = [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]
        ids = [sum(map(ord, text[a:b])) % 500 + 1 for a, b in offs]
        return {'input_ids': ids, 'offset_mapping': offs} if return_offsets_mapping else {'input_ids': ids}
fake = types.ModuleType('transformers'); fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: Tok())
sys.modules['transformers'] = fake
rng = random.Random(1)
rows = []
for i in range(120):
    n = rng.randint(30, 400)
    text = ''.join(rng.choice('abcdefgh ') for _ in range(n))
    a = rng.randint(0, n // 2); b = min(n, a + rng.randint(0, 60))
    rows.append({'instance_id': f'task{i % 60}__{i}', 'text': text, 'spans': [[a, b]] if b > a else []})
import tempfile
d = tempfile.mkdtemp()
open(f'{d}/sft.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
S = np.random.default_rng(0).random((6, 128))
np.savez(f'{d}/scores.npz', agent_ff=S)
rep = H.main({'sft': f'{d}/sft.jsonl', 'tokenizer': 'x', 'scores': f'{d}/scores.npz', 'out_dir': d, 'keep': 0.5,
              'multiple': 32, 'max_len': 64, 'eval_rows': 5, 'eval_len': 48, 'seed': 0, 'positions': 'all'})
# reference: prune_score.py's split/take logic, verbatim
held = lambda r: int(hashlib.md5(r['instance_id'].encode()).hexdigest(), 16) % 10 == 0
rr = random.Random(0); cal = [r for r in rows if not held(r)]; ho = [r for r in rows if held(r)]; rr.shuffle(cal); rr.shuffle(ho)
tok = Tok()
def enc(r, cap):
    e = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = e['input_ids'][:cap]; st = np.array([o[0] for o in e['offset_mapping'][:cap]])
    own = np.zeros(len(ids), bool)
    for s, t in r.get('spans') or []:
        own |= (st >= s) & (st < t)
    return ids, own
want = []
for r in ho:
    x = enc(r, 48)
    if x[1][1:].any():
        want.append((r['instance_id'], x))
    if len(want) == 5:
        break
ev = np.load(f'{d}/eval.npz'); tr = np.load(f'{d}/rows.npz'); mk = np.load(f'{d}/mask.npz')
assert list(ev['instance_ids']) == [w[0] for w in want], (list(ev['instance_ids']), [w[0] for w in want])
assert np.array_equal(ev['own'], np.concatenate([w[1][1] for w in want]))
assert np.array_equal(ev['ids'], np.concatenate([w[1][0] for w in want])) and np.array_equal(ev['labels'], ev['ids'])
assert rep['eval_own_tokens'] == sum(int(w[1][1].sum()) for w in want)
assert list(tr['sources']) == [r['instance_id'] for r in cal] and np.array_equal(tr['labels'], tr['ids'])
assert tr['lens'].max() <= 64 and rep['train_rows_cut_at_max_len'] == sum(1 for r in cal if len(r['text']) > 3 * 64)
assert mk['mask'].shape == (6, 128) and (mk['mask'].sum(1) == 64).all() and int(mk['keep']) == 64
assert all(set(np.nonzero(mk['mask'][i])[0]) == set(np.argsort(-S[i], kind='stable')[:64]) for i in range(6))
# own-only labels
H.main({'sft': f'{d}/sft.jsonl', 'tokenizer': 'x', 'scores': f'{d}/scores.npz', 'out_dir': d, 'keep': 0.5,
        'multiple': 32, 'max_len': 64, 'eval_rows': 5, 'eval_len': 48, 'seed': 0, 'positions': 'own'})
tr2 = np.load(f'{d}/rows.npz')
assert ((tr2['labels'] == -100) | (tr2['labels'] == tr2['ids'])).all() and (tr2['labels'] == -100).any()
# calls-only labels (B192, B207): every tool call in the row, from its <|tool_call> opener through <tool_call|> and the
# <|tool_response> marker, in any turn; rows without a call are dropped; eval.npz carries the calls mask in every mode
CALL1 = '<|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|><|tool_response>'
CALL2 = '<|tool_call>call:run_command{command:<|"|>ls<|"|>}<tool_call|>'
rows3 = []
for i in range(80):
    pre = ''.join(rng.choice('abcdefgh ') for _ in range(rng.randint(30, 150)))
    text = pre if i % 4 == 0 else pre + CALL1 + 'zz' * 10 + CALL2 + 'q' * 9
    rows3.append({'instance_id': f'c{i % 40}__{i}', 'text': text, 'spans': [[0, 6]]})
open(f'{d}/sft3.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rows3))
rep3 = H.main({'sft': f'{d}/sft3.jsonl', 'tokenizer': 'x', 'scores': f'{d}/scores.npz', 'out_dir': d, 'keep': 0.5,
               'multiple': 32, 'max_len': 400, 'eval_rows': 4, 'eval_len': 400, 'seed': 0, 'positions': 'calls'})
tr3, ev3 = np.load(f'{d}/rows.npz'), np.load(f'{d}/eval.npz')
rr = random.Random(0); cal3 = [r for r in rows3 if not held(r)]; ho3 = [r for r in rows3 if held(r)]; rr.shuffle(cal3); rr.shuffle(ho3)
def calls_mask(r, cap):
    e = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    st = np.array([o[0] for o in e['offset_mapping'][:cap]])
    m = np.zeros(len(st), bool)
    for c in (CALL1, CALL2):
        k = r['text'].find(c)
        if k >= 0:
            m |= (st >= k) & (st < k + len(c))
    return np.asarray(e['input_ids'][:cap]), m
keep3 = [r for r in cal3 if calls_mask(r, 400)[1][1:].any()]
assert list(tr3['sources']) == [r['instance_id'] for r in keep3]
assert rep3['train_rows_without_calls_dropped'] == len(cal3) - len(keep3) > 0
want_lab = np.concatenate([np.where(m, x, -100) for x, m in (calls_mask(r, 400) for r in keep3)])
assert np.array_equal(tr3['labels'], want_lab) and (tr3['labels'] != -100).any()
assert H.call_spans(CALL1 + CALL2) == [(0, len(CALL1)), (len(CALL1), len(CALL1 + CALL2))]
eo = np.concatenate([[0], np.cumsum(ev3['lens'])])
for j, iid in enumerate(ev3['instance_ids']):
    r = next(x for x in rows3 if x['instance_id'] == iid)
    assert np.array_equal(ev3['calls'][eo[j]:eo[j + 1]], calls_mask(r, 400)[1])
assert rep3['eval_calls_tokens'] == int(ev3['calls'].sum()) > 0
print('HEAL_DATA OK', rep)
