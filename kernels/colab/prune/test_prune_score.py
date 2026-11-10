#!/usr/bin/env python3
"""CPU test of prune_score: runs every code cell of the notebook with a tiny random Gemma 4 (the real config's
structure, 14 layers, real tokenizer; no downloads, no upload) on synthetic agent rows, and checks the outputs.
usage: PRUNE_TOK_DIR=<dir with config.json + tokenizer files> python3 kernels/colab/prune/test_prune_score.py
"""
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
src = [json.loads(x) for x in open(HERE.parents[2] / 'runs/sft-convert-smoke-cpu/sft/sft_v1.jsonl')]
rng = random.Random(0)
rows = []
for k in range(40):  # 40 synthetic tasks: one in ten lands in the held-out split
    r = dict(rng.choice(src))
    r['instance_id'] = f'task_{k}'
    rows.append(r)
(tmp / 'sft.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
code = []
for cell in json.load(open(HERE / 'prune_score.ipynb'))['cells']:
    if cell['cell_type'] == 'code':
        code.append(''.join(cell['source']))
script = '\n\n'.join(code).replace('SMOKE = True', 'SMOKE = True', 1)
(tmp / 'nb.py').write_text(script)
env = dict(os.environ, PRUNE_TINY='1', PRUNE_ROWS=str(tmp / 'sft.jsonl'), PRUNE_OUT=str(tmp / "out"))
p = subprocess.run([sys.executable, str(tmp / 'nb.py')], env=env, capture_output=True, text=True, cwd=tmp)
print(p.stdout[-4000:])
assert p.returncode == 0, p.stderr[-4000:]
rep = json.load(open(tmp / 'out/report.json'))
names = [r['variant'] for r in rep['damage']]
assert names == ['full', 'ff0.7-agent', 'ff0.7-wiki', 'ff0.7-random', 'drop6-agent', 'drop6-measured'], names
assert len(rep['layer_sweep']['damage_by_layer']) == 12 and rep['layer_sweep']['own_tokens'] > 0
assert any(v != 0 for v in rep['layer_sweep']['damage_by_layer'].values())
assert rep['damage'][0]['agent_damage'] == 0 and all(r['agent_own_nats'] > 0 for r in rep['damage'])
assert any(r['agent_damage'] != 0 for r in rep['damage'][1:])  # the masks actually change the model
assert rep['data']['heldout_rows'] >= 1 and rep['data']['cal_own_tokens'] > 0
assert len(rep['compare']['block_influence']['agent']) == 14 and (tmp / 'out/scores.png').exists()
import numpy as np
z = np.load(tmp / 'out/scores.npz')
assert z['agent_ff'].shape == (14, 128) and z['wiki_head'].shape == (14, 4) and float(z['n_agent']) > 0

# the same notebook again, reusing the agent measurements of the run above: Wikipedia is measured again, the agent
# numbers of agent-chosen cuts are carried over, the full model's agent number must match
(tmp / 'nb_reuse.py').write_text(script.replace('REUSE_AGENT = False', 'REUSE_AGENT = True', 1))
env2 = dict(env, PRUNE_OUT=str(tmp / 'out2'), PRUNE_PREV=str(tmp / 'out'))
p = subprocess.run([sys.executable, str(tmp / 'nb_reuse.py')], env=env2, capture_output=True, text=True, cwd=tmp)
print(p.stdout[-2000:])
assert p.returncode == 0, p.stderr[-4000:]
rep2 = json.load(open(tmp / 'out2/report.json'))
assert rep2['reuse']['full_agent_now'] == rep['damage'][0]['agent_own_nats'], rep2['reuse']
d1 = {r['variant']: r for r in rep['damage']}
for r in rep2['damage']:
    assert r['agent_own_nats'] == d1[r['variant']]['agent_own_nats'], r      # same model, same data: identical
    assert r['wiki_nats'] == d1[r['variant']]['wiki_nats'], r
assert [r['agent_source'] for r in rep2['damage']] == ['measured (reuse check)', 'from score-v1-smoke', 'measured',
                                                       'from score-v1-smoke', 'from score-v1-smoke', 'measured']
z2 = np.load(tmp / 'out2/scores.npz')
assert np.array_equal(z2['agent_ff'], z['agent_ff']) and np.allclose(z2['wiki_ff'], z['wiki_ff'])
assert rep['data']['wiki_format'].startswith('article inside a chat turn')

# a third run reusing the second one's agent part and its layer sweep (no new sweep): every agent number of a cut
# chosen without Wikipedia is carried over, the sweep order is the same
s3 = script.replace('REUSE_AGENT = False', 'REUSE_AGENT = True', 1).replace(
    '    LAYER_SWEEP_TOKENS, LAYER_SWEEP_OWN = 3000, 50', '    LAYER_SWEEP_TOKENS, LAYER_SWEEP_OWN = 0, 50', 1)
assert s3.count('LAYER_SWEEP_TOKENS, LAYER_SWEEP_OWN = 0, 50') == 1
(tmp / 'nb_reuse3.py').write_text(s3)
env3 = dict(env, PRUNE_OUT=str(tmp / 'out3'), PRUNE_PREV=str(tmp / 'out2'))
p = subprocess.run([sys.executable, str(tmp / 'nb_reuse3.py')], env=env3, capture_output=True, text=True, cwd=tmp)
assert p.returncode == 0, p.stderr[-4000:]
rep3 = json.load(open(tmp / 'out3/report.json'))
assert rep3['layer_sweep']['reused_from'] == 'score-v1-smoke'
assert rep3['layer_sweep']['order_least_damage_first'] == rep['layer_sweep']['order_least_damage_first']
for r in rep3['damage']:
    assert r['agent_own_nats'] == d1[r['variant']]['agent_own_nats'] and r['wiki_nats'] == d1[r['variant']]['wiki_nats'], r
assert [r['agent_source'] for r in rep3['damage']] == ['measured (reuse check)', 'from score-v1-smoke', 'measured',
                                                       'from score-v1-smoke', 'from score-v1-smoke', 'from score-v1-smoke']
print('TEST OK', tmp)
