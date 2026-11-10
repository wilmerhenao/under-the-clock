#!/usr/bin/env python3
"""CPU test of score-v4 (B202) without any download: a synthetic word-level tokenizer with Gemma 4's chat and tool
markers, a 14-layer random Gemma 4 (the notebook's TINY mode) and synthetic agent rows with tool calls.
1. the standard notebook (prune_score.py, SMOKE) plays the earlier run (score-v3): it also checks that the ROLE_ONLY
   edits left the standard path working;
2. the built score-v4 kernel (kernels/prune-score4-gpu/prune_score.py, switched to SMOKE) runs against it: the plain
   scores measured again in the one-product pass must equal the earlier run's, every role must be seen, and the output
   must keep every earlier array unchanged.
usage: python3 kernels/colab/prune/build_kaggle.py --role && <python with torch + transformers 5.13.1> \
       kernels/colab/prune/test_prune_roles.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
tmp = Path(tempfile.mkdtemp())
sys.path.insert(0, str(HERE))
import synth_agent  # noqa: E402

rows = synth_agent.rows(60)
(tmp / 'sft.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
tok_dir = synth_agent.tokenizer_dir(rows, tmp / 'tok')

env = dict(os.environ, PRUNE_TINY='1', PRUNE_TOK_DIR=str(tok_dir), PRUNE_ROWS=str(tmp / 'sft.jsonl'))
# 1. the standard notebook, as the earlier run
p = subprocess.run([sys.executable, str(HERE / 'prune_score.py')], env=dict(env, PRUNE_OUT=str(tmp / 'v3')),
                   capture_output=True, text=True, cwd=tmp)
print(p.stdout[-1500:])
assert p.returncode == 0, p.stderr[-4000:]
rep3 = json.load(open(tmp / 'v3/report.json'))
assert [r['variant'] for r in rep3['damage']][:2] == ['full', 'ff0.7-agent'] and rep3['data']['wiki_cal_tokens'] > 0
assert any(r['agent_damage'] != 0 for r in rep3['damage'][1:])

# 2. the built score-v4 kernel (SMOKE) against it
kern = (ROOT / 'kernels/prune-score4-gpu/prune_score.py').read_text()
assert kern.count('SMOKE = False ') == 1 and "RUN_NAME = 'score-v4' " in kern and 'ROLE_ONLY = True ' in kern
(tmp / 'v4.py').write_text(kern.replace('SMOKE = False ', 'SMOKE = True ', 1))
p = subprocess.run([sys.executable, str(tmp / 'v4.py')], env=dict(env, PRUNE_OUT=str(tmp / 'v4'), PRUNE_PREV=str(tmp / 'v3')),
                   capture_output=True, text=True, cwd=tmp)
print(p.stdout[-3000:])
assert p.returncode == 0, p.stderr[-4000:]
rep4 = json.load(open(tmp / 'v4/report.json'))
ro = rep4['roles']
assert rep4['run'] == 'score-v4-smoke' and ro['same_examples_ok'], ro['same_examples']
assert ro['sanity']['ok'] and ro['sanity']['agent_ff_rank_corr_min'] > 0.99999, ro['sanity']
assert all(v > 0 for v in ro['positions_per_role'].values()), ro['positions_per_role']
assert ro['roles_used'] == ['role_read', 'role_thought', 'role_args', 'role_syntax', 'role_close']
assert sum(ro['counts'][f'role_{n}'] for n in ('read', 'thought', 'args', 'syntax', 'close')) == \
    ro['counts']['agent'] - rep4['data']['cal_rows']                  # every position but each example's last one
assert 'wiki_damage' not in json.dumps(rep4) and rep4['data']['wiki_cal_tokens'] == 0 and 'damage' not in rep4
z3, z4 = np.load(tmp / 'v3/scores.npz'), np.load(tmp / 'v4/scores.npz')
assert set(z3.files) < set(z4.files) and all(np.array_equal(z3[k], z4[k]) for k in z3.files)   # earlier arrays unchanged
assert np.allclose(z4['agent_ff_v4'], z3['agent_ff'], rtol=1e-4) and np.allclose(z4['agent_own_ff_v4'], z3['agent_own_ff'], rtol=1e-4)
assert z4['role_ff'].shape == z3['agent_ff'].shape == (14, 128) and np.isfinite(z4['role_ff']).all()
assert not np.allclose(np.argsort(z4['role_ff'], 1), np.argsort(z3['agent_ff'], 1))            # a different ranking
for g in ('read', 'thought', 'args', 'syntax', 'close'):
    assert z4[f'role_{g}_ff'].shape == (14, 128) and int(z4[f'n_role_{g}']) == ro['counts'][f'role_{g}']
k = ro['compare']['keep_0.55']
assert k['k'] == 64 and 0 < k['kept_overlap_agent_ff_vs_role_ff'] < 1 and set(k) >= {'role_close_top5pct_kept'}
assert rep4['layer_sweep']['reused_from'] == 'score-v1-smoke' and (tmp / 'v4/scores.png').exists()
print('ROLES TEST OK', tmp, ro['positions_per_role'], ro['sanity'])
