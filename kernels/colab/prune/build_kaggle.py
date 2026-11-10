#!/usr/bin/env python3
"""Kaggle script kernel of the pruner notebook: kernels/prune-score-gpu/ (4 x L4, private, no internet: the
competition's GPU machines refuse it; WikiText from a private dataset).
--rerun: kernels/prune-score2-gpu/ (run score-v2): reuses the agent measurements of g4dev-prune-score-gpu (attached
as a kernel source), measures Wikipedia again (with <bos>, B168) and runs the single-layer removal test.
--rerun3: kernels/prune-score3-gpu/ (run score-v3): reuses score-v2's agent part and layer sweep, Wikipedia as a chat turn.
--role: kernels/prune-score4-gpu/ (run score-v4, B202): the notebook up to its first run of every step, then
token_roles.py and prune_roles.py: the 300 agent examples again, counted per role of the predicted token (role_ff);
score-v3 attached (its agent scores are the check and stay in the output).
usage: python3 kernels/colab/prune/build_kaggle.py [--smoke] [--rerun | --rerun3 | --role]"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = (ROOT / 'kernels/colab/prune/prune_score.py').read_text()
smoke = '--smoke' in sys.argv
src, n = re.subn(r'^SMOKE = True ', f'SMOKE = {smoke} ', src, count=1, flags=re.M)
assert n == 1
rerun = '--rerun' in sys.argv      # score-v2 (B168): reuse v1's agent part, Wikipedia with <bos>, layer sweep
rerun3 = '--rerun3' in sys.argv    # score-v3 (B171): reuse v2's agent part and layer sweep, Wikipedia in chat format
role = '--role' in sys.argv        # score-v4 (B202): per-role measurement
assert rerun + rerun3 + role <= 1
subs = {'rerun': (("RUN_NAME = 'score-v1' ", "RUN_NAME = 'score-v2' "), ('REUSE_AGENT = False ', 'REUSE_AGENT = True '),
                  ('LAYER_SWEEP_TOKENS = 32_000 ', 'LAYER_SWEEP_TOKENS = 24_000 ')),
        'rerun3': (("RUN_NAME = 'score-v1' ", "RUN_NAME = 'score-v3' "), ('REUSE_AGENT = False ', 'REUSE_AGENT = True '),
                   ('LAYER_SWEEP_TOKENS = 32_000 ', 'LAYER_SWEEP_TOKENS = 0 '))}
if role:
    cut = src.index('# %% [markdown]\n# ## (Optional) Reusing the agent measurements')
    src = (src[:cut] + (ROOT / 'kernels/colab/prune/token_roles.py').read_text() + '\n\n'
           + (ROOT / 'kernels/colab/prune/prune_roles.py').read_text())
subs['role'] = (("RUN_NAME = 'score-v1' ", "RUN_NAME = 'score-v4' "), ('ROLE_ONLY = False ', 'ROLE_ONLY = True '))
for mode, flag in (('rerun', rerun), ('rerun3', rerun3), ('role', role)):
    for old, new in (subs[mode] if flag else ()):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
gpu_image = re.search(r"^GPU_IMAGE = '([^']+)'", (ROOT / 'kernels/smoke/build.py').read_text(), re.M).group(1)
slug = ('g4dev-prune-score2-gpu' if rerun else 'g4dev-prune-score3-gpu' if rerun3 else 'g4dev-prune-score4-gpu' if role
        else 'g4dev-prune-score-gpu')
out = ROOT / 'kernels' / slug.replace('g4dev-', '')
out.mkdir(exist_ok=True)
(out / 'prune_score.py').write_text(src)
meta = {'id': f'verracodeguacas/{slug}', 'title': slug, 'code_file': 'prune_score.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaL4',
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse', 'verracodeguacas/g4dev-g11-data',
                            *([] if role else ['verracodeguacas/g4dev-wikitext103-eval']),
                            'verracodeguacas/g4dev-train-wheels'],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': (['verracodeguacas/g4dev-prune-score-gpu'] if rerun else
                           ['verracodeguacas/g4dev-prune-score2-gpu'] if rerun3 else
                           ['verracodeguacas/g4dev-prune-score3-gpu'] if role else []),
        'model_sources': ['google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2'], 'docker_image': gpu_image}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(out, 'SMOKE =', smoke, '| rerun =', rerun, '| rerun3 =', rerun3, '| role =', role)
