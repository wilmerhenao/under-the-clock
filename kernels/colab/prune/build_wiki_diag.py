#!/usr/bin/env python3
"""Kaggle kernel kernels/prune-wikidiag-gpu/ (B170): prune_score.py's setup and model-load cells + wiki_diag.py.
usage: python3 kernels/colab/prune/build_wiki_diag.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = (ROOT / 'kernels/colab/prune/prune_score.py').read_text()
cut = src.index('# %% [markdown]\n# ## The measuring probes')
head = src[:cut]
head, n = re.subn(r'^SMOKE = True ', 'SMOKE = False ', head, count=1, flags=re.M)
assert n == 1
head = head.replace("RUN_NAME = 'score-v1' ", "RUN_NAME = 'wikidiag' ", 1)
code = head + '\n' + (ROOT / 'kernels/colab/prune/wiki_diag.py').read_text()
compile(code, 'wiki_diag', 'exec')
gpu_image = re.search(r"^GPU_IMAGE = '([^']+)'", (ROOT / 'kernels/smoke/build.py').read_text(), re.M).group(1)
out = ROOT / 'kernels/prune-wikidiag-gpu'
out.mkdir(exist_ok=True)
(out / 'wiki_diag.py').write_text(code)
meta = {'id': 'verracodeguacas/g4dev-prune-wikidiag-gpu', 'title': 'g4dev-prune-wikidiag-gpu', 'code_file': 'wiki_diag.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaL4',
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse', 'verracodeguacas/g4dev-g11-data',
                            'verracodeguacas/g4dev-wikitext103-eval', 'verracodeguacas/g4dev-train-wheels'],
        'competition_sources': ['gemma-4-developer-agent'], 'kernel_sources': [],
        'model_sources': ['google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2'], 'docker_image': gpu_image}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(out)
