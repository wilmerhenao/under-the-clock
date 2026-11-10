#!/usr/bin/env python3
"""Where transformers puts Gemma 4 31B (bnb 4-bit, vision tower / embed_vision / lm_head kept bf16) on the GPUs, on CPU.

Builds the model on the meta device from its config.json and calls transformers' own `_get_device_map` (the code
`from_pretrained(device_map=...)` runs) with a stand-in for the 4-bit quantizer (Linear weights 0.5 bytes, the rest at
their dtype, memory caps x0.9 as bitsandbytes does). Prints GiB of weights and decoder layers per GPU. Also checks an
explicit `--layer-split` map built as `train_lora.py --layer-split` builds it: every parameter covered, weights per GPU.
Validated on G3 (2026-10-06): auto -> [3.69, 4.09, 4.09, 5.47] GiB, 0/18/18/24 layers; measured [3.6, 4.0, 4.0, 5.4].

usage: sim_placement.py runs/analysis/gemma4_31b_it_config.json [--auto 19,19,19,19 ...] [--split 3,19,19,19 ...]
"""
import argparse
import collections
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.integrations.accelerate import _get_device_map

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('config')
ap.add_argument('--auto', nargs='*', default=['19,19,19,19', '20,19,19,13'], help='per-GPU GiB caps for device_map auto')
ap.add_argument('--sequential', nargs='*', default=[], help='per-GPU GiB caps for device_map sequential')
ap.add_argument('--split', nargs='*', default=['3,19,19,19'], help='decoder layers per GPU (explicit map)')
a = ap.parse_args()
d = Path(tempfile.mkdtemp())
shutil.copy(a.config, d / 'config.json')
cfg = AutoConfig.from_pretrained(str(d))
with torch.device('meta'):
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
SKIP = ('vision_tower', 'embed_vision', 'lm_head')
MODS = dict(model.named_modules())
TIED = getattr(model, 'all_tied_weights_keys', {})


def esize(name, p):
    mod = MODS.get(name.rsplit('.', 1)[0])
    return 0.5 if isinstance(mod, nn.Linear) and name.endswith('.weight') and not any(s in name for s in SKIP) else p.element_size()


class Quant:  # the parts of Bnb4BitHfQuantizer that placement uses
    def param_element_size(self, model, name, param):
        return esize(name, param)

    def adjust_max_memory(self, mm):
        return {k: v * 0.90 for k, v in mm.items()}

    def validate_environment(self, *args, **kwargs):
        pass


def summary(dm, n):
    sizes, uncovered = collections.Counter(), 0
    for name, p in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers()):
        keys = [k for k in dm if k == '' or name == k or name.startswith(k + '.')]
        if not keys:
            uncovered += 1
            continue
        if name not in TIED:
            sizes[dm[max(keys, key=len)]] += p.numel() * esize(name, p)
    layer = lambda k: '.layers.' in k and 'language_model' in k and k.rsplit('.', 1)[-1].isdigit()  # noqa: E731
    return {'gib': [round(sizes[i] / 2**30, 2) for i in range(n)], 'layers': [sum(1 for k, v in dm.items() if layer(k) and v == i) for i in range(n)],
            'other': {k: v for k, v in dm.items() if not layer(k)}, 'uncovered_params': uncovered}


for mode, specs in (('auto', a.auto), ('sequential', a.sequential)):
    for caps in specs:
        g = [float(x) for x in caps.split(',')]
        dm = _get_device_map(model, mode, {i: f'{x}GiB' for i, x in enumerate(g)}, Quant())
        print(mode, caps, json.dumps(summary(dm, len(g))))
for split in a.split:
    counts = [int(x) for x in split.split(',')]
    lo = [sum(counts[:j]) for j in range(len(counts))]
    dm = {'model.vision_tower': 0, 'model.language_model.embed_tokens': 0, 'lm_head': 0}
    dm.update({f'model.language_model.layers.{i}': g for g, (s, c) in enumerate(zip(lo, counts)) for i in range(s, s + c)})
    dm.update({'model.language_model.norm': len(counts) - 1, 'model.language_model.rotary_emb': len(counts) - 1,
               'model.embed_vision': len(counts) - 1})
    print('split', split, json.dumps(summary(dm, len(counts))))
