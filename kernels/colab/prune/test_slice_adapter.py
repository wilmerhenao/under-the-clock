"""slice_adapter: a full-model PEFT adapter cut to a plan (kept channels, removed and renumbered layers) gives exactly the
sliced LoRA deltas; removed gate/up rows of an adapter trained on the cut report 0. usage: python3 kernels/colab/prune/test_slice_adapter.py"""
import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors.numpy import save_file, load_file
import slice_model as SM
rng = np.random.default_rng(0)
L, I, D, r = 6, 64, 16, 4
scores = {'agent_ff': rng.random((L, I))}
plan = SM.make_plan(scores, 0.5, drop=[2], multiple=16, group=1)
P = SM.PEFT_LAYERS
ad = {}
for i in range(L):
    for mod, (o, n) in {'gate_proj': (I, D), 'up_proj': (I, D), 'down_proj': (D, I), 'q_proj': (D, D)}.items():
        sec = 'mlp' if mod.endswith('proj') and mod != 'q_proj' else 'self_attn'
        ad[f'{P}{i}.{sec}.{mod}.lora_A.weight'] = rng.standard_normal((r, n)).astype(np.float32)
        B = rng.standard_normal((o, r)).astype(np.float32)
        ad[f'{P}{i}.{sec}.{mod}.lora_B.weight'] = B
src, dst = tempfile.mkdtemp(), tempfile.mkdtemp()
save_file(ad, f'{src}/adapter_model.safetensors'); json.dump({'r': r, 'lora_alpha': 8}, open(f'{src}/adapter_config.json', 'w'))
summ = SM.slice_adapter(src, dst, plan)
cut = load_file(f'{dst}/adapter_model.safetensors')
print(summ)
assert summ['tensors_of_removed_layers'] == 8 and summ['tensors'] == (L - 1) * 8 and summ['removed_rows_lora_b_absmax'] > 0
for new, old in enumerate(plan['kept_layers']):
    idx = plan['ff_idx'][old]
    for mod in ('gate_proj', 'up_proj'):
        dW_full = ad[f'{P}{old}.mlp.{mod}.lora_B.weight'] @ ad[f'{P}{old}.mlp.{mod}.lora_A.weight']
        dW_cut = cut[f'{P}{new}.mlp.{mod}.lora_B.weight'] @ cut[f'{P}{new}.mlp.{mod}.lora_A.weight']
        assert np.allclose(dW_full[idx], dW_cut)
    dW_full = ad[f'{P}{old}.mlp.down_proj.lora_B.weight'] @ ad[f'{P}{old}.mlp.down_proj.lora_A.weight']
    dW_cut = cut[f'{P}{new}.mlp.down_proj.lora_B.weight'] @ cut[f'{P}{new}.mlp.down_proj.lora_A.weight']
    assert np.allclose(dW_full[:, idx], dW_cut)
    assert np.array_equal(ad[f'{P}{old}.self_attn.q_proj.lora_A.weight'], cut[f'{P}{new}.self_attn.q_proj.lora_A.weight'])
assert json.load(open(f'{dst}/adapter_config.json'))['r'] == r
# an adapter trained on the cut (removed gate/up rows exactly 0) reports 0
for i in range(L):
    keep = np.zeros(I, bool); keep[plan['ff_idx'].get(i, [])] = True
    for mod in ('gate_proj', 'up_proj'):
        ad[f'{P}{i}.mlp.{mod}.lora_B.weight'][~keep] = 0
save_file(ad, f'{src}/adapter_model.safetensors')
assert SM.slice_adapter(src, tempfile.mkdtemp(), plan)['removed_rows_lora_b_absmax'] == 0.0
print('SLICE_ADAPTER OK')
