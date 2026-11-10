"""CPU test of slice_model.merge_lora_w4 (B192) on a synthetic compressed-tensors 4-bit checkpoint: a zero adapter
changes nothing; rounding to nearest loses a delta smaller than half a grid step (why the merge cannot round); stochastic
rounding keeps the trained grid, is unbiased (mean over seeds = W + dW), moves about |dW| / step of the values, and stays
within one step of W + dW; a 16-bit weight is merged exactly; every other tensor and file is copied unchanged.
usage: python3 kernels/colab/prune/test_merge_lora_w4.py"""
import json
import os
import sys
import tempfile

import numpy as np
import torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slice_model as SM  # noqa: E402

G, ROWS, COLS, R, ALPHA = 32, 48, 128, 4, 8
tmp = tempfile.mkdtemp()
src, dst, ad = (os.path.join(tmp, x) for x in ('src', 'dst', 'ad'))
for d in (src, ad):
    os.makedirs(d)
cfg = {'text_config': {'num_hidden_layers': 2}, 'quantization_config': {
    'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
    'config_groups': {'group_0': {'weights': {'num_bits': 4, 'strategy': 'group', 'symmetric': True, 'group_size': G}}}}}
json.dump(cfg, open(os.path.join(src, 'config.json'), 'w'))
open(os.path.join(src, 'tokenizer.json'), 'w').write('{"tok": 1}')
g = torch.Generator().manual_seed(0)
T, Q, S = {}, {}, {}
for li in range(2):
    base = f'model.language_model.layers.{li}.mlp.gate_proj'
    q = torch.randint(-8, 8, (ROWS, COLS), generator=g).to(torch.int8)
    s = (torch.rand(ROWS, COLS // G, generator=g) * 0.02 + 0.005).to(torch.bfloat16)
    T[base + '.weight_packed'], T[base + '.weight_scale'] = SM._pack4(q), s
    T[base + '.weight_shape'] = torch.tensor([ROWS, COLS])
    Q[base], S[base] = q, s
T['model.language_model.layers.0.self_attn.q_proj.weight'] = (torch.randn(ROWS, COLS, generator=g) * 0.02).to(torch.bfloat16)
T['model.language_model.norm.weight'] = torch.ones(COLS, dtype=torch.bfloat16)
save_file(T, os.path.join(src, 'model-00001-of-00001.safetensors'), metadata={'format': 'pt'})
json.dump({'metadata': {}, 'weight_map': {k: 'model-00001-of-00001.safetensors' for k in T}},
          open(os.path.join(src, 'model.safetensors.index.json'), 'w'))
targets = ['0.mlp.gate_proj', '1.mlp.gate_proj', '0.self_attn.q_proj']


def write_adapter(bscale, ascale=1.0, seed=1):
    gg = torch.Generator().manual_seed(seed)
    t = {}
    for name in targets:
        p = f'{SM.PEFT_LAYERS}{name}'
        t[p + '.lora_A.weight'] = (torch.randn(R, COLS, generator=gg) * ascale).numpy().astype(np.float32)
        t[p + '.lora_B.weight'] = (torch.randn(ROWS, R, generator=gg) * bscale).numpy().astype(np.float32)
    save_np(t, os.path.join(ad, 'adapter_model.safetensors'))
    json.dump({'r': R, 'lora_alpha': ALPHA, 'target_modules': ['gate_proj', 'q_proj']},
              open(os.path.join(ad, 'adapter_config.json'), 'w'))
    return t


def delta(t, name):
    p = f'{SM.PEFT_LAYERS}{name}'
    return (ALPHA / R) * torch.from_numpy(t[p + '.lora_B.weight']) @ torch.from_numpy(t[p + '.lora_A.weight'])


def merged(mode, seed=0):
    import shutil
    shutil.rmtree(dst, ignore_errors=True)
    summ = SM.merge_lora_w4(src, dst, ad, mode=mode, seed=seed, log=lambda *a: None)
    return load_file(os.path.join(dst, 'model-00001-of-00001.safetensors')), summ


def deq(out, base):
    return SM._unpack4(out[base + '.weight_packed'], COLS).float() * S[base].float().repeat_interleave(G, 1)


# 1. zero adapter: identical tensors
write_adapter(0.0)
out, summ = merged('stochastic')
assert all(torch.equal(out[k], T[k]) for k in T), 'a zero adapter changed the checkpoint'
assert summ['moved_share'] == 0.0 and summ['matrices'] == 3, summ
assert open(os.path.join(dst, 'tokenizer.json')).read() == '{"tok": 1}' and os.path.exists(os.path.join(dst, 'config.json'))

# 2. a small delta (~2% of a step, like the healing adapter): nearest rounding loses it entirely
t = write_adapter(1e-3, 0.05)
d0 = delta(t, '0.mlp.gate_proj')
step = S['model.language_model.layers.0.mlp.gate_proj'].float().repeat_interleave(G, 1)
print('small delta, mean |dW| / step =', round(float((d0 / step).abs().mean()), 4))
out, summ = merged('nearest')
assert torch.equal(out['model.language_model.layers.0.mlp.gate_proj.weight_packed'],
                   T['model.language_model.layers.0.mlp.gate_proj.weight_packed']), 'nearest kept a sub-step delta?'
assert summ['moved_share'] == 0.0, summ

# 3. stochastic: unbiased over seeds, moves about |dW| / step of the values, keeps the scales
base = 'model.language_model.layers.0.mlp.gate_proj'
W = deq(T, base)
acc, shares = torch.zeros_like(W), []
for sd in range(200):
    out, summ = merged('stochastic', seed=sd)
    assert torch.equal(out[base + '.weight_scale'], T[base + '.weight_scale']), 'scales changed'
    acc += deq(out, base)
    shares.append(summ['moved_share'])
mean_err = float(((acc / 200 - (W + d0)) / step).abs().mean())
exp_share = float((d0 / step).abs().clamp(max=1).mean())
print('stochastic: mean |E[merged] - (W + dW)| / step =', round(mean_err, 5), '; moved share', round(np.mean(shares), 4),
      'expected about', round(exp_share, 4))
assert mean_err < 0.01, mean_err                      # unbiased up to sampling noise (200 seeds)
assert abs(np.mean(shares) - exp_share) < 0.3 * exp_share + 1e-3, (np.mean(shares), exp_share)

# 4. a large delta (several steps): within one step of W + dW, except where clamped to the 4-bit range
t = write_adapter(0.2, 1.0, seed=3)
d0 = delta(t, '0.mlp.gate_proj')
out, summ = merged('stochastic', seed=0)
tgt = (W + d0) / step
got = deq(out, base) / step
inside = (tgt > -8) & (tgt < 7)
assert float((got - tgt)[inside].abs().max()) <= 1.0 + 1e-4, float((got - tgt)[inside].abs().max())
assert float(got.min()) >= -8 and float(got.max()) <= 7

# 5. 16-bit weight: exact merge
qb = 'model.language_model.layers.0.self_attn.q_proj'
exp = (T[qb + '.weight'].float() + delta(t, '0.self_attn.q_proj')).to(torch.bfloat16)
assert torch.equal(out[qb + '.weight'], exp), '16-bit merge is not exact'
# untouched tensors
assert torch.equal(out['model.language_model.norm.weight'], T['model.language_model.norm.weight'])
assert torch.equal(out['model.language_model.layers.1.mlp.gate_proj.weight_shape'],
                   T['model.language_model.layers.1.mlp.gate_proj.weight_shape'])

# 6. several shards, a module's packed weights in one and its scales in the next (slice_checkpoint closes a shard after
# whichever tensor crosses the size limit; the review found this on the real 5-shard cut): same result as one shard,
# bit for bit (one random stream per matrix), every shard keeps its tensors, the index is copied
src3, dst3 = os.path.join(tmp, 'src3'), os.path.join(tmp, 'dst3')
os.makedirs(src3)
for fn in ('config.json', 'tokenizer.json'):
    open(os.path.join(src3, fn), 'w').write(open(os.path.join(src, fn)).read())
keys = sorted(T)
cut_at = keys.index('model.language_model.layers.0.mlp.gate_proj.weight_packed') + 1   # scale goes to the next shard
parts = [keys[:cut_at], keys[cut_at:cut_at + 3], keys[cut_at + 3:]]
wm3 = {}
for j, ks in enumerate(parts):
    name = f'model-{j + 1:05d}-of-00003.safetensors'
    save_file({k: T[k] for k in ks}, os.path.join(src3, name), metadata={'format': 'pt'})
    wm3.update({k: name for k in ks})
assert wm3[base + '.weight_packed'] != wm3[base + '.weight_scale']
json.dump({'metadata': {}, 'weight_map': wm3}, open(os.path.join(src3, 'model.safetensors.index.json'), 'w'))
summ3 = SM.merge_lora_w4(src3, dst3, ad, mode='stochastic', seed=0, log=lambda *a: None)
out3 = {}
for j in range(3):
    part = load_file(os.path.join(dst3, f'model-{j + 1:05d}-of-00003.safetensors'))
    assert sorted(part) == sorted(parts[j]), 'a shard lost or gained tensors'
    out3.update(part)
assert sorted(out3) == sorted(out) and all(torch.equal(out3[k], out[k]) for k in out), 'layout changed the merge'
assert summ3['matrices'] == summ['matrices'] and summ3['moved_share'] == summ['moved_share'], (summ3, summ)
assert json.load(open(os.path.join(dst3, 'model.safetensors.index.json')))['weight_map'] == wm3
print('MERGE_LORA_W4 OK')
