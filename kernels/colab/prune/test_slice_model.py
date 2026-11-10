#!/usr/bin/env python3
"""CPU test of slice_model: a tiny random Gemma 4 (real config structure, 14 layers, random layer scales) is cut
(half the feed-forward channels, three layers removed, two of them next to each other) and the cut checkpoint,
loaded back with transformers, must give the same outputs as the full model with the same parts switched off by
prune_score's hooks (channels masked, removed layers replaced by their scale factor).
usage: PRUNE_TOK_DIR=<dir with the real config.json> python3 kernels/colab/prune/test_slice_model.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slice_model import make_plan, plan_masks, slice_checkpoint  # noqa: E402
from transformers import Gemma4Config, Gemma4ForConditionalGeneration  # noqa: E402

tmp = Path(tempfile.mkdtemp())
c = json.load(open(os.path.join(os.environ['PRUNE_TOK_DIR'], 'config.json')))
c['text_config'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=14, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, num_global_key_value_heads=1, global_head_dim=32,
                        layer_types=None, vocab_size=512, vocab_size_per_layer_input=512)
c['vision_config'].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=16)
torch.manual_seed(0)
model = Gemma4ForConditionalGeneration(Gemma4Config(**{k: v for k, v in c.items()
                                                       if k not in ('architectures', 'transformers_version')}))
TM = model.model.language_model
for lyr in TM.layers:
    lyr.layer_scalar.fill_(0.5 + torch.rand(1).item())
model = model.float().eval()   # the config says bfloat16 for some parts; compare in float32 throughout
src = tmp / 'src'
model.save_pretrained(src, max_shard_size='300KB')
assert len(list(src.glob('*.safetensors'))) > 1, 'want several shards'
L, I = 14, 128
assert json.load(open(src / 'config.json'))['text_config']['layer_types'], 'saved config lists layer types'

rng = np.random.default_rng(0)
scores = {'agent_ff': rng.random((L, I))}
plan = make_plan(scores, 0.5, drop=[3, 4, 9], multiple=16)
assert plan['keep_ff'] == 64 and plan['fold_into'] == {3: 2, 4: 2, 9: 8}, plan['fold_into']
try:
    make_plan(scores, 0.5, drop=[0])
    raise AssertionError('dropping layer 0 must be refused')
except AssertionError as e:
    assert 'cannot drop' in str(e)



def ff_hook(mask, li):
    def f(mod, args):
        return (args[0] * mask[li].to(args[0].dtype),)
    return f


def layer_hook(drop, li):
    def f(mod, args, kwargs, out):
        if li in drop:
            return args[0] * mod.layer_scalar.to(args[0].dtype)
    return f


ids = torch.randint(5, 500, (1, 1500))   # longer than the sliding window, so both attention kinds matter


def masked_logits(m, plan):
    """The full model m with the plan's parts switched off by prune_score-style hooks."""
    mask, drop = plan_masks(plan)
    hooks = []
    for li, lyr in enumerate(m.model.language_model.layers):
        hooks += [lyr.mlp.down_proj.register_forward_pre_hook(ff_hook(mask, li)),
                  lyr.register_forward_hook(layer_hook(drop, li), with_kwargs=True)]
    with torch.no_grad():
        out = m(input_ids=ids).logits
    for h in hooks:
        h.remove()
    return out


with torch.no_grad():
    base = model(input_ids=ids).logits


def check(plan, name, src_dir, ref_model, **kw):
    dst = tmp / name
    summ = slice_checkpoint(str(src_dir), str(dst), plan, shard_gb=0.0005)
    cut = Gemma4ForConditionalGeneration.from_pretrained(dst, dtype=torch.float32, **kw).eval()
    tc = cut.config.get_text_config()
    assert tc.num_hidden_layers == L - len(plan['drop']) and tc.intermediate_size == plan['keep_ff']
    full_types = model.config.get_text_config().layer_types
    assert tc.layer_types == [full_types[i] for i in plan['kept_layers']]
    assert (dst / 'tokenizer.json').exists() == (src_dir / 'tokenizer.json').exists()
    assert (dst / 'prune_plan.json').exists() and (dst / 'prune_plan_channels.npz').exists()
    ref = masked_logits(ref_model, plan)
    with torch.no_grad():
        got = cut(input_ids=ids).logits
    err = (ref - got).abs().max().item()
    moved = (ref - base).abs().max().item()
    print(f'{name}: {summ}; max |masked full - cut| = {err:.2e}; the cut moved the logits by up to {moved:.2e}')
    assert moved > 1e-2, 'the cut should change the model'
    assert err < 1e-4 * max(1.0, ref.abs().max().item()), err
    return dst, summ


# 1. single channels, three layers removed (two of them next to each other)
d1, s1 = check(plan, 'cut_channels', src, model)
assert s1['shards'] > 1 and s1['format'] == '16-bit'

# 2. whole blocks of 32 channels (what the 4-bit checkpoint can be cut by)
plan_b = make_plan(scores, 0.5, drop=[5], multiple=32, group=32)
idx = plan_b['ff_idx'][0]
assert len(idx) == 64
assert all(set(np.diff(idx.reshape(-1, 32), axis=1).ravel()) == {1} for idx in plan_b['ff_idx'].values())
G = scores['agent_ff'][0].reshape(4, 32).sum(1)
assert set(plan_b['ff_idx'][0][::32] // 32) == set(np.argsort(-G)[:2]), 'keeps the two best blocks'
check(plan_b, 'cut_blocks', src, model)

# 3. the same block cut on a 4-bit (compressed-tensors w4a16, blocks of 32) copy of the model: the cut 4-bit tensors
#    must be exactly the cut of the 4-bit tensors, and (if transformers can load it) give the masked model's outputs
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
qsrc = tmp / 'src_w4'
qsrc.mkdir()
sd = {}
for fn in sorted(src.glob('*.safetensors')):
    sd.update(load_file(fn))
fq = Gemma4ForConditionalGeneration(model.config).float().eval()   # the full model with 4-bit feed-forward weights
fq.load_state_dict(model.state_dict())
for k in [k for k in sd if '.mlp.' in k and 'language_model' in k]:
    w = sd.pop(k).float()
    gw = w.reshape(w.shape[0], -1, 32)
    sc = (gw.abs().amax(-1) / 7).clamp(min=1e-8)
    q = torch.round(gw / sc[..., None]).clamp(-8, 7)
    deq = (q * sc[..., None]).reshape(w.shape)
    fq.get_parameter(k).data.copy_(deq)
    base_k = k[:-len('.weight')]
    sd[base_k + '.weight_packed'] = pack_to_int32(q.reshape(w.shape).to(torch.int8), 4)
    sd[base_k + '.weight_scale'] = sc.float()
    sd[base_k + '.weight_shape'] = torch.tensor(w.shape)
save_file(sd, str(qsrc / 'model.safetensors'), metadata={'format': 'pt'})
real = json.load(open(os.path.join(os.environ['PRUNE_TOK_DIR'], 'config.json')))
assert real.get('quantization_config') is None   # the 16-bit config: borrow the 4-bit settings from the w4a16 config
w4cfg_path = os.environ.get('PRUNE_W4_CONFIG')
qc = json.load(open(w4cfg_path))['quantization_config'] if w4cfg_path else {
    'quant_method': 'compressed-tensors', 'format': 'pack-quantized', 'quantization_status': 'compressed',
    'config_groups': {'group_0': {'format': 'pack-quantized', 'targets': ['Linear'], 'input_activations': None,
                                  'output_activations': None, 'weights': {
                                      'num_bits': 4, 'type': 'int', 'symmetric': True, 'strategy': 'group',
                                      'group_size': 32, 'dynamic': False, 'actorder': None, 'observer': 'minmax'}}},
    'ignore': ['lm_head']}
qc['ignore'] = ['lm_head', 're:.*vision.*', 're:.*self_attn.*', 're:.*embed_vision.*']
cq = json.load(open(src / 'config.json'))
cq['quantization_config'] = qc
json.dump(cq, open(qsrc / 'config.json', 'w'))
dq = tmp / 'cut_w4'
sq = slice_checkpoint(str(qsrc), str(dq), plan_b, shard_gb=0.0005)
assert sq['format'] == 'w4 g32', sq
got = {}
for fn in sorted(dq.glob('*.safetensors')):
    got.update(load_file(fn))
for new, old in enumerate(plan_b['kept_layers']):
    ch = torch.as_tensor(plan_b['ff_idx'][old])
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        pk = f'model.language_model.layers.{{}}.mlp.{proj}.'
        o = pk.format(old)
        n = pk.format(new)
        full = unpack_from_int32(sd[o + 'weight_packed'], 4, torch.Size(sd[o + 'weight_shape'].tolist())).float()
        full = full * sd[o + 'weight_scale'].repeat_interleave(32, 1)
        shape = torch.Size(got[n + 'weight_shape'].tolist())
        mine = unpack_from_int32(got[n + 'weight_packed'], 4, shape).float() * got[n + 'weight_scale'].repeat_interleave(32, 1)
        want = full.index_select(0, ch) if proj != 'down_proj' else full.index_select(1, ch)
        assert shape == want.shape and torch.equal(mine, want), (old, proj)
print('4-bit tensors: cut is exact for every kept layer')
try:
    cutq = Gemma4ForConditionalGeneration.from_pretrained(dq, dtype=torch.float32).eval()
except Exception as e:   # transformers' compressed-tensors loader is optional here; vLLM loads it on Kaggle
    print('transformers could not load the 4-bit cut (tensor check above still holds):', str(e)[:300])
else:
    ref = masked_logits(fq, plan_b)
    with torch.no_grad():
        out = cutq(input_ids=ids).logits
    err = (ref - out).abs().max().item()
    print(f'4-bit cut loaded by transformers: max |masked 4-bit full - cut| = {err:.2e}')
    assert err < 1e-3 * max(1.0, ref.abs().max().item()), err

# 4. single channels on the 4-bit checkpoint: gate/up cut exactly, down_proj re-quantized in new blocks of 32
from slice_model import _pack4, _unpack4  # noqa: E402
qt = torch.randint(-8, 8, (7, 64), dtype=torch.int8)
assert torch.equal(_pack4(qt), pack_to_int32(qt, 4)) and torch.equal(_unpack4(pack_to_int32(qt, 4), 64), qt)
plan_c = make_plan(scores, 0.5, drop=[5], multiple=32, group=1)
assert any(set(np.diff(v.reshape(-1, 32), axis=1).ravel()) != {1} for v in plan_c['ff_idx'].values()), 'not blocks'
act = rng.random((L, I)) + 0.1
dc = tmp / 'cut_w4_channels'
sc_ = slice_checkpoint(str(qsrc), str(dc), plan_c, shard_gb=0.0005, act=act)
assert sc_['format'] == 'w4 g32, down_proj re-quantized' and 0 < sc_['requant_rel_err']['max'] < 0.2, sc_
got = {}
for fn in sorted(dc.glob('*.safetensors')):
    got.update(load_file(fn))
fq_c = Gemma4ForConditionalGeneration(model.config).float().eval()   # fq with the re-quantized down_proj values
fq_c.load_state_dict(fq.state_dict())
for new, old in enumerate(plan_c['kept_layers']):
    ch = torch.as_tensor(plan_c['ff_idx'][old])
    for proj in ('gate_proj', 'up_proj', 'down_proj'):
        o, n = f'model.language_model.layers.{old}.mlp.{proj}.', f'model.language_model.layers.{new}.mlp.{proj}.'
        full = unpack_from_int32(sd[o + 'weight_packed'], 4, torch.Size(sd[o + 'weight_shape'].tolist())).float()
        full = full * sd[o + 'weight_scale'].repeat_interleave(32, 1)
        shape = torch.Size(got[n + 'weight_shape'].tolist())
        mine = unpack_from_int32(got[n + 'weight_packed'], 4, shape).float() * got[n + 'weight_scale'].repeat_interleave(32, 1)
        if proj != 'down_proj':
            assert torch.equal(mine, full.index_select(0, ch)), (old, proj)
        else:
            want = full.index_select(1, ch)
            step = got[n + 'weight_scale'].repeat_interleave(32, 1)
            # every value is the nearest step of its block's new scale (or the end of the 4-bit range)
            assert shape == want.shape and torch.equal(mine, (want / step).round().clamp(-8, 7) * step), old
            # a new block made of old blocks with one shared scale keeps those values exactly
            same = (sd[o + 'weight_scale'].repeat_interleave(32, 1).index_select(1, ch).view(-1, len(ch) // 32, 32)
                    .float().std(-1) == 0).repeat_interleave(32, 1)
            assert torch.equal(mine[same], want[same]), old
            w = fq_c.get_parameter(f'model.language_model.layers.{old}.mlp.down_proj.weight').data
            w[:, ch] = mine
print('4-bit single-channel cut: gate/up exact, down_proj on the nearest step of its new scale;', sc_['requant_rel_err'])
cutc = Gemma4ForConditionalGeneration.from_pretrained(dc, dtype=torch.float32).eval()
ref = masked_logits(fq_c, plan_c)
with torch.no_grad():
    out = cutc(input_ids=ids).logits
err = (ref - out).abs().max().item()
print(f'4-bit single-channel cut loaded by transformers: max |masked (re-quantized) full - cut| = {err:.2e}')
assert err < 1e-3 * max(1.0, ref.abs().max().item()), err
print('TEST OK', tmp)
