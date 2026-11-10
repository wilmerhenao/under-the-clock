"""Numerical parity between Hugging Face's Gemma 4 (what vLLM implements, what the GPU trainer and the scorer use) and
the Tunix Gemma 4 port the TPU trainer uses, on the same weights:
1. base forward: final-norm hidden states and soft-capped logits (HF Gemma4ForCausalLM vs lora_tpu.Gemma4Hidden);
2. LoRA: the Tunix/Qwix LoRA model vs HF with the exported PEFT deltas merged into its weights (export orientation on
   real shapes: q, fused kv / k_eq_v, o, gate/up/down);
3. loss: the TPU trainer's masked CE at labelled positions (lora_tpu.target_loss) vs CE computed from HF logits with
   the GPU trainer's HF shift.
Weights: --tiny (random, tiny_config) or --ckpt <HF dir with config.json + safetensors> with --layer-ids (checkpoint
layers to keep, e.g. 0,5 = one sliding + one full-attention layer of the 31B; they become layers 0..n-1). HF runs
first and is freed before Tunix (CPU RAM). fp32 compute on both sides.
usage: python parity_hf_tunix.py --tiny | --ckpt DIR --layer-ids 0,5 --len 1536 [--npz rows.npz] [--out r.json]
"""
import argparse
import gc
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tiny', action='store_true')
ap.add_argument('--ckpt', default='')
ap.add_argument('--layer-ids', default='0,5')
ap.add_argument('--len', type=int, default=48)
ap.add_argument('--npz', default='')
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--b-scale', type=float, default=0.02)
ap.add_argument('--out', default='')
ap.add_argument('--ref', default='', help='G7 PEFT adapter: keys/shapes of the kept layers must match ours')
a = ap.parse_args()
t0 = time.time()
res = {'mode': 'tiny' if a.tiny else 'ckpt'}


def log(**kv):
    kv['t_s'] = round(time.time() - t0, 1)
    print(json.dumps(kv, default=str), flush=True)
    res.update(kv)


# ---- weights (HF multimodal naming model.language_model.*, as the real checkpoint and the Tunix loader use) ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
work = tempfile.mkdtemp(prefix='parity_')
if a.tiny:
    import lora_tpu as LT
    tcfg = LT.tiny_config()
    hf_w = LT.tiny_hf_weights(tcfg)
    rng = np.random.default_rng(3)
    for k in hf_w:
        if k.endswith('layer_scalar'):
            hf_w[k] = (0.5 + rng.random(1)).astype(np.float32)  # exercise layer_scalar too
    text = dict(vocab_size=tcfg.num_embed, hidden_size=tcfg.embed_dim, intermediate_size=tcfg.hidden_dim,
                num_hidden_layers=tcfg.num_layers, num_attention_heads=tcfg.num_heads,
                num_key_value_heads=tcfg.num_kv_heads, head_dim=tcfg.head_dim, global_head_dim=tcfg.global_key_size,
                num_global_key_value_heads=tcfg.num_global_kv_heads, sliding_window=tcfg.sliding_window_size,
                layer_types=['sliding_attention'] * 5 + ['full_attention'], attention_k_eq_v=True,
                final_logit_softcapping=30.0, hidden_size_per_layer_input=0, tie_word_embeddings=True)
    ids_keep = list(range(tcfg.num_layers))
else:
    import torch
    from safetensors import safe_open
    full = json.load(open(os.path.join(a.ckpt, 'config.json')))
    text = dict(full.get('text_config', full))
    ids_keep = [int(x) for x in a.layer_ids.split(',')]
    lt = text['layer_types']
    text['layer_types'] = [lt[i] for i in ids_keep]
    text['num_hidden_layers'] = len(ids_keep)
    hf_w = {}
    remap = {f'model.language_model.layers.{i}.': f'model.language_model.layers.{j}.' for j, i in enumerate(ids_keep)}
    for f in sorted(glob.glob(os.path.join(a.ckpt, '*.safetensors'))):
        with safe_open(f, 'pt') as fh:  # 'pt': the checkpoint is bf16, which numpy cannot hold
            for k in fh.keys():
                if not k.startswith('model.language_model.'):
                    continue
                if '.layers.' in k:
                    pre = k[:k.index('.layers.') + 8] + k[k.index('.layers.') + 8:].split('.')[0] + '.'
                    if pre not in remap:
                        continue
                    nk = remap[pre] + k[len(pre):]
                else:
                    nk = k
                hf_w[nk] = fh.get_tensor(k).to(torch.float32).numpy()
    log(stage='subset', tensors=len(hf_w), layer_types=text['layer_types'],
        gb=round(sum(v.nbytes for v in hf_w.values()) / 2**30, 2), dtypes=sorted({str(v.dtype) for v in hf_w.values()}))

T = a.len
if a.npz:
    d = np.load(a.npz)
    offs = np.concatenate([[0], np.cumsum(d['lens'])])
    j = int(np.argmax(d['lens'] >= T)) if (d['lens'] >= T).any() else int(np.argmax(d['lens']))
    tokens = d['ids'][offs[j]:offs[j] + T][None].astype(np.int64)
    labels = d['labels'][offs[j]:offs[j] + T][None].astype(np.int64)
    T = tokens.shape[1]
else:
    rng = np.random.default_rng(0)
    tokens = rng.integers(2, text['vocab_size'], size=(1, T))
    labels = np.full_like(tokens, -100)
    labels[0, T // 3:T // 3 + 40] = tokens[0, T // 3:T // 3 + 40]
    labels[0, T - 30:] = tokens[0, T - 30:]
log(stage='tokens', T=int(T), labelled=int((labels[0, 1:] != -100).sum()))


# ---- HF side ----
def hf_run(weights):
    import torch
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    torch.set_grad_enabled(False)
    keep = {k: v for k, v in text.items() if k in Gemma4TextConfig().to_dict() and k not in ('architectures',)}
    cfg = Gemma4TextConfig(**keep)
    cfg._attn_implementation = 'eager'
    m = Gemma4ForCausalLM(cfg).to(torch.float32).eval()
    sd = {k.replace('model.language_model.', 'model.'): torch.from_numpy(np.asarray(v, np.float32)) for k, v in weights.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != 'lm_head.weight']
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    out = m(input_ids=torch.from_numpy(tokens), output_hidden_states=True)
    h = out.hidden_states[-1].float().numpy()
    logits = out.logits.float().numpy()
    del m, sd, out
    gc.collect()
    return h, logits


def ce_from_logits(logits):
    """GPU trainer's HF shift: logits[t] predict labels[t+1]; mean over labelled positions."""
    lg = logits[0, :-1].astype(np.float64)
    tg = labels[0, 1:]
    sel = tg != -100
    lse = np.log(np.exp(lg[sel] - lg[sel].max(-1, keepdims=True)).sum(-1)) + lg[sel].max(-1)
    return float((lse - lg[sel][np.arange(sel.sum()), tg[sel]]).mean())


h_hf, lg_hf = hf_run(hf_w)
loss_hf = ce_from_logits(lg_hf)
log(stage='hf_base', loss=loss_hf)
from safetensors.numpy import save_file, load_file  # noqa: E402
save_file({k: np.asarray(v, np.float32) for k, v in hf_w.items()}, os.path.join(work, 'model.safetensors'))
del hf_w  # CPU RAM: the fp32 subset is reloaded from disk for the LoRA merge
gc.collect()

# ---- Tunix side ----
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lora_tpu as LT  # noqa: E402
from tunix.models.gemma4 import model as gm  # noqa: E402

def tunix_cfg(tc, layer_types):
    """Tunix ModelConfig from an HF Gemma 4 text config."""
    rp = tc.get('rope_parameters') or {}
    sl, fu = rp.get('sliding_attention', {}), rp.get('full_attention', {})
    A_ = gm.AttentionType
    return gm.ModelConfig(
        num_layers=len(layer_types), num_embed=tc['vocab_size'], embed_dim=tc['hidden_size'],
        hidden_dim=tc['intermediate_size'], num_heads=tc['num_attention_heads'], head_dim=tc['head_dim'],
        num_kv_heads=tc['num_key_value_heads'], num_global_kv_heads=tc.get('num_global_key_value_heads'),
        global_key_size=tc.get('global_head_dim') or tc['head_dim'], sliding_window_size=tc.get('sliding_window'),
        final_logit_softcap=tc.get('final_logit_softcapping'), k_eq_v_global=bool(tc.get('attention_k_eq_v')),
        local_base_frequency=int(sl.get('rope_theta', 10_000)), global_base_frequency=int(fu.get('rope_theta', 1_000_000)),
        local_rope_proportion=float(sl.get('partial_rotary_factor', 1.0)),
        global_rope_proportion=float(fu.get('partial_rotary_factor', 1.0)),
        per_layer_input_dim=int(tc.get('hidden_size_per_layer_input') or 0),
        attention_pattern=tuple(A_.GLOBAL if t == 'full_attention' else A_.LOCAL_SLIDING for t in layer_types))


if a.tiny:
    cfg = LT.tiny_config()
else:
    from transformers import Gemma4TextConfig
    hf_filled = Gemma4TextConfig(**{k: v for k, v in text.items() if k in Gemma4TextConfig().to_dict()}).to_dict()
    cfg = tunix_cfg(hf_filled, text['layer_types'])  # HF's defaults for any field the checkpoint leaves out
    log(stage='tunix_cfg', rope=hf_filled.get('rope_parameters'), sliding_window=cfg.sliding_window_size,
        global_rope_proportion=cfg.global_rope_proportion, local_rope_proportion=cfg.local_rope_proportion,
        softcap=cfg.final_logit_softcap, k_eq_v=cfg.k_eq_v_global)
    if full.get('text_config', full).get('num_hidden_layers') == 60:  # the real 31B: the runner's preset must match it
        tc_full = full.get('text_config', full)
        tc_full = Gemma4TextConfig(**{k: v for k, v in tc_full.items() if k in Gemma4TextConfig().to_dict()}).to_dict()
        ref_cfg = tunix_cfg(tc_full, tc_full['layer_types'])
        pre = gm.ModelConfig.gemma4_31b()
        fields = ['num_layers', 'num_embed', 'embed_dim', 'hidden_dim', 'num_heads', 'head_dim', 'num_kv_heads',
                  'num_global_kv_heads', 'global_key_size', 'sliding_window_size', 'final_logit_softcap',
                  'k_eq_v_global', 'local_base_frequency', 'global_base_frequency', 'local_rope_proportion',
                  'global_rope_proportion', 'per_layer_input_dim']
        mism = {f: [getattr(pre, f), getattr(ref_cfg, f)] for f in fields if getattr(pre, f) != getattr(ref_cfg, f)}
        pat = [pre.attention_pattern[i % len(pre.attention_pattern)] for i in range(60)]
        if pat != list(ref_cfg.attention_pattern):
            mism['attention_pattern'] = 'differs'
        log(stage='preset_check', preset_mismatch=mism, preset_ok=not mism)
cfg.param_dtype = cfg.dtype = jnp.float32
cfg.remat_config = gm.RematConfig.NONE
cfg.use_flash_attention = False
mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)
with mesh, jax.set_mesh(mesh):
    base = LT.load_hidden_model(work, cfg, mesh, dtype=jnp.float32)
    tk = jnp.asarray(tokens, jnp.int32)
    mask = LT.causal_mask(jnp.asarray([T]), T)
    h_tx = np.asarray(base(tk, None, None, mask), np.float32)
    lg_tx = jnp.asarray(h_tx) @ base.embedder.input_embedding.value.T
    if cfg.final_logit_softcap is not None:  # what the trainer's target_loss applies
        lg_tx = jnp.tanh(lg_tx / cfg.final_logit_softcap) * cfg.final_logit_softcap
    lg_tx = np.asarray(lg_tx, np.float32)

    def rel(x, y):
        return float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-9))
    top1 = float((lg_tx[0].argmax(-1) == lg_hf[0].argmax(-1)).mean())
    log(stage='base_parity', hidden_rel=rel(h_tx, h_hf), logits_rel=rel(lg_tx, lg_hf), top1_agree=top1,
        loss_tunix_from_logits=ce_from_logits(lg_tx))

    # trainer loss path (gathered positions, chunked, soft-capped) vs HF logits CE
    bt = LT.make_batch(tokens.astype(np.int32), labels.astype(np.int32), np.array([T], np.int32),
                       max_targets=int(np.ceil(max(1, (labels[0, 1:] != -100).sum()) / 8) * 8))
    s = LT.target_loss(base, cfg, jnp.asarray(h_tx), bt['pos'], bt['tgt'], bt['w'], chunk=8)
    loss_tr = float(s / bt['w'].sum())
    log(stage='loss_parity', loss_trainer=loss_tr, loss_hf=loss_hf, abs_diff=abs(loss_tr - loss_hf))

    # LoRA: Tunix LoRA model vs HF with the exported PEFT deltas merged
    lm = LT.apply_lora(base, a.rank, 2.0 * a.rank)
    LT.randomize_lora_b(lm, seed=1, scale=a.b_scale)
    peft = LT.export_peft(lm, cfg, a.rank, 2.0 * a.rank)
    h_lora = np.asarray(lm(tk, None, None, mask), np.float32)
    del base, lm
    gc.collect()
if a.ref:
    from safetensors import safe_open as so2
    with so2(a.ref, 'np') as r:
        ref_shapes = {k: tuple(r.get_slice(k).get_shape()) for k in r.keys()}
    P = LT.PEFT_PREFIX
    want = {}
    for j, i in enumerate(ids_keep):
        for k, shp in ref_shapes.items():
            if k.startswith(f'{P}{i}.'):
                want[f'{P}{j}.' + k[len(f'{P}{i}.'):]] = shp
    ours = {k: tuple(v.shape) for k, v in peft.items()}
    log(stage='ref_layout', ref_keys_for_layers=len(want), ours=len(ours),
        missing=sorted(set(want) - set(ours))[:5], extra=sorted(set(ours) - set(want))[:5],
        shape_mismatch=[k for k in set(want) & set(ours) if want[k] != ours[k]][:5],
        ref_match=set(want) == set(ours) and all(want[k] == ours[k] for k in want))
merged = LT.merge_into_hf(load_file(os.path.join(work, 'model.safetensors')), peft, a.rank, 2.0 * a.rank)
gc.collect()
h_hf2, lg_hf2 = hf_run(merged)
log(stage='lora_parity', peft_tensors=len(peft), hidden_rel=rel(h_lora, h_hf2),
    lora_moved_rel=rel(h_lora, h_tx), modules=sorted({k.split('.')[-3] for k in peft}))
# fp32 on identical weights: only op-order noise is allowed (a rope partial-factor mismatch gave 7e-4 on the tiny model)
ok = (res['hidden_rel'] < 1e-4 and res['top1_agree'] > 0.99 and res['abs_diff'] < 1e-4
      and res['lora_moved_rel'] > 20 * res['hidden_rel'] and res.get('ref_match', True) and res.get('preset_ok', True))
res['ok'] = bool(ok)
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
print('PARITY OK' if ok else 'PARITY FAILED', json.dumps({k: res.get(k) for k in ('hidden_rel', 'top1_agree', 'abs_diff',
                                                                                  'lora_moved_rel')}))
