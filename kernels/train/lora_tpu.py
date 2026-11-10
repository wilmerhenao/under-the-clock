"""Gemma 4 31B LoRA training pieces for TPU (JAX + Tunix's Gemma 4 port + Qwix LoRA), shared by the TPU kernel and the
CPU round-trip test (`kernels/train/test_lora_tpu.py`).

- `Gemma4Hidden`: Tunix's Gemma4 whose `__call__` returns final-norm hidden states instead of logits. Qwix intercepts
  only the top-level `__call__`, so LoRA stays active, and the loss can gather the trainable positions first: full
  logits for a 12k row would be 12k x 262k fp32 (12.9 GB).
- `make_train_step`: masked next-token CE at the labelled positions (labels from train_lora.py --dump-npz, i.e. the
  same tokens and masks the GPU trainer uses), logits in chunks against the tied embedding with Gemma's soft-cap 30.
- `export_peft`: Qwix LoRA factors -> PEFT safetensors keys/shapes as written by the GPU trainer
  (base_model.model.model.language_model.layers.N.{self_attn,mlp}.X.lora_{A,B}.weight). Tunix fuses k/v into
  `kv_einsum` (one A, B split per k/v) and full-attention layers share K and V (`k_einsum`; HF has no v_proj there).
- `check_export`: merges the exported adapter into the HF base weights, reloads them through Tunix's own loader and
  compares logits with the LoRA model; this exercises orientation, scaling and the k/v split end to end.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from flax import nnx
from qwix._src.providers import lora as qwix_lora
from tunix.models import safetensors_loader
from tunix.models.gemma4 import model as gm
from tunix.models.gemma4 import params_safetensors as gps

TARGETS = r'.*(q_einsum|kv_einsum|k_einsum|attn_vec_einsum|gate_proj|up_proj|down_proj)'
PEFT_PREFIX = 'base_model.model.model.language_model.layers.'
HF_PREFIX = 'model.language_model.layers.'


def tiny_config(mesh_test=False):
    """6-layer random-weight stand-in with the 31B's structure (5 sliding + 1 full layer, GQA, k_eq_v, 2x head_dim
    on the full layer) for CPU tests. mesh_test: head counts divisible by tp = 4 (8 fake CPU devices, fsdp 2 x tp 4)."""
    A = gm.AttentionType
    nh, kv = (8, 4) if mesh_test else (4, 2)
    return gm.ModelConfig(num_layers=6, num_embed=512, embed_dim=64, hidden_dim=128, num_heads=nh, head_dim=16,
                          num_kv_heads=kv, num_global_kv_heads=kv, global_key_size=32, sliding_window_size=8,
                          k_eq_v_global=True, attention_pattern=(A.LOCAL_SLIDING,) * 5 + (A.GLOBAL,),
                          param_dtype=jnp.float32, dtype=jnp.float32, remat_config=gm.RematConfig.DECODER)


def tiny_hf_weights(cfg, seed=0):
    """Random HF-layout (model.language_model.*) tensors for tiny_config()."""
    rng = np.random.default_rng(seed)
    pattern = cfg.attention_pattern

    def r(*s):
        return (rng.standard_normal(s) * 0.05).astype(np.float32)
    D, N, F, V = cfg.embed_dim, cfg.num_heads, cfg.hidden_dim, cfg.num_embed
    hf, P = {}, 'model.language_model.'
    hf[P + 'embed_tokens.weight'] = r(V, D)
    hf[P + 'norm.weight'] = r(D)
    for i in range(cfg.num_layers):
        g = pattern[i % len(pattern)] == gm.AttentionType.GLOBAL
        h, k = (cfg.global_key_size, cfg.num_global_kv_heads) if g else (cfg.head_dim, cfg.num_kv_heads)
        pre = f'{P}layers.{i}.'
        hf[pre + 'self_attn.q_proj.weight'] = r(N * h, D)
        hf[pre + 'self_attn.k_proj.weight'] = r(k * h, D)
        if not g:
            hf[pre + 'self_attn.v_proj.weight'] = r(k * h, D)
        hf[pre + 'self_attn.o_proj.weight'] = r(D, N * h)
        hf[pre + 'self_attn.q_norm.weight'] = r(h)
        hf[pre + 'self_attn.k_norm.weight'] = r(h)
        hf[pre + 'mlp.gate_proj.weight'] = r(F, D)
        hf[pre + 'mlp.up_proj.weight'] = r(F, D)
        hf[pre + 'mlp.down_proj.weight'] = r(D, F)
        for n in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                  'post_feedforward_layernorm'):
            hf[pre + n + '.weight'] = r(D)
        hf[pre + 'layer_scalar'] = np.ones((1,), np.float32)
    return hf


class Gemma4Hidden(gm.Gemma4):
    """Gemma4 returning the final-norm hidden states (no cache, no KV sharing: the 31B has neither)."""

    def __call__(self, tokens, positions=None, cache=None, attention_mask=None):
        del cache
        if positions is None:
            positions = jnp.tile(jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1))
        x = self.embedder.encode(tokens)
        assert self.config.per_layer_input_dim == 0 and all(i == j for i, j in enumerate(self.kv_cache_sharing_patterns))
        for layer in self.layers:
            _, x, _ = layer(x, positions, None, attention_mask, per_layer_input=None, kv_shared_cache=None)
        return self.final_norm(x)


def load_hidden_model(model_dir, cfg, mesh, dtype=None):
    """The loader's q/k/v preprocessing uses jnp ops on the raw tensors; under the default device those would all land
    on TPU 0 (~10.8 GB for the 31B, on top of its 7.8 GB share of the sharded weights), so they run on the host CPU and
    only the final device_put (explicit NamedShardings) goes to the TPU mesh."""
    with jax.default_device(jax.devices('cpu')[0]):
        return safetensors_loader.load_and_create_model(
            file_dir=model_dir, model_class=Gemma4Hidden, config=cfg, key_mapping=gps._get_key_and_transform_mapping,
            mesh=mesh, preprocess_fn=gps._make_preprocess_fn(cfg), dtype=dtype)


def random_init_model(cfg, mesh, seed=0, scale=0.02):
    """Same abstract model and shardings as the loader, but weights are drawn on device (jit with out_shardings), so a
    canary can test compile, HBM, step time and export without reading the 62 GB checkpoint (values are meaningless)."""
    with jax.set_mesh(mesh):
        abstract = nnx.eval_shape(lambda: Gemma4Hidden(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(abstract)
    shardings = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    shapes = abs_state.to_pure_dict()
    counter = iter(range(10**6))

    def make(sd, sh):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), next(counter))
        return jax.jit(lambda k: (jax.random.normal(k, sd.shape, jnp.float32) * scale).astype(sd.dtype),
                       out_shardings=sh)(key)

    state = jax.tree.map(make, shapes, shardings)
    return nnx.merge(graph_def, state)


# fp32 master LoRA weights with bf16 compute. Qwix creates lora_a/lora_b in the base weight's dtype (bf16 here). With
# bf16 params, Adam's ~lr-sized steps are below half a bf16 ulp for most lora_a entries (he_uniform, |a| ~ 0.02 ->
# ulp 1.2e-4 vs lr 5e-5), so A would barely train; PEFT on the GPU keeps adapters in fp32 (autocast_adapter_dtype).
# The params are therefore stored in fp32 (optimizer state fp32 too) and cast to the compute dtype where Qwix reads
# them, so the forward/backward math is unchanged and gradients flow back through the cast.
_LORA_COMPUTE = {'dtype': None}
_orig_get_or_create = qwix_lora._get_or_create_lora_params


def _get_or_create_cast(**kw):
    a, b = _orig_get_or_create(**kw)
    dt = _LORA_COMPUTE['dtype']
    if dt is not None:
        a, b = a.astype(dt), b.astype(dt)
    return a, b


qwix_lora._get_or_create_lora_params = _get_or_create_cast


def lora_dtypes(model):
    return sorted({str(v.value.dtype) for p in lora_pairs(model).values() for v in p.values()})


def apply_lora(base, rank, alpha, seed=0, master_fp32=True):
    """Qwix creates the LoRA params during one model call; inside nnx.remat(graph_updates=False) those new params are
    dropped, so the init call runs with remat off and the configured remat is restored afterwards (the modules share
    the config object and read remat_config at call time). master_fp32: see _LORA_COMPUTE."""
    lp = qwix.LoraProvider(module_path=TARGETS, rank=rank, alpha=alpha)
    cfg, remat, flash = base.config, base.config.remat_config, base.config.use_flash_attention
    cfg.remat_config = gm.RematConfig.NONE
    cfg.use_flash_attention = False  # the init call uses 2-token inputs; splash needs seq % block == 0
    try:
        lm = qwix.apply_lora_to_model(base, lp, rngs=nnx.Rngs(seed), **base.get_model_input())
    finally:
        cfg.remat_config, cfg.use_flash_attention = remat, flash
    assert lm.config is cfg and lm.layers[0].config is cfg
    pairs = lora_pairs(lm)
    assert len(pairs) > 0, 'Qwix created no LoRA params'
    if master_fp32:
        for p in pairs.values():
            for v in p.values():
                if v.value.dtype != jnp.float32:
                    v.value = v.value.astype(jnp.float32)
        _LORA_COMPUTE['dtype'] = cfg.dtype
    else:
        _LORA_COMPUTE['dtype'] = None
    return lm


def lora_pairs(model):
    """{module path: {'a': LoRAParam, 'b': LoRAParam}} keyed like 'layers.3.attn.kv_einsum'."""
    out = {}
    for path, v in nnx.iter_graph(model):
        if isinstance(v, nnx.LoRAParam):
            name = str(path[-1])
            out.setdefault('.'.join(map(str, path[:-1])), {})['a' if name.endswith('lora_a') else 'b'] = v
    return out


def randomize_lora_b(model, seed, scale):
    rng = np.random.default_rng(seed)
    for p in lora_pairs(model).values():
        p['b'].value = jnp.asarray(rng.standard_normal(p['b'].value.shape) * scale, p['b'].value.dtype)


def causal_mask(lens, T):
    """(B, T, T) bool: causal and inside each row's length (only the eager path reads it; splash builds its own)."""
    i = jnp.arange(T)
    m = (i[None, :, None] >= i[None, None, :]) & (i[None, None, :] < lens[:, None, None])
    return m


def make_batch(ids, labels, lens, max_targets):
    """Targets are next tokens: position t predicts ids[t+1] when labels[t+1] != -100 (HF shift)."""
    B, T = ids.shape
    pos = np.zeros((B, max_targets), np.int32)
    tgt = np.zeros((B, max_targets), np.int32)
    w = np.zeros((B, max_targets), np.float32)
    dropped = 0
    for b in range(B):
        idx = np.nonzero(labels[b, 1:lens[b]] != -100)[0]
        dropped += max(0, len(idx) - max_targets)
        idx = idx[:max_targets]
        pos[b, :len(idx)] = idx
        tgt[b, :len(idx)] = labels[b, idx + 1]
        w[b, :len(idx)] = 1.0
    return {'tokens': jnp.asarray(ids), 'lens': jnp.asarray(lens), 'pos': jnp.asarray(pos), 'tgt': jnp.asarray(tgt),
            'w': jnp.asarray(w), 'dropped': dropped}


def target_loss(model, cfg, h, pos, tgt, w, chunk):
    """Sum of CE over gathered positions, logits chunk by chunk (each chunk rematerialised in the backward)."""
    emb = model.embedder.input_embedding.value
    hs = jnp.take_along_axis(h, pos[..., None], axis=1)  # (B, M, D)
    B, M, D = hs.shape
    n = M // chunk

    @jax.checkpoint
    def one(args):
        hc, tc, wc = args
        logits = jnp.einsum('bmd,vd->bmv', hc.astype(cfg.dtype), emb.astype(cfg.dtype)).astype(jnp.float32)
        if cfg.final_logit_softcap is not None:
            logits = jnp.tanh(logits / cfg.final_logit_softcap) * cfg.final_logit_softcap
        lse = jax.nn.logsumexp(logits, axis=-1)
        gold = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
        return ((lse - gold) * wc).sum()

    parts = jax.lax.map(one, (hs.reshape(B, n, chunk, D).swapaxes(0, 1), tgt.reshape(B, n, chunk).swapaxes(0, 1),
                              w.reshape(B, n, chunk).swapaxes(0, 1)))
    return parts.sum()


def row_ce(model, cfg, h, pos, tgt, w, chunk):
    """Per-row sums of w * CE over gathered positions, shape (B,): target_loss without the sum over rows."""
    emb = model.embedder.input_embedding.value
    hs = jnp.take_along_axis(h, pos[..., None], axis=1)
    B, M, D = hs.shape
    n = M // chunk

    @jax.checkpoint
    def one(args):
        hc, tc, wc = args
        logits = jnp.einsum('bmd,vd->bmv', hc.astype(cfg.dtype), emb.astype(cfg.dtype)).astype(jnp.float32)
        if cfg.final_logit_softcap is not None:
            logits = jnp.tanh(logits / cfg.final_logit_softcap) * cfg.final_logit_softcap
        lse = jax.nn.logsumexp(logits, axis=-1)
        gold = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
        return ((lse - gold) * wc).sum(axis=-1)

    parts = jax.lax.map(one, (hs.reshape(B, n, chunk, D).swapaxes(0, 1), tgt.reshape(B, n, chunk).swapaxes(0, 1),
                              w.reshape(B, n, chunk).swapaxes(0, 1)))
    return parts.sum(axis=0)


def make_pair_step(cfg, chunk=512, eager_mask=False):
    """In-step preference optimisation (B163, the fix of B161): one pair per batch, row 0 chosen, row 1 rejected (both
    trained on their final step only). step(model, opt, tokens, lens, pos, tgt, w, ref, coef, lr) -> (loss, m,
    log_ratio (2,), n (2,), grad_norm, applied). ref (2,): each row's mean log-prob per target token under the reference
    (the starting LoRA: B = 0, i.e. the base, or an imported adapter); m = log_ratio_c - log_ratio_r from this forward
    pass, so the preference term saturates as the pair separates (the per-epoch signed weights of G16/G17 v1 could not:
    rejected steps fell to -70 nats/token in one epoch).
    coef = [beta, sft_chosen, ipo_target, is_ipo, ipo_norm, floor]: DPO -log sigmoid(beta m) (gradient beta*sigmoid(-beta
    m), 1 at m = 0 for beta 2, as the old weights) or IPO (m - t)^2 / (2 ipo_norm) (gradient (m - t) / ipo_norm); plus
    sft_chosen x the chosen step's mean NLL (RPO, Pang et al. 2024). A rejected log-ratio already below `floor` is
    held (stop-gradient): only the chosen side moves (bounded unlikelihood; -1e9 = off). beta 0 leaves SFT on the
    chosen step only (same compute: an ablation arm)."""
    @nnx.jit(donate_argnums=(0, 1))
    def step(model, opt, tokens, lens, pos, tgt, w, ref, coef, lr):
        beta, sft, target, is_ipo, norm, floor = coef[0], coef[1], coef[2], coef[3], coef[4], coef[5]

        def loss_fn(mdl):
            mask = causal_mask(lens, tokens.shape[1]) if eager_mask else None
            h = mdl(tokens, None, None, mask)
            ce = row_ce(mdl, cfg, h, pos, tgt, w, chunk)
            n = jnp.maximum((w != 0).sum(axis=-1).astype(jnp.float32), 1.0)
            ratio = -ce / n - ref
            r_rej = jnp.where(ratio[1] > floor, ratio[1], jax.lax.stop_gradient(ratio[1]))
            m = ratio[0] - r_rej
            dpo = -jax.nn.log_sigmoid(beta * m)
            ipo = (m - target) ** 2 / (2.0 * jnp.maximum(norm, 1e-6))
            loss = jnp.where(is_ipo > 0.5, ipo, dpo) + sft * ce[0] / n[0]
            return loss, (m, ratio, n)
        (loss, (m, ratio, n)), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam),
                                                           has_aux=True)(model)
        gnorm = optax.global_norm(jax.tree.leaves(grads))
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        grads = jax.tree.map(lambda g: jnp.where(ok, g, jnp.zeros_like(g)), grads)
        opt.update(model, grads, lr=jnp.where(ok, lr, 0.0).astype(jnp.float32))
        return loss, m, ratio, n, gnorm, ok
    return step


def make_row_logp(cfg, chunk=512, eager_mask=False):
    """rows(model, tokens, lens, pos, tgt, w) -> (sum of log p over each row's targets (w != 0), target count), both
    shape (B,). Forward only (DPO margins: policy and reference log-probs per row). The model is donated, as in the
    train step: nnx.jit hands the model's state back as an output, and without donation that output is a second copy of
    the weights (G15 v2 ran out of HBM on TPU: 16.89 GB needed, 15.75 GB available, B146)."""
    @nnx.jit(donate_argnums=(0,))
    def rows(model, tokens, lens, pos, tgt, w):
        mask = causal_mask(lens, tokens.shape[1]) if eager_mask else None
        h = model(tokens, None, None, mask)
        emb = model.embedder.input_embedding.value
        hs = jnp.take_along_axis(h, pos[..., None], axis=1)
        B, M, D = hs.shape
        n = M // chunk

        def one(args):
            hc, tc, wc = args
            logits = jnp.einsum('bmd,vd->bmv', hc.astype(cfg.dtype), emb.astype(cfg.dtype)).astype(jnp.float32)
            if cfg.final_logit_softcap is not None:
                logits = jnp.tanh(logits / cfg.final_logit_softcap) * cfg.final_logit_softcap
            lse = jax.nn.logsumexp(logits, axis=-1)
            gold = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
            return ((gold - lse) * (wc != 0)).sum(axis=-1)

        parts = jax.lax.map(one, (hs.reshape(B, n, chunk, D).swapaxes(0, 1), tgt.reshape(B, n, chunk).swapaxes(0, 1),
                                  w.reshape(B, n, chunk).swapaxes(0, 1)))
        return parts.sum(axis=0), (w != 0).sum(axis=-1)
    return rows


def make_tx(clip=1.0, b1=0.9, b2=0.999, eps=1e-8):
    """AdamW without weight decay (as before) whose learning rate is a per-step argument (`opt.update(..., lr=x)`), so
    the runner can re-plan the schedule's horizon from measured step times without recompiling."""
    def init(params):
        return optax.EmptyState()

    def update(updates, state, params=None, *, lr, **extra):
        return jax.tree.map(lambda u: -lr * u, updates), state
    return optax.chain(optax.clip_by_global_norm(clip), optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
                       optax.GradientTransformationExtraArgs(init, update))


def make_train_step(cfg, chunk=512, eager_mask=False):
    """step(model, opt, tokens, lens, pos, tgt, w, lr) -> (loss, n_targets, grad_norm, applied). A step whose loss or
    gradient norm is not finite applies no update (zero gradients, lr 0), so one bad batch cannot poison the adapter."""
    @nnx.jit(donate_argnums=(0, 1))
    def step(model, opt, tokens, lens, pos, tgt, w, lr):
        def loss_fn(mdl):
            mask = causal_mask(lens, tokens.shape[1]) if eager_mask else None
            h = mdl(tokens, None, None, mask)
            s = target_loss(mdl, cfg, h, pos, tgt, w, chunk)
            # mean over targets, not over weights: identical for SFT (w in {0, 1}); with DPO's signed per-row weights
            # the weight magnitude is the gradient scale and must not be normalised away
            n = jnp.maximum((w != 0).sum().astype(jnp.float32), 1.0)
            return s / n, n
        (loss, n), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)(model)
        gnorm = optax.global_norm(jax.tree.leaves(grads))
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        grads = jax.tree.map(lambda g: jnp.where(ok, g, jnp.zeros_like(g)), grads)
        opt.update(model, grads, lr=jnp.where(ok, lr, 0.0).astype(jnp.float32))
        return loss, n, gnorm, ok
    return step


# ---- healing a channel-cut model by distillation from the full model (pruner part 3) ----
# The cut model lives inside the full weights: an FFMask (0/1 per hidden unit of each FeedForward) multiplies the
# hidden activations, so a masked unit contributes nothing, exactly as if its gate/up rows and its down_proj column
# were removed. Its LoRA rows get no gradient (B rows of gate/up stay 0, the down_proj A columns only ever meet zeros),
# so the trained adapter slices exactly to the cut checkpoint. The teacher is the same buffers with the masks set to 1
# and the LoRA factors set to 0: the original full model, no second copy of the 62 GB in HBM.
class FFMask(nnx.Variable):
    """0/1 per hidden unit of a FeedForward (1 = kept)."""


def _ff_block_masked(self, x):
    """Tunix's FeedForward.block (down(gelu(gate(x)) * up(x))) with an optional FFMask on the hidden units; the same
    ops as the original when no mask is installed."""
    a = nnx.gelu(self.gate_proj(x)) * self.up_proj(x)
    m = getattr(self, 'ff_mask', None)
    if m is not None:
        a = a * m.value.astype(a.dtype)
    return self.down_proj(a)


def install_ff_mask(model, mask, mesh=None):
    """mask (L, I) of 0/1, the hidden units kept per layer; installs one FFMask per layer (replicated) and the masked
    FeedForward.block. Returns the kept count per layer."""
    gm.FeedForward.block = _ff_block_masked
    mask = np.asarray(mask)
    assert mask.shape[0] == len(model.layers), (mask.shape, len(model.layers))
    kept = []
    for i, layer in enumerate(model.layers):
        assert mask[i].shape == (layer.mlp.gate_proj.kernel.value.shape[1],), (i, mask[i].shape)
        v = jnp.asarray(mask[i], model.config.dtype)
        if mesh is not None:
            v = jax.device_put(v, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
        layer.mlp.ff_mask = FFMask(v)
        kept.append(int(np.asarray(mask[i]).sum()))
    return kept


def teacher_view(model):
    """The full model in the same buffers: every LoRA factor 0 (no adapter) and every FFMask 1 (no cut). A functional
    copy for use inside jitted functions; the model's own params are untouched."""
    gd, lora, ffm, rest = nnx.split(model, nnx.LoRAParam, FFMask, ...)
    return nnx.merge(gd, jax.tree.map(jnp.zeros_like, lora), jax.tree.map(jnp.ones_like, ffm), rest)


def _logits(cfg, hc, emb):
    logits = jnp.einsum('bmd,vd->bmv', hc.astype(cfg.dtype), emb.astype(cfg.dtype)).astype(jnp.float32)
    if cfg.final_logit_softcap is not None:
        logits = jnp.tanh(logits / cfg.final_logit_softcap) * cfg.final_logit_softcap
    return logits


def kd_loss(model, cfg, h, h_t, pos, w, chunk, temp=1.0):
    """Sum over the gathered positions of w * KL(teacher || student) over the full vocabulary (forward KL at
    temperature temp, times temp^2), chunk by chunk (each chunk rematerialised in the backward). h_t: the teacher's
    final hidden states (no gradient); both sides use the tied embedding and Gemma's soft-cap."""
    emb = model.embedder.input_embedding.value
    hs = jnp.take_along_axis(h, pos[..., None], axis=1)
    ht = jnp.take_along_axis(h_t, pos[..., None], axis=1)
    B, M, D = hs.shape
    n = M // chunk

    @jax.checkpoint
    def one(args):
        hc, tc, wc = args
        lps = jax.nn.log_softmax(_logits(cfg, hc, emb) / temp, axis=-1)
        lpt = jax.nn.log_softmax(_logits(cfg, tc, emb) / temp, axis=-1)
        kl = (jnp.exp(lpt) * (lpt - lps)).sum(-1)
        return (kl * wc).sum() * temp * temp

    parts = jax.lax.map(one, (hs.reshape(B, n, chunk, D).swapaxes(0, 1), ht.reshape(B, n, chunk, D).swapaxes(0, 1),
                              w.reshape(B, n, chunk).swapaxes(0, 1)))
    return parts.sum()


def make_distill_step(cfg, chunk=512, eager_mask=False, temp=1.0):
    """Healing step, same call and returns as make_train_step: step(model, opt, tokens, lens, pos, tgt, w, lr) ->
    (loss, n_targets, grad_norm, applied). Teacher: teacher_view(model), forward only; student: the model as it is (cut
    by its FFMasks, plus the LoRA being trained); loss = mean over the positions with w != 0 of KL(teacher || student).
    tgt is unused (the teacher's whole distribution is the target)."""
    @nnx.jit(donate_argnums=(0, 1))
    def step(model, opt, tokens, lens, pos, tgt, w, lr):
        del tgt
        mask = causal_mask(lens, tokens.shape[1]) if eager_mask else None
        h_t = jax.lax.stop_gradient(teacher_view(model)(tokens, None, None, mask))

        def loss_fn(mdl):
            h = mdl(tokens, None, None, mask)
            s = kd_loss(mdl, cfg, h, h_t, pos, w, chunk, temp)
            n = jnp.maximum((w != 0).sum().astype(jnp.float32), 1.0)
            return s / n, n
        (loss, n), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)(model)
        gnorm = optax.global_norm(jax.tree.leaves(grads))
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        grads = jax.tree.map(lambda g: jnp.where(ok, g, jnp.zeros_like(g)), grads)
        opt.update(model, grads, lr=jnp.where(ok, lr, 0.0).astype(jnp.float32))
        return loss, n, gnorm, ok
    return step


def make_eval_rows(cfg, chunk=512, eager_mask=False, teacher=False, kl=False):
    """rows(model, tokens, lens, pos, tgt, w, w2) -> per row (B,): (sum log p over w != 0, count, sum log p over w2 != 0,
    count) of the model as it is (the cut student) or, teacher=True, of teacher_view(model) (the full model); kl=True
    (student only) appends (sum over w != 0, sum over w2 != 0) of KL(teacher || student) per position: how far the cut
    model's whole next-token distribution is from the full model's. Forward only; the model is donated (see
    make_row_logp)."""
    assert not (teacher and kl)

    @nnx.jit(donate_argnums=(0,))
    def rows(model, tokens, lens, pos, tgt, w, w2):
        mdl = teacher_view(model) if teacher else model
        mask = causal_mask(lens, tokens.shape[1]) if eager_mask else None
        h = mdl(tokens, None, None, mask)
        emb = mdl.embedder.input_embedding.value
        hs = jnp.take_along_axis(h, pos[..., None], axis=1)
        B, M, D = hs.shape
        n = M // chunk
        if kl:
            ht = jnp.take_along_axis(teacher_view(model)(tokens, None, None, mask), pos[..., None], axis=1)
        else:
            ht = hs  # unused

        def one(args):
            hc, tc_h, tc, wc, wc2 = args
            logits = _logits(cfg, hc, emb)
            lps = jax.nn.log_softmax(logits, axis=-1)
            lp = jnp.take_along_axis(lps, tc[..., None], axis=-1)[..., 0]
            out = [(lp * (wc != 0)).sum(axis=-1), (lp * (wc2 != 0)).sum(axis=-1)]
            if kl:
                lpt = jax.nn.log_softmax(_logits(cfg, tc_h, emb), axis=-1)
                d = (jnp.exp(lpt) * (lpt - lps)).sum(-1)
                out += [(d * (wc != 0)).sum(axis=-1), (d * (wc2 != 0)).sum(axis=-1)]
            return tuple(out)

        sw = lambda x: x.reshape(B, n, chunk, *x.shape[2:]).swapaxes(0, 1)  # noqa: E731
        parts = jax.lax.map(one, (sw(hs), sw(ht), sw(tgt), sw(w), sw(w2)))
        res = (parts[0].sum(axis=0), (w != 0).sum(axis=-1), parts[1].sum(axis=0), (w2 != 0).sum(axis=-1))
        if kl:
            res += (parts[2].sum(axis=0), parts[3].sum(axis=0))
        return res
    return rows


def lora_alive(model):
    """False once a failed donated step has deleted the buffers (nothing left to train or export)."""
    try:
        return not any(v.value.is_deleted() for p in lora_pairs(model).values() for v in p.values())
    except Exception:
        return False


def _layer_kind(cfg, i):
    pattern = cfg.attention_pattern or gm.GEMMA4_ATTENTION_PATTERN
    return pattern[i % len(pattern)]


def export_peft(model, cfg, rank, alpha):
    """PEFT tensors {key: np.float32 array}; delta W_hf = (alpha / rank) * B @ A, as in PEFT's LoraLayer."""
    out = {}

    def put(i, mod, A, B):
        base = f'{PEFT_PREFIX}{i}.{"mlp" if mod.endswith("_proj") and mod[:-5] in ("gate", "up", "down") else "self_attn"}.{mod}'
        out[base + '.lora_A.weight'] = np.ascontiguousarray(A, np.float32)
        out[base + '.lora_B.weight'] = np.ascontiguousarray(B, np.float32)

    for path, p in lora_pairs(model).items():
        a = np.asarray(jax.device_get(p['a'].value), np.float32)
        b = np.asarray(jax.device_get(p['b'].value), np.float32)
        parts = path.split('.')
        i, mod = int(parts[1]), parts[-1]
        r = a.shape[-1] if mod != 'attn_vec_einsum' else a.shape[-1]
        assert r == rank, (path, a.shape, b.shape)
        if mod == 'q_einsum':  # w (N, D, H); a (D, r); b (r, N, H); HF q_proj[(n, h), d]
            put(i, 'q_proj', a.T, b.reshape(rank, -1).T)
        elif mod == 'kv_einsum':  # w (2, K, D, H); a (D, r); b (r, 2, K, H)
            put(i, 'k_proj', a.T, b[:, 0].reshape(rank, -1).T)
            put(i, 'v_proj', a.T, b[:, 1].reshape(rank, -1).T)
        elif mod == 'k_einsum':  # full attention, K = V: HF has k_proj only
            put(i, 'k_proj', a.T, b.reshape(rank, -1).T)
        elif mod == 'attn_vec_einsum':  # w (N, H, D); a (N, H, r); b (r, D); HF o_proj[d, (n, h)]
            put(i, 'o_proj', a.reshape(-1, rank).T, b.T)
        elif mod in ('gate_proj', 'up_proj', 'down_proj'):  # kernel (in, out); a (in, r); b (r, out)
            put(i, mod, a.T, b.T)
        else:
            raise ValueError(path)
    return out


def import_peft(model, cfg, peft, rank):
    """Inverse of export_peft: set the model's LoRA params from PEFT tensors (an adapter this trainer exported), each
    onto its param's existing sharding and dtype. Exact for our own exports, where k_proj and v_proj of a sliding layer
    share lora_A (Qwix's kv_einsum has one `a`); refuses an adapter whose k/v lora_A differ (e.g. one trained with
    PEFT on GPU) instead of silently averaging. Returns the number of PEFT tensors used."""
    used = set()

    def get(i, mod, ab):
        k = f'{PEFT_PREFIX}{i}.{"mlp" if mod in ("gate_proj", "up_proj", "down_proj") else "self_attn"}.{mod}.lora_{ab}.weight'
        used.add(k)
        return np.asarray(peft[k], np.float32)

    def put(v, x):
        assert x.size == v.value.size, (x.shape, v.value.shape)
        v.value = jax.device_put(jnp.asarray(x.reshape(v.value.shape), v.value.dtype), v.value.sharding)

    for path, p in lora_pairs(model).items():
        parts = path.split('.')
        i, mod = int(parts[1]), parts[-1]
        assert p['a'].value.shape[-1] == rank, (path, p['a'].value.shape)
        if mod == 'kv_einsum':
            ak, av = get(i, 'k_proj', 'A'), get(i, 'v_proj', 'A')
            assert np.array_equal(ak, av), f'{path}: k_proj and v_proj lora_A differ (not an adapter of this trainer)'
            put(p['a'], ak.T)
            bk, bv = get(i, 'k_proj', 'B').T, get(i, 'v_proj', 'B').T  # (r, K*H) each
            put(p['b'], np.stack([bk.reshape(rank, *p['b'].value.shape[2:]), bv.reshape(rank, *p['b'].value.shape[2:])], 1))
            continue
        hf = {'q_einsum': 'q_proj', 'k_einsum': 'k_proj', 'attn_vec_einsum': 'o_proj'}.get(mod, mod)
        if hf not in ('q_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
            raise ValueError(path)
        put(p['a'], get(i, hf, 'A').T)
        put(p['b'], get(i, hf, 'B').T)
    extra = sorted(set(peft) - used)
    assert not extra, f'{len(extra)} adapter tensors without a LoRA param here, e.g. {extra[:3]}'
    return len(used)


def peft_config(rank, alpha):
    return {'base_model_name_or_path': 'google/gemma-4-31b-it-qat-w4a16-ct', 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
            'r': rank, 'lora_alpha': alpha, 'lora_dropout': 0.0, 'bias': 'none', 'fan_in_fan_out': False,
            'inference_mode': True, 'init_lora_weights': True, 'use_rslora': False, 'use_dora': False,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}


def merge_into_hf(hf, peft, rank, alpha):
    merged = dict(hf)
    for k, A in peft.items():
        if not k.endswith('.lora_A.weight'):
            continue
        B = peft[k.replace('lora_A', 'lora_B')]
        key = HF_PREFIX + k[len(PEFT_PREFIX):].replace('.lora_A.weight', '.weight')
        assert key in merged, key
        merged[key] = (merged[key].astype(np.float32) + (alpha / rank) * (B @ A)).astype(merged[key].dtype)
    return merged


def check_export(lora_model, cfg, peft, hf, rank, alpha, tmpdir, mesh, tokens):
    """Max |logit diff| / max |logit| between the LoRA model and the base model loaded from HF + merged PEFT deltas."""
    from safetensors.numpy import save_file
    os.makedirs(tmpdir, exist_ok=True)
    save_file(merge_into_hf(hf, peft, rank, alpha), os.path.join(tmpdir, 'model.safetensors'))
    merged = load_hidden_model(tmpdir, cfg, mesh)
    T = tokens.shape[1]
    mask = causal_mask(jnp.full((tokens.shape[0],), T), T)
    e = merged.embedder.input_embedding.value
    h1 = lora_model(tokens, None, None, mask)
    h2 = merged(tokens, None, None, mask)
    l1, l2 = h1 @ e.T, h2 @ e.T
    return float(jnp.abs(l1 - l2).max() / jnp.abs(l1).max()), float(jnp.abs(h1 - h2).max())
