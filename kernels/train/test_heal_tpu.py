"""CPU tests of healing a cut model (pruner part 3) in the TPU trainer, on a tiny random Gemma 4 (no TPU, no real
weights).
1. lora_tpu pieces: an FFMask of all ones changes nothing; the masked student (LoRA B = 0) equals the model with the
   removed units' down_proj columns zeroed (an explicit cut); teacher_view equals the base model whatever the LoRA and
   the masks; the distillation step lowers the KL and never moves the teacher; the masked units' gate/up lora_B rows
   stay exactly 0 (so the adapter slices exactly to the cut checkpoint); the held-out eval's KL is 0 for the teacher
   against itself and shrinks with training.
2. train_tpu_runner.py end to end with 'heal' (1 device, and 8 fake devices as fsdp 2 x tp 4): arm a KL to the uncut
   model at every position, arm b the next-token loss on exactly the same batches; held-out evals of teacher and
   student (own/all tokens, KL) at the start, per epoch and after each arm.
usage: python kernels/train/test_heal_tpu.py   (needs jax, flax, google-tunix, qwix, optax, safetensors)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora_tpu as LT  # noqa: E402

RANK, ALPHA = 4, 8.0
tmp = tempfile.mkdtemp()
mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def tiny_weights(cfg_h):
    """tiny_hf_weights with peaked next-token distributions (embedding x30; at 0.05 they are ~uniform) and MLP outputs
    that weigh in the residual stream (post-feedforward norm gain 40; Gemma normalises each MLP output, so scaling the
    MLP weights alone changes nothing): cutting half the hidden units then moves the predictions by ~0.06 nats of KL,
    like the real c70 cut, instead of ~1e-5."""
    hw = LT.tiny_hf_weights(cfg_h)
    hw['model.language_model.embed_tokens.weight'] *= 30
    for k_ in list(hw):
        if k_.endswith('post_feedforward_layernorm.weight'):
            hw[k_] = np.full_like(hw[k_], 40.0)
    return hw


# ---- 1. the pieces ----
cfg = LT.tiny_config()
hf = tiny_weights(cfg)
V, L, I = cfg.num_embed, cfg.num_layers, cfg.hidden_dim
d0 = os.path.join(tmp, 'base')
os.makedirs(d0)
save_file(hf, os.path.join(d0, 'model.safetensors'))
rng = np.random.default_rng(0)
mask = np.zeros((L, I), np.float32)
for i in range(L):
    mask[i, rng.permutation(I)[:I // 2]] = 1
with jax.set_mesh(mesh):
    base = LT.load_hidden_model(d0, cfg, mesh)
    toks = jnp.asarray(rng.integers(1, V, size=(2, 32)), jnp.int32)
    cm = LT.causal_mask(jnp.full((2,), 32), 32)
    h_base = base(toks, None, None, cm)
    lm = LT.apply_lora(base, RANK, ALPHA)
    LT.randomize_lora_b(lm, seed=1, scale=0.2)
    h_lora = lm(toks, None, None, cm)
    LT.install_ff_mask(lm, np.ones((L, I), np.float32), mesh)
    assert float(jnp.abs(lm(toks, None, None, cm) - h_lora).max()) == 0.0, 'an all-ones mask must change nothing'
    kept = LT.install_ff_mask(lm, mask, mesh)
    assert kept == [I // 2] * L, kept
    d_t = float(jnp.abs(LT.teacher_view(lm)(toks, None, None, cm) - h_base).max())
    assert d_t == 0.0, d_t
    hf_cut = dict(hf)
    for i in range(L):
        k = f'model.language_model.layers.{i}.mlp.down_proj.weight'
        hf_cut[k] = hf[k] * mask[i][None, :]
    d1 = os.path.join(tmp, 'cut')
    os.makedirs(d1)
    save_file(hf_cut, os.path.join(d1, 'model.safetensors'))
    cut = LT.load_hidden_model(d1, cfg, mesh)
    for p in LT.lora_pairs(lm).values():
        p['b'].value = jnp.zeros_like(p['b'].value)
    d_c = float(jnp.abs(lm(toks, None, None, cm) - cut(toks, None, None, cm)).max())
    assert d_c == 0.0, d_c
    print('mask, teacher view, explicit cut: exact')

    ids = np.asarray(rng.integers(1, V, size=(2, 32)), np.int32)
    labels = ids.copy()
    lens = np.array([32, 27], np.int32)
    ids[1, 27:] = 0
    labels[1, 27:] = -100
    bt = LT.make_batch(ids, labels, lens, max_targets=32)
    own = np.asarray(bt['w']).copy()
    own[:, ::2] = 0
    opt = nnx.Optimizer(lm, LT.make_tx(), wrt=nnx.LoRAParam)
    step = LT.make_distill_step(cfg, chunk=16, eager_mask=True)
    ev_s = LT.make_eval_rows(cfg, chunk=16, eager_mask=True, kl=True)
    ev_t = LT.make_eval_rows(cfg, chunk=16, eager_mask=True, teacher=True)

    def ev(f):
        r = f(lm, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'], jnp.asarray(own), bt['w'])
        out = [float(-r[0].sum() / r[1].sum()), float(-r[2].sum() / r[3].sum())]
        if len(r) > 4:
            out += [float(r[4].sum() / r[1].sum()), float(r[5].sum() / r[3].sum())]
        return out
    t0, s0 = ev(ev_t), ev(ev_s)
    losses = []
    for t in range(8):
        loss, n, gn, ok = step(lm, opt, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'], bt['w'], np.float32(3e-3))
        assert bool(ok)
        losses.append(float(loss))
    t1, s1 = ev(ev_t), ev(ev_s)
    print('distill losses', [f'{x:.2e}' for x in losses], 'student KL own/all before', s0[2:], 'after', s1[2:])
    assert losses[-1] < losses[0] and abs(losses[0] - s0[3]) < 1e-7 + 0.02 * losses[0], (losses[0], s0)  # step 0 = eval KL
    assert t1 == t0, 'the teacher moved'
    assert s0[3] > 0 and s1[3] < s0[3] and s1[2] < s0[2], (s0, s1)
    for i, layer in enumerate(lm.layers):
        assert np.array_equal(np.asarray(layer.mlp.ff_mask.value), mask[i])
    peft = LT.export_peft(lm, cfg, RANK, ALPHA)
    for i in range(L):
        for mod in ('gate_proj', 'up_proj'):
            B = peft[f'{LT.PEFT_PREFIX}{i}.mlp.{mod}.lora_B.weight']
            assert np.abs(B[mask[i] == 0]).max() == 0.0 and np.abs(B[mask[i] == 1]).max() > 0, (i, mod)
    # the teacher's KL to itself is 0: a student whose mask is all ones and LoRA zero
    lm2 = LT.apply_lora(LT.load_hidden_model(d0, cfg, mesh), RANK, ALPHA)
    LT.install_ff_mask(lm2, np.ones((L, I), np.float32), mesh)
    r2 = LT.make_eval_rows(cfg, chunk=16, eager_mask=True, kl=True)(lm2, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'],
                                                                     jnp.asarray(own), bt['w'])
    assert float(jnp.abs(r2[5]).max()) < 1e-6, r2[5]
    print('distillation step and eval OK')

# ---- 2. the runner end to end ----
ref_dir = os.path.join(tmp, 'ref')
os.makedirs(ref_dir)
with jax.set_mesh(mesh):
    save_file(LT.export_peft(LT.apply_lora(LT.load_hidden_model(d0, cfg, mesh), RANK, ALPHA), cfg, RANK, ALPHA),
              os.path.join(ref_dir, 'adapter_model.safetensors'))
args = {'t0': time.time(), 'deadline_s': 1800, 'out': None, 'npz': None, 'model': None, 'buckets': '32,48',
        'flash': False, 'mesh': '1,1', 'rank': RANK, 'seed': 0, 'epochs': 2, 'batch': 2,
        'mode': 'train', 'canary_steps': 1, 'lr': 3e-3, 'loss_chunk': 2, 'max_targets': 8, 'cap_min': 20,
        'save_every_min': 0.01, 'reserve_s': 10, 'adapter_out': None, 'ref': os.path.join(ref_dir, 'adapter_model.safetensors')}


def run(name, env=None, **over):
    a = dict(args, t0=time.time(), **over)
    root = os.path.join(tmp, name)
    a['out'] = os.path.join(root, 'tpu')
    a['adapter_out'] = os.path.join(root, 'adapter_out')
    os.makedirs(a['out'])
    p = subprocess.run([sys.executable, os.path.join(HERE, 'train_tpu_runner.py'), json.dumps(a)],
                       env={**os.environ, 'G8_TINY': '1', **(env or {})}, capture_output=True, text=True)
    lines = [json.loads(x) for x in open(os.path.join(a['out'], 'log.jsonl')) if x.strip()]
    assert p.returncode == 0, (name, p.stderr[-3000:])
    return root, lines


def heal_files(cfg_h, tag, n_rows=9, seed=3, with_calls=False):
    r = np.random.default_rng(seed)
    d = os.path.join(tmp, f'heal_{tag}')
    os.makedirs(d)
    save_file(tiny_weights(cfg_h), os.path.join(d, 'model.safetensors'))
    m = np.zeros((cfg_h.num_layers, cfg_h.hidden_dim), np.uint8)
    for i in range(cfg_h.num_layers):
        m[i, r.permutation(cfg_h.hidden_dim)[:cfg_h.hidden_dim // 2]] = 1
    np.savez(os.path.join(d, 'mask.npz'), mask=m, keep=np.int32(cfg_h.hidden_dim // 2), sha=np.array('test'))
    ln = r.integers(20, 48, size=n_rows)
    ids_h = [r.integers(1, cfg_h.num_embed, size=n).astype(np.int32) for n in ln]
    np.savez(os.path.join(d, 'rows.npz'), ids=np.concatenate(ids_h), labels=np.concatenate(ids_h),
             lens=ln.astype(np.int32), sources=np.array([f'h{i}' for i in range(n_rows)]))
    el = r.integers(12, 32, size=5)
    eids = [r.integers(1, cfg_h.num_embed, size=n).astype(np.int32) for n in el]
    ow = [r.random(n) < 0.3 for n in el]
    cw = [r.random(n) < 0.2 for n in el]   # B192: tool-call bodies, a third view (only when with_calls)
    np.savez(os.path.join(d, 'eval.npz'), ids=np.concatenate(eids), labels=np.concatenate(eids), own=np.concatenate(ow),
             lens=el.astype(np.int32), instance_ids=np.array([f'e{i}' for i in range(5)]),
             **({'calls': np.concatenate(cw)} if with_calls else {}))
    n_calls = int(sum(int(c[1:].sum()) for c in cw)) if with_calls else None
    return d, m, int(sum(int(o[1:].sum()) for o in ow)), int(sum(n - 1 for n in el)), n_calls


def check_heal(name, cfg_h, d, m, n_own, n_all, n_calls=None, **over):
    heal = {'mask': os.path.join(d, 'mask.npz'), 'kd': True, 'temp': 1.0, 'eval_npz': os.path.join(d, 'eval.npz'),
            'eval_len': 32}
    root_h, lh = run(name, model=d, npz=os.path.join(d, 'rows.npz'), heal=heal, epochs=2,
                     phase2={'lr': 3e-3, 'epochs': 2, 'min_min': 0, 'match_a': True, 'heal': {'kd': False}}, **over)
    hm_ = [x for x in lh if x['stage'] == 'heal_mask'][0]
    assert hm_['kept_min'] == hm_['kept_max'] == cfg_h.hidden_dim // 2 and hm_['kd'] is True, hm_
    ev = {x['tag']: x for x in lh if x['stage'] == 'heal_eval'}
    assert set(ev) == {'start', 'a:ep0', 'a:final', 'b:ep0', 'b:final'}, sorted(ev)
    assert not [x for x in lh if x['stage'] == 'heal_eval_error'], [x for x in lh if x['stage'] == 'heal_eval_error']
    t0_, s0_ = ev['start']['teacher'], ev['start']['student']
    assert t0_['own_tokens'] == s0_['own_tokens'] == n_own and t0_['all_tokens'] == n_all, (t0_, n_own, n_all)
    assert 'kl_all' not in t0_ and s0_['kl_all'] > 0, (t0_, s0_)
    if n_calls is None:   # eval rows without a calls mask: no calls view (old eval.npz files)
        assert 'calls_tokens' not in s0_ and 'kl_calls' not in s0_, s0_
    else:                 # the calls view: its own token count, nats for both views, KL for the student
        assert t0_['calls_tokens'] == s0_['calls_tokens'] == n_calls, (t0_, s0_, n_calls)
        assert 'kl_calls' not in t0_ and s0_['kl_calls'] > 0 and 'calls_nats' in t0_, (t0_, s0_)
        assert all('kl_calls' in ev[t]['student'] for t in ev), ev
    sa = [x for x in lh if x['stage'] == 'step' and x['phase'] == 'a']
    sb = [x for x in lh if x['stage'] == 'step' and x['phase'] == 'b']
    assert [x for x in lh if x['stage'] == 'heal_arm'][0]['kd'] is False
    assert all(x['applied'] for x in sa + sb) and len(sa) == len(sb) >= 8, (len(sa), len(sb))
    assert all(x['targets'] == y['targets'] for x, y in zip(sa, sb))  # same batches, every position a target
    assert max(x['loss'] for x in sa) < 0.5 < min(x['loss'] for x in sb), ([x['loss'] for x in sa], [x['loss'] for x in sb])
    kl = {t: ev[t]['student']['kl_all'] for t in ('start', 'a:ep0', 'a:final', 'b:final')}
    print(name, 'KD loss first/last', sa[0]['loss'], sa[-1]['loss'], 'held-out KL(full || cut), all tokens', kl,
          'own nats', {t: ev[t]['student']['own_nats'] for t in ('start', 'a:final', 'b:final')}, 'teacher', t0_['own_nats'])
    assert np.mean([x['loss'] for x in sa[-3:]]) < np.mean([x['loss'] for x in sa[:3]]), [x['loss'] for x in sa]
    assert kl['a:final'] < kl['start'], kl  # distillation moved the cut model toward the full model
    for sub in ('adapter_out', 'adapter_out_b'):
        pe = load_file(os.path.join(root_h, sub, 'adapter_model.safetensors'))
        for i in range(cfg_h.num_layers):
            for mod in ('gate_proj', 'up_proj'):
                B_ = pe[f'{LT.PEFT_PREFIX}{i}.mlp.{mod}.lora_B.weight']
                assert np.abs(B_[m[i] == 0]).max() == 0.0 and np.abs(B_[m[i] == 1]).max() > 0, (sub, i, mod)
    return lh


d1_, m1, no1, na1, nc1 = heal_files(cfg, '1', with_calls=True)
check_heal('heal', cfg, d1_, m1, no1, na1, n_calls=nc1)
print('healing runner OK')

cfg8 = LT.tiny_config(mesh_test=True)
ref8 = os.path.join(tmp, 'ref8')
os.makedirs(ref8)
d8b = os.path.join(tmp, 'base8')
os.makedirs(d8b)
save_file(tiny_weights(cfg8), os.path.join(d8b, 'model.safetensors'))
with jax.set_mesh(mesh):
    save_file(LT.export_peft(LT.apply_lora(LT.load_hidden_model(d8b, cfg8, mesh), RANK, ALPHA), cfg8, RANK, ALPHA),
              os.path.join(ref8, 'adapter_model.safetensors'))
env8 = {'G8_TINY': '2', 'XLA_FLAGS': '--xla_force_host_platform_device_count=8', 'JAX_PLATFORMS': 'cpu'}
d8, m8, no8, na8, _ = heal_files(cfg8, '8')
check_heal('heal8', cfg8, d8, m8, no8, na8, env=env8, mesh='2,4', ref=os.path.join(ref8, 'adapter_model.safetensors'))
print('healing 8-device mesh OK')
print('OK')
