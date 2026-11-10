"""CPU tests for the TPU LoRA trainer on a tiny random Gemma 4 (no TPU, no real weights).
1. lora_tpu round trip: HF-format safetensors -> Tunix loader (Gemma4Hidden) -> Qwix LoRA -> PEFT export -> merge back
   into the HF weights -> reload -> logits must match; LoRA must actually move the outputs; masked steps lower the loss.
2. fp32 master LoRA weights on a bf16 model: params and optimizer state stay fp32, the forward stays bf16, lora_a
   actually moves at lr 5e-5 (with bf16 params almost none of it does), a non-finite batch applies no update.
3. train_tpu_runner.py end to end (G8_TINY=1, eager attention, mesh 1x1): npz rows -> buckets -> train -> per-epoch
   and final exports -> comparison with a reference adapter of the same layout; random-init canary.
4. Failure handling: simulated HBM failure (bucket demotion); flash attention requested on CPU (the splash kernel
   cannot run there, so the attention ladder must step down to eager and train); time-aware re-plan (slow steps and a
   short cap: the plan is cut at an epoch boundary and the cosine reaches its floor on the last step).
   Phase 2: LoRA reset to the same init, same data order, another lr, own dirs, no recompile; skipped if short.
   DPO / IPO pairs: per-epoch signed weights (clock targets, two arms) and in-step pairs (B163: margin from the step's
   own forward pass, beta-0 ablation arm, optional rejected floor, no training on a truncated pair).
5. 8 fake CPU devices, mesh fsdp 2 x tp 4 (G8_TINY=2): sharded load, LoRA, training, phase 2 and export; DPO and
   in-step DPO (mixed with a per-epoch pair).
usage: python kernels/train/test_lora_tpu.py   (needs jax, flax, google-tunix, qwix, optax, safetensors)
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
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora_tpu as LT  # noqa: E402

cfg = LT.tiny_config()
hf = LT.tiny_hf_weights(cfg)
V = cfg.num_embed
tmp = tempfile.mkdtemp()
save_file(hf, os.path.join(tmp, 'model.safetensors'))
rng = np.random.default_rng(0)

RANK, ALPHA = 4, 8.0
mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)
with jax.set_mesh(mesh):
    base = LT.load_hidden_model(tmp, cfg, mesh)
    lm = LT.apply_lora(base, RANK, ALPHA)
    pairs = LT.lora_pairs(lm)
    print('lora modules', len(pairs), sorted({p.split('.')[-1] for p in pairs}))
    LT.randomize_lora_b(lm, seed=1, scale=0.2)
    peft = LT.export_peft(lm, cfg, RANK, ALPHA)
    print('peft tensors', len(peft))
    assert len(peft) == 2 * (5 * 7 + 6), len(peft)  # 5 sliding layers x q,k,v,o,gate,up,down + full layer without v
    assert not any('layers.5.self_attn.v_proj' in k for k in peft), 'full-attention layer must not get v_proj'
    toks = jnp.asarray(rng.integers(1, V, size=(2, 24)), jnp.int32)
    rel, hmax = LT.check_export(lm, cfg, peft, hf, RANK, ALPHA, os.path.join(tmp, 'merged'), mesh, toks)
    print(f'export check: max rel logit diff {rel:.2e}, max hidden diff {hmax:.2e}')
    base_only = LT.load_hidden_model(tmp, cfg, mesh)
    mask = LT.causal_mask(jnp.full((2,), 24), 24)
    moved = float(jnp.abs(lm(toks, None, None, mask) - base_only(toks, None, None, mask)).max())
    print(f'LoRA moved hidden states by {moved:.2e} (must be >> export diff)')
    assert rel < 1e-4 and moved > 100 * hmax, (rel, hmax, moved)
    # import_peft is export_peft's inverse (stage 2 starts from an exported adapter, e.g. G11a): a fresh LoRA model takes
    # the adapter, exports it back bit for bit and computes the same hidden states; a PEFT adapter whose k_proj and
    # v_proj lora_A differ (not one of ours: kv_einsum has one shared `a`) is refused
    lm_i = LT.apply_lora(LT.load_hidden_model(tmp, cfg, mesh), RANK, ALPHA, seed=5)
    assert LT.import_peft(lm_i, cfg, peft, RANK) == len(peft)
    back = LT.export_peft(lm_i, cfg, RANK, ALPHA)
    assert set(back) == set(peft) and all(np.array_equal(back[k], peft[k]) for k in peft)
    assert float(jnp.abs(lm_i(toks, None, None, mask) - lm(toks, None, None, mask)).max()) == 0.0
    bad = dict(peft)
    kv = next(k for k in bad if 'v_proj.lora_A' in k)
    bad[kv] = bad[kv] + 1e-3
    try:
        LT.import_peft(lm_i, cfg, bad, RANK)
        raise RuntimeError('import_peft accepted different k/v lora_A')
    except AssertionError as e:
        assert 'differ' in str(e), e
    print('import_peft round trip OK')

    ids = rng.integers(1, V, size=(2, 40)).astype(np.int32)
    labels = np.full_like(ids, -100)
    labels[0, 10:20] = ids[0, 10:20]
    labels[1, 25:30] = ids[1, 25:30]
    lens = np.array([40, 33], np.int32)
    ids[1, 33:] = 0
    batch = LT.make_batch(ids, labels, lens, max_targets=32)
    opt = nnx.Optimizer(lm, LT.make_tx(), wrt=nnx.LoRAParam)
    step = LT.make_train_step(cfg, chunk=16, eager_mask=True)
    losses = []
    for t in range(6):
        loss, n, gn, ok = step(lm, opt, batch['tokens'], batch['lens'], batch['pos'], batch['tgt'], batch['w'],
                               np.float32(3e-3))
        losses.append(float(loss))
        print(f'step {t} loss {float(loss):.4f} targets {int(n)} grad_norm {float(gn):.3e} applied {bool(ok)}')
    assert int(n) == 10 + 5, n  # every label has a previous position (HF shift: logits[t] predict labels[t + 1])
    assert losses[-1] < losses[0], losses

# ---- 2. fp32 master weights on a bf16 model ----
cfg16 = LT.tiny_config()
cfg16.param_dtype = cfg16.dtype = jnp.bfloat16
with mesh, jax.set_mesh(mesh):
    m16 = LT.apply_lora(LT.load_hidden_model(tmp, cfg16, mesh, dtype=jnp.bfloat16), RANK, ALPHA)
    assert LT.lora_dtypes(m16) == ['float32'], LT.lora_dtypes(m16)
    a0 = {k: np.asarray(p['a'].value).copy() for k, p in LT.lora_pairs(m16).items()}
    opt16 = nnx.Optimizer(m16, LT.make_tx(), wrt=nnx.LoRAParam)
    step16 = LT.make_train_step(cfg16, chunk=16, eager_mask=True)
    for t in range(3):
        loss, n, gn, ok = step16(m16, opt16, batch['tokens'], batch['lens'], batch['pos'], batch['tgt'], batch['w'],
                                 np.float32(5e-5))
        assert bool(ok)
    moved_a = np.mean([np.mean(np.asarray(p['a'].value) != a0[k]) for k, p in LT.lora_pairs(m16).items()])
    opt_dt = sorted({str(x.dtype) for x in jax.tree.leaves(nnx.state(opt16)) if hasattr(x, 'dtype') and x.ndim > 0})
    h16 = m16(batch['tokens'], None, None, LT.causal_mask(batch['lens'], 40))
    print(f'fp32 master: lora {LT.lora_dtypes(m16)}, opt state {opt_dt}, hidden {h16.dtype}, '
          f'lora_a entries moved {moved_a:.3f}')
    assert LT.lora_dtypes(m16) == ['float32'] and opt_dt == ['float32'] and h16.dtype == jnp.bfloat16
    assert moved_a > 0.9, moved_a
    before = {k: np.asarray(p['b'].value).copy() for k, p in LT.lora_pairs(m16).items()}
    loss, n, gn, ok = step16(m16, opt16, batch['tokens'], batch['lens'], batch['pos'], batch['tgt'],
                             batch['w'] * jnp.nan, np.float32(5e-5))
    same = all(np.array_equal(np.asarray(p['b'].value), before[k]) for k, p in LT.lora_pairs(m16).items())
    print(f'non-finite batch: applied {bool(ok)}, params unchanged {same}, alive {LT.lora_alive(m16)}')
    assert not bool(ok) and same and LT.lora_alive(m16)
    LT._LORA_COMPUTE['dtype'] = None  # back to the fp32 test model's setting for the reference export below

# ---- 3. the runner, as the TPU kernel calls it ----
lens = rng.integers(20, 48, size=9)
ids = [rng.integers(1, V, size=n).astype(np.int32) for n in lens]
labs = []
for x in ids:
    lab = np.full_like(x, -100)
    lab[len(x) // 2:len(x) // 2 + 6] = x[len(x) // 2:len(x) // 2 + 6]
    labs.append(lab)
npz = os.path.join(tmp, 'rows.npz')
np.savez_compressed(npz, ids=np.concatenate(ids), labels=np.concatenate(labs), lens=lens.astype(np.int32),
                    instance_ids=np.array([f't{i}' for i in range(9)]), sources=np.array(['x/pre'] * 5 + ['x/post'] * 4))
ref_dir = os.path.join(tmp, 'ref')
os.makedirs(ref_dir)
save_file({k: v.astype(np.float32) for k, v in LT.export_peft(lm, cfg, RANK, ALPHA).items()},
          os.path.join(ref_dir, 'adapter_model.safetensors'))
args = {'t0': time.time(), 'deadline_s': 1800, 'out': None, 'npz': npz, 'model': tmp, 'buckets': '32,48',
        'flash': False, 'mesh': '1,1', 'rank': RANK, 'seed': 0, 'epochs': 2, 'batch': 2,
        'mode': 'train', 'canary_steps': 1, 'lr': 3e-3, 'loss_chunk': 2, 'max_targets': 8, 'cap_min': 20,
        'save_every_min': 0.01, 'reserve_s': 10, 'adapter_out': None,
        'ref': os.path.join(ref_dir, 'adapter_model.safetensors')}


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
    done = [x for x in lines if x['stage'] == 'done'][-1]
    print(name, {k: done.get(k) for k in ('stop', 'steps_ok', 'planned', 'step_errors', 'rung', 'demoted',
                                         'truncated_label_frac_kept', 'adapter_matches_ref')})
    return root, lines, done


root, lines, done = run('train')
steps = [x for x in lines if x['stage'] == 'step']
assert len(steps) >= 8 and all(x['targets'] > 0 and x['applied'] for x in steps), steps
assert done['stop'] == 'plan_done' and done['adapter_matches_ref'] is True
assert [x for x in lines if x['stage'] == 'lora'][0]['dtypes'] == ['float32']
for sub in ('adapter_out', 'adapters/ep0', 'adapters/ep1'):
    assert os.path.exists(os.path.join(root, sub, 'adapter_model.safetensors')), sub
    assert os.path.exists(os.path.join(root, sub, 'adapter_config.json')), sub
assert not [f for f in os.listdir(os.path.join(root, 'adapter_out')) if f.endswith('.tmp')]
lrs = [x['lr'] for x in steps]
assert abs(lrs[-1] - 0.1 * args['lr']) < 1e-9 + 0.02 * args['lr'], lrs  # cosine reaches its floor on the last step

root, l2, done = run('canary', mode='canary', random_init=True)
assert [x for x in l2 if x['stage'] == 'loaded'][0]['random_init'] is True
assert any(x['stage'] == 'step' for x in l2) and not any(x['stage'] == 'step_error' for x in l2), l2[-3:]
print('random-init canary OK')

# ---- 4. failure handling ----
root, l3, done = run('demote', env={'G8_FAIL': 'L48'})
assert any(x['stage'] == 'demote' for x in l3) and done['demoted'] == [48]
s3 = [x for x in l3 if x['stage'] == 'step']
assert any(x.get('as_len') == 32 for x in s3) and done['adapter_matches_ref'] is True
assert done['truncated_label_frac_kept'] is not None
print('bucket demotion OK', len(s3), 'steps')

root, l3b, done = run('demote2', env={'G8_FAIL': 'L48,L32'}, emergency_len=16)
assert done['demoted'] == [32, 48] and done['stop'] == 'plan_done', done
assert all(x.get('as_len') == 16 for x in l3b if x['stage'] == 'step') and done['truncated_label_frac_kept'] < 1
print('emergency length OK, labels kept', done['truncated_label_frac_kept'])

root, l3c, done = run('export_fail', env={'G8_FAIL': 'export'})
assert any(x['stage'] == 'export_error' for x in l3c) and done['stop'] == 'plan_done' and done['steps_ok'] == 10, done
assert done['adapter_matches_ref'] is True  # periodic exports failed, training continued, final export written
print('export failure tolerated OK')

root, l4, done = run('ladder', flash=True, flash_block=1024)
rung_moves = [x['to'] for x in l4 if x['stage'] == 'rung']
print('ladder moves', rung_moves, [x['kind'] for x in l4 if x['stage'] == 'step_error'])
assert rung_moves == ['flash512', 'flash256', 'eager8192'] and done['rung'] == 'eager8192', rung_moves
assert done['stop'] == 'plan_done' and done['steps_ok'] >= 8 and done['adapter_matches_ref'] is True
print('attention ladder OK')

first_step_s = [x for x in lines if x['stage'] == 'step'][0]['t_min'] * 60  # load + first compile on this machine
root, l5, done = run('replan', env={'G8_TEST_SLEEP_S': '1.0'}, epochs=4, cap_min=(first_step_s + 24) / 60,
                     save_every_min=100, replan_at=4, replan_every=3, reserve_s=0, compile_allow_s=2,
                     export_est_s=0.5)
cuts = [x for x in l5 if x['stage'] == 'replan_cut']
s5 = [x for x in l5 if x['stage'] == 'step']
print('replan cuts', [(x['steps'], x['whole_epochs']) for x in cuts], 'steps run', len(s5), 'last lr', s5[-1]['lr'])
assert cuts and done['stop'] == 'plan_done', (cuts, done)
assert abs(s5[-1]['lr'] - 0.1 * args['lr']) < 0.05 * args['lr'], s5[-1]['lr']
print('time-aware re-plan OK')

# phase 2: same init and data order, another lr, separate dirs; phase a's adapter is untouched
from safetensors.numpy import load_file  # noqa: E402
root, l7, done = run('phase2', phase2={'lr': 6e-3, 'epochs': 2, 'min_min': 0})
dones = [x for x in l7 if x['stage'] == 'done']
assert [x['phase'] for x in dones] == ['a', 'b'] and all(x['stop'] == 'plan_done' for x in dones), dones
sa = [x for x in l7 if x['stage'] == 'step' and x['phase'] == 'a']
sb = [x for x in l7 if x['stage'] == 'step' and x['phase'] == 'b']
assert sa[0]['loss'] == sb[0]['loss'] and sa[0]['k'] == sb[0]['k'] == 0, (sa[0], sb[0])  # identical init + batch
assert abs(max(x['lr'] for x in sb) - 6e-3) < 1e-9 and abs(max(x['lr'] for x in sa) - 3e-3) < 1e-9
first_b = {}
for x in sb:
    first_b.setdefault(x['bucket'], x['s'])
first_a = {}
for x in sa:
    first_a.setdefault(x['bucket'], x['s'])
print('first step seconds per length, a (compiles):', first_a, 'b:', first_b)
assert all(first_b[L] < 0.2 * first_a[L] for L in first_b), 'phase b recompiled'
p2 = [x for x in l7 if x['stage'] == 'phase2_start'][0]
assert p2['b_absmax_after_reset'] == 0.0 and p2['opt_absmax_after_reset'] == 0.0, p2
for sub in ('adapter_out', 'adapters/ep1', 'adapter_out_b', 'adapters_b/ep0', 'adapters_b/ep1'):
    assert os.path.exists(os.path.join(root, sub, 'adapter_model.safetensors')), sub
fa = [x for x in l7 if x['stage'] == 'export' and x['tag'] == 'a:final'][0]
fb = [x for x in l7 if x['stage'] == 'export' and x['tag'] == 'b:final'][0]
got = max(float(np.abs(v).max()) for k, v in load_file(os.path.join(root, 'adapter_out', 'adapter_model.safetensors')).items()
          if 'lora_B' in k)
assert abs(got - fa['b_absmax']) < 1e-12 and fa['b_absmax'] != fb['b_absmax'], (got, fa['b_absmax'], fb['b_absmax'])
print(f"phase 2 OK: same first loss {sa[0]['loss']}, final |B| a {fa['b_absmax']:.4f} vs b {fb['b_absmax']:.4f}")
root, l8, done = run('phase2skip', phase2={'lr': 6e-3, 'epochs': 2, 'min_min': 600})
assert any(x['stage'] == 'phase2_skipped' for x in l8) and not os.path.exists(os.path.join(root, 'adapter_out_b'))
print('phase 2 skipped when time is short OK')
# phase 2 on a subset: only the rows whose source tag contains '/post' (G10: post-compaction segments)
root, l9, done = run('phase2sub', phase2={'lr': 6e-3, 'epochs': 1, 'min_min': 0, 'source_contains': '/post'})
p9 = [x for x in l9 if x['stage'] == 'phase2_start'][0]
plans = {x['phase']: x for x in l9 if x['stage'] == 'plan'}
assert p9['rows'] == 4 and p9['source_contains'] == '/post', p9
assert 0 < plans['b']['steps'] < plans['a']['steps'] / plans['a']['epochs'], plans
assert [x['stop'] for x in l9 if x['stage'] == 'done'] == ['plan_done', 'plan_done']
assert os.path.exists(os.path.join(root, 'adapter_out_b', 'adapter_model.safetensors'))
dat = [x for x in l9 if x['stage'] == 'data'][0]
assert dat['by_source'] == {'x/pre': 5, 'x/post': 4}, dat
root, l10, done = run('phase2none', phase2={'lr': 6e-3, 'epochs': 1, 'min_min': 0, 'source_contains': 'nothing'})
assert any(x['stage'] == 'phase2_skipped' and 'nothing' in x['reason'] for x in l10)
assert not os.path.exists(os.path.join(root, 'adapter_out_b'))
print(f"phase 2 on a source subset OK: {p9['rows']} rows, {plans['b']['steps']} steps vs {plans['a']['steps']} in a")

# ---- 4b. DPO: pair rows (sources 'dpo:<pair>:c|r') get signed weights from the per-pair margin each epoch ----
P = 6
dl = rng.integers(20, 30, size=2 * P + 2)
d_ids = [rng.integers(1, V, size=n).astype(np.int32) for n in dl]
d_labs = []
for x in d_ids:
    lab = np.full_like(x, -100)
    lab[len(x) // 2:len(x) // 2 + 6] = x[len(x) // 2:len(x) // 2 + 6]
    d_labs.append(lab)
d_src = [f'dpo:p{k}~{"speed" if k % 2 else "fail"}:{side}' for k in range(P) for side in 'cr'] + ['dpo:lone~fail:r', 'x/post']
npz_d = os.path.join(tmp, 'rows_dpo.npz')
np.savez_compressed(npz_d, ids=np.concatenate(d_ids), labels=np.concatenate(d_labs), lens=dl.astype(np.int32),
                    instance_ids=np.array([f'd{i}' for i in range(len(dl))]), sources=np.array(d_src))
root, ld, done = run('dpo', npz=npz_d, buckets='32', epochs=3, batch=2,
                     dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0})
info = [x for x in ld if x['stage'] == 'dpo'][0]
refr = [x for x in ld if x['stage'] == 'dpo_refresh']
print('dpo', info, [(x['tag'], x['margin_p10_50_90'], x['acc'], x['weight_mean']) for x in refr])
assert info['pairs'] == P and info['rows'] == 2 * P
assert info['lone_rejected_rows'] == 1, info  # a rejected row without its chosen side gets weight 0, not SFT (B157)
assert [x['tag'] for x in refr] == ['start', 'ep1', 'ep2'], refr
assert refr[0]['margin_p10_50_90'] == [0.0, 0.0, 0.0] and abs(refr[0]['weight_mean'] - 1.0) < 1e-6
assert refr[-1]['margin_p10_50_90'][1] > 0 and refr[-1]['acc'] >= 0.8, refr[-1]
assert refr[-1]['weight_mean'] < refr[0]['weight_mean']  # separated pairs are down-weighted
assert set(refr[-1]['by_kind']) == {'fail', 'speed'} and refr[-1]['by_kind']['speed']['n'] == P // 2, refr[-1]['by_kind']
assert done['stop'] == 'plan_done' and all(x['applied'] for x in ld if x['stage'] == 'step')
print('DPO OK')
root, li, done = run('ipo', npz=npz_d, buckets='32', epochs=4, batch=2,
                     dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0, 'loss_speed': 'ipo', 'ipo_target_speed': 0.004})
ri = [x for x in li if x['stage'] == 'dpo_refresh']
print('ipo', [(x['tag'], x['by_kind']) for x in ri])
sp, fa = ri[-1]['by_kind']['speed'], ri[-1]['by_kind']['fail']
assert done['stop'] == 'plan_done' and sp['margin_p50'] > 0, ri[-1]
assert sp['margin_p50'] < fa['margin_p50'], (sp, fa)  # IPO holds speed pairs near the target, DPO keeps pushing fail pairs
print('IPO OK')
# clock-calibrated IPO (B154): per-pair targets from the two runs' estimated serving times (pairs.json)
meta_p = os.path.join(tmp, 'pairs_meta.json')
json.dump({f'p{k}~{"speed" if k % 2 else "fail"}': {'c': {'est_s': 100.0}, 'r': {'est_s': 100.0 * (1.5 + k)}} for k in range(P)},
          open(meta_p, 'w'))
root, lc, done = run('ipo_clock', npz=npz_d, buckets='32', epochs=4, batch=2,
                     dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0, 'loss_speed': 'ipo',
                          'ipo_target_speed': 'clock', 'ipo_tau_speed': 0.004, 'ipo_lo_speed': 0.001, 'ipo_hi_speed': 0.02,
                          'pairs_meta': meta_p})
info_c = [x for x in lc if x['stage'] == 'dpo'][0]
rc_ = [x for x in lc if x['stage'] == 'dpo_refresh']
print('ipo_clock', info_c['ipo_targets'], [(x['tag'], x['ipo']) for x in rc_])
t10, t50, t90 = info_c['ipo_targets']['speed']
assert info_c['meta_pairs'] == P and t90 > t10 > 0, info_c  # three speed pairs, three different targets
assert done['stop'] == 'plan_done' and rc_[-1]['ipo']['speed']['target_p10_50_90'][2] > rc_[-1]['ipo']['speed']['target_p10_50_90'][0]
print('clock-calibrated IPO OK')
# two preference arms in one run (B155): arm a = clock targets with the kind's mean target as the weight scale (pairs
# with a larger time gap start with a larger weight), arm b = one uniform target (G15's recipe), same reference, same steps
root, l2a, done = run('ipo_2arm', npz=npz_d, buckets='32', epochs=3, batch=2,
                      dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0, 'loss_speed': 'ipo',
                           'ipo_target_speed': 'clock', 'ipo_tau_speed': 0.004, 'ipo_lo_speed': 0.001, 'ipo_hi_speed': 0.02,
                           'ipo_norm_speed': 'mean', 'pairs_meta': meta_p},
                      phase2={'lr': args['lr'], 'epochs': 3, 'min_min': 0, 'a_share': 0.5, 'match_a': True,
                              'dpo': {'ipo_target_speed': 0.006, 'ipo_norm_speed': 'self'}})
infos = [x for x in l2a if x['stage'] == 'dpo']
r2 = [x for x in l2a if x['stage'] == 'dpo_refresh']
plans2 = {x['phase']: x for x in l2a if x['stage'] == 'plan'}
dones2 = {x['phase']: x for x in l2a if x['stage'] == 'done'}
print('ipo_2arm', [(x['arm'], x['ipo_targets'], x['ipo_norm']) for x in infos],
      [(x['tag'], x['ipo']['speed']['weight_p10_50_90']) for x in r2])
assert [x['arm'] for x in infos] == ['a', 'b'] and isinstance(infos[0]['ipo_norm']['speed'], float), infos
assert infos[1]['ipo_norm']['speed'] == 'self' and infos[1]['ipo_targets']['speed'] == [0.006] * 3, infos[1]
assert [x['tag'] for x in r2] == ['start', 'ep1', 'ep2'] * 2, [x['tag'] for x in r2]
wa, wb = r2[0]['ipo']['speed']['weight_p10_50_90'], r2[3]['ipo']['speed']['weight_p10_50_90']
assert wa[2] > 1.0 > wa[0] and abs(sum(wa) / 3 - 1.0) < 0.35, wa  # arm a: weights spread by time gap around 1
assert wb == [1.0, 1.0, 1.0], wb  # arm b: every pair starts at 1, as G15
assert r2[3]['margin_p10_50_90'] == [0.0, 0.0, 0.0]  # arm b restarts from the reference
assert plans2['b']['steps'] == dones2['a']['steps_ok'] and dones2['b']['stop'] == 'plan_done', (plans2, dones2)
assert os.path.exists(os.path.join(root, 'adapter_out_b', 'adapter_model.safetensors'))
print('two-arm IPO OK')

# ---- 4b'. in-step DPO (B163): each pair is one batch (row 0 chosen, row 1 rejected) and its preference term comes from
# that step's own forward pass, so it saturates as the pair separates (the per-epoch weights of G16/G17 v1 could not
# react within an epoch). Arm b: beta 0, i.e. SFT on the chosen steps only, same steps: the ablation ----
root, lis, done = run('dpo_instep', npz=npz_d, buckets='32', epochs=3, batch=2,
                      dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0, 'in_step': True},
                      phase2={'lr': args['lr'], 'epochs': 3, 'min_min': 0, 'a_share': 0.5, 'match_a': True,
                              'dpo': {'beta': 0.0}})
info_s = [x for x in lis if x['stage'] == 'dpo']
pe = {(x['phase'], x['ep']): x for x in lis if x['stage'] == 'pair_epoch'}
ps = [x for x in lis if x['stage'] == 'step' and 'pair' in x]
print('dpo_instep', [(k_, x['margin_p10_50_90'], x['acc'], x['chosen_ratio'][1], x['rejected_ratio'][1]) for k_, x in pe.items()])
assert info_s[0]['in_step_pairs'] == P and info_s[0]['lone_rejected_rows'] == 1, info_s[0]
assert [x['beta'] for x in info_s] == [2.0, 0.0], info_s
assert not any(x['stage'] == 'dpo_refresh' for x in lis)  # every pair is in-step: no per-epoch weights
assert sorted(pe) == [(ph, e) for ph in 'ab' for e in range(3)] and all(x['pairs'] == P for x in pe.values()), sorted(pe)
assert len([x for x in ps if x['phase'] == 'a']) == 3 * P and all(x['applied'] for x in ps), len(ps)
assert all(x['targets'] == 12 for x in ps), ps[:2]  # 6 trained tokens per row, both rows of the pair
a2, b2 = pe[('a', 2)], pe[('b', 2)]
assert a2['margin_p10_50_90'][1] > 0 and a2['acc'] >= 0.8 and a2['rejected_ratio'][1] < 0, a2
assert b2['chosen_ratio'][1] > 0 and b2['rejected_ratio'][1] > a2['rejected_ratio'][1], (a2, b2)  # beta 0: no push down
assert all(x['loss'] > 0.69 for x in ps if x['phase'] == 'b')  # -log sigmoid(0) = ln 2 plus the chosen NLL
print('in-step DPO OK')
# the optional in-step floor: a rejected step already below it is held (stop-gradient), only the chosen side moves
root, lf, done = run('dpo_instep_floor', npz=npz_d, buckets='32', epochs=3, batch=2,
                     dpo={'beta': 2.0, 'sft_chosen': 1.0, 'in_step': True, 'instep_floor': -0.001})
pf = [x for x in lf if x['stage'] == 'pair_epoch']
print('dpo_instep_floor', [(x['ep'], x['margin_p10_50_90'], x['chosen_ratio'][1], x['rejected_ratio'][1]) for x in pf])
assert done['stop'] == 'plan_done' and pf[-1]['rejected_ratio'][1] > a2['rejected_ratio'][1], (pf[-1], a2)
print('in-step floor OK')
# a pair is trained whole or not at all: with length 48 demoted (simulated HBM failure) a pair with a row longer than 32
# tokens is skipped, not trained on a truncated row (the trained step is the end of the row, as in the teacher rows)
zl = np.array([24, 26, 40, 44, 25, 23, 41, 38])  # pairs z0, z2 short; z1, z3 long
z_ids = [rng.integers(1, V, size=n).astype(np.int32) for n in zl]
z_labs = []
for x in z_ids:
    lab = np.full_like(x, -100)
    lab[-6:] = x[-6:]
    z_labs.append(lab)
npz_z = os.path.join(tmp, 'rows_dpo_z.npz')
np.savez_compressed(npz_z, ids=np.concatenate(z_ids), labels=np.concatenate(z_labs), lens=zl.astype(np.int32),
                    instance_ids=np.array([f'z{i}' for i in range(len(zl))]),
                    sources=np.array([f'dpo:z{k}~teacher:{side}' for k in range(4) for side in 'cr']))
root, lz, done = run('dpo_instep_demote', env={'G8_FAIL': 'L48'}, npz=npz_z, buckets='32,48', epochs=2, batch=2,
                     dpo={'beta': 2.0, 'sft_chosen': 1.0, 'in_step': True})
sk = [x for x in lz if x['stage'] == 'skip' and 'pair' in x]
pz = [x for x in lz if x['stage'] == 'pair_epoch']
print('dpo_instep_demote', [(x['ep'], x['pair']) for x in sk], [(x['ep'], x['pairs']) for x in pz])
assert done['demoted'] == [48] and sorted(x['pair'] for x in sk) == ['z1~teacher', 'z1~teacher', 'z3~teacher', 'z3~teacher']
assert [x['pairs'] for x in pz] == [2, 2] and done['stop'] == 'plan_done', (pz, done)
print('in-step pair truncation guard OK')
# stage 2 from an exported adapter (init_adapter: G17 v2 starts from G11a): the reference is the imported adapter (margins
# start at 0), and arm b restarts from the same imported adapter, not from B = 0
init_ad = os.path.join(ref_dir, 'adapter_model.safetensors')
root, lia, done = run('dpo_instep_init', npz=npz_d, buckets='32', epochs=2, batch=2, init_adapter=init_ad,
                      dpo={'beta': 2.0, 'sft_chosen': 1.0, 'in_step': True},
                      phase2={'lr': args['lr'], 'epochs': 2, 'min_min': 0, 'a_share': 0.5, 'match_a': True, 'dpo': {'beta': 0.0}})
ia = [x for x in lia if x['stage'] == 'init_adapter'][0]
p2s = [x for x in lia if x['stage'] == 'phase2_start'][0]
pia = {(x['phase'], x['ep']): x for x in lia if x['stage'] == 'pair_epoch'}
print('dpo_instep_init', ia, p2s['b_absmax_after_reset'], pia[('a', 0)]['margin_p10_50_90'], pia[('a', 1)]['margin_p10_50_90'])
assert ia['roundtrip_max_abs_diff'] == 0.0 and ia['tensors'] == len(peft) and ia['b_absmax'] > 0.1, ia
assert abs(p2s['b_absmax_after_reset'] - ia['b_absmax']) < 1e-6, (p2s, ia)  # arm b starts from the imported adapter
assert abs(pia[('a', 0)]['margin_p10_50_90'][0]) < 0.01 and pia[('a', 1)]['margin_p10_50_90'][1] > 0, pia
assert done['stop'] == 'plan_done' and [x['phase'] for x in lia if x['stage'] == 'done'] == ['a', 'b']
print('stage 2 from an exported adapter OK')

# ---- 4c. the forward-only row log-prob pass must not hold a second copy of the weights: nnx.jit returns the model's
# state, and without donation that output was a full copy (G15 v2 ran out of HBM on the TPU, B146) ----
with mesh, jax.set_mesh(mesh):
    m4 = LT.apply_lora(LT.load_hidden_model(tmp, cfg, mesh), RANK, ALPHA)
    i4 = rng.integers(1, V, size=(2, 32)).astype(np.int32)
    l4 = np.full((2, 32), -100, np.int32)
    l4[:, 10:20] = i4[:, 10:20]
    b4 = LT.make_batch(i4, l4, np.array([32, 30], np.int32), 8)
    f4 = LT.make_row_logp(cfg, chunk=4, eager_mask=True)
    ma = f4.lower(m4, b4['tokens'], b4['lens'], b4['pos'], b4['tgt'], b4['w']).compile().memory_analysis()
    print('row_logp memory: argument', ma.argument_size_in_bytes, 'output', ma.output_size_in_bytes, 'alias', ma.alias_size_in_bytes)
    assert ma.alias_size_in_bytes >= 0.99 * ma.argument_size_in_bytes, ma
    r_a = f4(m4, b4['tokens'], b4['lens'], b4['pos'], b4['tgt'], b4['w'])
    r_b = f4(m4, b4['tokens'], b4['lens'], b4['pos'], b4['tgt'], b4['w'])
    assert bool(jnp.all(r_a[0] == r_b[0])), (r_a, r_b)  # the donated model stays usable and unchanged
print('row_logp donation OK')

# ---- 5. 8 fake CPU devices, fsdp 2 x tp 4 ----
cfg8 = LT.tiny_config(mesh_test=True)
tmp8 = os.path.join(tmp, 'm8')
os.makedirs(tmp8)
save_file(LT.tiny_hf_weights(cfg8), os.path.join(tmp8, 'model.safetensors'))
with jax.set_mesh(mesh):
    ref8 = LT.export_peft(LT.apply_lora(LT.load_hidden_model(tmp8, cfg8, mesh), RANK, ALPHA), cfg8, RANK, ALPHA)
os.makedirs(os.path.join(tmp, 'ref8'))
save_file(ref8, os.path.join(tmp, 'ref8', 'adapter_model.safetensors'))
env8 = {'G8_TINY': '2', 'XLA_FLAGS': '--xla_force_host_platform_device_count=8', 'JAX_PLATFORMS': 'cpu'}
root, l6, done = run('mesh8', env=env8, model=tmp8, mesh='2,4', ref=os.path.join(tmp, 'ref8', 'adapter_model.safetensors'),
                     phase2={'lr': 6e-3, 'epochs': 1, 'min_min': 0})
assert [x['phase'] for x in l6 if x['stage'] == 'done'] == ['a', 'b']
lora6 = [x for x in l6 if x['stage'] == 'lora'][0]
print('mesh8 devices', [x for x in l6 if x['stage'] == 'jax'][0]['n'], 'lora example', lora6['example'])
assert [x for x in l6 if x['stage'] == 'jax'][0]['n'] == 8 and lora6['dtypes'] == ['float32']
da = [x for x in l6 if x['stage'] == 'done'][0]
assert da['stop'] == 'plan_done' and da['adapter_matches_ref'] is True and da['steps_ok'] >= 8, da
assert done['stop'] == 'plan_done' and done['adapter_matches_ref'] is True and done['steps_ok'] >= 4, done
print('8-device mesh OK')

# ---- 5b. DPO on the 8-device mesh: the reference pass shards its batch over fsdp 2 (G15 v1 died on batches of 1,
# B141); two length buckets and an odd number of pair rows (pair 0 has two rejected rows) ----
P8 = 6
m_src = [f'dpo:q{k}~{"speed" if k % 2 else "fail"}:{side}' for k in range(P8) for side in 'cr'] + ['dpo:q0~fail:r', 'x/pre', 'x/post']
ml = np.concatenate([rng.integers(20, 30, size=8), rng.integers(34, 46, size=len(m_src) - 8)])
m_ids = [rng.integers(1, cfg8.num_embed, size=n).astype(np.int32) for n in ml]
m_labs = []
for x in m_ids:
    lab = np.full_like(x, -100)
    lab[len(x) // 2:len(x) // 2 + 6] = x[len(x) // 2:len(x) // 2 + 6]
    m_labs.append(lab)
npz_m = os.path.join(tmp, 'rows_dpo8.npz')
np.savez_compressed(npz_m, ids=np.concatenate(m_ids), labels=np.concatenate(m_labs), lens=ml.astype(np.int32),
                    instance_ids=np.array([f'm{i}' for i in range(len(ml))]), sources=np.array(m_src))
root, l7, done = run('dpo8', env=env8, model=tmp8, mesh='2,4', ref=os.path.join(tmp, 'ref8', 'adapter_model.safetensors'),
                     npz=npz_m, buckets='32,48', epochs=3, batch=2, dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0})
r7 = [x for x in l7 if x['stage'] == 'dpo_refresh']
print('dpo8', [(x['tag'], x['margin_p10_50_90'], x['acc']) for x in r7])
assert [x for x in l7 if x['stage'] == 'dpo'][0]['rows'] == 2 * P8 + 1
assert [x['tag'] for x in r7] == ['start', 'ep1', 'ep2'] and r7[0]['margin_p10_50_90'] == [0.0, 0.0, 0.0], r7
assert done['stop'] == 'plan_done' and r7[-1]['margin_p10_50_90'][1] > 0, r7[-1]
print('DPO 8-device mesh OK')
# in-step DPO on the 8-device mesh, mixed: q0 (two rejected rows) stays on per-epoch weights, q1-q5 are in-step pairs; a
# pair batch has 2 rows (sharded over fsdp 2) at the longer row's length
root, l8s, done = run('dpo8_instep', env=env8, model=tmp8, mesh='2,4', ref=os.path.join(tmp, 'ref8', 'adapter_model.safetensors'),
                      npz=npz_m, buckets='32,48', epochs=3, batch=2,
                      dpo={'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -5.0, 'in_step': True})
i8 = [x for x in l8s if x['stage'] == 'dpo'][0]
r8 = [x for x in l8s if x['stage'] == 'dpo_refresh']
p8 = [x for x in l8s if x['stage'] == 'pair_epoch']
print('dpo8_instep', [(x['ep'], x['margin_p10_50_90'], x['acc']) for x in p8], [(x['tag'], x['pairs']) for x in r8])
assert i8['pairs'] == P8 and i8['in_step_pairs'] == P8 - 1, i8
assert [x['tag'] for x in r8] == ['start', 'ep1', 'ep2'] and all(x['pairs'] == 1 for x in r8), r8
assert [x['ep'] for x in p8] == [0, 1, 2] and all(x['pairs'] == P8 - 1 for x in p8), p8
assert {x['bucket'] for x in l8s if x['stage'] == 'step' and 'pair' in x} == {32, 48}
assert done['stop'] == 'plan_done' and p8[-1]['margin_p10_50_90'][1] > 0 and p8[-1]['acc'] >= 0.8, p8[-1]
print('in-step DPO 8-device mesh OK')

print('OK')
