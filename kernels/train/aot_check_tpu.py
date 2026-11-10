"""Trace-time canary for the G8 TPU trainer, on CPU: the REAL Gemma 4 31B config (60 layers, 32 heads, 16/4 KV heads,
head_dim 256/512, vocab 262144), Qwix LoRA r16 with fp32 master weights, the optimizer and the exact train step of
lora_tpu.make_train_step, traced with abstract (ShapeDtypeStruct) weights on 8 fake CPU devices in the fsdp 2 x tp 4
mesh, and lowered for the TPU platform. Nothing of the 62 GB is allocated.

This catches what killed canary v2 (a trace-time mesh/sharding error) and every other shape/sharding/API error in the
model, Qwix, the splash-attention kernel construction (block sizes vs 8192/12288, GQA ratios, head_dim 512, masks),
the loss and the optimizer, for each attention rung of the runner's ladder. It cannot see Mosaic's VMEM limit or HBM
(XLA:TPU compile) -- the runner's ladder/demotion handle those at run time.
--heal (pruner part 3): an FFMask cut of 70% of every layer's hidden units; traces and lowers the distillation step
(teacher_view inside the jit, KL at every position: max targets = the length) and the held-out eval (student and
teacher views) instead of the SFT step.
usage: python kernels/train/aot_check_tpu.py [--lens 8192,12288] [--rungs flash1024,flash512,flash256,eager8192] [--heal]
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_tpu as LT  # noqa: E402
from tunix.models.gemma4 import model as gm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--lens', default='8192,12288')
ap.add_argument('--rungs', default='flash1024,flash512,flash256,eager8192')
ap.add_argument('--batch', type=int, default=2)
ap.add_argument('--max-targets', type=int, default=4096)
ap.add_argument('--loss-chunk', type=int, default=512)
ap.add_argument('--layers', type=int, default=0, help='0 = all 60; fewer for a quick run')
ap.add_argument('--no-lower', action='store_true', help='trace only')
ap.add_argument('--heal', action='store_true', help='distillation step + held-out eval of a 70%% channel cut')
ap.add_argument('--eval-len', type=int, default=8192)
ap.add_argument('--out', default='')
ap.add_argument('--no-legacy-mesh', action='store_true',
                help='negative control: only jax.set_mesh (canary v2 bug); the flash rungs must then fail')
a = ap.parse_args()

t0 = time.time()
res = {'devices': len(jax.devices()), 'checks': []}
cfg = gm.ModelConfig.gemma4_31b()
if a.layers:
    cfg.num_layers = a.layers
cfg.param_dtype = cfg.dtype = jnp.bfloat16
cfg.remat_config = gm.RematConfig.DECODER
mesh = jax.make_mesh((2, 4), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def make():
    base = LT.Gemma4Hidden(cfg, rngs=nnx.Rngs(params=0))
    lm = LT.apply_lora(base, 16, 32.0)
    opt = nnx.Optimizer(lm, LT.make_tx(), wrt=nnx.LoRAParam)
    return lm, opt


import contextlib  # noqa: E402
with (contextlib.nullcontext() if a.no_legacy_mesh else mesh), jax.set_mesh(mesh):
    cfg.use_flash_attention = True
    lm, opt = nnx.eval_shape(make)
    pairs = LT.lora_pairs(lm)
    state = nnx.state((lm, opt))
    shard = nnx.get_named_sharding(state, mesh)

    def sds(x, s):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s)
    nnx.update((lm, opt), jax.tree.map(sds, state, shard))
    n_lora = sum(int(np.prod(v.value.shape)) for p in pairs.values() for v in p.values())
    lora_specs = {}
    for k, p in list(pairs.items())[:7]:
        lora_specs[k] = {n: [list(v.value.shape), str(v.value.dtype), str(v.value.sharding.spec)] for n, v in p.items()}
    base_bytes = sum(int(np.prod(v.shape)) * v.dtype.itemsize for v in jax.tree.leaves(nnx.state(lm, nnx.Param)))
    res.update({'layers': cfg.num_layers, 'lora_modules': len(pairs), 'lora_params': n_lora,
                'lora_dtypes': LT.lora_dtypes(lm), 'base_gb': round(base_bytes / 2**30, 2), 'lora_examples': lora_specs,
                'setup_s': round(time.time() - t0, 1)})
    print(json.dumps({k: v for k, v in res.items() if k != 'lora_examples'}), flush=True)
    assert res['lora_dtypes'] == ['float32'], res['lora_dtypes']
    if not a.layers:
        # G7's PEFT adapter has 122,429,440 params; Qwix shares one lora_a between k and v in the 50 sliding layers
        # (Tunix fuses them into kv_einsum) and the export duplicates it into PEFT's k_proj/v_proj: 50 x 5376 x 16 fewer
        assert n_lora == 122_429_440 - 50 * 5376 * 16, n_lora

    B, M = a.batch, a.max_targets
    if a.heal:
        I = cfg.hidden_dim
        hm = np.zeros((cfg.num_layers, I), np.float32)
        keep = int(round(0.7 * I / 512)) * 512
        for i in range(cfg.num_layers):
            hm[i, np.random.default_rng(i).permutation(I)[:keep]] = 1
        kept = LT.install_ff_mask(lm, hm, mesh)
        res['heal'] = {'kept_per_layer': kept[0], 'units': I}
        print(json.dumps(res['heal']), flush=True)
    for L in [int(x) for x in a.lens.split(',')] + ([-a.eval_len] if a.heal else []):
        for rung in a.rungs.split(','):
            flash = rung.startswith('flash')
            if not flash and rung.startswith('eager') and L > int(rung[5:] or 10**9):
                continue  # the eager rung caps length at 8192
            cfg.use_flash_attention = flash
            if flash:
                cfg.flash_attention_block_size = int(rung[5:])
            if a.heal and L < 0:  # held-out eval at -L tokens, both views, every position gathered
                EL = -L
                fns = [('eval_student', LT.make_eval_rows(cfg, chunk=a.loss_chunk, eager_mask=not flash, kl=True)),
                       ('eval_teacher', LT.make_eval_rows(cfg, chunk=a.loss_chunk, eager_mask=not flash, teacher=True))]
                ev_args = (lm, jax.ShapeDtypeStruct((B, EL), jnp.int32), jax.ShapeDtypeStruct((B,), jnp.int32),
                           jax.ShapeDtypeStruct((B, EL), jnp.int32), jax.ShapeDtypeStruct((B, EL), jnp.int32),
                           jax.ShapeDtypeStruct((B, EL), jnp.float32), jax.ShapeDtypeStruct((B, EL), jnp.float32))
                for kind, fn in fns:
                    r = {'len': EL, 'rung': rung, 'kind': kind}
                    t1 = time.time()
                    try:
                        tr = fn.trace(*ev_args)
                        r['trace_s'] = round(time.time() - t1, 1)
                        r['out'] = [f'{o.shape}:{o.dtype}' for o in jax.tree.leaves(tr.out_info)[-6:]]
                        if not a.no_lower:
                            t2 = time.time()
                            txt = tr.lower(lowering_platforms=('tpu',)).as_text()
                            r.update(lower_s=round(time.time() - t2, 1), hlo_chars=len(txt),
                                     mosaic_calls=txt.count('tpu_custom_call'))
                        r['ok'] = True
                    except Exception as e:
                        import traceback
                        r.update(ok=False, error=f'{type(e).__name__}: {str(e)[:800]}', tb=traceback.format_exc()[-1500:])
                    res['checks'].append(r)
                    print(json.dumps({k: v for k, v in r.items() if k != 'tb'}), flush=True)
                continue
            Mx = L if a.heal else M  # healing: every position of the row is a target
            step = (LT.make_distill_step(cfg, chunk=a.loss_chunk, eager_mask=not flash) if a.heal
                    else LT.make_train_step(cfg, chunk=a.loss_chunk, eager_mask=not flash))
            args = (lm, opt, jax.ShapeDtypeStruct((B, L), jnp.int32), jax.ShapeDtypeStruct((B,), jnp.int32),
                    jax.ShapeDtypeStruct((B, Mx), jnp.int32), jax.ShapeDtypeStruct((B, Mx), jnp.int32),
                    jax.ShapeDtypeStruct((B, Mx), jnp.float32), jax.ShapeDtypeStruct((), jnp.float32))
            r = {'len': L, 'rung': rung, **({'kind': 'distill'} if a.heal else {})}
            t1 = time.time()
            try:
                tr = step.trace(*args)
                r['trace_s'] = round(time.time() - t1, 1)
                outs = jax.tree.leaves(tr.out_info)
                r['out'] = [f'{o.shape}:{o.dtype}' for o in outs[-4:]]
                if not a.no_lower:
                    t2 = time.time()
                    low = tr.lower(lowering_platforms=('tpu',))
                    txt = low.as_text()
                    r['lower_s'] = round(time.time() - t2, 1)
                    r['hlo_chars'] = len(txt)
                    r['mosaic_calls'] = txt.count('tpu_custom_call')
                r['ok'] = True
            except Exception as e:
                import traceback
                r['ok'] = False
                r['error'] = f'{type(e).__name__}: {str(e)[:800]}'
                r['tb'] = traceback.format_exc()[-1500:]
            res['checks'].append(r)
            print(json.dumps({k: v for k, v in r.items() if k != 'tb'}), flush=True)
res['total_s'] = round(time.time() - t0, 1)
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
print('ALL OK' if all(c['ok'] for c in res['checks']) else 'FAILURES', res['total_s'], 's')
