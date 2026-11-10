"""Cut a Gemma 4 checkpoint down to the parts the agent uses (pruner part 2).

In plain words: part 1 (prune_score.py) measured, for every layer, how much each of the 21,504 feed-forward channels
matters on our agent's own runs, and how much each whole layer changes what flows through it. This file takes those
scores and writes a genuinely smaller model to disk:

* every layer keeps the same number of feed-forward channels (the ones that scored highest *in that layer*), because
  the model's config (and vLLM, our serving engine) allow one width for all layers;
* chosen whole layers are removed. Gemma 4 multiplies everything that flows out of a layer by a per-layer number
  (`layer_scalar`); a removed layer would still have applied that number, so it is folded into the kept layer before
  it. That makes the cut exact: the smaller model computes the same thing as the full model with those parts
  switched off (which is what part 1 measured).

The result is a normal Hugging Face checkpoint (config + safetensors + tokenizer) that vLLM and transformers load
without changes. It is written shard by shard, so memory stays at about one shard.

The competition serves a 4-bit copy of the model (compressed-tensors "w4a16": every 32 consecutive inputs of a weight
row share one scale, and 8 four-bit numbers are packed into one 32-bit integer). Cutting rows (gate/up) is exact in
that format too; cutting the *inputs* of down_proj is exact only if whole blocks of 32 channels go, so the plan can
be made block-aligned (`group=32`) and the same cut then applies, bit for bit, to the 4-bit checkpoint.

Block-aligned choice costs quality (part 1, B168: at 70% kept, the best blocks hold 89% of the importance that the
best single channels hold 97% of). The alternative: choose single channels and re-quantize only down_proj (unpack,
take the chosen inputs, quantize again in new blocks of 32, each block's scale chosen to minimise the error weighted
by how strongly the agent drives each channel). gate/up stay exact either way.
"""
import glob
import json
import os
import re
import shutil
import zlib

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r'^(model\.language_model\.layers\.)(\d+)\.(.+)$')


def make_plan(scores, keep, drop=(), score_key='agent_ff', multiple=512, group=1):
    """What to keep. scores: mapping with an (L, I) array under score_key (prune_score's scores.npz).
    keep: fraction of channels per layer (rounded to a multiple of `multiple`) or an exact count. 512 keeps every
    GPU's share (4-way tensor parallel) a multiple of 128, which the 4-bit kernels need.
    drop: layer indices to remove (never the first layer: there is no kept layer before it to take its scale; never
    the last).
    group: 1 = keep the best single channels; 32 = keep the best blocks of 32 consecutive channels (block score = sum
    of its channel scores), which the 4-bit checkpoint can be cut by exactly."""
    S = np.asarray(scores[score_key], dtype=np.float64)
    L, I = S.shape
    multiple = min(multiple, I)
    k = int(keep) if isinstance(keep, (int, np.integer)) else int(round(keep * I / multiple)) * multiple
    k = max(multiple, min(I, k))
    assert I % group == 0 and k % group == 0, (I, k, group)
    drop = sorted({int(x) for x in drop})
    assert all(0 < d < L - 1 for d in drop), f'cannot drop {drop}: only layers 1..{L - 2}'
    kept = [i for i in range(L) if i not in drop]
    if group == 1:
        ff_idx = {i: np.sort(np.argsort(-S[i], kind='stable')[:k]) for i in kept}
    else:
        G = S.reshape(L, I // group, group).sum(-1)
        ff_idx = {i: (np.sort(np.argsort(-G[i], kind='stable')[:k // group])[:, None] * group
                      + np.arange(group)[None]).ravel() for i in kept}
    fold_into = {d: max(i for i in kept if i < d) for d in drop}
    return {'layers': L, 'ff_channels': I, 'keep_ff': k, 'drop': drop, 'kept_layers': kept, 'fold_into': fold_into,
            'ff_idx': ff_idx, 'score_key': score_key, 'group': group}


def plan_masks(plan):
    """The same cut as switches for prune_score's hooks: (L, I) channel mask (dropped layers all ones) and drop set."""
    m = torch.ones(plan['layers'], plan['ff_channels'])
    for i, idx in plan['ff_idx'].items():
        m[i] = 0
        m[i, torch.as_tensor(idx)] = 1
    return m, set(plan['drop'])


def save_plan(plan, path):
    """Plan without the channel lists (JSON) + the channel lists (npz next to it)."""
    meta = {k: v for k, v in plan.items() if k != 'ff_idx'}
    meta['fold_into'] = {str(k): v for k, v in plan['fold_into'].items()}
    json.dump(meta, open(path, 'w'), indent=1)
    np.savez_compressed(os.path.splitext(path)[0] + '_channels.npz',
                        **{f'layer_{i}': np.asarray(v, dtype=np.int32) for i, v in plan['ff_idx'].items()})


PEFT_LAYERS = 'base_model.model.model.language_model.layers.'


def slice_adapter(src_dir, dst_dir, plan, prefix=PEFT_LAYERS):
    """A PEFT LoRA adapter of the full model -> the same adapter for the cut model of `plan`: gate_proj/up_proj lora_B
    keep the rows of the kept channels, down_proj lora_A the columns; the tensors of removed layers go and later layers
    are renumbered as slice_checkpoint renumbers them. Exact: every kept unit gets the same LoRA terms as in the masked
    full model, and a removed unit contributes nothing either way. Returns a summary, including the largest |lora_B|
    on removed gate/up rows (0 for an adapter trained on this cut: those rows never get a gradient)."""
    from safetensors.numpy import load_file, save_file as save_np
    t = load_file(os.path.join(src_dir, 'adapter_model.safetensors'))
    new_id = {old: new for new, old in enumerate(plan['kept_layers'])}
    I = plan['ff_channels']
    out, gone, removed_b = {}, 0, 0.0
    for k, v in t.items():
        m = re.match(re.escape(prefix) + r'(\d+)\.(.*)$', k)
        if not m:
            out[k] = v
            continue
        i, rest = int(m.group(1)), m.group(2)
        if i not in new_id:
            gone += 1
            continue
        idx = np.asarray(plan['ff_idx'][i])
        if rest in ('mlp.gate_proj.lora_B.weight', 'mlp.up_proj.lora_B.weight'):
            assert v.shape[0] == I, (k, v.shape)
            keep = np.zeros(I, bool)
            keep[idx] = True
            removed_b = max(removed_b, float(np.abs(v[~keep]).max()) if (~keep).any() else 0.0)
            v = v[idx]
        elif rest == 'mlp.down_proj.lora_A.weight':
            assert v.shape[1] == I, (k, v.shape)
            v = v[:, idx]
        out[f'{prefix}{new_id[i]}.{rest}'] = np.ascontiguousarray(v)
    os.makedirs(dst_dir, exist_ok=True)
    save_np(out, os.path.join(dst_dir, 'adapter_model.safetensors'))
    shutil.copy(os.path.join(src_dir, 'adapter_config.json'), os.path.join(dst_dir, 'adapter_config.json'))
    return {'tensors': len(out), 'tensors_of_removed_layers': gone, 'removed_rows_lora_b_absmax': removed_b}


def _unpack4(packed, cols):
    """compressed-tensors pack-quantized, 4 bits: int32 (rows, cols / 8) -> int8 (rows, cols) in [-8, 7]."""
    v = packed.to(torch.int64) & 0xFFFFFFFF
    out = torch.stack([(v >> (4 * i)) & 0xF for i in range(8)], -1).reshape(packed.shape[0], -1)[:, :cols]
    return (out - 8).to(torch.int8)


def _pack4(q):
    """int8 (rows, cols) in [-8, 7] -> int32 (rows, cols / 8), the same layout (value i of 8 in bits 4i..4i+3)."""
    v = (q.to(torch.int64) + 8).reshape(q.shape[0], -1, 8)
    p = (v << (4 * torch.arange(8, device=q.device))).sum(-1)
    return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32)


def requant_columns(packed, scale, cols, group, act=None, dev='cpu'):
    """Keep the input columns `cols` (any, not whole blocks) of a 4-bit weight: dequantize, select, quantize again in
    blocks of `group`. Per row and block, the scale is the best of a few candidates (least squared error, each column
    weighted by act**2 if given: the typical size of what flows into it). Returns (packed, scale, relative error of the
    kept weights)."""
    rows, n_in = packed.shape[0], packed.shape[1] * 8
    cols = cols.to(dev)
    q = _unpack4(packed.to(dev), n_in)
    s_col = scale.to(dev).float().repeat_interleave(group, 1).index_select(1, cols)   # each kept column's old scale
    W = (q.float().index_select(1, cols) * s_col)
    k = W.shape[1]
    G = W.view(rows, k // group, group)
    S_old = s_col.view(rows, k // group, group)
    wgt = (torch.ones(k, device=dev) if act is None else torch.as_tensor(act, dtype=torch.float32, device=dev)[cols] ** 2
           ).view(1, k // group, group)
    amax = G.abs().amax(-1)
    # candidate scales: the old scales of the columns in the block (exact for the columns that had it), and absmax/7
    # shrunk a little (clipping the largest value can lower the total error)
    cands = [S_old.amax(-1), S_old.amin(-1)] + [amax * m / 7 for m in (1.0, 0.95, 0.9, 0.875, 0.85, 0.8)]
    best_e = best_s = best_q = None
    for c in cands:
        sc = c.to(scale.dtype).float().clamp(min=1e-12)
        qq = torch.round(G / sc[..., None]).clamp(-8, 7)
        e = (((qq * sc[..., None] - G) ** 2) * wgt).sum(-1)
        if best_e is None:
            best_e, best_s, best_q = e, sc, qq
        else:
            b = e < best_e
            best_e, best_s, best_q = torch.where(b, e, best_e), torch.where(b, sc, best_s), torch.where(b[..., None], qq, best_q)
    rel = float(((best_q * best_s[..., None] - G) ** 2).sum().sqrt() / G.pow(2).sum().sqrt().clamp(min=1e-12))
    return _pack4(best_q.reshape(rows, k).to(torch.int8)).cpu(), best_s.to(scale.dtype).cpu(), rel


def _source_files(src):
    idx = os.path.join(src, 'model.safetensors.index.json')
    if os.path.exists(idx):
        wm = json.load(open(idx))['weight_map']
        return sorted(set(wm.values()))
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(src, '*.safetensors')))]


def _quant_group(cfg):
    """Block size of a compressed-tensors 4-bit checkpoint (None for a plain 16-bit one)."""
    q = cfg.get('quantization_config') or {}
    if not q:
        return None
    assert q.get('quant_method') == 'compressed-tensors' and q.get('format') == 'pack-quantized', q.get('format')
    (grp,) = q['config_groups'].values()
    w = grp['weights']
    assert w['num_bits'] == 4 and w['strategy'] == 'group' and w['symmetric'], w
    return int(w['group_size'])


def slice_checkpoint(src, dst, plan, shard_gb=4.0, log=print, act=None, dev=None):
    """Write the cut model to dst (16-bit or compressed-tensors 4-bit). Returns a small summary.
    4-bit checkpoint + a plan in whole blocks: exact. 4-bit + single channels: down_proj is re-quantized (act: optional
    (layers, channels) typical activation size, to weight the re-quantization error)."""
    os.makedirs(dst, exist_ok=True)
    files = _source_files(src)
    cfg = json.load(open(os.path.join(src, 'config.json')))
    qgroup = _quant_group(cfg)
    requant = bool(qgroup) and plan.get('group', 1) % qgroup != 0
    where = {}
    for fn in files:
        with safe_open(os.path.join(src, fn), 'pt') as f:
            where.update({k: fn for k in f.keys()})
    pending, rq_err = {}, []
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')     # where down_proj is re-quantized
    if qgroup and not requant:
        blocks = {i: torch.as_tensor(np.asarray(v)[::qgroup] // qgroup, dtype=torch.long) for i, v in plan['ff_idx'].items()}
        per_int = 8                                   # 4-bit numbers per packed 32-bit integer
        packed_cols = {i: (b[:, None] * (qgroup // per_int) + torch.arange(qgroup // per_int)[None]).reshape(-1)
                       for i, b in blocks.items()}
    kept, drop = plan['kept_layers'], set(plan['drop'])
    new_id = {old: new for new, old in enumerate(kept)}
    ff_idx = {i: torch.as_tensor(np.asarray(v), dtype=torch.long) for i, v in plan['ff_idx'].items()}

    # 1. the per-layer scale factors (tiny): kept layers absorb the scales of the removed layers after them
    scal = {}
    for fn in files:
        with safe_open(os.path.join(src, fn), 'pt') as f:
            for k in f.keys():
                m = LAYER_RE.match(k)
                if m and m.group(3) == 'layer_scalar':
                    scal[int(m.group(2))] = f.get_tensor(k)
    assert len(scal) == plan['layers'], (len(scal), plan['layers'])
    new_scal = {}
    for i in kept:
        v = scal[i].double()
        for d, into in plan['fold_into'].items():
            if into == i:
                v = v * scal[d].double()
        new_scal[i] = v.to(scal[i].dtype)

    # 2. stream every tensor: rename layers, cut the feed-forward channels, skip removed layers
    out, out_bytes, n_shard, weight_map, total = {}, 0, 0, {}, 0
    tmp_names = []

    def flush():
        nonlocal out, out_bytes, n_shard
        if not out:
            return
        n_shard += 1
        name = f'model-{n_shard:05d}.safetensors'
        save_file(out, os.path.join(dst, name), metadata={'format': 'pt'})
        for k in out:
            weight_map[k] = name
        tmp_names.append(name)
        log(f'  wrote {name}: {len(out)} tensors, {out_bytes / 2**30:.2f} GB')
        out, out_bytes = {}, 0

    for fn in files:
        with safe_open(os.path.join(src, fn), 'pt') as f:
            for k in f.keys():
                m = LAYER_RE.match(k)
                if m:
                    li, rest = int(m.group(2)), m.group(3)
                    if li in drop:
                        continue
                    proj, _, part = rest.rpartition('.')
                    if rest == 'layer_scalar':
                        t = new_scal[li].clone()
                    elif proj in ('mlp.gate_proj', 'mlp.up_proj') and part in ('weight', 'weight_packed', 'weight_scale'):
                        t = f.get_tensor(k).index_select(0, ff_idx[li]).contiguous()      # rows = channels
                    elif proj == 'mlp.down_proj' and part == 'weight':
                        t = f.get_tensor(k).index_select(1, ff_idx[li]).contiguous()      # inputs = channels
                    elif requant and proj == 'mlp.down_proj' and part in ('weight_packed', 'weight_scale'):
                        if li not in pending:                  # both parts at once, the other one is kept for later
                            base = f'{m.group(1)}{li}.mlp.down_proj.'

                            def get(name):
                                with safe_open(os.path.join(src, where[base + name]), 'pt') as g:
                                    return g.get_tensor(base + name)
                            pk, sc, err = requant_columns(get('weight_packed'), get('weight_scale'), ff_idx[li], qgroup,
                                                          act=None if act is None else act[li], dev=dev)
                            pending[li] = {'weight_packed': pk, 'weight_scale': sc}
                            rq_err.append(err)
                        t = pending[li].pop(part)
                        if not pending[li]:
                            del pending[li]
                    elif proj == 'mlp.down_proj' and part == 'weight_packed':
                        t = f.get_tensor(k).index_select(1, packed_cols[li]).contiguous()  # whole packed blocks
                    elif proj == 'mlp.down_proj' and part == 'weight_scale':
                        t = f.get_tensor(k).index_select(1, blocks[li]).contiguous()       # one scale per block
                    elif proj.startswith('mlp.') and part == 'weight_shape':
                        t = f.get_tensor(k).clone()
                        t[0 if proj != 'mlp.down_proj' else 1] = plan['keep_ff']
                    else:
                        assert not (proj.startswith('mlp.') and part not in ('weight',)), f'unexpected tensor {k}'
                        t = f.get_tensor(k)
                    name = f'{m.group(1)}{new_id[li]}.{rest}'
                else:
                    t, name = f.get_tensor(k), k
                out[name] = t
                nb = t.numel() * t.element_size()
                out_bytes += nb
                total += nb
                if out_bytes >= shard_gb * 2**30:
                    flush()
    flush()
    # final names model-0000k-of-0000n (the usual layout)
    n = len(tmp_names)
    for k, name in list(weight_map.items()):
        weight_map[k] = name.replace('.safetensors', f'-of-{n:05d}.safetensors')
    for name in tmp_names:
        os.replace(os.path.join(dst, name), os.path.join(dst, name.replace('.safetensors', f'-of-{n:05d}.safetensors')))
    json.dump({'metadata': {'total_size': total}, 'weight_map': dict(sorted(weight_map.items()))},
              open(os.path.join(dst, 'model.safetensors.index.json'), 'w'), indent=1)

    # 3. config: fewer layers, one narrower feed-forward width, the remaining layer types in order
    t = cfg.get('text_config', cfg)
    types = t.get('layer_types')
    assert types and len(types) == plan['layers'], 'config needs an explicit layer_types list'
    assert not t.get('num_kv_shared_layers') and not t.get('enable_moe_block') and not t.get('use_double_wide_mlp')
    t['layer_types'] = [types[i] for i in kept]
    t['num_hidden_layers'] = len(kept)
    t['intermediate_size'] = plan['keep_ff']
    json.dump(cfg, open(os.path.join(dst, 'config.json'), 'w'), indent=2)
    for p in glob.glob(os.path.join(src, '*')):
        b = os.path.basename(p)
        if os.path.isfile(p) and not b.endswith('.safetensors') and b not in ('config.json', 'model.safetensors.index.json'):
            shutil.copy(p, os.path.join(dst, b))
    save_plan(plan, os.path.join(dst, 'prune_plan.json'))
    assert not pending, 'down_proj packed weights and scales must come in pairs'
    summary = {'shards': n, 'bytes': total, 'gb': round(total / 2**30, 2), 'layers': len(kept),
               'ff_channels': plan['keep_ff'], 'dropped': plan['drop'],
               'format': (f'w4 g{qgroup}' + (', down_proj re-quantized' if requant else '')) if qgroup else '16-bit'}
    if requant:
        summary['requant_rel_err'] = {'mean': round(float(np.mean(rq_err)), 5), 'max': round(float(np.max(rq_err)), 5)}
    log('sliced', summary)
    return summary


def merge_lora_w4(src, dst, adapter_dir, mode='stochastic', seed=0, prefix=PEFT_LAYERS, log=print):
    """Merge a PEFT LoRA adapter (already cut to this checkpoint's shapes, see slice_adapter) into the weights of a
    compressed-tensors 4-bit checkpoint, so that vLLM serves the healed model without its LoRA path (which costs the
    cut's speed, B190). Writes a full checkpoint to dst. 16-bit weights, if any, are merged exactly.

    Why not round to nearest: the checkpoint's weights sit on their 4-bit grid (quantization-aware training) and a
    healing delta is a few hundredths of a grid step, so rounding W + dW to the nearest grid value gives back W exactly.
    mode 'stochastic' keeps every group's scale (the trained grid) and rounds W + dW up or down at random with the
    probabilities that make the result unbiased (E[merged] = W + dW): a weight moves one step with probability
    |dW| / step, so a small share of the values change. mode 'nearest' is the control: the delta is lost.
    Returns a summary: matrices merged, share of 4-bit values that changed, mean |dW| / step."""
    import torch
    from safetensors.numpy import load_file as load_np
    acfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    r = int(acfg['r'])
    scaling = float(acfg['lora_alpha']) / (r ** 0.5 if acfg.get('use_rslora') else r)
    ad = load_np(os.path.join(adapter_dir, 'adapter_model.safetensors'))
    pairs = {}
    for k, v in ad.items():
        m = re.match(re.escape(prefix) + r'(\d+)\.(.+)\.lora_([AB])\.weight$', k)
        assert m, f'unexpected adapter tensor {k}'
        pairs.setdefault(f'model.language_model.layers.{m.group(1)}.{m.group(2)}', {})[m.group(3)] = v
    assert pairs and all(set(p) == {'A', 'B'} for p in pairs.values()), 'adapter tensors must come in A/B pairs'
    cfg = json.load(open(os.path.join(src, 'config.json')))
    qgroup = _quant_group(cfg)
    os.makedirs(dst, exist_ok=True)
    files = _source_files(src)
    where = {}                      # tensor -> file: slice_checkpoint can put a module's packed weights and its scales in
    for fn in files:                # different shards (a shard closes after whichever tensor crosses the size limit)
        with safe_open(os.path.join(src, fn), 'pt') as f:
            where.update({k: fn for k in f.keys()})

    def tensor(out, key):           # from the shard in memory, else from the shard that holds it
        if key in out:
            return out[key]
        with safe_open(os.path.join(src, where[key]), 'pt') as g:
            return g.get_tensor(key)

    done, moved, total_q, rel = set(), 0, 0, []
    for fn in files:
        out = {}
        with safe_open(os.path.join(src, fn), 'pt') as f:
            keys = list(f.keys())
            for k in keys:
                out[k] = f.get_tensor(k)
        for base, ab in pairs.items():
            A = torch.from_numpy(ab['A']).float()
            B = torch.from_numpy(ab['B']).float()
            if base + '.weight_packed' in out:
                assert qgroup, f'{base}: packed weights in a checkpoint without a 4-bit config'
                pk, sc = out[base + '.weight_packed'], tensor(out, base + '.weight_scale')
                rows, cols = pk.shape[0], pk.shape[1] * 8
                assert B.shape[0] == rows and A.shape[1] == cols, (base, tuple(B.shape), tuple(A.shape), rows, cols)
                assert sc.shape == (rows, cols // qgroup), (base, tuple(sc.shape), rows, cols, qgroup)
                s_full = sc.float().repeat_interleave(qgroup, 1)                 # each weight's grid step
                q = _unpack4(pk, cols).float()
                t = q + (scaling * (B @ A)) / s_full                             # W + dW in units of the step
                if mode == 'stochastic':     # one stream per matrix: the result does not depend on the shard layout
                    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + zlib.crc32(base.encode()))
                    lo = torch.floor(t)
                    qn = lo + (torch.rand(t.shape, generator=gen) < (t - lo)).float()
                elif mode == 'nearest':
                    qn = torch.round(t)
                else:
                    raise ValueError(mode)
                qn = qn.clamp(-8, 7)
                moved += int((qn != q).sum())
                total_q += q.numel()
                rel.append(float((t - q).abs().mean()))
                out[base + '.weight_packed'] = _pack4(qn.to(torch.int8))
                done.add(base)
            elif base + '.weight' in out:                                         # a 16-bit weight: exact
                w = out[base + '.weight']
                assert w.shape == (B.shape[0], A.shape[1]), (base, tuple(w.shape))
                out[base + '.weight'] = (w.float() + scaling * (B @ A)).to(w.dtype)
                done.add(base)
        save_file(out, os.path.join(dst, fn), metadata={'format': 'pt'})
        log(f'  merged {fn}')
    missing = sorted(set(pairs) - done)
    assert not missing, f'{len(missing)} adapter targets not found in the checkpoint, e.g. {missing[:3]}'
    for p in glob.glob(os.path.join(src, '*')):
        b = os.path.basename(p)
        if os.path.isfile(p) and not b.endswith('.safetensors'):
            shutil.copy(p, os.path.join(dst, b))
    summary = {'mode': mode, 'seed': int(seed), 'matrices': len(done), 'lora_scaling': scaling,
               'moved_share': round(moved / max(1, total_q), 6), 'mean_abs_delta_in_steps': round(float(np.mean(rel)), 5)
               if rel else None}
    log('merged', summary)
    return summary
