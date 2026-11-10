"""Training loop for train_tpu.py (runs inside the Python 3.11 venv): load the bf16 Gemma 4 31B into a fsdp x tp mesh,
Qwix LoRA (fp32 master weights, bf16 compute), masked CE at labelled positions, PEFT export checked against a reference
GPU adapter.

Robustness for a run without a preceding canary (user 10-14):
- attention ladder: a step that fails to compile/run in the attention kernel (Mosaic VMEM, Pallas) is retried with a
  smaller splash block (1024 -> 512 -> 256), then with eager attention capped at 8192 tokens;
- an HBM failure demotes that length to the next shorter one (12288 -> 8192 -> emergency 4096; rows truncated,
  kept-label fraction logged);
- a failure after which the donated buffers are gone ends the loop without a final export (periodic exports remain);
- non-finite steps apply no update (lora_tpu.make_train_step); 5 in a row stop the run; exports refuse non-finite
  tensors and are written atomically;
- time-aware schedule: the learning rate is a step argument; after the first steady steps the remaining plan is cut
  (at an epoch boundary when possible) to what fits in the time left, and the cosine horizon follows the cut;
- per-epoch adapters in <adapters_dir>/ep<N>, the latest good adapter in adapter_out/;
- optional phase 2 ('phase2': {'lr', 'epochs', 'min_min', 'suffix', 'source_contains', 'dpo', 'a_share'}): after phase a
  (cut to a_share of the training time, if given) has finished
  and exported, the LoRA is reset to the same initial values, a fresh optimizer trains on the same data order (only the
  rows whose source tag contains 'source_contains', if given) with another learning rate (and 'dpo' overrides for the preference settings), and exports go to <adapter_out>_b /
  <adapters_dir>_b (no reload, no recompile: lr is a step argument).
- optional healing ('heal': {'mask', 'kd', 'temp', 'eval_npz', 'eval_len', 'eval_kl'}, pruner part 3): an FFMask cuts the model's
  hidden units (lora_tpu.install_ff_mask); kd True trains the LoRA on KL(full model || cut model) at every labelled
  position (the teacher is the same buffers with masks 1 and LoRA 0), kd False on the plain next-token loss; phase 2
  can switch it ('phase2': {'heal': {'kd': False}}); held-out nats of the teacher and the cut student at the start,
  per epoch and at the end of each phase ('heal_eval' log lines; own = agent-written tokens, all = every token; the
  student's lines also carry kl_own / kl_all, the mean KL(full model || cut model) per token).
usage: python train_tpu_runner.py '<json args>'
  G8_TINY=1|2: tiny random model on CPU (2: head counts for 8 fake devices), for kernels/train/test_lora_tpu.py;
  G8_FAIL=<rung or L<len>>,...: test hook, raise a simulated failure for that attention rung / sequence length;
  G8_TEST_SLEEP_S: test hook, extra seconds per step.
"""
import collections, json, math, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
from safetensors.numpy import save_file
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_tpu as LT
from tunix.models.gemma4 import model as gm

A = json.loads(sys.argv[1])
T0 = time.time()
TRAIN = A['mode'] == 'train'
LOG = open(A['out'] + '/log.jsonl', 'a')
TEST_FAIL = [x for x in os.environ.get('G8_FAIL', '').split(',') if x]
TEST_SLEEP = float(os.environ.get('G8_TEST_SLEEP_S', '0') or 0)


def log(name, **kv):
    kv = {'stage': name, 't_min': round((time.time() - A['t0']) / 60, 2), **kv}
    LOG.write(json.dumps(kv, default=str) + '\n'); LOG.flush(); print(json.dumps(kv, default=str), flush=True)


def left_s():
    return A['deadline_s'] - (time.time() - A['t0'])


def train_left_s():
    """Seconds of training left: notebook deadline minus the export reserve, and (train mode) the training cap."""
    s = left_s() - A['reserve_s']
    if TRAIN:
        s = min(s, A['cap_min'] * 60 - (time.time() - T0))
    if PHASE_END[0]:  # two arms in one notebook: arm a stops in time to leave arm b its share
        s = min(s, PHASE_END[0] - time.time())
    return s


PHASE_END = [None]


def hbm():
    try:
        s = [d.memory_stats() for d in jax.devices()]
        return {'peak_gb': round(max(x.get('peak_bytes_in_use', 0) for x in s) / 2**30, 2),
                'limit_gb': round(min(x.get('bytes_limit', 0) for x in s) / 2**30, 2)}
    except Exception as e:
        return {'err': str(e)[:200]}


log('jax', devices=[str(d) for d in jax.devices()], n=jax.device_count(), backend=jax.default_backend())
d = np.load(A['npz'])
offs = np.concatenate([[0], np.cumsum(d['lens'])])
rows = [(d['ids'][offs[i]:offs[i + 1]], d['labels'][offs[i]:offs[i + 1]]) for i in range(len(d['lens']))]
srcs = [str(x) for x in d['sources']] if 'sources' in d.files else [''] * len(rows)  # per-row source tag of the dump
buckets = sorted(int(b) for b in A['buckets'].split(','))
# demotion chain: the buckets plus an emergency length used only after HBM failures (never for initial bucketing)
chain = sorted(set(buckets) | ({int(A.get('emergency_len', 4096))} if A.get('emergency_len', 4096) else set()))
by_b = {b: [] for b in buckets}
for i, (ids, lab) in enumerate(rows):
    b = next(x for x in buckets if len(ids) <= x)
    by_b[b].append(i)
stats = {'rows': len(rows), 'tokens': int(d['lens'].sum()), 'trainable': int((d['labels'] != -100).sum()),
         'per_bucket': {b: len(v) for b, v in by_b.items()},
         'by_source': dict(sorted(((k, srcs.count(k)) for k in set(srcs)), key=lambda x: -x[1])[:10])}
json.dump(stats, open(A['out'] + '/data_stats.json', 'w'))
log('data', **stats)

tiny = os.environ.get('G8_TINY')
cfg = LT.tiny_config(mesh_test=tiny == '2') if tiny else gm.ModelConfig.gemma4_31b()
cfg.param_dtype = jnp.bfloat16
cfg.dtype = jnp.bfloat16
cfg.remat_config = gm.RematConfig.DECODER
cfg.use_flash_attention = A['flash']
cfg.flash_attention_block_size = int(A.get('flash_block', 1024))
f, t = (int(x) for x in A['mesh'].split(','))
mesh = jax.make_mesh((f, t), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2)
rank, alpha = A['rank'], 2.0 * A['rank']
t1 = time.time()
if A.get('random_init'):
    base = LT.random_init_model(cfg, mesh, seed=A['seed'])
else:
    base = LT.load_hidden_model(A['model'], cfg, mesh, dtype=jnp.bfloat16)  # host-side preprocessing, see lora_tpu
log('loaded', s=round(time.time() - t1, 1), hbm=hbm(), random_init=bool(A.get('random_init')),
    on=sorted({str(d.platform) for d in jax.tree.leaves(nnx.state(base))[0].devices()}))
# Tunix's splash-attention path reads the legacy mesh context (pxla.thread_resources), jax.set_mesh alone leaves it empty
with mesh, jax.set_mesh(mesh):
    lm = LT.apply_lora(base, rank, alpha, seed=A['seed'], master_fp32=A.get('master_fp32', True))
    del base
    _p = next(iter(LT.lora_pairs(lm).items()))
    log('lora', modules=len(LT.lora_pairs(lm)), dtypes=LT.lora_dtypes(lm), hbm=hbm(),
        example={'path': _p[0], 'a': [list(_p[1]['a'].value.shape), str(_p[1]['a'].value.sharding.spec)],
                 'b': [list(_p[1]['b'].value.shape), str(_p[1]['b'].value.sharding.spec)]})
    if A.get('init_adapter'):  # start from an adapter this trainer exported (e.g. G11a): stage 2 of SFT -> preferences
        t_i = time.time()
        with safe_open(A['init_adapter'], 'np') as fh:
            peft0 = {k_: fh.get_tensor(k_) for k_ in fh.keys()}
        n_used = LT.import_peft(lm, cfg, peft0, rank)
        back = LT.export_peft(lm, cfg, rank, alpha)
        diff = max(float(np.abs(back[k_] - peft0[k_].astype(np.float32)).max()) for k_ in peft0)
        log('init_adapter', path=A['init_adapter'], tensors=n_used, roundtrip_max_abs_diff=diff, s=round(time.time() - t_i, 1),
            b_absmax=float(max(np.abs(v).max() for k_, v in peft0.items() if 'lora_B' in k_)))
        assert set(back) == set(peft0) and diff == 0.0, ('init adapter round trip', diff)
        del peft0, back
    HEAL = A.get('heal')  # pruner part 3: the cut model (FFMask) trained toward the full model (teacher_view)
    if HEAL:
        hm = np.load(HEAL['mask'])
        kept = LT.install_ff_mask(lm, hm['mask'], mesh)
        log('heal_mask', layers=len(kept), kept_min=min(kept), kept_max=max(kept), units=int(hm['mask'].shape[1]),
            kept_share=round(sum(kept) / hm['mask'].size, 4), sha=str(hm['sha']) if 'sha' in hm.files else None,
            kd=bool(HEAL.get('kd')), temp=HEAL.get('temp', 1.0))
        del hm
    phase2 = A.get('phase2') if TRAIN else None
    # host copy of the initial LoRA values: phase 2 restarts from exactly the same initialisation
    lora_init = ({k: (np.asarray(jax.device_get(p['a'].value)), np.asarray(jax.device_get(p['b'].value)))
                  for k, p in LT.lora_pairs(lm).items()} if phase2 else None)

    def make_plan(epochs, only=None):
        """Epochs in order, rows shuffled within each epoch, batches of one length bucket (same seed -> same order);
        only(i) -> bool restricts the plan to a subset of the rows."""
        rng = np.random.default_rng(A['seed'])
        plan = []
        in_pairs = {i for v in instep.values() for i in v}
        row_b = {i: b for b, idx in by_b.items() for i in idx}
        for ep in range(epochs):
            ep_batches = []
            for pid in rng.permutation(sorted(instep)) if instep else []:
                c, r = instep[pid]
                if only is None or (only(c) and only(r)):
                    ep_batches.append((max(row_b[c], row_b[r]), [c, r], ep, str(pid)))
            for b, idx in by_b.items():
                idx = [i for i in idx if (only is None or only(i)) and i not in in_pairs]
                if not idx:
                    continue
                idx = list(rng.permutation(idx))
                if len(idx) % A['batch']:
                    idx += idx[:A['batch'] - len(idx) % A['batch']]  # wrap: a few rows twice per epoch
                ep_batches += [(b, idx[k:k + A['batch']], ep) for k in range(0, len(idx), A['batch'])]
            plan += [ep_batches[j] for j in rng.permutation(len(ep_batches))]
        if not TRAIN:
            plan = [x for b in buckets for x in [y for y in plan if y[0] == b][:A['canary_steps']]]
        return plan

    assert A['max_targets'] % A['loss_chunk'] == 0, (A['max_targets'], A['loss_chunk'])
    blk = int(A.get('flash_block', 1024))
    if A['flash']:
        rungs = [{'name': f'flash{bs}', 'flash': True, 'block': bs, 'max_len': None}
                 for bs in sorted({blk, 512, 256}, reverse=True) if bs <= blk]
        rungs.append({'name': 'eager8192', 'flash': False, 'block': blk, 'max_len': 8192})
    else:
        rungs = [{'name': 'eager', 'flash': False, 'block': blk, 'max_len': None}]
    if not A.get('ladder', True):
        rungs = rungs[:1]
    # hardware state shared by both phases: attention rung, demoted lengths, compiled shapes, steady step times
    S = {'rung': 0, 'demoted': set(), 'compiled': set(), 'times': {}, 'export_s': [], 'n_err': 0}

    def build_step():
        rg = rungs[S['rung']]
        cfg.use_flash_attention, cfg.flash_attention_block_size = rg['flash'], rg['block']
        S['pair_step'] = LT.make_pair_step(cfg, chunk=A['loss_chunk'], eager_mask=not rg['flash'])
        if HEAL and HEAL.get('kd'):
            return LT.make_distill_step(cfg, chunk=A['loss_chunk'], eager_mask=not rg['flash'],
                                        temp=float(HEAL.get('temp', 1.0)))
        return LT.make_train_step(cfg, chunk=A['loss_chunk'], eager_mask=not rg['flash'])

    S['step'] = build_step()

    def eff_len(b):
        L = b
        cap = rungs[S['rung']]['max_len']
        if cap and L > cap:
            L = max([x for x in chain if x <= cap] or [cap])
        while L in S['demoted']:
            L = max(x for x in chain if x < L)
        return L

    def export(tag, od):
        """Never raises (a full disk or I/O error must not end training); returns {'error': ...} instead."""
        try:
            return _export(tag, od)
        except Exception as e:
            import traceback
            log('export_error', tag=tag, dir=od, error=f'{type(e).__name__}: {str(e)[:500]}',
                tb=traceback.format_exc()[-1500:])
            return {'match': None, 'error': str(e)[:200]}

    def _export(tag, od):
        if 'export' in TEST_FAIL and ':step' in tag:
            raise OSError(28, 'No space left on device (simulated)')
        t2 = time.time()
        peft = LT.export_peft(lm, cfg, rank, alpha)
        bad = [k for k, v in peft.items() if not np.isfinite(v).all()]
        if bad:
            log('export_refused', tag=tag, nonfinite_tensors=len(bad), example=bad[0])
            return {'match': None, 'refused': True}
        os.makedirs(od, exist_ok=True)
        save_file(peft, od + '/adapter_model.safetensors.tmp')
        os.replace(od + '/adapter_model.safetensors.tmp', od + '/adapter_model.safetensors')
        with open(od + '/adapter_config.json.tmp', 'w') as fh:
            json.dump(LT.peft_config(rank, alpha), fh, indent=1)
        os.replace(od + '/adapter_config.json.tmp', od + '/adapter_config.json')
        chk = {}
        if A['ref']:
            with safe_open(A['ref'], 'np') as r:
                rk = set(r.keys())
                shapes_bad = [k for k in rk & set(peft) if tuple(r.get_slice(k).get_shape()) != peft[k].shape]
                dt = {r.get_slice(k).get_dtype() for k in list(rk)[:5]}
            chk = {'ref_keys': len(rk), 'ours': len(peft), 'missing': sorted(rk - set(peft))[:5],
                   'extra': sorted(set(peft) - rk)[:5], 'shape_mismatch': shapes_bad[:5], 'ref_dtypes': sorted(dt)}
            chk['match'] = not chk['missing'] and not chk['extra'] and not shapes_bad
        S['export_s'].append(time.time() - t2)
        log('export', tag=tag, dir=od, s=round(S['export_s'][-1], 1),
            bytes=os.path.getsize(od + '/adapter_model.safetensors'),
            b_absmax=float(max(np.abs(v).max() for k, v in peft.items() if 'lora_B' in k)), **chk)
        return chk

    EV = None
    if HEAL and HEAL.get('eval_npz'):
        e_ = np.load(HEAL['eval_npz'])
        eo = np.concatenate([[0], np.cumsum(e_['lens'])])
        has_calls = 'calls' in e_.files   # heal_data.py since B192: tool-call bodies, a third view of the eval
        EV = [(e_['ids'][eo[i]:eo[i + 1]], e_['labels'][eo[i]:eo[i + 1]], e_['own'][eo[i]:eo[i + 1]].astype(bool),
               e_['calls'][eo[i]:eo[i + 1]].astype(bool) if has_calls else None)
              for i in range(len(e_['lens']))]
        del e_
    S['eval_fns'], S['eval_s'] = {}, []

    def heal_eval(tag, views=('student',)):
        """Held-out nats per token of the cut student (the model as it is) and/or the full teacher (teacher_view): on
        the agent-written tokens (own, the number part 1 and the speed test report), on every token (all) and, when
        the eval rows carry a calls mask, on the tool-call bodies (calls: a second pass with that mask). Never raises."""
        if not EV:
            return None
        out, t_all = {}, time.time()
        try:
            EL = eff_len(int(HEAL.get('eval_len', 8192)))
            B = A['batch']
            for view in views:
                t_e = time.time()
                kl = view == 'student' and bool(HEAL.get('eval_kl', True))  # distance to the full model, too
                key = (view, rungs[S['rung']]['name'], EL, kl)
                if key not in S['eval_fns']:
                    rg = rungs[S['rung']]
                    cfg.use_flash_attention, cfg.flash_attention_block_size = rg['flash'], rg['block']
                    S['eval_fns'][key] = LT.make_eval_rows(cfg, chunk=A['loss_chunk'], eager_mask=not rg['flash'],
                                                           teacher=view == 'teacher', kl=kl)
                so = no = sa = na = ko = ka = sc = nc = kc = 0.0
                with_calls = EV[0][3] is not None
                for k0 in range(0, len(EV), B):
                    grp = EV[k0:k0 + B]
                    real = len(grp)
                    grp = grp + [grp[-1]] * (B - real)  # a short last batch repeats its last row (not counted)
                    ids = np.zeros((B, EL), np.int32)
                    lab = np.full((B, EL), -100, np.int32)
                    own = np.zeros((B, EL), bool)
                    calls = np.zeros((B, EL), bool)
                    lens = np.zeros(B, np.int32)
                    for b, (x, y, o, cm) in enumerate(grp):
                        n = min(len(x), EL)
                        ids[b, :n], lab[b, :n], own[b, :n], lens[b] = x[:n], y[:n], o[:n], n
                        if cm is not None:
                            calls[b, :n] = cm[:n]
                    bt = LT.make_batch(ids, lab, lens, EL)
                    pos = np.asarray(bt['pos'])
                    w_own = np.take_along_axis(own, np.minimum(pos + 1, EL - 1), axis=1) & (np.asarray(bt['w']) != 0)
                    r_ = S['eval_fns'][key](lm, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'],
                                            jnp.asarray(w_own, jnp.float32), bt['w'])
                    a, an, c, cn = r_[:4]
                    if kl:
                        ko += float(np.asarray(r_[4])[:real].sum())
                        ka += float(np.asarray(r_[5])[:real].sum())
                    so += float(np.asarray(a)[:real].sum())
                    no += float(np.asarray(an)[:real].sum())
                    sa += float(np.asarray(c)[:real].sum())
                    na += float(np.asarray(cn)[:real].sum())
                    if with_calls:
                        w_calls = np.take_along_axis(calls, np.minimum(pos + 1, EL - 1), axis=1) & (np.asarray(bt['w']) != 0)
                        r2 = S['eval_fns'][key](lm, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'],
                                                jnp.asarray(w_calls, jnp.float32), bt['w'])
                        sc += float(np.asarray(r2[0])[:real].sum())
                        nc += float(np.asarray(r2[1])[:real].sum())
                        if kl:
                            kc += float(np.asarray(r2[4])[:real].sum())
                out[view] = {'own_nats': round(-so / max(no, 1.0), 5), 'all_nats': round(-sa / max(na, 1.0), 5),
                             'own_tokens': int(no), 'all_tokens': int(na), 's': round(time.time() - t_e, 1),
                             **({'kl_own': round(ko / max(no, 1.0), 5), 'kl_all': round(ka / max(na, 1.0), 5)}
                                if kl else {}),
                             **({'calls_nats': round(-sc / max(nc, 1.0), 5), 'calls_tokens': int(nc),
                                 **({'kl_calls': round(kc / max(nc, 1.0), 5)} if kl else {})} if with_calls else {})}
            log('heal_eval', tag=tag, len=EL, rows=len(EV), **out)
        except Exception as e:
            import traceback
            log('heal_eval_error', tag=tag, error=f'{type(e).__name__}: {str(e)[:800]}', tb=traceback.format_exc()[-1500:])
        S['eval_s'].append(time.time() - t_all)
        return out

    def classify(msg):
        m = msg.lower()
        if any(x in m for x in ('vmem', 'mosaic', 'pallas', 'splash')):
            return 'attention'
        if any(x in m for x in ('hbm', 'resource_exhausted', 'out of memory')):
            return 'memory'
        return 'other'

    # ---- DPO (optional, A['dpo'] = {'beta', 'sft_chosen', 'reject_floor'}): rows tagged 'dpo:<pair>:c' / ':r' in the
    # dump's sources are the passing / failing runs of one task (all their segments). The loss stays a target-weighted
    # CE; each epoch the pair rows get signed weights from the current per-token log-ratio margin against the
    # reference (the initial LoRA, B = 0, i.e. the base): chosen sft_chosen + a, rejected -a, a = beta * sigmoid(-beta
    # * m), m = mean log-ratio(chosen) - mean log-ratio(rejected) (length-normalised, as SimPO). A rejected side whose
    # mean log-ratio is already below reject_floor gets weight 0 (bounded unlikelihood). Other rows: SFT, weight 1.
    DPO = A.get('dpo') if TRAIN else None
    row_w = np.ones(len(rows), np.float32)
    pairs, pair_rows, ref = {}, [], {}
    instep = {}  # pair id -> (chosen row, rejected row): trained one pair per batch, objective in-step (B163)
    if DPO:
        for i, s_ in enumerate(srcs):
            if s_.startswith('dpo:'):
                _, pid, side = s_.rsplit(':', 2)
                pairs.setdefault(pid, {'c': [], 'r': []})[side].append(i)
        lone_r = [i for v in pairs.values() if not v['c'] for i in v['r']]
        pairs = {k_: v for k_, v in pairs.items() if v['c'] and v['r']}
        pair_rows = sorted({i for v in pairs.values() for side in 'cr' for i in v[side]})
        for i in lone_r:  # a rejected row whose chosen side was dropped (too long, no labels) must not be trained as SFT
            row_w[i] = 0.0
        if DPO.get('in_step'):
            instep = {k_: (v['c'][0], v['r'][0]) for k_, v in pairs.items() if len(v['c']) == 1 and len(v['r']) == 1}
        S['row_logp'] = {}
        # Clock-calibrated IPO (B154): DPO['ipo_target_<kind>'] = 'clock' sets each pair's IPO target from the two runs'
        # estimated serving times (pairs.json from dpo_pairs.py: est_s = fixed seconds per model call + seconds per
        # generated token, B142): target = tau * ln(est_s(rejected) / est_s(chosen)), clipped to [lo, hi]. A pair whose
        # runs differ by little time asks for a small log-ratio margin, a pair with a large time gap for a larger one.
        meta = json.load(open(DPO['pairs_meta'])) if DPO.get('pairs_meta') and os.path.exists(DPO['pairs_meta']) else {}
        pair_target, tnorm = {}, {}

        def set_targets(arm):
            """IPO target per pair under the current DPO config, and the per-kind weight scale: 'self' divides (t - m)
            by the pair's own target (every pair starts at weight 1), 'mean' by the kind's mean target, so a pair with a
            larger time gap pushes harder from the first step while the kind's mean starting weight stays 1 (B155)."""
            pair_target.clear(), tnorm.clear()
            for pid in pairs:
                kind = pid.rsplit('~', 1)[1] if '~' in pid else 'fail'
                tg = DPO.get(f'ipo_target_{kind}', DPO.get('ipo_target', 0.01))
                if tg == 'clock':
                    mm = meta.get(pid) or {}
                    c_s, r_s = (mm.get('c') or {}).get('est_s'), (mm.get('r') or {}).get('est_s')
                    if c_s and r_s and c_s > 0 and r_s > 0:
                        tg = DPO.get(f'ipo_tau_{kind}', 0.02) * math.log(r_s / c_s)
                        tg = min(DPO.get(f'ipo_hi_{kind}', 0.03), max(DPO.get(f'ipo_lo_{kind}', 0.002), tg))
                    else:
                        tg = DPO.get(f'ipo_fallback_{kind}', 0.01)
                pair_target[pid] = float(tg)
            tq = {k_: [t for p_, t in pair_target.items() if p_.endswith('~' + k_)] for k_ in {p_.rsplit('~', 1)[-1] for p_ in pairs}}
            for k_, v in tq.items():
                tnorm[k_] = float(np.mean(v)) if v and DPO.get(f'ipo_norm_{k_}', DPO.get('ipo_norm', 'self')) == 'mean' else None
            log('dpo', arm=arm, pairs=len(pairs), in_step_pairs=len(instep), rows=len(pair_rows), lone_rejected_rows=len(lone_r), meta_pairs=len(meta), **DPO,
                ipo_targets={k_: [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] for k_, v in tq.items() if v},
                ipo_norm={k_: (round(v, 5) if v else 'self') for k_, v in tnorm.items()})
        set_targets('a')

    def row_logps(idx_list):
        """{row: (sum log p over its targets, n targets)} for the current LoRA, forward only, A['batch'] rows of one
        length per call (the batch axis is sharded over fsdp: G15 v1 died on batches of 1 with fsdp 2, B141); a short
        last batch repeats its last row."""
        out, by_len = {}, collections.defaultdict(list)
        for i in idx_list:
            by_len[eff_len(next(x for x in buckets if len(rows[i][0]) <= x))].append(i)
        B = A['batch']
        for L, ix in sorted(by_len.items()):
            t_l = time.time()
            for k in range(0, len(ix), B):
                grp = ix[k:k + B]
                grp_full = grp + [grp[-1]] * (B - len(grp))
                ids = np.zeros((B, L), np.int32)
                lab = np.full((B, L), -100, np.int32)
                lens = np.zeros(B, np.int32)
                for b, i in enumerate(grp_full):
                    n = min(len(rows[i][0]), L)
                    ids[b, :n], lab[b, :n], lens[b] = rows[i][0][:n], rows[i][1][:n], n
                bt = LT.make_batch(ids, lab, lens, A['max_targets'])
                key = (rungs[S['rung']]['name'], L)
                if key not in S['row_logp']:
                    rg = rungs[S['rung']]
                    cfg.use_flash_attention, cfg.flash_attention_block_size = rg['flash'], rg['block']
                    S['row_logp'][key] = LT.make_row_logp(cfg, chunk=A['loss_chunk'], eager_mask=not rg['flash'])
                lp, cnt = S['row_logp'][key](lm, bt['tokens'], bt['lens'], bt['pos'], bt['tgt'], bt['w'])
                for b, i in enumerate(grp):
                    out[i] = (float(lp[b]), int(cnt[b]))
            log('row_logp', len=L, rows=len(ix), s=round(time.time() - t_l, 1))  # also feeds the no-progress watchdog
        return out

    def pair_coef(pid):
        """[beta, sft_chosen, ipo_target, is_ipo, ipo_norm, floor] of an in-step pair, read from DPO at call time (arm b
        may override it). The rejected floor of in-step pairs is 'instep_floor' (default off): the per-epoch
        'reject_floor' (-0.1) was set for weights that could not react within an epoch."""
        kind = pid.rsplit('~', 1)[1] if '~' in pid else 'fail'
        tgt = pair_target.get(pid, 0.01)
        fl = DPO.get('instep_floor')
        return np.array([DPO.get(f'beta_{kind}', DPO['beta']), DPO.get('sft_chosen', 1.0), tgt,
                         float(DPO.get(f'loss_{kind}', DPO.get('loss', 'dpo')) == 'ipo'), tnorm.get(kind) or tgt,
                         -1e9 if fl is None else fl], np.float32)

    def pair_epoch(name, ep):
        """Margins of the in-step pairs as each was met in this epoch (before its own update)."""
        pts = S.setdefault('pair_pts', {}).pop((name, ep), [])
        if not pts:
            return
        q = lambda x: [round(float(y), 4) for y in np.percentile(x, [10, 50, 90])]  # noqa: E731
        log('pair_epoch', phase=name, ep=ep, pairs=len(pts), margin_p10_50_90=q([x[0] for x in pts]),
            acc=round(float(np.mean([x[0] > 0 for x in pts])), 3), chosen_ratio=q([x[1] for x in pts]),
            rejected_ratio=q([x[2] for x in pts]))

    def dpo_refresh(tag):
        t_r = time.time()
        in_rows = {i for v in instep.values() for i in v}
        cur = ref if tag == 'start' else row_logps([i for i in pair_rows if i not in in_rows])
        floor, sft_c = DPO.get('reject_floor', -0.1), DPO.get('sft_chosen', 1.0)
        ms, dcs, drs, ws, frozen = [], [], [], [], 0
        by_kind = collections.defaultdict(list)  # pair id '<task>~<kind>' (fail / speed, B130): margins per kind
        ipo_pts = {}  # kind -> [(target, margin)]: does the learned margin follow the time gap? (B154)

        def mean_ratio(ix, d):
            return (sum(d[i][0] for i in ix) - sum(ref[i][0] for i in ix)) / max(1, sum(d[i][1] for i in ix))
        for pid, v in pairs.items():
            if pid in instep:
                continue
            kind = pid.rsplit('~', 1)[1] if '~' in pid else 'fail'
            beta = DPO.get(f'beta_{kind}', DPO['beta'])
            dc, dr = mean_ratio(v['c'], cur), mean_ratio(v['r'], cur)
            m = dc - dr
            by_kind[kind].append(m)
            if DPO.get(f'loss_{kind}', DPO.get('loss', 'dpo')) == 'ipo':
                # IPO (Azar et al. 2023, arXiv:2310.12036): squared loss (m - target)^2 regresses the margin to a target
                # instead of pushing it to infinity, so near-ties and noisy labels (speed pairs) are not over-fitted.
                # Weight = -dL/dm scaled to 1 at m = 0 (DPO's beta/2 = 1 at beta 2), 0 at the target, negative past it.
                tgt = pair_target[pid]
                nrm = tnorm.get(kind) or tgt
                a = max(-1.0, min(DPO.get('ipo_amax', 3.0) if tnorm.get(kind) else 1.0, (tgt - m) / nrm))
                ipo_pts.setdefault(kind, []).append((tgt, m, a))
            else:
                a = beta / (1.0 + math.exp(min(50.0, beta * m)))
            for i in v['c']:
                row_w[i] = sft_c + a
            for i in v['r']:
                row_w[i] = -a if (dr > floor or a < 0) else 0.0
            frozen += dr <= floor and a > 0
            ms.append(m), dcs.append(dc), drs.append(dr), ws.append(a)
        q = lambda x: [round(float(y), 4) for y in np.percentile(x, [10, 50, 90])] if x else None
        log('dpo_refresh', tag=tag, pairs=len(ms), margin_p10_50_90=q(ms), acc=round(float(np.mean([m > 0 for m in ms])), 3)
            if ms else None, chosen_ratio=q(dcs), rejected_ratio=q(drs), weight_mean=round(float(np.mean(ws)), 4) if ws else None,
            rejected_frozen=frozen, s=round(time.time() - t_r, 1),
            by_kind={k_: {'n': len(v), 'margin_p50': q(v)[1], 'acc': round(float(np.mean([m > 0 for m in v])), 3)}
                     for k_, v in by_kind.items()},
            ipo={k_: {'target_p10_50_90': q([t for t, _, _ in v]), 'weight_p10_50_90': q([w_ for _, _, w_ in v]),
                      'margin_target_corr': round(float(np.corrcoef([t for t, _, _ in v], [m_ for _, m_, _ in v])[0, 1]), 3)
                      if len(v) > 2 and np.std([t for t, _, _ in v]) > 0 and np.std([m_ for _, m_, _ in v]) > 0 else None,
                      'reached_target': round(float(np.mean([m_ >= t for t, m_, _ in v])), 3)} for k_, v in ipo_pts.items()})

    def run_phase(name, plan, LR, adapter_out, adapters_dir):
        """Train lm over `plan` with warmup + cosine to 10% of LR (horizon re-planned to the time left); export per
        epoch to adapters_dir/ep<N>, periodically and finally to adapter_out. Returns (alive, stop, ok_steps)."""
        FLOOR = 0.1 * LR
        warm = max(1, len(plan) // 20)
        horizon = [len(plan)]

        def lr_at(k):
            if k < warm:
                return LR * (k + 1) / warm
            prog = min(1.0, (k - warm) / max(1, horizon[0] - 1 - warm))  # the last planned step runs at the floor
            return FLOOR + (LR - FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

        def replan(k):
            rem, times = plan[k:], S['times']
            if not rem or not times:
                return
            per_tok = max(float(np.median(v)) / L for L, v in times.items() if v)  # conservative for unseen lengths
            lens_rem = [eff_len(x[0]) for x in rem]
            est = [float(np.median(times[L])) if times.get(L) else per_tok * L for L in lens_rem]
            unseen = {L for L in lens_rem if not times.get(L)}
            n_exports = 1 + sum(1 for j in range(1, len(rem)) if rem[j][2] != rem[j - 1][2])
            n_exports += int(sum(est) / 60 / max(A['save_every_min'], 1e-3))
            budget = (train_left_s() - n_exports * (max(S['export_s']) if S['export_s'] else A.get('export_est_s', 60.0))
                      - A.get('compile_allow_s', 240) * len(unseen))
            if HEAL and EV:  # one held-out eval per epoch boundary and one at the end
                n_ev = 1 + sum(1 for j in range(1, len(rem)) if rem[j][2] != rem[j - 1][2])
                budget -= n_ev * (max(S['eval_s']) if S['eval_s'] else A.get('eval_est_s', 240.0))
            cum = np.cumsum(est)
            fit = int(np.searchsorted(cum, budget, side='right'))
            log('replan', phase=name, k=k, remaining=len(rem), fit=fit, est_min=round(cum[-1] / 60, 1),
                budget_min=round(budget / 60, 1), step_s={L: round(float(np.median(v)), 2) for L, v in times.items() if v})
            if fit >= len(rem):
                return
            end = k + fit
            bounds = [j for j in range(k + 1, end + 1) if j < len(plan) and plan[j][2] != plan[j - 1][2]]
            if bounds and bounds[-1] - k >= 0.7 * fit:
                end = bounds[-1]
            del plan[end:]
            horizon[0] = len(plan)
            log('replan_cut', phase=name, steps=len(plan), horizon=horizon[0],
                last_epoch=plan[-1][2] if plan else None, whole_epochs=bool(bounds) and end == bounds[-1])

        log('plan', phase=name, steps=len(plan), epochs=len({x[2] for x in plan}), warmup=warm, lr=LR,
            buckets={b: sum(1 for x in plan if x[0] == b) for b in buckets}, rungs=[x['name'] for x in rungs])
        k, t_last, tok_done, bad_run, ok_steps = 0, time.time(), 0, 0, 0
        alive, stop = True, 'plan_done'
        labels_total = labels_kept = 0
        next_replan = A.get('replan_at', 12)
        while k < len(plan):
            b, idx, ep = plan[k][:3]
            pid = plan[k][3] if len(plan[k]) > 3 else None  # an in-step preference pair: idx = [chosen, rejected]
            if TRAIN and k and ep != plan[k - 1][2] and ok_steps:
                export(f'{name}:ep{plan[k - 1][2]}', f'{adapters_dir}/ep{plan[k - 1][2]}')
                if HEAL:
                    heal_eval(f'{name}:ep{plan[k - 1][2]}')
                t_last = time.time()
            if DPO and pairs and (k == 0 or ep != plan[k - 1][2]):
                if k == 0 and not ref:
                    ref.update(row_logps(pair_rows))  # the starting LoRA (B = 0: the base; or the init adapter) is the reference
                if k and instep:
                    pair_epoch(name, plan[k - 1][2])
                if len(pairs) > len(instep):  # pairs still on per-epoch signed weights (several rows per side)
                    dpo_refresh('start' if k == 0 else f'ep{ep}')
                t_last = time.time()
            if train_left_s() < 0:
                stop = 'time'
                log('stop', phase=name, reason='time', step=k)
                break
            L = eff_len(b)
            ids = np.zeros((len(idx), L), np.int32)
            lab = np.full((len(idx), L), -100, np.int32)
            lens = np.zeros(len(idx), np.int32)
            n_lab = 0
            for j, i in enumerate(idx):
                n = min(len(rows[i][0]), L)  # truncation only when the length was demoted / capped
                ids[j, :n], lab[j, :n], lens[j] = rows[i][0][:n], rows[i][1][:n], n
                n_lab += int((rows[i][1][1:] != -100).sum())
            kept = int((lab[:, 1:] != -100).sum())
            if pid is not None and any(len(rows[i][0]) > L for i in idx):
                log('skip', phase=name, k=k, ep=ep, bucket=b, as_len=L, pair=pid, reason='pair truncated (demoted length)')
                k += 1
                continue
            if L < b:
                labels_total += n_lab
                labels_kept += kept
            if not kept:
                log('skip', phase=name, k=k, bucket=b, as_len=L, reason='no labels left after truncation')
                k += 1
                continue
            bt = LT.make_batch(ids, lab, lens, L if HEAL else A['max_targets'])
            if DPO and pid is None:
                bt['w'] = bt['w'] * jnp.asarray(row_w[np.asarray(idx)])[:, None]
            lr = lr_at(k)
            key = (rungs[S['rung']]['name'], L, pid is not None, bool(HEAL and HEAL.get('kd')))  # each step kind compiles
            t3 = time.time()
            try:
                if rungs[S['rung']]['name'] in TEST_FAIL or f'L{L}' in TEST_FAIL:
                    raise RuntimeError(('Mosaic: scoped vmem limit exceeded (simulated)'
                                        if rungs[S['rung']]['name'] in TEST_FAIL
                                        else 'RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm (simulated)'))
                extra = {}
                if pid is not None:
                    c_, r_ = idx
                    rmean = np.array([ref[c_][0] / max(1, ref[c_][1]), ref[r_][0] / max(1, ref[r_][1])], np.float32)
                    loss, m_, ratio_, n2, gn, ok = S['pair_step'](
                        lm, S['opt'], bt['tokens'], bt['lens'], bt['pos'], bt['tgt'], bt['w'], jnp.asarray(rmean),
                        jnp.asarray(pair_coef(pid)), np.float32(lr))
                    n = float(np.asarray(n2).sum())
                    m_, ratio_ = float(m_), [float(x) for x in np.asarray(ratio_)]
                    extra = {'pair': pid, 'm': round(m_, 4), 'ratio_c': round(ratio_[0], 4), 'ratio_r': round(ratio_[1], 4)}
                    if np.isfinite(m_):
                        S.setdefault('pair_pts', {}).setdefault((name, ep), []).append((m_, ratio_[0], ratio_[1]))
                else:
                    loss, n, gn, ok = S['step'](lm, S['opt'], bt['tokens'], bt['lens'], bt['pos'], bt['tgt'], bt['w'],
                                                np.float32(lr))
                loss, ok = float(loss), bool(ok)
                if TEST_SLEEP:
                    time.sleep(TEST_SLEEP)
            except Exception as e:
                import traceback
                msg = f'{type(e).__name__}: {str(e)[:1500]}'
                kind = classify(msg)
                log('step_error', phase=name, k=k, bucket=b, as_len=L, rung=rungs[S['rung']]['name'], kind=kind,
                    error=msg, tb=traceback.format_exc()[-2500:], **hbm())
                S['n_err'] += 1
                if not LT.lora_alive(lm):
                    alive, stop = False, 'state_lost'
                    log('stop', phase=name, reason='state_lost: donated buffers deleted by the failed step; '
                                                   'keeping earlier exports')
                    break
                if not TRAIN:
                    k += 1  # canary: report every bucket's failure
                    continue
                smaller = [x for x in chain if x < L]
                if kind == 'memory' and smaller:
                    S['demoted'].add(L)
                    log('demote', length=L, to=max(smaller))
                elif kind != 'memory' and S['rung'] + 1 < len(rungs):
                    S['rung'] += 1
                    S['step'] = build_step()
                    log('rung', to=rungs[S['rung']]['name'])
                elif smaller:
                    S['demoted'].add(L)
                    log('demote', length=L, to=max(smaller))
                else:
                    stop = 'error'
                    log('stop', phase=name, reason='no fallback left', step=k)
                    break
                if S['n_err'] > 6:
                    stop = 'error'
                    log('stop', phase=name, reason='too many step errors', step=k)
                    break
                continue  # retry the same batch with the new setting
            dt = time.time() - t3
            first = key not in S['compiled']
            S['compiled'].add(key)
            if not first:
                S['times'].setdefault(L, []).append(dt)
            tok_done += int(lens.sum())
            log('step', phase=name, k=k, bucket=b, ep=ep, **({'as_len': L} if L != b else {}), **extra, loss=round(loss, 4),
                applied=ok, targets=int(n), dropped=bt['dropped'], grad_norm=float(gn), s=round(dt, 2),
                tok_s=round(int(lens.sum()) / dt, 1), lr=float(lr), compile=first, rung=rungs[S['rung']]['name'],
                **(hbm() if k < 2 * len(buckets) + 2 or k % 25 == 0 or first else {}))
            if ok:
                ok_steps += 1
                bad_run = 0
            else:
                bad_run += 1
                if bad_run >= 5:
                    stop = 'nonfinite'
                    log('stop', phase=name, reason='5 consecutive non-finite steps (no update applied)', step=k)
                    break
            k += 1
            if TRAIN and k >= next_replan:
                try:
                    replan(k)
                except Exception as e:  # planning is an optimisation; training goes on with the current plan
                    log('replan_error', phase=name, k=k, error=f'{type(e).__name__}: {str(e)[:500]}')
                next_replan = k + A.get('replan_every', 40)
            if TRAIN and time.time() - t_last > A['save_every_min'] * 60 and ok_steps:
                export(f'{name}:step{k}', adapter_out)
                t_last = time.time()
        if instep and k:
            pair_epoch(name, plan[min(k, len(plan)) - 1][2])
        chk = {}
        if alive and (ok_steps or not TRAIN):
            chk = export(f'{name}:final', adapter_out)
            if TRAIN and stop == 'plan_done' and plan and not chk.get('refused') and not chk.get('error'):
                import shutil  # the plan ended on an epoch boundary: the final adapter is that epoch's adapter
                shutil.copytree(adapter_out, f'{adapters_dir}/ep{plan[-1][2]}', dirs_exist_ok=True)
                log('export_copy', phase=name, to=f'{adapters_dir}/ep{plan[-1][2]}')
        elif alive:
            log('no_final_export', phase=name, reason='no successful training step')
        if HEAL and alive and ok_steps:
            heal_eval(f'{name}:final')
        log('done', phase=name, stop=stop, steps_ok=ok_steps, planned=len(plan), tokens=tok_done,
            step_errors=S['n_err'], rung=rungs[S['rung']]['name'], demoted=sorted(S['demoted']),
            truncated_label_frac_kept=round(labels_kept / labels_total, 3) if labels_total else None,
            min=round((time.time() - T0) / 60, 1), adapter_matches_ref=chk.get('match'))
        return alive, stop, ok_steps

    adapter_out = A['adapter_out'].rstrip('/')
    adapters_dir = A.get('adapters_dir') or os.path.join(os.path.dirname(adapter_out), 'adapters')
    S['opt'] = nnx.Optimizer(lm, LT.make_tx(), wrt=nnx.LoRAParam)
    if phase2 and phase2.get('a_share'):  # arm a gets this share of the training time left now (the rest is arm b's)
        PHASE_END[0] = time.time() + phase2['a_share'] * train_left_s()
        log('phase_a_end', left_min=round(train_left_s() / 60, 1), a_share=phase2['a_share'])
    plan_a = make_plan(A['epochs'])
    if HEAL:
        heal_eval('start', views=('teacher', 'student'))
    alive, stop, ok1 = run_phase('a', plan_a, A['lr'], adapter_out, adapters_dir)
    PHASE_END[0] = None

    # ---- phase 2 (optional): same init, same data order, different learning rate; only after phase a is saved ----
    if phase2:
        need_s = phase2.get('min_min', 60) * 60
        sub = phase2.get('source_contains') or ''
        only = (lambda i: sub in srcs[i]) if sub else None
        n_sub = sum(1 for i in range(len(rows)) if only is None or only(i))
        if not (alive and ok1 and (stop == 'plan_done' or (stop == 'time' and phase2.get('a_share')))):
            log('phase2_skipped', reason=f'phase a ended with {stop}, alive={alive}, steps={ok1}')
        elif n_sub < A['batch']:
            log('phase2_skipped', reason=f'{n_sub} rows with a source containing {sub!r}')
        elif train_left_s() < need_s:
            log('phase2_skipped', reason=f'{round(train_left_s() / 60, 1)} min left < {need_s / 60} min')
        else:
            for kk, p in LT.lora_pairs(lm).items():
                a0, b0 = lora_init[kk]
                p['a'].value = jax.device_put(a0, p['a'].value.sharding)
                p['b'].value = jax.device_put(b0, p['b'].value.sharding)
            del lora_init
            # fresh optimizer state in place (Adam count/mu/nu = 0 is its init): a new nnx.Optimizer object would make
            # nnx.jit retrace and recompile every length (measured on CPU: 9 s per length; minutes on the TPU)
            ost = nnx.state(S['opt'], nnx.OptState)
            nnx.update(S['opt'], jax.tree.map(jnp.zeros_like, ost))
            log('phase2_start', lr=phase2['lr'], epochs=phase2.get('epochs', A['epochs']), source_contains=sub, rows=n_sub,
                left_min=round(train_left_s() / 60, 1), b_absmax_after_reset=float(max(
                    float(jnp.abs(p['b'].value).max()) for p in LT.lora_pairs(lm).values())),
                opt_absmax_after_reset=float(max([float(jnp.abs(x).max()) for x in jax.tree.leaves(
                    nnx.state(S['opt'], nnx.OptState)) if hasattr(x, 'size') and x.size] or [0.0])))
            if DPO and pairs and phase2.get('dpo'):  # arm b's preference settings; the reference (B = 0) is unchanged
                DPO.update(phase2['dpo'])
                set_targets('b')
            if HEAL and phase2.get('heal'):  # arm b's healing objective (kd False: the plain next-token loss)
                HEAL.update(phase2['heal'])
                S['step'] = build_step()
                log('heal_arm', arm='b', kd=bool(HEAL.get('kd')), temp=HEAL.get('temp', 1.0))
            sfx = phase2.get('suffix', '_b')
            plan_b = make_plan(phase2.get('epochs', A['epochs']), only)
            if phase2.get('match_a'):  # a paired ablation: arm b runs exactly arm a's steps (same rows, same order)
                del plan_b[len(plan_a):]
            run_phase('b', plan_b, phase2['lr'], adapter_out + sfx, adapters_dir + sfx)
