"""LoRA SFT of Gemma 4 31B on a Kaggle TPU v5e-8 (JAX + Tunix's Gemma 4 port + Qwix LoRA).

Same data, tokens and loss masks as the GPU trainer (train_lora.py --dump-npz on the rendered rows), same LoRA targets
(q/k/v/o/gate/up/down, rank 16, alpha 32) and the same PEFT key layout, so the adapter drops into the submission and
the GPU paired eval unchanged. Base weights: the bf16 QAT checkpoint (gemma-4-31b-it-qat-q4_0-unquantized) in bf16,
sharded fsdp 2 x tp 4 (tp <= 4: full-attention layers have 4 KV heads); splash attention; decoder-level remat; loss at
the labelled positions only (lora_tpu.target_loss).

MODE 'canary': a few steps per length bucket (compile time, step time, tokens/s, peak HBM), export, compare the
adapter's keys/shapes/dtypes with the GPU adapter of g4dev-train-g7-gpu, write adapter_out/. MODE 'train': EPOCHS over
the rows, re-planned to the time left (TRAIN_CAP_MIN, DEADLINE_S), exporting every SAVE_EVERY_MIN minutes, per epoch
and at the end; fallbacks for attention-kernel and HBM failures (see train_tpu_runner.py). LoRA params are fp32
(bf16 compute), as PEFT keeps them on the GPU.
Outputs: /kaggle/working/tpu/{log.jsonl, report.json, data_stats.json}, /kaggle/working/adapter_out/ (latest good
adapter), /kaggle/working/adapters/ep<N>/ (per epoch); with PHASE2_LR, the same under adapter_out_b/, adapters_b/. After the runner starts the notebook itself never fails, so the
version stays COMPLETE and its output can be mounted by the GPU eval.
HEAL (pruner part 3): instead of the SFT rows, heal_data.py builds every-token rows from the agent rows, the pruner's
held-out rows and the channel cut (scores under /kaggle/input); the runner trains the cut model's LoRA toward the full
model (KL, arm a) or on the plain next-token loss (PHASE2_HEAL, arm b) and evaluates both on the held-out rows. The cut
(heal_mask.npz) and the data summary (heal_data.json) are copied to the output for the GPU speed/quality test.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = 'canary'  # @define
DEADLINE_S = 3000  # @define (whole notebook)
TRAIN_CAP_MIN = 30  # @define
LR = 5e-5  # @define
RANK = 16  # @define
EPOCHS = 1  # @define
BUCKETS = '8192,12288'  # @define
BATCH = 2  # @define (rows per step; must be a multiple of the fsdp axis)
MESH = '2,4'  # @define (fsdp, tp)
MAX_TARGETS = 4096  # @define (trainable positions per row kept; the rest are dropped and counted)
LOSS_CHUNK = 512  # @define
CANARY_STEPS = 2  # @define (per bucket)
SAVE_EVERY_MIN = 20  # @define
FLASH = True  # @define
FLASH_BLOCK = 1024  # @define (splash block; the ladder falls back to 512, 256, then eager at 8192)
LADDER = True  # @define
MASTER_FP32 = True  # @define
PHASE2_LR = 0.0  # @define (> 0: after phase a, retrain from the same LoRA init and data order at this lr -> *_b dirs)
PHASE2_EPOCHS = 3  # @define
PHASE2_MIN_MIN = 60  # @define (phase 2 runs only with at least this many training minutes left)
PHASE2_SOURCE = ''  # @define (non-empty: phase 2 trains only the rows whose source tag contains it)
PHASE2_DPO = {}  # @define (overrides of DPO for phase 2: a second preference arm on the same data and reference)
PHASE2_A_SHARE = 0.0  # @define (> 0: phase a stops at this share of the training time left when it starts)
PHASE2_MATCH_A = False  # @define (phase 2 runs exactly phase a's steps: a paired ablation)
# DPO: non-empty (e.g. {'beta': 2.0, 'sft_chosen': 1.0, 'reject_floor': -0.1}) -> rows tagged 'dpo:<task>:c|r'
# (kernels/analysis/dpo_pairs.py) get per-epoch signed preference weights (B127, B154)
DPO = {}  # @define
INIT_ADAPTER = ''  # @define (a path pattern under /kaggle/input: start from this adapter, e.g. G11a's, instead of B = 0)
# HEAL: non-empty -> heal a channel-cut model (pruner part 3), e.g. {'scores': 'prune/score-v*/scores.npz', 'keep': 0.7,
# 'multiple': 512, 'kd': True, 'temp': 1.0, 'positions': 'all', 'eval_rows': 40, 'eval_len': 8192}
HEAL = {}  # @define
PHASE2_HEAL = {}  # @define (overrides of HEAL for phase 2, e.g. {'kd': False}: the same steps on the next-token loss)
RANDOM_INIT = False  # @define (canary: random weights, skips the 62 GB read)
PREFETCH_GB = 0  # @define (0 off, -1 whole checkpoint, N first N GB: parallel read-ahead of the mounted weights)
PREFETCH_WAIT_MIN = 30  # @define (longest wait for the read-ahead before loading anyway)
STALL_MIN = 75  # @define (runner killed after this long without a new log line; the 62 GB load took 38 min in v2)
TRAIN_ON = 'actions'  # @define
MASK_MISTAKES = True  # @define
SFT_NAME = 'render/sft.jsonl'  # @define
KEEP_MAX_CALLS = 0  # @define (> 0: train only on runs that resolved the task within this many tool calls)
SEED = 0  # @define
# exact versions resolved by canary v2 (10-13), so a later run gets the same stack
PINS = ['jax[tpu]==0.10.2', 'libtpu==0.0.42.1', 'google-tunix==0.1.7', 'qwix==0.1.8', 'flax==0.12.8', 'optax==0.2.8',
        'safetensors==0.7.0', 'numpy==2.4.6', 'transformers==5.12.1', 'tokenizers==0.22.2',
        'sentencepiece==0.2.2']  # @define
TORCH_PIN = 'torch==2.14.1'  # @define (CPU wheel; train_lora.py --dump-npz imports torch)
LORA_TPU = None  # @embed ../train/lora_tpu.py
TRAIN_PY = None  # @embed ../train/train_lora.py
RUNNER = None  # @embed ../train/train_tpu_runner.py
HEAL_DATA = None  # @embed ../train/heal_data.py

T0 = time.time()
WORKING = Path('/kaggle/working')
OUT = WORKING / 'tpu'
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/tmp/g8')
WORK.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'log.jsonl', 'a')


def stage(name, **kv):
    kv = {'stage': name, 't_min': round((time.time() - T0) / 60, 2), **kv}
    LOG.write(json.dumps(kv, default=str) + '\n')
    LOG.flush()
    print(json.dumps(kv, default=str), flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return Path(hits[0]) if hits else None


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout[-3000:] + r.stderr[-3000:])


# ---- 0. environment: Python 3.11 venv (Tunix needs >= 3.11) with pinned JAX/Tunix/Qwix and CPU torch ----
def host():
    out = {}
    try:
        mem = dict(x.split(':', 1) for x in open('/proc/meminfo').read().splitlines() if ':' in x)
        out.update({k: round(int(mem[k].split()[0]) / 2**20, 1) for k in ('MemTotal', 'MemAvailable', 'Cached')
                    if k in mem})
    except Exception as e:
        out['mem_err'] = str(e)[:100]
    import shutil
    for d in ('/kaggle/working', '/tmp', '/dev/shm'):
        try:
            out['free_gb:' + d] = round(shutil.disk_usage(d).free / 2**30, 1)
        except Exception:
            pass
    out['cpus'] = os.cpu_count()
    return out


stage('start', mode=MODE, python=sys.version.split()[0], pins=PINS, host=host())
os.environ['UV_HTTP_TIMEOUT'] = '300'  # rb2: uv's 30 s default timed out on a large wheel
rc, out = sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
VENV = Path('/tmp/venv311')
rc, out = sh(['uv', 'venv', '-p', '3.11', str(VENV)]) if rc == 0 else (rc, out)
stage('venv', rc=rc, out=out[-500:])
PY = str(VENV / 'bin' / 'python')
rc, out = sh(['uv', 'pip', 'install', '-p', PY, *PINS])
stage('pip', rc=rc, out=out[-1500:])
rc2, out2 = sh(['uv', 'pip', 'install', '-p', PY, TORCH_PIN, '--index-url', 'https://download.pytorch.org/whl/cpu'])
stage('torch_cpu', rc=rc2, out=out2[-500:])
if rc or rc2:
    raise SystemExit('environment failed')

# ---- 1. inputs ----
qat_bf16 = (find('gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')
            or find('gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')).parent
sft = find(SFT_NAME)
ref = find('adapter_out/adapter_model.safetensors')
init_ad = find(INIT_ADAPTER) if INIT_ADAPTER else None
if INIT_ADAPTER and not init_ad:
    raise SystemExit(f'INIT_ADAPTER {INIT_ADAPTER!r} not found under /kaggle/input')  # before the 62 GB load, not after
scores = find(HEAL['scores']) if HEAL else None
if HEAL and not scores:
    raise SystemExit(f"HEAL scores {HEAL['scores']!r} not found under /kaggle/input")
if KEEP_MAX_CALLS:
    rows = [json.loads(x) for x in open(sft) if x.strip()]
    keep = [r for r in rows if r.get('tool_calls') is None or r['tool_calls'] <= KEEP_MAX_CALLS]
    if len(keep) >= max(40, len(rows) // 4):  # never train on a sliver: below that, keep every row
        sft = WORK / 'sft_kept.jsonl'
        sft.write_text(''.join(json.dumps(r) + '\n' for r in keep))
    stage('filter', rows=len(rows), kept=len(keep), used=str(sft), max_calls=KEEP_MAX_CALLS,
          unknown_calls=sum(1 for r in rows if r.get('tool_calls') is None),
          kept_tokens=sum(r.get('n_tokens') or 0 for r in keep), all_tokens=sum(r.get('n_tokens') or 0 for r in rows))
    del rows, keep
stage('inputs', model=str(qat_bf16), sft=str(sft), ref_adapter=str(ref), init_adapter=str(init_ad) if init_ad else None,
      heal_scores=str(scores) if scores else None,
      model_files=sorted(p.name for p in qat_bf16.iterdir())[:12])

# ---- 1b. parallel read-ahead of the checkpoint (the 62 GB read took 38 min page by page in canary v2) ----
import threading
from concurrent.futures import ThreadPoolExecutor
PF = {'bytes': 0, 'sec': None}


def prefetch():
    files = sorted(qat_bf16.glob('*.safetensors'))
    limit = None if PREFETCH_GB < 0 else int(PREFETCH_GB * 2**30)
    chunk = 128 * 2**20
    jobs = []
    for f in files:
        size = f.stat().st_size
        jobs += [(f, o, min(chunk, size - o)) for o in range(0, size, chunk)]
    if limit is not None:
        jobs = jobs[:max(1, limit // chunk)]
    t = time.time()

    def read(job):
        f, o, n = job
        fd = os.open(f, os.O_RDONLY)
        try:
            done = 0
            while done < n:
                got = len(os.pread(fd, min(n - done, 16 * 2**20), o + done))
                if not got:
                    break
                done += got
            return done
        finally:
            os.close(fd)
    with ThreadPoolExecutor(32) as ex:
        for d in ex.map(read, jobs):
            PF['bytes'] += d
    PF['sec'] = time.time() - t
    stage('prefetch_done', gb=round(PF['bytes'] / 2**30, 1), s=round(PF['sec'], 1),
          mb_s=round(PF['bytes'] / 2**20 / max(PF['sec'], 1e-6)), files=len(files))


pf_thread = None
if PREFETCH_GB and not RANDOM_INIT or (PREFETCH_GB > 0 and RANDOM_INIT):
    pf_thread = threading.Thread(target=prefetch, daemon=True)
    pf_thread.start()

# ---- 2. tokens + labels exactly as the GPU trainer makes them ----
(WORK / 'train_lora.py').write_text(TRAIN_PY)
npz = WORK / 'rows.npz'
if HEAL:  # every-token rows, the pruner's held-out rows and the channel cut (heal_data.py)
    import shutil
    (WORK / 'heal_data.py').write_text(HEAL_DATA)
    ha = {'sft': str(sft), 'tokenizer': str(qat_bf16), 'scores': str(scores), 'out_dir': str(WORK), 'keep': HEAL['keep'],
          'multiple': HEAL.get('multiple', 512), 'max_len': max(int(b) for b in BUCKETS.split(',')),
          'eval_rows': HEAL.get('eval_rows', 40), 'eval_len': HEAL.get('eval_len', 8192), 'seed': SEED,
          'positions': HEAL.get('positions', 'all'), 'score_key': HEAL.get('score_key', 'agent_ff')}
    rc, out = sh([PY, str(WORK / 'heal_data.py'), json.dumps(ha)])
    stage('heal_data', rc=rc, out=out[-2500:])
    if rc:
        raise SystemExit('heal data failed')
    shutil.copy(WORK / 'mask.npz', OUT / 'heal_mask.npz')
    shutil.copy(WORK / 'heal_data.json', OUT / 'heal_data.json')
else:
    rc, out = sh([PY, str(WORK / 'train_lora.py'), '--model', str(qat_bf16), '--data', str(sft), '--out', str(OUT / 'dump'),
                  '--train-on', TRAIN_ON, *(['--mask-mistakes'] if MASK_MISTAKES else []),
                  '--max-tokens', str(max(int(b) for b in BUCKETS.split(','))), '--dump-npz', str(npz)])
    stage('dump', rc=rc, out=out[-1500:])
    if rc:
        raise SystemExit('token dump failed')

# ---- 3. train (runner in the venv) ----
if pf_thread:  # the runner's loader then finds the weights in the page cache
    pf_thread.join(timeout=PREFETCH_WAIT_MIN * 60)  # bounded: a slow read-ahead must not eat the training time
    stage('prefetch_wait', done=not pf_thread.is_alive(), gb=round(PF['bytes'] / 2**30, 1), host=host())
(WORK / 'lora_tpu.py').write_text(LORA_TPU)
(WORK / 'runner.py').write_text(RUNNER)
args = {'t0': T0, 'deadline_s': DEADLINE_S, 'out': str(OUT), 'npz': str(npz), 'model': str(qat_bf16),
        'buckets': BUCKETS, 'flash': FLASH, 'mesh': MESH, 'rank': RANK, 'seed': SEED, 'epochs': EPOCHS, 'batch': BATCH,
        'mode': MODE, 'canary_steps': CANARY_STEPS, 'lr': LR, 'loss_chunk': LOSS_CHUNK, 'max_targets': MAX_TARGETS,
        'cap_min': TRAIN_CAP_MIN, 'save_every_min': SAVE_EVERY_MIN, 'reserve_s': 300,
        'adapter_out': '/kaggle/working/adapter_out', 'adapters_dir': '/kaggle/working/adapters',
        'ref': str(ref) if ref else '', 'random_init': RANDOM_INIT, 'flash_block': FLASH_BLOCK, 'ladder': LADDER,
        'master_fp32': MASTER_FP32, 'init_adapter': str(init_ad) if init_ad else '',
        'phase2': {'lr': PHASE2_LR, 'epochs': PHASE2_EPOCHS, 'min_min': PHASE2_MIN_MIN, 'source_contains': PHASE2_SOURCE,
                   'dpo': PHASE2_DPO, 'a_share': PHASE2_A_SHARE, 'match_a': PHASE2_MATCH_A,
                   'heal': PHASE2_HEAL} if PHASE2_LR > 0 else None,
        'heal': ({'mask': str(WORK / 'mask.npz'), 'eval_npz': str(WORK / 'eval.npz'), 'kd': bool(HEAL.get('kd', True)),
                  'temp': float(HEAL.get('temp', 1.0)), 'eval_len': int(HEAL.get('eval_len', 8192))} if HEAL else None),
        'dpo': ({**DPO, 'pairs_meta': str(Path(find(SFT_NAME)).parent / 'pairs.json')} if DPO else None)}  # pair costs (B154)
env = dict(os.environ, PYTHONUNBUFFERED='1',
           XLA_FLAGS=(os.environ.get('XLA_FLAGS', '') + ' --xla_llvm_disable_expensive_passes=true').strip())  # compile ~8 -> ~3 min (arXiv 2605.25645)
# From here on nothing raises: whatever the runner did, the adapters it exported and the report stay in the output.
rc = None
try:
    p = subprocess.Popen([PY, str(WORK / 'runner.py'), json.dumps(args)], env=env)
    why = None
    while rc is None:
        try:
            rc = p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if time.time() - T0 > DEADLINE_S - 120:
                why = 'deadline'
            elif time.time() - os.path.getmtime(OUT / 'log.jsonl') > STALL_MIN * 60:
                why = f'no log line for {STALL_MIN} min'
            if why:
                p.terminate()
                try:
                    rc = p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = 'killed'
                stage('runner_timeout', rc=rc, why=why)
except Exception as e:
    stage('runner_launch_error', error=str(e)[:500])
stage('runner_exit', rc=rc, host=host())


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def make_report(rc):
    report = {'mode': MODE, 'rc': rc, 'total_min': round((time.time() - T0) / 60, 1)}
    try:
        lines = []
        for x in open(OUT / 'log.jsonl'):
            try:
                lines.append(json.loads(x))
            except Exception:
                pass
        steps = [x for x in lines if x.get('stage') == 'step']
        steady = [x for x in steps if not x.get('compile')]
        eps = sorted({(x.get('phase', 'a'), x['ep']) for x in steps if x.get('ep') is not None})
        lens = sorted({x.get('as_len', x['bucket']) for x in steady})
        report.update({
            'steps': len(steps), 'steps_applied': sum(1 for x in steps if x.get('applied', True)),
            'first_loss': steps[0]['loss'] if steps else None, 'last_loss': steps[-1]['loss'] if steps else None,
            'loss_by_epoch': {f'{ph}:ep{e}': round(median([x['loss'] for x in steps
                                                           if x.get('ep') == e and x.get('phase', 'a') == ph]), 4)
                              for ph, e in eps},  # median per epoch
            'step_s_median_by_len': {L: median([x['s'] for x in steady if x.get('as_len', x['bucket']) == L])
                                     for L in lens},
            'tok_s_median': median([x['tok_s'] for x in steady]),
            'compile_s': [(x.get('as_len', x['bucket']), x.get('rung'), x['s']) for x in steps if x.get('compile')],
            'peak_hbm_gb': max((x.get('peak_gb') or 0 for x in lines), default=None),
            'exports': [{k: x.get(k) for k in ('tag', 'dir', 'match', 'bytes', 'b_absmax', 't_min')}
                        for x in lines if x.get('stage') == 'export'],
            'events': [{k: v for k, v in x.items() if k != 'tb'} for x in lines if x.get('stage') in (
                'stop', 'step_error', 'rung', 'demote', 'replan_cut', 'export_refused', 'no_final_export',
                'runner_timeout', 'phase2_start', 'phase2_skipped', 'export_error', 'replan_error')],
            'done': [x for x in lines if x.get('stage') == 'done'],
        })
        if HEAL:  # held-out nats per token: full model (teacher) and cut model (student) before, per epoch, after
            report['heal'] = {'mask': next(({k: v for k, v in x.items() if k not in ('stage', 't_min')}
                                            for x in lines if x.get('stage') == 'heal_mask'), None),
                              'evals': [{k: v for k, v in x.items() if k != 'stage'} for x in lines
                                        if x.get('stage') in ('heal_eval', 'heal_eval_error')],
                              'arms': {'a': {'kd': bool(HEAL.get('kd', True))}, 'b': PHASE2_HEAL}}
            try:
                report['heal']['data'] = json.loads((OUT / 'heal_data.json').read_text())
            except Exception:
                pass
        report['adapters'] = {str(q.parent.relative_to(WORKING)): q.stat().st_size
                              for q in WORKING.glob('**/adapter_model.safetensors')}
    except Exception as e:
        report['report_error'] = f'{type(e).__name__}: {e}'[:500]
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return report


make_report(rc)
