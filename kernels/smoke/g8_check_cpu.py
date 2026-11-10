"""One-shot insurance for the G8 TPU run, on a Kaggle CPU machine (no TPU queue): everything that can be checked
without a TPU, with the real checkpoint, tokenizer, rendered rows and G7 adapter mounted exactly as the TPU kernel
mounts them.
1. env: the TPU kernel's exact pins in a uv Python 3.11 venv (+ CPU torch); jax runs on CPU (JAX_PLATFORMS=cpu).
2. read speed of the mounted 62 GB checkpoint: one thread vs the TPU kernel's 32-thread pread read-ahead (first GB).
3. the token dump (train_lora.py --dump-npz, the TPU kernel's exact command): rows, tokens, trainable, length buckets.
4. HF-vs-Tunix parity on the real weights (parity_hf_tunix.py): checkpoint layers 0 (sliding) and 5 (full attention),
   one real rendered row of >= 1536 tokens (beyond the 1024 sliding window): base forward, trainer loss vs HF CE,
   exported LoRA merged into HF vs the Qwix LoRA model, our PEFT keys/shapes vs G7's for those layers, and Tunix's
   gemma4_31b preset (used by the TPU runner) vs the checkpoint's config.json.
5. trace-time canary (aot_check_tpu.py): the full 60-layer train step traced and lowered for TPU at 8192/12288 for
   every attention rung, mesh fsdp 2 x tp 4 on 8 fake CPU devices.
Outputs: /kaggle/working/g8check/{report.json, log.jsonl, parity.json, aot.json, data_stats.json}.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PINS = ['jax[tpu]==0.10.2', 'libtpu==0.0.42.1', 'google-tunix==0.1.7', 'qwix==0.1.8', 'flax==0.12.8', 'optax==0.2.8',
        'safetensors==0.7.0', 'numpy==2.4.6', 'transformers==5.12.1', 'tokenizers==0.22.2',
        'sentencepiece==0.2.2']  # @define
TORCH_PIN = 'torch==2.14.1'  # @define
SFT_NAME = 'render/sft.jsonl'  # @define
PARITY_LEN = 1536  # @define
PARITY_LAYERS = '0,5'  # @define
READ_GB = 1  # @define
LORA_TPU = None  # @embed ../train/lora_tpu.py
TRAIN_PY = None  # @embed ../train/train_lora.py
PARITY = None  # @embed ../train/parity_hf_tunix.py
AOT = None  # @embed ../train/aot_check_tpu.py

T0 = time.time()
OUT = Path('/kaggle/working/g8check')
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path('/tmp/g8c')
WORK.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'log.jsonl', 'a')
REPORT = {}


def stage(name, **kv):
    kv = {'stage': name, 't_min': round((time.time() - T0) / 60, 2), **kv}
    LOG.write(json.dumps(kv, default=str) + '\n')
    LOG.flush()
    print(json.dumps(kv, default=str), flush=True)
    REPORT[name] = kv
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=1, default=str))


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return Path(hits[0]) if hits else None


def sh(cmd, timeout=None, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return r.returncode, (r.stdout[-6000:] + r.stderr[-4000:])
    except subprocess.TimeoutExpired as e:
        return 'timeout', str(e)[-2000:]


def host():
    out = {}
    try:
        mem = dict(x.split(':', 1) for x in open('/proc/meminfo').read().splitlines() if ':' in x)
        out.update({k: round(int(mem[k].split()[0]) / 2**20, 1) for k in ('MemTotal', 'MemAvailable') if k in mem})
    except Exception as e:
        out['mem_err'] = str(e)[:100]
    out['cpus'] = os.cpu_count()
    return out


stage('start', host=host(), pins=PINS)
# ---- 1. environment (the TPU kernel's commands) ----
os.environ['UV_HTTP_TIMEOUT'] = '300'
rc, out = sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
VENV = Path('/tmp/venv311')
rc, out = sh(['uv', 'venv', '-p', '3.11', str(VENV)]) if rc == 0 else (rc, out)
PY = str(VENV / 'bin' / 'python')
rc1, out1 = sh(['uv', 'pip', 'install', '-p', PY, *PINS])
rc2, out2 = sh(['uv', 'pip', 'install', '-p', PY, TORCH_PIN, '--index-url', 'https://download.pytorch.org/whl/cpu'])
rc3, out3 = sh([PY, '-c', 'import jax, tunix, qwix, flax, optax, transformers, torch; '
                'print(jax.__version__, flax.__version__, optax.__version__, transformers.__version__, torch.__version__)'],
               env=dict(os.environ, JAX_PLATFORMS='cpu'))
stage('env', rc=[rc, rc1, rc2, rc3], versions=out3.strip()[-300:], pip_tail=out1[-800:])
if rc1 or rc2 or rc3:
    raise SystemExit('environment failed')

# ---- inputs ----
model = (find('gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')
         or find('gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')).parent
sft = find(SFT_NAME)
ref = find('adapter_out/adapter_model.safetensors')
stage('inputs', model=str(model), sft=str(sft), ref=str(ref),
      shards=[(p.name, round(p.stat().st_size / 2**30, 2)) for p in sorted(model.glob('*.safetensors'))])

# ---- 2. read speed of the mounted checkpoint ----
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
files = sorted(model.glob('*.safetensors'))


def read_range(f, o, n):
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


try:
    n = READ_GB * 2**30
    t = time.time()
    b1 = read_range(files[-1], 0, n)  # last shard: not touched before
    s1 = time.time() - t
    chunk = 128 * 2**20
    t = time.time()
    with ThreadPoolExecutor(32) as ex:
        b32 = sum(ex.map(lambda o: read_range(files[0], o, chunk), range(0, n, chunk)))
    s32 = time.time() - t
    stage('read_speed', one_thread_mb_s=round(b1 / 2**20 / s1), threads32_mb_s=round(b32 / 2**20 / s32),
          gb_each=READ_GB, implied_62gb_min_one=round(62 * 1024 / (b1 / 2**20 / s1) / 60, 1),
          implied_62gb_min_32=round(62 * 1024 / (b32 / 2**20 / s32) / 60, 1))
except Exception as e:
    stage('read_speed', error=f'{type(e).__name__}: {e}'[:300])

# ---- 3. token dump, exactly as the TPU kernel runs it ----
(WORK / 'train_lora.py').write_text(TRAIN_PY)
npz = WORK / 'rows.npz'
rc, out = sh([PY, str(WORK / 'train_lora.py'), '--model', str(model), '--data', str(sft), '--out', str(OUT / 'dump'),
              '--train-on', 'actions', '--mask-mistakes', '--max-tokens', '12288', '--dump-npz', str(npz)])
stats = {}
if rc == 0:
    import numpy as np
    d = np.load(npz)
    L = d['lens']
    stats = {'rows': int(len(L)), 'tokens': int(L.sum()), 'trainable': int((d['labels'] != -100).sum()),
             'le_8192': int((L <= 8192).sum()), 'gt_8192': int((L > 8192).sum()), 'max_len': int(L.max()),
             'max_targets_per_row': int(max(((d['labels'][o:o + n] != -100).sum()) for o, n in
                                            zip(np.concatenate([[0], np.cumsum(L)[:-1]]), L))),
             'sources': {str(k): int(v) for k, v in zip(*np.unique(d['sources'], return_counts=True))}}
    json.dump(stats, open(OUT / 'data_stats.json', 'w'), indent=1)
stage('dump', rc=rc, stats=stats, out=out[-800:])

# ---- 4. HF vs Tunix parity on real weights ----
(WORK / 'lora_tpu.py').write_text(LORA_TPU)
(WORK / 'parity_hf_tunix.py').write_text(PARITY)
(WORK / 'aot_check_tpu.py').write_text(AOT)
env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONUNBUFFERED='1')
rc, out = sh([PY, '-W', 'ignore', str(WORK / 'parity_hf_tunix.py'), '--ckpt', str(model), '--layer-ids', PARITY_LAYERS,
              '--len', str(PARITY_LEN), *(['--npz', str(npz)] if npz.exists() else []),
              *(['--ref', str(ref)] if ref else []), '--out', str(OUT / 'parity.json')], timeout=4 * 3600, env=env)
par = json.load(open(OUT / 'parity.json')) if (OUT / 'parity.json').exists() else {}
stage('parity', rc=rc, ok=par.get('ok'), **{k: par.get(k) for k in (
    'hidden_rel', 'logits_rel', 'top1_agree', 'loss_trainer', 'loss_hf', 'abs_diff', 'lora_moved_rel', 'ref_match',
    'preset_ok', 'preset_mismatch', 'T')}, tail=[x for x in out.splitlines() if x.startswith(('{', 'PARITY', 'Trace',
                                                                                         'Assert', 'Value'))][-12:])

# ---- 5. trace-time canary of the full 60-layer step for TPU ----
rc, out = sh([PY, '-W', 'ignore', str(WORK / 'aot_check_tpu.py'), '--out', str(OUT / 'aot.json')], timeout=3 * 3600,
             env=dict(env, XLA_FLAGS='--xla_force_host_platform_device_count=8'))
aot = json.load(open(OUT / 'aot.json')) if (OUT / 'aot.json').exists() else {}
stage('aot', rc=rc, all_ok=bool(aot.get('checks')) and all(c['ok'] for c in aot.get('checks', [])),
      checks=[{k: c.get(k) for k in ('len', 'rung', 'ok', 'error')} for c in aot.get('checks', [])],
      lora_params=aot.get('lora_params'), tail=out.splitlines()[-3:])
stage('done', host=host(), total_min=round((time.time() - T0) / 60, 1),
      verdict={'env': REPORT['env']['rc'] == [0, 0, 0, 0], 'dump': REPORT['dump']['rc'] == 0,
               'parity': bool(REPORT['parity'].get('ok')), 'aot': REPORT['aot']['all_ok']})
