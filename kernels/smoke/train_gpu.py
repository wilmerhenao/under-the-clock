"""Train a QLoRA adapter for Gemma 4 31B on agent trajectories, then measure it against the base model.

1. Train (separate process, so GPU memory is fully released): QAT bf16 weights loaded in 4-bit NF4.
2. Serve the competition model (w4a16) with the adapter via vLLM multi-LoRA, exactly like scoring would.
3. Logprob check: base vs adapter must differ on a fixed prompt (catches silently-zeroed adapters).
4. Held-out tasks: base and adapter evaluated side by side in the same session/environment.
Fails fast at any stage; no fallbacks.
"""
import collections
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
TRAIN_PY = None  # @embed ../train/train_lora.py
SLICE_SRC = None  # @embed ../colab/prune/slice_model.py
TOLERANT_SRC = None  # @embed ../colab/prune/gemma4_tolerant.py
BUNDLES = None  # @embed_dir ../bundles/sub-v5b ../bundles/sub-v9 ../bundles/sub-v9c ../bundles/sub-v9t ../bundles/sub-v9r ../bundles/sub-v9k ../bundles/sub-v9g
DEADLINE_S = 3 * 3600  # @define
TRAIN_CAP_MIN = 75  # @define
ENV_ARGS = []  # @define  ('@kit' / '@extra' / '@wh313' / '@rw' resolved at runtime)
EVAL_REPOGRAPH = ''  # @define ('priv_eval': EVAL_SET runs on debmalya84/repograph-verified-swe-tasks, ids trusted by
                    # g4dev-repograph-controls-cpu minus REPOGRAPH_EXCLUDE repos; canaries stay on the competition data)
REPOGRAPH_EXCLUDE = []  # @define (repos in our training pools, docs/external_datasets.md)
EVAL_MAX_TASKS = 0  # @define (> 0: at most this many EVAL_SET tasks, a seeded random subset)
EPOCHS = 2  # @define
MAX_EXAMPLES = 0  # @define (0 = all)
EVAL_SET = 'heldout'  # @define ('canary' = the two canary tasks, for smoke tests)
EVAL_CONC = 12  # @define
SFT_NAME = 'sft_dev_v1.jsonl'  # @define (training file, found anywhere under /kaggle/input)
PATCH_REASONING = False  # @define (True: make vLLM read ADK's `reasoning_content`; the scorer's wheel decides, so default off)
PROBE_LENS = ''  # @define (e.g. '8192,16384,24576': memory probe before training)
VAL_NAME = ''  # @define (held-out JSONL for validation loss before/after training, e.g. 'mix/val.jsonl')
VAL_N = 24  # @define
GRAD_ACCUM = 4  # @define
SCHED = 'steps'  # @define ('time': cosine over TRAIN_CAP_MIN, for runs stopped by the time cap)
SAVE_EVERY_MIN = 0  # @define (also checkpoint the adapter every N training minutes)
MAX_MEM = ''  # @define (per-GPU GiB caps for the trainer's device map, weights only; '' = 19 each)
DEVICE_MAP = 'auto'  # @define (auto: caps bind only on the last GPU; sequential: caps decide the layer split)
LAYER_SPLIT = ''  # @define (decoder layers per GPU, e.g. '3,19,19,19'; auto gave 0/18/18/24)
PROBE_REAL = False  # @define (also probe the longest real line under each probe length)
PROBE_ONLY = False  # @define (stop after the memory probe: no training, serving or eval)
REPLICAS = 1  # @define (2: data-parallel replicas on GPUs 0,1 and 2,3, REPLICA_SPLIT decoder layers per GPU; ~2x tok/s)
REPLICA_SPLIT = '26,34'  # @define (GPU 0 of each replica also holds the embeddings/tied lm_head, ~2.8 GB)
ADAPTER_FROM = ''  # @define (glob under /kaggle/input of a trained adapter dir: skip training, serve and evaluate it)
ADAPTER_B_FROM = ''  # @define (glob of a second trained adapter: arm 'base' = ADAPTER_FROM's adapter, arm 'adapter' = this one; adapter A/B on one server)
ADAPTER_B_ALPHA = 0  # @define (> 0: serve the second adapter with this lora_alpha, e.g. half strength = 16 for r16/alpha32)
# CUT (pruner, B175/B176): serve a channel-cut copy of the competition model instead of the full one, e.g. {'scores':
# '**/prune/score-v*/scores.npz', 'keep': 0.7, 'multiple': 512, 'group': 1, 'mask': '**/tpu/heal_mask.npz'}: cut by
# slice_model.make_plan (the speed test's cut), checked against the healing adapter's training mask if one is mounted;
# the adapters (full-size PEFT) are cut to the same shape. Arm 'base' is then the cut model without adapter.
CUT = {}  # @define
# MODELS (B192): a same-session comparison of different checkpoints, each served alone (TP 4, no LoRA) on the same
# tasks, e.g. [{'name': 'full'}, {'name': 'c70', 'cut': {'scores': '**/prune/score-v*/scores.npz', 'keep': 0.7}},
# {'name': 'c70kd', 'cut': {...same cut..., 'mask': '**/tpu/heal_mask.npz'}, 'merge': '**/<healing nb>/adapter_out'}]
# ('merge': a healing adapter merged into the cut's 4-bit weights, slice_model.merge_lora_w4). The EVAL_SET tasks, in
# the harness's own order (--order-seed 0), are cut into BLOCKS blocks of BLOCK_TASKS; even blocks run the models in the
# listed order, odd blocks in reverse (A B C | C B A), so neither order nor time in the session favours a model. No
# training, no LoRA server, no canaries: a segment that finishes no task stops the run.
MODELS = []  # @define
BLOCKS = 2  # @define
BLOCK_TASKS = 64  # @define
SEG_CAP_MIN = 95  # @define (wall-clock cap of one segment, BLOCK_TASKS tasks with EVAL_CONC at a time)
BLOCK_EXCLUDE_REPOS = []  # @define (MODELS mode: repos dropped from the blocks, e.g. ones found in a training pool)
BLOCK_IDS = {}  # @define (MODELS mode, B209: fixed blocks {'block0': [instance ids], ...} instead of the computed ones,
               # e.g. an earlier session's exact tasks; every id must be in EVAL_SET)
TRAIN_ONLY = False  # @define (stop after training: adapter to /kaggle/working/adapter_out, no serving or eval; the
                    # g5b adapter from this trainer was served on the current wheels by g4dev-runtime-paired-gpu, 10-10)
LOSS_CHUNK = 1024  # @define (trainable positions per logits chunk)
LR = 1e-4  # @define (LoRA learning rate; recipe v2 plans 5e-5)
TRAIN_ON = 'all'  # @define ('actions' = loss on tool-call arguments only; thoughts and text stay context)
MASK_MISTAKES = False  # @define (with TRAIN_ON 'actions': no loss on calls the next thought calls a mistake, B38)
SKIP_KINDS = ''  # @define (comma-separated row kinds left out of training and validation, e.g. 'summary')
TRAIN_EXTRA_MIN = 40  # @define (process timeout = TRAIN_CAP_MIN + this: model load, probe, validation)
PROBE_GATE = False  # @define (True: train only on lengths the probe fits; stop before training if none fits)
EVAL_BUNDLE = ''  # @define (e.g. 'sub-v5b': evaluate through the deployment bundle: its prompt, sampling and caps)
EVAL_BUNDLE_B = ''  # @define (non-empty: the second arm is the BASE model through this bundle instead of the adapter: a prompt A/B on one server)
EVAL_AB_ADAPTER = False  # @define (with EVAL_BUNDLE_B: both arms use the adapter, bundle A vs bundle B)
EVAL_TIME_SCALE = 1.0  # @define (with EVAL_BUNDLE: x max_time_minutes, concurrent runs decode slower than the scorer)
EVAL_PASSES = 1  # @define (passes over EVAL_SET; base and adapter side by side in each)
EVAL_GATE = False  # @define (run EVAL_SET only if validation loss fell, the adapter moves logprobs, both canaries are healthy)
GATE_REASONING_MIN = 0.5  # @define (with EVAL_GATE: adapter/base per-step text p90 on the canaries; G2b: 0.11, B28)
GATE_MAX_REPEAT = 5  # @define (with EVAL_GATE: fail if the adapter repeats one call this often with no edit in between; G2b: 26)
ADAPTER = 'g4dev_sft_v1'
ADAPTER_B = 'g4dev_sft_b'
RANDOM_ADAPTER = 'g4dev_random'  # same shapes, random non-zero B: tests serving plumbing independently of training

T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'train_eval'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'machine': 'gpu-4xL4', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
GEMMA_SAMPLING = ('temperature: 0.6\ntop_p: 0.95\nmax_output_tokens: 16384\n'
                  'thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n')


def save():
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    save()
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:3000]}', flush=True)


def trace_behaviour(run_dir):
    """Per-step agent text length (the step's message) and stale repeats (the same tool call with no edit since its previous
    occurrence), from a run's ATIF traces. G2b's adapter (B26/B28/B30): per-step text p90 404 vs 1,681 chars; stale
    repeats median 13% of calls, 29/70 dev runs with a stale call >= 5 times (base 0/53)."""
    lens, rep_share, max_rep = [], [], 0
    for f in sorted(Path(run_dir).glob('results/traces/trace_*.json')):
        try:
            steps = [s for s in json.loads(f.read_text()).get('steps', []) if s.get('source') == 'agent']
        except (OSError, ValueError):
            continue
        lens += [len(s.get('message') or '') for s in steps]
        calls = [(c.get('function_name'), json.dumps([c.get('function_name'), c.get('arguments')], sort_keys=True))
                 for s in steps for c in (s.get('tool_calls') or [])]
        # a stale repeat: the same call again with no edit since its previous occurrence (re-running a repro script after
        # an edit is normal; G2b's adapter re-read the same lines 26 times without editing)
        last_edit, seen, n_stale, chain = 0, {}, 0, collections.Counter()
        for i, (name, key) in enumerate(calls):
            if name in ('edit_file', 'write_file'):
                last_edit = i + 1
                continue
            if key in seen and seen[key] >= last_edit:
                n_stale += 1
                chain[key] += 1
            seen[key] = i
        if calls:
            rep_share.append(n_stale / len(calls))
            max_rep = max(max_rep, max(chain.values()) + 1 if chain else 1)
    lens.sort()
    return {'steps': len(lens), 'text_p50': lens[len(lens) // 2] if lens else None,
            'text_p90': lens[int(0.9 * (len(lens) - 1) + 0.5)] if lens else None,
            'stale_repeat_share': round(sum(rep_share) / len(rep_share), 3) if rep_share else None, 'max_stale_chain': max_rep}


def find(pattern_tail, depth=6):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def repograph_data(comp):
    """Competition-layout data dir over the repograph split: snapshots, graphs under the harness's file names
    (<repo name>_<base_commit>.json), stand-in embeddings so the scorer's Code Intelligence prompt section appears, and
    the evaluation ids (trusted by the CPU controls, excluded repos removed, optionally a seeded subset)."""
    import hashlib
    import random
    import re
    import numpy as np
    rg = Path(sorted(glob.glob(f'/kaggle/input/**/repograph-verified-swe-tasks/{EVAL_REPOGRAPH}/tasks.jsonl',
                               recursive=True))[0]).parent
    data = WORK / 'rgdata'
    for d in ('snapshots', 'graphs', 'embeddings'):
        (data / d).mkdir(parents=True, exist_ok=True)
    for d in ('wheels', 'docker', 'sandbox', 'sample_submission'):
        if not (data / d).exists():
            os.symlink(comp / d, data / d)
    rows = [json.loads(x) for x in (rg / 'tasks.jsonl').read_text().splitlines() if x.strip()]
    trusted, src_csv, rg_ids = None, [], {r['instance_id'] for r in rows}
    for f in sorted(glob.glob('/kaggle/input/**/controls/controls.csv', recursive=True)):  # union over control runs
        with open(f) as fh:
            got = {r['instance_id']: r['label_trusted'] == 'True' for r in __import__('csv').DictReader(fh)}
        if got and set(got) <= rg_ids:
            trusted = (trusted or set()) | {i for i, ok in got.items() if ok}
            src_csv.append(f)
    if trusted is None:
        raise RuntimeError('no repograph controls.csv mounted (g4dev-repograph-controls-cpu output)')
    excl = {x.lower() for x in REPOGRAPH_EXCLUDE}

    def vec(text, dim=64):  # hashed signed bag of identifier pieces: a stand-in, not the official embedding model
        v = np.zeros(dim, np.float32)
        for tok in re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])', text):
            h = int(hashlib.md5(tok.lower().encode()).hexdigest()[:8], 16)
            v[h % dim] += 1.0 if (h >> 12) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v
    graphs, embs = 0, 0
    for r in rows:
        os.symlink(rg / 'snapshots' / f"{r['instance_id']}.tgz", data / 'snapshots' / f"{r['instance_id']}.tgz")
        short, sha = r['repo'].split('/')[-1], r['base_commit']
        dst = data / 'graphs' / f'{short}_{sha}.json'
        src = next((p for p in (rg / 'graphs' / f"{r['instance_id'].rsplit('_', 1)[0]}_{sha}.json",
                                rg / 'graphs' / f'{short}_{sha}.json') if p.exists()), None)
        if src is None or dst.exists():
            continue
        os.symlink(src, dst)
        graphs += 1
        g = json.loads(src.read_text())
        keys = {}
        for n in g.get('nodes', []):
            nid = str(n.get('id') or n.get('name') or '')
            if nid and '/' not in nid:
                keys[nid] = vec(nid + ' ' + str(n.get('text') or '')[:600])
        if keys:
            np.savez_compressed(data / 'embeddings' / f'{short}_{sha}.npz', **keys)
            embs += 1
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    ids = sorted(r['instance_id'] for r in rows if r['instance_id'] in trusted and r['repo'].lower() not in excl)
    if EVAL_MAX_TASKS and len(ids) > EVAL_MAX_TASKS:
        ids = sorted(random.Random(0).sample(ids, EVAL_MAX_TASKS))
    stage('repograph_data', split=EVAL_REPOGRAPH, tasks=len(rows), trusted=len(trusted), eval_ids=len(ids),
          excluded_repos=len(excl), graphs=graphs, embeddings=embs, controls=src_csv)
    return data, ids


def run_models(serve_model, data, splits, env_args):
    """MODELS mode (B192): prepare every checkpoint, then run the blocks of tasks model by model on one server at a time."""
    global server
    import random
    import numpy as np
    from adk_submission import VllmConfig, VllmServer
    (WORK / 'slice_model.py').write_text(SLICE_SRC)
    sys.path.insert(0, str(WORK))
    from slice_model import make_plan, slice_checkpoint, slice_adapter, merge_lora_w4
    # 1. the tasks, in the order the harness runs them with --order-seed 0, cut into blocks
    ids = set(json.loads(Path(splits).read_text())[EVAL_SET])
    tasks = [json.loads(x) for x in (Path(data) / 'tasks.jsonl').read_text().splitlines() if x.strip()]
    order = sorted(tasks, key=lambda t: t['instance_id'])
    random.Random(0).shuffle(order)
    # after the EVAL_MAX_TASKS sample (the earlier sessions' tasks); matched on the repo's NAME, not owner/name: same4
    # (B197) missed jsonargparse/syrupy, listed in the eval set under their new owners (mauvilsa/, syrupy-project/)
    short = lambda r: str(r or '').lower().rstrip('/').split('/')[-1]  # noqa: E731
    excl_b = {short(x) for x in BLOCK_EXCLUDE_REPOS}
    seq = [t['instance_id'] for t in order if t['instance_id'] in ids and short(t.get('repo')) not in excl_b]
    if excl_b:
        stage('block_exclude', repos=sorted(excl_b), dropped=sum(1 for t in order if t['instance_id'] in ids
                                                                  and short(t.get('repo')) in excl_b))
    if BLOCK_IDS:   # B209: an earlier session's exact blocks
        blocks = {k: list(v) for k, v in BLOCK_IDS.items()}
        missing = sorted(i for v in blocks.values() for i in v if i not in ids)
        if missing or sorted(blocks) != [f'block{b}' for b in range(BLOCKS)]:
            raise RuntimeError(f'BLOCK_IDS: {len(missing)} ids not in {EVAL_SET} ({missing[:5]}), keys {sorted(blocks)}')
    else:
        if len(seq) < BLOCKS * BLOCK_TASKS:
            raise RuntimeError(f'{len(seq)} tasks for {BLOCKS} blocks of {BLOCK_TASKS}')
        blocks = {f'block{b}': seq[b * BLOCK_TASKS:(b + 1) * BLOCK_TASKS] for b in range(BLOCKS)}
    bfile = WORK / 'blocks.json'
    bfile.write_text(json.dumps(blocks, indent=1))
    stage('models_blocks', sizes={k: len(v) for k, v in blocks.items()}, first=[v[:3] for v in blocks.values()])
    # 1b. B209: a model served with the tolerant tool-call parser (vLLM plugin) is checked here first, in this process,
    # against the installed vLLM's own Gemma 4 parser and the model's tokenizer: a malformed call must come out repaired
    plugin = WORK / 'gemma4_tolerant.py'
    if any(m.get('tool_parser') == 'gemma4_tolerant' for m in MODELS):
        plugin.write_text(TOLERANT_SRC)
        import importlib.util as _iu
        import types as _types
        from transformers import AutoTokenizer
        _spec = _iu.spec_from_file_location('gemma4_tolerant', str(plugin))
        _mod = _iu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        cls = ToolParserManager.get_tool_parser('gemma4_tolerant')
        tp = cls(AutoTokenizer.from_pretrained(str(serve_model)))
        q = '<|"|>'
        req = _types.SimpleNamespace(tools=[_types.SimpleNamespace(function=_types.SimpleNamespace(
            name='read_file', parameters={'properties': {'filepath': {}, 'start_line': {}, 'end_line': {}}}))])
        got = {}
        for label, body in (('backtick', f'filepath:{q}src/a.py`,start_line:70,end_line:170'),
                            ('json_key', f'filepath:{q}src/a.py{q},start_line":70,end_line":170'),
                            ('clean', f'filepath:{q}src/a.py{q},start_line:70,end_line:170')):
            r = tp.extract_tool_calls('<|tool_call>call:read_file{' + body + '}<tool_call|>', req)
            got[label] = json.loads(r.tool_calls[0].function.arguments) if r.tools_called else None
        want = {'filepath': 'src/a.py', 'start_line': 70, 'end_line': 170}
        stage('tolerant_selftest', parser=f'{cls.__module__}.{cls.__name__}', got=got)
        if any(v != want for v in got.values()):
            raise RuntimeError(f'tolerant parser self-test failed: {got}')
    # 2. the checkpoints: the full model, cuts (sliced once per distinct cut), merged healing adapters
    dirs, cut_cache = {}, {}
    for m in MODELS:
        t = time.time()
        c = m.get('cut')
        if not c:
            dirs[m['name']] = serve_model
            stage('models_prepared', model=m['name'], dir=str(serve_model))
            continue
        z = np.load(Path(sorted(glob.glob(f"/kaggle/input/{c['scores']}", recursive=True))[0]))
        key = c.get('score_key', 'agent_ff')
        plan = make_plan(z, c['keep'], drop=c.get('drop', []), score_key=key, multiple=c.get('multiple', 512),
                         group=c.get('group', 1))
        # act_key (B202): whose activation sizes weight down_proj's re-quantization; prune_serve uses agent_ff for every
        # cut, so a role_ff cut set to 'agent_ff' here is exactly serve-v6's
        act_key = c.get('act_key', key)
        ck = json.dumps({k: c.get(k) for k in ('scores', 'keep', 'drop', 'group', 'score_key', 'multiple', 'act_key')},
                        sort_keys=True)
        rec = {}
        if ck not in cut_cache:
            act = z[act_key] / np.maximum(z['down_norm'], 1e-12) if 'down_norm' in z.files else None
            d = Path(f"/tmp/model_{m['name']}")
            cut_cache[ck] = (d, slice_checkpoint(str(serve_model), str(d), plan, log=lambda *a: None, act=act, dev='cpu'))
        d, rec['cut'] = cut_cache[ck]
        if m.get('merge'):
            if c.get('mask'):  # the healing adapter was trained on exactly this cut
                hm = np.load(sorted(glob.glob(f"/kaggle/input/{c['mask']}", recursive=True))[0])['mask']
                if not all(np.array_equal(np.nonzero(hm[i])[0], np.asarray(plan['ff_idx'][i])) for i in plan['kept_layers']):
                    raise RuntimeError(f"{m['name']}: the cut differs from the mask the adapter was trained with")
            src = Path(sorted(glob.glob(f"/kaggle/input/{m['merge']}", recursive=True))[0])
            ad = WORK / f"adapter_{m['name']}"
            rec['adapter_slice'] = slice_adapter(str(src), str(ad), plan)
            md = Path(f"/tmp/model_{m['name']}_merged")
            rec['merge'] = merge_lora_w4(str(d), str(md), str(ad), mode=m.get('merge_mode', 'stochastic'),
                                         seed=m.get('merge_seed', 0), log=lambda *a: None)
            d = md
        dirs[m['name']] = d
        stage('models_prepared', model=m['name'], dir=str(d), minutes=round((time.time() - t) / 60, 1), **rec)
    # 3. the deployment bundle (its prompt, sampling and caps; the clock x EVAL_TIME_SCALE)
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    bdir = WORK / 'bundles' / EVAL_BUNDLE
    for rel, text in BUNDLES[EVAL_BUNDLE].items():
        (bdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (bdir / rel).write_text(text)
    bundle_args = ['--bundle', str(bdir), '--time-scale', str(EVAL_TIME_SCALE)]
    # 4. segments: block b in the listed order (b even) or reversed (b odd); one server per model change
    names = [m['name'] for m in MODELS]
    mcfg = {m['name']: m for m in MODELS}
    segs = [(b, n) for b in range(BLOCKS) for n in (names if b % 2 == 0 else names[::-1])]
    cur, mid, results = None, None, {}
    for k, (b, name) in enumerate(segs):
        starts, prev = 0, cur
        for _, n in segs[k:]:
            starts += n != prev
            prev = n
        left = DEADLINE_S - (time.time() - T0) - 10 * 60
        budget = min(SEG_CAP_MIN * 60, (left - starts * 7 * 60) / (len(segs) - k))
        if budget < 15 * 60:
            stage('models_out_of_time', segment=k, block=b, model=name, left_min=round(left / 60, 1))
            break
        if name != cur:
            if server is not None:
                server.stop()
                server = None
                time.sleep(20)                                        # let the GPUs empty before the next model
            t = time.time()
            parser = mcfg[name].get('tool_parser', 'gemma4')          # B209: 'gemma4_tolerant' = the vLLM plugin
            extra = ['--tool-parser-plugin', str(plugin)] if parser == 'gemma4_tolerant' else []
            if extra:
                os.environ['G4DEV_TOLERANT_LOG'] = str(OUT / f'tolerant_rewrites_{name}.jsonl')
            else:
                os.environ.pop('G4DEV_TOLERANT_LOG', None)
            server = VllmServer(VllmConfig(
                model=str(dirs[name]), port=8000, host='127.0.0.1', dtype='auto', gpu_memory_utilization=0.92,
                tool_call_parser=parser, reasoning_parser='gemma4', max_model_len=32768, enable_auto_tool_choice=True,
                enable_lora=False, tensor_parallel_size=4, startup_timeout=1500, extra_args=extra))
            server.start()
            cur = name
            mid = str(getattr(server, 'effective_model_path', None) or server.config.model)
            stage('models_server_up', model=name, startup_s=round(time.time() - t, 1), model_id=mid)
        out = OUT / f'{EVAL_SET}_{name}_b{b}'
        stage('models_segment_start', segment=k, block=b, model=name, budget_min=round(budget / 60, 1))
        t = time.time()
        try:
            subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data), '--base-url',
                            server.base_url, '--model-id', mid, '--tag', name, '--out', str(out), '--tasks', 'file',
                            '--task-ids-file', str(bfile), '--task-ids-key', f'block{b}', '--order-seed', '0',
                            '--concurrency', str(EVAL_CONC), '--max-time-minutes', '20', '--budget-s', str(int(budget)),
                            *bundle_args, *env_args], timeout=budget + 900)
        except subprocess.TimeoutExpired:
            pass
        f = out / 'summary_smoke.json'
        sm = json.loads(f.read_text()) if f.exists() else {}
        results[f'{name}_b{b}'] = {x: sm.get(x) for x in ('status', 'healthy', 'n_tasks', 'n_finished', 'n_resolved',
                                                          'n_errors', 'n_llm_calls', 'tool_calls_per_llm_call')}
        stage('models_segment_done', segment=k, block=b, model=name, minutes=round((time.time() - t) / 60, 1),
              **results[f'{name}_b{b}'])
        rw = OUT / f'tolerant_rewrites_{name}.jsonl'
        if rw.exists():   # B209: calls the tolerant parser rewrote so far (and the server's 'registered' marker)
            ev = [json.loads(x) for x in rw.read_text().splitlines() if x.strip()]
            stage('tolerant_rewrites', model=name, registered=sum(e.get('event') == 'registered' for e in ev),
                  rewritten=sum('tool' in e for e in ev))
        if not sm.get('n_finished'):
            raise RuntimeError(f'segment {k} ({name}, block {b}) finished no task: stopping to save quota')
    if server is not None:
        server.stop()
        server = None
    stage('models_done', results=results)


for k, v in {
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'VLLM_ENGINE_READY_TIMEOUT_S': '1500',
    'VLLM_NO_USAGE_STATS': '1', 'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ[k] = v

server = None
try:
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip().splitlines()
    stage('env', python=sys.version.split()[0], gpus=gpus, cpus=os.cpu_count())
    if len(gpus) < 4:
        raise SystemExit(f'expected 4 GPUs, got {len(gpus)}: stopping to save quota')

    # ---- wheelhouse (official procedure) + training libs if the image lacks them ----
    wh = find('swegemma-*.whl').parent
    for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
        try:
            os.unlink(pth)
        except OSError:
            pass
    tmp_whl = Path('/tmp/wheelhouse')
    tmp_whl.mkdir(parents=True, exist_ok=True)
    for w in wh.glob('*.whl'):
        if 'cutlass' in w.name.lower():
            continue
        name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
        if not (tmp_whl / name).exists():
            os.symlink(w, tmp_whl / name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                    *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
    installed = {}
    for mod, pat in (('peft', 'peft-*.whl'), ('accelerate', 'accelerate-*.whl'), ('bitsandbytes', 'bitsandbytes-*.whl')):
        if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode != 0:
            w = find(pat)
            if w:  # our wheel dataset
                rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(w)]).returncode
                installed[mod] = f'wheel rc={rc}'
            else:  # needs internet (off for submission notebooks, allowed for our own runs)
                rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', mod]).returncode
                installed[mod] = f'pypi rc={rc}'
            if subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode != 0:
                raise RuntimeError(f'{mod} is missing and could not be installed ({installed[mod]}): stopping before the model loads')
    stage('train_libs', installed=installed)
    importlib.invalidate_caches()
    versions = subprocess.run([sys.executable, '-c', 'import torch, transformers, peft, bitsandbytes, accelerate;'
                               'print(torch.__version__, transformers.__version__, peft.__version__,'
                               'bitsandbytes.__version__, accelerate.__version__)'],
                              capture_output=True, text=True)
    stage('libs', versions=versions.stdout.strip() or versions.stderr[-800:])

    import vllm
    cu = Path(vllm.__file__).parent / 'entrypoints' / 'chat_utils.py'
    src = cu.read_text()
    old = 'reasoning = message.get("reasoning")'
    # Does this wheel replay past thoughts? ADK sends them as `reasoning_content`; older wheels read only `reasoning`,
    # so past thoughts were silently dropped from the prompt. Training data must be rendered the way the scorer does.
    reads_rc = 'reasoning_content' in src
    needs_patch = old in src and 'reasoning_content' not in src.split(old, 1)[1][:80]
    if needs_patch and PATCH_REASONING:
        cu.write_text(src.replace(old, 'reasoning = message.get("reasoning") or message.get("reasoning_content")', 1))
    stage('vllm', version=vllm.__version__, chat_utils_mentions_reasoning_content=reads_rc,
          wheel_drops_past_thoughts=needs_patch, patched=bool(needs_patch and PATCH_REASONING),
          reasoning_lines=[l.strip()[:160] for l in src.splitlines() if 'reasoning' in l][:14])

    DATA = find('tasks.jsonl').parent
    sft = find(SFT_NAME) or Path(sorted(glob.glob(f'/kaggle/input/**/{SFT_NAME}', recursive=True))[0])
    splits = find('splits.json')
    qat_bf16 = (find('gemma-4-31b-it-qat-q4_0-unquantized/*/config.json') or
                find('gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')).parent
    serve_model = (find('gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*/config.json')
                   or find('gemma-4-31b-it-qat-w4a16-ct/*/config.json')).parent
    kit, extra = find('py313.tar.gz*'), find('dirty_equals-*.whl')
    wh313 = sorted(glob.glob('/kaggle/input/**/g4dev-py313-wheels/**/pytest_timeout-*.whl', recursive=True))
    rw = sorted(glob.glob('/kaggle/input/**/repograph-verified-swe-tasks-wheels', recursive=True))
    sub = {'@kit': str(kit.parent) if kit else None, '@extra': str(extra.parent) if extra else None,
           '@wh313': str(Path(wh313[0]).parent) if wh313 else None, '@rw': rw[0] if rw else None}
    env_args = [sub.get(x, x) if x.startswith('@') else x for x in ENV_ARGS]
    if None in env_args:
        raise RuntimeError(f'unresolved env args {ENV_ARGS} -> {env_args}')
    EVAL_DATA = DATA
    if EVAL_REPOGRAPH:
        EVAL_DATA, rg_ids = repograph_data(DATA)
        splits = WORK / 'repograph_split.json'
        splits.write_text(json.dumps({EVAL_SET: rg_ids}))
    stage('inputs', sft=str(sft), qat_bf16=str(qat_bf16), serve=str(serve_model), env_args=env_args,
          eval_data=str(EVAL_DATA))
    if MODELS:
        run_models(serve_model, EVAL_DATA, splits, env_args)
        raise SystemExit('models: done')

    # ---- 1. train (own process), or reuse a trained adapter ----
    if ADAPTER_FROM:
        src_ad = Path(sorted(glob.glob(f'/kaggle/input/{ADAPTER_FROM}', recursive=True))[0])
        shutil.copytree(src_ad, OUT / 'train' / 'adapter', dirs_exist_ok=True)
        stage('adapter_reused', src=str(src_ad), files=sorted(x.name for x in src_ad.iterdir()))
    (WORK / 'train_lora.py').write_text(TRAIN_PY)
    t = time.time()
    val_file = (find(VAL_NAME) or Path(sorted(glob.glob(f'/kaggle/input/**/{VAL_NAME}', recursive=True))[0])) if VAL_NAME else None
    split = REPLICA_SPLIT if REPLICAS > 1 else LAYER_SPLIT
    cmd = [sys.executable, str(WORK / 'train_lora.py'), '--model', str(qat_bf16), '--data', str(sft),
           '--out', str(OUT / 'train'), '--time-cap-min', str(TRAIN_CAP_MIN), '--epochs', str(EPOCHS), '--lr', str(LR),
           '--max-examples', str(MAX_EXAMPLES), '--probe-lens', PROBE_LENS, '--grad-accum', str(GRAD_ACCUM),
           '--sched', SCHED, '--save-every-min', str(SAVE_EVERY_MIN), *(['--probe-gate'] if PROBE_GATE else []),
           *(['--max-mem', MAX_MEM] if MAX_MEM else []), '--device-map', DEVICE_MAP, '--loss-chunk', str(LOSS_CHUNK),
           *(['--layer-split', split] if split else []), *(['--probe-real'] if PROBE_REAL else []),
           *(['--probe-only'] if PROBE_ONLY else []), '--train-on', TRAIN_ON,
           *(['--skip-kinds', SKIP_KINDS] if SKIP_KINDS else []),
           *(['--mask-mistakes'] if MASK_MISTAKES else []),
           *(['--val-data', str(val_file), '--val-n', str(VAL_N)] if val_file else [])]
    if ADAPTER_FROM:
        rc = 0
    elif REPLICAS > 1:  # one process per replica, each sees its own pair of GPUs; gloo all-reduce of LoRA gradients
        per = 4 // REPLICAS
        procs = [subprocess.Popen(cmd + ['--replicas', str(REPLICAS)], env=dict(
            os.environ, RANK=str(r), WORLD_SIZE=str(REPLICAS), MASTER_ADDR='127.0.0.1', MASTER_PORT='29533',
            CUDA_VISIBLE_DEVICES=','.join(str(g) for g in range(r * per, (r + 1) * per)))) for r in range(REPLICAS)]
        t_end = time.time() + (TRAIN_CAP_MIN + TRAIN_EXTRA_MIN) * 60
        rcs = []
        for pr in procs:
            try:
                rcs.append(pr.wait(timeout=max(1, t_end - time.time())))
            except subprocess.TimeoutExpired:
                for q in procs:
                    q.kill()
                rcs.append('timeout')
        rc = next((x for x in rcs if x != 0), 0)
        stage('replicas_done', returncodes=rcs)
    else:
        try:
            r = subprocess.run(cmd, timeout=(TRAIN_CAP_MIN + TRAIN_EXTRA_MIN) * 60)
            rc = r.returncode
        except subprocess.TimeoutExpired:  # the last checkpoint (SAVE_EVERY_MIN) is still served and checked
            rc = 'timeout'
    log_lines = (OUT / 'train' / 'train_log.jsonl').read_text().splitlines() if (OUT / 'train' / 'train_log.jsonl').exists() else []
    ex_lines = [json.loads(x) for x in log_lines if '"event": "example"' in x]
    stage('train_done', returncode=rc, minutes=round((time.time() - t) / 60, 1),
          first=log_lines[:3], last=log_lines[-4:], examples=len(ex_lines),
          tok_per_s=ex_lines[-1].get('tok_per_s') if ex_lines else None,
          loss_first=[x['loss'] for x in ex_lines[:3]], loss_last=[x['loss'] for x in ex_lines[-3:]],
          oom_skips=sum('"event": "oom_skip"' in x for x in log_lines),
          probe=[json.loads(x) for x in log_lines if any(f'"event": "{e}"' in x for e in ('probe', 'sdpa_backends', 'probe_gate'))],
          nan_or_abort=[json.loads(x) for x in log_lines if '"event": "nan_loss"' in x or '"event": "abort"' in x][:5],
          steps=next((json.loads(x).get('steps') for x in log_lines if '"event": "saved"' in x), None),
          checkpoints=sum('"event": "checkpoint"' in x for x in log_lines),
          val=[{k: v for k, v in json.loads(x).items() if k != 'per_example'} for x in log_lines if '"event": "val"' in x])
    if PROBE_ONLY:
        raise SystemExit('probe only: done (see train_done.probe)')
    adapter_src = OUT / 'train' / 'adapter'
    if not (adapter_src / 'adapter_model.safetensors').exists() or (rc != 0 and not SAVE_EVERY_MIN):
        raise RuntimeError('training failed; see train_log.jsonl')
    if TRAIN_ONLY:
        shutil.copytree(adapter_src, WORK / 'adapter_out', dirs_exist_ok=True)
        stage('adapter_saved', files=sorted(x.name for x in (WORK / 'adapter_out').iterdir()),
              mb=round(sum(x.stat().st_size for x in (WORK / 'adapter_out').iterdir()) / 1e6, 1))
        raise SystemExit('train only: done')

    # ---- 1b. (CUT) the cut model and cut adapters ----
    cut_plan = None
    if CUT:
        import numpy as np
        (WORK / 'slice_model.py').write_text(SLICE_SRC)
        sys.path.insert(0, str(WORK))
        from slice_model import make_plan, slice_checkpoint, slice_adapter
        t = time.time()
        sc = Path(sorted(glob.glob(f"/kaggle/input/{CUT['scores']}", recursive=True))[0])
        z = np.load(sc)
        key = CUT.get('score_key', 'agent_ff')
        cut_plan = make_plan(z, CUT['keep'], drop=CUT.get('drop', []), score_key=key, multiple=CUT.get('multiple', 512),
                             group=CUT.get('group', 1))
        same = None
        hm_hits = sorted(glob.glob(f"/kaggle/input/{CUT['mask']}", recursive=True)) if CUT.get('mask') else []
        if hm_hits:
            hm = np.load(hm_hits[0])['mask']
            same = all(np.array_equal(np.nonzero(hm[i])[0], np.asarray(cut_plan['ff_idx'][i])) for i in cut_plan['kept_layers'])
            if not same:
                raise RuntimeError('the cut differs from the mask the adapter was trained with')
        act = z[key] / np.maximum(z['down_norm'], 1e-12) if 'down_norm' in z.files else None
        cut_dir = Path('/tmp/cut_model')
        summ = slice_checkpoint(str(serve_model), str(cut_dir), cut_plan, log=lambda *a: None, act=act, dev='cpu')
        serve_model = cut_dir
        ad_summ = slice_adapter(str(adapter_src), str(OUT / 'train' / 'adapter_cut'), cut_plan)
        adapter_src = OUT / 'train' / 'adapter_cut'
        stage('cut', scores=str(sc), keep_ff=cut_plan['keep_ff'], dropped=cut_plan['drop'], summary=summ,
              adapter=ad_summ, matches_training_mask=same, minutes=round((time.time() - t) / 60, 1))

    # ---- 2. serve base + adapter (competition model, multi-LoRA) ----
    policy = WORK / 'serve_policy'
    shutil.rmtree(policy, ignore_errors=True)
    (policy / 'adapters').mkdir(parents=True)
    shutil.copytree(adapter_src, policy / 'adapters' / ADAPTER, ignore=shutil.ignore_patterns('README.md'))
    shutil.copytree(adapter_src, policy / 'adapters' / RANDOM_ADAPTER, ignore=shutil.ignore_patterns('README.md'))
    if ADAPTER_B_FROM:
        src_b = Path(sorted(glob.glob(f'/kaggle/input/{ADAPTER_B_FROM}', recursive=True))[0])
        if cut_plan is not None:
            stage('adapter_b_cut', **slice_adapter(str(src_b), str(OUT / 'train' / 'adapter_b_cut'), cut_plan))
            src_b = OUT / 'train' / 'adapter_b_cut'
        shutil.copytree(src_b, policy / 'adapters' / ADAPTER_B, ignore=shutil.ignore_patterns('README.md'))
        stage('adapter_b', src=str(src_b), files=sorted(x.name for x in src_b.iterdir()))
        if ADAPTER_B_ALPHA:
            cfg_b = policy / 'adapters' / ADAPTER_B / 'adapter_config.json'
            cb = json.loads(cfg_b.read_text())
            stage('adapter_b_alpha', old=cb.get('lora_alpha'), new=ADAPTER_B_ALPHA, r=cb.get('r'))
            cb['lora_alpha'] = ADAPTER_B_ALPHA
            cfg_b.write_text(json.dumps(cb, indent=2))
    import torch
    from safetensors.torch import load_file, save_file
    rnd = policy / 'adapters' / RANDOM_ADAPTER / 'adapter_model.safetensors'
    g = torch.Generator().manual_seed(0)
    sd = {k: (torch.randn(v.shape, generator=g, dtype=torch.float32) * 0.02).to(v.dtype) if 'lora_B' in k else v
          for k, v in load_file(str(rnd)).items()}
    save_file(sd, str(rnd), metadata={'format': 'pt'})
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    manifest = discover_adapters(str(policy), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    t = time.time()
    server = VllmServer(VllmConfig(
        model=str(serve_model), port=8000, host='127.0.0.1', dtype='auto', gpu_memory_utilization=0.92,
        tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768, enable_auto_tool_choice=True,
        enable_lora=True, max_loras=3 if ADAPTER_B_FROM else 2, max_lora_rank=64, tensor_parallel_size=4, startup_timeout=1500,
    ), adapter_manifest=manifest)
    server.start()
    base_id = str(getattr(server, 'effective_model_path', None) or server.config.model)
    served = json.loads(urllib.request.urlopen(server.base_url + '/models', timeout=30).read())
    served_ids = [m['id'] for m in served.get('data', [])]
    stage('server_up', startup_s=round(time.time() - t, 1), served=served_ids)
    if ADAPTER_B_FROM and ADAPTER_B not in served_ids:
        raise RuntimeError(f'second adapter {ADAPTER_B} not served; got {served_ids}')
    if ADAPTER not in served_ids or RANDOM_ADAPTER not in served_ids:
        raise RuntimeError(f'adapters {ADAPTER}/{RANDOM_ADAPTER} not both served; got {served_ids}')

    # ---- 3. the adapter must actually change the model ----
    def top_logprobs(model_id):
        body = json.dumps({'model': model_id, 'prompt': 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
                           'max_tokens': 8, 'temperature': 0, 'logprobs': 5}).encode()
        req = urllib.request.Request(server.base_url + '/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        out = json.loads(urllib.request.urlopen(req, timeout=120).read())['choices'][0]
        return out['text'], out['logprobs']['top_logprobs']
    def diff(a_lp, b_lp):
        return round(max(abs(a_lp[i].get(k, -99) - v) for i in range(min(len(a_lp), len(b_lp))) for k, v in b_lp[i].items()), 4)
    base_txt, base_lp = top_logprobs(base_id)
    rnd_txt, rnd_lp = top_logprobs(RANDOM_ADAPTER)
    lora_txt, lora_lp = top_logprobs(ADAPTER)
    trained_effect = diff(base_lp, lora_lp)
    stage('lora_check', base=base_txt, random_adapter=rnd_txt, trained_adapter=lora_txt,
          random_diff=diff(base_lp, rnd_lp), trained_diff=trained_effect)
    if diff(base_lp, rnd_lp) < 1e-3:
        raise RuntimeError('a random non-zero adapter has no effect on logprobs: adapters are not applied when served')
    # a trained adapter with no effect is reported, not fatal: the canaries still say whether serving works end to end

    # ---- 4. base vs adapter, same session, same environment: canaries, then EVAL_SET (gated) ----
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    bundle_args, caps = [], ['--sampling', GEMMA_SAMPLING, '--max-tool-calls', '100', '--max-turns', '150',
                             '--timeout-seconds', '120']
    if EVAL_BUNDLE:  # the bundle's own sampling.yaml and eval_config.yaml caps (28 calls, 8 min x EVAL_TIME_SCALE)
        bdir = WORK / 'bundles' / EVAL_BUNDLE
        for rel, text in BUNDLES[EVAL_BUNDLE].items():
            (bdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (bdir / rel).write_text(text)
        bundle_args, caps = ['--bundle', str(bdir), '--time-scale', str(EVAL_TIME_SCALE)], []
    bundle_args_b = bundle_args
    if EVAL_BUNDLE_B:  # prompt A/B: arm 2 = base model + bundle B (same caps source, same time scale)
        bdir_b = WORK / 'bundles' / EVAL_BUNDLE_B
        for rel, text in BUNDLES[EVAL_BUNDLE_B].items():
            (bdir_b / rel).parent.mkdir(parents=True, exist_ok=True)
            (bdir_b / rel).write_text(text)
        bundle_args_b = ['--bundle', str(bdir_b), '--time-scale', str(EVAL_TIME_SCALE)]
        stage('prompt_ab', a=EVAL_BUNDLE, b=EVAL_BUNDLE_B)

    def run_pair(name, task_args, budget_s, conc, minutes, seed=None, data=None):
        procs = {}
        mid_a = ADAPTER if ((EVAL_BUNDLE_B and EVAL_AB_ADAPTER) or ADAPTER_B_FROM) else base_id
        mid_b = ADAPTER_B if ADAPTER_B_FROM else (ADAPTER if EVAL_AB_ADAPTER else base_id) if EVAL_BUNDLE_B else ADAPTER
        for tag, mid in (('base', mid_a), ('adapter', mid_b)):
            procs[tag] = subprocess.Popen([
                sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data or DATA), '--base-url', server.base_url,
                '--model-id', mid, '--tag', tag, '--out', str(OUT / f'{name}_{tag}'), *task_args,
                '--concurrency', str(conc), '--max-time-minutes', minutes, '--budget-s', str(int(budget_s)),
                *(['--order-seed', str(seed)] if seed is not None else []),
                *(bundle_args_b if tag == 'adapter' else bundle_args), *caps, *env_args])
        stage(f'{name}_start', budget_min=round(budget_s / 60, 1), concurrency=conc)
        res = {}
        for tag, proc in procs.items():
            try:
                proc.wait(timeout=budget_s + 900)
            except subprocess.TimeoutExpired:
                proc.kill()
            f = OUT / f'{name}_{tag}' / 'summary_smoke.json'
            sm = json.loads(f.read_text()) if f.exists() else {}
            res[tag] = {k: sm.get(k) for k in ('status', 'healthy', 'n_tasks', 'n_finished', 'n_resolved', 'n_errors',
                                                'n_llm_calls', 'tool_calls_per_llm_call', 'resolved_ids')}
        stage(f'{name}_done', **res)
        return res

    left = DEADLINE_S - (time.time() - T0) - 10 * 60
    canary = run_pair('canary', ['--tasks', 'canary'], min(25 * 60, left), 2, '8')
    beh = {tag: trace_behaviour(OUT / f'canary_{tag}') for tag in ('base', 'adapter')}
    stage('canary_behaviour', **beh)
    if EVAL_SET != 'canary':
        val = {json.loads(x)['when']: json.loads(x)['loss_tok'] for x in log_lines if '"event": "val"' in x}
        why = [] if splits else ['splits.json not mounted']
        if EVAL_GATE:
            if 'start' in val and 'end' in val and not val['end'] < val['start']:
                why.append(f'validation loss did not fall ({val})')
            if trained_effect < 1e-3:
                why.append('the trained adapter does not move logprobs')
            for tag in ('base', 'adapter'):
                if not (canary.get(tag) or {}).get('healthy'):
                    why.append(f'{tag} canary unhealthy')
            b, a_ = beh['base'], beh['adapter']
            if b['text_p90'] and a_['text_p90'] is not None and a_['text_p90'] < GATE_REASONING_MIN * b['text_p90']:
                why.append(f"adapter reasoning collapsed: per-step text p90 {a_['text_p90']} vs base {b['text_p90']}")
            if a_['max_stale_chain'] >= GATE_MAX_REPEAT:
                why.append(f"adapter loops: one call repeated {a_['max_stale_chain']} times without an edit in a canary")
        stage('eval_gate', go=not why, why=why, val=val, behaviour=beh)
        for k in range(EVAL_PASSES if not why else 0):
            left = DEADLINE_S - (time.time() - T0) - 10 * 60
            if left < 20 * 60:
                stage('eval_out_of_time', pass_k=k)
                break
            run_pair(f'{EVAL_SET}_p{k}', ['--tasks', 'file', '--task-ids-file', str(splits), '--task-ids-key', EVAL_SET],
                     left / (EVAL_PASSES - k), EVAL_CONC, '20', seed=k, data=EVAL_DATA)
except SystemExit as e:
    stage('stopped', reason=str(e))
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-4000:], tb=traceback.format_exc()[-3000:])
finally:
    if server is not None:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    save()
    print(json.dumps(REPORT, indent=2, default=str)[-4000:], flush=True)
