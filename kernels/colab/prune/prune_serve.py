# %% [markdown]
# # Pruning Gemma 4 31B for a coding agent, part 2: is the cut model actually faster?
#
# **Private notebook. Do not share or publish.**
#
# ## In plain words
#
# Part 1 measured which pieces of the model our agent uses. This notebook takes those measurements, **cuts the
# competition's own 4-bit copy of the model** down to the pieces that matter most, and runs each cut model in vLLM
# (the serving engine the competition uses) on the same 4 x L4 machine. For every model it measures:
#
# * **speed of reading** the prompt (prefill, tokens per second) and **speed of writing** (decode, tokens per second),
#   for one request and for 8 requests at once. Our agent spends most of its time writing, one token at a time, and
#   each written token needs a full pass over the model's weights, so fewer weights should mean faster writing;
# * **quality**: how surprised the model is by what our agent actually wrote on held-out tasks (the same number as in
#   part 1: lower is better);
# * a short **sanity check**: it completes a small Python function, so we can see it still writes code.
#
# The cuts follow part 1's choice exactly (the cutter is tested against it). A "b" cut keeps whole blocks of 32
# channels, which the 4-bit format can be cut by without touching a single number; a "c" cut keeps the best single
# channels and re-quantizes the one matrix whose blocks it splits (down_proj). v3 (B175) showed the c cuts lose half
# as much quality at the same size and speed.
#
# A cut can also come with a **healing adapter**: a small LoRA trained on TPU so that the cut model imitates the full
# model (the teacher) or simply learns the agent's text again (part 3). The adapter is trained on the full-size model
# with the removed channels switched off, so it is cut to the same shape here and served with vLLM's LoRA path; speed
# then includes LoRA's cost.
#
# **Later runs add two measurements.** (1) *The next tool call:* at real moments of held-out agent runs, each model
# writes its next step, and we check whether its tool call is well formed and whether it is the call the full model
# (or the agent itself) made there. A cut model that breaks its calls fails here in minutes, long before a task run.
# (2) *Loss per kind of token* (serve-v6): the same "surprise" number, split by what the token is: text the agent
# reads, its thoughts, argument values in its calls, call syntax, and the quote that closes a string argument. serve-v6
# compares the 55% cut chosen by the plain score with the 55% cut chosen by the role-aware score of part 1b, which
# gives each kind of token an equal vote. serve-v7 instead leaves the model whole and tries *speculative decoding*: the
# engine guesses the next few tokens by looking them up in the prompt (agents often repeat text they have read) and
# the model checks all the guesses in one pass, so the output should stay the same while writing gets faster.

# %%
# ---- Settings ----
RUN_NAME = 'serve-v1'  # v4: 'serve-v2' (set by the builder)
# each cut: share of feed-forward channels kept in every layer (in blocks of 32), and how many whole layers to
# remove (the ones that change their input least on agent text; never the first or the last)
# Part 1 (B168): narrower feed-forward costs little (keep 85% -> +0.025 nats, 70% -> +0.08, 55% -> +0.18); removing
# 6 layers costs much more (+0.97) for a similar share of weights. d6 stays in to measure how much speed a depth cut
# buys, since that is its usual selling point.
# b = best whole blocks of 32 channels (exact in the 4-bit format); c = best single channels (gate/up exact, down_proj
# re-quantized in new blocks of 32): part 1 says single channels keep much more of what matters, re-quantizing costs
# a little; this measures which wins.
CUTS = [{'name': 'b70', 'keep': 0.70, 'drop': 0, 'group': 32},
        {'name': 'c70', 'keep': 0.70, 'drop': 0, 'group': 1},
        {'name': 'c85', 'keep': 0.85, 'drop': 0, 'group': 1},
        {'name': 'b85', 'keep': 0.85, 'drop': 0, 'group': 32},
        {'name': 'c55', 'keep': 0.55, 'drop': 0, 'group': 1},
        {'name': 'd6', 'keep': 1.0, 'drop': 6, 'group': 32}]
# optional per cut: 'order': 'measured' removes the layers whose single removal hurt the agent least (part 1's layer
# sweep) instead of the lowest block influence; 'adapter': a path pattern under /kaggle/input of a healing adapter
# (PEFT, full-size) to cut and serve with this cut
# optional per cut (serve-v5): 'score': the part-1 score that chooses its channels ('agent_own_ff' = only the tokens
# the agent wrote; 'mix_ff' = per-layer normalized agent_ff + agent_own_ff: what it reads and what it writes); 'merge':
# a path pattern of a healing adapter merged into the cut's 4-bit weights (served without LoRA, B190), 'merge_mode'
SCORE_KEY = 'agent_ff'      # which part-1 scores choose the channels
TURNS_PER_ROW = 1           # next-action states per held-out example (serve-v5: several, spread over the run)
SAMPLE_TURNS_FULL = 0.0     # > 0: the full model's turns once more at this temperature (its own agreement ceiling)
FULL_REPLICA = {}           # non-empty (serve-v5): the full model again as 'full_b', in an engine with these settings
                            # (serve_bench.py PRUNE_MAX_BATCHED / PRUNE_GPU_UTIL): other numerics only, the floor of the
                            # greedy next-action agreement that every cut is compared with
HELDOUT_ROWS = 40           # held-out agent examples for the quality number (same split and order as part 1)
QUALITY_ROWS = 0            # > 0 (serve-v6): the quality number on only the first this many (the states use them all)
QUALITY_BY_ROLE = False     # (serve-v6, B202) the quality numbers also per role of the token (token_roles.py)
BENCH_ENV = {}              # (serve-v7) settings for every model's serve_bench.py process (PRUNE_SKIP_QUALITY, ...)
MAX_TOKENS = 8192
SPEED_SHORT, SPEED_LONG = 2000, 16000   # prompt lengths (tokens) for the speed tests
MULTIPLE = 512              # channels kept per layer: a multiple of 512 (4 GPUs x 128, what the 4-bit kernels need)
SEED = 0
BENCH_TIMEOUT_MIN = 20      # one model (load + measurements) took 11 min in serve-v1 v2; a hung one is killed after this
MAX_FAILS_IN_A_ROW = 2      # after this many cuts in a row fail, the rest are skipped (a shared cause; saves GPU time)

# %%
import os, sys, json, time, glob, hashlib, shutil, subprocess, random, signal
T0 = time.time()
TEST = os.environ.get('PRUNE_TEST') == '1'   # internal CPU test: tiny model, a stand-in for vLLM, no installs
if TEST:
    HELDOUT_ROWS, MAX_TOKENS, SPEED_SHORT, SPEED_LONG, MULTIPLE = 3, 6144, 100, 300, 32
    BENCH_TIMEOUT_MIN = float(os.environ.get('PRUNE_BENCH_TIMEOUT_MIN', BENCH_TIMEOUT_MIN))


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return hits[0] if hits else None


OUT = os.environ.get('PRUNE_OUT') or f'/kaggle/working/prune/{RUN_NAME}'
os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get('PRUNE_TMP', '/tmp')
LIB = f'{TMP}/prune_lib'
os.makedirs(LIB, exist_ok=True)
open(f'{LIB}/slice_model.py', 'w').write(SLICE_SRC)      # noqa: F821  (inserted by the builder)
open(f'{LIB}/serve_bench.py', 'w').write(BENCH_SRC)      # noqa: F821
open(f'{LIB}/token_roles.py', 'w').write(ROLES_SRC)      # noqa: F821
sys.path.insert(0, LIB)
if QUALITY_ROWS:
    os.environ['PRUNE_QUALITY_ROWS'] = str(QUALITY_ROWS)    # read by serve_bench.py in every model's process
os.environ.update({k: str(v) for k, v in BENCH_ENV.items()})
for k, v in {'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_NO_USAGE_STATS': '1',
             'VLLM_ENGINE_READY_TIMEOUT_S': '1500', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
             # count CUDA graph memory in vLLM's budget, as our eval kernels do (serve-v2: without it a LoRA engine's
             # doubled graphs, 4.5 GB per GPU, used up the headroom and the sampler warm-up ran out of memory)
             'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1'}.items():
    os.environ[k] = v
if not TEST:
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True,
                         text=True).stdout, flush=True)
    # the competition's wheelhouse (vLLM 0.19.1 and friends), installed the official way
    wh = os.path.dirname(find('swegemma-*.whl'))
    for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
        try:
            os.unlink(pth)
        except OSError:
            pass
    tmp_whl = '/tmp/wheelhouse'
    os.makedirs(tmp_whl, exist_ok=True)
    for w in glob.glob(f'{wh}/*.whl'):
        name = os.path.basename(w)
        if 'cutlass' in name.lower():
            continue
        name = name.replace('cu128', '+cu128') if ('cu128' in name and '+' not in name) else name
        if not os.path.exists(f'{tmp_whl}/{name}'):
            os.symlink(w, f'{tmp_whl}/{name}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                    *sorted(glob.glob(f'{tmp_whl}/*.whl'))], check=True)

import numpy as np
REPORT = {'run': RUN_NAME, 'cuts': CUTS, 'score_key': SCORE_KEY, 'timings_min': {}, 'models': {}}


def save_report():
    json.dump(REPORT, open(f'{OUT}/report.json', 'w'), indent=1)


if TEST:
    model_dir, scores_path, sft_path = os.environ['PRUNE_MODEL'], os.environ['PRUNE_SCORES'], os.environ['PRUNE_ROWS']
else:
    model_dir = os.path.dirname(find('gemma-4-31b-it-qat-w4a16-ct/*/config.json')
                                or find('**/gemma-4-31b-it-qat-w4a16-ct/**/config.json'))
    scores_path = find('**/scores.npz')
    sft_path = find('g4dev-g11-data/sft.jsonl') or find('**/sft.jsonl')
part1 = json.load(open(os.path.join(os.path.dirname(scores_path), 'report.json')))
heal_mask_path = os.environ.get('PRUNE_HEAL_MASK') if TEST else find('**/tpu/heal_mask.npz')
REPORT['inputs'] = {'model': model_dir, 'scores': scores_path, 'agent_rows': sft_path, 'part1_run': part1.get('run'),
                    'heal_mask': heal_mask_path}
REPORT['disk_free_gb'] = {'/tmp': round(shutil.disk_usage(TMP).free / 2**30, 1),
                          'out': round(shutil.disk_usage(OUT).free / 2**30, 1)}
log('inputs', REPORT['inputs'], 'free disk', REPORT['disk_free_gb'])
save_report()
# a cut model takes up to ~22 GB on disk while it is measured: stop now (not after the full model's 20 minutes) if
# there is no room
assert TEST or REPORT['disk_free_gb']['/tmp'] > 1.1 * os.path.getsize(glob.glob(f'{model_dir}/*.safetensors')[0]) / 2**30, \
    f"not enough free disk in {TMP} for a cut model: {REPORT['disk_free_gb']}"

# %% [markdown]
# ## The test inputs
#
# Held-out agent examples: the same tasks part 1 kept aside (1 task in 10, chosen by a hash of its name), in the same
# order. Speed prompts: real agent text of 2,000 and 16,000 tokens (the longest agent examples), and 8 different
# 2,000-token prompts for the "8 at once" test.

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir)
rows = [json.loads(x) for x in open(sft_path) if x.strip()]
held = lambda r: int(hashlib.md5(r['instance_id'].encode()).hexdigest(), 16) % 10 == 0  # noqa: E731
rng = random.Random(SEED)
cal_rows = [r for r in rows if not held(r)]
ho_rows = [r for r in rows if held(r)]
rng.shuffle(cal_rows)
rng.shuffle(ho_rows)
MAX_TOKENS = min(MAX_TOKENS, int(part1.get('settings', {}).get('MAX_TOKENS') or MAX_TOKENS))


def encode(r, n=MAX_TOKENS):
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids'][:n]
    starts = np.array([o[0] for o in enc['offset_mapping'][:n]])
    own = np.zeros(len(ids), bool)
    for s, e in r.get('spans') or []:
        own |= (starts >= s) & (starts < e)
    return ids, own


from token_roles import ROLE_NAMES, char_roles
heldout = []
for r in ho_rows:
    ids, own = encode(r)
    pos = [int(t) for t in np.nonzero(own)[0] if t >= 1]
    if pos:
        heldout.append({'ids': [int(x) for x in ids], 'own': pos})
        if QUALITY_BY_ROLE:   # the role of every token itself (its logprob is the prediction of that token)
            cr = char_roles(r['text'])
            st = np.array([o[0] for o in tok(r['text'], add_special_tokens=False,
                                             return_offsets_mapping=True)['offset_mapping'][:len(ids)]])
            heldout[-1]['roles'] = [int(x) for x in cr[np.clip(st, 0, len(cr) - 1)]] if len(cr) else [0] * len(ids)
    if len(heldout) == HELDOUT_ROWS:
        break
assert heldout, 'no held-out example has agent-written tokens within MAX_TOKENS'
long_rows = sorted((encode(r, SPEED_LONG)[0] for r in cal_rows[:200]), key=len, reverse=True)
p_long = [t for x in long_rows for t in x][:SPEED_LONG]     # the longest examples, end to end if one is too short
p_short = [x[:SPEED_SHORT] for x in long_rows[1:10]]
assert len(p_short) == 9 and len(p_short[-1]) == SPEED_SHORT and len(p_long) == SPEED_LONG, 'agent rows too short'
DATA = {'heldout': heldout, 'speed': {'p2k': [int(x) for x in p_short[0]], 'p16k': [int(x) for x in p_long]},
        'streams': [[int(x) for x in p] for p in p_short[1:9]], 'role_names': list(ROLE_NAMES),
        # as token ids: the chat template already starts with <bos>; a text prompt would get a second one from vLLM
        'sanity': tok(tok.apply_chat_template([{'role': 'user', 'content': 'Write a Python function `is_prime(n)` '
                                                                         'with a docstring. Reply with the code only.'}],
                                              tokenize=False, add_generation_prompt=True),
                      add_special_tokens=False)['input_ids']}
# agent states for the next-action test: in each held-out example, the last agent turn that fits whole in MAX_TOKENS;
# the prompt is everything before it (what the agent saw), the recorded turn is what the agent did
import re
CALL = re.compile(r'<\|tool_call>(.*?)<tool_call\|>', re.S)
turns, turn_calls, turn_rows, rows_used = [], [], [], 0
for r in ho_rows:
    if rows_used == HELDOUT_ROWS:
        break
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) < 2:
        continue
    starts = [o[0] for o in enc['offset_mapping']]
    fits = []
    for a, b in r.get('spans') or []:
        i0 = next((i for i, x in enumerate(starts) if x >= a), None)
        i1 = next((i for i, x in enumerate(starts) if x >= b), len(starts))
        if i0 and i1 <= MAX_TOKENS:
            fits.append((i0, a, b))
    if not fits:
        continue
    rows_used += 1
    # the last turn that fits whole (serve-v2's single state) and, with TURNS_PER_ROW > 1, earlier ones spread evenly
    picks = sorted({int(round(x)) for x in np.linspace(0, len(fits) - 1, min(TURNS_PER_ROW, len(fits)))}) \
        if TURNS_PER_ROW > 1 else [len(fits) - 1]
    for k in picks:
        i0, a, b = fits[k]
        m = CALL.search(r['text'][a:b])
        turns.append([int(x) for x in enc['input_ids'][:i0]])
        turn_calls.append(m.group(1) if m else None)
        turn_rows.append(rows_used - 1)
DATA['turns'] = turns
json.dump(DATA, open(f'{TMP}/bench_data.json', 'w'))
REPORT['data'] = {'heldout_rows': len(heldout), 'heldout_own_tokens': sum(len(h['own']) for h in heldout),
                  'quality_rows': min(len(heldout), QUALITY_ROWS or len(heldout)),
                  'quality_tokens_per_role': ({n: sum(sum(1 for x in h['roles'][1:] if x == j)
                                                      for h in heldout[:QUALITY_ROWS or len(heldout)])
                                               for j, n in enumerate(ROLE_NAMES)} if QUALITY_BY_ROLE else None),
                  'max_tokens': MAX_TOKENS, 'streams': len(DATA['streams']), 'turn_states': len(turns),
                  'turn_states_with_recorded_call': sum(c is not None for c in turn_calls),
                  'turn_prompt_tokens': sum(len(t) for t in turns), 'turn_rows': rows_used,
                  'turns_per_row': TURNS_PER_ROW}
# per state (serve-v5 review): its source row (states of one row are not independent: intervals cluster by row), its
# prompt length and the call the agent recorded there
REPORT['turn_state_rows'] = turn_rows
REPORT['turn_state_prompt_tokens'] = [len(t) for t in turns]
REPORT['turn_state_recorded_calls'] = turn_calls
log('data', REPORT['data'])
save_report()

# %% [markdown]
# ## Cut, serve, measure: one model at a time
#
# The full model first (the reference), then each cut. A cut model is written to local disk (about 15-20 GB), served,
# measured and deleted before the next one.

# %%
from slice_model import make_plan, slice_checkpoint, slice_adapter, merge_lora_w4
z = np.load(scores_path)
z = {k: z[k] for k in z.files}
if 'agent_own_ff' in z:  # what the agent reads (all tokens) and what it writes, each normalized per layer
    z['mix_ff'] = (z['agent_ff'] / z['agent_ff'].mean(1, keepdims=True)
                   + z['agent_own_ff'] / z['agent_own_ff'].mean(1, keepdims=True))
L = z[SCORE_KEY].shape[0]
bi = z['agent_bi'] if 'agent_bi' in z else None
order = [int(i) for i in np.argsort(bi) if 0 < i < L - 1] if bi is not None else []
# part 1's measured order: the layers whose removal alone cost the agent least first (score-v2/v3 layer sweep)
order_m = [int(i) for i in (part1.get('layer_sweep') or {}).get('order_least_damage_first', []) if 0 < int(i) < L - 1]
REPORT['layer_orders'] = {'block_influence': order[:12], 'measured': order_m[:12]}
# typical size of what flows into each channel on agent text (score = size x outgoing weight norm), to weight the
# re-quantization error of single-channel cuts
ACT = z[SCORE_KEY] / np.maximum(z['down_norm'], 1e-12) if 'down_norm' in z else None


def gpu_free_gb():
    if TEST:
        return None
    q = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout.split()
    return [round(int(x) / 1024, 1) for x in q]


def bench(name, path, adapter=None, extra_env=None):
    """One model through serve_bench.py in its own process group. Its log and its results (saved after every stage) go
    to the output folder, so a failure keeps what was measured. A process that hangs (for example one vLLM worker
    failing while the others wait) is killed with all its workers, and the next model gets clean GPUs."""
    t = time.time()
    out, logf = f'{OUT}/bench_{name}.json', f'{OUT}/bench_{name}.log'
    free_before = gpu_free_gb()
    with open(logf, 'w') as lf:
        p = subprocess.Popen([sys.executable, f'{LIB}/serve_bench.py', path, out, f'{TMP}/bench_data.json',
                              *([adapter] if adapter else [])],
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, **(extra_env or {})})
        try:
            rc = p.wait(timeout=BENCH_TIMEOUT_MIN * 60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = 'timeout'
        else:
            try:  # workers left behind by a process that exited (an engine that failed to start leaves them)
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if not TEST:
        time.sleep(15)                                        # let the killed or finished workers release the GPUs
    res = json.load(open(out)) if os.path.exists(out) else {}
    res.update(rc=rc, wall_min=round((time.time() - t) / 60, 1), gpu_free_gb_before=free_before,
               gpu_free_gb_after=gpu_free_gb())
    if rc:
        res['log_tail'] = open(logf, errors='replace').read()[-3000:]
        log(name, 'FAILED', rc, res['log_tail'][-1500:])
    return res


REPORT['models']['full'] = bench('full', model_dir, extra_env={'PRUNE_SAMPLE_TURNS': str(SAMPLE_TURNS_FULL)}
                                 if SAMPLE_TURNS_FULL else None)
log('full', {k: v for k, v in REPORT['models']['full'].items() if k != 'sanity'})
save_report()
assert REPORT['models']['full']['rc'] == 0, 'the full model did not run: stop before cutting'
if FULL_REPLICA:   # same weights, other engine numerics (a failure here does not stop the cuts)
    REPORT['models']['full_b'] = bench('full_b', model_dir, extra_env={k: str(v) for k, v in FULL_REPLICA.items()})
    log('full_b', {k: v for k, v in REPORT['models']['full_b'].items() if k not in ('sanity', 'log_tail')})
    save_report()
fails = 0
cur = {'key': None, 'dst': None, 'summ': None}   # the cut checkpoint on disk, reused by the next cut with the same plan
for ci, c in enumerate(CUTS):
    if fails >= MAX_FAILS_IN_A_ROW:
        REPORT['models'][c['name']] = {'rc': 'skipped', 'why': f'the {fails} cuts before it failed'}
        log(c['name'], 'skipped: the cuts before it failed')
        save_report()
        continue
    t = time.time()
    drop = (order_m if c.get('order') == 'measured' else order)[:c['drop']]
    assert len(drop) == c['drop'], f"{c['name']}: only {len(drop)} layers in the {c.get('order', 'block influence')} order"
    skey = c.get('score', SCORE_KEY)
    plan = make_plan(z, 1.0 if c['keep'] >= 1 else c['keep'], drop=drop, score_key=skey, multiple=MULTIPLE,
                     group=c['group'])
    key = (c['keep'], tuple(drop), c['group'], skey)
    extra = {}
    try:
        if key != cur['key']:
            if cur['dst']:
                shutil.rmtree(cur['dst'], ignore_errors=True)
            dst = f'{TMP}/cut_{c["name"]}'
            # on the CPU: this process must not hold GPU memory while vLLM starts (serve-v1 v2: the first re-quantized
            # cut, sliced on GPU 0 in this process, hung vLLM's start-up)
            cur.update(key=key, dst=dst, summ=slice_checkpoint(model_dir, dst, plan, log=lambda *a: None, act=ACT,
                                                               dev='cpu'))
        else:
            extra['reused_cut_of'] = CUTS[ci - 1]['name']
        adapter, serve_dir = None, cur['dst']
        if c.get('adapter') or c.get('merge'):
            pat = c.get('adapter') or c.get('merge')
            src = os.environ.get('PRUNE_ADAPTER_' + c['name'].upper().replace('+', '_').replace('~', 'M')) if TEST else find(pat)
            assert src and os.path.exists(f'{src}/adapter_model.safetensors'), f"{c['name']}: adapter {pat} not found"
            # the mask its own healing notebook trained with (serve-v8 mounts two), else the one found at start-up
            own_mask = os.path.join(os.path.dirname(src), 'tpu', 'heal_mask.npz')
            mask_path = own_mask if os.path.exists(own_mask) else heal_mask_path
            if mask_path:  # the adapter was trained on exactly this cut
                extra['adapter_mask'] = mask_path
                hm = np.load(mask_path)['mask']
                same = hm.shape == (L, z[SCORE_KEY].shape[1]) and all(
                    np.array_equal(np.nonzero(hm[i])[0], np.asarray(plan['ff_idx'][i])) for i in plan['kept_layers'])
                extra['adapter_cut_matches_training_mask'] = bool(same)
                assert same, f"{c['name']}: this cut differs from the mask the adapter was trained with"
            adapter = f'{TMP}/adapter_{c["name"].replace("+", "_").replace("~", "m")}'
            extra['adapter_src'] = src
            extra['adapter_slice'] = slice_adapter(src, adapter, plan)
            if c.get('merge'):  # merged into the 4-bit weights: a separate checkpoint, served without LoRA (B190)
                serve_dir = f'{TMP}/merged_{c["name"].replace("+", "_").replace("~", "m")}'
                t_m = time.time()
                extra['merge'] = merge_lora_w4(cur['dst'], serve_dir, adapter, mode=c.get('merge_mode', 'stochastic'),
                                               seed=c.get('merge_seed', 0), log=lambda *a: None)
                extra['merge']['minutes'] = round((time.time() - t_m) / 60, 1)
                adapter = None
        res = bench(c['name'], serve_dir, adapter)
        if serve_dir != cur['dst']:
            shutil.rmtree(serve_dir, ignore_errors=True)
    except Exception as e:
        res = {'rc': f'setup error: {type(e).__name__}: {str(e)[:300]}', 'wall_min': 0}
        log(c['name'], 'FAILED before serving:', res['rc'])
    res.update(plan_summary=cur['summ'], dropped_layers=drop, slice_min=round((time.time() - t) / 60 - res['wall_min'], 1),
               **extra)
    REPORT['models'][c['name']] = res
    fails = fails + 1 if res['rc'] else 0
    log(c['name'], {k: v for k, v in res.items() if k not in ('sanity', 'log_tail')})
    save_report()
if cur['dst']:
    shutil.rmtree(cur['dst'], ignore_errors=True)

# %% [markdown]
# ## Summary: speed gained vs quality lost (before any repair)

# %%
full = REPORT['models']['full']
table = []
for name, r in REPORT['models'].items():
    if r.get('rc') or 'speed' not in r or 'quality' not in r:
        table.append({'model': name, 'ok': False})
        continue
    table.append({'model': name, 'ok': True,
                  'size_gb': (r.get('plan_summary') or {}).get('gb'),
                  'decode_speedup_2k': round(r['speed']['p2k']['decode_tok_s'] / full['speed']['p2k']['decode_tok_s'], 3),
                  'decode_speedup_16k': round(r['speed']['p16k']['decode_tok_s'] / full['speed']['p16k']['decode_tok_s'], 3),
                  'prefill_speedup_16k': round(r['speed']['p16k']['prefill_tok_s'] / full['speed']['p16k']['prefill_tok_s'], 3),
                  'streams8_speedup': round(r['speed']['streams8']['decode_tok_s_total']
                                            / full['speed']['streams8']['decode_tok_s_total'], 3),
                  'agent_own_nats': r['quality']['agent_own_nats'],
                  'agent_damage': round(r['quality']['agent_own_nats'] - full['quality']['agent_own_nats'], 4)})
def calls(r, key='turns'):
    return [(m.group(1) if m else None) for m in (CALL.search(t) for t in (r.get(key) or []))]


def malformed(c):
    """A tool call whose syntax broke (B188/B189): a quote outside Gemma's <|"|> strings (a JSON-style key), or inside
    a string a backtick followed by ', key:' (the string closed with the wrong delimiter swallowed the next argument)."""
    if not c:
        return False
    parts = c.split('<|"|>')
    return '"' in ''.join(parts[0::2]) or any(re.search(r'`\s*,\s*[a-z_]{2,20}\s*:', x) for x in parts[1::2])


def malformed_stats(cs):
    made = [x for x in cs if x is not None]
    bad = sum(malformed(x) for x in made)
    return {'calls': len(made), 'malformed': bad, 'share': round(bad / max(1, len(made)), 4)}


def agree(xs, ys):
    """Share of states where both made a tool call and the calls are identical (name and arguments), and where the
    tool names match (the text before the first '{')."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return None
    name = lambda c: c.split('{', 1)[0] if c else None  # noqa: E731
    return {'states': len(pairs), 'same_call': round(sum(x == y for x, y in pairs) / len(pairs), 3),
            'same_tool': round(sum(name(x) == name(y) for x, y in pairs) / len(pairs), 3)}


full_calls = calls(full)
for row in table:
    r = REPORT['models'].get(row['model'], {})
    if row.get('ok') and r.get('turns'):
        mine = calls(r)
        row['next_action_vs_full'] = agree(mine, full_calls)        # does the cut act as the full model would?
        row['next_action_vs_recorded'] = agree(mine, turn_calls)    # ... as the agent actually did (sampled)?
        row['malformed_calls'] = malformed_stats(mine)
    if row.get('ok') and (r.get('quality') or {}).get('agent_all_nats') is not None:
        row['agent_all_nats'] = r['quality']['agent_all_nats']
        row['all_damage'] = round(r['quality']['agent_all_nats'] - full['quality'].get('agent_all_nats', float('nan')), 4)
    if row.get('ok') and (r.get('quality') or {}).get('by_role'):
        fb = full['quality'].get('by_role') or {}
        row['role_nats'] = {g: v['nats'] for g, v in r['quality']['by_role'].items()}
        row['role_damage'] = {g: round(v['nats'] - fb[g]['nats'], 4) for g, v in r['quality']['by_role'].items() if g in fb}
if full.get('turns_sampled'):  # the full model against itself at the agent's temperature: the ceiling of agreement
    fs = calls(full, 'turns_sampled')
    table.append({'model': f"full@T{full.get('turns_sampled_temperature')}", 'ok': True,
                  'next_action_vs_full': agree(fs, full_calls), 'next_action_vs_recorded': agree(fs, turn_calls),
                  'malformed_calls': malformed_stats(fs)})
REPORT['recorded_calls_malformed'] = malformed_stats(turn_calls)
REPORT['table'] = table
REPORT['timings_min']['total'] = round((time.time() - T0) / 60, 1)
save_report()
print(json.dumps(table, indent=1))
for name, r in REPORT['models'].items():
    print(f'--- {name} sanity ---\n{(r.get("sanity") or "")[:400]}')
log('done')
