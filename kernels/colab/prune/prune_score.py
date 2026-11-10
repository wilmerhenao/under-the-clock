# %% [markdown]
# # Pruning Gemma 4 31B for a coding agent, part 1: what does the agent actually use?
#
# **Private notebook. Do not share or publish.**
#
# ## The idea in plain words
#
# Gemma 4 31B is a general model: it can write poems, answer history questions and translate Thai. Our agent only
# reads code, runs tests and edits files. Every time the agent writes one word (a "token"), the GPU has to read the
# whole model, all 31 billion numbers, from memory. That reading is what makes each step slow, and in our
# competition runs slow steps were what lost tasks: when the agent ran out of time it solved nothing.
#
# So the question is: **which parts of the model does the agent actually use, and can we remove the rest to make it
# faster?**
#
# A model like this is a stack of 60 *layers*. Each layer has two parts:
#
# * an **attention** part (it decides which earlier words to look at; 32 "heads", each looking for something
#   different), and
# * a **feed-forward** part: 21,504 small units ("channels") that each fire on some pattern. Most of what the
#   model *knows* lives here, and it is about 70% of all the numbers in the model.
#
# This notebook runs the full model over the agent's own work (real runs of our agent: reading issues, reading files,
# thinking, calling tools) and **measures how much every channel, every attention head and every layer contributes**.
# It does the same on ordinary Wikipedia text, for comparison: parts that matter for Wikipedia but not for the agent
# are the ones we would like to cut.
#
# Then it **tries some cuts without any retraining** and measures the damage: how much worse the model gets at
# predicting what our agent actually wrote, and how much worse at Wikipedia. If the cuts are aimed well, the agent
# damage should stay small while the Wikipedia damage grows. Part 2 (a separate notebook) makes the real smaller
# model and repairs it by letting the full model teach it.
#
# ## Where it runs
#
# * **Kaggle** (how we run it): a private script on the 4 x L4 GPU machine (4 x 22.5 GB). The model is spread over
#   the four GPUs; it reads the model, our agent data and the Wikipedia text from attached Kaggle inputs (these
#   machines have no internet) and writes its results to the run's output.
# * **Colab** (alternative): G4 or H100 runtime; it downloads the model with your Kaggle credentials (Colab
#   Secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`) and uploads the results to a private Kaggle dataset.
#
# Leave `SMOKE = True` for a quick test on a few examples; `SMOKE = False` is the real measurement (about 1.5-2.5
# hours on 4 x L4). Either way the notebook first checks how long an example the GPUs can take and runs every step
# once on one example, so a problem shows up in the first minutes, not after an hour.

# %%
# ---- Settings: the only cell you normally edit ----
SMOKE = True                 # True: quick test on a few short examples. False: the full measurement.
RUN_NAME = 'score-v1'        # name of this run (results folder; on Colab also the private dataset g4dev-prune-<name>)
MODEL_HANDLE = 'google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2'   # the model we train on (bf16)
AGENT_DATA = 'verracodeguacas/g4dev-g11-data'   # our agent's own successful runs (private)
CAL_ROWS = 300               # agent examples used to measure what the model uses
HELDOUT_ROWS = 40            # agent examples kept aside (other tasks) to measure the damage of each cut
MAX_TOKENS = 8192            # longest example (tokens); lowered automatically if the GPUs run out of memory
WIKI_TOKENS = 300_000        # Wikipedia text for comparison (measurement)
WIKI_HELDOUT_CHUNKS = 24     # Wikipedia pieces of 2,048 tokens kept aside to measure damage
KEEP_FRACS = [0.85, 0.70, 0.55]   # share of feed-forward channels kept in every layer
DROP_LAYERS = [6, 12]        # numbers of whole layers removed
LAYER_SWEEP_TOKENS = 32_000  # tokens (in total) for trying each layer's removal on its own; 0 = skip that test
LAYER_SWEEP_OWN = 300        # each example used there is read up to its first 300 agent-written tokens
REUSE_AGENT = False          # True: take the agent measurements from an earlier run of this notebook (attached as an
                             # input) and measure only Wikipedia again; the agent numbers are checked to match first
SEED = 0
ROLE_ONLY = False            # True (score-v4, B202; set by build_kaggle.py --role): only the agent examples again, counted
                             # separately for each role of the token being written (prune_roles.py cells appended);
                             # no Wikipedia, no layer test, no cuts here
if SMOKE:
    CAL_ROWS, HELDOUT_ROWS, WIKI_TOKENS, WIKI_HELDOUT_CHUNKS = 3, 2, 6_000, 2
    KEEP_FRACS, DROP_LAYERS = [0.70], [6]
    LAYER_SWEEP_TOKENS, LAYER_SWEEP_OWN = 3000, 50
    RUN_NAME += '-smoke'

# %% [markdown]
# ## Setup
#
# Checks the GPUs and installs the library versions this was tested with (on Kaggle: the competition's own
# wheelhouse, the same stack our agent evaluations use).

# %%
import os, sys, json, time, math, glob, hashlib, shutil, subprocess, random
T0 = time.time()
TINY = os.environ.get('PRUNE_TINY') == '1'   # internal CPU test with a tiny random model (no downloads, no upload)
KAGGLE = os.path.isdir('/kaggle/input') and not TINY
COLAB = not KAGGLE and not TINY
if TINY:
    MAX_TOKENS = min(MAX_TOKENS, 6144)


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return hits[0] if hits else None


if not TINY:
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                         capture_output=True, text=True).stdout, flush=True)
if KAGGLE:
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    wh = os.path.dirname(find('swegemma-*.whl'))      # the competition wheelhouse: torch, transformers 5.13.1, ...
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
    if subprocess.run([sys.executable, '-c', 'import accelerate'], capture_output=True).returncode:
        # no internet on the competition's GPU machines: our private wheel dataset has it
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', find('accelerate-*.whl')], check=True)
elif COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.13.1', 'accelerate', 'kagglehub',
                    'kaggle', 'matplotlib', 'pyarrow'], check=True)


def kaggle_auth():
    """Colab only: Kaggle credentials from Colab secrets or Drive's kaggle.json; values are never printed."""
    have = lambda: bool(os.environ.get('KAGGLE_API_TOKEN') or (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')))  # noqa: E731
    if have():
        return 'environment'
    try:
        from google.colab import userdata
        for k in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
            try:
                v = userdata.get(k)
                if v:
                    os.environ[k] = v
            except Exception:
                pass
        if have():
            return 'Colab secrets'
        from google.colab import drive
        if not os.path.isdir('/content/drive'):
            drive.mount('/content/drive')
    except ImportError:
        pass
    p = '/content/drive/MyDrive/kaggle.json'
    if os.path.exists(p):
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        shutil.copy(p, os.path.expanduser('~/.kaggle/kaggle.json'))
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        cred = json.load(open(p))
        os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = cred['username'], cred['key']
        return 'Drive kaggle.json'
    raise SystemExit('Kaggle credentials not found: add KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets (key icon).')


if COLAB:
    log('Kaggle credentials from', kaggle_auth())
    free_gb = shutil.disk_usage('/root').free / 2**30
    log(f'free disk {free_gb:.0f} GB')
    assert free_gb > 75, 'need about 75 GB of free disk for the 62 GB model'

import numpy as np
import torch
import torch.nn.functional as F
OUT = os.path.abspath(os.environ.get('PRUNE_OUT') or (f'/kaggle/working/prune/{RUN_NAME}' if KAGGLE
                                                       else f'/content/prune_results/{RUN_NAME}'))
os.makedirs(OUT, exist_ok=True)
REPORT = {'run': RUN_NAME, 'smoke': SMOKE, 'platform': 'kaggle' if KAGGLE else 'colab' if COLAB else 'cpu-test',
          'model': MODEL_HANDLE, 'agent_data': AGENT_DATA, 'settings': {
              'CAL_ROWS': CAL_ROWS, 'HELDOUT_ROWS': HELDOUT_ROWS, 'MAX_TOKENS': MAX_TOKENS, 'WIKI_TOKENS': WIKI_TOKENS,
              'KEEP_FRACS': KEEP_FRACS, 'DROP_LAYERS': DROP_LAYERS, 'LAYER_SWEEP_TOKENS': LAYER_SWEEP_TOKENS,
              'LAYER_SWEEP_OWN': LAYER_SWEEP_OWN, 'REUSE_AGENT': REUSE_AGENT, 'SEED': SEED, 'ROLE_ONLY': ROLE_ONLY},
          'timings_min': {}}
if torch.cuda.is_available():
    REPORT['gpus'] = [f'{torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GB'
                      for i in range(torch.cuda.device_count())]
    log('torch', torch.__version__, '| GPUs:', REPORT['gpus'])


def save_report():
    json.dump(REPORT, open(f'{OUT}/report.json', 'w'), indent=1)

# %% [markdown]
# ## The model and the agent's runs
#
# On Kaggle both are attached inputs. On Colab the 62 GB model is downloaded first (roughly 10-30 minutes). The agent
# data is our own private dataset of successful agent runs: each example is the exact text the model saw and wrote
# during one run, with marks showing which parts the model itself wrote.

# %%
if TINY:
    model_dir, sft_path = os.environ['PRUNE_TOK_DIR'], os.environ['PRUNE_ROWS']
elif KAGGLE:
    model_dir = os.path.dirname(find('gemma-4-31b-it-qat-q4_0-unquantized/*/config.json')
                                or find('**/gemma-4-31b-it-qat-q4_0-unquantized/**/config.json'))
    sft_path = find('g4dev-g11-data/sft.jsonl') or find('**/sft.jsonl')
else:
    import kagglehub
    model_dir = kagglehub.model_download(MODEL_HANDLE)
    sft_path = sorted(glob.glob(f'{kagglehub.dataset_download(AGENT_DATA)}/**/sft.jsonl', recursive=True))[0]
REPORT['timings_min']['inputs'] = round((time.time() - T0) / 60, 1)
log('model at', model_dir, '| agent rows at', sft_path)

# %% [markdown]
# ## Load the model
#
# The full model in 16-bit numbers takes about 62 GB. On 4 x L4 it is split across the four GPUs, about 16 GB each,
# leaving room for the work. We only run it forward (no training here).

# %%
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(model_dir)
if TINY:
    from transformers import Gemma4Config
    c = json.load(open(os.path.join(model_dir, 'config.json')))
    c['text_config'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=14, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, num_global_key_value_heads=1, global_head_dim=32,
                            layer_types=None)
    c['vision_config'].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                              num_key_value_heads=2, head_dim=16)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(Gemma4Config(**{k: v for k, v in c.items()
                                                           if k not in ('architectures', 'transformers_version')}))
    for lyr in model.model.language_model.layers:  # random layer scalars, so dropping a layer must keep its scale
        lyr.layer_scalar.fill_(0.5 + torch.rand(1).item())
else:
    t_load = time.time()
    n_gpu = torch.cuda.device_count()
    per = torch.cuda.get_device_properties(0).total_memory / 2**30
    # leave room for activations on every GPU: on 4 x 22.5 GB about 18 GB of weights each, on one big GPU all of it
    cap = {i: f'{int(per - 4.5) if n_gpu > 1 else int(per - 3)}GiB' for i in range(n_gpu)}
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='auto',
                                                           max_memory=cap, attn_implementation='sdpa')
    REPORT['timings_min']['load'] = round((time.time() - t_load) / 60, 1)
    REPORT['weights_per_gpu_gb'] = [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]
model.eval()
TM = model.model.language_model          # the text part: embeddings, 60 layers, final norm
LAYERS = list(TM.layers)
L = len(LAYERS)
TCFG = model.config.get_text_config()
I = TCFG.intermediate_size
H = TCFG.num_attention_heads
CAP = TCFG.final_logit_softcapping
DEV0 = TM.embed_tokens.weight.device     # where the input goes (and where the scores are kept)
HEAD_DEV = model.lm_head.weight.device
REPORT['model_shape'] = {'layers': L, 'ff_channels_per_layer': I, 'heads': H, 'hidden': TCFG.hidden_size,
                         'layer_types': list(TCFG.layer_types), 'vocab': TCFG.vocab_size}
REPORT['layer_scalar'] = [round(float(x.layer_scalar.float()), 4) for x in LAYERS]
log(f'{L} layers, {I} feed-forward channels per layer, {H} heads; weights per GPU (GB):', REPORT.get('weights_per_gpu_gb'))
save_report()

# %% [markdown]
# ## The measuring probes and the cut switches
#
# Small "hooks" are attached inside every layer. While the model reads an example they record:
#
# * **feed-forward channels:** how strongly each of the 21,504 channels fires, times how much its output can move
#   the model (the size of its outgoing weights). A channel that rarely fires, or barely moves anything, scores low.
# * **attention heads:** the same idea per head.
# * **layers:** how much each layer changes its input. A layer whose output is almost the same as its input is doing
#   little (this is the "block influence" used in earlier pruning work).
#
# Each is counted separately for: all agent tokens, the tokens the agent wrote itself, and Wikipedia.
#
# The same hooks can also **switch parts off**, which gives exactly the same answers as a smaller model with those
# parts removed: a channel switched off contributes nothing, and a skipped layer passes its input through unchanged
# (keeping the layer's scale factor, which in Gemma 4 multiplies everything after each layer).

# %%
ROLE_NAMES = ('read', 'thought', 'args', 'syntax', 'close')   # (ROLE_ONLY) the token roles, see token_roles below
RG = tuple(f'role_{n}' for n in ROLE_NAMES)
GROUPS = ('agent', 'agent_own', 'wiki') + (RG if ROLE_ONLY else ())
ACC = {g: {'ff_abs': torch.zeros(L, I, device=DEV0), 'ff_sq': torch.zeros(L, I, device=DEV0),
           'head_sq': torch.zeros(L, H, device=DEV0), 'bi': torch.zeros(L, device=DEV0), 'n': 0} for g in GROUPS}
STATE = {'sel': [],      # [(group, CPU index tensor or None)] for the current example; empty = measure nothing
         'w': None}      # (ROLE_ONLY) (tokens x groups weights, group names): feed-forward sums of every group at once
MASK = {'ff': None, 'drop': set()}


def ff_hook(li):
    def f(mod, args):
        x = args[0]
        if MASK['ff'] is not None:                                # switched-off channels contribute nothing
            x = x * MASK['ff'][li].to(x.device, x.dtype)
        if STATE['w'] is not None:                                # (ROLE_ONLY) one product for all groups
            W, names = STATE['w']
            xf = x[0].float()
            sq = W.to(xf.device).T @ (xf * xf)                    # (groups, channels): sums of squares per group
            for j, g in enumerate(names):
                ACC[g]['ff_sq'][li] += sq[j].to(DEV0)
        if STATE['sel']:
            xf = x[0].float()                                     # (tokens, channels): what goes into down_proj
            for g, sel in STATE['sel']:
                xs = xf if sel is None else xf[sel.to(xf.device)]
                ACC[g]['ff_abs'][li] += xs.abs().sum(0).to(DEV0)
                ACC[g]['ff_sq'][li] += (xs * xs).sum(0).to(DEV0)
        if MASK['ff'] is not None:
            return (x,)
    return f


def head_hook(li):
    def f(mod, args):
        if not STATE['sel']:
            return
        x = args[0][0].float()                                    # (tokens, heads * head size): into o_proj
        x = x.reshape(x.shape[0], H, -1)
        for g, sel in STATE['sel']:
            xs = x if sel is None else x[sel.to(x.device)]
            ACC[g]['head_sq'][li] += (xs * xs).sum((0, 2)).to(DEV0)
    return f


def layer_hook(li):
    def f(mod, args, kwargs, out):
        # on several GPUs a layer's input can still sit on the previous GPU (the layer moves its own copy): align first
        h_in = args[0].to(out.device)
        if li in MASK['drop']:          # skip the layer but keep its scale factor: same as removing it and folding the
            return h_in * mod.layer_scalar.to(out.device, h_in.dtype)   # scale into the layer before (exact)
        if STATE['sel']:
            cos = F.cosine_similarity(h_in[0].float(), out[0].float(), dim=-1)   # 1 = the layer changed nothing
            for g, sel in STATE['sel']:
                ACC[g]['bi'][li] += (1 - (cos if sel is None else cos[sel.to(cos.device)])).sum().to(DEV0)
    return f


HOOKS = []
for li, lyr in enumerate(LAYERS):
    HOOKS += [lyr.mlp.down_proj.register_forward_pre_hook(ff_hook(li)),
              lyr.self_attn.o_proj.register_forward_pre_hook(head_hook(li)),
              lyr.register_forward_hook(layer_hook(li), with_kwargs=True)]


@torch.no_grad()
def forward_only(ids):
    TM(input_ids=ids[None].to(DEV0), use_cache=False)


@torch.no_grad()
def surprise(examples, own_only):
    """Mean cross-entropy (nats per token) of predicting the next token, on the agent's own tokens or on all tokens."""
    tot, n = 0.0, 0
    for item in examples:
        ids, own = item if own_only else (item, torch.ones(len(item), dtype=torch.bool))
        h = TM(input_ids=ids[None].to(DEV0), use_cache=False).last_hidden_state[0]
        pos = torch.nonzero(own[1:]).squeeze(1)        # position t predicts token t + 1
        for ch in pos.split(1024):
            logits = model.lm_head(h[ch.to(h.device)].to(HEAD_DEV)).float()
            if CAP:
                logits = torch.tanh(logits / CAP) * CAP
            tot += float(F.cross_entropy(logits, ids[ch + 1].to(logits.device), reduction='sum'))
            n += len(ch)
    return tot / max(1, n)


def masks_from(score, frac, how):
    m = torch.zeros(L, I, device=DEV0)
    g = torch.Generator().manual_seed(SEED)
    for i in range(L):
        k = max(1, round(frac * I))
        idx = (torch.randperm(I, generator=g)[:k].to(DEV0) if how == 'random' else score[i].topk(k).indices)
        m[i, idx] = 1.0
    return m

# %% [markdown]
# ## How long an example fits?
#
# Longer examples need more GPU memory for the work in between. Before measuring anything, we try one example of
# 8,192 tokens with the probes on; if a GPU runs out of memory we try 6,144, then 4,096 tokens. The longest that fits
# is used for everything below.

# %%
if not TINY:
    fit = None
    STATE['sel'] = [('agent', None)]                 # probes on (realistic memory); the counts are reset after
    for n_try in [x for x in (MAX_TOKENS, 6144, 4096, 3072) if x <= MAX_TOKENS]:
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 100000, (n_try,))
            if ROLE_ONLY:                            # the role measurement's own probe (its memory, not the other's)
                STATE['sel'], STATE['w'] = [], (torch.ones(n_try, 2 + len(RG)), ['agent', 'agent_own', *RG])
            forward_only(ids)
            surprise([(ids, torch.ones(n_try, dtype=torch.bool))], True)
            fit = n_try
            REPORT['peak_gpu_memory_gb_at_max_tokens'] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                                                         for i in range(torch.cuda.device_count())]
            break
        except torch.OutOfMemoryError as e:
            log(f'{n_try} tokens did not fit ({str(e)[:120]}); trying shorter')
            torch.cuda.empty_cache()
    assert fit, 'even 3,072 tokens do not fit'
    for g in GROUPS:
        for k in ('ff_abs', 'ff_sq', 'head_sq', 'bi'):
            ACC[g][k].zero_()
    STATE['sel'], STATE['w'] = [], None
    MAX_TOKENS = fit
    REPORT['settings']['MAX_TOKENS'] = fit
    log(f'examples up to {fit} tokens; peak memory per GPU (GB):', REPORT['peak_gpu_memory_gb_at_max_tokens'])
    save_report()

# %% [markdown]
# ## The examples
#
# Agent examples are split **by task**: the tasks used for measuring are never the tasks used for judging damage.
# For each example we also mark which tokens the model itself wrote (its reasoning and tool calls), because that is
# what we most need to keep working. Wikipedia (WikiText-103) is the "everything else" comparison.

# %%
rows = [json.loads(x) for x in open(sft_path) if x.strip()]
held = lambda r: int(hashlib.md5(r['instance_id'].encode()).hexdigest(), 16) % 10 == 0  # noqa: E731  (1 task in 10)
rng = random.Random(SEED)
cal_rows = [r for r in rows if not held(r)]
ho_rows = [r for r in rows if held(r)]
rng.shuffle(cal_rows)
rng.shuffle(ho_rows)


def encode(r):
    """Token ids (first MAX_TOKENS) and a mask of the tokens the model wrote itself (the row's `spans`)."""
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids'][:MAX_TOKENS]
    starts = np.array([o[0] for o in enc['offset_mapping'][:MAX_TOKENS]])
    own = np.zeros(len(ids), bool)
    for s, e in r.get('spans') or []:
        own |= (starts >= s) & (starts < e)
    return torch.tensor(ids), torch.from_numpy(own)


def take(cands, n, src=None):
    """The first n examples (in shuffled order) that contain tokens the agent wrote within their first MAX_TOKENS
    (src: a list that receives the rows they came from)."""
    out = []
    for r in cands:
        x = encode(r)
        if x[1][1:].any():
            out.append(x)
            if src is not None:
                src.append(r)
        if len(out) == n:
            break
    return out


CAL_SRC, HO_SRC = [], []
CAL = take(cal_rows, CAL_ROWS, CAL_SRC)
HO = take(ho_rows, HELDOUT_ROWS, HO_SRC)
assert CAL and HO, 'no examples with agent-written tokens in their first MAX_TOKENS tokens'


def wiki_text():
    if TINY:
        return ' '.join(['The river flows through the old city, past the cathedral and the market where farmers sell '
                         'apples, cheese and bread every Saturday morning. Historians disagree about its founding.'] * 400)
    import io
    import urllib.request
    import pandas as pd
    parts = []
    for split in ('validation', 'test'):
        if KAGGLE:   # a private copy of the two WikiText-103 files (the GPU machines have no internet)
            data = open(find(f'wikitext-103-raw-v1-{split}.parquet'), 'rb').read()
        else:
            url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/{split}-00000-of-00001.parquet'
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        parts += [x for x in pd.read_parquet(io.BytesIO(data))['text'] if x.strip()]
    return '\n'.join(parts)


def wiki_chunks(n_tokens, n_heldout, size):
    """Pieces of `size` tokens: Wikipedia text presented as the model's own answer to "Write an encyclopedia article."
    (its chat format, like every agent example), plus a mask of the article's tokens (the only ones counted).
    Why: this chat-tuned model expects its turn structure. On bare text it keeps predicting its own turn markers
    (`<|channel>`) and scores 5.9 (no <bos>, run score-v1) to 7.6 (<bos>, score-v2) nats/token; inside a chat turn
    the same text scores ~2.3-2.7 (diagnostic, B171)."""
    ids = tok(wiki_text(), add_special_tokens=False)['input_ids']
    pre = tok(tok.apply_chat_template([{'role': 'user', 'content': 'Write an encyclopedia article.'}], tokenize=False,
                                      add_generation_prompt=True), add_special_tokens=False)['input_ids']
    assert pre[0] == tok.bos_token_id, 'the chat prompt should start with <bos>'
    body = size - len(pre)
    chunks = []
    for i in range(0, len(ids) - body, body):
        x = torch.tensor(pre + ids[i:i + body])
        m = torch.zeros(len(x), dtype=torch.bool)
        m[len(pre):] = True
        chunks.append((x, m))
    need = math.ceil(n_tokens / body)
    return chunks[:need], chunks[need:need + n_heldout]


WIKI_CAL, WIKI_HO = (wiki_chunks(WIKI_TOKENS, WIKI_HELDOUT_CHUNKS, size=2048 if not TINY else 256)
                     if not ROLE_ONLY else ([], []))       # (ROLE_ONLY: agent examples only)
REPORT['data'] = {'agent_rows_total': len(rows), 'cal_rows': len(CAL), 'cal_tokens': int(sum(len(x[0]) for x in CAL)),
                  'cal_own_tokens': int(sum(int(x[1].sum()) for x in CAL)), 'heldout_rows': len(HO),
                  'heldout_tasks_share': round(len(ho_rows) / max(1, len(rows)), 3),
                  'heldout_own_tokens': int(sum(int(x[1].sum()) for x in HO)),
                  'wiki_cal_tokens': int(sum(int(m.sum()) for _, m in WIKI_CAL)), 'wiki_heldout_chunks': len(WIKI_HO),
                  'wiki_format': 'article inside a chat turn (model answer)'}
log('data', REPORT['data'])

# %% [markdown]
# ## A first run of every step on one example
#
# Before the long measurement: switch some channels and two layers off at random and compute the damage on one
# held-out example. This exercises every piece once, so a problem shows up now.

# %%
t_pre = time.time()
base_pre = surprise(HO[:1], True)
MASK['ff'], MASK['drop'] = masks_from(None, 0.7, 'random'), {0, 1}
cut_pre = surprise(HO[:1], True)
MASK['ff'], MASK['drop'] = None, set()
assert math.isfinite(base_pre) and math.isfinite(cut_pre) and cut_pre != base_pre, (base_pre, cut_pre)
REPORT['preflight'] = {'full_nats': round(base_pre, 4), 'random_cut_nats': round(cut_pre, 4),
                       's': round(time.time() - t_pre, 1)}
log('preflight', REPORT['preflight'])
save_report()

# %% [markdown]
# ## (Optional) Reusing the agent measurements of an earlier run
#
# The agent half of the measurement takes about 1.5 hours. With `REUSE_AGENT = True` the notebook takes the agent
# scores (and the agent damage of cuts that depend only on them) from an earlier run of this notebook, attached as an
# input, and measures only the rest again. First it checks that it really has the same model and the same held-out
# agent examples: the full model's agent number must come out the same as last time.

# %%
PREV, PREV_AGENT, FULL_AGENT_NOW = None, {}, None
if REUSE_AGENT:
    prev_dir = os.environ.get('PRUNE_PREV') or os.path.dirname(find('**/scores.npz') or '')
    assert prev_dir and os.path.exists(f'{prev_dir}/report.json'), 'REUSE_AGENT needs an earlier run attached'
    PREV = np.load(f'{prev_dir}/scores.npz')
    prev_rep = json.load(open(f'{prev_dir}/report.json'))
    for k in ('CAL_ROWS', 'HELDOUT_ROWS', 'MAX_TOKENS', 'SEED'):
        assert prev_rep['settings'][k] == REPORT['settings'][k], (k, prev_rep['settings'][k], REPORT['settings'][k])
    PREV_AGENT = {r['variant']: r['agent_own_nats'] for r in prev_rep['damage']}
    t_chk = time.time()
    FULL_AGENT_NOW = round(surprise(HO, True), 4)
    REPORT['reuse'] = {'from_run': prev_rep['run'], 'from_dir': prev_dir, 'full_agent_prev': PREV_AGENT['full'],
                       'full_agent_now': FULL_AGENT_NOW, 'check_min': round((time.time() - t_chk) / 60, 1)}
    log('reuse check', REPORT['reuse'])
    save_report()
    assert abs(FULL_AGENT_NOW - PREV_AGENT['full']) < 0.005, 'not the same model or examples as the earlier run'

# %% [markdown]
# ## Measuring what each part contributes
#
# Now the model reads every measurement example (agent first, then Wikipedia) with the probes on.

# %%
t_meas = time.time()
n_tok = 0
for k, (ids, own) in enumerate(CAL if PREV is None else []):   # (reused: Wikipedia only)
    own_idx = torch.nonzero(own).squeeze(1)
    STATE['sel'] = [('agent', None)] + ([('agent_own', own_idx)] if len(own_idx) else [])
    forward_only(ids)
    ACC['agent']['n'] += len(ids)
    ACC['agent_own']['n'] += len(own_idx)
    n_tok += len(ids)
    if k % 25 == 0:
        log(f'agent example {k + 1}/{len(CAL)}, {n_tok:,} tokens, {n_tok / max(1e-6, time.time() - t_meas):,.0f} tok/s')
for ids, m in WIKI_CAL:
    art_idx = torch.nonzero(m).squeeze(1)              # the article's tokens only (not the short chat prompt)
    STATE['sel'] = [('wiki', art_idx)]
    forward_only(ids)
    ACC['wiki']['n'] += len(art_idx)
    n_tok += len(ids)
STATE['sel'] = []
dt = time.time() - t_meas
REPORT['timings_min']['measure'] = round(dt / 60, 1)
REPORT['measure_tok_per_s'] = round(n_tok / dt, 1)
if torch.cuda.is_available():
    REPORT['peak_gpu_memory_gb'] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                                    for i in range(torch.cuda.device_count())]
log(f'measured {n_tok:,} tokens in {dt / 60:.1f} min; peak GPU memory', REPORT.get('peak_gpu_memory_gb'))
save_report()

# %% [markdown]
# ## Turning the measurements into scores
#
# Score of a feed-forward channel = (typical size of its activity) x (size of its outgoing weights). Score of a
# layer = average "how much it changed its input". The comparison with Wikipedia tells us whether the agent uses a
# different part of the model: if agent and Wikipedia scores were the same, there would be nothing agent-specific to
# cut.

# %%
with torch.no_grad():
    down_norm = torch.stack([lyr.mlp.down_proj.weight.float().norm(dim=0).to(DEV0) for lyr in LAYERS])     # (L, I)
    o_norm = torch.stack([lyr.self_attn.o_proj.weight.float().view(lyr.self_attn.o_proj.weight.shape[0], H, -1)
                          .pow(2).sum((0, 2)).sqrt().to(DEV0) for lyr in LAYERS])                             # (L, H)
SC = {}
for g in GROUPS:
    n = max(1, ACC[g]['n'])
    SC[g] = {'ff': (ACC[g]['ff_sq'] / n).sqrt() * down_norm, 'ff_abs': ACC[g]['ff_abs'] / n,
             'head': (ACC[g]['head_sq'] / n).sqrt() * o_norm, 'bi': ACC[g]['bi'] / n}
if PREV is not None:   # the earlier run's agent scores (same model and examples: checked above)
    for g in ('agent', 'agent_own'):
        SC[g] = {k: torch.from_numpy(PREV[f'{g}_{k}']).float().to(DEV0) for k in ('ff', 'ff_abs', 'head', 'bi')}
        ACC[g]['n'] = int(PREV[f'n_{g}'])


def rank_corr(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def kept(score_row, frac):
    return set(score_row.topk(max(1, round(frac * score_row.numel()))).indices.tolist())


cmp = {'per_layer_rank_corr_agent_vs_wiki': [round(rank_corr(SC['agent']['ff'][i], SC['wiki']['ff'][i]), 3)
                                             for i in range(L)],
       'per_layer_rank_corr_agent_vs_agent_own': [round(rank_corr(SC['agent']['ff'][i], SC['agent_own']['ff'][i]), 3)
                                                  for i in range(L)]}
for f in KEEP_FRACS:  # how different are the channels the agent keeps from the ones Wikipedia would keep?
    ov = [len(kept(SC['agent']['ff'][i], f) & kept(SC['wiki']['ff'][i], f)) / max(1, round(f * I)) for i in range(L)]
    cmp[f'kept_overlap_agent_vs_wiki_at_{f}'] = round(float(np.mean(ov)), 3)
# channels important for Wikipedia (top half) that the agent barely uses (bottom 30%): "general knowledge" the agent
# does not need
ag_r = SC['agent']['ff'].argsort(1).argsort(1).float() / I
wk_r = SC['wiki']['ff'].argsort(1).argsort(1).float() / I
cmp['share_wiki_top_half_and_agent_bottom_30pct'] = round(float(((wk_r >= 0.5) & (ag_r < 0.3)).float().mean()), 4)
cmp['block_influence'] = {g: [round(float(x), 5) for x in SC[g]['bi']] for g in GROUPS}
REPORT['compare'] = cmp
np.savez_compressed(f'{OUT}/scores.npz', **{f'{g}_{k}': v.cpu().numpy() for g in GROUPS for k, v in SC[g].items()},
                    down_norm=down_norm.cpu().numpy(), o_norm=o_norm.cpu().numpy(),
                    layer_scalar=np.array(REPORT['layer_scalar']), **{f'n_{g}': np.array(ACC[g]['n']) for g in GROUPS})
log('agent vs Wikipedia channel ranking, mean rank correlation over layers:',
    round(float(np.mean(cmp['per_layer_rank_corr_agent_vs_wiki'])), 3))
save_report()

# %% [markdown]
# ## Which single layers can go? (each tried on its own)
#
# "How much a layer changes its input" (above) is a cheap guess at which layers matter, and it can mislead. The
# direct test: remove one layer at a time and measure the damage on a few *measurement* examples (never the held-out
# ones). The layers whose removal hurts least are the honest candidates for removing whole layers, so below the
# layer cuts are tried both ways: chosen by the cheap guess and chosen by this test.

# %%
SWEEP = None
if not LAYER_SWEEP_TOKENS and PREV is not None and 'layer_sweep' in prev_rep:   # the earlier run's test, reused
    dbl = prev_rep['layer_sweep']['damage_by_layer']
    SWEEP = [dbl[str(i)] for i in range(1, L - 1)]
    order_m = prev_rep['layer_sweep']['order_least_damage_first']
    REPORT['layer_sweep'] = dict(prev_rep['layer_sweep'], reused_from=prev_rep['run'])
    log('layer sweep reused from', prev_rep['run'], order_m[:12])
elif LAYER_SWEEP_TOKENS:
    t_sw = time.time()
    sw, n_sw = [], 0
    for ids, own in CAL:                               # measurement examples (shuffled), never the held-out ones
        pos = torch.nonzero(own[1:]).squeeze(1) + 1
        end = int(pos[min(len(pos), LAYER_SWEEP_OWN) - 1]) + 1   # later tokens cannot change the score
        sw.append((ids[:end], own[:end]))
        n_sw += end
        if n_sw >= LAYER_SWEEP_TOKENS:
            break
    base_sw = surprise(sw, True)
    SWEEP = []
    for i in range(1, L - 1):                          # never the first or the last layer
        MASK['drop'] = {i}
        SWEEP.append(round(surprise(sw, True) - base_sw, 5))
        if i % 10 == 0:
            log(f'layer sweep {i}/{L - 2}: {(time.time() - t_sw) / 60:.1f} min')
    MASK['drop'] = set()
    order_m = [int(i) + 1 for i in np.argsort(SWEEP, kind='stable')]
    REPORT['layer_sweep'] = {'rows': len(sw), 'tokens': int(sum(len(x[0]) for x in sw)),
                             'own_tokens': int(sum(int(x[1][1:].sum()) for x in sw)), 'base_nats': round(base_sw, 4),
                             'damage_by_layer': {str(i): d for i, d in zip(range(1, L - 1), SWEEP)},
                             'order_least_damage_first': order_m, 'min': round((time.time() - t_sw) / 60, 1)}
    REPORT['timings_min']['layer_sweep'] = REPORT['layer_sweep']['min']
    log('layer sweep: least damage first', order_m[:12])
    save_report()

# %% [markdown]
# ## Trying some cuts (no retraining yet)
#
# * **narrower feed-forward:** keep the top 85% / 70% / 55% of channels in every layer, chosen by the *agent* scores.
#   Two controls at the middle setting: channels chosen by *Wikipedia* scores, and channels chosen *at random*. If
#   our idea is right, the agent-chosen cut hurts the agent least.
# * **fewer layers:** remove the layers that change their input least on agent text (never the last layer).
#
# Damage is measured as the average "surprise" (cross-entropy, lower is better) on the held-out agent tasks, on the
# tokens the agent wrote itself, and on held-out Wikipedia text.

# %%
variants = [('full', None, set())]
for f in KEEP_FRACS:
    variants.append((f'ff{f}-agent', masks_from(SC['agent']['ff'], f, 'top'), set()))
mid = KEEP_FRACS[len(KEEP_FRACS) // 2]
variants += [(f'ff{mid}-wiki', masks_from(SC['wiki']['ff'], mid, 'top'), set()),
             (f'ff{mid}-random', masks_from(None, mid, 'random'), set())]
order = [i for i in SC['agent']['bi'].argsort().tolist() if i != L - 1]
for k in DROP_LAYERS:
    if k < L - 1:
        variants.append((f'drop{k}-agent', None, set(order[:k])))
        if SWEEP:
            variants.append((f'drop{k}-measured', None, set(order_m[:k])))
# when reusing an earlier run: cuts chosen only from the agent scores have the same agent number as last time
reusable = {n for n, _, _ in variants if n in PREV_AGENT and not n.endswith('-wiki')
            and (not n.endswith('-measured') or 'reused_from' in REPORT.get('layer_sweep', {}))}
t_cut = time.time()
damage = []
for name, ffm, drop in variants:
    MASK['ff'], MASK['drop'] = ffm, drop
    if name == 'full' and FULL_AGENT_NOW is not None:
        agent_nats, how = FULL_AGENT_NOW, 'measured (reuse check)'
    elif name in reusable:
        agent_nats, how = PREV_AGENT[name], f"from {REPORT['reuse']['from_run']}"
    else:
        agent_nats, how = round(surprise(HO, True), 4), 'measured'
    row = {'variant': name, 'agent_own_nats': agent_nats, 'agent_source': how,
           'wiki_nats': round(surprise(WIKI_HO, True), 4), 'dropped_layers': sorted(drop),
           'ff_keep': None if ffm is None else round(float(ffm.mean()), 3)}
    damage.append(row)
    log(row)
    REPORT['damage'] = damage
    save_report()
MASK['ff'], MASK['drop'] = None, set()
base = damage[0]
for r in damage:
    r['agent_damage'] = round(r['agent_own_nats'] - base['agent_own_nats'], 4)
    r['wiki_damage'] = round(r['wiki_nats'] - base['wiki_nats'], 4)
# share of the model's numbers removed by each cut (weights read per generated token, ignoring the vocabulary table)
ff_w = 3 * TCFG.hidden_size * I
lay_w = [sum(p.numel() for p in lyr.parameters()) for lyr in LAYERS]
for r in damage:
    removed = 0
    if r['ff_keep'] is not None:
        removed += L * ff_w * (1 - r['ff_keep'])
    removed += sum(lay_w[i] for i in r['dropped_layers'])
    r['layer_weights_removed_share'] = round(removed / sum(lay_w), 3)
REPORT['damage'] = damage
REPORT['timings_min']['cuts'] = round((time.time() - t_cut) / 60, 1)
for h in HOOKS:
    h.remove()
HOOKS.clear()
save_report()

# %% [markdown]
# ## Pictures and the report
#
# 1. How much each layer changes its input, on agent text and on Wikipedia.
# 2. Per layer, how similar the agent's channel ranking is to Wikipedia's (1 = the same channels matter).
# 3. Damage of each cut: agent (x) vs Wikipedia (y). Good cuts sit low on x and high on y.

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4 if SWEEP else 3, figsize=(21 if SWEEP else 16, 4.2))
for g in GROUPS:
    ax[0].plot(cmp['block_influence'][g], label=g)
ax[0].set(title='How much each layer changes its input', xlabel='layer', ylabel='1 - cosine(in, out)')
ax[0].legend()
ax[1].plot(cmp['per_layer_rank_corr_agent_vs_wiki'], label='agent vs Wikipedia')
ax[1].plot(cmp['per_layer_rank_corr_agent_vs_agent_own'], label="agent vs agent's own tokens")
ax[1].set(title='Do the same channels matter?', xlabel='layer', ylabel='rank correlation')
ax[1].legend()
for r in damage[1:]:
    ax[2].scatter(r['agent_damage'], r['wiki_damage'])
    ax[2].annotate(r['variant'], (r['agent_damage'], r['wiki_damage']), fontsize=8)
ax[2].set(title='Damage of each cut (no retraining)', xlabel='agent: extra nats/token', ylabel='Wikipedia: extra nats/token')
if SWEEP:
    ax[3].bar(range(1, L - 1), SWEEP)
    ax[3].set(title='Removing one layer at a time', xlabel='layer', ylabel='agent: extra nats/token', yscale='symlog')
plt.tight_layout()
plt.savefig(f'{OUT}/scores.png', dpi=120)
REPORT['timings_min']['total'] = round((time.time() - T0) / 60, 1)
save_report()
print(json.dumps({k: REPORT[k] for k in ('data', 'timings_min', 'damage')}, indent=1))
if SMOKE:
    assert damage[0]['agent_own_nats'] > 0 and all(math.isfinite(r['agent_own_nats']) for r in damage)
    print('SMOKE TEST PASSED; set SMOKE = False for the real measurement.')

# %% [markdown]
# ## Save the results (private)
#
# On Kaggle the results are simply the run's output (private). On Colab they are uploaded to a **private** Kaggle
# dataset and copied to Google Drive if it is mounted. Nothing is made public.

# %%
if COLAB:
    slug = f'g4dev-prune-{RUN_NAME}'
    meta = {'title': slug, 'id': f"{os.environ.get('KAGGLE_USERNAME') or 'verracodeguacas'}/{slug}",
            'licenses': [{'name': 'other'}]}
    json.dump(meta, open(f'{OUT}/dataset-metadata.json', 'w'))
    r = subprocess.run(['kaggle', 'datasets', 'create', '-p', OUT, '-r', 'skip'], capture_output=True, text=True)
    if r.returncode != 0 or 'already' in (r.stdout + r.stderr).lower():
        r = subprocess.run(['kaggle', 'datasets', 'version', '-p', OUT, '-m', RUN_NAME, '-r', 'skip'],
                           capture_output=True, text=True)
    print('Kaggle upload:', 'ok' if r.returncode == 0 else 'FAILED', (r.stdout + r.stderr)[-300:])
    if os.path.isdir('/content/drive/MyDrive'):
        shutil.copytree(OUT, f'/content/drive/MyDrive/g4dev-prune/{RUN_NAME}', dirs_exist_ok=True)
        print('copied to Drive: MyDrive/g4dev-prune/' + RUN_NAME)
log('done')
