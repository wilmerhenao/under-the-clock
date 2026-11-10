"""QLoRA SFT of Gemma 4 31B (QAT weights, 4-bit NF4) on rendered agent trajectories.

Input: JSONL from build_sft.py ({text, spans}); only characters inside `spans` (model-generated) are trained.
Loss is computed only at trainable positions, in chunks, against the tied embedding matrix with Gemma's
final-logit soft-cap (30) - full logits for a 28k-token example would need ~29 GB.
Attention (default `--attn chunked`): Gemma 4's full-attention layers have head_dim 512, for which PyTorch has no fused
SDPA backward on L4 (sm89); the math fallback materialises heads x L x L scores (G1, 2026-10-05: every 13-25k-token example
went OOM). The chunked function attends in query chunks (each checkpointed, sliding window honoured), so attention memory
is heads x chunk x L. Verified against eager attention on a tiny Gemma 4 (logits 1e-5, grads 1e-6).
`--probe-lens 8192,16384,...`: before training, one forward+backward per length with peak memory per GPU, plus a check of
which SDPA backends support backward at head_dim 256 and 512.
Output: PEFT adapter whose keys match the competition sample (base_model.model.model.language_model...).
"""
import argparse
import bisect
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--time-cap-min', type=float, default=75)
ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--grad-accum', type=int, default=4)
ap.add_argument('--max-tokens', type=int, default=28672)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--no-4bit', action='store_true', help='CPU smoke tests with a tiny model')
ap.add_argument('--max-examples', type=int, default=0)
ap.add_argument('--attn', default='chunked', help='chunked | sdpa | eager')
ap.add_argument('--chunk', type=int, default=1024)
ap.add_argument('--loss-chunk', type=int, default=1024, help='trainable positions per logits chunk (262k vocab: 1 GB fp32 per 1024)')
ap.add_argument('--probe-headroom-gb', type=float, default=2.5, help='a probed length counts as fitting only with this much GPU memory to spare (G3: 12k peaked at 20.1 of 22 GB, then training OOMed)')
ap.add_argument('--max-mem', default='', help='per-GPU GiB caps for the device map, e.g. 4.4,5.2,5.2,4.2 (weights only)')
ap.add_argument('--device-map', default='auto', help="auto: transformers caps GPUs 0..n-2 at ~model/n and the last GPU "
                'takes the rest, so --max-mem binds only on the last GPU (G3: 20,19,19,13 left placement unchanged); '
                'sequential: fill GPU 0 to its cap, then GPU 1, ... (the caps decide the layer split)')
ap.add_argument('--train-on', default='all', choices=['all', 'actions'], help='actions: loss only on tool-call bodies '
                '(thoughts, channel markers, the opening <|tool_call> and plain text masked): teachers can show what to do '
                'without teaching how long to think (G2b copied narration-length thoughts, B26-B33)')
ap.add_argument('--mask-mistakes', action='store_true', help='with --train-on actions: no loss on a tool call whose next '
                'thought calls it a mistake (B38: a teacher edit with a garbage new_string, followed by "I accidentally corrupted the file")')
ap.add_argument('--skip-kinds', default='', help='comma-separated line kinds to drop from train and validation, e.g. summary '
                '(deterministic template summaries: memorizable, and they made the validation loss look good in G2b)')
ap.add_argument('--probe-lens', default='', help='comma-separated sequence lengths for a memory probe before training; '
                'L:f probes L tokens with a trained fraction f (default 0.25)')
ap.add_argument('--probe-real', action='store_true', help='also probe the longest real line <= each length (G3: real 10-12k '
                'lines OOMed under a synthetic 12k that passed); a length fits only if both fit')
ap.add_argument('--probe-only', action='store_true', help='stop after the probe (no validation, no training, no adapter)')
ap.add_argument('--labels-only', action='store_true', help='tokenizer only: audit which tokens are trainable (labels_audit.json) '
                'and stop before loading the model')
ap.add_argument('--layer-split', default='', help='decoder layers per GPU, e.g. 3,19,19,19: explicit device map with the '
                'vision tower, embeddings and tied lm_head on GPU 0, final norm/rotary/embed_vision on the last GPU '
                '(auto put 0/18/18/24 layers on G3, simulated from the config and matching its measured weights)')
ap.add_argument('--dump-npz', default='', help='tokenizer only: write the training token ids and labels (after '
                '--train-on/--mask-mistakes/--max-tokens) to this .npz and stop before loading the model')
ap.add_argument('--val-data', default='', help='held-out JSONL: mean token loss before and after training')
ap.add_argument('--val-n', type=int, default=24, help='validation examples (seeded sample of --val-data)')
ap.add_argument('--save-every-min', type=float, default=0, help='also save the adapter every N training minutes')
ap.add_argument('--sched', default='steps', help="steps: cosine over the planned steps | time: cosine over --time-cap-min")
ap.add_argument('--replicas', type=int, default=1, help='data-parallel replicas, one process each (launched by the caller '
                'with RANK, WORLD_SIZE, MASTER_ADDR/PORT and its own CUDA_VISIBLE_DEVICES): each replica holds the whole '
                'model on its GPUs and trains its shard; LoRA gradients are averaged (gloo) every optimizer step')
ap.add_argument('--probe-gate', action='store_true',
                help='after the probe, train only on lengths that fit (largest passing probe length); exit 3 if none fits')
a = ap.parse_args()
T0 = time.time()
RANK, WORLD = int(os.environ.get('RANK', '0')), max(1, a.replicas)
if WORLD > 1:
    import datetime
    import torch.distributed as dist
    dist.init_process_group('gloo', rank=RANK, world_size=WORLD, timeout=datetime.timedelta(hours=3))
OUT = Path(a.out) if RANK == 0 else Path(a.out) / f'rank{RANK}'
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'train_log.jsonl', 'w')


def log(**kv):
    kv['t_min'] = round((time.time() - T0) / 60, 2)
    LOG.write(json.dumps(kv) + '\n')
    LOG.flush()
    print(json.dumps(kv), flush=True)


from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained(a.model)
examples = [json.loads(line) for line in open(a.data)]


TOOL_CALL = re.compile(r'<\|tool_call>(.*?<tool_call\|>)(<\|tool_response>)?', re.S)


MISTAKE = re.compile(r"accidentally|corrupt|garbage|\boops\b|my mistake|made a mistake|that was (?:wrong|a mistake)|"
                     r"(?:broke|messed up|mangled) the (?:file|code)|wrong (?:file|place|line)", re.I)
NEXT_THOUGHT = re.compile(r'<\|channel>thought\n(.*?)<channel\|>', re.S)
MASKED = {'calls': 0, 'masked': 0}
MASKED_SAMPLES = []


def regretted(text, end):
    """True if the first thought after position `end` (the call's tool response) calls the call a mistake."""
    m = NEXT_THOUGHT.search(text, end)
    hit = bool(m and MISTAKE.search(m.group(1)[:600]))
    if hit and len(MASKED_SAMPLES) < 40:
        MASKED_SAMPLES.append({'call': text[max(0, end - 300):end], 'thought': m.group(1)[:400]})
    return hit


def action_spans(text, spans):
    """The trainable spans cut down to tool-call bodies: from after `<|tool_call>` through `<tool_call|>` and the
    `<|tool_response>` stop marker that follows. Thoughts, channel markers and plain text stay context only."""
    out = []
    for m in TOOL_CALL.finditer(text):
        lo, hi = m.start(1), m.end()
        if a.mask_mistakes and any(s0 <= lo < e0 for s0, e0 in spans):
            MASKED['calls'] += 1
            if regretted(text, hi):
                MASKED['masked'] += 1
                continue
        for s0, e0 in spans:
            a_, b_ = max(lo, s0), min(hi, e0)
            if a_ < b_:
                out.append((a_, b_))
    return sorted(out)


def encode(ex, train_on=None, offsets=False):
    enc = tok(ex['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    # an explicit argument wins, then the row's own 'train_on' (G17 teacher pairs train the whole final step, plan
    # included, B157), then --train-on
    spans = action_spans(ex['text'], ex['spans']) if (train_on or ex.get('train_on') or a.train_on) == 'actions' else ex['spans']
    starts = [s for s, _ in spans]
    labels = []
    for (c0, _), t in zip(enc['offset_mapping'], ids):
        k = bisect.bisect_right(starts, c0) - 1
        labels.append(t if k >= 0 and spans[k][0] <= c0 < spans[k][1] else -100)
    if offsets:
        return torch.tensor(ids), torch.tensor(labels), enc['offset_mapping']
    return torch.tensor(ids), torch.tensor(labels)


SKIP_KINDS = {k for k in a.skip_kinds.split(',') if k}
n_all = len(examples)
examples = [ex for ex in examples if ex.get('kind', 'segment') not in SKIP_KINDS]
log(event='kinds', skipped=n_all - len(examples), skip_kinds=sorted(SKIP_KINDS), train_on=a.train_on)


data, n_empty = [], 0
for ex in examples[:a.max_examples or None]:
    ids, labels = encode(ex)
    if len(ids) > a.max_tokens:
        log(event='skip_long', instance_id=ex['instance_id'], tokens=len(ids))
        continue
    if not int((labels[1:] != -100).sum()):  # nothing to learn (e.g. no tool call under --train-on actions)
        n_empty += 1
        continue
    data.append((ex['instance_id'], ex['source'], ids, labels))
log(event='mistakes', **MASKED, mask_mistakes=a.mask_mistakes)
log(event='data', examples=len(data), tokens=sum(len(d[2]) for d in data),
    trainable=sum(int((d[3] != -100).sum()) for d in data), skipped_no_trainable=n_empty)
val = []
if a.val_data:
    vex = [ex for ex in map(json.loads, open(a.val_data)) if ex.get('kind', 'segment') not in SKIP_KINDS]
    random.Random(a.seed).shuffle(vex)
    for ex in vex:
        ids, labels = encode(ex)
        if len(ids) <= a.max_tokens and len(val) < a.val_n and int((labels[1:] != -100).sum()):
            val.append((ex['instance_id'], ex['source'], ids, labels))
    log(event='val_data', examples=len(val), tokens=sum(len(d[2]) for d in val))

THOUGHT = re.compile(r'<\|channel>thought\n.*?<channel\|>', re.S)


def labels_audit(n_show=6):
    """Per line: tokens, trainable tokens under 'all' and 'actions', tool calls; checks for the current mode that every
    trainable run of an 'actions' line is one tool call (starts with 'call:', ends at '<|tool_response>' or
    '<tool_call|>') and that no trainable token starts inside a thought."""
    rows, shown, bad = [], [], {'run_start': 0, 'run_end': 0, 'in_thought': 0, 'runs_ne_calls': 0}
    for ex in examples[:a.max_examples or None]:
        ids, labels, offs = encode(ex, offsets=True)
        if len(ids) > a.max_tokens:
            continue
        _, lab_all = encode(ex, train_on='all')
        _, lab_act = encode(ex, train_on='actions')
        n_calls = len(TOOL_CALL.findall(ex['text']))
        model_calls = sum(1 for m in TOOL_CALL.finditer(ex['text'])
                          if any(s0 <= m.start(1) < e0 for s0, e0 in ex['spans']))
        thoughts = [(m.start(), m.end()) for m in THOUGHT.finditer(ex['text'])]
        tr = (labels != -100).tolist()
        runs, i = [], 0
        while i < len(tr):
            if tr[i]:
                j = i
                while j + 1 < len(tr) and tr[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        for i, j in runs:
            txt = tok.decode(ids[i:j + 1].tolist())
            if a.train_on == 'actions':
                bad['run_start'] += not txt.startswith('call:')
                bad['run_end'] += not (txt.endswith('<|tool_response>') or txt.endswith('<tool_call|>'))
            if len(shown) < n_show and (len(rows) % 97 == 0 or j - i > 400):
                c0 = offs[i][0]
                shown.append({'instance_id': ex['instance_id'], 'kind': ex.get('kind', 'segment'), 'tokens': j - i + 1,
                              'before': ex['text'][max(0, c0 - 300):c0], 'trained': txt[:1200]})
        bad['in_thought'] += sum(1 for k, t in enumerate(tr) if t and any(s0 <= offs[k][0] < e0 for s0, e0 in thoughts))
        if a.train_on == 'actions':
            bad['runs_ne_calls'] += len(runs) != len(action_spans(ex['text'], ex['spans']))  # masked mistakes excluded
        rows.append({'instance_id': ex['instance_id'], 'source': ex['source'], 'kind': ex.get('kind', 'segment'),
                     'tokens': len(ids), 'trainable': int(sum(tr)), 'trainable_all': int((lab_all != -100).sum()),
                     'trainable_actions': int((lab_act != -100).sum()), 'tool_calls': n_calls, 'model_calls': model_calls,
                     'runs': len(runs)})

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    summ = {'lines': len(rows), 'train_on': a.train_on, 'skip_kinds': sorted(SKIP_KINDS), 'max_tokens': a.max_tokens,
            'tokens': sum(r['tokens'] for r in rows), 'trainable': sum(r['trainable'] for r in rows),
            'trainable_all': sum(r['trainable_all'] for r in rows),
            'trainable_actions': sum(r['trainable_actions'] for r in rows),
            'lines_no_trainable': sum(1 for r in rows if not r['trainable']),
            'trainable_per_line_p10_p50_p90': [q([r['trainable'] for r in rows], p) for p in (0.1, 0.5, 0.9)],
            'calls_per_line_p50': q([r['model_calls'] for r in rows], 0.5), 'violations': bad,
            'by_kind': {k: sum(1 for r in rows if r['kind'] == k) for k in sorted({r['kind'] for r in rows})}}
    summ['mistakes'] = dict(MASKED, mask_mistakes=a.mask_mistakes)
    (OUT / 'labels_audit.json').write_text(json.dumps({'summary': summ, 'examples': shown, 'masked_samples': MASKED_SAMPLES, 'rows': rows}, indent=1))
    log(event='labels_audit', **summ)


if a.labels_only:
    labels_audit()
    raise SystemExit(0)

if a.dump_npz:  # the TPU trainer (train_tpu.py) reads these exact token ids and labels
    import numpy as np
    np.savez_compressed(a.dump_npz, ids=np.concatenate([d[2].numpy() for d in data]).astype(np.int32),
                        labels=np.concatenate([d[3].numpy() for d in data]).astype(np.int32),
                        lens=np.array([len(d[2]) for d in data], np.int32),
                        instance_ids=np.array([d[0] for d in data]), sources=np.array([d[1] for d in data]))
    log(event='dump_npz', path=a.dump_npz, examples=len(data))
    raise SystemExit(0)

from peft import LoraConfig, get_peft_model
from transformers import (AttentionInterface, AttentionMaskInterface, AutoConfig, AutoModelForImageTextToText,
                          BitsAndBytesConfig)


def _attn_block(q, k, v, q0, k0, window, scaling):
    qi = torch.arange(q0, q0 + q.shape[2], device=q.device)[:, None]
    kj = torch.arange(k0, k0 + k.shape[2], device=q.device)[None, :]
    allowed = kj <= qi
    if window:
        allowed = allowed & (qi - kj < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scaling)


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Causal (sliding-window on sliding layers) attention in query chunks; one unpadded sequence; mask argument unused."""
    L = query.shape[2]
    rep = query.shape[1] // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
    window = getattr(module, 'sliding_window', None)
    outs = []
    for s in range(0, L, a.chunk):
        e = min(L, s + a.chunk)
        k0 = max(0, s - window + 1) if window else 0
        args = (query[:, :, s:e], key[:, :, k0:e], value[:, :, k0:e], s, k0, window, scaling)
        outs.append(checkpoint(_attn_block, *args, use_reentrant=False) if torch.is_grad_enabled() else _attn_block(*args))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


AttentionInterface.register('chunked', chunked_attention)
AttentionMaskInterface.register('chunked', lambda *args_, **kw_: None)  # no L x L mask: the function builds its own per chunk

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         llm_int8_skip_modules=['vision_tower', 'embed_vision', 'lm_head'])
n_gpu = torch.cuda.device_count()
if a.device_map == 'sequential' and not a.max_mem:  # the default 19 GiB caps would put the whole model on GPU 0
    raise SystemExit('--device-map sequential needs --max-mem')
device_map = a.device_map
if a.layer_split:
    counts = [int(x) for x in a.layer_split.split(',')]
    n_layers = AutoConfig.from_pretrained(a.model).get_text_config().num_hidden_layers
    if sum(counts) != n_layers or len(counts) > max(1, n_gpu):
        raise SystemExit(f'--layer-split {counts} does not cover {n_layers} layers on {n_gpu} GPUs')
    device_map = {'model.vision_tower': 0, 'model.language_model.embed_tokens': 0, 'lm_head': 0}
    device_map.update({f'model.language_model.layers.{i}': g for g, (lo, c) in
                       enumerate(zip([sum(counts[:j]) for j in range(len(counts))], counts)) for i in range(lo, lo + c)})
    last = len(counts) - 1
    device_map.update({'model.language_model.norm': last, 'model.language_model.rotary_emb': last, 'model.embed_vision': last})
if a.no_4bit:
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32)
else:
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, quantization_config=bnb, device_map=device_map, dtype=torch.bfloat16, attn_implementation=a.attn,
        max_memory=({i: f'{g}GiB' for i, g in enumerate(a.max_mem.split(','))} if a.max_mem else {i: '19GiB' for i in range(n_gpu)}))
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
log(event='model_loaded', gpus=n_gpu, load_min=round((time.time() - T0) / 60, 1), mem_gb=[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpu)])
_dm = getattr(model, 'hf_device_map', None) or {}
_layer = lambda k: k.rsplit('.', 1)[-1].isdigit() and '.layers.' in k  # noqa: E731
log(event='device_map', layers_per_device={str(d): sum(1 for k, v in _dm.items() if _layer(k) and v == d) for d in sorted(set(_dm.values()), key=str)},
    layer_ranges={str(d): [min(i), max(i)] for d in set(_dm.values()) if (i := [int(k.rsplit('.', 1)[-1]) for k, v in _dm.items() if _layer(k) and v == d])},
    other={k: str(v) for k, v in _dm.items() if not _layer(k)})

lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                  target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)')
model = get_peft_model(model, lcfg)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_params(tensors):
    return torch.cat([t.detach().float().reshape(-1).cpu() for t in tensors])


def unflat_into(flat, tensors):
    i = 0
    for t in tensors:
        n = t.numel()
        t.copy_(flat[i:i + n].view_as(t).to(device=t.device, dtype=t.dtype))
        i += n


if WORLD > 1:  # LoRA A is randomly initialised: every replica starts from rank 0's weights
    _ps = [p for p in model.parameters() if p.requires_grad]
    _flat = flat_params(_ps)
    dist.broadcast(_flat, 0)
    with torch.no_grad():
        unflat_into(_flat, _ps)
    log(event='replicas', rank=RANK, world=WORLD, lora_params=n_train)
log(event='lora', trainable_params=n_train)

inner = model.base_model.model  # Gemma4ForConditionalGeneration
W = inner.lm_head.weight  # tied with the input embeddings, frozen
softcap = getattr(inner.config.get_text_config(), 'final_logit_softcapping', None) or 0.0


def chunk_loss(h, t):
    logits = (h.to(W.device) @ W.T).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, t.to(W.device), reduction='sum')


def example_loss(ids, labels):
    dev0 = next(inner.model.language_model.parameters()).device
    out = inner.model(input_ids=ids[None].to(dev0), attention_mask=torch.ones_like(ids)[None].to(dev0),
                      use_cache=False)
    h = out.last_hidden_state[0]
    tgt = labels[1:]
    pos = (tgt != -100).nonzero().squeeze(-1)
    hs, ts = h[:-1][pos.to(h.device)], tgt[pos]
    c = a.loss_chunk
    total = sum(checkpoint(chunk_loss, hs[i:i + c], ts[i:i + c], use_reentrant=False) for i in range(0, len(pos), c))
    return total / max(1, len(pos)), len(pos)


def sdpa_backends():
    """Which fused SDPA kernels run forward+backward at Gemma 4's head sizes on this GPU."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    out = {}
    for d in (256, 512):
        for name, be in (('flash', SDPBackend.FLASH_ATTENTION), ('efficient', SDPBackend.EFFICIENT_ATTENTION),
                         ('math', SDPBackend.MATH)):
            q, k, v = (torch.randn(1, 32, 2048, d, device='cuda:0', dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
                out[f'{name}_d{d}'] = 'ok'
            except Exception as e:  # noqa: BLE001
                out[f'{name}_d{d}'] = f'{type(e).__name__}: {str(e)[:120]}'
            del q, k, v
    torch.cuda.empty_cache()
    return out


probe_ok = {}
model.train()  # HF runs gradient checkpointing only in training mode (G2 probed in eval mode: OOM at 8k, 2026-10-06)
if a.probe_lens and torch.cuda.is_available():
    log(event='sdpa_backends', **sdpa_backends())
    base_ids = max(data, key=lambda d: len(d[2]))[2]

    def probe_one(ids, labels, **tags):
        for i in range(n_gpu):
            torch.cuda.reset_peak_memory_stats(i)
        t_ = time.time()
        fwd_gb = None
        try:
            loss, n_pos = example_loss(ids, labels)
            fwd_gb = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]
            loss.backward()
            torch.cuda.synchronize()
            res = dict(ok=True, loss=round(loss.item(), 4), seconds=round(time.time() - t_, 1))
        except torch.cuda.OutOfMemoryError as e:
            res = dict(ok=False, error=str(e)[:200])
        loss = n_pos = None  # a failed backward leaves its graph on `loss` (G3: it stayed alive through training)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]
        cap = torch.cuda.get_device_properties(0).total_memory / 2**30
        if res['ok'] and max(peak) > cap - a.probe_headroom_gb:
            res.update(ok=False, error=f'fits, but peak {max(peak)} GB leaves less than {a.probe_headroom_gb} GB of {cap:.1f}')
        log(event='probe', tokens=len(ids), trainable=int((labels[1:] != -100).sum()), attn=a.attn, training=model.training,
            peak_gb=peak, fwd_peak_gb=fwd_gb, after_gb=[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpu)],
            **tags, **res)
        return res['ok']

    for spec in [x for x in a.probe_lens.split(',') if x]:
        L, _, f = spec.partition(':')
        L, f = int(L), float(f or 0.25)
        ids = base_ids.repeat(L // len(base_ids) + 1)[:L]
        labels = ids.clone()
        labels[:int(L * (1 - f))] = -100
        ok = probe_one(ids, labels, kind='synthetic', trained_fraction=f)
        if f == 0.25:
            probe_ok[L] = ok and probe_ok.get(L, True)
        if a.probe_real and f == 0.25:
            real = max((d for d in data if len(d[2]) <= L), key=lambda d: len(d[2]), default=None)
            if real is not None:
                probe_ok[L] = probe_one(real[2], real[3], kind='real', limit=L, instance_id=real[0], source=real[1]) and probe_ok[L]
    if a.probe_only:
        log(event='probe_only', fits={str(k): v for k, v in probe_ok.items()})
        raise SystemExit(0)
    if a.probe_gate:
        fits = [L for L, ok in probe_ok.items() if ok]
        fit_max = max(fits) if fits else 0
        if WORLD > 1:  # every replica trains the same length cap: the smallest that passed everywhere
            _t = torch.tensor([fit_max], dtype=torch.int64)
            dist.all_reduce(_t, op=dist.ReduceOp.MIN)
            fit_max = int(_t.item())
        if not fit_max:
            log(event='probe_gate', verdict='nothing fits: stopping before training')
            raise SystemExit(3)
        # always cap at the longest length that passed: an unprobed length is untested (shake-g5: every probe passed,
        # nothing was capped, the first 14.7k line OOMed and the run never trained, B43)
        a.max_tokens = min(a.max_tokens, fit_max)
        n0, v0 = len(data), len(val)
        data = [d for d in data if len(d[2]) <= a.max_tokens]
        val = [d for d in val if len(d[2]) <= a.max_tokens]
        log(event='probe_gate', verdict=f'training only up to {a.max_tokens} tokens', all_fit=all(probe_ok.values()),
            dropped=n0 - len(data), val_dropped=v0 - len(val))

def val_loss(when):
    """Token-weighted mean loss on the validation sample (no dropout, no grad); the zero-init adapter = base model."""
    if not val or RANK != 0:
        return
    model.eval()
    t_, tot, n_tok, per_ex, skipped = time.time(), 0.0, 0, [], []
    with torch.no_grad():
        for iid, src, ids, labels in val:
            try:
                loss, n_pos = example_loss(ids, labels)
            except torch.cuda.OutOfMemoryError:
                skipped.append(iid)  # start and end compare only if both skip the same examples
                gc.collect()
                torch.cuda.empty_cache()
                continue
            tot += loss.item() * n_pos
            n_tok += n_pos
            per_ex.append(round(loss.item(), 4))
    model.train()
    log(event='val', when=when, loss_tok=round(tot / max(1, n_tok), 4), loss_ex=round(sum(per_ex) / max(1, len(per_ex)), 4),
        examples=len(per_ex), skipped=skipped, trainable_tokens=n_tok, seconds=round(time.time() - t_, 1), per_example=per_ex)


def save_adapter(event, **kv):
    if RANK != 0:
        return
    # Same metadata shape as the competition's sample adapter
    model.peft_config['default'].base_model_name_or_path = 'google/gemma-4-31b-it-qat-w4a16-ct'
    adapter_dir = OUT / 'adapter'
    model.save_pretrained(adapter_dir)
    # vLLM's LoRA loader expects a plain module-name list, not PEFT's regex form
    cfg_path = adapter_dir / 'adapter_config.json'
    acfg = json.loads(cfg_path.read_text())
    acfg['target_modules'] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    cfg_path.write_text(json.dumps(acfg, indent=2))
    log(event=event, dir=str(adapter_dir), files={p.name: p.stat().st_size for p in adapter_dir.iterdir()}, **kv)


val_loss('start')
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
steps_total = math.ceil(len(data) * a.epochs / (a.grad_accum * WORLD))
t_train = time.time()


def lr_factor(s):
    warm = min(1.0, (s + 1) / 2)
    done = (time.time() - t_train) / 60 / a.time_cap_min if a.sched == 'time' else s / max(1, steps_total)
    return warm * 0.5 * (1 + math.cos(math.pi * min(1.0, done)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
random.seed(a.seed)
model.train()
step, seen, tokens_done, stop = 0, 0, 0, False
t_saved = time.time()
nan_run = 0
oom_run = 0
base_mem = [torch.cuda.memory_allocated(i) for i in range(n_gpu)]  # leak check after an OOM (shake-g5: 14 GB stayed on GPU 3)
for epoch in range(a.epochs if WORLD == 1 else 0):  # WORLD > 1: replica loop below
    order = list(range(len(data)))
    random.shuffle(order)
    for k in order:
        if (time.time() - t_train) / 60 > a.time_cap_min:  # training minutes; model loading is not counted
            stop = True
            break
        iid, src, ids, labels = data[k]
        oom = False
        try:
            loss, n_pos = example_loss(ids, labels)
            (loss / a.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:  # outside the except block: the traceback and `loss` held the failed graph (G3: 674 skips in a row)
            loss = n_pos = None
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            mem = [torch.cuda.memory_allocated(i) for i in range(n_gpu)]
            oom_run += 1
            log(event='oom_skip', instance_id=iid, tokens=len(ids), run=oom_run, mem_gb=[round(m / 2**30, 1) for m in mem])
            leaked = max(m - b for m, b in zip(mem, base_mem)) > 2 * 2**30
            if oom_run >= 3 or leaked:  # every later example fails the same way: stop and keep what was trained
                log(event='abort', reason='leaked memory after an OOM' if leaked else '3 OOMs in a row',
                    leaked_gb=round(max(m - b for m, b in zip(mem, base_mem)) / 2**30, 1))
                stop = True
                break
            continue
        oom_run = 0
        if not math.isfinite(loss.item()):
            nan_run += 1
            opt.zero_grad(set_to_none=True)
            log(event='nan_loss', instance_id=iid, run=nan_run)
            if nan_run >= 3:
                log(event='abort', reason='3 non-finite losses in a row')
                stop = True
                break
            continue
        nan_run = 0
        seen += 1
        tokens_done += len(ids)
        log(event='example', epoch=epoch, instance_id=iid, source=src, tokens=len(ids), trainable=n_pos,
            loss=round(loss.item(), 4), tok_per_s=round(tokens_done / (time.time() - t_train), 1))
        if seen % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log(event='step', step=step, lr=sched.get_last_lr()[0])
            if a.save_every_min and (time.time() - t_saved) / 60 >= a.save_every_min:
                save_adapter('checkpoint', steps=step, examples_seen=seen)
                t_saved = time.time()
    if stop:
        break
if WORLD == 1 and seen % a.grad_accum:
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1

if WORLD > 1:
    # Rounds: each replica runs up to grad_accum examples of its shard, then all replicas average their LoRA gradients
    # and a stop flag (time cap, end of shard, abort) in one all-reduce, so they always step together and stop together.
    shard = []
    for epoch in range(a.epochs):
        order = list(range(len(data)))
        random.shuffle(order)  # same seed everywhere: the shards partition each epoch's order
        shard += order[RANK::WORLD]
    it = iter(shard)
    while True:
        want_stop, got = (time.time() - t_train) / 60 > a.time_cap_min, 0
        while not want_stop and got < a.grad_accum:
            k = next(it, None)
            if k is None:
                want_stop = True
                break
            iid, src, ids, labels = data[k]
            oom = False
            try:
                loss, n_pos = example_loss(ids, labels)
                (loss / a.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                loss = n_pos = None
                gc.collect()
                torch.cuda.empty_cache()
                oom_run += 1
                log(event='oom_skip', instance_id=iid, tokens=len(ids), run=oom_run)
                if oom_run >= 3:
                    log(event='abort', reason='3 OOMs in a row')
                    want_stop = True
                continue  # the partial graph is freed; gradients of earlier examples in this round stay
            oom_run = 0
            if not math.isfinite(loss.item()):
                nan_run += 1
                log(event='nan_loss', instance_id=iid, run=nan_run)
                if nan_run >= 3:
                    log(event='abort', reason='3 non-finite losses in a row')
                    want_stop = True
                continue
            nan_run = 0
            seen += 1
            got += 1
            tokens_done += len(ids)
            log(event='example', instance_id=iid, source=src, tokens=len(ids), trainable=n_pos, rank=RANK,
                loss=round(loss.item(), 4), tok_per_s=round(tokens_done / (time.time() - t_train), 1))
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch.cat([flat_params(grads), torch.tensor([float(got), float(want_stop)])])
        dist.all_reduce(flat)
        n_all, stop_all = flat[-2].item(), flat[-1].item() > 0
        if n_all > 0:
            with torch.no_grad():  # each replica scaled by 1/grad_accum; average over the replicas that contributed
                for p_, g_ in zip(params, grads):
                    if p_.grad is None:
                        p_.grad = g_
                unflat_into(flat[:-2] * (a.grad_accum / n_all), [p_.grad for p_ in params])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            log(event='step', step=step, lr=sched.get_last_lr()[0], examples_all=int(n_all))
            if a.save_every_min and (time.time() - t_saved) / 60 >= a.save_every_min:
                save_adapter('checkpoint', steps=step, examples_seen=seen)
                t_saved = time.time()
        opt.zero_grad(set_to_none=True)
        if stop_all:
            stop = True
            break

if WORLD > 1:  # identical updates keep the replicas identical: the checksums must match across ranks
    log(event='replica_sync', rank=RANK, steps=step, checksum=round(float(flat_params(params).double().abs().sum()), 6))
save_adapter('saved', steps=step, examples_seen=seen, stopped_by_time=stop)
val_loss('end')
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
