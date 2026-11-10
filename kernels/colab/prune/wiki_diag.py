# Diagnostic (B170): why does the full model score 5.9 (no <bos>) / 7.6 (with <bos>) nats per token on WikiText-103
# when a healthy 31B model should be near 2, while it scores 0.26 on the agent's own tokens? Appended by
# build_wiki_diag.py to prune_score.py's setup + model-load cells (same machine, same model, same stack).
import hashlib
import io
import pandas as pd
import torch.nn.functional as F

DIAG = {}


def save_diag():
    json.dump(DIAG, open(f'{OUT}/wiki_diag.json', 'w'), indent=1)


def wiki_text():
    parts = []
    for split in ('validation', 'test'):
        wd = os.environ.get('PRUNE_WIKI_DIR')     # CPU test only
        data = open(f'{wd}/wikitext-103-raw-v1-{split}.parquet' if wd else find(f'wikitext-103-raw-v1-{split}.parquet'), 'rb').read()
        parts += [x for x in pd.read_parquet(io.BytesIO(data))['text'] if x.strip()]
    return '\n'.join(parts)


@torch.no_grad()
def token_losses(ids):
    """Cross-entropy of every next token (position t predicts t+1), top-1 hit, and the logits' top-5 at a few spots."""
    ids = torch.as_tensor(ids)
    h = TM(input_ids=ids[None].to(DEV0), use_cache=False).last_hidden_state[0]
    ce, hit = [], []
    for ch in torch.arange(len(ids) - 1).split(512):
        lg = model.lm_head(h[ch.to(h.device)].to(HEAD_DEV)).float()
        if CAP:
            lg = torch.tanh(lg / CAP) * CAP
        tgt = ids[ch + 1].to(lg.device)
        ce.append(F.cross_entropy(lg, tgt, reduction='none').cpu())
        hit.append((lg.argmax(-1) == tgt).cpu())
    return torch.cat(ce), torch.cat(hit)


BUCKETS = [(0, 16), (16, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]


def summary(res, masks=None):
    """res: [(ce, hit)] per sequence; buckets are positions within each sequence; masks: optional [bool] per sequence."""
    masks = masks or [torch.ones(len(c), dtype=torch.bool) for c, _ in res]
    ce = torch.cat([c[m] for (c, _), m in zip(res, masks)])
    out = {'nats': round(float(ce.mean()), 4), 'tokens': len(ce)}
    for a, b in BUCKETS:
        cs, hs = [], []
        for (c, h), m in zip(res, masks):
            sel = torch.zeros(len(c), dtype=torch.bool)
            sel[a:b] = True
            sel &= m
            cs.append(c[sel])
            hs.append(h[sel])
        cs, hs = torch.cat(cs), torch.cat(hs)
        if len(cs):
            out[f'{a}-{b}'] = {'nats': round(float(cs.mean()), 3), 'top1': round(float(hs.float().mean()), 3), 'n': len(cs)}
    return out


def top5(ids, positions):
    ids = torch.as_tensor(ids)
    with torch.no_grad():
        h = TM(input_ids=ids[None].to(DEV0), use_cache=False).last_hidden_state[0]
        res = []
        for p in positions:
            lg = model.lm_head(h[p].to(HEAD_DEV)).float()
            lg = torch.tanh(lg / CAP) * CAP if CAP else lg
            lp = torch.log_softmax(lg, -1)
            v, i = lp.topk(5)
            res.append({'pos': p, 'context_end': tok.decode(ids[max(0, p - 12):p + 1].tolist()),
                        'actual_next': tok.decode(ids[p + 1:p + 2].tolist()),
                        'top5': [(tok.decode([int(a)]), round(float(b), 2)) for a, b in zip(i, v)]})
    return res


t_d = time.time()
text = wiki_text()
wids = tok(text, add_special_tokens=False)['input_ids']
bos = tok.bos_token_id
start = 147 * 2047                       # where score-v2's held-out Wikipedia chunks began
chunks = [[bos] + wids[start + k * 2047:start + (k + 1) * 2047] for k in range(4)]
DIAG['bos_id'] = bos
DIAG['chunk0_text_start'] = tok.decode(chunks[0][:80])

# 1. as in score-v2: <bos> + raw WikiText, sdpa attention
r = [token_losses(c) for c in chunks]
DIAG['wiki_bos_sdpa'] = summary(r)
log('wiki <bos> sdpa', DIAG['wiki_bos_sdpa'])
DIAG['top5_wiki'] = top5(chunks[0], [5, 50, 300, 900, 1500, 2000])
save_diag()

# 2. no <bos> (score-v1)
r = [token_losses(c[1:]) for c in chunks]
DIAG['wiki_nobos_sdpa'] = summary(r)
log('wiki no <bos>', DIAG['wiki_nobos_sdpa'])
save_diag()

# 3. short pieces: only the first 512 tokens of each chunk (does length matter?)
r = [token_losses(c[:512]) for c in chunks]
DIAG['wiki_bos_512'] = summary(r)
log('wiki <bos> 512', DIAG['wiki_bos_512'])
save_diag()

# 4. the same text with WikiText's odd spacing undone (" , " -> ", ", " @-@ " -> "-", ...)
def detok(s):
    for a, b in ((' @-@ ', '-'), (' @,@ ', ','), (' @.@ ', '.'), (' , ', ', '), (' . ', '. '), (' ; ', '; '), (' : ', ': '),
                 (" 's", "'s"), (' ( ', ' ('), (' ) ', ') '), (' \n', '\n')):
        s = s.replace(a, b)
    return s


piece = tok.decode(wids[start:start + 4 * 2047])
dids = tok(detok(piece), add_special_tokens=False)['input_ids']
dchunks = [[bos] + dids[k * 2047:(k + 1) * 2047] for k in range(len(dids) // 2047)][:4]
r = [token_losses(c) for c in dchunks]
DIAG['wiki_detok_bos'] = summary(r)
log('wiki detokenized', DIAG['wiki_detok_bos'])
save_diag()

# 5. the same text inside a chat turn (the model's own format): user asks for an article, the model writes it
for name, c in (('chat', chunks[0]),):
    body = tok.decode(c[1:1500])
    msgs = [{'role': 'user', 'content': 'Write an encyclopedia article.'}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pids = tok(prompt, add_special_tokens=False)['input_ids']
    bids = tok(body, add_special_tokens=False)['input_ids']
    ce, hit = token_losses(pids + bids)
    mask = torch.zeros(len(ce), dtype=torch.bool)
    mask[len(pids) - 1:] = True                  # positions that predict the article's tokens
    DIAG['wiki_in_chat'] = summary([(ce, hit)], [mask])
    DIAG['chat_prompt'] = prompt
    log('wiki in chat', DIAG['wiki_in_chat'])
save_diag()

# 6. attention implementation: eager instead of sdpa on the same <bos> chunks
try:
    model.set_attn_implementation('eager')
    r = [token_losses(c) for c in chunks[:2]]
    DIAG['wiki_bos_eager'] = summary(r)
    log('wiki <bos> eager', DIAG['wiki_bos_eager'])
    model.set_attn_implementation('sdpa')
except Exception as e:
    DIAG['wiki_bos_eager'] = f'failed: {str(e)[:300]}'
save_diag()

# 7. agent rows (held-out tasks): every token, the agent's own tokens and the rest, by position
rows = [json.loads(x) for x in open(sft_path) if x.strip()]
held = [r_ for r_ in rows if int(hashlib.md5(r_['instance_id'].encode()).hexdigest(), 16) % 10 == 0][:3]
ag = []
for r_ in held:
    enc = tok(r_['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids'][:8192]
    st = np.array([o[0] for o in enc['offset_mapping'][:8192]])
    own = np.zeros(len(ids), bool)
    for a, b in r_.get('spans') or []:
        own |= (st >= a) & (st < b)
    ce, hit = token_losses(ids)
    ag.append((ce, hit, torch.from_numpy(own[1:])))
res = [(x[0], x[1]) for x in ag]
DIAG['agent_all'] = summary(res)
DIAG['agent_own'] = summary(res, [x[2] for x in ag])
DIAG['agent_not_own'] = summary(res, [~x[2] for x in ag])
DIAG['agent_first_ids'] = [tok.convert_ids_to_tokens(int(i)) for i in tok(held[0]['text'], add_special_tokens=False)['input_ids'][:8]]
log('agent all / own / not own', DIAG['agent_all']['nats'], DIAG['agent_own']['nats'], DIAG['agent_not_own']['nats'])
DIAG['minutes'] = round((time.time() - t_d) / 60, 1)
save_diag()
print(json.dumps(DIAG, indent=1)[:12000])
log('done')
