"""Is mix8 what it should be? (CPU, no internet; reads the finished mix8 and mix7 outputs)

mix8 = mix7 with the compaction summaries written by Gemma 4 31B (B39, B41), in 4 chains. Checks:
- every post-compaction segment's `For context` message carries a Gemma summary, never our template
  (template markers: "My notes so far, oldest first:" and "Files mentioned so far:");
- context-summary length per chain vs mix7's template summaries;
- chains per compacted conversation (train/val_chains.jsonl), and that chain copies differ from chain 0 only in context;
- alignment with mix7: same conversations (ref), line and token counts, trainable tokens.
Outputs: /kaggle/working/audit/audit.json.
"""
import collections
import glob
import json
import re
from pathlib import Path

OUT = Path('/kaggle/working/audit')
OUT.mkdir(parents=True, exist_ok=True)
MIX8 = '/kaggle/input/**/g4dev-train-mix8-cpu/mix'  # @define
MIX7 = '/kaggle/input/**/g4dev-train-mix7-cpu/mix'  # @define
CTX = re.compile(r'For context:.*?\[model\] said: (.*?)<turn\|>', re.S)
TEMPLATE = ('My notes so far, oldest first:', 'Files mentioned so far:')


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def lines(path):
    return [json.loads(x) for x in open(path)] if path and path.exists() else []


def pct(v):
    v = sorted(v)
    return {f'p{q}': v[min(len(v) - 1, int(len(v) * q / 100))] for q in (10, 50, 90)} if v else {}


def contexts(rows):
    out = []
    for r in rows:
        if r.get('kind') != 'segment':
            continue
        m = CTX.search(r['text'][:60000])
        if m:
            out.append(m.group(1))
    return out


d8, d7 = find(MIX8), find(MIX7)
rep = {'mix8_dir': str(d8), 'mix7_dir': str(d7)}
for split in ('train', 'val'):
    m8 = lines(d8 / f'{split}.jsonl')
    ch = lines(d8 / f'{split}_chains.jsonl')
    m7 = lines(d7 / f'{split}.jsonl') if d7 else []
    c8, c7 = contexts(m8), contexts(m7)
    by_chain = collections.defaultdict(list)
    for r in ch:
        by_chain[r['chain']].append(r)
    chain_ctx = {k: contexts(v) for k, v in by_chain.items()}
    conv_chains = collections.Counter()
    refs_c0 = {r['ref'] for r in m8 if r.get('kind') == 'segment' and CTX.search(r['text'][:60000])}
    for ref in refs_c0:
        conv_chains[1 + sum(1 for k, v in by_chain.items() if any(x['ref'] == ref for x in v))] += 1
    # chain copies: same trained text as chain 0 where the line exists in both (targets identical, context differs)
    c0_by = {(r['ref'], r['part']): r for r in m8}
    same_target = diff_target = 0
    for r in ch:
        r0 = c0_by.get((r['ref'], r['part']))
        if r0 is None or r.get('kind') != 'segment':
            continue
        t0 = ''.join(r0['text'][a:b] for a, b in r0['spans'])
        t1 = ''.join(r['text'][a:b] for a, b in r['spans'])
        same_target += t0 == t1
        diff_target += t0 != t1
    rep[split] = {
        'mix8': {'lines': len(m8), 'kinds': dict(collections.Counter(r.get('kind') for r in m8)),
                 'refs': len({r['ref'] for r in m8}), 'tokens': sum(r['n_tokens'] for r in m8),
                 'trainable': sum(r['n_trainable'] for r in m8)},
        'mix7': {'lines': len(m7), 'kinds': dict(collections.Counter(r.get('kind') for r in m7)),
                 'refs': len({r['ref'] for r in m7}), 'tokens': sum(r['n_tokens'] for r in m7),
                 'trainable': sum(r['n_trainable'] for r in m7)},
        'refs_shared': len({r['ref'] for r in m8} & {r['ref'] for r in m7}),
        'post_compaction_segments': {'mix8': len(c8), 'mix7': len(c7)},
        'context_template_like': {'mix8': sum(any(t in c for t in TEMPLATE) for c in c8),
                                  'mix7': sum(any(t in c for t in TEMPLATE) for c in c7),
                                  'chains': {k: sum(any(t in c for t in TEMPLATE) for c in v) for k, v in chain_ctx.items()}},
        'context_chars': {'mix8_chain0': pct([len(c) for c in c8]), 'mix7_template': pct([len(c) for c in c7]),
                          **{f'chain{k}': pct([len(c) for c in v]) for k, v in sorted(chain_ctx.items())}},
        'context_openings': collections.Counter(c[:20] for c in c8).most_common(6),
        'chains_per_compacted_conversation': dict(conv_chains),
        'chain_lines': {k: len(v) for k, v in sorted(by_chain.items())},
        'chain_copies_target_equal_vs_differ': [same_target, diff_target],
    }
(OUT / 'audit.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
