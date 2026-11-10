"""How much the teachers think: reasoning length per assistant turn in the training mixes, by source (CPU, no internet).

G2b (2026-10-07, BITACORA B26): the adapter lost Gemma's long-reasoning tail (per-step reasoning p90 1,678 -> 404 chars
in the dev traces), edited in 21/53 runs vs 40/53 and solved fewer tasks. Question: do the training rows carry short
narration ("Let me look at X") instead of reasoning, and which sources do? Decision: which sources / turns G4 trains on.
For every segment line of each mounted mix: the agent's steps (the template keeps a whole tool-using episode in one model
turn, so a step is the text before each `<|tool_call>`, cut at the previous `<tool_response|>`), the thought of each step
(`<|channel>thought\\n` ... `<channel|>`), its length in characters; per source: steps, share without a thought, share
under 200 characters, p10/p50/p90, and the share of conversations whose median step is under 200 characters.
Outputs: /kaggle/working/thought/audit.json.
"""
import collections
import glob
import json
import re
from pathlib import Path

OUT = Path('/kaggle/working/thought')
OUT.mkdir(parents=True, exist_ok=True)
MIX_GLOB = '/kaggle/input/**/mix/train.jsonl'  # @define
TURN = re.compile(r'<\|turn>model\n(.*?)(?:<turn\|>|$)', re.S)
THOUGHT = re.compile(r'<\|channel>thought\n(.*?)<channel\|>', re.S)


def q(xs, p):
    return xs[int(p * (len(xs) - 1) + 0.5)] if xs else None


rep = {}
for f in sorted(glob.glob(MIX_GLOB, recursive=True)):
    name = f.split('/')[-3]
    per = collections.defaultdict(list)
    conv_med = collections.defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        if r.get('kind', 'segment') != 'segment':
            continue
        lens = []
        for turn in TURN.findall(r['text']):
            for piece in turn.split('<tool_response|>'):
                if '<|tool_call>' not in piece:
                    continue  # a closing text-only answer, not an action step
                m = THOUGHT.search(piece.split('<|tool_call>', 1)[0])
                lens.append(len(m.group(1).strip()) if m else -1)
        if not lens:
            continue
        per[r['source']] += lens
        s = sorted(lens)
        conv_med[r['source']].append(s[len(s) // 2])
    out = {}
    for src in sorted(per):
        L = sorted(per[src])
        have = [x for x in L if x >= 0]
        cm = conv_med[src]
        out[src] = {'steps': len(L), 'no_thought': round(sum(x < 0 for x in L) / len(L), 3),
                    'under_200': round(sum(0 <= x < 200 for x in L) / len(L), 3),
                    'over_1000': round(sum(x >= 1000 for x in L) / len(L), 3),
                    'p10_p50_p90': [q(have, .1), q(have, .5), q(have, .9)],
                    'conversations': len(cm), 'conv_median_under_200': round(sum(0 <= x < 200 or x < 0 for x in cm) / len(cm), 3)}
    rep[name] = out
    print(name, json.dumps(out)[:2500], flush=True)
(OUT / 'audit.json').write_text(json.dumps(rep, indent=1))
