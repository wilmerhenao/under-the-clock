#!/usr/bin/env python3
"""Can Gemini 3 Flash write compaction summaries the way Gemma 4 31B does? (experiment B, B39)

usage: summary_style.py GEMMA_PROBE_JSON INPUTS_JSON OUT_JSON [--model google/gemini-3-flash-preview] [--n-shots 3]

GEMMA_PROBE_JSON: the benchmark task's output (`g4dev_gemma_probe.json`, two Gemma samples per summarizer prompt).
INPUTS_JSON: the task's inputs (the scorer's summarizer requests). The first n-shots prompts are the few-shot
examples (prompt -> Gemma sample 0 as earlier chat turns); the rest are held out. Conditions on each held-out prompt:
zero-shot; few-shot; few-shot plus a one-line style instruction. Scores against Gemma's two samples: length, style
features (bold labels vs markdown headers, bullets, an "Unresolved" section, third-person agent mentions), ROUGE-L and
bag-of-words cosine, with Gemma sample vs Gemma sample as the ceiling. The proxy token is minted from ambient Kaggle
auth (CLI path; Gemini answers there), kept in memory and never printed.
"""
import argparse
import collections
import json
import math
import re
import time

import httpx
from kaggle.api.kaggle_api_extended import KaggleApi

STYLE = ('Write the summary in exactly the same style, structure and length as your previous summaries in this '
         'conversation.')


def tokens(s):
    return re.findall(r'[a-z0-9_]+', s.lower())


def rouge_l(a, b):
    x, y = tokens(a), tokens(b)
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 2 * p * r / (p + r) if p + r else 0.0


def cosine(a, b):
    ca, cb = collections.Counter(tokens(a)), collections.Counter(tokens(b))
    num = sum(ca[t] * cb[t] for t in ca)
    den = math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return num / den if den else 0.0


def style(s):
    lines = s.splitlines()
    return {'chars': len(s),
            'bold_labels': len(re.findall(r'^\s*(?:[*-]\s*)?\*\*[^*\n]{2,60}:?\*\*:?', s, re.M)),
            'md_headers': sum(1 for l in lines if re.match(r'^\s*#{1,6} ', l)),
            'bullets': sum(1 for l in lines if re.match(r'^\s*(?:[*-]|\d+\.)\s', l)),
            'unresolved': bool(re.search(r'unresolved', s, re.I)),
            'agent_3rd_person': len(re.findall(r'\b(?:the (?:ai )?agent|the assistant|the user)\b', s, re.I)),
            'first_person': len(re.findall(r'\b(?:I|my|we)\b', s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gemma')
    ap.add_argument('inputs')
    ap.add_argument('out')
    ap.add_argument('--model', default='google/gemini-3-flash-preview')
    ap.add_argument('--n-shots', type=int, default=3)
    a = ap.parse_args()
    g = json.load(open(a.gemma))
    inputs = json.load(open(a.inputs))
    gem = collections.defaultdict(list)
    for r in g['sdk']:
        if r.get('ok') and r.get('content'):
            gem[r['capture']].append(r['content'])
    prompts = [(s['capture'], s['request']['messages'][0]['content']) for s in inputs['summaries']]
    shots = [(c, p) for c, p in prompts[:a.n_shots] if gem.get(c)]
    held = [(c, p) for c, p in prompts[a.n_shots:] if len(gem.get(c, [])) >= 1]
    few = []
    for c, p in shots:
        few += [{'role': 'user', 'content': p}, {'role': 'assistant', 'content': gem[c][0]}]

    api = KaggleApi()
    api.authenticate()
    env = api._fetch_model_proxy_env()
    base = env['MODEL_PROXY_URL'].rstrip('/')
    for suffix in ('/openapi', '/genai'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    key = env['MODEL_PROXY_API_KEY']

    def ask(messages):
        t = time.time()
        for k in range(3):
            r = httpx.post(base + '/openapi/chat/completions', json={'model': a.model, 'messages': messages, 'max_tokens': 8192},
                           headers={'Authorization': f'Bearer {key}'}, timeout=300)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(15 * (k + 1))
        j = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
        msg = ((j.get('choices') or [{}])[0].get('message') or {})
        cost = sum(((j.get('usage') or {}).get('cost') or {}).values()) / 1e9
        return {'status': r.status_code, 'seconds': round(time.time() - t, 1), 'content': msg.get('content') or '',
                'usage': j.get('usage'), 'cost_usd': cost, 'error': None if r.status_code == 200 else r.text.replace(key, '<r>')[:300]}

    rows, spend = [], 0.0
    for c, p in held:
        row = {'capture': c, 'gemma': gem[c], 'cand': {}}
        for cond, msgs in (('zero_shot', [{'role': 'user', 'content': p}]),
                           ('few_shot', few + [{'role': 'user', 'content': p}]),
                           ('few_shot_style', few + [{'role': 'user', 'content': p + '\n\n' + STYLE}])):
            res = ask(msgs)
            spend += res['cost_usd']
            row['cand'][cond] = res
            print(c, cond, res['status'], res['seconds'], 's | chars', len(res['content']), '| $%.4f' % res['cost_usd'],
                  '|', res['error'] or '', flush=True)
        rows.append(row)

    def score(text, refs):
        return {'rouge_l': sum(rouge_l(text, r) for r in refs) / len(refs), 'cosine': sum(cosine(text, r) for r in refs) / len(refs)}

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        refs = row['gemma']
        if len(refs) >= 2:
            s = score(refs[1], refs[:1])
            for k, v in s.items():
                agg['gemma_vs_gemma'][k].append(v)
            for k, v in style(refs[1]).items():
                agg['gemma_vs_gemma']['style_' + k].append(float(v))
        for k, v in style(refs[0]).items():
            agg['gemma'][('style_' + k)].append(float(v))
        for cond, res in row['cand'].items():
            if res['status'] != 200 or not res['content']:
                continue
            for k, v in score(res['content'], refs).items():
                agg[cond][k].append(v)
            for k, v in style(res['content']).items():
                agg[cond]['style_' + k].append(float(v))
    summary = {cond: {k: round(sum(v) / len(v), 3) for k, v in d.items()} | {'n': max(len(v) for v in d.values())}
               for cond, d in agg.items()}
    json.dump({'model': a.model, 'shots': [c for c, _ in shots], 'held_out': [c for c, _ in held], 'spend_usd': round(spend, 4),
               'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print('spend $%.4f' % spend)


if __name__ == '__main__':
    main()
