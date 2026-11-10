"""One G10 summary round: collect the Gemma summaries of g4dev-g10-summaries, keep the good ones, version the dataset
g4dev-g10-summaries, and (with --pending) build the next pending dataset from a builder round's pending.jsonl.

  python3 kernels/analysis/g10_round.py collect --round N      # download, filter, write summaries_rN.jsonl, version dataset
  python3 kernels/analysis/g10_round.py pending --round N --from runs/g10-mix-cpu-rN/mix/pending.jsonl

A summary is kept when the request finished normally ('stop') and the text reads as a summary: not empty, no tool-call
markup, not opening with a code block, at most MAX_CHARS characters (mix8: p90 1,564; Gemma sometimes continues as the
agent instead of summarising, B41). A rejected (prompt, chain) is re-drawn in the next round with another seed (salt).
State: runs/g10-summaries/state.json ({hash:chain -> salt}), per-round evidence in runs/g10-summaries/.
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / 'runs/g10-summaries'
SCR = Path('/tmp/claude-0/-home-user-kaggle-stuff/d6870b9e-a3a7-5a0e-a5b8-a882d849026f/scratchpad')
SUM_DIR, PEND_DIR = SCR / 'g10_summaries', SCR / 'g10_pending'
MAX_CHARS = 6000
MAX_DRAWS = 3  # rejected draws of one (prompt, chain) before it is dropped (the builder then leaves that chain out)
BAD_MARKS = ('<|tool_call>', '<tool_call', '"function_name"', 'call:run_command', 'call:read_file', '<|channel>')


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, '\n'.join(x for x in (r.stdout + r.stderr).splitlines() if 'token' not in x.lower())


def state():
    f = EV / 'state.json'
    return json.loads(f.read_text()) if f.exists() else {'salt': {}, 'rounds': {}}


def why_bad(x):
    s = (x.get('summary') or '').strip()
    if x.get('finish') not in (None, 'stop'):
        return f"finish={x.get('finish')}"
    if not s:
        return 'empty'
    if len(s) > MAX_CHARS:
        return f'{len(s)} chars'
    if s.startswith('```'):
        return 'opens with code'
    if any(m in s for m in BAD_MARKS):
        return 'tool-call markup'
    return None


def collect(a):
    EV.mkdir(parents=True, exist_ok=True)
    out = SCR / f'g10_task_r{a.round}'
    rc, log = run(['kaggle', 'benchmarks', 'tasks', 'download', 'g4dev-g10-summaries', '-m', 'gemma-4-31b-it',
                   '--output', str(out)])
    files = sorted(glob.glob(str(out / '**/summaries.jsonl'), recursive=True))
    errs = sorted(glob.glob(str(out / '**/errors.jsonl'), recursive=True))
    if rc or not files:
        sys.exit(f'download rc {rc}, summaries files {files}\n{log[-1500:]}')
    st = state()
    good, bad, chars, cost = [], [], [], 0.0
    for line in open(files[-1]):
        if not line.strip():
            continue
        x = json.loads(line)
        cost += sum(((x.get('usage') or {}).get('cost') or {}).values()) / 1e9
        w = why_bad(x)
        if w:
            bad.append({'hash': x['hash'], 'chain': x['chain'], 'why': w, 'head': (x.get('summary') or '')[:200]})
            key = f"{x['hash']}:{x['chain']}"
            st['salt'][key] = st['salt'].get(key, 0) + 1
            if st['salt'][key] >= MAX_DRAWS:  # this prompt keeps making Gemma act instead of summarise: leave the chain out
                good.append({'chain': x['chain'], 'hash': x['hash'], 'drop': True, 'why': w})
        else:
            good.append({k: x[k] for k in ('chain', 'hash', 'summary', 'finish', 'seconds') if k in x})
            chars.append(len(x['summary']))
    good_n = sum(1 for g in good if not g.get('drop'))
    n_err = 0
    for f in errs:  # a completed request that called a tool or hit the cap is re-drawn like a rejected summary
        for line in open(f):
            if not line.strip():
                continue
            n_err += 1
            e = json.loads(line)
            if e.get('status') == 200 and e.get('finish') in ('tool_calls', 'length'):
                key = f"{e['hash']}:{e['chain']}"
                st['salt'][key] = st['salt'].get(key, 0) + 1
                if st['salt'][key] >= MAX_DRAWS:
                    good.append({'chain': e['chain'], 'hash': e['hash'], 'drop': True, 'why': f"finish={e['finish']}"})
    (SUM_DIR / f'summaries_r{a.round}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in good))
    chars.sort()
    rep = {'round': a.round, 'good': good_n, 'dropped': len(good) - good_n, 'rejected': len(bad), 'errors': n_err, 'cost_usd': round(cost, 4),
           'chars': {q: chars[min(len(chars) - 1, int(len(chars) * q / 100))] for q in (10, 50, 90)} if chars else {},
           'rejected_examples': bad[:8]}
    st['rounds'][str(a.round)] = {k: v for k, v in rep.items() if k != 'rejected_examples'}
    (EV / 'state.json').write_text(json.dumps(st, indent=1))
    (EV / f'round{a.round}.json').write_text(json.dumps(rep, indent=1))
    rc, log = run(['kaggle', 'datasets', 'version', '-p', str(SUM_DIR), '-m', f'round {a.round}: {good_n} summaries'])
    print(json.dumps({k: v for k, v in rep.items() if k != 'rejected_examples'}), '| rejected', [b['why'] for b in bad],
          '| dataset version rc', rc, log[-200:])


def pending(a):
    st = state()
    rows = [json.loads(line) for line in open(a.src) if line.strip()]
    for e in rows:  # a rejected (prompt, chain) is re-drawn with another seed
        salt = {str(k): st['salt'][f"{e['hash']}:{k}"] for k in e['chains'] if f"{e['hash']}:{k}" in st['salt']}
        if salt:
            e['salt'] = salt
    for f in PEND_DIR.glob('*.jsonl'):
        f.unlink()
    (PEND_DIR / f'pending_r{a.round}.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in rows))
    rc, log = run(['kaggle', 'datasets', 'version', '-p', str(PEND_DIR), '-m', f'round {a.round}'])
    print({'round': a.round, 'prompts': len(rows), 'requests': sum(len(e['chains']) for e in rows),
           'salted': sum(1 for e in rows if e.get('salt')), 'dataset rc': rc}, log[-200:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['collect', 'pending'])
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--from', dest='src')
    main_a = ap.parse_args()
    collect(main_a) if main_a.cmd == 'collect' else pending(main_a)
