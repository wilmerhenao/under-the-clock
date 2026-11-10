"""Next-action test of a serve run (serve-v5, B192), offline: per model, its greedy first tool call at each held-out agent
state against the full model's, compared three ways (exact text, same tool, same parsed call = tool name + argument
dict), with 95% intervals from a bootstrap over source rows (several states come from one trajectory, so states are not
independent); the floor is full vs full_b (same weights, other engine numerics) and the sampling row full vs full at
T 0.2. Malformed calls without the regex's false positives on code: a call that does not parse, or a string argument
that swallowed another of the tool's own parameters (`a.py`,start_line:350 -> one argument), by tool.
usage: python3 kernels/analysis/serve_next_action.py <dir with report.json and bench_<model>.json> [out.json]"""
import json
import random
import re
import sys
from pathlib import Path

Q = '<|"|>'
CALL = re.compile(r'<\|tool_call>(.*?)<tool_call\|>', re.S)
ROOT = Path(__file__).resolve().parents[2]
PARAMS = {t['function']['name'] if 'function' in t else t['name']:
          set(((t.get('function', t)).get('parameters') or {}).get('properties', {}))
          for t in json.load(open(ROOT / 'private/harness_tools_openai.json'))}
ALL_PARAMS = set().union(*PARAMS.values())


def parse_call(c):
    """(name, args) of a Gemma 4 tool call body, e.g. 'call:read_file{filepath:<|"|>a.py<|"|>,start_line:3}'; None if it
    does not parse (unclosed string, repeated key, junk between arguments)."""
    m = re.match(r'\s*call:([A-Za-z_]\w*)\{(.*)\}\s*$', c or '', re.S)
    if not m:
        return None
    name, body, args, i = m.group(1), m.group(2), {}, 0
    while i < len(body):
        km = re.match(r'\s*([A-Za-z_]\w*)\s*:', body[i:])
        if not km:
            return None
        key, i = km.group(1), i + km.end()
        if body.startswith(Q, i):
            j = body.find(Q, i + len(Q))
            if j < 0:
                return None
            val, i = ('s', body[i + len(Q):j]), j + len(Q)
        else:
            depth, j = 0, i
            while j < len(body):
                if body.startswith(Q, j):
                    k = body.find(Q, j + len(Q))
                    if k < 0:
                        return None
                    j = k + len(Q)
                    continue
                ch = body[j]
                depth += ch in '{['
                depth -= ch in '}]'
                if ch == ',' and depth == 0:
                    break
                j += 1
            val, i = ('v', body[i:j].strip()), j
        if key in args:
            return None
        args[key] = val
        if i < len(body):
            if body[i] != ',':
                return None
            i += 1
    return name, args


def swallowed(name, args):
    """A string argument that contains another of this tool's parameters as ', key:' after a stray quote or backtick."""
    keys = PARAMS.get(name) or ALL_PARAMS
    for k, (kind, v) in args.items():
        if kind == 's' and any(re.search(r'[`"\']\s*,\s*' + re.escape(p) + r'\s*:', v) for p in keys if p != k):
            return True
    return False


def first_call(text):
    m = CALL.search(text or '')
    return m.group(1) if m else None


def summarize(calls):
    made = [c for c in calls if c is not None]
    parsed = [parse_call(c) for c in made]
    by_tool = {}
    for c, p in zip(made, parsed):
        tool = p[0] if p else (re.match(r'\s*call:([A-Za-z_]\w*)', c) or [None, '?'])[1]
        bad = p is None or swallowed(*p)
        t = by_tool.setdefault(tool, [0, 0])
        t[0] += 1
        t[1] += bad
    bad = sum(v[1] for v in by_tool.values())
    return {'states': len(calls), 'calls': len(made), 'malformed': bad, 'share': round(bad / max(1, len(made)), 4),
            'unparsed': sum(p is None for p in parsed), 'by_tool': {k: {'calls': v[0], 'malformed': v[1]}
                                                                  for k, v in sorted(by_tool.items())}}


def same(x, y, how):
    if x is None or y is None:
        return None
    if how == 'text':
        return x.strip() == y.strip()
    px, py = parse_call(x), parse_call(y)
    if how == 'tool':   # the tool name even when the arguments do not parse
        tool = lambda c: (re.match(r'\s*call:([A-Za-z_]\w*)', c) or [None, c.split('{', 1)[0]])[1]  # noqa: E731
        return tool(x) == tool(y)
    return px is not None and px == py


def agree(xs, ys, rows, how, boot=2000, seed=0):
    """Share of the states where the reference (ys) made a call at which the model made the same one (no call = a
    mismatch, as in prune_serve.agree); 95% interval by resampling source rows."""
    per = {}
    for x, y, r in zip(xs, ys, rows):
        if y is None:
            continue
        per.setdefault(r, []).append(bool(same(x, y, how)))
    if not per:
        return None
    keys = sorted(per)
    est = sum(sum(v) for v in per.values()) / sum(len(v) for v in per.values())
    rng, sims = random.Random(seed), []
    for _ in range(boot):
        pick = [per[rng.choice(keys)] for _ in keys]
        n = sum(len(v) for v in pick)
        sims.append(sum(sum(v) for v in pick) / n if n else est)
    sims.sort()
    return {'share': round(est, 3), 'ci95': [round(sims[int(0.025 * boot)], 3), round(sims[int(0.975 * boot) - 1], 3)],
            'states': sum(len(v) for v in per.values()), 'rows': len(keys)}


def main(d, out=None):
    d = Path(d)
    R = json.load(open(d / 'report.json'))
    models = {}
    for name, r in R['models'].items():
        b = d / f'bench_{name}.json'
        src = json.load(open(b)) if b.exists() else r
        if src.get('turns'):
            models[name] = [first_call(t) for t in src['turns']]
            if src.get('turns_sampled'):
                models[f"{name}@T{src.get('turns_sampled_temperature')}"] = [first_call(t) for t in src['turns_sampled']]
    n = len(models['full'])
    rows = R.get('turn_state_rows') or list(range(n))   # serve-v2/v2b: one state per row
    assert len(rows) == n, (len(rows), n)
    recorded = R.get('turn_state_recorded_calls')
    res = {'states': n, 'rows': len(set(rows)), 'models': {}}
    if recorded:
        res['recorded'] = summarize(recorded)
    for name, cs in models.items():
        m = {'malformed': summarize(cs)}
        for ref in ('full', 'full_b'):
            if ref in models and ref != name:
                m[f'vs_{ref}'] = {how: agree(cs, models[ref], rows, how) for how in ('text', 'parsed', 'tool')}
        if recorded:
            m['vs_recorded'] = {how: agree(cs, recorded, rows, how) for how in ('parsed', 'tool')}
        res['models'][name] = m
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    print(f"{n} states from {res['rows']} rows")
    for name, m in res['models'].items():
        vf = m.get('vs_full') or {}
        fmt = lambda a: f"{a['share']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]" if a else '-'  # noqa: E731
        print(f"{name:>12}: parsed {fmt(vf.get('parsed'))}  tool {fmt(vf.get('tool'))}  text {fmt(vf.get('text'))}  "
              f"malformed {m['malformed']['malformed']}/{m['malformed']['calls']} (unparsed {m['malformed']['unparsed']})")
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
