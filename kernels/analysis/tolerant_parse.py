"""Offline: how many malformed next calls would a tolerant call parser repair? (paper Section 4, B208)

The cuts' typical failure is a string argument closed with a backtick or a JSON quote instead of Gemma's delimiter
<|"|>, so the string runs on and swallows the next argument (`a.py`,start_line:350 -> one argument), or a JSON-style
key (start_line": 151). The tolerant parser is serve_next_action.parse_call after two rewrites of the call body:
(1) a backtick, a double quote or an escaped double quote directly before ', <one of the tool's parameters>:' or before
the closing brace, inside an open <|"|> string, becomes <|"|>; (2) a double quote between a key and its colon is dropped. A malformed call counts
as repaired if the rewritten call parses and swallows no parameter; 'exact' if it then equals the full model's parsed
call at the same state. Nothing is re-run: the inputs are the serve runs' saved greedy turns.
usage: python3 kernels/analysis/tolerant_parse.py [out.json]"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_next_action import PARAMS, ALL_PARAMS, first_call, parse_call, swallowed  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'colab' / 'prune'))
from gemma4_tolerant import rewrite_args  # noqa: E402  (the same rewrite the serving kernel runs, B209)

ROOT = Path(__file__).resolve().parents[2]
RUNS = {  # next-action tests whose saved turns we re-parse: (directory, models)
    'serve-v5 (109 states)': ('runs/prune-serve5-gpu/v1/prune/serve-v5',
                              ['full', 'full_b', 'c85', 'c70', 'c70+kd~', 'c70own', 'c55']),
    'serve-v6 (141 states)': ('runs/prune-serve6-gpu/v1/prune/serve-v6', ['full', 'c55', 'c55role']),
    'serve-v7 (109 states)': ('runs/prune-serve7-gpu/v2/prune/serve-v7', ['full', 'full_b']),
}


def bad(c):
    p = parse_call(c)
    return p is None or swallowed(*p)


def rewrite(c):
    """The call with the plugin's rewrites (kernels/colab/prune/gemma4_tolerant.py: the rule the serving kernel runs)."""
    m = re.match(r'(\s*call:([A-Za-z_]\w*)\{)(.*)(\}\s*)$', c, re.S)
    if not m:
        return c
    return m.group(1) + rewrite_args(m.group(3), PARAMS.get(m.group(2)) or ALL_PARAMS) + m.group(4)


def main(out=None):
    res, tot = {}, {'malformed': 0, 'repaired': 0, 'exact': 0}
    for label, (d, models) in RUNS.items():
        d = ROOT / d
        calls = {}
        for mname in models:
            b = d / f'bench_{mname}.json'
            if b.exists():
                calls[mname] = [first_call(t) for t in json.load(open(b)).get('turns') or []]
        full = calls['full']
        rows = {}
        for mname, cs in calls.items():
            mal = rep = ex = 0
            for k, c in enumerate(cs):
                if c is None or not bad(c):
                    continue
                mal += 1
                r = rewrite(c)
                if not bad(r):
                    rep += 1
                    ref = full[k] if k < len(full) else None
                    ex += ref is not None and mname != 'full' and parse_call(r) == parse_call(ref)
            rows[mname] = {'calls': sum(c is not None for c in cs), 'malformed': mal, 'repaired': rep, 'exact_full': ex}
            for key, v in (('malformed', mal), ('repaired', rep), ('exact', ex)):
                tot[key] += v
        res[label] = rows
    res['total'] = tot
    for label, rows in res.items():
        if label == 'total':
            continue
        print(label)
        for mname, r in rows.items():
            print(f"  {mname:>8}: malformed {r['malformed']:>2} of {r['calls']:>3} calls, repaired {r['repaired']:>2}, "
                  f"then equal to the full model's call {r['exact_full']:>2}")
    print('total', tot)
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
