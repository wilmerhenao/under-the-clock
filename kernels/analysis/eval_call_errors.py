"""Malformed tool calls in a GPU eval's traces (B189): calls where one argument's value swallowed the next argument
because the model closed a string with the wrong delimiter (a backtick or a JSON quote instead of Gemma's <|"|>), seen
as `value`,key:...` or `value\\", key\\": ...` inside a string argument; plus the share of tool results with status
"error" and the sessions where malformed calls repeat.
usage: python3 kernels/analysis/eval_call_errors.py <out.json> name=<traces dir> [name2=<traces dir> ...]
(traces dir = .../repograph_p0_<arm>/results/traces)"""
import ast
import collections
import glob
import json
import re
import sys

BAD = re.compile(r'(`|\\?")\s*,\s*\\?"?[a-z_]{3,20}\\?"?\s*:')   # a closing quote/backtick, then ", key:" inside a value


def as_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
    except (ValueError, SyntaxError):
        return []


def status(obs):
    m = re.search(r'"status":\s*"(\w+)"', str(obs))
    return m.group(1) if m else None


out = {}
for spec in sys.argv[2:]:
    name, d = spec.split('=', 1)
    calls = errors = bad = 0
    bad_sessions, rep_bad_sessions, by_tool, sessions = 0, 0, collections.Counter(), 0
    for f in sorted(glob.glob(f'{d}/*.json')):
        t = json.load(open(f))
        sessions += 1
        nbad = 0
        for s in t.get('steps', []):
            if s.get('source') != 'agent':
                continue
            tcs = as_list(s.get('tool_calls'))
            for c in tcs:
                calls += 1
                args = c.get('arguments') or {}
                vals = [v for v in (args.values() if isinstance(args, dict) else []) if isinstance(v, str)]
                if any(BAD.search(v) for v in vals):
                    bad += 1
                    nbad += 1
                    by_tool[c.get('function_name')] += 1
            if tcs and status(s.get('observation')) == 'error':
                errors += 1
        bad_sessions += nbad > 0
        rep_bad_sessions += nbad >= 3
    out[name] = {'sessions': sessions, 'tool_calls': calls, 'malformed_calls': bad,
                 'malformed_share': round(bad / max(1, calls), 4), 'malformed_by_tool': dict(by_tool.most_common()),
                 'sessions_with_a_malformed_call': bad_sessions, 'sessions_with_3_plus': rep_bad_sessions,
                 'steps_with_tool_error': errors, 'tool_error_share': round(errors / max(1, calls), 4)}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
