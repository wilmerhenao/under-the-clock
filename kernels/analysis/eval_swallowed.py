"""Malformed tool calls in task-session traces without the regex detector's false positives (B194): a call counts only if
one of its string arguments swallowed another of that tool's own parameters (`a.py`,start_line:350 -> one argument),
the test of serve_next_action.swallowed; plus the tool-error share as the harness reports it. Per arm, by tool.
usage: python3 kernels/analysis/eval_swallowed.py <out.json> name=<traces dir> [name2=<traces dir> ...]"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_next_action import PARAMS, ALL_PARAMS  # noqa: E402


def swallowed(name, args):
    keys = PARAMS.get(name) or ALL_PARAMS
    for k, v in (args or {}).items():
        if isinstance(v, str) and any(re.search(r'[`"\']\s*,\s*' + re.escape(p) + r'\s*:', v) for p in keys if p != k):
            return True
    return False


out = {}
for spec in sys.argv[2:]:
    name, d = spec.split('=', 1)
    calls, bad, sess_bad, by_tool, sessions = 0, 0, 0, collections.Counter(), 0
    for f in sorted(glob.glob(f'{d}/*.json')):
        tr = json.load(open(f))
        sessions += 1
        nb = 0
        for s in tr.get('steps', []):
            if s.get('source') != 'agent':
                continue
            for c in s.get('tool_calls') or []:
                calls += 1
                if swallowed(c.get('function_name'), c.get('arguments')):
                    bad += 1
                    nb += 1
                    by_tool[c.get('function_name')] += 1
        sess_bad += nb > 0
    out[name] = {'sessions': sessions, 'tool_calls': calls, 'swallowed_calls': bad,
                 'share': round(bad / max(1, calls), 4), 'sessions_with_one': sess_bad, 'by_tool': dict(by_tool)}
    print(name, json.dumps(out[name]))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
