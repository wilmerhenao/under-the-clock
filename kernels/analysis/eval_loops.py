"""Loop breakdown of a two-arm GPU eval from its traces (B184): per session the longest run of identical consecutive tool
calls (same function, same arguments), which tool it repeats, and how the session ended; repeated calls by tool.
usage: python3 kernels/analysis/eval_loops.py <traces root: .../train_eval> <runs/eval-... dir> <out.json> <name of base arm> <name of adapter arm>
Traces and task results as in eval_trace_stats.py. A session "loops" when one call repeats 3+ times in a row;
submit_patch{} runs (an empty submit repeated after a patch) are counted apart from real loops."""
import collections
import glob
import json
import sys

TR, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
names = {'base': sys.argv[4], 'adapter': sys.argv[5]}
out = {}
for tag, name in names.items():
    res = {}
    for line in open(f'{RES}/train_eval/repograph_p0_{tag}/results/task_results.jsonl'):
        if line.strip():
            r = json.loads(line)
            res[r['instance_id']] = r
    ends, loop_ends, rep_by_tool, loop_tool, longest, calls = (collections.Counter(), collections.Counter(),
                                                                 collections.Counter(), collections.Counter(), [], [])
    for f in sorted(glob.glob(f'{TR}/repograph_p0_{tag}/results/traces/*.json')):
        iid = f.split('trace_', 1)[1][:-5]
        d = json.load(open(f))
        last, run, best, best_fn, best_key, seen, n = None, 0, 0, None, None, set(), 0
        for s in (x for x in d.get('steps', []) if x.get('source') == 'agent'):
            for c in (s.get('tool_calls') or []):
                n += 1
                key = c['function_name'] + json.dumps(c.get('arguments') or {}, sort_keys=True)
                rep_by_tool[c['function_name']] += key in seen
                seen.add(key)
                run = run + 1 if key == last else 1
                if run > best:
                    best, best_fn, best_key = run, c['function_name'], key
                last = key
        calls.append(n)
        r = res.get(iid, {})
        e = r.get('error') or ''
        kind = ('resolved' if r.get('resolved') else 'timeout' if e.startswith('Agent exceeded session')
                else 'other_error' if e else 'ended_unresolved')
        ends[kind] += 1
        if best >= 3:
            empty_submit = best_key == 'submit_patch{}'
            loop_tool['submit_patch{}' if empty_submit else best_fn] += 1
            loop_ends[('submit_patch{}' if empty_submit else 'real') + ':' + kind] += 1
            longest.append([best, kind, best_key[:100]])
    longest.sort(reverse=True)
    real = sum(v for k, v in loop_tool.items() if k != 'submit_patch{}')
    out[name] = {'sessions': len(calls), 'tool_calls_mean_max': [round(sum(calls) / max(1, len(calls)), 1), max(calls or [0])],
                 'session_ends': dict(ends), 'loop_sessions': len(longest), 'real_loop_sessions': real,
                 'loop_sessions_by_tool': dict(loop_tool.most_common()), 'loop_sessions_by_end': dict(loop_ends),
                 'repeated_calls_by_tool': dict(rep_by_tool.most_common()), 'longest_runs': longest[:12]}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({k: {x: v[x] for x in ('sessions', 'loop_sessions', 'real_loop_sessions', 'loop_sessions_by_tool',
                                        'loop_sessions_by_end', 'session_ends')} for k, v in out.items()}, indent=1))
