"""Per-arm step statistics of a two-arm GPU eval from its traces (B159, B165): tokens and text per step, calls without a
tool call, seconds per step, long steps, repeated identical tool calls (loops) and steps per task by outcome.
usage: python3 kernels/analysis/eval_trace_stats.py <traces root: .../train_eval> <runs/eval-... dir> <out.json> <name of base arm> <name of adapter arm> [eval prefix, default repograph_p0; dev evals: dev_p0]
The traces root holds repograph_p0_{base,adapter}/results/traces/*.json (kaggle kernels output ... --file-pattern
'train_eval/repograph_p0_(base|adapter)/results/traces/.*\\.json$' into a scratch dir); task results come from the runs dir."""
import json, glob, statistics as stt, sys
TR, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]   # traces root, runs/eval-... root, output json
names = {'base': sys.argv[4], 'adapter': sys.argv[5]}
PFX = sys.argv[6] if len(sys.argv) > 6 else 'repograph_p0'
def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else None
out = {}
for tag, name in names.items():
    results = {}
    rp = f'{RES}/train_eval/{PFX}_{tag}/results/task_results.jsonl'
    for line in (open(rp) if glob.glob(rp) else []):   # without task results every session counts as 'other'
        r = json.loads(line)
        results[r['instance_id']] = r
    toks, chars, steps_per, nocall, gaps, long_sess = [], [], [], 0, [], []
    rep = {'calls': 0, 'repeats': 0, 'sessions_with_3_identical_calls_in_a_row': 0}
    by_outcome = {'resolved': [], 'timeout': [], 'other': []}
    for f in glob.glob(f'{TR}/{PFX}_{tag}/results/traces/*.json'):
        iid = f.split('trace_', 1)[1][:-5]
        d = json.load(open(f))
        ag = [s for s in d.get('steps', []) if s.get('source') == 'agent']
        steps_per.append(len(ag))
        prev = 0.0
        mx = 0
        seen, run, best_run, last = set(), 0, 0, None
        for s in ag:
            for c in (s.get('tool_calls') or []):
                key = c['function_name'] + json.dumps(c.get('arguments') or {}, sort_keys=True)
                rep['calls'] += 1
                rep['repeats'] += key in seen
                seen.add(key)
                run = run + 1 if key == last else 1
                best_run = max(best_run, run)
                last = key
            ct = (s.get('metrics') or {}).get('completion_tokens')
            if ct is not None:
                toks.append(ct)
                mx = max(mx, ct)
            chars.append(len(s.get('message') or ''))
            if not s.get('tool_calls'):
                nocall += 1
            el = (s.get('extra') or {}).get('elapsed_s')
            if el is not None:
                gaps.append(el - prev)
                prev = el
        rep['sessions_with_3_identical_calls_in_a_row'] += best_run >= 3
        r = results.get(iid, {})
        kind = 'resolved' if r.get('resolved') else 'timeout' if (r.get('error') or '').startswith('Agent exceeded session') else 'other'
        by_outcome[kind].append(len(ag))
        long_sess.append((mx > 1540, kind))
    n_steps = sum(steps_per)
    ls = [k for b, k in long_sess if b]
    out[name] = {'traces': len(steps_per), 'agent_steps': n_steps,
                 'completion_tokens_p50_p90_mean': [q(toks, .5), q(toks, .9), round(stt.mean(toks))] if toks else None,
                 'step_text_chars_p50_p90': [q(chars, .5), q(chars, .9)], 'steps_per_task_mean': round(stt.mean(steps_per), 1),
                 'agent_steps_without_tool_call': nocall, 'share_without_tool_call': round(nocall / max(1, n_steps), 3),
                 'seconds_per_step_p50_p90': [round(q(gaps, .5), 1), round(q(gaps, .9), 1)] if gaps else None,
                 'calls_over_1024_tokens_share': round(sum(t > 1024 for t in toks) / max(1, len(toks)), 3),
                 'token_share_of_calls_over_1024': round(sum(t for t in toks if t > 1024) / max(1, sum(toks)), 3),
                 'sessions_with_a_step_over_1540_tokens': {'n': len(ls), 'resolved': ls.count('resolved'), 'timeout': ls.count('timeout')},
                 'repeated_calls_share': round(rep['repeats'] / max(1, rep['calls']), 3),
                 'sessions_with_3_identical_calls_in_a_row': rep['sessions_with_3_identical_calls_in_a_row'],
                 'steps_per_task_by_outcome': {k: (round(stt.mean(v), 1) if v else None, len(v)) for k, v in by_outcome.items()}}
    print(name, json.dumps(out[name]))
json.dump(out, open(OUT, 'w'), indent=1)
