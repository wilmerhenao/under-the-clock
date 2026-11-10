"""What an agent does after its first submit_patch (B179/B181: G17 v2a kept re-submitting until the clock).

Per arm of a two-arm GPU eval: sessions that submitted, timeouts after a submit, share of all session time spent after
the first submit, first-submit time, steps after it, whether the patch changed after it (submit_patch's patch_size),
and how the model's responses after it look. The harness leaves the agent loop at the first model response without a
function call once a patch is submitted (`data/HARNESS_README.md` 5.3); a response that writes a closing text AND calls
a tool keeps the session going.
usage: python3 kernels/analysis/eval_post_submit.py <traces root: .../train_eval> <runs/eval-... dir> <out.json>
(traces as for eval_trace_stats.py; task results from the runs dir)
"""
import collections
import glob
import json
import sys

TR, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EDITS = ('edit_file', 'write_file')


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else None


def el_of(s):
    e = (s.get('extra') or {}).get('elapsed_s')
    if e is None and s.get('tool_calls'):
        e = (s['tool_calls'][0].get('extra') or {}).get('elapsed_s')
    return e


out = {}
for tag in ('base', 'adapter'):
    res = {}
    for line in open(f'{RES}/train_eval/repograph_p0_{tag}/results/task_results.jsonl'):
        if line.strip():
            r = json.loads(line)
            res[r['instance_id']] = r
    rows, R, R_ct = [], collections.Counter(), []
    for f in glob.glob(f'{TR}/repograph_p0_{tag}/results/traces/*.json'):
        iid = f.split('trace_', 1)[1][:-5]
        r = res.get(iid)
        if r is None:
            continue
        st = [s for s in json.load(open(f)).get('steps', []) if s.get('source') == 'agent']
        groups = collections.OrderedDict()  # agent events of one model response share its elapsed time
        last = 0.0
        for s in st:
            e = el_of(s)
            last = e if e is not None else last
            groups.setdefault(round(last, 1), []).append(s)
        t_sub, sizes, steps_after, edits_after = None, [], 0, 0
        ends = [e for s in st for e in (el_of(s), ((s.get('observation') or {}).get('extra') or {}).get('elapsed_s'))
                if e is not None]
        last_event = max(ends) if ends else 0.0
        for e, g in groups.items():
            calls = [c['function_name'] for s in g for c in (s.get('tool_calls') or [])]
            text = any((s.get('extra') or {}).get('event_type') == 'text' and (s.get('message') or '').strip() for s in g)
            if t_sub is not None:
                steps_after += 1
                edits_after += any(c in EDITS for c in calls)
                R['responses_after_submit'] += 1
                R['after_submit_text_and_call'] += bool(text and calls)
                R['after_submit_call_only'] += bool(calls and not text)
                R['after_submit_no_call'] += not calls
                R_ct += [(s.get('metrics') or {}).get('completion_tokens') or 0 for s in g]
            for s in g:
                for c in (s.get('tool_calls') or []):
                    if c['function_name'] == 'submit_patch':
                        try:
                            sizes.append(json.loads((s.get('observation') or {}).get('content') or '{}').get('patch_size'))
                        except ValueError:
                            sizes.append(None)
            if t_sub is None and 'submit_patch' in calls:
                t_sub = e
        rows.append({'resolved': bool(r['resolved']), 'dur': r['duration_seconds'], 't_sub': t_sub, 'n_sub': len(sizes),
                     'timeout': (r.get('error') or '').startswith('Agent exceeded session timeout'),
                     'changed': len(set(sizes)) > 1, 'steps_after': steps_after, 'edits_after': edits_after,
                     'silent_tail': r['duration_seconds'] - last_event})
    sub = [x for x in rows if x['t_sub'] is not None]
    post = [x['dur'] - x['t_sub'] for x in sub]
    tot = sum(x['dur'] for x in rows)
    out[tag] = {'sessions': len(rows), 'submitted': len(sub), 'resolved': sum(x['resolved'] for x in rows),
                'timeouts': sum(x['timeout'] for x in rows), 'timeouts_after_submit': sum(x['timeout'] for x in sub),
                'submit_calls_mean_max': [round(sum(x['n_sub'] for x in rows) / len(rows), 2), max(x['n_sub'] for x in rows)],
                'first_submit_s_p50_p90': [round(q([x['t_sub'] for x in sub], .5)), round(q([x['t_sub'] for x in sub], .9))],
                'session_s_total': round(tot), 'post_submit_s_total': round(sum(post)),
                'post_submit_share_of_session_time': round(sum(post) / tot, 3),
                'post_submit_s_p50_p90': [round(q(post, .5)), round(q(post, .9))],
                'responses_after_submit_p50_p90': [q([x['steps_after'] for x in sub], .5), q([x['steps_after'] for x in sub], .9)],
                'sessions_editing_after_submit': sum(x['edits_after'] > 0 for x in sub),
                'sessions_patch_changed_after_first_submit': sum(x['changed'] for x in sub),
                'resolved_with_patch_changed_after_first_submit': sum(x['resolved'] and x['changed'] for x in sub),
                'responses_after_submit': dict(R),
                # task end minus the last event the trace logged: verification time, plus any model request still
                # running when the session clock ran out (a response that never returned is never logged)
                'silent_tail_s_p50_p90_submitted': [round(q([x['silent_tail'] for x in sub], .5)),
                                                    round(q([x['silent_tail'] for x in sub], .9))],
                'silent_tail_s_p50_p90_not_submitted': [round(q([x['silent_tail'] for x in rows if x['t_sub'] is None], .5) or 0),
                                                        round(q([x['silent_tail'] for x in rows if x['t_sub'] is None], .9) or 0)],
                'max_completion_tokens_after_submit': max([c for c in R_ct] or [0])}
    print(tag, json.dumps(out[tag]))
json.dump(out, open(OUT, 'w'), indent=1)
