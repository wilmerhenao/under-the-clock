#!/usr/bin/env python3
"""How often the scorer's compaction hits Gemma, and what it costs, from the 129 long-cap Gemma traces (gen-gpu-v1 pass 0).

A compaction is detected as an agent step whose prompt is >2k tokens shorter than the previous step's (the context only
grows otherwise). Reports: tasks compacted, first compaction (tool-call index), compactions per task, share of steps after
the first compaction, resolved tasks and where their edits fall, prompt size right after a compaction, residual seconds
at compaction steps (the summary request, decoded at the proxy's 8.5 tok/s), and repeated calls before/after compaction
(exact repeats of an earlier call; read_file on a file already read before the compaction).
usage: compaction_in_traces.py [out.json]
"""
import glob
import json
import statistics as st
import sys

ok = set(json.load(open('runs/reverify-gemma-v5d/resolved_ids.json')))
paths = sorted(glob.glob('runs/gen-gpu-v1/gen/gemma-4-31b-qat/pass0/results/traces/*.json'))


def key(tc):
    n, a = tc.get('function_name'), tc.get('arguments') or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            a = {}
    if n == 'read_file':
        return ('read', a.get('filepath') or a.get('path'), a.get('start_line'), a.get('end_line'))
    if n == 'run_command':
        return ('cmd', (a.get('command') or '').strip())
    return (n, json.dumps(a, sort_keys=True)[:200])


R = dict(tasks=len(paths), compacted=0, first_call=[], per_task=[], steps=0, steps_after=0, resolved=0,
         resolved_compacted=0, last_edit_after=0, first_edit_after=0, prompt_before=[], prompt_after=[],
         resid_comp=[], resid_other=[], calls_before=0, rep_before=0, calls_after=0, rep_after=0,
         reads_after=0, reads_after_seen=0)
for p in paths:
    iid = p.split('trace_')[-1][:-5]
    steps = [s for s in json.load(open(p))['steps'] if s.get('source') == 'agent']
    prev_pt = prev_e = None
    calls, comp, n_comp, first, last_edit_after, first_edit_after, first_edit_seen = 0, False, 0, None, False, False, False
    edits_so_far, comp_with_edits = 0, []
    seen_all, files_before = set(), set()
    for s in steps:
        m = s.get('metrics') or {}
        pt, c = m.get('prompt_tokens') or 0, m.get('completion_tokens') or 0
        e = (s.get('extra') or {}).get('elapsed_s')
        dropped = prev_pt is not None and pt < prev_pt - 2000
        if dropped:
            comp_with_edits.append(edits_so_far > 0)
            R['prompt_before'].append(prev_pt)
            R['prompt_after'].append(pt)
            n_comp += 1
            if not comp:
                first = calls
            comp = True
        if e is not None and prev_e is not None:
            (R['resid_comp'] if dropped else R['resid_other']).append((e - prev_e) - c / 8.5)
        prev_pt, prev_e = pt, e
        R['steps'] += 1
        R['steps_after'] += comp
        tcs = s.get('tool_calls') or []
        for tc in tcs:
            k = key(tc)
            if k[0] == 'submit_patch':
                continue
            dup = k in seen_all
            if comp:
                R['calls_after'] += 1
                R['rep_after'] += dup
                if k[0] == 'read':
                    R['reads_after'] += 1
                    R['reads_after_seen'] += k[1] in files_before
            else:
                R['calls_before'] += 1
                R['rep_before'] += dup
                if k[0] == 'read':
                    files_before.add(k[1])
            seen_all.add(k)
        calls += len([x for x in tcs if x.get('function_name') != 'submit_patch'])
        if any(x.get('function_name') in ('edit_file', 'write_file') for x in tcs):
            edits_so_far += 1
            last_edit_after = comp
            if not first_edit_seen:
                first_edit_seen, first_edit_after = True, comp
    R.setdefault('comp_events', []).extend(comp_with_edits)
    if comp_with_edits:
        R.setdefault('first_comp_with_edit', []).append(comp_with_edits[0])
    if comp:
        R['compacted'] += 1
        R['first_call'].append(first)
        R['per_task'].append(n_comp)
    if iid in ok:
        R['resolved'] += 1
        R['resolved_compacted'] += comp
        R['last_edit_after'] += last_edit_after
        R['first_edit_after'] += first_edit_after
out = {
    'tasks': R['tasks'], 'compacted': R['compacted'],
    'first_compaction_tool_call': {'median': st.median(R['first_call']), 'min': min(R['first_call']), 'max': max(R['first_call'])},
    'compacted_before_call': {cap: sum(1 for f in R['first_call'] if f < cap) for cap in (20, 24, 28, 32, 40)},
    'compactions_per_compacted_task': round(st.mean(R['per_task']), 2),
    'share_of_steps_after_first_compaction': round(R['steps_after'] / R['steps'], 3),
    'resolved': R['resolved'], 'resolved_compacted': R['resolved_compacted'],
    'resolved_last_edit_after_compaction': R['last_edit_after'], 'resolved_first_edit_after_compaction': R['first_edit_after'],
    'prompt_tokens_before_compaction_median': st.median(R['prompt_before']),
    'prompt_tokens_after_compaction_median': st.median(R['prompt_after']),
    'residual_s_at_compaction_median': round(st.median(R['resid_comp'])), 'residual_s_other_median': round(st.median(R['resid_other'])),
    'exact_repeat_rate_before': round(R['rep_before'] / R['calls_before'], 3),
    'exact_repeat_rate_after': round(R['rep_after'] / R['calls_after'], 3),
    'reads_after_of_files_read_before': round(R['reads_after_seen'] / R['reads_after'], 3),
    'calls_before': R['calls_before'], 'calls_after': R['calls_after'],
    'compactions_with_an_earlier_edit': round(sum(R['comp_events']) / len(R['comp_events']), 3),
    'first_compactions_with_an_earlier_edit': round(sum(R['first_comp_with_edit']) / len(R['first_comp_with_edit']), 3),
    'compaction_events': len(R['comp_events']),
}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    open(sys.argv[1], 'w').write(json.dumps(out, indent=1) + '\n')
