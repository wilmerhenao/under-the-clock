"""B202's pre-registered test of the role-aware score on serve-v6 (c55 = agent_ff vs c55role = role_ff, same session).
Primary: c55role's malformed share (unparsed, or a string argument that swallowed another parameter; of the states
where it made a call) at most half of c55's, with a one-sided Fisher exact p < 0.05. Secondary: recorded-call
agreement (parsed) c55role >= c55 + 0.05; served damage (agent_own_nats) c55role <= c55 + 0.03. Descriptive: paired
McNemar over the states where both made a call, agreement with the full model, served loss per role of the token, and
whether c55 reproduces serve-v5's c55 on the states the two runs share. Run 2 (tasks) only if the primary p < 0.01.
usage: python3 kernels/analysis/role_test.py <serve-v6 dir> [<serve-v5 dir>] [out.json]"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_next_action import first_call, main as next_action, parse_call, swallowed  # noqa: E402


def fisher_less(a, n1, b, n2):
    """One-sided Fisher exact p that group 1's rate (a of n1) is lower than group 2's (b of n2): P(X <= a)."""
    N, K = n1 + n2, a + b
    tot = math.comb(N, n1)
    return sum(math.comb(K, k) * math.comb(N - K, n1 - k) for k in range(max(0, K - n2), a + 1)) / tot


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bad(c):
    p = parse_call(c)
    return p is None or swallowed(*p)


def run(d, d5=None, out=None):
    d = Path(d)
    na = next_action(d)
    R = json.load(open(d / 'report.json'))
    M = na['models']
    m55, mr = M['c55']['malformed'], M['c55role']['malformed']
    s55, sr = m55['malformed'] / max(1, m55['calls']), mr['malformed'] / max(1, mr['calls'])
    p = fisher_less(mr['malformed'], mr['calls'], m55['malformed'], m55['calls'])
    res = {'states': na['states'], 'rows': na['rows'],
           'primary': {'c55': [m55['malformed'], m55['calls'], round(s55, 4)],
                       'c55role': [mr['malformed'], mr['calls'], round(sr, 4)],
                       'fisher_one_sided_p': round(p, 6), 'half_or_less': sr <= 0.5 * s55,
                       'pass': bool(sr <= 0.5 * s55 and p < 0.05), 'run2': bool(sr <= 0.5 * s55 and p < 0.01)}}
    rec = {n: (M[n].get('vs_recorded') or {}).get('parsed') for n in M}
    q = {n: R['models'][n]['quality'] for n in ('full', 'c55', 'c55role')}
    res['secondary'] = {
        'recorded_agreement': {n: rec[n]['share'] if rec[n] else None for n in M},
        'recorded_agreement_pass': bool(rec['c55role'] and rec['c55'] and rec['c55role']['share'] >= rec['c55']['share'] + 0.05),
        'agent_own_nats': {n: v['agent_own_nats'] for n, v in q.items()},
        'served_damage_pass': q['c55role']['agent_own_nats'] <= q['c55']['agent_own_nats'] + 0.03}
    # paired over states: malformed or not where both cuts made a call
    turns = {n: [first_call(t) for t in json.load(open(d / f'bench_{n}.json'))['turns']] for n in ('full', 'c55', 'c55role')}
    both = [(bad(x), bad(y)) for x, y in zip(turns['c55'], turns['c55role']) if x is not None and y is not None]
    b_ = sum(1 for x, y in both if x and not y)      # c55 malformed, c55role fine
    c_ = sum(1 for x, y in both if y and not x)
    res['descriptive'] = {
        'paired_states_both_called': len(both), 'c55_only_malformed': b_, 'c55role_only_malformed': c_,
        'mcnemar_exact_p': round(mcnemar_exact(b_, c_), 6),
        'no_call_states': {n: sum(c is None for c in v) for n, v in turns.items()},
        'malformed_full': [M['full']['malformed']['malformed'], M['full']['malformed']['calls']],
        'vs_full_parsed': {n: (M[n].get('vs_full') or {}).get('parsed') for n in M if n != 'full'},
        'role_nats': {n: {g: v['nats'] for g, v in (q[n].get('by_role') or {}).items()} for n in q},
        'role_tokens': {g: v['tokens'] for g, v in (q['full'].get('by_role') or {}).items()}}
    if d5:   # the states the two runs share (serve-v5's first rows, same order): the same cut, the same outputs?
        d5 = Path(d5)
        t5 = json.load(open(d5 / 'bench_c55.json'))['turns']
        r5 = json.load(open(d5 / 'report.json'))
        k = len(t5)
        same_prompts = R['turn_state_prompt_tokens'][:k] == r5['turn_state_prompt_tokens']
        res['descriptive']['vs_serve_v5'] = {
            'shared_states': k, 'same_prompt_lengths': same_prompts,
            'c55_identical_text': sum(a == b for a, b in zip(json.load(open(d / 'bench_c55.json'))['turns'][:k], t5)),
            'full_identical_text': sum(a == b for a, b in zip(json.load(open(d / 'bench_full.json'))['turns'][:k],
                                                                json.load(open(d5 / 'bench_full.json'))['turns']))}
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    pr = res['primary']
    print(f"PRIMARY malformed c55role {pr['c55role'][0]}/{pr['c55role'][1]} ({pr['c55role'][2]:.1%}) vs c55 "
          f"{pr['c55'][0]}/{pr['c55'][1]} ({pr['c55'][2]:.1%}); one-sided Fisher p {pr['fisher_one_sided_p']}; "
          f"pass {pr['pass']}; run 2 {pr['run2']}")
    print('secondary', res['secondary'])
    print('descriptive', {k: v for k, v in res['descriptive'].items() if k not in ('role_nats',)})
    print('role nats', res['descriptive']['role_nats'])
    return res


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], a[1] if len(a) > 1 and not a[1].endswith('.json') else None,
        a[-1] if len(a) > 1 and a[-1].endswith('.json') else None)
