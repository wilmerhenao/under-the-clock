"""B207's pre-registered test of tool-call healing on serve-v8: c55kd~ (heal-c55's arm a, every token, merged) against
c55calls~ (B207 Run A's arm a, the tool calls only, merged); c55callsb~ (Run A's arm b, no teacher) and the full model
in the same session.
Primary: failed call attempts (B204's definition, fixed in B207: a parsed call that is malformed, a call closed by
<tool_call|> without its <|tool_call> opener, or an empty output; over the states where the model attempted a call) of
c55calls~ below c55kd~'s: one-sided exact McNemar over the states where both attempted, p < 0.05.
Secondary: c55calls~'s failed-attempt share at most twice the full model's; the parsed-only malformed share
(B202's measure, one-sided Fisher); agreement with the recorded call; served damage on the agent's tokens and per role.
Descriptive: arm b against arm a; the kinds of failure; agreement with the full model; the merges.
usage: python3 kernels/analysis/heal_test.py <serve-v8 dir> [out.json]"""
import collections
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from role_test import fisher_less  # noqa: E402
from serve_next_action import first_call, main as next_action, parse_call, swallowed  # noqa: E402

Q = '<|"|>'
MODELS = ('full', 'c55kd~', 'c55calls~', 'c55callsb~')


def kind(t):
    """None: no call attempted (prose only); 'ok'; else the kind of failure (B204)."""
    c = first_call(t)
    if c is None:
        if not t.strip():
            return 'empty'
        return 'missing opener' if '<tool_call|>' in t else None
    p = parse_call(c)
    if p is not None and not swallowed(*p):
        return 'ok'
    parts = c.split(Q)
    inside, outside = parts[1::2], ''.join(parts[0::2])
    if len(parts) % 2 == 0 or any(re.search(r'[`"\\]\s*[,}]', x) for x in inside):
        return 'wrong delimiter'
    if re.search(r'\w"\s*:', outside) or re.search(r'"\w+"\s*:', outside):
        return 'quoted key'
    if re.search(r'\w\s*=\s*\S', outside):
        return '= for :'
    return 'other'


def failed(k):
    return k not in (None, 'ok')


def one_sided_mcnemar(b, c):
    """P(X >= b), X ~ Bin(b + c, 1/2): b = states where only the reference failed, c = where only the tested model did."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def paired(K, ref, tested):
    pairs = [(failed(a), failed(b)) for a, b in zip(K[ref], K[tested]) if a is not None and b is not None]
    b_ = sum(1 for x, y in pairs if x and not y)
    c_ = sum(1 for x, y in pairs if y and not x)
    return {'both_attempted': len(pairs), f'only_{ref}_failed': b_, f'only_{tested}_failed': c_,
            'one_sided_p': round(one_sided_mcnemar(b_, c_), 6)}


def run(d, out=None):
    d = Path(d)
    na = next_action(d)
    R = json.load(open(d / 'report.json'))
    names = [n for n in MODELS if (d / f'bench_{n}.json').exists()]
    K = {n: [kind(t) for t in json.load(open(d / f'bench_{n}.json'))['turns']] for n in names}
    share = {}
    for n in names:
        a = sum(k is not None for k in K[n])
        f = sum(failed(k) for k in K[n])
        share[n] = {'failed': f, 'attempts': a, 'share': round(f / max(1, a), 4), 'no_attempt': sum(k is None for k in K[n]),
                    'kinds': dict(collections.Counter(k for k in K[n] if failed(k)))}
    res = {'states': na['states'], 'rows': na['rows'], 'models': names, 'failed_attempts': share}
    if {'c55kd~', 'c55calls~'} <= set(names):
        pr = paired(K, 'c55kd~', 'c55calls~')
        res['primary'] = dict(pr, **{'pass': pr['one_sided_p'] < 0.05})
    M = na['models']
    q = {n: (R['models'].get(n) or {}).get('quality') or {} for n in names}
    sec = {'parsed_malformed': {n: [M[n]['malformed']['malformed'], M[n]['malformed']['calls']] for n in names if n in M},
           'recorded_agreement': {n: ((M.get(n) or {}).get('vs_recorded') or {}).get('parsed', {}).get('share') for n in names},
           'agent_own_nats': {n: q[n].get('agent_own_nats') for n in names},
           'role_damage': {t['model']: t.get('role_damage') for t in R.get('table', []) if t['model'] in names}}
    if 'full' in share and 'c55calls~' in share:
        sec['calls_within_2x_full'] = share['c55calls~']['share'] <= 2 * max(share['full']['share'], 1e-9)
    if {'c55kd~', 'c55calls~'} <= set(sec['parsed_malformed']):
        (a, n1), (b, n2) = sec['parsed_malformed']['c55calls~'], sec['parsed_malformed']['c55kd~']
        sec['parsed_malformed_fisher_one_sided_p'] = round(fisher_less(a, n1, b, n2), 6)
    res['secondary'] = sec
    desc = {'vs_full_parsed': {n: ((M.get(n) or {}).get('vs_full') or {}).get('parsed') for n in names if n != 'full'},
            'merge': {n: (R['models'].get(n) or {}).get('merge') for n in names if n != 'full'},
            'adapter_mask_ok': {n: (R['models'].get(n) or {}).get('adapter_cut_matches_training_mask') for n in names if n != 'full'},
            'rc': {n: (R['models'].get(n) or {}).get('rc') for n in MODELS}}
    if {'c55calls~', 'c55callsb~'} <= set(names):
        desc['arm_b_vs_arm_a'] = paired(K, 'c55callsb~', 'c55calls~')
    res['descriptive'] = desc
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    if 'primary' in res:
        p = res['primary']
        print(f"PRIMARY failed attempts c55calls~ {share['c55calls~']['failed']}/{share['c55calls~']['attempts']} vs c55kd~ "
              f"{share['c55kd~']['failed']}/{share['c55kd~']['attempts']}; paired {p['only_c55kd~_failed']} vs "
              f"{p['only_c55calls~_failed']} of {p['both_attempted']}, one-sided p {p['one_sided_p']}; pass {p['pass']}")
    print('failed attempts', {n: (v['failed'], v['attempts'], v['kinds']) for n, v in share.items()})
    print('secondary', {k: v for k, v in sec.items() if k != 'role_damage'})
    print('role damage', sec['role_damage'])
    print('descriptive', desc)
    return res


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
