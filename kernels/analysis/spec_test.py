"""B202 Run 2' (serve-v7): does n-gram speculative decoding speed the full model up while keeping its behaviour?
Pre-registered (rebuttal, 11-09): behaviour is kept if the speculative engine's next calls agree with plain greedy
decoding's at least as often as the numerics floor allows (serve-v5's full vs full_b: 0.78, 95% interval 0.70-0.86, so
>= 0.70) and its malformed calls are at most 3 per 100. Speed: decode tok/s at 2k / 16k context and 8 streams (synthetic
continuations) and on 16 real agent turns at 1 and 8 streams, as ratios to the plain engine in the same session.
usage: python3 kernels/analysis/spec_test.py <serve-v7 dir> [out.json]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_next_action import main as next_action  # noqa: E402


def run(d, out=None):
    d = Path(d)
    na = next_action(d)
    b = {n: json.load(open(d / f'bench_{n}.json')) for n in ('full', 'full_b')}
    sp = {k: round(b['full_b']['speed'][k]['decode_tok_s'] / b['full']['speed'][k]['decode_tok_s'], 3)
          for k in ('p2k', 'p16k')}
    sp['streams8'] = round(b['full_b']['speed']['streams8']['decode_tok_s_total']
                           / b['full']['speed']['streams8']['decode_tok_s_total'], 3)
    for k in ('streams1', 'streams8'):
        sp[f'turns_{k}'] = round(b['full_b']['turn_speed'][k]['tok_s'] / b['full']['turn_speed'][k]['tok_s'], 3)
    m = na['models']['full_b']
    agree = (m.get('vs_full') or {}).get('parsed')
    mal = m['malformed']
    kept = bool(agree and agree['share'] >= 0.70 and mal['malformed'] * 100 <= 3 * max(1, mal['calls']))
    same_text = sum(x == y for x, y in zip(b['full']['turns'], b['full_b']['turns']))
    res = {'states': na['states'], 'speedup': sp, 'agreement_parsed': agree, 'malformed': [mal['malformed'], mal['calls']],
           'malformed_plain': [na['models']['full']['malformed']['malformed'], na['models']['full']['malformed']['calls']],
           'identical_turn_text': same_text, 'behaviour_kept': kept, 'engine': b['full_b']['engine'],
           'turn_speed': {n: b[n]['turn_speed'] for n in b}}
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'turn_speed'}, indent=1))
    return res


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
