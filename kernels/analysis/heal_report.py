"""Read a healing run's log (pruner part 3, B176) and score it against the pre-registered criteria.

usage: python3 kernels/analysis/heal_report.py runs/heal-c70-tpu/v1/tpu/log.jsonl [out.json]

Prints, per 'heal_eval' line: held-out nats on agent-written tokens (own) and on all tokens, for the teacher (the full
model: mask 1, LoRA 0) and the student (the cut model with its adapter), the student's damage (student - teacher own
nats) and its KL to the teacher. Then the training loss per phase and epoch, and the B176 criteria:
  healing works   : arm a's final own-token damage < +0.04 AND arm a's final kl_own <= 0.5 x the start kl_own
  teacher matters : arm a's final kl_own clearly below arm b's at equal steps (reported as the ratio a / b; "clearly"
                    is read as <= 0.8, stated here before the result is seen)
Consistency targets from part 1 / serve v3: teacher own ~0.258, start damage ~+0.08, own tokens 25,896.
Runs that train the tool calls (B195 / B207) also log a calls view (calls_nats, kl_calls); their criteria, at epoch 2
(tag ep1: tags count epochs from 0, the last is 'final') and epoch 4 ('final'):
  calls healing works      : arm a's kl_calls at epoch 2 <= 0.5 x the start's AND its own-token damage <= +0.04
  teacher matters on calls : arm a's kl_calls below arm b's at epoch 2 and at epoch 4
"""
import collections
import json
import statistics as stt
import sys

path = sys.argv[1]
L = [json.loads(x) for x in open(path) if x.strip()]
ev = [r for r in L if r.get('stage') == 'heal_eval']
errs = [r for r in L if r.get('stage') == 'heal_eval_error']
out = {'log': path, 'lines': len(L), 'heal_eval_errors': [e.get('error') for e in errs]}

teacher = next((r['teacher'] for r in ev if 'teacher' in r), None)
out['teacher'] = teacher
rows = []
for r in ev:
    s = r.get('student')
    if not s:
        continue
    row = {'tag': r['tag'], 't_min': r.get('t_min'), 'own_nats': s['own_nats'], 'all_nats': s['all_nats'],
           'kl_own': s.get('kl_own'), 'kl_all': s.get('kl_all'), 'own_tokens': s['own_tokens']}
    if s.get('calls_nats') is not None:
        row.update(calls_nats=s['calls_nats'], kl_calls=s.get('kl_calls'), calls_tokens=s.get('calls_tokens'))
    if teacher:
        row['damage_own'] = round(s['own_nats'] - teacher['own_nats'], 4)
        row['damage_all'] = round(s['all_nats'] - teacher['all_nats'], 4)
        if s.get('calls_nats') is not None and teacher.get('calls_nats') is not None:
            row['damage_calls'] = round(s['calls_nats'] - teacher['calls_nats'], 4)
    rows.append(row)
out['student'] = rows

# training loss per phase / epoch (mean of applied steps), steps and tokens
steps = [r for r in L if r.get('stage') == 'step']
by = collections.defaultdict(list)
for r in steps:
    if r.get('applied'):
        by[(r['phase'], r['ep'])].append(r['loss'])
out['train_loss'] = {f'{p}:ep{e}': {'steps': len(v), 'mean': round(stt.mean(v), 4), 'first': v[0], 'last': v[-1]}
                     for (p, e), v in sorted(by.items())}
out['steps_per_phase'] = dict(collections.Counter(r['phase'] for r in steps if r.get('applied')))
out['done'] = [{k: r.get(k) for k in ('phase', 'stop', 'steps_ok', 'planned', 'tokens', 'step_errors', 'rung', 'min')}
               for r in L if r.get('stage') == 'done']
out['heal_mask'] = next(({k: v for k, v in r.items() if k not in ('stage',)} for r in L if r.get('stage') == 'heal_mask'),
                        None)
out['heal_arm'] = [r for r in L if r.get('stage') == 'heal_arm']


def pick(tag):
    return next((r for r in rows if r['tag'] == tag), None)


start, fa, fb = pick('start'), pick('a:final'), pick('b:final')
crit = {}
if teacher:
    crit['consistency'] = {'teacher_own': teacher['own_nats'], 'target_teacher_own': 0.258,
                           'own_tokens': teacher['own_tokens'], 'target_own_tokens': 25896,
                           'start_damage': start and start.get('damage_own'), 'target_start_damage': 0.08}
if start and fa:
    crit['healing_works'] = {
        'a_final_damage': fa['damage_own'], 'damage_below_0.04': fa['damage_own'] < 0.04,
        'kl_own_start': start['kl_own'], 'kl_own_a_final': fa['kl_own'],
        'kl_ratio': round(fa['kl_own'] / start['kl_own'], 3) if start['kl_own'] else None,
        'kl_halved': fa['kl_own'] <= 0.5 * start['kl_own'] if start['kl_own'] else None}
    crit['healing_works']['verdict'] = bool(crit['healing_works']['damage_below_0.04'] and crit['healing_works']['kl_halved'])
if fa and fb:
    r_ab = fa['kl_own'] / fb['kl_own'] if fb['kl_own'] else None
    crit['teacher_matters'] = {'kl_own_a': fa['kl_own'], 'kl_own_b': fb['kl_own'],
                               'ratio_a_over_b': round(r_ab, 3) if r_ab is not None else None,
                               'damage_a': fa['damage_own'], 'damage_b': fb['damage_own'],
                               'steps_a': out['steps_per_phase'].get('a'), 'steps_b': out['steps_per_phase'].get('b'),
                               'verdict': bool(r_ab is not None and r_ab <= 0.8)}
# equal steps: arm b runs arm a's batches in a's order (PHASE2_MATCH_A), but may stop early on time; compare per epoch
eq = {}
for r in rows:
    if r['tag'].startswith('a:ep'):
        rb = pick('b:' + r['tag'][2:])
        if rb and rb['kl_own']:
            eq[r['tag'][2:]] = {'kl_own_a': r['kl_own'], 'kl_own_b': rb['kl_own'],
                                'ratio_a_over_b': round(r['kl_own'] / rb['kl_own'], 3),
                                'damage_a': r.get('damage_own'), 'damage_b': rb.get('damage_own')}
if eq:
    crit['teacher_matters_per_epoch'] = eq
# the calls view (B195 / B207): epoch 2 = tag ep1, epoch 4 = final
a2, b2, fb_ = pick('a:ep1'), pick('b:ep1'), fb
if start and start.get('kl_calls') is not None and a2 and a2.get('kl_calls') is not None:
    w = {'kl_calls_start': start['kl_calls'], 'kl_calls_a_ep2': a2['kl_calls'],
         'kl_calls_ratio': round(a2['kl_calls'] / start['kl_calls'], 3) if start['kl_calls'] else None,
         'kl_calls_halved': bool(start['kl_calls'] and a2['kl_calls'] <= 0.5 * start['kl_calls']),
         'damage_own_a_ep2': a2.get('damage_own'), 'damage_own_le_0.04': bool(a2.get('damage_own') is not None
                                                                               and a2['damage_own'] <= 0.04),
         'damage_calls_start': start.get('damage_calls'), 'damage_calls_a_ep2': a2.get('damage_calls'),
         'epoch4': fa and {'kl_calls_a': fa.get('kl_calls'), 'damage_own_a': fa.get('damage_own'),
                           'damage_calls_a': fa.get('damage_calls')}}
    w['verdict'] = bool(w['kl_calls_halved'] and w['damage_own_le_0.04'])
    crit['calls_healing_works'] = w
    if b2 and fa and fb_ and all(x.get('kl_calls') is not None for x in (b2, fa, fb_)):
        crit['teacher_matters_on_calls'] = {'epoch2': [a2['kl_calls'], b2['kl_calls']], 'epoch4': [fa['kl_calls'], fb_['kl_calls']],
                                            'verdict': bool(a2['kl_calls'] < b2['kl_calls'] and fa['kl_calls'] < fb_['kl_calls'])}
out['criteria'] = crit

print(f"{'tag':<12}{'own':>9}{'all':>9}{'dmg_own':>9}{'dmg_all':>9}{'kl_own':>9}{'kl_all':>9}{'dmg_calls':>10}{'kl_calls':>9}")
if teacher:
    print(f"{'teacher':<12}{teacher['own_nats']:>9.4f}{teacher['all_nats']:>9.4f}")
for r in rows:
    f = lambda x: f'{x:>9.4f}' if isinstance(x, (int, float)) else f'{"-":>9}'  # noqa: E731
    print(f"{r['tag']:<12}{f(r['own_nats'])}{f(r['all_nats'])}{f(r.get('damage_own'))}{f(r.get('damage_all'))}"
          f"{f(r['kl_own'])}{f(r['kl_all'])} {f(r.get('damage_calls'))}{f(r.get('kl_calls'))}")
print(json.dumps({k: out[k] for k in ('criteria', 'train_loss', 'done', 'heal_eval_errors')}, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
