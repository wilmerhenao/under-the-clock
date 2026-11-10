#!/usr/bin/env python3
"""Daily AI-quota cycle for the private benchmark tasks g4dev-daily-s<k> (docs/status_2026-10-12.md).

usage: daily_cycle.py quota
       daily_cycle.py plan    --day YYYY-MM-DD [--usd 7] [--shards 2] [--conc 5] [--hours 8.5] [--per-task 0.011]
       daily_cycle.py build   --day D        # kernels/bench/build/g4dev-daily-<D>-s<k>.py from kernels/bench/daily_task.py
       daily_cycle.py launch  --day D        # push each shard (creation run checks inputs), then run on Gemma 4 31B
       daily_cycle.py status  --day D
       daily_cycle.py collect --day D        # download, ledger, small evidence in runs/daily/<D>/, dataset with traces
       daily_cycle.py auto    [--usd 7]      # collect finished days; if nothing runs and quota allows, plan+build+launch

State lives in Git: private/daily/pool_meta.json (pool order and tiers, daily_pool.py), private/daily/days/<D>.json
(plan, launch, collection), private/daily/ledger.jsonl (one row per started task attempt). A task the gate never
started (cap or deadline) is not an attempt and returns to the queue. Selection: fresh pool tasks first (tiers A, B, C
in pool order), then (default, B127) repeats of solved tasks at the bundle temperature, up to REPEAT_MAX attempts each,
for pass/fail pairs; with --no-repeat, retries at temperature 1.0 (tier R and failed attempts, MAX_ATTEMPTS each).
Money: each shard gets usd/shards; the gate stops new tasks at that minus 0.15 and refuses calls at that plus 0.30.
The account quota (10 USD/day) is read before launching; the plan shrinks to fit what is left minus 1 USD, and to
the model proxy's monthly cap (100 USD, refill 11-06) minus 5 USD minus MONTH_FLOOR for each later day of the window.
Kaggle auth is ambient (KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY); nothing secret is printed or written.
"""
import argparse
import datetime
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / 'private/daily'
DAYS = PD / 'days'
LEDGER = PD / 'ledger.jsonl'
TEMPLATE = ROOT / 'kernels/bench/daily_task.py'
BUILD = ROOT / 'kernels/bench/build'
RUNS = ROOT / 'runs/daily'
DL = RUNS / '_dl'  # ignored by Git
POOL_DS = 'verracodeguacas/g4dev-daily-pool'
WHEELS_DS = 'metric/gemma-4-developer-agent-wheelhouse'
MODEL = 'gemma-4-31b-it'
MAX_ATTEMPTS = 2
RETRY_TEMP = 1.0
MONTH_FLOOR = 2.0  # USD/day kept for each remaining day of the proxy's monthly window
REPEAT_MAX = 4  # attempts per already-solved task in the repeat phase (DPO pairs, user 10-25, B127)


def task_name(day, k):
    return f'g4dev-daily-s{k}'


def load_day(day):
    p = DAYS / f'{day}.json'
    return json.loads(p.read_text()) if p.exists() else None


def save_day(day, st):
    DAYS.mkdir(parents=True, exist_ok=True)
    (DAYS / f'{day}.json').write_text(json.dumps(st, indent=1))


def ledger():
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = '\n'.join(x for x in (r.stdout + r.stderr).splitlines() if 'outdated `kaggle` version' not in x)
    return r.returncode, out


def client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def quota():  # daily AI quota (benchmark tasks) and the model proxy's monthly cap (100 USD on 10-12)
    from kagglesdk.benchmarks.types.benchmark_tasks_api_service import ApiGetBenchmarkTaskQuotaRequest
    from kagglesdk.models.types.model_proxy_api_service import ApiGetModelProxyQuotasRequest
    with client().build_kaggle_client() as k:
        r = k.benchmarks.benchmark_tasks_api_client.get_benchmark_task_quota(ApiGetBenchmarkTaskQuotaRequest())
        q = {'used': round(r.daily_quota_used, 4), 'allowed': r.total_daily_quota_allowed,
             't': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        try:
            m = k.models.model_proxy_api_client.get_model_proxy_quotas(ApiGetModelProxyQuotasRequest())
            for b in m.quota_balances:
                if 'MONTH' in str(b.refill_period).upper():
                    q['month_used'], q['month_allowed'] = round(b.quota_used, 4), b.total_quota_allowed
                    q['month_refill'] = str(b.refill_time)[:19]
        except Exception as e:  # noqa: BLE001
            q['month_err'] = type(e).__name__
    return q


def runs_of(task):
    from kagglesdk.benchmarks.types.benchmark_tasks_api_service import ApiListBenchmarkTaskRunsRequest
    api = client()
    with api.build_kaggle_client() as k:
        req = ApiListBenchmarkTaskRunsRequest()
        req.task_slug = api._make_task_slug(task)
        req.page_size = 50
        try:
            resp = k.benchmarks.benchmark_tasks_api_client.list_benchmark_task_runs(req)
        except Exception as e:  # noqa: BLE001
            return [{'error': f'{type(e).__name__}: {str(e)[:160]}'}]
    return [{'id': r.id, 'model': r.model_version_slug, 'state': str(r.state).split('.')[-1],
             'start': str(r.start_time)[:19] if r.start_time else None, 'end': str(r.end_time)[:19] if r.end_time else None,
             'error': r.error_message or None} for r in resp.runs]


def pool_meta():
    return json.loads((PD / 'pool_meta.json').read_text())


def plan(a):
    meta = pool_meta()
    if load_day(a.day) and load_day(a.day).get('launched'):
        sys.exit(f'{a.day} already launched')
    attempts, solved, failed, best_calls = {}, set(), set(), {}
    for r in ledger():
        if not r.get('llm_calls') and r.get('error'):  # infrastructure failure before any model call (10-19 litellm, B92)
            continue
        attempts[r['instance_id']] = attempts.get(r['instance_id'], 0) + 1
        (solved if r.get('resolved') else failed).add(r['instance_id'])
        if r.get('resolved'):
            best_calls[r['instance_id']] = min(best_calls.get(r['instance_id'], 99), r.get('tool_calls') or 0)
    busy = set()  # planned on days launched but not collected yet
    for p in sorted(DAYS.glob('*.json')) if DAYS.exists() else []:
        st = json.loads(p.read_text())
        if st.get('launched') and not st.get('collected'):
            busy |= {i for sh in st['shards'] for ph in sh['phases'] for i in ph['ids']}
    fresh = [i for i in meta['order'] if meta['tier'][i] in 'ABC' and i not in attempts and i not in busy]
    prior = {i: attempts.get(i, 0) + (meta['tier'][i] == 'R') for i in meta['order']}  # tier R: one bank5 attempt
    fresh_set = set(fresh)
    retry = [i for i in meta['order'] if i not in solved and i not in busy and i not in fresh_set
             and 0 < prior[i] < MAX_ATTEMPTS]
    retry.sort(key=lambda i: prior[i])
    # repeat: solved tasks again at the deployment temperature, for pass/fail pairs on one task (DPO). Tasks without a
    # failure yet first, hardest first: on 10-25 a repeat failed 19/71 times when the best success took > 16 tool calls
    # and 10/193 times otherwise (B129); then the fewest attempts.
    repeat = [i for i in meta['order'] if i in solved and i not in busy and attempts.get(i, 0) < REPEAT_MAX]
    repeat.sort(key=lambda i: (i in failed, -best_calls.get(i, 0), attempts.get(i, 0)))
    if a.repeat:
        retry = []
    usd = a.usd
    if not a.no_quota:
        q = quota()
        usd = min(usd, q['allowed'] - q['used'] - 1.0)
        if 'month_allowed' in q:  # keep MONTH_FLOOR USD/day for every later day until the monthly refill
            days = max(1, (datetime.datetime.fromisoformat(q['month_refill']) - datetime.datetime.utcnow()).days)
            usd = min(usd, q['month_allowed'] - q['month_used'] - 5.0 - MONTH_FLOOR * (days - 1))
        print('quota', q, '-> plan usd', round(usd, 2))
    if usd < 1.0:
        sys.exit('not enough AI quota left today')
    per_shard = usd / a.shards
    n = math.ceil(per_shard / a.per_task * a.over)  # over-plan: the gate stops at the cap; leftovers stay queued
    ids_f, ids_r = fresh[:n * a.shards], retry[:max(0, n * a.shards - len(fresh))]
    ids_p = repeat[:max(0, n * a.shards - len(ids_f) - len(ids_r))] if a.repeat else []
    shards = []
    for k in range(a.shards):
        phases = [{'name': 'fresh', 'temperature': None, 'ids': ids_f[k::a.shards]},
                  {'name': 'retry', 'temperature': RETRY_TEMP, 'ids': ids_r[k::a.shards]},
                  {'name': 'repeat', 'temperature': None, 'ids': ids_p[k::a.shards]}]
        shards.append({'task': task_name(a.day, k), 'shard': k, 'target_usd': round(per_shard, 3),
                       'soft_usd': round(per_shard - 0.15, 3), 'hard_usd': round(per_shard + 0.30, 3),
                       'concurrency': a.conc, 'budget_s': int(a.hours * 3600), 'time_scale': 3.0,
                       'admit_margin_s': 1800, 'min_phase_s': 2400, 'clone_threads': 4,
                       'phases': [p for p in phases if p['ids']]})
    st = {'day': a.day, 'planned_at': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'), 'usd': round(usd, 3),
          'per_task_est': a.per_task, 'queue': {'fresh_left': len(fresh), 'retry_left': len(retry),
                                                'repeat_left': len(repeat) if a.repeat else None}, 'shards': shards}
    save_day(a.day, st)
    print(json.dumps({'day': a.day, 'usd': round(usd, 2), 'per_shard_ids': [sum(len(p['ids']) for p in s['phases'])
                                                                            for s in shards],
                      'fresh': len(ids_f), 'retry': len(ids_r), 'repeat': len(ids_p), 'queue': st['queue']}))


def build(a):
    st = load_day(a.day)
    BUILD.mkdir(parents=True, exist_ok=True)
    tpl = TEMPLATE.read_text()
    head, rest = tpl.split('\n', 1)
    assert head.strip() == '# %%'
    for sh in st['shards']:
        plan_json = json.dumps({'day': a.day, **sh})
        src = f'{head}\nPLAN_JSON = {plan_json!r}\n{rest}'.replace('__TASK_NAME__', sh['task'])
        out = BUILD / f"g4dev-daily-{a.day}-s{sh['shard']}.py"
        out.write_text(src)
        compile(src, str(out), 'exec')
        sh['built'] = str(out.relative_to(ROOT))
        print('built', out.relative_to(ROOT), len(src), 'chars')
    save_day(a.day, st)


def launch(a):
    st = load_day(a.day)
    for sh in st['shards']:
        if sh.get('run_started'):
            continue
        rc, out = run(['kaggle', 'benchmarks', 'tasks', 'push', sh['task'], '-f', str(ROOT / sh['built']),
                       '-d', POOL_DS, '-d', WHEELS_DS, '--wait', '1500'], timeout=1800)
        sh['push'] = {'rc': rc, 'tail': out[-600:]}
        print(sh['task'], 'push rc', rc, out[-300:])
        if rc != 0:
            save_day(a.day, st)
            sys.exit(f"push failed for {sh['task']}")
        rc, out = run(['kaggle', 'benchmarks', 'tasks', 'run', sh['task'], '-m', MODEL], timeout=600)
        sh['run'] = {'rc': rc, 'tail': out[-600:]}
        print(sh['task'], 'run rc', rc, out[-300:])
        if rc == 0:
            sh['run_started'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        save_day(a.day, st)
    st['launched'] = all(sh.get('run_started') for sh in st['shards'])
    save_day(a.day, st)


def gemma_run(sh):
    rs = [r for r in runs_of(sh['task']) if 'gemma' in str(r.get('model', ''))]
    rs = [r for r in rs if (r.get('start') or '9')[:13] >= sh.get('run_started', '')[:13]]  # this day's run (or not started)
    return sorted(rs, key=lambda r: r.get('id') or 0)[-1] if rs else None


def status(a):
    st = load_day(a.day)
    out = {'day': a.day, 'quota': quota(), 'shards': []}
    for sh in st['shards']:
        out['shards'].append({'task': sh['task'], 'run': gemma_run(sh), 'all_runs': runs_of(sh['task'])[:4]})
    print(json.dumps(out, indent=1))
    return out


def collect(a):
    st = load_day(a.day)
    if st.get('collected'):
        print(a.day, 'already collected')
        return True
    done = True
    for sh in st['shards']:
        if sh.get('collected'):
            continue
        r = gemma_run(sh)
        if not r or not r['state'].endswith(('COMPLETED', 'ERRORED', 'FAILED', 'CANCELLED')):
            print(sh['task'], 'not finished:', r)
            done = False
            continue
        dest = DL / a.day / f"s{sh['shard']}"
        shutil.rmtree(dest, ignore_errors=True)
        rc, out = run(['kaggle', 'benchmarks', 'tasks', 'download', sh['task'], '-m', MODEL, '-o', str(dest)],
                      timeout=3600)
        files = [Path(p) for p in glob.glob(f'{dest}/**/*', recursive=True) if Path(p).is_file()]
        rep = [p for p in files if p.name == 'daily.json']
        if rc != 0 or not rep:
            print(sh['task'], 'download problem', rc, out[-400:])
            done = False
            continue
        rdir = rep[0].parent  # the Gemma run's working dir
        ev = RUNS / a.day / f"s{sh['shard']}"
        ev.mkdir(parents=True, exist_ok=True)
        shutil.copy(rep[0], ev / 'daily.json')
        rows = []
        for jl in sorted(rdir.glob('out/*/results/task_results.jsonl')):
            phase = jl.parts[-3]
            temp = next((p.get('temperature') for p in sh['phases'] if p['name'] == phase), None)
            (ev / phase).mkdir(exist_ok=True)
            shutil.copy(jl, ev / phase / 'task_results.jsonl')
            for x in jl.read_text().splitlines():
                if not x.strip():
                    continue
                t = json.loads(x)
                if 'daily gate' in str(t.get('error')):
                    continue
                rows.append({'day': a.day, 'shard': sh['shard'], 'phase': phase, 'temperature': temp,
                             'instance_id': t['instance_id'], 'resolved': bool(t.get('resolved')),
                             'llm_calls': t.get('total_llm_calls'), 'tool_calls': t.get('tool_calls'),
                             'seconds': t.get('duration_seconds'), 'patch': bool(t.get('agent_patch_size')),
                             'error': str(t.get('error'))[:160] if t.get('error') else None})
        cost = {}
        cl = rdir / 'cost.jsonl'
        if cl.exists():
            by = {}
            for x in cl.read_text().splitlines():
                try:
                    c = json.loads(x)
                except ValueError:
                    continue
                by[c.get('task') or '-'] = by.get(c.get('task') or '-', 0.0) + float(c.get('usd') or 0)
            for r_ in rows:
                r_['usd'] = round(by.get(r_['instance_id'], 0.0), 5)
            cost = {'usd': round(sum(by.values()), 4), 'unattributed': round(by.get('-', 0.0), 4)}
            (ev / 'cost_summary.json').write_text(json.dumps(cost, indent=1))
        with open(LEDGER, 'a') as f:
            f.writelines(json.dumps(r_) + '\n' for r_ in rows)
        tgz = next((p for p in files if p.name == 'out.tar.gz'), None)
        sh['collected'] = {'run': r, 'attempts': len(rows), 'resolved': sum(r_['resolved'] for r_ in rows), **cost,
                           'tarball_mb': round(tgz.stat().st_size / 1e6, 1) if tgz else None}
        print(sh['task'], sh['collected'])
        save_day(a.day, st)
    if not done:
        return False
    # traces for training: one private dataset per day with each shard's out.tar.gz
    slug = f"g4dev-daily-out-{a.day.replace('-', '')}"
    up = DL / a.day / 'upload'
    up.mkdir(parents=True, exist_ok=True)
    for sh in st['shards']:
        tgz = sorted(glob.glob(f"{DL}/{a.day}/s{sh['shard']}/**/out.tar.gz", recursive=True))
        if tgz:
            shutil.copy(tgz[0], up / f"{a.day}-s{sh['shard']}.tar.gz")
            shutil.copy(next(Path(p) for p in glob.glob(f"{DL}/{a.day}/s{sh['shard']}/**/daily.json", recursive=True)),
                        up / f"{a.day}-s{sh['shard']}-daily.json")
    (up / 'dataset-metadata.json').write_text(json.dumps(
        {'title': slug, 'id': f'verracodeguacas/{slug}', 'licenses': [{'name': 'other'}]}))
    rc, out = run(['kaggle', 'datasets', 'create', '-p', str(up)], timeout=3600)
    st['dataset'] = {'slug': f'verracodeguacas/{slug}', 'rc': rc, 'tail': out[-300:]}
    st['collected'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    st['totals'] = {k: sum((sh.get('collected') or {}).get(k) or 0 for sh in st['shards'])
                    for k in ('attempts', 'resolved', 'usd')}
    save_day(a.day, st)
    print('collected', a.day, st['totals'], 'dataset rc', rc)
    return True


def auto(a):
    today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    active = False
    for p in sorted(DAYS.glob('*.json')) if DAYS.exists() else []:
        st = json.loads(p.read_text())
        if st.get('launched') and not st.get('collected'):
            a.day = st['day']
            if not collect(a):
                active = True
    if active:
        print('a daily run is still active; nothing launched')
        return
    if a.collect_only:  # user 10-28: no new AI spend until G15's results (B149)
        print('collect-only: nothing launched')
        return
    if load_day(today) and load_day(today).get('launched'):
        print(today, 'already launched')
        return
    q = quota()
    if q['used'] > 1.0:  # the daily window has not reset yet (or something else spent today): try again later
        print('daily quota not fresh:', q)
        return
    a.day = today
    plan(a)
    build(a)
    launch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['quota', 'plan', 'build', 'launch', 'status', 'collect', 'auto'])
    ap.add_argument('--day', default=datetime.datetime.utcnow().strftime('%Y-%m-%d'))
    ap.add_argument('--usd', type=float, default=3.0)  # user 10-12: 3 USD/day (100 USD monthly proxy cap)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--conc', type=int, default=5)
    ap.add_argument('--hours', type=float, default=8.5)
    ap.add_argument('--per-task', type=float, default=0.011)
    ap.add_argument('--over', type=float, default=1.35)
    ap.add_argument('--no-quota', action='store_true', help='plan without reading the account quota')
    ap.add_argument('--collect-only', action='store_true', help='auto: collect finished days, never launch')
    ap.add_argument('--repeat', action=argparse.BooleanOptionalAction, default=True,
                    help='after fresh tasks, re-attempt solved tasks (DPO pairs) instead of temperature-1.0 retries '
                         '(default since 10-25, B127; --no-repeat restores fresh + retry)')
    a = ap.parse_args()
    if a.cmd == 'quota':
        print(quota())
    else:
        globals()[a.cmd](a)


if __name__ == '__main__':
    main()
