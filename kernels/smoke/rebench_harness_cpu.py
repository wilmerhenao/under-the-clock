"""PR-Gym integration test: do SWE-rebench tasks run end to end through the official swegemma harness?

Builds a harness-format data dir from g4dev-rebench-feasibility-cpu's verified tasks (tasks.jsonl rows with FAIL_TO_PASS
+ snapshots), installs the competition harness wheels, and runs gold and empty-patch controls with per-task uv envs
(eval_harness --uv-task-env). Success = the harness reproduces the standalone labels (gold passes, empty fails), which
means GPU rollouts on these tasks only need a model endpoint. CPU + internet (uv fetches Pythons and packages).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
GOLD_CONC = 4  # @define (2 gives cleaner verification timings on the 4-core CPU kernel)
NONE_CONC = 4  # @define
FRESHSWE = False  # @define (True: tasks from the FreshSWE-Py dataset, converted to PR-Gym rows, instead of bank outputs)
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'rebench_harness'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:2000]}', flush=True)


def find(pattern_tail, depth=7):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    # 1. harness wheels (same install rule as controls_cpu: only wheels this interpreter supports)
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    stage('harness_installed', wheels=len(ok), python=sys.version.split()[0])

    # 2. data dir in competition layout
    comp = find('competitions/gemma-4-developer-agent/tasks.jsonl') or find('gemma-4-developer-agent/tasks.jsonl')
    data = WORK / 'rbdata'
    (data / 'snapshots').mkdir(parents=True, exist_ok=True)
    for d in ('wheels', 'graphs', 'embeddings'):
        (data / d).mkdir(exist_ok=True)
    shutil.copytree(comp.parent / 'sample_submission', data / 'sample_submission', dirs_exist_ok=True)
    rows, seen = [], set()
    if FRESHSWE:  # FreshSWE-Py (moshegolderson/freshswe-py): competition schema + locks/<base_commit>.txt, CPython 3.13
        fs = find('freshswe-py/tasks.jsonl') or find('tasks.jsonl')
        fs_root = fs.parent
        for line in open(fs):
            if not line.strip():
                continue
            x = json.loads(line)
            snap, lock = fs_root / 'snapshots' / f"{x['instance_id']}.tgz", fs_root / 'locks' / f"{x['base_commit']}.txt"
            if x['instance_id'] in seen or not snap.exists() or not lock.exists():
                continue
            seen.add(x['instance_id'])
            r = {k: x.get(k) for k in ('instance_id', 'repo', 'base_commit', 'patch', 'test_patch', 'problem_statement',
                                       'hints_text', 'created_at')}
            r.update({'python': '3.13', 'requirements': lock.read_text(), 'install': 'pip install -e .',
                      'license_name': '', 'source': 'moshegolderson/freshswe-py', 'statement_source': x.get('statement_source'),
                      'src_lines_changed': x.get('src_lines_changed'), 'f2p_count': x.get('f2p_count')})
            os.symlink(snap, data / 'snapshots' / f"{r['instance_id']}.tgz")
            rows.append(r)
        (OUT / 'tasks_verified.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for f in [] if FRESHSWE else sorted(glob.glob('/kaggle/input/**/rebench/tasks_verified.jsonl', recursive=True)):  # all mounted banks
        snaps = Path(f).parent / 'snapshots'
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            src = snaps / f'{r["instance_id"].replace("/", "__")}.tgz'
            if r['instance_id'] in seen or not src.exists():
                continue
            seen.add(r['instance_id'])
            os.symlink(src, data / 'snapshots' / f'{r["instance_id"]}.tgz')
            rows.append(r)
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    stage('data_ready', tasks=len(rows), snapshots=len(list((data / 'snapshots').glob('*.tgz'))))

    # 3. controls through the official Evaluator
    for control in ('gold', 'none'):
        out_dir = OUT / control
        rc = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data), '--tag', control,
                             '--out', str(out_dir), '--tasks', 'all', '--control', control, '--uv-task-env',
                             '--time-verify', '--concurrency', str(GOLD_CONC if control == 'gold' else NONE_CONC),
                             '--budget-s', str(5.5 * 3600), '--timeout-seconds', '600']).returncode
        f = out_dir / 'summary_smoke.json'
        summ = json.loads(f.read_text()) if f.exists() else None
        if summ:
            summ.pop('run_ids', None)
        stage(f'{control}_done', returncode=rc, summary=summ)

    def load(c):
        f = OUT / c / 'results' / 'task_results.jsonl'
        return {r['instance_id']: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}
    gold, none = load('gold'), load('none')
    agree = [i for i in gold if gold[i].get('resolved') and none.get(i, {}).get('resolved') is False]
    (OUT / 'harness_trusted_ids.json').write_text(json.dumps(sorted(agree)))
    env = [json.loads(l) for l in (OUT / 'gold' / 'env_setup.jsonl').read_text().splitlines() if l.strip()] \
        if (OUT / 'gold' / 'env_setup.jsonl').exists() else []
    built = sorted(r['env_s'] for r in env if 'env_s' in r)
    stage('env_timing', builds=len(built), reused=sum(1 for r in env if r.get('reused')),
          median_s=built[len(built) // 2] if built else None, p90_s=built[int(len(built) * 0.9)] if built else None,
          max_s=built[-1] if built else None,
          bulk_ok=sum(1 for r in env if r.get('frozen_rc') == 0),
          nodeps_ok=sum(1 for r in env if r.get('frozen_nodeps_rc') == 0),
          per_line=sum(1 for r in env if 'line_fail' in r),
          install_fail=sum(1 for r in env if r.get('install_rc') not in (0, None)))
    # verification pytest wall time of the gold runs (the scorer uses the bundle's timeout_seconds for this command)
    vt = [json.loads(l) for l in (OUT / 'gold' / 'verify_times.jsonl').read_text().splitlines() if l.strip()] \
        if (OUT / 'gold' / 'verify_times.jsonl').exists() else []
    vt_ok = sorted((r['secs'], r.get('task')) for r in vt if r.get('task') in set(agree))
    secs = [x for x, _ in vt_ok]
    stage('verify_timing', rows=len(vt), trusted_rows=len(secs), gold_conc=GOLD_CONC,
          median_s=secs[len(secs) // 2] if secs else None, p90_s=secs[int(len(secs) * 0.9)] if secs else None,
          p99_s=secs[int(len(secs) * 0.99)] if secs else None, max_s=secs[-1] if secs else None,
          over_30=sum(x > 30 for x in secs), over_60=sum(x > 60 for x in secs), over_120=sum(x > 120 for x in secs),
          over_240=sum(x > 240 for x in secs), slowest=vt_ok[-15:])
    prev = set()
    for f in glob.glob('/kaggle/input/**/rebench_harness/harness_trusted_ids.json', recursive=True):
        prev |= set(json.loads(Path(f).read_text()))
    if prev:
        stage('vs_previous_list', previous=len(prev), now=len(agree), both=len(prev & set(agree)),
              only_previous=sorted(prev - set(agree))[:60], only_now=sorted(set(agree) - prev)[:60])
    stage('result', n=len(rows), harness_trusted=len(agree),
          gold_fail=sorted(i for i in gold if not gold[i].get('resolved'))[:40],
          none_pass=sorted(i for i in none if none[i].get('resolved')),
          gold_errors=sorted({str(gold[i].get('error'))[:150] for i in gold if gold[i].get('error')})[:10])
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
