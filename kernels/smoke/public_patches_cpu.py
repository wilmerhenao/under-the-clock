"""Verify public agent trajectories on our task bank with the official Evaluator (CPU, no model).

Public datasets hold agent trajectories for many of the SWE-rebench tasks in our bank (scout of 2026-10-04), some without
any outcome label. Here every final patch of those trajectories is applied and tested by the same harness that scores
the competition (per-task uv env, gold-verified tasks only). Result: execution-verified labels in our own environment,
the teacher's solve rate on our bank, how verification depends on trajectory length, and (where the dataset has its own
label) how often the two agree. Outputs: /kaggle/working/public_patches/{verified.jsonl,summary.json,report.json}.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
# (dataset, passes, shard cap in MB); pass j verifies each task's j-th shortest trajectory
SOURCES = [('nvidia/SWE-Hero-openhands-trajectories', 4, 6000), ('nebius/SWE-rebench-openhands-trajectories', 3, 6000), ('nebius/SWE-agent-trajectories', 2, 3000), ('nvidia/SWE-Zero-openhands-trajectories', 2, 13000)]  # @define
DEADLINE_S = 9.5 * 3600  # @define
START_PASS = 0  # @define (skip passes below this on tasks a previous run already covered; banks in NEW_BANKS start at 0)
NEW_BANKS = []  # @define (e.g. ['bank4']: standalone-verified banks whose tasks are candidates, all passes)
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'public_patches'
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/pp')
TMP.mkdir(parents=True, exist_ok=True)
HUB = 'https://huggingface.co'
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
ID_COLS = ('instance_id', 'task_id', 'problem_id', 'id')
PATCH_COLS = ('model_patch', 'generated_patch', 'git_patch', 'pred_patch', 'prediction', 'diff', 'patch')
LABEL_COLS = ('resolved', 'target', 'success', 'reward', 'is_resolved')
TRAJ_COLS = ('trajectory', 'messages', 'history', 'conversations', 'traj')
TID_COLS = ('trajectory_id', 'traj_id', 'run_id', 'id')
MODEL_COLS = ('model_name', 'model', 'agent_model')


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:2500]}', flush=True)


def find(pattern_tail, depth=7):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-public-patches'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep((40 if '429' in str(e) else 4) * (k + 1))


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-public-patches'})
            with urllib.request.urlopen(req, timeout=900) as r, open(dest, 'wb') as fh:
                shutil.copyfileobj(r, fh, 1 << 22)
            with open(dest, 'rb') as fh:  # a truncated download has no parquet footer
                fh.seek(-4, 2)
                if fh.read(4) == b'PAR1':
                    return dest.stat().st_size
            raise RuntimeError('no parquet footer (truncated download)')
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def count_calls(traj):
    """(tool calls, messages, characters) for the trajectory layouts seen in the scout."""
    if isinstance(traj, str):
        chars = len(traj)
        try:
            traj = json.loads(traj)
        except Exception:  # noqa: BLE001
            return None, None, chars
    else:
        chars = len(json.dumps(traj, default=str))
    if not isinstance(traj, list):
        return None, None, chars
    calls = 0
    for m in traj:
        if not isinstance(m, dict):
            continue
        tc = m.get('tool_calls')
        if tc is not None and hasattr(tc, '__len__') and len(tc):
            calls += len(tc)
        elif m.get('role') == 'ai' or (m.get('role') == 'assistant' and tc is None and 'tool_calls' not in m):
            calls += 1  # text-command formats: one action per assistant turn
    return calls, len(traj), chars


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    # 1. harness
    wh = find('swegemma-*.whl').parent
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok = [w for w in sorted(wh.glob('*.whl')) if 'cutlass' not in w.name.lower()
          and any(str(t) in supported for t in parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    import pyarrow.parquet as pq

    # 2. data dir: every mounted bank; candidates = harness-trusted ids + bank part 3 (standalone-verified)
    comp = find('competitions/gemma-4-developer-agent/tasks.jsonl') or find('gemma-4-developer-agent/tasks.jsonl')
    data = WORK / 'rbdata'
    (data / 'snapshots').mkdir(parents=True, exist_ok=True)
    for d in ('wheels', 'graphs', 'embeddings'):
        (data / d).mkdir(exist_ok=True)
    shutil.copytree(comp.parent / 'sample_submission', data / 'sample_submission', dirs_exist_ok=True)
    rows, seen, bank3, new_tasks = [], set(), set(), set()
    for f in sorted(glob.glob('/kaggle/input/**/rebench/tasks_verified.jsonl', recursive=True)):
        snaps = Path(f).parent / 'snapshots'
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            src = snaps / f'{r["instance_id"].replace("/", "__")}.tgz'
            if r['instance_id'] in seen or not src.exists():
                continue
            seen.add(r['instance_id'])
            if 'bank3' in f:
                bank3.add(r['instance_id'])
            if any(b in f for b in NEW_BANKS):
                new_tasks.add(r['instance_id'])
            os.symlink(src, data / 'snapshots' / f'{r["instance_id"]}.tgz')
            rows.append(r)
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    trusted = set()
    for f in glob.glob('/kaggle/input/**/rebench_harness/harness_trusted_ids.json', recursive=True):
        trusted |= set(json.loads(Path(f).read_text()))
    cand = (trusted | bank3 | new_tasks) & seen
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    stage('data_ready', bank=len(rows), harness_trusted=len(trusted), bank3=len(bank3), new_bank_tasks=len(new_tasks),
          candidates=len(cand), start_pass=START_PASS)
    if len(cand) < 100:
        raise RuntimeError('bank or trusted lists not mounted')

    # 3. collect the trajectories of each source that end on a candidate task
    recs = []
    for ds, n_pass, cap_mb in SOURCES:
        try:
            tree = get(f'{HUB}/api/datasets/{ds}/tree/refs%2Fconvert%2Fparquet?recursive=true')
            files = sorted((t['path'], t.get('size', 0)) for t in tree if t.get('path', '').endswith('.parquet'))
            got, n_rows, n_hit, cols_seen, shards = 0, 0, 0, None, 0
            per_task = {}
            for path, size in files:
                if got + size > cap_mb * 1e6 or time.time() - T0 > 0.35 * DEADLINE_S:
                    break
                dest = TMP / 'shard.parquet'
                download(f'{HUB}/datasets/{ds}/resolve/refs%2Fconvert%2Fparquet/{urllib.parse.quote(path)}', dest)
                got += size
                shards += 1
                pf = pq.ParquetFile(dest)
                cols = pf.schema_arrow.names
                idc = next((c for c in ID_COLS if c in cols), None)
                patchc = next((c for c in PATCH_COLS if c in cols), None)
                labelc = next((c for c in LABEL_COLS if c in cols), None)
                trajc = next((c for c in TRAJ_COLS if c in cols), None)
                tidc = next((c for c in TID_COLS if c in cols and c != idc), None)
                modelc = next((c for c in MODEL_COLS if c in cols), None)
                cols_seen = {'id': idc, 'patch': patchc, 'label': labelc, 'traj': trajc, 'traj_id': tidc, 'model': modelc,
                             'all': cols}
                if not idc or not patchc:
                    break
                n_rows += pf.metadata.num_rows
                ids = set(pq.read_table(dest, columns=[idc]).column(0).to_pylist()) & cand
                if ids:
                    keep = [c for c in (idc, patchc, labelc, trajc, tidc, modelc) if c]
                    for b in pq.read_table(dest, columns=keep, filters=[(idc, 'in', sorted(ids))]).to_batches(max_chunksize=50):
                        for r in b.to_pylist():
                            calls, msgs, chars = count_calls(r.get(trajc)) if trajc else (None, None, None)
                            rec = {'dataset': ds, 'instance_id': r[idc], 'traj_id': str(r.get(tidc)) if tidc else None,
                                   'label': r.get(labelc) if labelc else None, 'model': r.get(modelc) if modelc else None,
                                   'tool_calls': calls, 'messages': msgs, 'chars': chars, 'patch': r.get(patchc) or ''}
                            per_task.setdefault(r[idc], []).append(rec)
                            n_hit += 1
                dest.unlink(missing_ok=True)
            for iid, lst in per_task.items():
                lst.sort(key=lambda x: (x['tool_calls'] if x['tool_calls'] is not None else 1e9, x['chars'] or 0))
                if ds == 'nebius/SWE-agent-trajectories':  # labelled: shortest solved and shortest unsolved per task
                    pos = [x for x in lst if x['label'] is True][:1]
                    neg = [x for x in lst if x['label'] is False][:1]
                    lst = pos + neg
                    for x in lst:
                        x['pass'] = 0 if x['label'] is True else 1
                else:
                    lst = lst[:n_pass]
                    for j, x in enumerate(lst):
                        x['pass'] = j
                    if iid not in new_tasks:  # passes a previous run already verified are skipped
                        lst = [x for x in lst if x['pass'] >= START_PASS]
                recs += lst
            stage('source', dataset=ds, shards=shards, of=len(files), mb=round(got / 1e6), rows=n_rows, on_candidates=n_hit,
                  tasks=len(per_task), columns=cols_seen)
        except Exception as e:  # noqa: BLE001
            stage('source_failed', dataset=ds, error=f'{type(e).__name__}: {e}'[:400], tb=traceback.format_exc()[-800:])

    # 4. verify, one pass at a time (one patch per task per pass)
    def write_out():
        with open(OUT / 'verified.jsonl', 'w') as fh:
            for x in recs:
                y = dict(x)
                if not y.get('resolved'):
                    y['patch'] = None  # keep the text of verified patches only
                fh.write(json.dumps(y, default=str) + '\n')

    for ds, n_pass, _ in SOURCES:
        for j in range(n_pass):
            todo = [x for x in recs if x['dataset'] == ds and x.get('pass') == j and x['patch'].strip()]
            left = DEADLINE_S - (time.time() - T0)
            if not todo or left < 20 * 60:
                continue
            tag = ds.split('/')[-1][:24] + f'_p{j}'
            pdir = WORK / 'pp_patches' / tag
            pdir.mkdir(parents=True, exist_ok=True)
            for x in todo:
                p = x['patch'] if x['patch'].endswith('\n') else x['patch'] + '\n'
                (pdir / f'{x["instance_id"]}.patch').write_text(p)
            (WORK / f'ids_{tag}.json').write_text(json.dumps(sorted(x['instance_id'] for x in todo)))
            out_dir = OUT / 'runs' / tag
            budget = int(min(2.5 * 3600, left - 10 * 60))
            rc = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data), '--tag', tag,
                                 '--out', str(out_dir), '--tasks', 'file', '--task-ids-file', str(WORK / f'ids_{tag}.json'),
                                 '--control', 'patches', '--patch-dir', str(pdir), '--uv-task-env', '--concurrency', '4',
                                 '--budget-s', str(budget), '--timeout-seconds', '600']).returncode
            f = out_dir / 'results' / 'task_results.jsonl'
            res = {r['instance_id']: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}
            for x in todo:
                r = res.get(x['instance_id'])
                if r is not None:
                    x.update(resolved=bool(r.get('resolved')), test_exit_code=r.get('test_exit_code'),
                             error=str(r.get('error') or '')[:160] or None)
            done = [x for x in todo if 'resolved' in x]
            stage('pass', dataset=ds, k=j, tasks=len(todo), finished=len(done), resolved=sum(x['resolved'] for x in done),
                  returncode=rc)
            shutil.rmtree(out_dir / 'results' / 'test_outputs', ignore_errors=True)
            write_out()

    # 5. summary
    summary = {}
    for ds, _, _ in SOURCES:
        mine = [x for x in recs if x['dataset'] == ds]
        done = [x for x in mine if 'resolved' in x]
        if not mine:
            continue
        ok_ = [x for x in done if x['resolved']]
        buckets = {}
        for x in done:
            c = x['tool_calls']
            b = 'unknown' if c is None else '<=30' if c <= 30 else '31-60' if c <= 60 else '61-100' if c <= 100 else '>100'
            d = buckets.setdefault(b, {'n': 0, 'resolved': 0})
            d['n'] += 1
            d['resolved'] += x['resolved']
        calls = sorted(x['tool_calls'] for x in ok_ if x['tool_calls'] is not None)
        s = {'trajectories_selected': len(mine), 'verified_runs': len(done), 'resolved': len(ok_),
             'tasks_with_trajectory': len({x['instance_id'] for x in mine}),
             'tasks_with_verified_solution': len({x['instance_id'] for x in ok_}),
             'tasks_with_verified_solution_le30_calls': len({x['instance_id'] for x in ok_
                                                             if x['tool_calls'] is not None and x['tool_calls'] <= 30}),
             'by_tool_calls': buckets, 'resolved_tool_calls_median': calls[len(calls) // 2] if calls else None,
             'models': sorted({str(x['model']) for x in mine})[:8],
             'errors': sorted({x['error'] for x in done if x.get('error')})[:6]}
        if any(x['label'] is not None for x in done):
            s['label_vs_ours'] = {f'label={lab}': {'n': sum(1 for x in done if x['label'] is lab),
                                                    'ours_resolved': sum(1 for x in done if x['label'] is lab and x['resolved'])}
                                  for lab in (True, False)}
        summary[ds] = s
    summary['all'] = {'tasks_with_verified_solution': len({x['instance_id'] for x in recs if x.get('resolved')}),
                      'candidates': len(cand)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    (OUT / 'solved_ids.json').write_text(json.dumps(sorted({x['instance_id'] for x in recs if x.get('resolved')})))
    write_out()
    stage('summary', **summary)
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
