"""Label controls on CPU (no accelerator, no GPU quota): for every public task, verify
  none: an empty patch must FAIL the task's tests (otherwise the task is solved "for free")
  gold: the reference patch must PASS (otherwise the local environment can't verify this task)
with the official harness's Phase-2 verification. label_trusted = gold passes and none fails.
Outputs: /kaggle/working/controls/{none,gold}/ + controls.csv + report.json.
"""
import csv
import glob
import json
import re
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

EVAL_PY = None  # @embed eval_harness.py
VARIANT_ARGS = []  # @define
CONTROLS = ['none', 'gold']  # @define
PATCH_GLOB = ''  # @define
CONC = 0  # @define (0 = one per CPU; 1 = sequential like the scorer, for timing)
EXTRA_ARGS = []  # @define (passed to eval_harness as they are)
REPOGRAPH = ''  # @define ('priv_eval' / 'priv_train': controls on debmalya84/repograph-verified-swe-tasks, per-repo wheelhouses)
REPOGRAPH_IDS = []  # @define (non-empty: only these instance ids, e.g. the tasks a previous run could not label)
BERTH = ''  # @define ('core' / 'extended': controls on jayp2005/berth-swe-tasks, harness_compatible rows, its wheels/ as /wheels)

T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'controls'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'machine': 'cpu', 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}


def save():
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    save()
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:3000]}', flush=True)


def find(pattern_tail, depth=6):
    for d in range(1, depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * (d - 1) + pattern_tail))
        if hits:
            return Path(hits[0])
    return None


os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
try:
    wh = find('swegemma-*.whl').parent
    # same wheel set as the official notebook (the base image supplies the rest); GPU-only wheels are inert here
    tmp_whl = Path('/tmp/wheelhouse')
    tmp_whl.mkdir(parents=True, exist_ok=True)
    for w in wh.glob('*.whl'):
        if 'cutlass' in w.name.lower():
            continue
        name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
        if not (tmp_whl / name).exists():
            os.symlink(w, tmp_whl / name)
    # install only wheels this interpreter/platform supports: CPU sessions can land on images where the GPU-only
    # wheels (apache_tvm_ffi, ...) are rejected, which failed the whole install (reverify-tpu v1, 2026-10-02)
    import platform
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:  # pip always vendors packaging
        from pip._vendor.packaging.tags import sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
    supported = {str(t) for t in sys_tags()}
    ok, skipped = [], []
    for w in sorted(tmp_whl.glob('*.whl')):
        tags = parse_wheel_filename(w.name)[3]
        (ok if any(str(t) in supported for t in tags) else skipped).append(w)
    stage('platform', python=sys.version.split()[0], machine=platform.machine(), libc=platform.libc_ver(),
          wheels_ok=len(ok), skipped=[w.name.split('-')[0] for w in skipped])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *map(str, ok)], check=True)
    DATA = (find('competitions/gemma-4-developer-agent/tasks.jsonl') or find('tasks.jsonl')).parent
    if REPOGRAPH:  # competition-layout data dir over the repograph split; each repo's wheelhouse is its /wheels
        rg = Path(sorted(glob.glob(f'/kaggle/input/**/repograph-verified-swe-tasks/{REPOGRAPH}/tasks.jsonl',
                                   recursive=True))[0]).parent
        rw = Path(sorted(glob.glob('/kaggle/input/**/repograph-verified-swe-tasks-wheels', recursive=True))[0])
        comp, DATA = DATA, WORK / 'rgdata'
        for d in ('snapshots', 'graphs', 'embeddings'):
            (DATA / d).mkdir(parents=True, exist_ok=True)
        for d in ('wheels', 'docker', 'sandbox', 'sample_submission'):
            if not (DATA / d).exists():
                os.symlink(comp / d, DATA / d)
        rows = [json.loads(x) for x in (rg / 'tasks.jsonl').read_text().splitlines() if x.strip()]
        for r in rows:
            os.symlink(rg / 'snapshots' / f"{r['instance_id']}.tgz", DATA / 'snapshots' / f"{r['instance_id']}.tgz")
        for g in (rg / 'graphs').glob('*.json'):
            os.symlink(g, DATA / 'graphs' / g.name)
        (DATA / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        prefixes = sorted({r['instance_id'].rsplit('_', 1)[0] for r in rows})
        no_wheels = [x for x in prefixes if not (rw / x).is_dir()]
        VARIANT_ARGS = [*VARIANT_ARGS, '--repo-wheels', str(rw)]
        if REPOGRAPH_IDS:
            (WORK / 'repograph_ids.json').write_text(json.dumps(REPOGRAPH_IDS))
            EXTRA_ARGS = [*EXTRA_ARGS, '--tasks', 'file', '--task-ids-file', str(WORK / 'repograph_ids.json')]
        stage('repograph_data', split=REPOGRAPH, tasks=len(rows), repos=len(prefixes), no_wheel_dir=no_wheels,
              graphs=len(list((DATA / 'graphs').glob('*.json'))), wheel_root=str(rw))
    if BERTH:  # competition-layout data dir over a Berth split (B142); one shared wheelhouse, official-env dedup
        bt = Path(sorted(glob.glob('/kaggle/input/**/berth-swe-tasks/tasks.jsonl', recursive=True))[0]).parent
        comp, DATA = DATA, WORK / 'btdata'
        for d in ('snapshots', 'graphs', 'embeddings'):
            (DATA / d).mkdir(parents=True, exist_ok=True)
        os.symlink(bt / 'wheels', DATA / 'wheels')
        for d in ('docker', 'sandbox', 'sample_submission'):
            if (comp / d).exists() and not (DATA / d).exists():
                os.symlink(comp / d, DATA / d)
        cols = list(json.loads((comp / 'tasks.jsonl').read_text().splitlines()[0]))
        allrows = [json.loads(x) for x in (bt / f'{BERTH}.jsonl').read_text().splitlines() if x.strip()]
        rows = [r for r in allrows if r.get('harness_compatible')]
        for r in rows:
            os.symlink(bt / 'snapshots' / f"{r['instance_id']}.tgz", DATA / 'snapshots' / f"{r['instance_id']}.tgz")
        for g in (bt / 'graphs').glob('*.json'):
            os.symlink(g, DATA / 'graphs' / g.name)
        for g in (bt / 'embeddings').glob('*.npz'):
            os.symlink(g, DATA / 'embeddings' / g.name)
        (DATA / 'tasks.jsonl').write_text(''.join(json.dumps({k: r.get(k) for k in cols}) + '\n' for r in rows))
        stage('berth_data', split=BERTH, rows=len(allrows), compatible=len(rows), cols=cols,
              post_cutoff=sum(1 for r in rows if r.get('post_cutoff')), repos=sorted({r['repo'] for r in rows}),
              wheels=len(list((bt / 'wheels').glob('*.whl'))), graphs=len(list((DATA / 'graphs').glob('*.json'))))
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    extra = find('dirty_equals-*.whl')
    kit = find('py313.tar.gz*')
    # our per-task wheelhouse by dataset name (a wheel-name lookup hits the competition's wheels/ first)
    hits = sorted(glob.glob('/kaggle/input/**/g4dev-py313-wheels/**/pytest_timeout-*.whl', recursive=True))
    wh313 = str(Path(hits[0]).parent) if hits else None  # pip --find-links does not recurse
    if '@wh313' in VARIANT_ARGS and not wh313:
        raise RuntimeError('py313 wheelhouse (g4dev-py313-wheels with pytest-timeout) not found under /kaggle/input')
    variant = [str(extra.parent) if x == '@extra' else str(kit.parent) if x == '@kit' else wh313 if x == '@wh313' else x for x in VARIANT_ARGS]
    stage('ready', python=sys.version.split()[0], cpus=os.cpu_count(), data=str(DATA), variant=variant)

    conc = str(CONC or max(2, (os.cpu_count() or 4)))
    patch_dirs = sorted(Path(p) for p in glob.glob(PATCH_GLOB, recursive=True)) if PATCH_GLOB else []
    stage('patch_dirs', dirs=[str(d) for d in patch_dirs])
    jobs = [(c, None) for c in CONTROLS if c != 'patches']
    jobs += [('patches', d) for d in patch_dirs] if 'patches' in CONTROLS else []
    for control, pdir in jobs:
        out_dir = OUT / (control if pdir is None else 'reverify_' + pdir.parts[-4] + '_' + pdir.parts[-3])
        rc = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(DATA), '--tag', control,
                             '--out', str(out_dir), '--tasks', 'all', '--control', control,
                             '--concurrency', conc, '--budget-s', str(5 * 3600), *variant, *EXTRA_ARGS,
                             *(['--patch-dir', str(pdir)] if pdir else [])]).returncode
        f = out_dir / 'summary_smoke.json'
        summary = json.loads(f.read_text()) if f.exists() else None
        if summary:
            summary.pop('run_ids', None)
        stage(f'{out_dir.name}_done', returncode=rc, summary=summary)
        vf, rf = out_dir / 'verify_times.jsonl', out_dir / 'results' / 'task_results.jsonl'
        if vf.exists():  # wall time of the verification pytest (the scorer runs it with the bundle's timeout_seconds)
            vt = [json.loads(x) for x in vf.read_text().splitlines() if x.strip()]
            res = [json.loads(x) for x in rf.read_text().splitlines() if x.strip()] if rf.exists() else []
            if not any(r.get('task') for r in vt) and len(vt) == len(res) and conc == '1':
                for r, t in zip(vt, res):  # snapshot HEAD is not the task's base_commit: sequential run, same order
                    r['task'] = t['instance_id']
            ok = {t['instance_id'] for t in res if t.get('resolved')}
            pairs = sorted((r['secs'], r.get('task')) for r in vt)
            secs = [x for x, _ in pairs]
            by_repo = {}
            for x, i in pairs:
                by_repo.setdefault(str(i).rsplit('_', 1)[0], []).append(x)
            stage(f'{out_dir.name}_verify_timing', rows=len(vt), conc=conc, mapped=sum(1 for r in vt if r.get('task')),
                  median_s=secs[len(secs) // 2] if secs else None, p90_s=secs[int(len(secs) * 0.9)] if secs else None,
                  max_s=secs[-1] if secs else None, over_30=sum(x > 30 for x in secs), over_60=sum(x > 60 for x in secs),
                  over_120=sum(x > 120 for x in secs), over_240=sum(x > 240 for x in secs),
                  resolved_over_60=sum(1 for x, i in pairs if x > 60 and i in ok),
                  by_repo={k: {'n': len(v), 'median': sorted(v)[len(v) // 2], 'max': max(v)} for k, v in by_repo.items()},
                  slowest=pairs[-20:])
    if 'none' not in CONTROLS:
        raise SystemExit(0)

    def load(control):
        f = OUT / control / 'results' / 'task_results.jsonl'
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
        return {r['instance_id']: r for r in rows}

    none, gold = load('none'), load('gold')
    ids = sorted(set(none) | set(gold))
    with open(OUT / 'controls.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['instance_id', 'none_resolved', 'gold_resolved', 'label_trusted', 'none_exit', 'gold_exit',
                    'gold_error', 'gold_hint'])

        def hint(i):  # first error-looking line of the gold test log (why a reference patch fails)
            f = OUT / 'gold' / 'results' / 'test_outputs' / f'{i}.log'
            if not f.exists():
                return ''
            for line in f.read_text(errors='replace').splitlines():
                if re.search(r'Error|No module named|^FAILED|^ERROR', line):
                    return line.strip()[:200]
            return ''
        for i in ids:
            n, g = none.get(i, {}), gold.get(i, {})
            trusted = bool(g.get('resolved')) and n.get('resolved') is False
            w.writerow([i, n.get('resolved'), g.get('resolved'), trusted, n.get('test_exit_code'),
                        g.get('test_exit_code'), str(g.get('error') or '')[:200],
                        '' if g.get('resolved') else hint(i)])
    trusted = sum(1 for i in ids if gold.get(i, {}).get('resolved') and none.get(i, {}).get('resolved') is False)
    stage('controls_written', n=len(ids), trusted=trusted,
          free_pass=[i for i in ids if none.get(i, {}).get('resolved')],
          gold_fail=[i for i in ids if not gold.get(i, {}).get('resolved')])
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-4000:], tb=traceback.format_exc()[-3000:])
finally:
    REPORT['total_min'] = round((time.time() - T0) / 60, 1)
    save()
