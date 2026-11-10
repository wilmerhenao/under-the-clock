"""Backlog diff-to-trajectory feasibility: CPU replay + official verification, no teacher calls.

Gold-guided read/edit plans are privileged training candidates, NEVER evaluation policy.
Mechanically replay a short plan; extract its git diff; regrade through the existing Evaluator.
Does not publish data or train an adapter. Preserve all existing bank outputs.
"""
import ast
import difflib
import glob
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

EVAL_PY = None  # @embed eval_harness.py
MAX_TASKS = 100  # @define
DEADLINE_S = 4 * 3600  # @define
MAX_CALLS = 28  # @define
EXCLUDED_REPOS = {'fastapi/fastapi', 'encode/httpx', 'psf/requests', 'Textualize/rich'}
T0 = time.monotonic()
WORK = Path(os.getenv('DIFF_WORK', '/kaggle/working'))
OUT = WORK / 'diff_replay'
OUT.mkdir(parents=True, exist_ok=True)


def git(root, *args):
    r = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(root), *args],
                       capture_output=True, timeout=60)
    if r.returncode:
        raise ValueError(f'git {args[0]} failed: {r.stderr.decode(errors="replace")[-300:]}')
    return r.stdout.decode('utf-8', errors='strict')


def protected(p):
    path = Path(p)
    return (path.name in {'conftest.py', 'pytest.ini', '.pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg',
                         'sitecustomize.py', 'usercustomize.py', '_swegemma_stubs.py'} or path.suffix == '.pth'
            or path.name.startswith('test_') or path.name.endswith('_test.py')
            or bool({'tests', 'test', 'testing'} & set(path.parts)))


def make_plan(old, new, path):
    """Reverse-order nonoverlapping replacements, each with a unique <=150-line read window."""
    before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
    changes = [op for op in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
               if op[0] != 'equal']
    working, actions = old, []
    for _, i, j, a, b in reversed(changes):
        selected = None
        for pad in range(0, 6):
            start, end = max(0, i - pad), min(len(before), j + pad)
            needle = ''.join(before[start:end])
            replacement = ''.join(before[start:i] + after[a:b] + before[j:end])
            if needle and working.count(needle) == 1 and end - start <= 150 and len(needle) <= 10000:
                selected = start, end, needle, replacement
                break
        if selected is None:
            raise ValueError('no_unique_short_anchor')
        start, end, needle, replacement = selected
        actions.extend([
            {'tool': 'read_file', 'arguments': {'filepath': path, 'start_line': start + 1, 'end_line': end}},
            {'tool': 'edit_file', 'arguments': {'filepath': path, 'old_string': needle, 'new_string': replacement}}])
        working = working.replace(needle, replacement, 1)
    if working != new:
        raise ValueError('plan_does_not_reconstruct_file')
    return actions


def replay_task(row, snapshot):
    with tempfile.TemporaryDirectory(prefix='gold_diff_', dir='/tmp') as tmp:
        dest = Path(tmp)
        with tarfile.open(snapshot) as tf:
            tf.extractall(dest, filter='data')
        candidates = [dest] + [p for p in dest.iterdir() if p.is_dir()]
        roots = [p for p in candidates if (p / '.git').exists()]
        root = roots[0] if roots else (candidates[1] if len(candidates) == 2 else dest)
        if (root / '.git').is_dir():
            shutil.rmtree(root / '.git')
        elif (root / '.git').exists():
            (root / '.git').unlink()
        git(root, 'init', '-q')
        git(root, 'add', '.')
        git(root, '-c', 'user.name=CPU replay', '-c', 'user.email=cpu@local', 'commit', '-qm', 'baseline', '--allow-empty')
        baseline = git(root, 'rev-parse', 'HEAD').strip()
        patch = dest / 'gold.patch'
        patch.write_text(row['patch'])
        # If root==dest, gold.patch is untracked; remove it before extracting the synthesized patch.
        git(root, 'apply', '--check', str(patch))
        git(root, 'apply', str(patch))
        patch.unlink()
        git(root, 'add', '-N', '.')  # include new files so v1 rejects, rather than silently losing them
        paths = git(root, 'diff', '--name-only', '-z').strip('\0').split('\0')
        if not paths or any(not p or protected(p) for p in paths):
            raise ValueError('protected_or_empty_diff')
        states = {}
        for p in paths:
            if not (root / p).is_file() or (root / p).is_symlink():
                raise ValueError('deleted_or_symlink_file')
            old = git(root, 'show', f'{baseline}:{p}')  # v1 excludes newly created files
            new = (root / p).read_text()
            if '\x00' in old or '\x00' in new:
                raise ValueError('binary_file')
            states[p] = old, new
        git(root, 'reset', '--hard', '-q', baseline)
        actions, syntax = [], {}
        for p, (old, new) in states.items():
            plan = make_plan(old, new, p)
            actions.extend(plan)
            working = (root / p).read_text()
            for action in plan:
                if action['tool'] == 'edit_file':
                    a = action['arguments']
                    if working.count(a['old_string']) != 1:
                        raise ValueError('replay_anchor_not_unique')
                    working = working.replace(a['old_string'], a['new_string'], 1)
            if working != new:
                raise ValueError('replay_mismatch')
            (root / p).write_text(working)
            if p.endswith('.py'):
                syntax[p] = {}
                for label, text in [('baseline', old), ('candidate', new)]:
                    try:
                        ast.parse(text)
                        syntax[p][label] = 'ok'
                    except SyntaxError as e:
                        syntax[p][label] = f'{e.msg}:{e.lineno}'
        if len(actions) > MAX_CALLS:
            raise ValueError('over_tool_budget')
        for p, (_, new) in states.items():
            if (root / p).read_text() != new:
                raise ValueError('file_mismatch')
        # Same extraction mechanism as submit_patch; diff is not checked for byte identity to gold.
        git(root, 'add', '-N', '.')
        extracted = git(root, 'diff', '--binary', baseline)
        if not extracted:
            raise ValueError('empty_extracted_patch')
        actions.append({'tool': 'submit_patch', 'arguments': {}})  # free tool call
        return extracted, actions, syntax


def main():
    data = WORK / 'diff_data'
    (data / 'snapshots').mkdir(parents=True, exist_ok=True)
    for name in ['wheels', 'graphs', 'embeddings']:
        (data / name).mkdir(exist_ok=True)
    patches = OUT / 'patches'
    patches.mkdir(exist_ok=True)
    trusted = set()
    for f in glob.glob('/kaggle/input/**/harness_trusted_ids.json', recursive=True):
        trusted.update(json.loads(Path(f).read_text()))
    if not trusted:
        raise RuntimeError('Missing trusted IDs: mount g4dev-rebench-harness-bank-cpu.')
    candidates = {}
    for f in sorted(glob.glob('/kaggle/input/**/rebench/tasks_verified.jsonl', recursive=True)):
        for row in map(json.loads, Path(f).read_text().splitlines()):
            iid = row['instance_id']
            snapshot = Path(f).parent / 'snapshots' / f'{iid.replace("/", "__")}.tgz'
            if iid in trusted and row.get('repo') not in EXCLUDED_REPOS and snapshot.is_file():
                candidates.setdefault(iid, (row, snapshot))
    todo = sorted(candidates.values(), key=lambda x: x[0]['instance_id'])
    random.Random(20261009).shuffle(todo)
    accepted, rejected = [], []
    with open(OUT / 'plans.jsonl', 'w') as plans:
        for row, snapshot in todo[:MAX_TASKS]:
            if time.monotonic() - T0 > min(DEADLINE_S / 3, 3600):
                break
            try:
                patch, actions, syntax = replay_task(row, snapshot)
                iid = row['instance_id']
                (patches / f'{iid}.patch').write_text(patch)
                (data / 'snapshots' / f'{iid}.tgz').symlink_to(snapshot)
                plans.write(json.dumps({'instance_id': iid, 'privileged_gold_guided': True,
                                        'verification': 'pending', 'actions': actions, 'syntax': syntax}) + '\n')
                plans.flush()
                accepted.append(row)
            except Exception as e:
                rejected.append({'instance_id': row['instance_id'], 'reason': str(e)[:300]})
    report = {'question': 'Can gold diffs produce short read/edit plans that survive official patch verification?',
              'candidates': len(todo), 'attempted': len(accepted) + len(rejected), 'plans': len(accepted),
              'rejected': rejected, 'verification': 'not_run',
              'limitations': ['Gold-guided plans do not measure autonomous repair accuracy.',
                              'Plans are not rendered SFT examples; no model observations are invented.',
                              'v1 excludes additions/deletions, protected paths, and nonunique/long edit anchors.',
                              'Syntax results use this CPU Python version, not necessarily the task Python.']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    if not accepted:
        raise RuntimeError('No replayable candidates; see report.json.')
    (data / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in accepted))
    comp = next(Path('/kaggle/input').rglob('sample_submission/agent.yaml'), None)
    if comp is None:
        raise RuntimeError('Competition sample_submission missing.')
    shutil.copytree(comp.parent, data / 'sample_submission', dirs_exist_ok=True)
    wheels = next(Path('/kaggle/input').rglob('swegemma-*.whl'), None)
    if wheels is None:
        raise RuntimeError('Harness wheelhouse missing.')
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    supported = set(sys_tags())
    compatible = [str(w) for w in wheels.parent.glob('*.whl') if 'cutlass' not in w.name.lower()
                  and supported.intersection(parse_wheel_filename(w.name)[3])]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *compatible], check=True, timeout=600)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True, timeout=300)
    (WORK / 'eval_harness.py').write_text(EVAL_PY)
    remaining = DEADLINE_S - (time.monotonic() - T0) - 120
    if remaining <= 0:
        raise RuntimeError('Deadline exhausted before verification.')
    rc = subprocess.run([sys.executable, str(WORK / 'eval_harness.py'), '--data', str(data), '--tag', 'diff-replay',
                         '--out', str(OUT / 'verification'), '--tasks', 'all', '--control', 'patches',
                         '--patch-dir', str(patches), '--uv-task-env', '--time-verify', '--concurrency', '2',
                         '--timeout-seconds', '240', '--budget-s', str(int(remaining))],
                        timeout=remaining + 60).returncode
    results = OUT / 'verification' / 'results' / 'task_results.jsonl'
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()] if results.exists() else []
    report.update(verification='finished' if rc == 0 else 'failed_or_partial', returncode=rc,
                  finished=len(rows), resolved=sum(bool(r.get('resolved')) for r in rows),
                  resolved_ids=[r['instance_id'] for r in rows if r.get('resolved')],
                  minutes=(time.monotonic() - T0) / 60)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        f = OUT / 'report.json'
        report = json.loads(f.read_text()) if f.exists() else {}
        report.update(fatal=f'{type(e).__name__}: {e}', minutes=(time.monotonic() - T0) / 60)
        f.write_text(json.dumps(report, indent=2))
        raise
