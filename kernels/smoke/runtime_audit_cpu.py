"""CPU-only submission budget and existing-patch audit. No model calls or credentials.

Mounted outputs are historical samples, not hidden-scoring timing measurements.
Queue delay is never treated as execution time. Reports bounds and explicit unknowns.
"""
import ast
import collections
import json
import os
from pathlib import Path
import re
import statistics
import zipfile

TASK_COUNTS = [58, 120]  # @define (120 is the repository's estimate, not a live host guarantee)
RESERVE_SECONDS = 3600  # @define (scenario assumption for setup/model loading/other overhead)
INPUT = Path(os.getenv('AUDIT_INPUT', '/kaggle/input'))
OUT = Path(os.getenv('AUDIT_OUT', '/kaggle/working/runtime_audit'))
OUT.mkdir(parents=True, exist_ok=True)


def protected(path):
    p = Path(path)
    return (p.name in {'conftest.py', 'pytest.ini', '.pytest.ini', 'pyproject.toml', 'tox.ini',
                       'setup.cfg', 'sitecustomize.py', 'usercustomize.py', '_swegemma_stubs.py'}
            or p.suffix == '.pth' or p.name.startswith('test_') or p.name.endswith('_test.py')
            or bool({'tests', 'test', 'testing'} & set(p.parts)))


def patch_paths(text):
    # Parse quoted git paths too; uncertain names are flagged rather than silently accepted.
    import shlex
    paths = []
    for line in text.splitlines():
        if line.startswith('diff --git '):
            try:
                fields = shlex.split(line)
                if len(fields) == 4:
                    paths.extend(x[2:] if x.startswith(('a/', 'b/')) else x for x in fields[2:])
            except ValueError:
                paths.append('<unparsed-path>')
    return sorted(set(paths))


def main():
    report = {'question': 'What runtime headroom and mechanical diff risks do existing artifacts show?',
              'limits': ['No CPU test measures adapter decode speed or reviewer score gain.',
                         'Queue time is excluded; actual scorer timing is required for a calibrated forecast.',
                         '120 tasks and one-hour reserve are assumptions; setup and final verification are separate.',
                         'A reviewer is an additional LLM call, not just the time needed to execute git diff.'],
              'bundles': [], 'traces': [], 'patches': [], 'budget_scenarios': []}
    import yaml
    for f in sorted(INPUT.rglob('submission.zip')):
        try:
            with zipfile.ZipFile(f) as z:
                names = z.namelist()
                configs = [x for x in names if Path(x).name == 'eval_config.yaml']
                for config in configs:
                    if z.getinfo(config).file_size > 100000:
                        raise ValueError('oversized config')
                    cfg = yaml.safe_load(z.read(config)) or {}
                    report['bundles'].append({'artifact': str(f), 'config_path': config,
                                              'evaluation': cfg.get('evaluation', {}),
                                              'adapter_files': [x for x in names if 'adapter' in x.lower()]})
        except Exception as e:
            report['bundles'].append({'artifact': str(f), 'error': str(e)})
    groups = collections.defaultdict(list)
    for f in sorted(INPUT.rglob('trace_*.json')):
        try:
            d = json.loads(f.read_text())
            steps = d.get('steps', [])
            elapsed = [s.get('extra', {}).get('elapsed_s') for s in steps if s.get('extra')]
            elapsed = [x for x in elapsed if isinstance(x, (int, float)) and x >= 0]
            row = {'artifact': str(f), 'instance_id': d.get('instance_id', f.stem.removeprefix('trace_')),
                   'recorded_elapsed_s': max(elapsed) if elapsed else None,
                   'completion_tokens': sum((s.get('metrics') or {}).get('completion_tokens', 0) or 0 for s in steps),
                   'tool_calls': sum(len(s.get('tool_calls') or []) for s in steps),
                   'context': 'Historical trace; concurrency and model identity must be checked before extrapolation.'}
            report['traces'].append(row)
            if elapsed:
                groups[str(f.parent)].append(max(elapsed))
        except Exception as e:
            report['traces'].append({'artifact': str(f), 'error': str(e)})
    report['trace_groups'] = {k: {'n': len(v), 'mean_recorded_s': statistics.mean(v),
                                    'median_recorded_s': statistics.median(v), 'max_recorded_s': max(v)}
                              for k, v in groups.items()}
    for f in sorted(INPUT.rglob('*.patch')):
        text = f.read_text(errors='replace')
        paths = patch_paths(text)
        report['patches'].append({'artifact': str(f), 'bytes': f.stat().st_size, 'paths': paths,
                                 'protected_paths': [p for p in paths if protected(p)],
                                 'scratch_candidates': [p for p in paths if re.search(r'(^|/)(repro|debug|scratch)[^/]*\.py$', p)],
                                 'unparsed_path': '<unparsed-path>' in paths})
    for n in TASK_COUNTS:
        for cap in [3, 4, 5, 6, 7, 8]:
            for verify_s in [3, 30, 120, 240]:
                available = 12 * 3600 - RESERVE_SECONDS - n * verify_s
                upper = n * cap * 60 + RESERVE_SECONDS + n * verify_s
                report['budget_scenarios'].append({
                    'tasks': n, 'agent_cap_minutes': cap, 'assumed_mean_verification_seconds': verify_s,
                    'assumed_other_overhead_seconds': RESERVE_SECONDS, 'cap_envelope_hours': upper / 3600,
                    'fits_if_overhead_assumptions_hold': upper < 12 * 3600,
                    'maximum_mean_agent_seconds': available / n,
                    'note': 'Envelope assumes every task reaches its cap; not an empirical prediction.'})
    report['reviewer_budget_scenarios'] = [
        {'tasks': n, 'additional_seconds_per_task': s, 'additional_minutes': n * s / 60}
        for n in TASK_COUNTS for s in [5, 15, 30, 60]]
    report['patch_summary'] = {'count': len(report['patches']),
                               'with_protected_paths': sum(bool(r['protected_paths']) for r in report['patches']),
                               'with_scratch_candidates': sum(bool(r['scratch_candidates']) for r in report['patches'])}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('question', 'patch_summary', 'trace_groups', 'limits')}, indent=2))


if __name__ == '__main__':
    main()
