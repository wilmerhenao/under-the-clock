"""Scout: can the short Claude trajectories of R2E-Gym and SWE-Gym be replayed on real checkouts, like SWE-smith?

Their SFT datasets (mirrored privately on Kaggle) hold only messages: no task id, repo or commit. Here every trajectory is
matched to its task by the issue text (R2E-Gym `<github_issue>`, SWE-Gym `<pr_description>`) against the task datasets on
the Hub (R2E-Gym-V1 / R2E-Gym-Subset: repo_name, commit_hash, docker_image; SWE-Gym: repo, base_commit). For a sample of
matched trajectories the teacher's first file view is compared with the file in a GitHub checkout at the candidate
revisions (R2E-Gym: the commit and its parent; SWE-Gym: base_commit), which says which repository state the teacher saw.
Also measured: tool-call syntax, calls per trajectory, first-edit position. Outputs: /kaggle/working/match_scout/
{mapping.jsonl, report.json}.
"""
import collections
import glob
import json
import re
import shutil
import subprocess
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

VALIDATE_N = 30  # @define (matched trajectories per dataset whose first file view is checked against GitHub)
ONLY = []  # @define (subset of SOURCES keys to run; empty = all)
ALL_SPLITS = False  # @define (also read non-train splits of the task datasets, e.g. R2E-Gym-Lite dev_*)
T0 = time.time()
WORK = Path('/kaggle/working')
OUT = WORK / 'match_scout'
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/ms')
TMP.mkdir(parents=True, exist_ok=True)
HUB = 'https://huggingface.co'
REPORT = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
R2E_REPOS = {'pandas': 'pandas-dev/pandas', 'numpy': 'numpy/numpy', 'pillow': 'python-pillow/Pillow',
             'aiohttp': 'aio-libs/aiohttp', 'tornado': 'tornadoweb/tornado', 'coveragepy': 'nedbat/coveragepy',
             'datalad': 'datalad/datalad', 'pyramid': 'Pylons/pyramid', 'orange3': 'biolab/orange3', 'scrapy': 'scrapy/scrapy',
             'sympy': 'sympy/sympy', 'twisted': 'twisted/twisted', 'requests': 'psf/requests', 'flask': 'pallets/flask',
             'django': 'django/django', 'matplotlib': 'matplotlib/matplotlib', 'astropy': 'astropy/astropy',
             'moto': 'getmoto/moto', 'pyarrow': 'apache/arrow', 'xarray': 'pydata/xarray'}
SOURCES = {  # SFT mirror (Kaggle dataset dir name) -> (Hub task datasets, issue tag, result prefix)
    'r2egym-sft-trajectories': (['R2E-Gym/R2E-Gym-V1', 'R2E-Gym/R2E-Gym-Subset', 'R2E-Gym/R2E-Gym-Lite'], 'github_issue',
                                'Execution output of ['),
    'swe-gym-openhands-sft-trajectories': (['SWE-Gym/SWE-Gym'], 'pr_description', 'EXECUTION RESULT of ['),
}


def stage(name, **kv):
    REPORT['stages'].append({'stage': name, 't': round(time.time() - T0, 1), **kv})
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2, default=str))
    print(f'[{(time.time() - T0) / 60:6.1f} min] {name} {json.dumps(kv, default=str)[:2500]}', flush=True)


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-match-scout'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep((40 if '429' in str(e) else 4) * (k + 1))


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'g4dev-match-scout'})
            with urllib.request.urlopen(req, timeout=900) as r, open(dest, 'wb') as fh:
                shutil.copyfileobj(r, fh, 1 << 22)
            return dest
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def sh(cmd, cwd=None, timeout=900):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors='replace')
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'timeout'


def norm(s):
    """Issue text for matching: R2E-Gym's [ISSUE] markers and markdown emphasis removed, whitespace collapsed, lower case."""
    s = re.sub(r'\[/?ISSUE\]', ' ', s or '').replace('*', '')
    return re.sub(r'\s+', ' ', s).strip().lower()


CALL_RE = re.compile(r'<function=([\w.-]+)>(.*?)</function>', re.S)
PARAM_RE = re.compile(r'<parameter=([\w.-]+)>(.*?)</parameter>', re.S)


def parse(msgs, tag, prefix):
    """Issue text, repo root, and the (tool, params, observation) sequence of a text-embedded-call trajectory."""
    user0 = next((m['content'] for m in msgs if m['role'] == 'user'), '')
    mm = re.search(rf'<{tag}>(.*?)</{tag}>', user0, re.S)
    issue = mm.group(1) if mm else ''
    mr = re.search(r'<uploaded_files>\s*(\S+)', user0)
    root = mr.group(1).rstrip('/') if mr else '/testbed'
    calls = []
    for i, m in enumerate(msgs):
        if m['role'] != 'assistant':
            continue
        for fm in CALL_RE.finditer(m['content'] or ''):
            params = {k: v.strip('\n') for k, v in PARAM_RE.findall(fm.group(2))}
            obs = ''
            if i + 1 < len(msgs) and msgs[i + 1]['role'] in ('user', 'tool') and (msgs[i + 1]['content'] or '').startswith(prefix):
                obs = msgs[i + 1]['content']
            calls.append((fm.group(1), params, obs))
    return issue, root, calls


def file_views(calls, root):
    """[(relative path, {line_no: text})] of file views whose output shows cat -n lines, in order (max 6)."""
    views = []
    for name, p, obs in calls:
        if name in ('file_editor', 'str_replace_editor') and p.get('command') == 'view' and p.get('path') \
                and 'result of running `cat -n`' in obs:  # full listings only (R2E-Gym's condensed views fold code)
            lines = {int(a): b for a, b in re.findall(r'^\s*(\d+)(?:\t| +)(.*)$', obs, re.M)}
            path = p['path']
            rel = path[len(root) + 1:] if path.startswith(root + '/') else None
            if rel and len(lines) >= 5 and rel not in [v[0] for v in views]:
                views.append((rel, lines))
    return views[:6]


def first_file_view(calls, root):
    """(relative path, {line_no: text}) of the first file view whose output shows cat -n lines."""
    for name, p, obs in calls:
        if name in ('file_editor', 'str_replace_editor') and p.get('command') == 'view' and p.get('path'):
            lines = {int(a): b for a, b in re.findall(r'^\s*(\d+)(?:\t| +)(.*)$', obs, re.M)}
            if len(lines) >= 5:
                path = p['path']
                rel = path[len(root) + 1:] if path.startswith(root + '/') else None
                if rel:
                    return rel, lines
    return None, None


try:
    import pyarrow.parquet as pq

    # 1. task datasets from the Hub: normalized problem statement -> task records
    tasks = {}
    for src, (hub_sets, _, _) in [(k, v) for k, v in SOURCES.items() if not ONLY or k in ONLY]:
        recs = []
        for ds in hub_sets:
            try:
                tree = get(f'{HUB}/api/datasets/{ds}/tree/refs%2Fconvert%2Fparquet?recursive=true')
                files = sorted(t['path'] for t in tree if t.get('path', '').endswith('.parquet') and (ALL_SPLITS or '/train' in t['path']))
                for path in files:
                    dest = download(f'{HUB}/datasets/{ds}/resolve/refs%2Fconvert%2Fparquet/{urllib.parse.quote(path)}',
                                    TMP / 'tasks.parquet')
                    cols = pq.ParquetFile(dest).schema_arrow.names
                    keep = [c for c in ('problem_statement', 'repo_name', 'commit_hash', 'docker_image', 'repo',
                                        'base_commit', 'instance_id', 'version') if c in cols]
                    for r in pq.read_table(dest, columns=keep).to_pylist():
                        r['hub'] = ds
                        recs.append(r)
            except Exception as e:  # noqa: BLE001
                stage('task_source_failed', dataset=ds, error=f'{type(e).__name__}: {e}'[:300])
        index = collections.defaultdict(list)
        for r in recs:
            index[norm(r.get('problem_statement'))[:150]].append(r)
        tasks[src] = (recs, index)
        stage('tasks', source=src, rows=len(recs), distinct_statements=len(index),
              repos=collections.Counter(r.get('repo_name') or r.get('repo') for r in recs).most_common(25))

    # 2. SFT trajectories (Kaggle mirrors) -> parse + match
    summary = {}
    mapping = open(OUT / 'mapping.jsonl', 'w')
    validate = {}
    for src, (_, tag, prefix) in [(k, v) for k, v in SOURCES.items() if not ONLY or k in ONLY]:
        files = sorted(glob.glob(f'/kaggle/input/**/{src}/**/*.parquet', recursive=True)) or \
            sorted(glob.glob(f'/kaggle/input/{src}/*.parquet'))
        recs, index = tasks[src]
        c = collections.Counter()
        calls_n, first_edit, tools = [], [], collections.Counter()
        samples = []
        row_no = 0
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=['messages']):
                for r in b.to_pylist():
                    row_no += 1
                    msgs = r['messages'] if isinstance(r['messages'], list) else json.loads(r['messages'])
                    issue, root, calls = parse(msgs, tag, prefix)
                    c['rows'] += 1
                    n_issue = norm(issue)
                    cands = [t for t in index.get(n_issue[:150], []) if norm(t.get('problem_statement')) == n_issue] or \
                        index.get(n_issue[:150], [])
                    edits = [k for k, (nm, p, _) in enumerate(calls)
                             if nm in ('file_editor', 'str_replace_editor') and p.get('command') in ('str_replace', 'create', 'insert')]
                    calls_n.append(len(calls))
                    if edits:
                        first_edit.append(edits[0] + 1)
                    for nm, p, _ in calls:
                        tools[nm + (':' + p.get('command', '') if 'editor' in nm else '')] += 1
                    keys = sorted({t.get('docker_image') or t.get('instance_id') for t in cands})
                    commits = sorted({t.get('commit_hash') or t.get('base_commit') for t in cands})
                    repo = (cands[0].get('repo') or cands[0].get('repo_name')) if cands else None
                    c['matched'] += bool(cands)
                    c['unique_commit'] += len(commits) == 1
                    c['no_issue'] += not issue
                    rec = {'source': src, 'row': row_no - 1, 'matched': bool(cands), 'n_candidates': len(cands),
                           'task_keys': keys[:5], 'repo': repo, 'commits': commits[:5], 'calls': len(calls),
                           'first_edit': edits[0] + 1 if edits else None, 'root': root}
                    mapping.write(json.dumps(rec) + '\n')
                    if len(commits) == 1 and len(samples) < 3 * VALIDATE_N:
                        views = file_views(calls, root)
                        if views:
                            samples.append((rec, views))
                    if len(c) and c['rows'] <= 2:
                        REPORT.setdefault('call_syntax_examples', {}).setdefault(src, []).append(
                            [(nm, sorted(p)) for nm, p, _ in calls[:6]])
        validate[src] = samples
        summary[src] = {**c, 'calls': {f'p{q}': sorted(calls_n)[int(len(calls_n) * q / 100)] for q in (10, 50, 90)} if calls_n else {},
                        'share_le28_calls': round(sum(x <= 28 for x in calls_n) / max(1, len(calls_n)), 3),
                        'first_edit': {f'p{q}': sorted(first_edit)[int(len(first_edit) * q / 100)] for q in (10, 50, 90)} if first_edit else {},
                        'tools': tools.most_common(15)}
        stage('matched', source=src, **summary[src])
    mapping.close()

    # 3. which repository state did the teacher see? first file view vs GitHub at the candidate revisions
    val = {}
    for src, samples in validate.items():
        res = collections.Counter()
        detail = []
        random_order = sorted(samples, key=lambda s: hash(s[0]['commits'][0]) % 1000)[:VALIDATE_N]
        for rec, views in random_order:
            if time.time() - T0 > 3.5 * 3600:
                break
            repo = rec['repo'] or ''
            slug = repo if '/' in repo else R2E_REPOS.get(repo.lower())
            if not slug:
                res['unknown_repo'] += 1
                detail.append({'repo': repo, 'result': 'unknown_repo'})
                continue
            d = TMP / slug.replace('/', '__')
            if not d.exists():
                rc, _, err = sh(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', f'https://github.com/{slug}.git', str(d)],
                                timeout=1800)
                if rc != 0:
                    res['clone_failed'] += 1
                    detail.append({'repo': slug, 'result': 'clone_failed', 'err': err[-200:]})
                    continue
            commit = rec['commits'][0]
            refs = [commit, f'{commit}^'] if 'r2egym' in src else [commit]
            # prefer a viewed file the commit changes: only there do the candidate revisions differ
            changed = set(sh(['git', '-C', str(d), 'show', '--name-only', '--format=', commit], timeout=300)[1].split())
            rel, lines = next(((r_, l_) for r_, l_ in views if r_ in changed), views[0])
            out = {}
            for ref in refs:
                rc, txt, _ = sh(['git', '-C', str(d), 'show', f'{ref}:{rel}'], timeout=300)
                if rc != 0:
                    out[ref] = 'missing'
                    continue
                file_lines = txt.splitlines()
                shown = sorted(lines)[:40]
                same = sum(1 for k in shown if k - 1 < len(file_lines) and file_lines[k - 1].strip() == lines[k].strip())
                out[ref] = round(same / max(1, len(shown)), 3)
            exact = [ref for ref, sc in out.items() if sc == 1.0]
            verdict = ('both_equal' if len(exact) > 1 else 'commit' if exact == [commit] else 'parent' if exact else 'no_exact_match')
            res[verdict] += 1
            detail.append({'repo': slug, 'file': rel, 'changed_by_commit': rel in changed, 'scores': out, 'result': verdict})
        val[src] = {'counts': dict(res), 'detail': detail[:VALIDATE_N]}
        stage('validated', source=src, counts=dict(res))
    REPORT['summary'] = summary
    REPORT['validation'] = val
    stage('done', minutes=round((time.time() - T0) / 60, 1))
except Exception as e:  # noqa: BLE001
    stage('fatal', error=f'{type(e).__name__}: {e}'[-2000:], tb=traceback.format_exc()[-2500:])
