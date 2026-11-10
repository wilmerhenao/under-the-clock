"""Rank the repository's source definitions by how well they match an issue text.

Parses every non-test .py file with ast (async functions included, current working tree), scores each
function/class by overlap with the identifiers, quoted strings and paths in the query (rare terms count more),
and prints: the source packages, the best-matching definitions with line ranges, the files that hold them, and
the existing test files that mention the top symbols. Read-only; output stays under a few thousand characters.
"""
import argparse
import ast
import math
import os
import re
import sys
from collections import Counter, defaultdict

SKIP = {'.git', '.venv', 'venv', 'env', 'build', 'dist', 'node_modules', '__pycache__', '.tox', '.eggs',
        'site-packages', '.mypy_cache', '.pytest_cache'}
SECONDARY = ('docs/', 'docs_src/', 'examples/', 'scripts/', 'benchmarks/', 'tools/')
STOP = set('''the and for with that this from into when then than are was were has have had not but you your our
its use used uses using add adds added fix fixes fixed issue bug feature support allow allows should would could
can will does doing done make makes made new old one two get set none true false self cls def class return
returns import pass raise value values type types like also only just more some any all each other same such
please thanks thank example examples description summary checklist pull request changes change test tests'''.split())
WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
PART = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+')


def workspace():
    for cand in (os.environ.get('SWE_WORKSPACE'), os.environ.get('PWD'), '/workspace', os.getcwd()):
        if cand and os.path.isdir(os.path.join(cand, '.git')):
            return cand
    return '/workspace'


def is_test(rel):
    low = rel.lower()
    name = low.rsplit('/', 1)[-1]
    return bool(re.search(r'(^|/)(tests?|testing)/', low)) or name.startswith('test_') or name.endswith('_test.py') \
        or name == 'conftest.py'


def split_ident(word):
    parts = [p.lower() for p in PART.findall(word)] + [p.lower() for p in word.split('_') if p]
    return [p for p in parts if len(p) > 2 and p not in STOP]


def terms(text):
    out = []
    for w in WORD.findall(text):
        low = w.lower()
        if len(low) > 2 and low not in STOP:
            out.append(low)
        if '_' in w or any(c.isupper() for c in w[1:]):
            out.extend(split_ident(w))
    return out


def definitions(path, rel):
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
        tree = ast.parse(src)
    except Exception:  # noqa: BLE001
        return []
    lines = src.splitlines()
    found = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = prefix + child.name
                end = getattr(child, 'end_lineno', child.lineno)
                if isinstance(child, ast.ClassDef):
                    kind, sig = 'class', ''
                else:
                    kind = 'async def' if isinstance(child, ast.AsyncFunctionDef) else 'def'
                    names = [a.arg for a in child.args.posonlyargs + child.args.args + child.args.kwonlyargs]
                    sig = '(' + ', '.join(n for n in names if n not in ('self', 'cls'))[:70] + ')'
                body = '\n'.join(lines[child.lineno - 1:min(end, child.lineno + 300)])
                idents = []
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Name):
                        idents.append(sub.id)
                    elif isinstance(sub, ast.Attribute):
                        idents.append(sub.attr)
                    elif isinstance(sub, ast.arg):
                        idents.append(sub.arg)
                    elif isinstance(sub, ast.keyword) and sub.arg:
                        idents.append(sub.arg)
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        idents.append(sub.name)
                    elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and len(sub.value) <= 60:
                        idents.append(sub.value)  # short strings: messages, keys, option names
                found.append((rel, child.lineno, end, kind, qual, sig, body, ' '.join(idents)))
                if isinstance(child, ast.ClassDef):
                    visit(child, qual + '.')
    visit(tree, '')
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', default='')
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('rest', nargs='*')
    opts, extra = ap.parse_known_args()
    query = ' '.join([opts.query, *opts.rest, *[e for e in extra if not e.startswith('--')]]).strip()
    if not query:
        print('Pass the issue title plus its key identifiers, messages and file names as args query.')
        return
    root = workspace()
    q_terms = Counter(terms(query))
    exact = {w for w in WORD.findall(query) if len(w) > 3 and ('_' in w or any(c.isupper() for c in w[1:]))}
    exact |= {p.split('.')[-1] for p in re.findall(r'`([A-Za-z_][\w.]*)`', query)}
    quoted = [s.strip() for s in re.findall(r'["\']([^"\'\n]{6,80})["\']', query)]
    paths = {p.strip('./') for p in re.findall(r'[\w./-]+\.py\b', query)}

    defs, packages = [], Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not d.startswith('.'))
        for fname in filenames:
            if not fname.endswith('.py') or fname.startswith('.'):  # skips the harness's .adk_exec_* wrapper
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, root).replace(os.sep, '/')
            if is_test(rel) or os.path.getsize(path) > 500_000:
                continue
            top = rel.split('/')[0] if '/' in rel else '(root)'
            packages[top] += 1
            defs.extend(definitions(path, rel))
    if not defs:
        print('No Python definitions found under', root)
        return

    docs = []
    df = Counter()
    lengths = []
    for rel, start, end, kind, qual, sig, body, idents in defs:
        bag = Counter(terms(qual) * 3 + terms(rel) * 2 + terms(idents))
        docs.append(bag)
        lengths.append(sum(bag.values()))
        df.update(bag.keys())
    n = len(docs)
    avg_len = sum(lengths) / n
    wants_docs = bool(re.search(r'\b(docs?|tutorial|documentation|docs_src)\b', query, re.I))
    scored = []
    for bag, length, (rel, start, end, kind, qual, sig, body, idents) in zip(docs, lengths, defs):
        s = 0.0
        norm = 0.25 + 0.75 * length / avg_len  # BM25 length normalization: long functions match everything
        for t, qf in q_terms.items():
            tf = bag.get(t, 0)
            if tf:
                idf = math.log(1 + n / (1 + df[t]))
                s += idf * min(qf, 3) * tf * 2.2 / (tf + 1.2 * norm)
        short = qual.rsplit('.', 1)[-1]
        if short in exact or qual in exact:
            s += 15.0
        elif any(e in body for e in exact):
            s += 4.0
        s += sum(8.0 for q in quoted if q in body)
        if rel in paths or any(rel.endswith(p) for p in paths):
            s += 10.0
        if rel.startswith(SECONDARY) and not wants_docs:
            s *= 0.5
        scored.append((s, rel, start, end, kind, qual, sig))
    scored.sort(key=lambda x: -x[0])

    print('Source packages (non-test .py files): ' +
          ', '.join(f'{p}/ ({c})' for p, c in packages.most_common(8)))
    print(f'Best-matching definitions (score  path:lines  kind name):')
    shown = 0
    for s, rel, start, end, kind, qual, sig in scored:
        if s <= 0 or shown >= opts.top:
            break
        print(f'  {s:6.1f}  {rel}:{start}-{end}  {kind} {qual}{sig}')
        shown += 1
    by_file = defaultdict(list)
    for s, rel, *_ in scored[:80]:
        by_file[rel].append(s)
    by_file = {rel: v[0] + 0.25 * sum(v[1:3]) for rel, v in by_file.items()}  # best match, not file size
    print('Most relevant files: ' + ', '.join(rel for rel, _ in sorted(by_file.items(), key=lambda kv: -kv[1])[:6]))

    names = [q.rsplit('.', 1)[-1] for _, _, _, _, _, q, _ in scored[:4] if len(q.rsplit('.', 1)[-1]) > 3]
    hits = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith('.')]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root).replace(os.sep, '/')
            if not (fname.endswith('.py') and is_test(rel)) or not names:
                continue
            try:
                with open(os.path.join(dirpath, fname), encoding='utf-8', errors='replace') as fh:
                    text = fh.read()
            except OSError:
                continue
            c = sum(len(re.findall(r'\b' + re.escape(nm) + r'\b', text)) for nm in names)
            if c:
                hits[rel] = c
    if hits:
        print('Existing tests mentioning ' + ', '.join(names) + ': ' +
              ', '.join(rel for rel, _ in hits.most_common(5)))
    print('Next: read the top locations (read_file with a line range) and confirm before answering.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f'locate.py failed: {type(e).__name__}: {e}. Fall back to git grep.', file=sys.stdout)
