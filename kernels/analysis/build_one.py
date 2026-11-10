#!/usr/bin/env python3
"""Build only the named kernels (so other kernel dirs are not regenerated): build_one.py <name> [<name> ...]"""
import sys
from pathlib import Path

build = Path(__file__).resolve().parent.parent / 'smoke' / 'build.py'
src = build.read_text()
loop = 'for name, k in KERNELS.items():'
assert loop in src
exec(compile(src.replace(loop, f'for name, k in [(n, KERNELS[n]) for n in {sys.argv[1:]!r}]:'), str(build), 'exec'),
     {'__file__': str(build), '__name__': '__main__'})
