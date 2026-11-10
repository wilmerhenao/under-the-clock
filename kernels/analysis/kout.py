#!/usr/bin/env python3
"""Page through ALL output files of a kernel (the CLI stops after the first page) and download those matching a regex.

usage: kout.py <kernel-slug> <dest-dir> <regex> [--list]   (kernel of user verracodeguacas; regex on the output path)
"""
import re
import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

slug, dest, pattern = sys.argv[1], Path(sys.argv[2]), re.compile(sys.argv[3])
only_list = '--list' in sys.argv
api = KaggleApi()
api.authenticate()
files, token = [], None
with api.build_kaggle_client() as k:
    while True:
        r = ApiListKernelSessionOutputRequest()
        r.user_name, r.kernel_slug, r.page_size = 'verracodeguacas', slug, 100
        if token:
            r.page_token = token
        resp = k.kernels.kernels_api_client.list_kernel_session_output(r)
        files += [(f.file_name, f.url) for f in resp.files]
        token = resp.next_page_token
        if not token:
            break
        time.sleep(1.5)
hits = [(n, u) for n, u in files if pattern.search(n)]
print(f'{len(files)} output files, {len(hits)} match {pattern.pattern!r}')
for name, url in hits:
    if only_list:
        print('  ', name)
        continue
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(out, 'wb') as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    print('  ', out, out.stat().st_size)
