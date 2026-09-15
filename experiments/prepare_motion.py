#!/usr/bin/env python
"""Download the official HumanAct12 joint arrays, with a pinned file manifest.

Requires gdown (uv pip install --python .venv/bin/python gdown).
No external Python or pickled objects are executed. Existing valid .npy files
are reused; incomplete files are replaced atomically.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

FOLDER = 'https://drive.google.com/drive/folders/1TBY2x-gD6f3yzQ0WNmXP2-be3xu3qDkV'


def main():
    import gdown
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/humanact12')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    listing = out / 'download_manifest.json'
    if listing.exists():
        rows = json.loads(listing.read_text())
    else:
        files = gdown.download_folder(FOLDER, output=str(out/'raw'), skip_download=True, quiet=True)
        rows = [dict(id=f.id, path=f.path) for f in files]
        listing.write_text(json.dumps(rows, indent=2)+'\n')
    print(f'DOWNLOAD {len(rows)} files, {args.workers} workers', flush=True)

    def fetch(row):
        target = out / 'raw' / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        def valid():
            if not target.is_file():
                return False
            if target.suffix != '.npy':
                return target.stat().st_size > 0
            try:
                x = np.load(target, allow_pickle=False)
                return x.ndim == 3 and x.shape[1:] == (24, 3) and np.isfinite(x).all()
            except (ValueError, OSError, EOFError):
                return False
        if not valid():
            for attempt in range(3):
                try:
                    tmp = target.with_suffix(target.suffix+'.partial')
                    url = 'https://drive.usercontent.google.com/download?id=' + row['id'] + '&export=download&confirm=t'
                    with urllib.request.urlopen(url, timeout=60) as response:
                        tmp.write_bytes(response.read())
                    tmp.replace(target)
                    if not valid():
                        raise ValueError(f'Invalid array: {target}')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        return {**row, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = []
        for i, row in enumerate(pool.map(fetch, rows), 1):
            hashes.append(row)
            if i % 100 == 0:
                print(f'DOWNLOAD {i}/{len(rows)}', flush=True)
    (out/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(f'COMPLETE {len(hashes)} verified files', flush=True)


if __name__ == '__main__':
    main()
