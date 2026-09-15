#!/usr/bin/env python
"""Export completed motion runs, diagnostics, and an interactive comparison."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import hashlib
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import has_valid_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for path in sorted(Path(args.root).glob('*/summary.json')):
        s = json.loads(path.read_text())
        if not has_valid_summary(str(path.parent), s['config'], s['provenance']):
            raise ValueError(f'Uncertified or stale run: {path.parent}')
        with zipfile.ZipFile(path.parent/'source.zip') as archive:
            for name, digest in s['provenance']['sources'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f'Source archive mismatch: {name}')
        dest = out/path.parent.name
        dest.mkdir(exist_ok=True)
        for name in ['summary.json', 'config.yaml', 'provenance.json', 'run_grid_complete.json',
                     'environment.json', 'metrics.jsonl', 'viewer.html', 'futures.gif',
                     'train_windows.json', 'validation_windows.json', 'test_windows.json']:
            shutil.copy2(path.parent/name, dest/name)
        manifest = json.loads((path.parent/'data_manifest.json').read_text())
        shared_manifest = out/'data_manifest.json'
        if shared_manifest.exists() and json.loads(shared_manifest.read_text()) != manifest:
            raise ValueError('Dataset mismatch across runs')
        shared_manifest.write_text(json.dumps(manifest, indent=2)+'\n')
        from lib.motion_visuals import render
        test = dict(np.load(path.parent/'test_samples.npz', allow_pickle=False))
        render(dest, {'test': test}, s['config'])
        (dest/'visual_provenance.json').write_text(json.dumps(dict(
            renderer_sha256=hashlib.sha256((ROOT/'lib/motion_visuals.py').read_bytes()).hexdigest(),
            note='Regenerated from saved arrays; correct source downward-positive Y for display only.'), indent=2)+'\n')
        row = dict(name=path.parent.name, **{k:v for k,v in s['final']['test'].items() if k!='actions'},
                   validation_ade=s['final']['validation']['single_ade'],
                   validation_sw1=s['final']['validation']['class_sw1'],
                   train_seconds=s['train_seconds'], prefix_sensitivity=s['prefix_sensitivity'])
        rows.append(row)
        cards.append(f'<h2>{path.parent.name}</h2><a href="{path.parent.name}/futures.gif">GIF</a> · <a href="{path.parent.name}/viewer.html">Open viewer</a><iframe src="{path.parent.name}/viewer.html" width="100%" height="780"></iframe>')
    if not rows:
        raise ValueError('No completed runs')
    rows.sort(key=lambda r:r['validation_sw1'])
    table = '| D | Val SW1 ↓ | Test SW1 ↓ | Single ADE ↓ | Best-of-8 ADE ↓ | Diversity | Bone rel. error ↓ | Boundary accel. ↓ | Train s |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        table += f'| {r["name"]} | {r["validation_sw1"]:.4f} | {r["class_sw1"]:.4f} | {r["single_ade"]:.4f} | {r["best_of_k_ade"]:.4f} | {r["diversity"]:.4f} | {r["bone_relative_error"]:.4f} | {r["boundary_acceleration"]:.4f} | {r["train_seconds"]:.1f} |\n'
    (out/'TABLE.md').write_text(table+'\nSorted by validation class-marginal SW1; no single metric establishes motion quality. Spatial units are normalized source coordinates; dynamics are per frame.\n')
    (out/'leaderboard.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Motion completion comparison</title><style>body{font:16px system-ui;background:#101923;color:#eee;margin:20px}a{color:#84c9ff}iframe{border:0}</style><h1>Particle DDGAN: human motion completion</h1><p>Held-out subjects. Same sampler and matched training exposure. Eight futures per evaluated prefix; viewers show the first four samples for fixed clips.</p>'+''.join(cards))
    print(table)


if __name__=='__main__':
    main()
