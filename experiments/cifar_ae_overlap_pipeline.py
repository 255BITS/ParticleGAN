#!/usr/bin/env python
"""Certify frozen 80k/160k noise sweeps sequentially on GPU 1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline


def main(smoke=False):
    track = 'particle_overlap_sweep' + ('_smoke' if smoke else '')
    pipeline.COMMON = ROOT/f'runs/cifar_particle_ae/{track}/PIPELINE.log'
    configs = {}
    for step, reference in [(80000, 15.7526821490), (160000, 19.0584392557)]:
        if smoke and step != 80000:
            continue
        k = step//1000
        path = Path(f'runs/cifar_particle_ae/particle_16k_{k}k/16k_{k}k/checkpoint_{step:06d}.pt')
        cfg = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                   out_dir=f'runs/cifar_particle_ae/{track}/{k}k', expected_fid=reference)
        if smoke:
            cfg.update(samples=128, quality_samples=128, latent_samples=512, noise_scales=[1., .5])
        configs[f'{k}k'] = cfg
    rows = pipeline.grid(track, 'experiments/probe_cifar_ae_overlap.py', '1', configs)
    report = ROOT/f'reports/cifar-particle-ae/{track}'
    pipeline.write(report/'results.json', rows)
    table = ['# Frozen particle noise sweep', '', '| Checkpoint | Noise scale | FID50k | Density | Coverage | Latent confusion |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        assert row['parent_unchanged'] and row['frozen_state_unchanged']
        for r in row['results']:
            table.append(f"| {r['step']} | {r['noise_scale']} | {r['fid']:.4f} | {r['quality']['density']:.4f} | {r['quality']['coverage']:.4f} | {r['geometry']['wrong_nearest_fraction']:.5%} |")
    table += ['', 'Inference-only changes, not training improvements. Coverage uses 10k real/fake, k5. Same draws across scales.']
    if smoke:
        table = ['# Smoke only: reduced sample counts, not benchmark scores', ''] + [s.replace('FID50k', 'FID128').replace('10k real/fake', '128 real/fake') for s in table]
    (report/'LEADERBOARD.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
