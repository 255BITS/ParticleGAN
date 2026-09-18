#!/usr/bin/env python
"""Verify and tabulate the saved conditional variation audit."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/cifar_particle_ae/variation'
    out = ROOT / 'reports/cifar-particle-ae/variation'
    summary = json.loads((run / 'summary.json').read_text())
    protocol = summary['protocol']
    assert summary['checkpoint_unchanged'] and summary['model_state_unchanged']
    source = ROOT / 'experiments/measure_cifar_particle_variation.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['script_sha256']
    assert (run / 'source.py').read_bytes() == source.read_bytes()
    checkpoint = ROOT / 'runs/cifar_particle_ae/scout/bounded/checkpoint.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == protocol['checkpoint_sha256']
    assert len(summary['rows']) == 6
    with np.load(run / 'per_input_metrics.npz') as metrics:
        assert metrics['selected_ids'].shape == (protocol['inputs'],)
        for row in summary['rows']:
            for key in ('pair_pixel_mse', 'pair_feature_cosine', 'identical_pair_fraction',
                        'unique_draws', 'recon_mse', 'feature_to_input',
                        'feature_to_deterministic_reconstruction', 'own_anchor_nearest_fraction'):
                values = metrics[row['condition'] + '__' + key]
                assert values.shape == (protocol['inputs'],) and np.isfinite(values).all()
                # Torch and NumPy use different float32 reduction orders.
                assert np.isclose(values.astype(np.float64).mean(), row[key], rtol=1e-6, atol=1e-7)
            if row['condition'] == 'encoded_0':
                assert row['pair_pixel_mse'] == 0 and row['unique_draws'] == 1
    out.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'protocol.json', 'per_input_metrics.npz'):
        shutil.copy2(run / name, out / name)
    lines = ['# Conditional variation: diversity/fidelity table', '',
             'One frozen EMA checkpoint. 512 held-out inputs, eight draws each, all 28 unordered pairs per input. Ordered by noise strength, not a universal quality ranking.', '',
             '| Sampling condition | Unique /8 | Pair pixel RMSE /255 | Feature diversity /unrelated recon | Input MSE | MSE increase | Own anchor nearest |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['rows']:
        name = row['condition'].replace('encoded_', 'z_X + sigma × ').replace('selected_center_1', 'p[k] + sigma × noise')
        if row['condition'].startswith('encoded_'):
            name += ' × noise'
        lines.append(f"| {name} | {row['unique_draws']:.1f} | {row['pair_pixel_rmse']:.2f} | {row['feature_diversity_percent_unrelated_reconstruction']:.1f}% | {row['recon_mse']:.5f} | {row['recon_mse_change_percent']:+.1f}% | {100*row['own_anchor_nearest_fraction']:.2f}% |")
    ref = summary['reference']
    lines += ['', 'Feature diversity = mean pairwise cosine distance in normalized Inception pool2048 features, divided by the distance between unrelated deterministic reconstructions.',
              f"Unrelated reference: cosine distance {ref['unrelated_reconstruction_feature_cosine']:.6f}, pixel RMSE {ref['unrelated_reconstruction_pixel_rmse']:.2f}/255. Unrelated real images have cosine distance {ref['unrelated_real_feature_cosine']:.6f}.",
              'Own-anchor retention retrieves the nearest of 512 deterministic reconstructions in feature space. It is not original-image identity or class accuracy.',
              'MSE uses [-1,1] float pixels. Uniqueness and feature extraction use uint8 images. Zero-noise cosine distance has a ~1e-8 floating-point floor despite identical pixels.',
              f"Runtime {summary['total_seconds']:.1f} sec; peak {summary['peak_memory_gb']:.2f} GiB. No training, image grids, visual inspection, or checkpoint changes.", '']
    (out / 'LEADERBOARD.md').write_text('\n'.join(lines))
    print('\n'.join(lines).rstrip())


if __name__ == '__main__':
    main()
