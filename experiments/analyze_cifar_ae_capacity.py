#!/usr/bin/env python
"""Certified capacity/duration leaderboards and FID50k learning curves."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_ae_capacity.py')
    defaults = trainer_defaults(trainer)
    provenance = code_provenance(trainer, sys.executable)
    configs = [load_config(p, defaults) for p in json.loads(manifest.read_text())]
    rows, missing = [], []
    for cfg in configs:
        run = ROOT / cfg['out_dir']
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        s = json.loads((run / 'summary.json').read_text())
        if (s['final']['step'] != cfg['steps'] or s['final']['samples'] != 50000
                or not math.isfinite(s['final']['fid']) or not s['frozen_features_unchanged']
                or not s['sigma_unchanged']):
            raise ValueError(f'invalid final result: {run}')
        if s['start_step'] != (s['metadata']['resume']['step'] if cfg['resume_checkpoint'] else 0):
            raise ValueError(f'invalid continuation: {run}')
        curve = []
        for line in (run / 'metrics.jsonl').read_text().splitlines():
            m = json.loads(line)
            if 'generation' in m:
                assert m['generation']['samples'] == 50000 and math.isfinite(m['generation']['fid'])
                curve.append({'step': m['step'], 'fid50k': m['generation']['fid'],
                              'recon_mse': m['reconstruction']['recon_mse']})
        rows.append({'name': run.name, 'curve': curve, **s})
    fresh = [s for s in rows if not s['config']['resume_checkpoint']]
    if len(fresh) > 1:
        assert len({s['metadata']['D_E_initialization_sha256'] for s in fresh}) == 1
        allowed = {'g_width', 'g_depth', 'out_dir'}
        assert all({k: v for k, v in s['config'].items() if k not in allowed}
                   == {k: v for k, v in fresh[0]['config'].items() if k not in allowed} for s in fresh)
    rows.sort(key=lambda s: s['final']['fid'])
    lines = ['# CIFAR AE-GAN: ' + manifest.parent.name, '',
             f'{len(rows)}/{len(configs)} certified runs complete. Target: FID50k below 13.', '',
             '| Rank | Run | G params | Updates (start → end) | Final FID50k ↓ | Test MSE ↓ | Updates/s | New train min |',
             '|---:|---|---:|---|---:|---:|---:|---:|']
    for rank, s in enumerate(rows, 1):
        lines.append(f"| {rank} | {s['name']} | {s['metadata']['parameters']['G']:,} | "
                     f"{s['start_step']:,} → {s['final']['step']:,} | {s['final']['fid']:.3f} | "
                     f"{s['final']['reconstruction']['recon_mse']:.5f} | "
                     f"{(s['final']['step']-s['start_step'])/s['train_seconds']:.2f} | {s['train_seconds']/60:.2f} |")
    lines += ['', 'N=8 double-backprop bcap (coefficient multiplied by 8); scratch encoder; frozen pretrained '
              'discriminator features. Width 32 for D/E throughout. Same seed for configuration control; no seed sweep. '
              'FID uses 50k generated images, CIFAR train50k reference, TF-compatible Inception and EMA weights. '
              'Reconstruction uses 10k test images. Both GPUs run concurrently, so timing includes shared-host effects.', '',
              'Historical width32 baseline: FID50k **19.611 at 10k**, **20.105 at 20k**, **19.439 at 30k** '
              '([audited curve](../lazy-long/README.md)). This is a historical reference, not a repeated control.', '',
              '## Learning curves', '', '| Run | Step | FID50k ↓ | Test MSE ↓ |', '|---|---:|---:|---:|']
    for s in rows:
        for c in s['curve']:
            lines.append(f"| {s['name']} | {c['step']:,} | {c['fid50k']:.3f} | {c['recon_mse']:.5f} |")
    lines += ['', '## Recommendation', '']
    if missing:
        lines.append('Provisional: incomplete or uncertified runs: ' + ', '.join(missing) + '.')
    elif rows:
        best = rows[0]
        fid = best['final']['fid']
        lines.append(f"Best final measurement: **{best['name']}, FID50k {fid:.3f}**. "
                     + ('The below-13 target is reached; inspect samples and benchmark protocol before claiming parity.'
                        if fid < 13 else f'The remaining gap to 13 is {fid-13:.3f}.'))
        if fresh:
            lines.append('Extend the leading capacity configuration if its improvement is meaningful and its curve '
                         'supports further training. Compare its time per update with the duration track. '
                         'If both remain near or above the historical 20k result (20.105), these scouts do not support '
                         'spending more compute on width/depth alone.')
        else:
            gain = 19.439097618143478 - fid
            lines.append(f'Continuation changed FID by {(-gain):+.3f} relative to the saved 30k checkpoint. '
                         'Use the last several FID50k measurements to judge remaining progress; improving '
                         'reconstruction alone does not establish that generation FID will catch up.')
        lines.append('Selection uses final measurements; intermediate values describe the curve. '
                     'Review both track reports before selecting the next long run.')
    report.mkdir(parents=True, exist_ok=True)
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    (report / 'leaderboard.json').write_text(json.dumps({'complete': not missing, 'missing': missing, 'rows': rows},
                                                       indent=2, allow_nan=False) + '\n')
    print('\n'.join(lines), flush=True)
    return 1 if missing else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(analyze(args.config_manifest, args.report))
