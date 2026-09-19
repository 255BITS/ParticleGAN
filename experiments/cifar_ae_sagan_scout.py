#!/usr/bin/env python
"""Active-from-initialization SAGAN-style G/D attention scout on GPU 1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline

TRAINER = 'experiments/train_cifar_ae_sagan.py'
TRACK = 'sagan_gd_16k_scout'
ARM = 'sagan_gd_16k'
BASELINE_TRACK = 'deconv_wide_norm_16k_scout'
pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'


def configuration(smoke=False):
    path = ROOT / f'configs/cifar_particle_ae/{BASELINE_TRACK}/deconv_wide_norm_16k.yaml'
    baseline = yaml.safe_load(path.read_text())
    track = TRACK + ('_smoke' if smoke else '')
    cfg = {**baseline, 'generator_arch': 'sagan_gd',
           'out_dir': f'runs/cifar_particle_ae/{track}/{ARM}'}
    assert {k for k in cfg if cfg[k] != baseline[k]} == {'generator_arch', 'out_dir'}
    if smoke:
        cfg.update(steps=16, eval_interval=8, log_interval=8,
                   eval_samples=128, final_samples=128, recon_samples=64)
    return track, cfg, baseline


def run(smoke=False):
    track, cfg, baseline = configuration(smoke)
    r = pipeline.grid(track, TRAINER, '1', {ARM: cfg})[0]
    assert r['start_step'] == 0 and r['final']['step'] == cfg['steps']
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    assert r['metadata']['parameters']['prior'] == 16384 * 64
    assert abs(r['metadata']['sigma'] - cfg['fixed_sigma']) < 1e-8
    out = ROOT / cfg['out_dir']
    metrics = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert [x['step'] for x in curve] == list(range(cfg['eval_interval'], cfg['steps'] + 1, cfg['eval_interval']))
    assert all(x['generation']['samples'] == cfg['final_samples'] for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    report = ROOT / 'reports/cifar-particle-ae' / track
    pipeline.write(report / 'results.json', [{'name': ARM, 'curve': curve, **r}])
    if smoke:
        print('SMOKE certified: 16k particles, active G/D attention, bcap double backward, FID/reconstruction/checkpoints', flush=True)
        return
    best = min(curve, key=lambda x: x['generation']['fid'])
    checkpoints = {}
    for label, step, fid in [('final', cfg['steps'], r['final']['fid']),
                             ('best_sampled', best['step'], best['generation']['fid'])]:
        path = out / f'checkpoint_{step:06d}.pt'
        checkpoints[label] = dict(path=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), step=step, fid50k=fid)
    pipeline.write(report / 'CHECKPOINTS.json', checkpoints)
    baseline_metrics = ROOT / baseline['out_dir'] / 'metrics.jsonl'
    reference = {x['step']: x['generation']['fid'] for x in
                 (json.loads(line) for line in baseline_metrics.read_text().splitlines()) if 'generation' in x}
    lines = ['# SAGAN-style attention in G and D, 16k particles', '',
             '| Step | Attention FID50k | No-attention wide GN FID50k | Difference |', '|---|---:|---:|---:|']
    for x in curve:
        fid, step = x['generation']['fid'], x['step']
        old = reference.get(step)
        cells = f'{old:.4f} | {fid-old:+.4f}' if old is not None else 'pending | pending'
        lines.append(f'| {step} | {fid:.4f} | {cells} |')
    lines += ['', f"Best sampled {best['generation']['fid']:.4f} at {best['step']}; final {r['final']['fid']:.4f}. "
              f"Training minutes {r['train_seconds']/60:.2f}.", '',
              'Same scratch particle initialization and existing base G/D/E weights, sigma and recipe. '
              'Attention is active from initialization with a fixed unit residual coefficient. '
              'This jointly changes G and D; it does not isolate which side helps. '
              'SAGAN-style attention adaptation, not a reproduction of the original paper. No automatic promotion.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    (report / 'FINDINGS.md').write_text('# Interpretation\n\n'
        'Review the matched-step curve, sample grids, attention gradient norms and final training stability. '
        'If attention helps, G-only/D-only checkpoint or architecture comparisons can separate contributions. '
        'Stable training in this run would support compatibility with this recipe; it would not establish '
        'that all GAN instability mechanisms are eliminated. Retain best and final checkpoints separately. '
        'No further run queued.\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.smoke)
