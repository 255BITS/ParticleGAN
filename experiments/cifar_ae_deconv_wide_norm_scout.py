#!/usr/bin/env python
"""Scratch wider GroupNorm deconv on GPU 0, paired with the plain deconv scout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline

TRAINER = 'experiments/train_cifar_ae_deconv_wide_norm.py'
TRACK = 'deconv_wide_norm_16k_scout'
ARM = 'deconv_wide_norm_16k'
pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'


def configuration(smoke=False):
    reference = json.loads((ROOT / 'reports/cifar-particle-ae/deconv_16k_scout/results.json').read_text())[0]
    track = TRACK + ('_smoke' if smoke else '')
    cfg = {**reference['config'], 'generator_arch': 'deconv_wide_norm',
           'steps': 16 if smoke else 40000, 'eval_interval': 8 if smoke else 5000,
           'log_interval': 8 if smoke else 100, 'eval_samples': 128 if smoke else 50000,
           'final_samples': 128 if smoke else 50000, 'recon_samples': 64 if smoke else 10000,
           'out_dir': f'runs/cifar_particle_ae/{track}/{ARM}', 'max_train_seconds': 10800.}
    if not smoke:
        changed = {k for k in cfg if cfg[k] != reference['config'][k]}
        assert changed == {'generator_arch', 'out_dir', 'max_train_seconds'}
    return track, cfg, reference


def run(smoke=False):
    track, cfg, reference = configuration(smoke)
    r = pipeline.grid(track, TRAINER, '0', {ARM: cfg})[0]
    assert r['start_step'] == 0 and r['final']['step'] == cfg['steps']
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    assert r['metadata']['parameters']['prior'] == 16384 * 64
    assert r['metadata']['sigma'] == reference['metadata']['sigma']
    assert r['metadata']['D_E_initialization_sha256'] == reference['metadata']['D_E_initialization_sha256']
    out = ROOT / cfg['out_dir']
    metrics = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert [x['step'] for x in curve] == list(range(cfg['eval_interval'], cfg['steps'] + 1, cfg['eval_interval']))
    assert all(x['generation']['samples'] == cfg['final_samples'] for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    report = ROOT / 'reports/cifar-particle-ae' / track
    pipeline.write(report / 'results.json', [{'name': ARM, 'curve': curve, **r}])
    if smoke:
        print('SMOKE certified: wider GroupNorm G; original D/E initialization, sigma and recipe', flush=True)
        return
    best = min(curve, key=lambda x: x['generation']['fid'])
    checkpoints = {}
    for label, step, fid in [('final', cfg['steps'], r['final']['fid']),
                             ('best_sampled', best['step'], best['generation']['fid'])]:
        path = out / f'checkpoint_{step:06d}.pt'
        checkpoints[label] = dict(path=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), step=step, fid50k=fid)
    pipeline.write(report / 'CHECKPOINTS.json', checkpoints)
    baseline = {x['step']: x['generation']['fid'] for x in reference['curve']}
    lines = ['# Wider deconv with GroupNorm, 16k particles', '',
             '| Step | Wide + GN FID50k | Small no norm FID50k | Difference |', '|---|---:|---:|---:|']
    lines += [f"| {x['step']} | {x['generation']['fid']:.4f} | {baseline[x['step']]:.4f} | "
              f"{x['generation']['fid']-baseline[x['step']]:+.4f} |" for x in curve]
    lines += ['', f"Best sampled {best['generation']['fid']:.4f} at {best['step']}; final {r['final']['fid']:.4f}. "
              f"Training minutes {r['train_seconds']/60:.2f}.", '',
              'Matched scratch prior initialization, D/E initialization, shared sigma, recipe and evaluation steps. '
              'This tests width plus normalization jointly and cannot separate their individual effects. '
              'No automatic promotion queued.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    delta = r['final']['fid'] - reference['final']['fid']
    text = (f"# Interpretation\n\nFinal FID difference versus small unnormalized deconv: {delta:+.4f}. "
            'Inspect the complete curve and image grids; neither endpoint FID nor parameter count alone establishes a capacity limit. '
            'If the combined upgrade helps, isolate width and normalization in subsequent tests. '
            'Recommend continuation only after reviewing whether late training still improves. No further run queued.\n')
    (report / 'FINDINGS.md').write_text(text)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.smoke)
