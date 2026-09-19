#!/usr/bin/env python
"""GPU 1 plain deconvolution scout with certified results and retained checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline

TRAINER = 'experiments/train_cifar_ae_deconv.py'
ARM = 'deconv_16k_no_norm'
TRACK = 'deconv_16k_scout'
pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'


def configuration(smoke=False):
    cfg = yaml.safe_load((ROOT / 'configs/cifar_particle_ae/particle_scaling_scout/split_16384.yaml').read_text())
    reference = json.loads((ROOT / 'runs/cifar_particle_ae/particle_16k_80k/16k_80k/metadata.json').read_text())
    track = TRACK + ('_smoke' if smoke else '')
    cfg.update(generator_arch='deconv', num_particles=16384, expansion_factor=1,
               fixed_sigma=reference['sigma'], resume_checkpoint='', resume_sha256='',
               steps=16 if smoke else 40000, eval_interval=8 if smoke else 5000,
               log_interval=8 if smoke else 100,
               eval_samples=128 if smoke else 50000, final_samples=128 if smoke else 50000,
               initial_eval_samples=0, recon_samples=64 if smoke else 10000,
               max_train_seconds=7200., keep_checkpoints=True,
               out_dir=f'runs/cifar_particle_ae/{track}/{ARM}')
    return track, cfg


def run(smoke=False):
    track, cfg = configuration(smoke)
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
        print('SMOKE certified: 16k particles, lazy double backprop, FID/reconstruction and checkpoints', flush=True)
        return
    best = min(curve, key=lambda x: x['generation']['fid'])
    checkpoints = {}
    for label, step, fid in [('final', cfg['steps'], r['final']['fid']),
                             ('best_sampled', best['step'], best['generation']['fid'])]:
        path = out / f'checkpoint_{step:06d}.pt'
        checkpoints[label] = dict(path=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), step=step, fid50k=fid)
    pipeline.write(report / 'CHECKPOINTS.json', checkpoints)
    lines = ['# Plain deconvolution, 16k particles, no normalization', '',
             '| Step | FID50k |', '|---|---:|']
    lines += [f"| {x['step']} | {x['generation']['fid']:.4f} |" for x in curve]
    lines += ['', f"Best sampled: {best['generation']['fid']:.4f} at {best['step']}; "
              f"final: {r['final']['fid']:.4f}. Training: {r['train_seconds']/60:.2f} minutes.", '',
              '| Historical reference | Steps | FID50k |', '|---|---:|---:|',
              '| Residual CNN, 16k expanded at 10k | 40000 | 17.0982 |',
              '| Residual CNN, 16k best | 80000 | 15.7527 |', '',
              'New G/E/D head and 16,384 independent particle rows trained from scratch. '
              'Frozen pretrained D features and shared noise scale match the historical recipe. '
              'Historical CNN runs expanded a trained 1,024-particle model at 10k; '
              'these are contextual references, not a matched architecture ablation.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    late = curve[-1]['generation']['fid'] - curve[-2]['generation']['fid']
    recommendation = ('FID improved in the final interval; inspect sample quality and consider a checkpoint continuation.'
                      if late < 0 else 'FID did not improve in the final interval; inspect the best checkpoint and curve before extending.')
    (report / 'FINDINGS.md').write_text(
        '# Interpretation\n\n' + recommendation + '\n\n'
        'Architecture and prior initialization history both differ from the historical CNN. '
        'This scout tests viability; it cannot isolate the causal effect of removing normalization. '
        'Review generated grids for artifacts and diversity. No automatic continuation is queued.\n')
    print('\n'.join(lines), flush=True)
    print('RECOMMENDATION ' + recommendation, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.smoke)
