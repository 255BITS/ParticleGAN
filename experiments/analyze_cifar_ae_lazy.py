#!/usr/bin/env python
"""Verify matched lazy-bcap scouts and estimate the cost of longer FID50k runs."""
import argparse
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def runtime_estimate(summary, evaluations, steps):
    """Project train time plus measured evaluation costs; no quality extrapolation."""
    cfg = summary['config']
    small = next(e for e in evaluations if e['generation']['samples'] == cfg['eval_samples'])
    final = evaluations[-1]
    count = (steps - 1) // 5000
    train = summary['train_seconds'] * steps / cfg['steps']
    generation = count * small['generation_seconds'] + final['generation_seconds']
    # Proposed long config reconstructs 10k images at every evaluation.
    recon = (count + 1) * final['reconstruction_seconds'] * 10000 / cfg['recon_samples']
    residual = max(0., summary['total_seconds'] - summary['train_seconds'] - sum(
        e['generation_seconds'] + e['reconstruction_seconds'] for e in evaluations))
    return {'steps': steps, 'train_minutes': train / 60,
            'estimated_total_minutes': (train + generation + recon + residual) / 60}


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_particle_ae.py')
    provenance = code_provenance(trainer, sys.executable)
    configs = [load_config(p, trainer_defaults(trainer)) for p in json.loads(manifest.read_text())]
    shared = lambda c: {k: v for k, v in c.items() if k not in ('reg_every', 'out_dir')}
    if (len(configs) != 3 or {c['reg_every'] for c in configs} != {4, 8, 16}
            or any(shared(c) != shared(configs[0]) for c in configs)
            or configs[0]['final_samples'] != 50000 or configs[0]['eval_samples'] != 5000):
        raise ValueError('requires a matched N=4/8/16 grid with FID5k intermediate and FID50k final')
    rows, missing = [], []
    for cfg in configs:
        run = Path(cfg['out_dir'])
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        s = json.loads((run / 'summary.json').read_text())
        final = s['final']
        if (final['step'] != cfg['steps'] or final['samples'] != 50000
                or not math.isfinite(final['fid']) or not s['sigma_unchanged']
                or not s['frozen_features_unchanged']):
            raise ValueError(f'invalid final result: {run}')
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        ev = [v for v in history if 'generation' in v]
        for e in ev:
            for key in ('generation_seconds', 'reconstruction_seconds'):
                if not math.isfinite(e[key]) or e[key] < 0:
                    raise ValueError(f'invalid evaluation timing: {run}')
        rows.append({'name': run.name, **s, 'evaluations': ev,
                     'runtime_estimates': [runtime_estimate(s, ev, n) for n in (30000, 50000, 100000)]})
    if rows and any((s['metadata']['initialization_sha256'], s['rng_sha256']) !=
                    (rows[0]['metadata']['initialization_sha256'], rows[0]['rng_sha256']) for s in rows):
        raise ValueError('initialization or training random streams differ')
    rows.sort(key=lambda s: (s['final']['fid'], s['config']['reg_every']))
    baseline = next((s for s in rows if s['config']['reg_every'] == 4), None)
    lines = ['# CIFAR-10 AE-GAN lazy bcap scouts', '',
             f'{len(rows)}/3 certified runs; identical initialization, one shared seed, one GPU at a time.', '',
             '| Rank | N | FID50k ↓ | Test MSE ↓ | Steps/s | Train min | Total min | Speedup vs N=4 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rank, s in enumerate(rows, 1):
        speedup = f"{baseline['train_seconds']/s['train_seconds']:.2f}x" if baseline else 'pending'
        lines.append(f"| {rank} | {s['config']['reg_every']} | {s['final']['fid']:.3f} | "
                     f"{s['final']['reconstruction']['recon_mse']:.5f} | {s['config']['steps']/s['train_seconds']:.2f} | "
                     f"{s['train_seconds']/60:.2f} | {s['total_seconds']/60:.2f} | {speedup} |")
    lines += ['', 'Exact double-backprop bcap is applied every N steps with coefficient multiplied by N. '
              'Optimizer rates/betas are fixed. Only N varies. The scratch encoder won the previous scout; '
              'the discriminator still uses frozen ImageNet features.', '',
              '## Runtime projections', '',
              '| N | Measured final FID50k eval min | 30k updates total min | 50k updates total min | 100k updates total min |',
              '|---:|---:|---:|---:|---:|']
    for s in rows:
        estimates = ' | '.join(f"{v['estimated_total_minutes']:.1f}" for v in s['runtime_estimates'])
        lines.append(f"| {s['config']['reg_every']} | {s['evaluations'][-1]['generation_seconds']/60:.2f} | {estimates} |")
    lines += ['', 'Projections assume unchanged hardware/architecture, constant measured training throughput, '
              'FID5k every 5k steps, final FID50k and 10k test reconstructions at each evaluation. '
              'Reconstruction cost is scaled linearly from 1k images; setup/checkpoint overhead uses the scout residual. '
              'These estimate compute cost, not the time required to attain a target FID.', '',
              '## BigGAN comparison', '',
              'The original [BigGAN paper, appendix C.2](https://arxiv.org/html/1809.11096#A3.SS2) '
              'reports CIFAR-10 FID 14.73 and IS 9.22 without truncation. This is a historical context value, '
              'not a matched baseline in this grid. BigGAN uses class conditioning; our model is unconditional '
              'and uses an ImageNet-pretrained discriminator. We measure FID50k against CIFAR train50k using '
              'torch-fidelity TF-compatible Inception. We have not established an exact match to every detail '
              'of the published evaluation. The [author implementation](https://github.com/ajbrock/BigGAN-PyTorch#an-important-note-on-inception-metrics) '
              'explicitly distinguishes its PyTorch monitoring scores from official TF scores. '
              'A strict comparison should re-evaluate a specified CIFAR BigGAN checkpoint through our evaluator. '
              'Do not compare intermediate FID5k with final FID50k as a learning trend.', '',
              '## Recommendation', '']
    report.mkdir(parents=True, exist_ok=True)
    promoted = report / 'winner_long.yaml'
    if missing:
        promoted.unlink(missing_ok=True)
        lines += ['Provisional: pending, failed or uncertified runs: ' + ', '.join(missing) + '.']
    else:
        winner = rows[0]
        cfg = {**winner['config'], 'steps': 30000, 'eval_interval': 5000, 'recon_samples': 10000,
               'max_train_seconds': 7200., 'out_dir': f"runs/cifar_particle_ae/lazy_long/{winner['name']}"}
        promoted.write_text(yaml.safe_dump(cfg, sort_keys=False))
        lines += [f"Promote N={winner['config']['reg_every']} by lowest final FID50k. "
                  'Inspect the speed/quality tradeoff and samples if gaps are small. '
                  'Prepared winner_long.yaml for 30k updates from the same initial seed; longer training is not launched. '
                  'The scout alone cannot predict whether the configuration will reach BigGAN quality.']
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
