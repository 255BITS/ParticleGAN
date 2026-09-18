#!/usr/bin/env python
"""Certify and summarize the isolated baseline/bundle FID scout."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_ae_bundle.py')
    defaults = trainer_defaults(trainer)
    configs = [load_config(p, defaults) for p in json.loads(manifest.read_text())]
    shared = lambda c: {k: v for k, v in c.items() if k not in ('out_dir', 'speed_bundle')}
    if (len(configs) != 2 or {c['speed_bundle'] for c in configs} != {False, True}
            or shared(configs[0]) != shared(configs[1]) or configs[0]['final_samples'] != 50000):
        raise ValueError('requires matched baseline/bundle FID50k configs')
    provenance = code_provenance(trainer, sys.executable)
    rows, missing = [], []
    for cfg in configs:
        run = Path(cfg['out_dir'])
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        s = json.loads((run / 'summary.json').read_text())
        if (s['final']['step'] != cfg['steps'] or s['final']['samples'] != 50000
                or not math.isfinite(s['final']['fid']) or not s['frozen_features_unchanged']
                or not s['sigma_unchanged']):
            raise ValueError(f'invalid final result: {run}')
        rows.append({'name': run.name, **s})
    if len(rows) == 2:
        assert rows[0]['metadata']['initialization_sha256'] == rows[1]['metadata']['initialization_sha256']
        assert rows[0]['rng_sha256'] == rows[1]['rng_sha256']
    rows.sort(key=lambda s: s['final']['fid'])
    lines = ['# AE-GAN optimization bundle: FID scout', '',
             f'{len(rows)}/2 certified runs complete. Same initialization and seed; GPU 1, sequential execution.', '',
             '| Rank | Implementation | FID50k ↓ | Test MSE ↓ | Steps/s | Train min | Total min |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for rank, s in enumerate(rows, 1):
        lines.append(f"| {rank} | {s['name']} | {s['final']['fid']:.3f} | "
                     f"{s['final']['reconstruction']['recon_mse']:.5f} | "
                     f"{s['config']['steps']/s['train_seconds']:.2f} | {s['train_seconds']/60:.2f} | {s['total_seconds']/60:.2f} |")
    lines += ['', '5k updates, N=8 exact double-backprop bcap, same optimizer settings, same scratch encoder, '
              'and frozen pretrained discriminator. The bundle batches D real/fake forwards, reuses their logits '
              'for bcap, and uses foreach EMA. Both arms use two CPU threads. Final FID50k uses the CIFAR train50k '
              'reference and TF-compatible Inception; reconstruction uses 1k test images. '
              'GPU 0 concurrently trains the existing long run, so shared-host variation can affect timing.', '',
              '## Recommendation', '']
    if missing:
        lines.append('Provisional: missing or uncertified runs: ' + ', '.join(missing) + '. Finish these before selecting.')
    else:
        base = next(s for s in rows if not s['config']['speed_bundle'])
        bundle = next(s for s in rows if s['config']['speed_bundle'])
        delta = bundle['final']['fid'] - base['final']['fid']
        speedup = base['train_seconds'] / bundle['train_seconds']
        lines += [f'Bundle minus baseline FID: **{delta:+.3f}** (negative is better). '
                  f'Training throughput: **{speedup:.3f}x** baseline.', '']
        if delta <= 0 and speedup > 1:
            lines.append('The bundle is a candidate for longer validation: it improved measured speed without a '
                         'FID regression in this scout. Inspect sample grids before promotion.')
        elif delta > 0:
            lines.append('Keep the current baseline for the ongoing long run. The bundle had worse final FID; '
                         'weigh the size of that gap against its speed gain before further validation.')
        else:
            lines.append('No demonstrated speed benefit in the full scout; retain baseline pending evidence otherwise.')
        lines += ['', 'This is one configuration comparison, not a seed sweep or proof of equivalent long-run '
                  'quality. Hard routing and Adam can amplify floating-point differences. No implementation '
                  'is automatically installed in the production trainer.']
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
