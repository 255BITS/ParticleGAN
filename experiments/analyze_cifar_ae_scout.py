#!/usr/bin/env python
"""Rank manifest-listed CIFAR AE scouts and prepare a longer winner config."""
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


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_particle_ae.py')
    defaults = trainer_defaults(trainer)
    provenance = code_provenance(trainer, sys.executable)
    configs = [load_config(path, defaults) for path in json.loads(manifest.read_text())]
    if not configs:
        raise ValueError('empty scout manifest')
    for key in ('arm', 'seed', 'steps', 'batch_size', 'final_samples', 'eval_samples',
                'eval_interval', 'recon_samples', 'data_dir', 'fid_cache'):
        if any(c[key] != configs[0][key] for c in configs):
            raise ValueError(f'unmatched scout protocol: {key}')
    if configs[0]['arm'] != 'bounded' or configs[0]['final_samples'] < 2:
        raise ValueError('requires AE scouts with FID evaluation')
    rows, missing = [], []
    for cfg in configs:
        run = Path(cfg['out_dir'])
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        s = json.loads((run / 'summary.json').read_text())
        f = s['final']
        if (f['step'] != cfg['steps'] or f['samples'] != cfg['final_samples']
                or not math.isfinite(f['fid']) or not s['sigma_unchanged']
                or not s['frozen_features_unchanged']
                or (cfg['encoder_backbone'] != 'scratch' and not s['frozen_encoder_features_unchanged'])):
            raise ValueError(f'invalid completed result: {run}')
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        evaluations = [v for v in history if 'generation' in v]
        rows.append({'name': run.name, **s, 'evaluations': evaluations})
    rows.sort(key=lambda s: (s['final']['fid'], s['name']))
    report.mkdir(parents=True, exist_ok=True)
    lines = ['# CIFAR-10 AE-GAN pretrained encoder scouts', '',
             f"{len(rows)}/{len(configs)} certified runs complete; {configs[0]['steps']} updates each. "
             f"Ranked by final EMA FID{configs[0]['final_samples']:,} (lower is better).", '',
             '| Rank | Config | FID ↓ | Test MSE ↓ | Effective particles ↑ | Offset saturation | Train min |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for i, s in enumerate(rows, 1):
        r = s['final']['reconstruction']
        lines.append(f"| {i} | {s['name']} | {s['final']['fid']:.3f} | {r['recon_mse']:.5f} | "
                     f"{r['effective_particles']:.1f} | {100*r['offset_saturation']:.1f}% | {s['train_seconds']/60:.2f} |")
    lines += ['', 'One shared seed; no seed sweep. FID uses unconditional generated images and the CIFAR-10 '
              'train50k reference. Reconstruction uses the first 1,000 unaugmented test images; labels are unused. '
              'FID5k is a scouting metric and cannot be compared directly with historical FID50k. '
              'Particle usage describes encoder routing, not unconditional class coverage.', '',
              '## Interpretation', '',
              'The scratch control and pretrained baseline differ only in encoder architecture/initialization. '
              'The pretrained backbone is frozen through layer3, with trainable spatial query and offset heads. '
              'The reconstruction-weight variant changes 1.0 to 0.3; the learning-rate variant halves G/E, D '
              'and prior rates together. Every arm retains the existing pretrained discriminator.', '']
    base = next((s for s in rows if s['name'] == '02_pretrained'), None)
    control = next((s for s in rows if s['name'] == '01_scratch_control'), None)
    if base and control:
        delta = base['final']['fid'] - control['final']['fid']
        lines += [f"Pretrained baseline minus scratch FID: {delta:+.3f}. "
                  'This is one matched trajectory, not evidence of seed-to-seed reliability.', '']
    for s in rows:
        ev = s['evaluations']
        if len(ev) > 1 and ev[-2]['generation']['samples'] == ev[-1]['generation']['samples']:
            change = ev[-1]['generation']['fid'] - ev[-2]['generation']['fid']
            lines.append(f"- {s['name']}: FID changed {change:+.3f} over the last evaluation interval "
                         '(negative means improvement).')
    lines += ['', '## Recommendation', '']
    promoted = report / 'winner_long.yaml'
    if missing:
        lines += ['Pending, failed, or uncertified: ' + ', '.join(missing) + '.',
                  'Ranking is provisional. Finish or repair these runs before selecting the longer experiment.']
        promoted.unlink(missing_ok=True)
    elif rows:
        winner = rows[0]
        longer = {**winner['config'], 'steps': 30000, 'eval_interval': 5000,
                  'eval_samples': 5000, 'final_samples': 50000, 'recon_samples': 10000,
                  'max_train_seconds': 7200.,
                  'out_dir': f"runs/cifar_particle_ae/pretrained_long/{winner['name']}"}
        promoted.write_text(yaml.safe_dump(longer, sort_keys=False))
        lines += [f"Select **{winner['name']}** by final scout FID ({winner['final']['fid']:.3f}). "
                  'Inspect its sample grids, reconstruction and late FID trend before the longer run. '
                  'Small scout gaps may reflect finite-sample noise.', '',
                  'Prepared `winner_long.yaml`: 30k updates, FID50k at the end, all 10k test reconstructions, '
                  'and retained evaluation checkpoints. This starts a fresh longer trajectory with the winning '
                  'configuration and same seed; it does not resume the scout checkpoint. It is not launched automatically.', '',
                  '```sh', '.venv/bin/python -u experiments/follow_grid.py \\',
                  '  --root runs/cifar_particle_ae/pretrained_long \\',
                  '  --log runs/cifar_particle_ae/pretrained_long/PIPELINE.log -- \\',
                  f"  --configs '{promoted}' --gpus 0 --workers_per_gpu 1 \\",
                  '  --python .venv/bin/python --trainer experiments/train_cifar_particle_ae.py', '```']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    (report / 'leaderboard.json').write_text(json.dumps({'complete': not missing, 'missing': missing,
                                                       'rows': rows}, indent=2, allow_nan=False) + '\n')
    print('\n'.join(lines), flush=True)
    return 1 if missing else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(analyze(args.config_manifest, args.report))
