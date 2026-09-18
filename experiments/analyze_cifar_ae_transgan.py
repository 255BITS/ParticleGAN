#!/usr/bin/env python
"""Certified scratch transformer/routing scouts with an existing CNN reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def historical_cnn():
    """Reuse verified training and previously audited FID50k; never retrain."""
    root = ROOT / 'runs/cifar_particle_ae'
    summaries = []
    for name in ('lazy_long/n08', 'duration_100k/n08'):
        run = root / name
        summary = json.loads((run / 'summary.json').read_text())
        provenance = json.loads((run / 'provenance.json').read_text())
        assert has_valid_summary(str(run), summary['config'], code_provenance(provenance['trainer'], sys.executable))
        summaries.append(summary)
    early = json.loads((ROOT / 'reports/cifar-particle-ae/lazy-long/curve.json').read_text())
    curve = []
    for value in early:
        step = value['step']
        checkpoint = root / 'lazy_long/n08' / f'checkpoint_{step:06d}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == value['checkpoint_sha256']
        if step != 30000:
            audit = json.loads((root / 'lazy_long/fid50k_curve' / f'step_{step:06d}/summary.json').read_text())
            assert audit['checkpoint_sha256'] == value['checkpoint_sha256'] and audit['checkpoint_unchanged']
            assert audit['final']['samples'] == 50000 and audit['final']['fid'] == value['fid50k']
        else:
            assert summaries[0]['final']['samples'] == 50000 and summaries[0]['final']['fid'] == value['fid50k']
        curve.append({'step': step, 'fid': value['fid50k'], 'recon_mse': value['recon_mse'],
                      'train_seconds': value['train_minutes'] * 60})
    for line in (root / 'duration_100k/n08/metrics.jsonl').read_text().splitlines():
        value = json.loads(line)
        if 'generation' in value and value['step'] <= 50000:
            assert value['generation']['samples'] == 50000
            curve.append({'step': value['step'], 'fid': value['generation']['fid'],
                          'recon_mse': value['reconstruction']['recon_mse'],
                          'train_seconds': summaries[0]['train_seconds'] + value['train_seconds']})
    assert curve[-1]['step'] == 50000
    return {'name': 'cnn_all_historical', 'curve': curve, 'parameters': summaries[0]['metadata']['parameters']['G'],
            'fid_protocol': summaries[0]['metadata']['fid_protocol'], 'historical': True}


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_ae_transgan.py')
    provenance, defaults = code_provenance(trainer, sys.executable), trainer_defaults(trainer)
    rows, missing = [], []
    historical = historical_cnn()
    for path in json.loads(manifest.read_text()):
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['start_step'] == 0 and summary['final']['step'] == cfg['steps'] == 50000
        assert summary['final']['samples'] == 50000 and math.isfinite(summary['final']['fid'])
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        assert summary['metadata']['fid_protocol'] == historical['fid_protocol']
        curve = []
        for line in (run / 'metrics.jsonl').read_text().splitlines():
            value = json.loads(line)
            if 'generation' in value:
                assert value['generation']['samples'] == 50000
                curve.append({'step': value['step'], 'fid': value['generation']['fid'],
                              'recon_mse': value['reconstruction']['recon_mse'],
                              'train_seconds': value['train_seconds']})
        best = min(curve, key=lambda value: value['fid'])
        rows.append({'name': run.name, 'curve': curve, 'best_observed': {**best,
                     'checkpoint': str(run / f"checkpoint_{best['step']:06d}.pt")}, **summary})
    rows.sort(key=lambda row: row['final']['fid'])
    by_name = {row['name']: row for row in rows}
    lines = ['# CIFAR AE-GAN: scratch transformer and reconstruction routing', '',
             f'{len(rows)}/{len(rows) + len(missing)} certified scouts complete.', '',
             'All new runs start from scratch for 50k updates. Same frozen pretrained ResNet18 D, scratch E, '
             'particle prior, optimizer rates, one D update, EMA and lazy bcap every8 with coefficient×8. '
             'E-only reconstruction detaches particle centers and freezes G parameters for the reconstruction '
             'forward while preserving the gradient through G into E. Adversarial gradients still train G and the prior.', '',
             '| Rank | New run | G parameters | Final FID50k ↓ | Test MSE ↓ | Train min | Wall min | Updates/s |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for rank, row in enumerate(rows, 1):
        lines.append(f"| {rank} | {row['name']} | {row['metadata']['parameters']['G']:,} | {row['final']['fid']:.4f} | "
                     f"{row['final']['reconstruction']['recon_mse']:.5f} | {row['train_seconds']/60:.2f} | "
                     f"{row['total_seconds']/60:.2f} | {50000/row['train_seconds']:.2f} |")
    baseline = historical['curve'][-1]
    lines += ['', f"Historical CNN with full reconstruction at50k: **FID {baseline['fid']:.4f}**, "
              f"test MSE {baseline['recon_mse']:.5f}; {historical['parameters']:,} G parameters. "
              'Its original scratch trajectory resumed at30k with full optimizer/RNG state and unchanged recipe. '
              'It is reused at user request and is not a new concurrent control.', '',
              'FID uses 50k prior samples, EMA G/prior and the existing TF-compatible Inception CIFAR train50k '
              'reference. Reconstruction uses test10k. All arms use the same seed; no seed sweeps. '
              'Transformer size differs from CNN size, so this tests architecture plus capacity. '
              'It is a TransGAN-style generator inside AE-GAN, with a normalized low-gain tanh RGB head '
              'to prevent preflight saturation, not a reproduction of the full TransGAN recipe.', '',
              '| Run | Step | FID50k ↓ | Test MSE ↓ | Training min |', '|---|---:|---:|---:|---:|']
    for row in [historical] + rows:
        for value in row['curve']:
            lines.append(f"| {row['name']} | {value['step']:,} | {value['fid']:.4f} | {value['recon_mse']:.5f} | {value['train_seconds']/60:.2f} |")
    effects = None
    lines += ['', '## Interpretation and recommendation', '']
    if missing:
        lines.append('Pending or uncertified: ' + ', '.join(missing) + '. No winner selected.')
    elif set(by_name) == {'transgan_all', 'transgan_e_only', 'cnn_e_only'}:
        f = {name: row['final']['fid'] for name, row in by_name.items()}
        effects = {'architecture_under_e_only': f['transgan_e_only'] - f['cnn_e_only'],
                   'e_only_in_transformer': f['transgan_e_only'] - f['transgan_all'],
                   'e_only_in_cnn_historical': f['cnn_e_only'] - baseline['fid'],
                   'architecture_full_reconstruction_historical': f['transgan_all'] - baseline['fid']}
        lines += [f"Transformer − CNN under E-only reconstruction: {effects['architecture_under_e_only']:+.4f} FID. "
                  f"E-only − full reconstruction within transformer: {effects['e_only_in_transformer']:+.4f} FID. "
                  'These compare new scratch runs; negative favors the first named intervention.',
                  f"Relative to historical full-reconstruction CNN, CNN E-only changes FID by {effects['e_only_in_cnn_historical']:+.4f}, "
                  f"and transformer full reconstruction by {effects['architecture_full_reconstruction_historical']:+.4f}. "
                  'These historical contrasts are less controlled; small differences should not be overinterpreted.']
        winner = rows[0]
        last = winner['curve'][-2:]
        trend = last[-1]['fid'] - last[0]['fid']
        lines.append(f"Lowest new final endpoint: **{winner['name']} ({winner['final']['fid']:.4f})**; "
                     f"last evaluation change {trend:+.4f}. "
                     'Endpoint ranking is separate from intermediate minima and does not prove a unique cause of the plateau.')
        if winner['final']['fid'] < baseline['fid'] - 1 and trend <= 0:
            lines.append('Recommendation: consider continuing this saved checkpoint after reviewing its compute cost and sample quality. No long promotion is automatic.')
        elif winner['final']['fid'] < baseline['fid']:
            lines.append('Recommendation: review the final trend and wall-clock cost before extending; a small or rebounding gain is insufficient evidence for automatic promotion.')
        else:
            lines.append('Recommendation: these endpoints do not improve on the historical CNN baseline. Inspect critic/gradient diagnostics and samples before committing to longer training.')
        best_run = min(rows, key=lambda row: row['best_observed']['fid'])
        best = best_run['best_observed']
        lines.append(f"Best observed new measurement: {best['fid']:.4f}, {best_run['name']} at{best['step']:,}. "
                     f"Checkpoint `{best['checkpoint']}`. This minimum is selected across evaluations.")
        lines.append('Target below13 reached at a final endpoint.' if winner['final']['fid'] < 13 else 'Target below13 remains unmet at final endpoints.')
    report.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for row in [historical] + rows:
        curve = row['curve']
        style = '--o' if row.get('historical') else '-o'
        axes[0].plot([v['step']/1000 for v in curve], [v['fid'] for v in curve], style, label=row['name'])
        axes[1].plot([v['train_seconds']/60 for v in curve], [v['fid'] for v in curve], style)
        axes[2].plot([v['step']/1000 for v in curve], [v['recon_mse'] for v in curve], style)
    for ax, title, xlabel in zip(axes, ['Generation FID50k', 'FID versus training cost', 'Test reconstruction MSE'],
                                 ['Updates (thousands)', 'Training minutes', 'Updates (thousands)']):
        ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(alpha=.2)
    axes[0].axhline(13, color='black', linestyle=':', label='target13')
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(report / 'curves.png', dpi=180); plt.close(fig)
    lines += ['', '![Learning curves and cost](curves.png)']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    (report / 'leaderboard.json').write_text(json.dumps({'complete': not missing, 'missing': missing,
                                                       'effects': effects, 'historical': historical, 'rows': rows}, indent=2, allow_nan=False) + '\n')
    print('\n'.join(lines), flush=True)
    return int(bool(missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(analyze(args.config_manifest, args.report))
