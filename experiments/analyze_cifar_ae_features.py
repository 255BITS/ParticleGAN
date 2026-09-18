#!/usr/bin/env python
"""Certified pretrained-feature and selective reconstruction scouts."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(manifest, report):
    trainer = str(ROOT / 'experiments/train_cifar_ae_features.py')
    provenance = code_provenance(trainer, sys.executable)
    defaults = trainer_defaults(trainer)
    rows, missing = [], []
    for path in json.loads(manifest.read_text()):
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name)
            continue
        s = json.loads((run / 'summary.json').read_text())
        assert s['final']['step'] == cfg['steps'] and s['final']['samples'] == 50000
        assert math.isfinite(s['final']['fid']) and s['frozen_features_unchanged'] and s['sigma_unchanged']
        curve = []
        for line in (run / 'metrics.jsonl').read_text().splitlines():
            m = json.loads(line)
            if 'generation' in m:
                assert m['generation']['samples'] == 50000
                curve.append({'step': m['step'], 'fid': m['generation']['fid'],
                              'recon_mse': m['reconstruction']['recon_mse'],
                              'train_seconds': m['train_seconds']})
        best = min(curve, key=lambda c: c['fid'])
        rows.append({'name': run.name, 'curve': curve, 'best_observed': {**best,
                     'checkpoint': str(run / f"checkpoint_{best['step']:06d}.pt")}, **s})
    rows.sort(key=lambda r: r['final']['fid'])
    by_name = {r['name']: r for r in rows}
    control = by_name.get('control')
    lines = ['# CIFAR AE-GAN feature and selective-gradient scouts', '',
             f'{len(rows)}/{len(rows) + len(missing)} certified runs complete.', '',
             'All arms resume the original one-D 50k checkpoint (FID50k18.9012). '
             'ResNet34 replaces only the frozen discriminator backbone while retaining trainable heads and Adam state. '
             'This changes D scores immediately and tests replacement plus adaptation, not capacity alone. '
             'G growth adds identity-initialized refinements; reconstruction gradients are blocked only on those '
             'new parameters, retaining reconstruction updates on old G, E and prior. '
             'No extra D warmup, head reset, or change in the one-D update ratio.', '',
             '| Rank | Run | Final FID50k ↓ | Δ vs control ↓ | Test MSE ↓ | Train min | Wall min | G updates/s |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for rank, r in enumerate(rows, 1):
        delta = r['final']['fid'] - control['final']['fid'] if control else None
        r['fid_delta_vs_control'] = delta
        delta_text = f'{delta:+.4f}' if delta is not None else 'pending'
        speed = (r['final']['step'] - r['start_step']) / r['train_seconds']
        lines.append(f"| {rank} | {r['name']} | {r['final']['fid']:.4f} | {delta_text} | "
                     f"{r['final']['reconstruction']['recon_mse']:.5f} | {r['train_seconds']/60:.2f} | "
                     f"{r['total_seconds']/60:.2f} | {speed:.2f} |")
    lines += ['', 'FID uses 50,000 prior samples, EMA G/prior, and the existing TF-compatible Inception '
              'CIFAR train50k reference. Reconstruction uses the 10k test split. Same seed and parent state '
              'throughout; these are architecture interventions, not seed experiments. Final endpoints '
              'determine ranking. Intermediate minima are reported separately.', '',
              '| Run | Global step | FID50k ↓ | Test MSE ↓ | Train min since 50k |',
              '|---|---:|---:|---:|---:|']
    for r in rows:
        for c in r['curve']:
            lines.append(f"| {r['name']} | {c['step']} | {c['fid']:.4f} | {c['recon_mse']:.5f} | {c['train_seconds']/60:.2f} |")
    lines += ['', '## Interpretation and recommendation', '',
              'The earlier all-gradient G-growth scout ended at FID22.8981; its contemporaneous control was19.2033. '
              'Use that as historical context for selective routing, not a same-run paired estimate. '
              'Reconstruction is never an unconditional-generation benchmark.']
    effects = None
    if missing:
        lines.append('Pending or uncertified: ' + ', '.join(missing) + '. Do not select a winner yet.')
    elif set(by_name) == {'control', 'grow_g_adv', 'resnet34', 'both'}:
        f = {k: v['final']['fid'] for k, v in by_name.items()}
        effects = {'g_only_delta': f['grow_g_adv'] - f['control'],
                   'd_only_delta': f['resnet34'] - f['control'],
                   'both_delta': f['both'] - f['control'],
                   'interaction': f['both'] - f['grow_g_adv'] - f['resnet34'] + f['control']}
        lines.append(f"G growth with selective routing: {effects['g_only_delta']:+.4f} FID; ResNet34: {effects['d_only_delta']:+.4f}; "
                     f"both: {effects['both_delta']:+.4f}, relative to the contemporaneous control.")
        lines.append(f"Factorial interaction (both − G − D + control): {effects['interaction']:+.4f}. "
                     'A negative value means the combined result improves more than the sum of the individual '
                     'changes on this FID scale. This single trajectory provides no uncertainty estimate or proof of a unique bottleneck.')
        winner = rows[0]
        if winner['name'] == 'control':
            lines.append('Recommendation: do not promote the tested expansions. They did not beat unchanged continuation at the matched endpoint.')
        else:
            gain = f['control'] - winner['final']['fid']
            cost = winner['train_seconds'] / control['train_seconds']
            lines.append(f"Best final endpoint: **{winner['name']}**, improving {gain:.4f} FID for {cost:.2f}× training time. "
                         'Review its last two evaluations and compute cost before extending; small gains alone may not justify expansion.')
            if gain >= 1:
                lines.append('Recommendation: this is a candidate for a longer checkpoint continuation, subject to its recent trend and cost.')
            else:
                lines.append('Recommendation: the final gain is under one FID point; given the stated cost preference, discuss the tradeoff before a long promotion.')
        best_run = min(rows, key=lambda r: r['best_observed']['fid'])
        best = best_run['best_observed']
        lines.append(f"Best observed measurement: {best['fid']:.4f}, {best_run['name']} at {best['step']:,}. "
                     f"Checkpoint: `{best['checkpoint']}`. This minimum is selected across evaluations.")
        lines.append('Target below 13 reached at a final endpoint.' if winner['final']['fid'] < 13 else 'Target below 13 remains unmet at the final endpoints.')
    report.mkdir(parents=True, exist_ok=True)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for r in rows:
            c = r['curve']
            steps = [50] + [v['step']/1000 for v in c]
            fids = [18.901233842] + [v['fid'] for v in c]
            axes[0].plot(steps, fids, '-o', label=r['name'])
            axes[1].plot([0] + [v['train_seconds']/60 for v in c], fids, '-o', label=r['name'])
            axes[2].plot([v['step']/1000 for v in c], [v['recon_mse'] for v in c], '-o', label=r['name'])
        for ax, title, xlabel in zip(axes, ['Generation FID50k', 'FID versus training cost', 'Test reconstruction MSE'],
                                     ['Global updates (thousands)', 'Training minutes since checkpoint', 'Global updates (thousands)']):
            ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(alpha=.2)
        axes[0].axhline(13, color='black', linestyle=':', label='target 13')
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(report / 'curves.png', dpi=180); plt.close(fig)
        lines += ['', '![Learning curves and cost](curves.png)']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    (report / 'leaderboard.json').write_text(json.dumps({'complete': not missing, 'missing': missing,
                                                       'effects': effects, 'rows': rows}, indent=2, allow_nan=False) + '\n')
    print('\n'.join(lines), flush=True)
    return int(bool(missing))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config_manifest', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    sys.exit(analyze(args.config_manifest, args.report))
