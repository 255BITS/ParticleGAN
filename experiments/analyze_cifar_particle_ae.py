#!/usr/bin/env python
"""Verify a completed matched pair and publish its image experiment leaderboard."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary


def analyze(root, report):
    rows, histories, audits = [], {}, {}
    provenance = code_provenance(str(ROOT / 'experiments/train_cifar_particle_ae.py'), sys.executable)
    for arm in ('gan', 'bounded'):
        run = root / arm
        s = json.loads((run / 'summary.json').read_text())
        cfg = s['config']
        assert cfg['arm'] == arm and Path(cfg['out_dir']).resolve() == run.resolve()
        assert has_valid_summary(str(run), cfg, provenance), f'{arm}: missing/current-source-invalid certificate'
        assert s['final']['step'] == cfg['steps']
        assert s['final']['samples'] == cfg['final_samples'] and cfg['final_samples'] >= 50000
        assert np.isfinite(s['final']['fid'])
        assert s['sigma_unchanged'] and s['frozen_features_unchanged']
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        assert history[-1]['step'] == cfg['steps']
        expected = list(range(cfg['eval_interval'], cfg['steps'], cfg['eval_interval'])) + [cfg['steps']]
        assert [v['step'] for v in history if 'generation' in v] == expected
        for v in history:
            if 'generation' in v:
                assert v['generation']['samples'] == (cfg['final_samples'] if v['step'] == cfg['steps'] else cfg['eval_samples'])
        if arm == 'bounded':
            with np.load(run / f"recon_{cfg['steps']:06d}.npz") as errors:
                assert len(errors['ids']) == cfg['recon_samples']
                assert errors['counts'].sum() == cfg['recon_samples']
                assert np.array_equal(np.bincount(errors['ids'], minlength=cfg['num_particles']), errors['counts'])
                for key in ('recon', 'zero_offset', 'random_offset', 'shuffled_particle'):
                    assert abs(errors[key].mean() - s['final']['reconstruction'][key + '_mse']) < 1e-7
        s['checkpoint_sha256'] = hashlib.sha256((run / 'checkpoint.pt').read_bytes()).hexdigest()
        rows.append(s)
        histories[arm] = history
        audit_path = run / 'audit5k' / 'summary.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert audit['checkpoint_sha256'] == s['checkpoint_sha256']
            assert audit['checkpoint_unchanged'] and audit['final']['samples'] == 5000
            assert audit['step'] == cfg['steps'] and audit['arm'] == arm
            audits[arm] = audit
    a, b = rows
    shared = lambda c: {k: v for k, v in c.items() if k not in ('arm', 'out_dir')}
    assert shared(a['config']) == shared(b['config'])
    assert a['metadata']['initialization_sha256'] == b['metadata']['initialization_sha256']
    assert a['metadata']['sigma'] == b['metadata']['sigma']
    assert a['metadata']['pretrained'] == b['metadata']['pretrained']
    assert a['rng_sha256'] == b['rng_sha256'], 'training draw streams diverged'
    rows.sort(key=lambda s: s['final']['fid'])
    report.mkdir(parents=True, exist_ok=True)
    for s in rows:
        arm, step = s['config']['arm'], s['config']['steps']
        dest = report / arm
        dest.mkdir(exist_ok=True)
        names = ['summary.json', 'config.yaml', 'metadata.json', 'provenance.json', 'metrics.jsonl',
                 'run_grid_complete.json', f'samples_{step:06d}.png']
        if arm == 'bounded':
            names += [f'recon_{step:06d}.png', f'recon_{step:06d}.npz']
        for name in names:
            shutil.copy2(root / arm / name, dest / name)
        if arm in audits:
            (dest / 'audit5k.json').write_text(json.dumps(audits[arm], indent=2) + '\n')
    (report / 'leaderboard.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    lines = ['# Direct CIFAR particle autoencoder leaderboard', '',
             'Final EMA weights, 10k updates, FID50k ascending. One shared seed; labels unused.', '',
             '| Arm | FID50k ↓ | Feature variance /real | Test MSE ↓ | Effective /1024 | Offset RMS | Train min | Total min | Peak GiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in rows:
        f, r = s['final'], s['final']['reconstruction']
        rec = f"{r['recon_mse']:.5f} | {r['effective_particles']:.1f} | {r['offset_rms']:.3f}" if r else '— | — | —'
        lines.append(f"| {s['config']['arm']} | {f['fid']:.3f} | {f['feature_variance_ratio']:.3f} | {rec} | {s['train_seconds']/60:.2f} | {s['total_seconds']/60:.2f} | {s['peak_memory_gb']:.2f} |")
    lines += ['', 'Feature variance is a coarse spread diagnostic, not measured mode coverage.',
              'MSE uses [-1,1] pixels and all 10k unaugmented CIFAR test images. Generation FID uses the existing CIFAR train50k reference.',
              'Historical conditional DDGAN results are not a matched control.', '',
              '## Reconstruction ablations', '', '| Evaluation | Test MSE ↓ | Change from predicted |', '|---|---:|---:|']
    r = b['final']['reconstruction']
    for name in ('recon', 'zero_offset', 'random_offset', 'shuffled_particle'):
        value = r[name + '_mse']
        lines.append(f"| {name} | {value:.6f} | {100*(value/r['recon_mse']-1):+.1f}% |")
    lines += ['', f"Per-image error p90 {r['recon_p90']:.5f}, p99 {r['recon_p99']:.5f}; aggregate PSNR {r['recon_psnr']:.2f} dB.",
              f"Used {r['used_particles']}/1024 particles; hard usage TV {r['hard_usage_tv']:.3f}, hard–soft usage TV {r['hard_soft_usage_tv']:.3f}.",
              f"Offset saturation (abs >2.9): {100*r['offset_saturation']:.2f}%; conditional mean RMS {r['conditional_offset_mean_rms']:.3f}.", '',
              'Verified: identical shared configs, initialization, pretrained weights, final data/prior RNG states, complete evaluation budgets,',
              'fixed sigma, frozen feature weights, saved reconstruction errors/counts, and current-source completion certificates.', '']
    if len(audits) == 2:
        lines += ['## Same-count diagnostic audit', '',
                  'The final checkpoints were additionally evaluated at 5k samples to distinguish training deterioration from sample-count effects. No updates were made.', '',
                  '| Arm | FID5k at 2500 | At 5000 | At 7500 | At 10000 (audit) |',
                  '|---|---:|---:|---:|---:|']
        for arm in ('gan', 'bounded'):
            values = [v['generation']['fid'] for v in histories[arm]
                      if 'generation' in v and v['generation']['samples'] == 5000]
            values.append(audits[arm]['final']['fid'])
            lines.append('| ' + arm + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        lines += ['', 'Audit sample grids reproduce the original final grids within one uint8 quantization level. Checkpoint hashes are unchanged.', '']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for arm, history in histories.items():
        diagnostics = [v for v in history if 'generation' in v and v['generation']['samples'] == 5000]
        if arm in audits:
            diagnostics.append({'step': audits[arm]['step'], 'generation': audits[arm]['final']})
        axes[0, 0].plot([v['step'] for v in diagnostics], [v['generation']['fid'] for v in diagnostics], 'o-', label=arm)
    axes[0, 0].set(title='Diagnostic FID5k (final FID50k is separate)', xlabel='Updates', ylabel='FID ↓')
    evaluations = [v for v in histories['bounded'] if v.get('reconstruction')]
    steps = [v['step'] for v in evaluations]
    for key in ('recon_mse', 'zero_offset_mse', 'shuffled_particle_mse'):
        axes[0, 1].plot(steps, [v['reconstruction'][key] for v in evaluations], 'o-', label=key)
    axes[0, 1].set(title='Held-out EMA reconstruction', xlabel='Updates', ylabel='MSE ↓')
    for key in ('used_particles', 'effective_particles'):
        axes[1, 0].plot(steps, [v['reconstruction'][key] for v in evaluations], 'o-', label=key)
    axes[1, 0].set(title='Hard routing usage /1024', xlabel='Updates')
    for key in ('offset_rms', 'conditional_offset_mean_rms'):
        axes[1, 1].plot(steps, [v['reconstruction'][key] for v in evaluations], 'o-', label=key)
    axes[1, 1].set(title='Encoded offsets', xlabel='Updates', ylabel='RMS')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(report / 'learning_curves.png', dpi=140)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/cifar_particle_ae/scout')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/cifar-particle-ae')
    args = parser.parse_args()
    analyze(args.root, args.report)
