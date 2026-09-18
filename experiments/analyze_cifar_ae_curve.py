#!/usr/bin/env python
"""Summarize the saved long-run checkpoints at a consistent FID50k sample count."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary


def main():
    run = ROOT / 'runs/cifar_particle_ae/lazy_long/n08'
    audits = run.parent / 'fid50k_curve'
    report = ROOT / 'reports/cifar-particle-ae/lazy-long'
    summary = json.loads((run / 'summary.json').read_text())
    assert has_valid_summary(str(run), summary['config'], code_provenance(
        str(ROOT / 'experiments/train_cifar_particle_ae.py'), sys.executable))
    history = {r['step']: r for r in map(json.loads, (run / 'metrics.jsonl').read_text().splitlines())
               if 'generation' in r}
    final_audit = json.loads((run / 'audit5k/summary.json').read_text())
    assert final_audit['checkpoint_sha256'] == hashlib.sha256((run / 'checkpoint.pt').read_bytes()).hexdigest()
    rows = []
    for step in (5000, 10000, 15000, 20000, 25000, 30000):
        path = run / f'checkpoint_{step:06d}.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if step == 30000:
            assert digest == final_audit['checkpoint_sha256'] and final_audit['checkpoint_unchanged']
            measured = summary['final']
            fid5k = final_audit['final']['fid']
        else:
            audit = json.loads((audits / f'step_{step:06d}' / 'summary.json').read_text())
            assert audit['step'] == step and audit['checkpoint_sha256'] == digest
            assert audit['checkpoint_unchanged'] and audit['grid_max_uint8_difference'] <= 1
            assert audit['config'] == summary['config']
            assert audit['fid_protocol'] == summary['metadata']['fid_protocol']
            measured = audit['final']
            fid5k = history[step]['generation']['fid']
        assert measured['samples'] == 50000
        rows.append({'step': step, 'fid50k': measured['fid'], 'fid5k': fid5k,
                     'recon_mse': history[step]['reconstruction']['recon_mse'],
                     'train_minutes': history[step]['train_seconds'] / 60,
                     'checkpoint_sha256': digest})
    ranked = sorted(rows, key=lambda r: r['fid50k'])
    rank = {r['step']: i for i, r in enumerate(ranked, 1)}
    early, final = rows[1], rows[-1]
    improvement = early['fid50k'] - final['fid50k']
    recon_improvement = 100 * (1 - final['recon_mse'] / early['recon_mse'])
    lines = ['# AE-GAN checkpoint curve: consistent FID50k', '',
             'All six checkpoints now use 50,000 generated samples against the same CIFAR train50k reference, '
             'EMA G/prior, evaluation RNG seed, batch size and TF-compatible Inception protocol. '
             'Earlier checkpoints were evaluated without training; the existing final result was reused.', '',
             '| Updates | FID50k ↓ | Rank | Test MSE ↓ | Training min |',
             '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['step']:,} | {r['fid50k']:.3f} | {rank[r['step']]} | {r['recon_mse']:.5f} | {r['train_minutes']:.2f} |")
    lines += ['', f"The final checkpoint improves FID by only {improvement:.3f} from 10k to 30k updates, "
              f"while reconstruction MSE improves {recon_improvement:.1f}%. "
              'Generation improves quickly through 10k, then fluctuates within a narrow range through 30k. '
              'The curve shows a modest late recovery, without a large delayed improvement in FID.', '',
              '![FID50k and reconstruction across saved checkpoints](curve.png)', '',
              '## Interpretation and next decision', '',
              'The apparent large final jump in the original log mixed FID5k with FID50k. '
              f"At the same final checkpoint, those scores are {final['fid5k']:.3f} and {final['fid50k']:.3f}. "
              'The new curve removes that sample-count mismatch. It does not establish a causal conflict '
              'between reconstruction and generation, and cannot rule out improvements after 30k.', '',
              'Use roughly 10k updates as a cost-effective budget for the next configuration comparison. '
              'These results alone do not justify assuming a much longer unchanged run will produce a large '
              'FID gain. Keep the reconstruction-lag hypothesis open, but distinguish it from measured evidence. '
              'No additional training was launched.', '',
              '## Audit', '',
              'Five earlier checkpoints were evaluated across GPUs 0 and 1. All checkpoint hashes remained '
              'unchanged; first-100 sample grids reproduced within one uint8 quantization level. '
              'The original final-run completion certificate remains valid. Small floating-point differences '
              'are permitted by the original CUDA/TF32 protocol. Reconstruction uses all 10k test images.', '',
              'Evaluation logs: `runs/cifar_particle_ae/lazy_long/fid50k_curve/PIPELINE.log`.', '',
              '```sh', '.venv/bin/python experiments/audit_cifar_particle_ae.py \\',
              '  runs/cifar_particle_ae/lazy_long/n08 \\',
              '  --checkpoint checkpoint_010000.pt --samples 50000 --out /tmp/cifar_10k_fid50k',
              '.venv/bin/python experiments/analyze_cifar_ae_curve.py', '```']
    report.mkdir(parents=True, exist_ok=True)
    (report / 'curve.json').write_text(json.dumps(rows, indent=2) + '\n')
    (report / 'README.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    x = [r['step'] / 1000 for r in rows]
    for ax, metric, label, color in zip(axes, ('fid50k', 'recon_mse'),
                                       ('FID50k (lower is better)', 'Test reconstruction MSE'), ('#2369bd', '#c56b18')):
        ax.plot(x, [r[metric] for r in rows], 'o-', color=color)
        ax.set(xlabel='Training updates (thousands)', ylabel=label, xticks=x)
        ax.grid(alpha=.25)
    fig.suptitle('Same trajectory: generation levels off while reconstruction improves')
    fig.savefig(report / 'curve.png', dpi=180)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
