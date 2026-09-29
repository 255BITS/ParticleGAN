#!/usr/bin/env python
"""Rank completed train_100gaussians runs using their clean EMA sample metrics."""
import argparse
import json
import math
from pathlib import Path


def leaderboard(paths):
    runs = [(path, json.loads(path.read_text())) for path in paths]
    for path, run in runs:
        for key in ('modes', 'hq', 'sw1', 'mode_tv'):
            value = run.get('final', {}).get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{path}: missing/nonfinite {key}; use completed non-MoG summaries')
    runs.sort(key=lambda item: (-item[1]['final']['modes'], -item[1]['final']['hq'], item[1]['final']['sw1']))
    lines = ['# 100 Gaussians leaderboard', '',
             'Rank: coverage descending, HQ descending, then sliced W1 ascending. '
             'These are final clean EMA samples, not best-checkpoint results.', '',
             '| Rank | Run | Pack | Steps | Eval N | Modes ↑ | HQ ↑ | SW1 ↓ | Mode TV ↓ | Core σ ratio ≈1 | Train s ↓ |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rank, (path, run) in enumerate(runs, 1):
        cfg, metrics = run['config'], run['final']
        width = metrics.get('per_mode_core_ratio')
        width_text = f'{width:.3f}' if isinstance(width, (int, float)) and math.isfinite(width) else 'unmeasured'
        lines.append(f"| {rank} | {path.parent.name} | {cfg.get('pack_size', 1)} | "
                     f"{cfg['epochs'] * cfg['steps_per_epoch']} | {cfg['final_samples']} | "
                     f"{metrics['modes']}/100 | {metrics['hq']:.4f} | {metrics['sw1']:.4f} | "
                     f"{metrics['mode_tv']:.4f} | {width_text} | {run['train_seconds']:.1f} |")
    lines += ['', 'HQ is the fraction within 3σ (0.09) of a target center. '
              'Mode TV measures imbalance; SW1 measures distribution mismatch. '
              'A core width ratio below 0.8 flags overly narrow modes even with good coverage/HQ.', '',
              'Compare runs with the same update budget, batch size, evaluation count, seed and code version. '
              'A one-row board describes one experiment; it does not establish improvement.', '',
              'Recommended next comparison: a pack_size=1 control with the same no_regularizer settings '
              'and seed. This isolates packing. Then compare against the regularized default; '
              'that comparison also changes noise and optimizer stabilization.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='*', type=Path,
                        default=[Path('results/100gaussians/pacgan8_no_reg/summary.json')])
    parser.add_argument('--out', type=Path, default=Path('results/100gaussians/LEADERBOARD.md'))
    args = parser.parse_args()
    report = leaderboard(args.summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report, end='')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
