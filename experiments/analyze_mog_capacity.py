#!/usr/bin/env python
"""Audit and summarize the matched small-generator MoG/DDGAN screen."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from experiments.run_grid import has_valid_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/denoising/mog_capacity/manifest.json')
    parser.add_argument('--out', default='reports/denoising-toy/mog_capacity')
    parser.add_argument('--baseline-manifest', help='Shorter runs to audit and compare against')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in json.loads(Path(args.manifest).read_text()):
        cfg = read_config(path)
        directory = Path(cfg['out_dir'])
        summary = json.loads((directory / 'summary.json').read_text())
        assert has_valid_summary(str(directory), cfg, summary['provenance']), path
        history = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert history[-1]['step'] == cfg['steps'], path
        assert all(np.isfinite(row['conditional_sw1']) for row in history), path
        runs.append(dict(name=directory.name, cfg=cfg, summary=summary, history=history))
    assert len(runs) == 4
    # Keep plot order/colors stable even when the runner schedules DDGAN first.
    runs.sort(key=lambda run: (run['cfg']['model'] == 'ddgan', run['cfg']['sigma_rel']))
    assert {(r['cfg']['model'], r['cfg']['sigma_rel']) for r in runs} == {
        ('gan', 0.), ('gan', .025), ('ddgan', 0.), ('ddgan', .025)}
    ignored = {'model', 'sigma_rel', 'out_dir'}
    contexts = {json.dumps({k: v for k, v in r['cfg'].items() if k not in ignored}, sort_keys=True)
                for r in runs}
    assert len(contexts) == 1, 'unmatched settings'
    assert len({json.dumps(r['summary']['provenance'], sort_keys=True) for r in runs}) == 1
    assert runs[0]['cfg']['lr_floor'] == 1., 'screen requires constant learning rates'
    common_seconds = min(r['history'][-1]['train_seconds'] for r in runs)
    metrics = ['conditional_sw1', 'joint_hq', 'conditional_mode_tv', 'per_mode_core_ratio',
               'modes', 'cond_acc', 'tail_10sigma']
    rows = []
    for run in runs:
        summary, history = run['summary'], run['history']
        final = summary['final']
        timing = [h['train_seconds'] for h in history]
        row = {'name': run['name'], **{m: final[m] for m in metrics},
               'train_seconds': summary['train_seconds'],
               'wall_seconds': summary['total_seconds'],
               'sampling_ms_per_1000': final['sampling_ms_per_1000'],
               'unique_outputs': final['unique_outputs'],
               **{f'parameters_{k}': v for k, v in summary['parameters'].items()}}
        for metric in metrics:
            row['equal_time_' + metric] = float(np.interp(common_seconds, timing, [h[metric] for h in history]))
        # A descriptive tail change, not an estimated asymptote or a stopping rule.
        row['last_4k_sw1_change'] = history[-1]['conditional_sw1'] - history[-5]['conditional_sw1']
        late = [h for h in history if h['step'] > .8 * run['cfg']['steps']]
        for metric in metrics:
            values = [h[metric] for h in late if h[metric] is not None]
            row['late_mean_' + metric] = float(np.mean(values)) if values else None
        rows.append(row)
    rows.sort(key=lambda row: row['conditional_sw1'])
    with (out / 'leaderboard.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    contrasts = {}
    for model in ('gan', 'ddgan'):
        pair = {r['cfg']['sigma_rel']: r for r in runs if r['cfg']['model'] == model}
        contrasts[model + '_mog_minus_atoms'] = {
            m: pair[.025]['summary']['final'][m] - pair[0.]['summary']['final'][m] for m in metrics}
    result = {'completed': len(runs), 'common_train_seconds': common_seconds,
              'rows': rows, 'mog_minus_atoms': contrasts,
              'reference_floor': {r['name']: r['summary']['reference_floor'] for r in runs}}
    if args.baseline_manifest:
        baselines = {}
        for path in json.loads(Path(args.baseline_manifest).read_text()):
            cfg = read_config(path)
            directory = Path(cfg['out_dir'])
            summary = json.loads((directory / 'summary.json').read_text())
            assert has_valid_summary(str(directory), cfg, summary['provenance']), path
            baselines[directory.name] = (cfg, summary, directory)
        audits, comparisons = [], []
        timing_keys = {'train_seconds', 'wall_seconds', 'sampling_ms_per_1000'}
        for run in runs:
            cfg, summary, directory = baselines[run['name']]
            changed = {k for k in cfg if cfg[k] != run['cfg'][k]}
            assert changed == {'steps', 'out_dir'}, changed
            assert cfg['steps'] < run['cfg']['steps']
            assert summary['provenance'] == run['summary']['provenance']
            history = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            prefix = [h for h in run['history'] if h['step'] <= cfg['steps']]
            stripped = lambda rows: [{k: v for k, v in row.items() if k not in timing_keys} for row in rows]
            assert stripped(history) == stripped(prefix), f"training prefix differs: {run['name']}"
            audits.append(dict(name=run['name'], matching_checkpoints=len(prefix),
                               matching_through_step=cfg['steps'], exact_non_timing_match=True))
            comparisons.append(dict(name=run['name'], baseline_steps=cfg['steps'],
                                    final_steps=run['cfg']['steps'],
                                    baseline={m: summary['final'][m] for m in metrics},
                                    final={m: run['summary']['final'][m] for m in metrics}))
        result['prefix_audit'] = audits
        result['baseline_comparison'] = comparisons
        (out / 'prefix_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    milestones = []
    for run in runs:
        for row in run['history']:
            if row['step'] in (14000, 28000, 50000, 75000, 100000):
                milestones.append({'name': run['name'], 'step': row['step'],
                                   **{m: row[m] for m in metrics}})
    if milestones:
        with (out / 'milestones.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(milestones[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(milestones)
    cfg = runs[0]['cfg']
    lines = ['# Small-generator MoG/DDGAN comparison', '',
             'Four certified runs, one shared training seed; no seed sweep. '
             f"G width {cfg['generator_hidden']}, D width {cfg['hidden']}, depth {cfg['depth']}, "
             f"{cfg['num_particles']} learned components, {cfg['steps']:,} updates, constant LR. "
             'Atoms and MoG share standardization and optimizer settings; sigma_rel is 0 versus .025.', '',
             'Sorted by final conditional SW1 (lower is better), evaluated on 20,000 draws. '
             'HQ, coverage, balance, width and tails must also be considered. Width ratio should approach 1.', '',
             '| Rank | Run | Conditional SW1 ↓ | HQ % ↑ | Modes ↑ | Conditional TV ↓ | Core width | Tail % ↓ | Train seconds |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for rank, row in enumerate(rows, 1):
        lines.append(f"| {rank} | {row['name']} | {row['conditional_sw1']:.4f} | {100*row['joint_hq']:.2f} | "
                     f"{row['modes']} | {row['conditional_mode_tv']:.4f} | {row['per_mode_core_ratio']:.3f} | "
                     f"{100*row['tail_10sigma']:.2f} | {row['train_seconds']:.1f} |")
    lines += ['', f'Equal-time comparison at {common_seconds:.1f} training seconds. '
              'These are interpolated 8,192-draw checkpoint metrics, not the 20k-draw final scores.', '',
              '| Run | Equal-time conditional SW1 ↓ | Last 4k SW1 change ↓ | G parameters | Prior parameters | Sampling ms / 1k |',
              '|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda row: row['equal_time_conditional_sw1']):
        lines.append(f"| {row['name']} | {row['equal_time_conditional_sw1']:.4f} | "
                     f"{row['last_4k_sw1_change']:+.4f} | {row['parameters_G']} | "
                     f"{row['parameters_prior']} | {row['sampling_ms_per_1000']:.2f} |")
    lines += ['', 'Mean checkpoint metrics over the final 20% of updates (8,192 draws per checkpoint). '
              'These summarize the training trajectory, not independent trials or seed uncertainty.', '',
              '| Run | Mean HQ % | Mean SW1 ↓ | Mean conditional TV ↓ | Mean width |',
              '|---|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda row: row['late_mean_joint_hq'], reverse=True):
        lines.append(f"| {row['name']} | {100*row['late_mean_joint_hq']:.2f} | "
                     f"{row['late_mean_conditional_sw1']:.4f} | {row['late_mean_conditional_mode_tv']:.4f} | "
                     f"{row['late_mean_per_mode_core_ratio']:.3f} |")
    lines += ['', 'The GAN/DDGAN comparison changes the training objective, conditioning inputs, and sampling computation. '
              'It does not isolate representational capacity. A small-G screen also needs a matched wider-G '
              'comparison before claiming a capacity interaction. A fixed-budget loss does not rule out a later crossover.', '',
              'Artifacts: `leaderboard.csv`, `comparison.json`, `curves.png`, `samples.png`. '
              f"Raw logs/checkpoints: `{Path(cfg['out_dir']).parent}/<run>/`.", '']
    if args.baseline_manifest:
        lines += ['All non-timing checkpoint metrics exactly reproduce the shorter runs through '
                  f"{comparisons[0]['baseline_steps']:,} updates. See `prefix_audit.json`.", '',
                  '| Run | HQ before → after | Modes before → after | Width before → after | SW1 before → after |',
                  '|---|---:|---:|---:|---:|']
        for row in comparisons:
            a, b = row['baseline'], row['final']
            lines.append(f"| {row['name']} | {100*a['joint_hq']:.2f}% → {100*b['joint_hq']:.2f}% | "
                         f"{a['modes']} → {b['modes']} | {a['per_mode_core_ratio']:.3f} → "
                         f"{b['per_mode_core_ratio']:.3f} | {a['conditional_sw1']:.4f} → {b['conditional_sw1']:.4f} |")
    (out / 'TABLE.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for run in runs:
        history = run['history']
        for row, xkey in enumerate(('step', 'train_seconds')):
            for col, metric in enumerate(('conditional_sw1', 'joint_hq', 'conditional_mode_tv', 'per_mode_core_ratio')):
                ax = axes[row, col]
                ax.plot([h[xkey] for h in history], [h[metric] for h in history], label=run['name'])
                ax.set(xlabel=xkey, ylabel=metric)
                if metric == 'per_mode_core_ratio':
                    ax.set_yscale('log')
                    ax.axhline(1, color='grey', linestyle=':', linewidth=.8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, run in zip(axes, runs):
        data = np.load(Path(run['cfg']['out_dir']) / 'final_samples.npz')
        ax.scatter(*data['x'][:8000].T, c=data['c'][:8000], cmap='tab10', s=1, alpha=.35, vmin=0, vmax=9)
        ax.set(title=run['name'], xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect='equal')
    fig.tight_layout()
    fig.savefig(out / 'samples.png', dpi=140)
    plt.close(fig)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
