#!/usr/bin/env python
"""Audit and report the fixed-seed component-count/budget experiments."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from lib.mog_metrics import allocation_null, pass_metrics

OUT = ROOT / 'results/mog'
CRITERIA = json.loads((ROOT / 'configs/mog/stage1_criteria.json').read_text())
REFERENCES = ['stage0/C0_s1', 'noise_check/n400_r0_s1',
              'noise_check/n400_r1over32_s1', 'longer/n400_r0p03125_14k_s1',
              'refine_noise/n400_r1over40_14k_s1']


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                          for k, v in r.items()} for r in rows)


def load_rows(final=False):
    assert hashlib.sha256((OUT / 'results.csv').read_bytes()).hexdigest() == CRITERIA['baseline_sha256']
    paths = [(OUT / name, 'reference') for name in REFERENCES]
    for phase in ('component_scale', 'scale_longer'):
        for config in sorted((ROOT / 'configs/mog' / phase).glob('*.yaml')):
            cfg = yaml.safe_load(config.read_text())
            folder = ROOT / cfg['out_dir']
            if not (folder / 'run_grid_complete.json').exists():
                if final:
                    raise RuntimeError(f'Incomplete: {folder}')
                continue
            assert json.loads((folder / 'summary.json').read_text())['config'] == cfg
            paths.append((folder, phase))
    rows = []
    for folder, phase in paths:
        summary_bytes = (folder / 'summary.json').read_bytes()
        summary = json.loads(summary_bytes)
        certificate = json.loads((folder / 'run_grid_complete.json').read_text())
        assert hashlib.sha256(summary_bytes).hexdigest() == certificate['summary_sha256']
        cfg, m = summary['config'], summary['final']
        assert cfg == certificate['config'] and cfg['seed'] == 1
        assert cfg['final_samples'] == 200000
        if phase != 'reference':
            assert cfg['mog_pass_criteria'] == CRITERIA['thresholds']
            assert summary['provenance'] == certificate['provenance']
            traces = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
            expected = cfg['epochs'] * cfg['steps_per_epoch'] // cfg['log_interval']
            assert len(traces) == expected, folder
            assert len({t['step'] for t in traces}) == expected
            assert all(np.isfinite(t[k]) for t in traces for k in ('hq_ratio', 'width_ratio', 'kl_balance'))
        row = dict(run=folder.name, phase=phase, **cfg, **m,
                   steps=cfg['epochs'] * cfg['steps_per_epoch'],
                   git_sha=summary['git_sha'], total_seconds=summary['total_seconds'],
                   train_seconds=summary['train_seconds'])
        row.update(pass_metrics(m, CRITERIA['thresholds']))
        row['passed_c0_mean'] = pass_metrics(m, CRITERIA['mean_thresholds'])['passed']
        row['width_error'] = abs(m['width_ratio'] - 1)
        t = CRITERIA['thresholds']
        row['failed_metrics'] = ','.join(k for k, good in {
            'coverage': m['modes'] == 100, 'HQ': m['hq_ratio'] >= t['hq_ratio_min'],
            'width': t['width_ratio_min'] <= m['width_ratio'] <= t['width_ratio_max'],
            'balance': m['kl_balance'] <= t['kl_balance_max']}.items() if not good)
        rows.append(row)
    c0 = next(r for r in rows if r['run'] == 'C0_s1')
    for row in rows:
        row['quality_dominates_c0_s1'] = (
            row['modes'] >= c0['modes'] and row['hq_ratio'] >= c0['hq_ratio']
            and row['width_error'] <= c0['width_error'] and row['kl_balance'] <= c0['kl_balance']
            and any(row[k] != c0[k] for k in ('hq_ratio', 'width_error', 'kl_balance')))
        row['pareto_quality'] = not any(
            o['modes'] >= row['modes'] and o['hq_ratio'] >= row['hq_ratio']
            and o['width_error'] <= row['width_error'] and o['kl_balance'] <= row['kl_balance']
            and any(o[k] != row[k] for k in ('modes', 'hq_ratio', 'width_error', 'kl_balance'))
            for o in rows)
        def same_recipe(other):
            keys = ('num_particles', 'steps', 'standardize', 'particle_lr_multiplier')
            return (all(row[k] == other[k] for k in keys)
                    and (row['particle_beta1'] or row['beta1']) == (other['particle_beta1'] or other['beta1']))
        atoms = next((o for o in rows if o['sigma_rel'] == 0 and same_recipe(o)), None)
        row['matched_atoms'] = atoms['run'] if atoms and row['sigma_rel'] > 0 else None
        if row['matched_atoms']:
            row['delta_hq_vs_atoms'] = row['hq_ratio'] - atoms['hq_ratio']
            row['delta_width_error_vs_atoms'] = row['width_error'] - atoms['width_error']
            row['delta_kl_vs_atoms'] = row['kl_balance'] - atoms['kl_balance']
    return rows


def rank(r):
    return (-r['passed'], -r['hq_ratio'], r['width_error'], r['kl_balance'])


def table(rows):
    lines = ['| Setting | N | Steps | HQ/real | Width/real | KL | Pass | Failed | Seconds |',
             '|---|---:|---:|---:|---:|---:|:---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['num_particles']} | {r['steps']} | {r['hq_ratio']:.6f} | "
                     f"{r['width_ratio']:.4f} | {r['kl_balance']:.5f} | {'yes' if r['passed'] else 'no'} | "
                     f"{r['failed_metrics'] or '—'} | {r['total_seconds']:.1f} |")
    return '\n'.join(lines)


def plots(rows):
    refs = list(csv.DictReader((OUT / 'results.csv').open()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {0.: 'tab:blue', .025: 'tab:orange', .0625: 'tab:green'}
    for axes_row, mult in zip(axes, (10, 1)):
        for ax, key in zip(axes_row, ('hq_ratio', 'width_ratio', 'kl_balance')):
            for radius in (0., .025, .0625):
                group = sorted([r for r in rows if r['phase'] == 'component_scale' and r['standardize']
                                and r['particle_lr_multiplier'] == mult and r['sigma_rel'] == radius],
                               key=lambda r: r['num_particles'])
                if group:
                    ax.plot([r['num_particles'] for r in group], [r[key] for r in group],
                            '-o', color=colors[radius], label=f'std on, r={radius:g}')
                unstd = [r for r in rows if r['phase'] == 'component_scale' and not r['standardize']
                         and r['particle_lr_multiplier'] == mult and r['sigma_rel'] == radius]
                if unstd:
                    ax.scatter([r['num_particles'] for r in unstd], [r[key] for r in unstd],
                               marker='x', color=colors[radius], s=60, label=f'std off, r={radius:g}')
            baseline = [float(r[key]) for r in refs if r['arm'] == 'C0']
            ax.axhline(np.mean(baseline), ls='--', color='black', label='C0 mean (3 seeds)')
            ax.axhspan(min(baseline), max(baseline), color='gray', alpha=.15, label='C0 range')
            if key == 'width_ratio':
                ax.axhline(1, ls=':', color='black')
            if key == 'kl_balance':
                ns = [1600, 6400, 20000]
                ax.plot(ns, [allocation_null(n)['kl_balance'] for n in ns], ':', color='gray', label='allocation null')
            ax.set(xscale='log', xlabel='Number of components', ylabel=key,
                   title=f"{'Fast: LR x10, beta1=.5' if mult == 10 else 'Shipped: LR x1, beta1=0'}")
            ax.set_xticks([1600, 6400, 20000], ['1.6k', '6.4k', '20k'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle('Component-count screen: 7k steps, seed 1 only')
    fig.tight_layout(); fig.savefig(OUT / 'component_scale_count.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, xkey in zip(axes[0], ('width_ratio', 'kl_balance')):
        for phase, marker in [('reference', 'x'), ('component_scale', 'o'), ('scale_longer', 's')]:
            group = [r for r in rows if r['phase'] == phase]
            ax.scatter([r[xkey] for r in group], [r['hq_ratio'] for r in group], marker=marker, label=phase)
        ax.axhline(CRITERIA['thresholds']['hq_ratio_min'], color='gray', ls='--')
        ax.set(xlabel=xkey, ylabel='HQ / real HQ', title='All settings, including failures')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    passing = [r for r in rows if r['passed']]
    for ax, xkey in zip(axes[1], ('width_ratio', 'kl_balance')):
        for i, r in enumerate(passing):
            label = f"C0 {r['steps']//1000}k" if r['run'].startswith('C0') else (
                f"MoG N={r['num_particles']}, r=1/{round(1/r['sigma_rel'])}, {r['steps']//1000}k")
            ax.scatter(r[xkey], r['hq_ratio'], color=f'C{i}', marker='x' if r['sigma_rel'] == 0 else 'o',
                       label=label, s=55)
        ax.axhline(CRITERIA['thresholds']['hq_ratio_min'], color='gray', ls='--', label='Frozen HQ floor')
        ax.set(xlabel=xkey, ylabel='HQ / real HQ', title='Zoom: settings passing the frozen envelope')
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc='best')
    fig.suptitle('Quality tradeoffs; single seed, training/table costs reported separately')
    fig.tight_layout(); fig.savefig(OUT / 'component_scale_tradeoffs.png', dpi=160); plt.close(fig)
    longer = [r for r in rows if r['phase'] == 'scale_longer']
    if longer:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for row in longer:
            trace = [json.loads(line) for line in (ROOT / row['out_dir'] / 'metrics.jsonl').read_text().splitlines()]
            for ax, key in zip(axes, ('hq_ratio', 'width_ratio', 'kl_balance')):
                line, = ax.plot([t['step'] for t in trace], [t[key] for t in trace],
                                label=row['run'].replace('_28k_s1', ''), alpha=.8, lw=1)
                ax.scatter([row['steps']], [row[key]], color=line.get_color(), marker='s', s=25)
        for ax, key in zip(axes, ('hq_ratio', 'width_ratio', 'kl_balance')):
            baseline = [float(r[key]) for r in refs if r['arm'] == 'C0']
            ax.axhline(np.mean(baseline), ls='--', color='black', label='7k C0 historical mean')
            ax.set(xlabel='Training step', ylabel=key); ax.grid(alpha=.2)
            if key == 'kl_balance':
                ax.set_yscale('log')
            ax.legend(fontsize=6)
        fig.suptitle('Longer-budget trajectories: 20k eval draws per point; final squares use 200k')
        fig.tight_layout(); fig.savefig(OUT / 'component_scale_training.png', dpi=160); plt.close(fig)


def validate_prefixes(rows):
    checks = []
    for row in rows:
        if row['phase'] != 'scale_longer':
            continue
        if row['run'] == 'C0_28k_s1':
            parent = OUT / 'stage0/C0_s1'
        elif row['run'] == 'n400_r1over40_28k_s1':
            parent = OUT / 'refine_noise/n400_r1over40_14k_s1'
        else:
            parent = OUT / 'component_scale' / row['run'].replace('_28k_', '_7k_')
        cfg = json.loads((parent / 'summary.json').read_text())['config']
        cutoff = cfg['epochs'] * cfg['steps_per_epoch'] * cfg['lr_anneal_start']
        def lines(folder):
            return [line for line in (folder / 'log.txt').read_text().splitlines()
                    if line.startswith('[epoch ') and float(re.search(r'step (\d+)', line).group(1)) < cutoff]
        old, new = lines(parent), lines(ROOT / row['out_dir'])
        assert old and old == new, row['run']
        current = json.loads((ROOT / row['out_dir'] / 'summary.json').read_text())['config']
        differences = {k: [cfg.get(k), current.get(k)] for k in sorted(cfg.keys() | current.keys())
                       if cfg.get(k) != current.get(k)}
        assert set(differences) <= {'epochs', 'out_dir', 'mog_pass_criteria'}, differences
        checks.append(dict(run=row['run'], parent=parent.name, exact_prefix_lines=len(old),
                           before_step=cutoff, config_differences=differences))
    (OUT / 'component_scale_prefix_check.json').write_text(json.dumps(checks, indent=2) + '\n')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    rows = load_rows(args.final)
    leaderboard = sorted(rows, key=rank)
    save_csv(OUT / 'component_scale_results.csv', rows)
    save_csv(OUT / 'component_scale_leaderboard.csv', leaderboard)
    print(table(leaderboard))
    if not args.final:
        return
    prefix_checks = validate_prefixes(rows)
    plots(rows)
    nulls = [allocation_null(n) for n in sorted({r['num_particles'] for r in rows})]
    (OUT / 'component_scale_nulls.json').write_text(json.dumps(nulls, indent=2) + '\n')
    new = [r for r in rows if r['phase'] != 'reference']
    new_mog = [r for r in new if r['sigma_rel'] > 0]
    report = ['# More MoG components and training', '',
              'Exploratory single-seed configuration screen (seed 1), not a seed sweep or pass-rate estimate. '
              'All final metrics use 200k noise-on EMA samples and the same-size real reference. '
              'The C0 acceptance envelope remains frozen; passing it is weaker than beating C0 on every metric.', '',
              f"Completed {len(new)} new configurations, including {len(new_mog)} positive-noise MoGs. "
              f"{sum(r['passed'] for r in new_mog)} MoGs pass the envelope. "
              f"{sum(r['quality_dominates_c0_s1'] for r in new_mog)} MoGs dominate seed-1 C0 on measured coverage, HQ, width error and KL. "
              'That comparison ignores training and table costs and is not a statistical claim.', '',
              'Historical C0 across three seeds: HQ/real 0.99978, width/real 0.84343, KL 0.03478. '
              'The table uses the matching seed-1 C0 (width 0.79092, KL 0.03750), which was the '
              'weakest historical C0 width/balance result. The per-run CSV also records passed_c0_mean '
              'as a sensitivity check; no new repetitions were run.', '',
              '## Outcome and recommendation', '',
              '**MoG now has passing settings at both 400 and 20,000 components.** '
              'The 400-component r=1/40 model passes at 28k steps: HQ/real 0.999535, width/real 0.92636, '
              'KL 0.02888. This is 50 times fewer components and four times the training steps of original C0. '
              'It improves width and balance relative to original C0, but HQ is lower. Its raw scale grows '
              '4.11 times, a recorded optimizer/gauge warning even though standardized reads keep output '
              'latent scale controlled. There is no matched 400-atom 28k control yet.', '',
              '**At 20k components, extra training helps both priors.** At 28k, the unstandardized r=1/16 '
              'MoG has HQ/real 1.000046, width/real 0.95429, KL 0.026566; matched atoms have 0.999110, '
              '0.94375, 0.026554. MoG has better measured HQ and slightly better width, with effectively '
              'tied balance (KL difference +0.000012). This is not evidence of universal superiority '
              'from one training seed. r=1/40 also passes, with width 0.95669 and KL 0.02742.', '',
              'The 20k r=1/16, 28k result is the only new MoG that also clears all historical C0-mean '
              'thresholds. It uses four times the training budget. The unstandardized 20k r=1/40 setting '
              'already passes the frozen envelope at 7k, but not all C0-mean thresholds. Raw scale grows '
              'only about 11–13% in the longer 20k runs; effective sigma/spacing increases rather than '
              'collapsing to zero. Their effective-radius drift flags are retained.', '',
              '**More components alone are not the solution.** At 7k, the fast small-table optimizer '
              'fails as N grows; switching to the shipped optimizer restores width but the standardized '
              '1,600/6,400-component models still miss balance. Unstandardized and standardized 20k '
              'settings differ substantially on this seed. These observations do not establish a monotonic '
              'count law or imply standardization is always harmful.', '',
              'Recommendation: keep both prior types. Use C0 as the simple 7k reference; use 400-component '
              'MoG at 28k when a compact prior with continuous support is valuable. For a larger-table '
              'quality comparison, retain 20k r=1/16 at 28k and matched 28k atoms. The next useful training '
              'comparison is a 400-atom 28k control, followed by lower-cost schedule refinements of the '
              'passing 400-component MoG. Avoid further increases in count before establishing a benefit '
              'at matched optimizer and budget. No additional training is left running.', '',
              '## Leaderboard', '', table(leaderboard), '',
              'Rank: frozen pass first, then HQ, then distance of width from real, then balance. '
              'Use the separate columns and quality frontier when a different tradeoff matters.', '',
              '## Interpretation limits', '',
              '- N>1,024 uses sampled-row raw VICReg, as shipped; N=400 references use the full table. '
              'At fixed r, increasing N also changes initial spacing and absolute sigma.',
              '- Fast optimizer means particle LR multiplier 10 (initial LR 0.06), beta1=0.5. '
              'Shipped means multiplier 1 (LR 0.006), beta1=0. No generator/discriminator retuning.',
              '- Standardized reads are used except explicitly named std0 controls and C0. '
              'Raw scale and effective-radius drift flags remain in the per-run CSV.',
              '- Longer runs restart from the same seed and scale the delayed cosine schedule to their budget; '
              'they do not resume EMA checkpoints. Matched-budget controls are needed before crediting noise for budget gains.',
              '- Purity is HQ-conditioned. With 200k samples and 20k components, each component gets only about '
              'ten evaluation draws on average; per-component diagnostics are less precise than at N=400.',
              '- Times include metrics and output artifacts; concurrent execution changes elapsed time per run. '
              'These are observed experiment costs, not isolated throughput benchmarks.', '',
              '- Adding positive noise consumes additional training RNG draws; zero-noise follows the '
              'original RNG contract. Fixed seed does not mean identical subsequent random streams.', '',
              '## Matched noise versus atoms comparisons', '',
              'Same N, budget, standardization and optimizer. Positive HQ delta favors MoG; '
              'negative width-error and KL deltas favor MoG.', '',
              '| MoG run | Atoms control | Δ HQ/real | Δ absolute width error | Δ KL |',
              '|---|---|---:|---:|---:|']
    for r in rows:
        if r['matched_atoms']:
            report.append(f"| {r['run']} | {r['matched_atoms']} | {r['delta_hq_vs_atoms']:+.6f} | "
                          f"{r['delta_width_error_vs_atoms']:+.5f} | {r['delta_kl_vs_atoms']:+.5f} |")
    report += ['',
              '## Geometry audit', '',
              '| Run | sigma | d0 | r_eff | Raw std final/initial | r flag | Scale flag | Purity |',
              '|---|---:|---:|---:|---:|:---:|:---:|---:|']
    for r in rows:
        report.append(f"| {r['run']} | {r['sigma']:.6g} | {r['d0']:.5g} | {r['r_eff']:.5g} | "
                      f"{r['raw_std_live_ratio']:.3f} | {r['r_eff_drift_flag']} | {r['raw_std_drift_flag']} | {r['purity_mean']:.5f} |")
    report += ['', '## Validation and artifacts', '',
               'Every new result was checked against its generated config, completion certificate, '
               'summary SHA-256, source provenance, frozen criteria, and expected metric-trace count. '
               'No training implementation changed. Historical results are reused.', '',
               f"Checked {sum(r['steps']//r['log_interval'] for r in new)} new trace entries. "
               f"All {len(prefix_checks)} longer runs have identical original-log prefixes to their shorter "
               'parents before the shorter schedule starts annealing; only budget, output path and '
               '(for historical C0) evaluation pass thresholds differ in config. '
               'Details: component_scale_prefix_check.json.', '',
               '- [Design and launch plan](COMPONENT_SCALE_PLAN.md)',
               '- [All per-run metrics](component_scale_results.csv) and [leaderboard](component_scale_leaderboard.csv)',
               '- [Component-count plot](component_scale_count.png) and [quality tradeoffs](component_scale_tradeoffs.png)',
               '- [Longer-budget trajectories](component_scale_training.png)',
               '- Raw outputs and JSONL traces: component_scale/<run>/ and scale_longer/<run>/.', '']
    (OUT / 'COMPONENT_SCALE.md').write_text('\n'.join(report))


if __name__ == '__main__':
    main()
