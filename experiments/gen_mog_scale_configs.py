#!/usr/bin/env python
"""Single-seed component-count screen; reuse the existing MoG training recipe."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.gen_mog_configs import frozen_criteria, save
from experiments.train_100gaussians import DEFAULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['screen', 'optimizer', 'longer'], default='screen')
    parser.add_argument('--selected', help='Completed positive-noise run name for longer-budget comparison')
    parser.add_argument('--extra', action='append', default=[], help='Additional completed screen run to extend to 28k')
    args = parser.parse_args()
    folder = ROOT / 'configs/mog/component_scale'
    base = {**DEFAULTS, 'seed': 1, 'prior_kind': 'mog', 'mog_metrics': True,
            'mog_pass_criteria': frozen_criteria()['thresholds'],
            'standardize': True, 'particle_lr_multiplier': 10., 'particle_beta1': .5,
            'log_interval': 100, 'final_samples': 200000}
    if args.stage == 'longer':
        if not args.selected:
            parser.error('--selected is required for the longer follow-up')
        parent = ROOT / 'results/mog/component_scale' / args.selected
        if not (parent / 'run_grid_complete.json').exists():
            parser.error('Selected run must be certified complete')
        selected = json.loads((parent / 'summary.json').read_text())['config']
        assert selected['sigma_rel'] > 0 and selected['seed'] == 1
        c0 = json.loads((ROOT / 'results/mog/stage0/C0_s1/summary.json').read_text())['config']
        small = json.loads((ROOT / 'results/mog/refine_noise/n400_r1over40_14k_s1/summary.json').read_text())['config']
        configs = [(args.selected.replace('_7k_', '_28k_'), selected),
                   ('matched_atoms_' + args.selected.replace('_7k_', '_28k_'), {**selected, 'sigma_rel': 0.}),
                   ('C0_28k_s1', c0), ('n400_r1over40_28k_s1', small)]
        for name in args.extra:
            extra = ROOT / 'results/mog/component_scale' / name
            if not (extra / 'run_grid_complete.json').exists():
                parser.error('Extra run must be certified complete')
            cfg = json.loads((extra / 'summary.json').read_text())['config']
            assert cfg['seed'] == 1
            configs.append((name.replace('_7k_', '_28k_'), cfg))
        folder = ROOT / 'configs/mog/scale_longer'
        for name, cfg in configs:
            cfg = {**cfg, 'epochs': 28, 'mog_pass_criteria': frozen_criteria()['thresholds'],
                   'out_dir': f'results/mog/scale_longer/{name}'}
            # C0 itself may already be the selected arm's matched atoms control.
            if name.startswith('matched_atoms_') and all(
                    cfg[k] == c0[k] for k in c0 if k not in ('epochs', 'out_dir', 'mog_pass_criteria', 'particle_beta1')):
                if cfg['particle_beta1'] in (None, 0.):
                    continue
            save(cfg, name, folder)
        return
    if args.stage == 'optimizer':
        for n in (1600, 6400):
            for tag, radius in [('r0', 0.), ('r1over40', 1/40)]:
                name = f'n{n}_{tag}_shipped_std1_7k_s1'
                cfg = {**base, 'num_particles': n, 'sigma_rel': radius,
                       'particle_lr_multiplier': 1., 'particle_beta1': 0.,
                       'out_dir': f'results/mog/component_scale/{name}'}
                save(cfg, name, folder)
        return
    cells = []
    for n in (1600, 6400, 20000):
        for tag, radius in [('r0', 0.), ('r1over40', 1/40), ('r1over16', 1/16)]:
            cells.append((f'n{n}_{tag}_fast', dict(num_particles=n, sigma_rel=radius)))
    # Separate C0's optimizer and standardization from the count/noise screen.
    for tag, radius, standardized in [('r0', 0., True), ('r1over40', 1/40, True),
                                      ('r1over40', 1/40, False), ('r1over16', 1/16, False)]:
        cells.append((f'n20000_{tag}_shipped_std{int(standardized)}',
                      dict(num_particles=20000, sigma_rel=radius, standardize=standardized,
                           particle_lr_multiplier=1., particle_beta1=0.)))
    for name, changes in cells:
        name += '_7k_s1'
        cfg = {**base, **changes, 'out_dir': f'results/mog/component_scale/{name}'}
        save(cfg, name, folder)


if __name__ == '__main__':
    main()
