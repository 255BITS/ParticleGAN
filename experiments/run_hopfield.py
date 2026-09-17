#!/usr/bin/env python
"""Run the single-seed kill test, then the grid only if its gate passes.

Logs: results/hopfield/PIPELINE.log and results/hopfield/runs/*/log.txt.
Use an external tee to capture this process's unbuffered status output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def arm_config(root, dataset, read, particles, beta=16, learn_beta=False,
               seed=1234, device='cuda', steps=7000):
    label = 'uniform' if read == 'uniform' else ('hopfield_learn16' if learn_beta else f'hopfield_b{beta}')
    name = f'{dataset}_m{particles}_{label}_s{seed}'
    if steps != 7000:
        name += f'_{steps}steps'
    return {'dataset': dataset, 'read': read, 'beta': float(beta),
            'learn_beta': learn_beta, 'num_particles': particles, 'seed': seed,
            'device': device, 'epochs': 1, 'steps_per_epoch': steps,
            'out_dir': str(root / 'runs' / name)}


def generate(root, seed, device):
    grid = [arm_config(root, dataset, read, particles, beta, learn, seed, device)
            for dataset in ('uniform', 'imbalanced')
            for particles in (100, 1000, 20000)
            for read, beta, learn in [('uniform', 16, False), ('hopfield', 4, False),
                                      ('hopfield', 16, False), ('hopfield', 64, False),
                                      ('hopfield', 16, True)]]
    kill = [cfg for cfg in grid if cfg['dataset'] == 'imbalanced'
            and cfg['num_particles'] == 100 and cfg['beta'] == 16 and not cfg['learn_beta']]
    rest = [cfg for cfg in grid if cfg not in kill]
    config_dir = root / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for stage, configs in [('kill', kill), ('grid', rest)]:
        paths = []
        for cfg in configs:
            path = config_dir / (Path(cfg['out_dir']).name + '.yaml')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(str(path))
        manifest = config_dir / f'{stage}_manifest.json'
        manifest.write_text(json.dumps(paths, indent=2) + '\n')
        manifests[stage] = manifest
    return kill, manifests


def gate(kill):
    finals = {cfg['read']: json.loads((Path(cfg['out_dir']) / 'summary.json').read_text())['final']
              for cfg in kill}
    improvement = finals['uniform']['tv'] - finals['hopfield']['tv']
    return {'passed': improvement >= .20, 'tv_improvement': improvement,
            'required_improvement': .20, 'uniform_tv': finals['uniform']['tv'],
            'hopfield_tv': finals['hopfield']['tv'], 'seeds_per_arm': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results/hopfield')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--generate_only', action='store_true')
    parser.add_argument('--kill_only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    kill, manifests = generate(root, args.seed, args.device)
    print(f'One seed per arm ({args.seed}); kill test 2 runs, gated remainder 28 runs.', flush=True)
    if args.generate_only:
        return

    def run(manifest):
        subprocess.run([sys.executable, '-u', str(ROOT / 'experiments/run_grid.py'),
                        '--config_manifest', str(manifest), '--python', sys.executable,
                        '--trainer', str(ROOT / 'experiments/train_hopfield.py'),
                        '--gpus', args.gpus, '--workers_per_gpu', str(args.workers_per_gpu)],
                       cwd=ROOT, check=True)

    def analyze():
        subprocess.run([sys.executable, str(ROOT / 'experiments/analyze_hopfield.py'),
                        '--runs_dir', str(root / 'runs'), '--out_dir', str(root)], cwd=ROOT, check=True)

    run(manifests['kill'])
    decision = gate(kill)
    (root / 'kill_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print('KILL TEST: ' + json.dumps(decision), flush=True)
    analyze()
    if not decision['passed']:
        print('STOP: Hopfield failed the predeclared +0.20 TV improvement gate.', flush=True)
        return
    if args.kill_only:
        return
    run(manifests['grid'])
    # A longer baseline distinguishes slow headcount migration from failure to
    # converge within 7k. This is a new duration arm, not a seed experiment.
    baseline = arm_config(root, 'imbalanced', 'uniform', 20000, seed=args.seed,
                          device=args.device)
    baseline_final = json.loads((Path(baseline['out_dir']) / 'summary.json').read_text())['final']
    if baseline_final['steps_to_tv'] is not None:
        analyze()
        return
    long_cfg = arm_config(root, 'imbalanced', 'uniform', 20000, seed=args.seed,
                          device=args.device, steps=21000)
    path = root / 'configs' / 'uniform_m20000_long.yaml'
    path.write_text(yaml.safe_dump(long_cfg, sort_keys=False))
    manifest = root / 'configs' / 'long_manifest.json'
    manifest.write_text(json.dumps([str(path)]) + '\n')
    run(manifest)
    analyze()


if __name__ == '__main__':
    main()
