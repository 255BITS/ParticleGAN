#!/usr/bin/env python
"""Toggle component noise in frozen capacity-screen checkpoints; no training."""
import copy
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_denoising import make_prior
from particlegan import DDGAN
from particlegan.diffusion import DrawSource
from lib.denoising_toy import GaussianGrid, ToyGenerator, generate, grid_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/denoising/mog_capacity')
    parser.add_argument('--out', default='reports/denoising-toy/mog_capacity')
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    rows = []
    for name in ('gan_atoms', 'gan_mog', 'ddgan_atoms', 'ddgan_mog'):
        directory = ROOT / args.root / name
        checkpoint = torch.load(directory / 'final.pt', map_location=device, weights_only=False)
        cfg = checkpoint['config']
        g = ToyGenerator(cfg).to(device).eval()
        g.load_state_dict(checkpoint['G'])
        prior = make_prior(cfg, device)
        noise = DrawSource(cfg['noise'], cfg['noise_particles'], 2, cfg['seed'] + 102, device)
        noise.load_state_dict(checkpoint['noise'])
        schedule = DDGAN(cfg['alpha_bar'], validate_args=False).to(device)
        toy = GaussianGrid(device, cfg['std'], cfg['classes'])
        c = torch.arange(cfg['final_samples'], device=device) % cfg['classes']
        for sigma_rel in (0., .025):
            state = copy.deepcopy(checkpoint['prior'])
            state['sigma'] = state['d0'] * sigma_rel
            state['_extra_state']['sigma_rel'] = sigma_rel
            prior.load_state_dict(state)
            rngs = [torch.Generator(device=device).manual_seed(99000 + k) for k in range(4)]
            real = toy.sample(c, rngs[3])
            x = generate(g, prior, noise, schedule, c, *rngs[:3])
            metrics = grid_metrics(x, c, toy, real)
            if sigma_rel == cfg['sigma_rel']:
                original = json.loads((directory / 'summary.json').read_text())['final']
                assert abs(metrics['conditional_sw1'] - original['conditional_sw1']) < 1e-5
            rows.append(dict(name=name, training_sigma_rel=cfg['sigma_rel'],
                             sampling_sigma_rel=sigma_rel, metrics=metrics))
            print(name, sigma_rel, 'SW1', metrics['conditional_sw1'], 'core', metrics['per_mode_core_ratio'], flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'noise_probe.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# Frozen-checkpoint noise intervention', '',
             'No retraining. All network weights, raw component means, and calibrated initial spacing stay fixed. '
             'Only sampling sigma_rel changes. Baseline rows reproduce the final evaluation. '
             'Noise toggling changes RNG consumption, so individual draw paths are not paired.', '',
             '| Trained model | Sampling sigma_rel | Conditional SW1 ↓ | HQ % | Modes | Core width (ideal 1) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        lines.append(f"| {row['name']} | {row['sampling_sigma_rel']} | {m['conditional_sw1']:.4f} | "
                     f"{100*m['joint_hq']:.2f} | {m['modes']} | {m['per_mode_core_ratio']:.3f} |")
    (out / 'NOISE_PROBE.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
