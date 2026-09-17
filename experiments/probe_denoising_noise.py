#!/usr/bin/env python
"""Evaluate noise-support versus covariance substitutions in a frozen DDGAN.

These are inference interventions, not independently trained ablations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particlegan import DDGAN
from particlegan.diffusion import DrawSource
from experiments.train_denoising import make_prior
from lib.denoising_toy import GaussianGrid, ToyGenerator, generate, grid_metrics


def matrix_power(cov, exponent):
    values, vectors = torch.linalg.eigh(cov)
    return (vectors * values.clamp_min(1e-8).pow(exponent)) @ vectors.T


class AffineSource:
    def __init__(self, source, center, transform, mean):
        self.source, self.center, self.transform, self.mean = source, center, transform, mean

    def sample(self, n, rng):
        x, ids = self.source.sample(n, rng)
        return (x - self.center) @ self.transform + self.mean, ids


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=99000, help="Evaluation RNG seed, independent of training")
    args = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    assert cfg["model"] == "ddgan" and cfg["noise"] == "learned"
    assert args.samples % cfg["classes"] == 0
    toy = GaussianGrid(device, cfg["std"], cfg["classes"])
    schedule = DDGAN(cfg["alpha_bar"], validate_args=False).to(device)
    g = ToyGenerator(cfg).to(device)
    prior = make_prior(cfg, device)
    noise = DrawSource("learned", cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    for module, key in ((g, "G"), (prior, "prior"), (noise, "noise")):
        module.load_state_dict(ckpt[key])
        module.eval()
    gaussian = DrawSource("gaussian", cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    fixed = DrawSource("fixed", cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    zero = DrawSource("zero", cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    mean = noise.table.mean(0)
    # Population covariance: particles are sampled with replacement.
    cov = torch.cov(noise.table.T, correction=0)
    root, invroot = matrix_power(cov, .5), matrix_power(cov, -.5)
    fixed_cov = torch.cov(fixed.table.T, correction=0)
    variants = {
        "learned_table": noise,
        "matched_gaussian": AffineSource(gaussian, 0, root, mean),
        "matched_fixed_table": AffineSource(fixed, fixed.table.mean(0),
                                            matrix_power(fixed_cov, -.5) @ root, mean),
        "standard_gaussian": gaussian,
        "standardized_learned_table": AffineSource(noise, mean, invroot, 0),
        "zero_noise": zero,
    }
    c = torch.arange(args.samples, device=device) % cfg["classes"]
    real = toy.sample(c, torch.Generator(device=device).manual_seed(args.seed + 3))
    results = {}
    for name, source in variants.items():
        rngs = [torch.Generator(device=device).manual_seed(args.seed + k) for k in range(3)]
        x = generate(g, prior, source, schedule, c, *rngs)
        results[name] = grid_metrics(x, c, toy, real)
        m = results[name]
        print(f"{name}: HQ={m['joint_hq']:.4f}, TV={m['conditional_mode_tv']:.4f}, "
              f"SW1={m['conditional_sw1']:.4f}", flush=True)
    report = {"checkpoint": args.checkpoint,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": cfg, "samples": args.samples, "evaluation_seed": args.seed,
              "noise_mean": mean.cpu().tolist(), "noise_population_covariance": cov.cpu().tolist(),
              "metrics": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
