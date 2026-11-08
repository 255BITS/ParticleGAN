#!/usr/bin/env python
"""Explain BISPF domain dispatch with a fixed linear critic; no training needed.

Run from any directory: python /path/to/ParticleGAN/examples/bispf_input_domains.py
See docs/bispf-input-domains.md for the categorical training example.
"""
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlegan import get_recipe


def probe(width, domain):
    """Measure the public penalty on D(x) = 0.2 * sum(x) + 0.3.

    This hand-set critic is an arithmetic fixture, not a trained model.
    Every input coordinate has derivative 0.2, so the raw gradient length
    is 0.2 * sqrt(width); the RMS length is 0.2.
    """
    critic = nn.Linear(width, 1).double()
    with torch.no_grad():
        critic.weight.fill_(0.2)
        critic.bias.fill_(0.3)
    recipe = get_recipe("bispf_optimized", batch_size=4)
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(
        optimizer, input_domain=domain, collect_stats=True)
    # Both batches really are categorical rows, even for a small vocabulary.
    # Using the same rows for every domain isolates the declaration's effect.
    real = F.one_hot(torch.arange(4) % width, width).double()
    fake = real.roll(1, dims=1)
    value = penalty(critic, real, fake)
    gradient = torch.autograd.grad(value, critic.weight)[0]
    stats = penalty.last_stats
    categorical = domain == "categorical" or domain == "auto" and width >= 170
    expected_norm = math.sqrt(0.04 * width + 1e-12)
    if not categorical and width > 4:
        expected_norm /= math.sqrt(width)
    if not math.isclose(stats["mean_grad_norm"], expected_norm, rel_tol=1e-9):
        raise RuntimeError("Gradient norm does not match the analytic fixture")
    if not torch.isfinite(value) or not torch.isfinite(gradient).all():
        raise RuntimeError("Nonfinite penalty or backward gradient")
    if categorical and (stats["floor_pen"] != 0 or stats["gauge_loss"] != 0):
        raise RuntimeError("Categorical branch unexpectedly applied floor/gauge")
    return {
        "width": width, "input_domain": domain,
        "penalty_branch": "categorical" if categorical else "continuous",
        "mean_grad_norm": stats["mean_grad_norm"],
        "floor_pen": stats["floor_pen"], "gauge_loss": stats["gauge_loss"],
        "penalty": float(value.detach()), "finite_backward": True,
    }


def main():
    torch.set_num_threads(1)
    print("Fixed critic: D(x) = 0.2 * sum(x) + 0.3. No model is trained.")
    print("Columns: width, declaration, selected branch, gradient norm, floor, gauge")
    results = []
    for width in (4, 170, 1024):
        for domain in ("auto", "continuous", "categorical"):
            result = probe(width, domain)
            results.append(result)
            print(f"{width:4d}  {domain:11s}  {result['penalty_branch']:11s}  "
                  f"{result['mean_grad_norm']:.6f}  {result['floor_pen']:.6f}  "
                  f"{result['gauge_loss']:.6f}")
    print("PASS: all nine analytic gradient and finite-backward checks.")
    print(json.dumps({"checks": results, "passed": True}))


if __name__ == "__main__":
    main()
