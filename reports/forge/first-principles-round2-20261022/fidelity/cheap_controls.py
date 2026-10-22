#!/usr/bin/env python3
"""Deterministic analytic admission and bounded CPU cost measurement; no training."""
import json
import math
from pathlib import Path
import time

import torch

from particlegan.bregman_game import bregman_cost_matrix, critic_game_loss, generator_game_loss


def run():
    torch.set_num_threads(1)
    initial_rng = torch.get_rng_state().clone()
    saturated = []
    for value, target in ((-40., 1.), (40., 0.)):
        a = torch.full((2, 1), value, dtype=torch.float64, requires_grad=True)
        y = torch.full_like(a, target)
        loss = generator_game_loss(bregman_cost_matrix(a.sigmoid(), y, kind="sigmoid", prelink=a), torch.zeros(2))
        gradient, = torch.autograd.grad(loss, a)
        saturated.append({"prelink": value, "target": target, "value": float(loss.detach()),
                          "minimum_absolute_gradient": float(gradient.abs().min())})
        assert torch.isfinite(loss) and gradient.abs().min() > .49
    x = torch.tensor([[-math.sqrt(2)], [0.], [0.], [math.sqrt(2)]], dtype=torch.float64)
    y = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    wrong_shape = generator_game_loss(bregman_cost_matrix(x, y), torch.zeros(4))
    assert wrong_shape > .29
    x = torch.tensor([[-3.], [-3.], [-1.], [1.]], dtype=torch.float64)
    y = torch.tensor([[-3.], [-1.], [1.], [3.]], dtype=torch.float64)
    price = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    cost = bregman_cost_matrix(x, y)
    price_gradient, = torch.autograd.grad(critic_game_loss(cost, price), price)
    torch.testing.assert_close(price_gradient, torch.tensor([.25, 0., 0., -.25], dtype=torch.float64))
    assert generator_game_loss(cost, price) == 0
    timings = []
    # Each measurement includes one detached D cost/loss and one G cost/backward.
    # Sizes are the actual native pair shape and selected image batch shape.
    for kind, n, d in (("identity", 2048, 2), ("sigmoid", 32, 64)):
        values = torch.arange(n*d, dtype=torch.float32).reshape(n, d)
        target = torch.sin(values * .017)
        if kind == "sigmoid":
            target = (target + 1) / 2
        elapsed = []
        for _ in range(3):
            a = torch.cos(values * .019).requires_grad_()
            fake = a.sigmoid() if kind == "sigmoid" else a
            arguments = {"kind": kind, "prelink": a if kind == "sigmoid" else None}
            price = torch.linspace(-.1, .1, n, requires_grad=True)
            started = time.process_time()
            c = bregman_cost_matrix(fake, target, **arguments)
            critic_game_loss(c, price).backward()
            c = bregman_cost_matrix(fake, target, **arguments)
            generator_game_loss(c, price).backward()
            elapsed.append(time.process_time() - started)
        timings.append({"kind": kind, "batch": n, "coordinates": d,
                        "pair_matrix_bytes_float32": n*n*4,
                        "cpu_seconds_D_plus_G_cost_and_backward": elapsed,
                        "native_7000_step_cost_only_cpu_projection_seconds": max(elapsed)*7000 if n == 2048 else None})
    assert torch.equal(initial_rng, torch.get_rng_state())
    return {"schema": 1, "candidate": "bispf-bregman-game-v1", "seed_study": False,
            "scope": "analytic and CPU operation controls; no optimizer updates, sampling draws or qualification",
            "saturated_prelink": saturated, "same_moments_wrong_shape_zero_price_value": float(wrong_shape),
            "missing_mode_zero_price_generator_value": 0., "missing_mode_critic_loss_price_gradient": price_gradient.tolist(),
            "constant_law": "exact-zero cost/force and gauge control covered by unit tests",
            "global_torch_rng_unchanged": True, "cost_measurements": timings,
            "runtime_limit": "O(n*m*d) device-local cost plus row min; no CPU assignment, iterations or extra forwards"}


if __name__ == "__main__":
    destination = Path(__file__).with_name("cheap-controls.json")
    result = run()
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
