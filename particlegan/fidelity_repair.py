"""Tail-sensitive empirical output transport, without task/evaluator geometry.

This opt-in G-phase objective compares caller-owned batches. It consumes no
randomness or persistent state and never accesses labels, target components,
prior kernels, held-out samples or a second network forward.
"""
import math

import torch


def sliced_quartic_transport(fake, real):
    """Mean sorted projected fourth-power residual / real coordinate variance².

    One dimension gives empirical W4 to the fourth power, normalized by the
    squared real variance. Two dimensions use 32 evenly spaced unoriented
    directions. Higher dimensions use up to 32 evenly spaced coordinate axes
    and 32 fixed unit trigonometric directions. This surrogate is neither full
    multivariate transport nor rotation invariant. Quartic costs emphasize
    remote contamination; they do not guarantee any component's moments.

    Real targets and normalization are detached. Exactly constant target
    batches use unit scale (in the task's output units), avoiding the enormous
    precision-floor amplification seen in earlier transport controls. A
    nonconstant real batch retains its own positive variance without a fitted
    coefficient or task-dependent width. All rows participate with equal mass.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device or fake[0].numel() == 0):
        raise ValueError("quartic transport needs matching floating batches of at least two samples")
    x, y = fake.flatten(1), real.detach().flatten(1)
    dimension = x.shape[1]
    if dimension == 1:
        frame = x.new_ones(1, 1)
    elif dimension == 2:
        angles = torch.arange(32, dtype=x.dtype, device=x.device) * (math.pi / 32)
        frame = torch.stack((angles.cos(), angles.sin()))
    else:
        coordinates = torch.arange(1, dimension + 1, dtype=x.dtype, device=x.device)[:, None]
        directions = torch.arange(1, 33, dtype=x.dtype, device=x.device)[None, :]
        oblique = (torch.sin(coordinates * directions * math.sqrt(2))
                   + torch.cos(coordinates * directions * math.sqrt(3)))
        oblique = oblique / oblique.norm(dim=0, keepdim=True)
        axis_count = min(dimension, 32)
        axes = x.new_zeros(dimension, axis_count)
        rows = torch.arange(axis_count, device=x.device) * (dimension - 1) // (axis_count - 1)
        axes[rows, torch.arange(axis_count, device=x.device)] = 1
        frame = torch.cat((axes, oblique), dim=1)
    scale = (y - y.mean(0)).square().mean()
    scale = torch.where(scale > 0, scale, scale.new_ones(()))
    # Normalize before the fourth power to avoid an avoidable scale² underflow.
    residual = ((x @ frame).sort(dim=0).values - (y @ frame).sort(dim=0).values) / scale.sqrt()
    return residual.square().square().mean()
