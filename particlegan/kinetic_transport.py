"""Deterministic sliced quadratic transport signal on ordinary training batches.

This is a finite-direction empirical objective, not an evaluator or a sampler.
It moves equal sample masses by projected quantile matching. No labels, target
centers, new random draws, persistent state or altered sampling weights enter.
"""
import math

import torch


def kinetic_transport_loss(fake, real, *, projections=32):
    """Mean projected W2 squared, divided by detached real coordinate variance.

    In two dimensions use evenly spaced angles in [0, pi); one dimension is
    exact empirical W2. Higher dimensions use a fixed trigonometric frame.
    These finite directions are a surrogate, with no rotation-invariance or
    full Wasserstein convergence claim. Batches must have equal cardinality.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.device != real.device):
        raise ValueError("kinetic transport needs matching floating batches of at least two samples")
    if type(projections) is not int or projections < 1:
        raise ValueError("kinetic transport projections must be a positive integer")
    x, y = fake.flatten(1), real.detach().flatten(1)
    dimension = x.shape[1]
    if dimension == 1:
        frame = x.new_ones(1, 1)
    elif dimension == 2:
        angles = torch.arange(projections, dtype=x.dtype, device=x.device) * (math.pi / projections)
        frame = torch.stack((angles.cos(), angles.sin()))
    else:
        coordinates = torch.arange(1, dimension + 1, dtype=x.dtype, device=x.device)[:, None]
        directions = torch.arange(1, projections + 1, dtype=x.dtype, device=x.device)[None, :]
        frame = torch.sin(coordinates * directions * math.sqrt(2)) + torch.cos(coordinates * directions * math.sqrt(3))
        frame = frame / frame.norm(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    scale = (y - y.mean(0)).square().mean().clamp_min(torch.finfo(x.dtype).eps)
    projected_x = (x @ frame).sort(dim=0).values
    projected_y = (y @ frame).sort(dim=0).values
    return (projected_x - projected_y).square().mean() / scale
