"""Deterministic sliced quadratic transport signal on ordinary training batches.

This is a finite-direction empirical objective, not an evaluator or a sampler.
It moves equal sample masses by projected quantile matching. No labels, target
centers, new random draws, persistent state or altered sampling weights enter.
"""
import math

import torch


def balanced_assignment_loss(fake, real, *, block_size=128):
    """Conservative joint-coordinate quadratic transport in contiguous blocks.

    Every row has equal mass and is matched exactly once to a real row in its
    block. The last block retains all its rows. Assignment is solved on detached
    float64 CPU coordinates; gradients flow through the selected squared costs.
    This is exact empirical W2 within each block, divided by dimension and the
    detached whole-real-batch coordinate variance. It is a minibatch surrogate,
    not full-batch OT when multiple blocks are used, or a population divergence.
    No labels, RNG, changed prior weights or persistent state enter.

    Requires SciPy (available in ParticleGAN's ``mog`` extra). At assignment ties
    the solver selects a minimizing branch; gradients need not be unique.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device):
        raise ValueError("balanced assignment needs matching floating batches of at least two samples")
    if type(block_size) is not int or block_size < 2:
        raise ValueError("balanced assignment block_size must be an integer >= 2")
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    x, y = fake.flatten(1), real.detach().flatten(1)
    detached_x = x.detach().to(device="cpu", dtype=torch.float64).numpy()
    detached_y = y.to(device="cpu", dtype=torch.float64).numpy()
    matches = []
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        _, columns = linear_sum_assignment(cdist(detached_x[start:stop], detached_y[start:stop], "sqeuclidean"))
        matches.extend((columns + start).tolist())
    indices = torch.tensor(matches, device=x.device, dtype=torch.long)
    scale = (y - y.mean(0)).square().mean().clamp_min(torch.finfo(x.dtype).eps)
    return (x - y[indices]).square().mean() / scale


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


def kinetic_transport_local_loss(fake, real):
    """Relative local kernel moments at detached real-batch anchors.

    Each anchor uses its fourth-other-neighbor radius (or n-1 for n<5), at
    fixed multipliers 1, 2 and 4. The squared fake/real feature-mean residual
    is divided by the squared real feature mean. This finite, data-dependent
    feature MMD is not an unbiased population divergence or fitted density
    ratio estimator. All anchors, widths and normalizers are detached.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device):
        raise ValueError("local transport needs matching floating batches of at least two samples")
    x, y = fake.flatten(1), real.detach().flatten(1)
    distance_real = torch.cdist(y, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    neighbor_distance = distance_real.clone()
    neighbor_distance.fill_diagonal_(float("inf"))
    scale = (y - y.mean(0)).square().mean().clamp_min(torch.finfo(x.dtype).eps)
    width_squared = neighbor_distance.kthvalue(min(4, len(y)-1), dim=0).values
    width_squared = width_squared.clamp_min(torch.finfo(x.dtype).eps * scale)
    distance_fake = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
    residuals = []
    for multiplier in (1., 2., 4.):
        denominator = 2 * multiplier**2 * width_squared[None, :]
        p = torch.exp(-distance_real / denominator).mean(0)
        q = torch.exp(-distance_fake / denominator).mean(0)
        # p >= 1/n because every anchor contributes its own unit kernel value.
        residuals.append(((q-p)/p).square().mean())
    return torch.stack(residuals).mean()
