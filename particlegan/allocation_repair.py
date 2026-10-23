"""A long-range output distribution witness on the existing training batches.

Energy distance uses attraction to real samples and repulsion between generated
samples. There are no target labels/centres, fitted kernels, random projections,
extra draws, persistent state, or changes to the public sampling law.
"""
from __future__ import annotations

import torch


def energy_distance(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Return a target-scale-normalized unbiased *numerator* energy estimate.

    The numerator is ``E|X-Y| - .5 E|X-X'| - .5 E|Y-Y'|`` with ordered
    off-diagonal self pairs (U statistics), so a finite estimate may be
    negative. Population energy distance is twice this numerator. Dividing by
    a batch-derived normalizer means the complete ratio is not an unbiased
    population estimator. No nonnegative clipping is applied to its gradient.

    Normalize by the detached root-mean-square distance of distinct target
    pairs, using one when that distance is exactly zero. This avoids an
    epsilon denominator amplifying constant-target forces. A nonconstant
    target's normalizer still follows its physical scale. Targets and the
    normalizer receive no gradient. Euclidean norms use their zero subgradient
    at coincident samples; no jitter or bandwidth is introduced.
    """
    if (fake.ndim < 2 or real.ndim != fake.ndim
            or fake.shape[1:] != real.shape[1:] or len(fake) < 2 or len(real) < 2
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device or fake[0].numel() == 0):
        raise ValueError("energy distance needs matching floating sample shapes and at least two rows per batch")
    x, y = fake.flatten(1), real.detach().flatten(1)
    n, m = len(x), len(y)
    # Direct differences preserve small separations and exact coincident-pair
    # zeros; matrix-multiplication cdist can invent distances via cancellation.
    cross = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").mean()
    xx = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist").sum() / (n * (n - 1))
    yy_distances = torch.cdist(y, y, compute_mode="donot_use_mm_for_euclid_dist")
    yy = yy_distances.sum() / (m * (m - 1))
    scale = (yy_distances.square().sum() / (m * (m - 1))).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (cross - .5 * xx - .5 * yy) / scale
