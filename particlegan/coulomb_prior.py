"""Bounded deterministic repulsion with a checkpointed initial confining well."""
import math

import torch
from torch import nn


CENTER_BUFFER = "_coulomb_initial_center"
RADIUS_BUFFER = "_coulomb_initial_radius"


def capture_coulomb_reference(prior):
    """Capture once, after public initialization; consume no random numbers.

    These O(d) persistent prior buffers travel with live/EMA/checkpoint states.
    Rebinding an already initialized policy never moves the reference well.
    """
    if hasattr(prior, CENTER_BUFFER) != hasattr(prior, RADIUS_BUFFER):
        raise ValueError("incomplete Coulomb initial reference")
    if hasattr(prior, CENTER_BUFFER):
        return
    rows = prior.z.detach()
    if rows.ndim != 2 or not len(rows) or not torch.isfinite(rows).all():
        raise ValueError("Coulomb reference requires finite initialized location rows")
    center = rows.mean(0)
    radius = (rows - center).square().sum(1).mean().sqrt()
    prior.register_buffer(CENTER_BUFFER, center.clone())
    prior.register_buffer(RADIUS_BUFFER, radius.clone())


def coulomb_reference(priors):
    """Return the initial centroid/radius of one table or a declared union."""
    priors = tuple(priors) if isinstance(priors, (list, tuple)) else (priors,)
    if not priors or any(not hasattr(p, CENTER_BUFFER) for p in priors):
        raise ValueError("bind Coulomb prior policy after public initialization first")
    if len(priors) == 1:
        return getattr(priors[0], CENTER_BUFFER), getattr(priors[0], RADIUS_BUFFER)
    total = sum(len(p.z) for p in priors)
    center = sum(getattr(p, CENTER_BUFFER) * (len(p.z) / total) for p in priors)
    radius2 = sum((getattr(p, RADIUS_BUFFER).square()
                   + (getattr(p, CENTER_BUFFER) - center).square().sum())
                  * (len(p.z) / total) for p in priors)
    return center, radius2.sqrt()


class CoulombPriorRegularizer(nn.Module):
    """Dimensionless softened inverse-distance energy plus harmonic confinement.

    q_i = (z_i - initial_centroid) / max(initial_RMS_radius, 1e-6)
    E = mean_{i,k} (||q_i - q_{i+offset_k}||² + softening²)^(-1/2)
        + confinement / 2 * mean_i ||q_i||².

    Select at most ``neighbors`` deterministic cyclic offsets, evenly spaced
    over 1..n-1. Self pairs are excluded; every row participates. The arithmetic
    is O(n*d*neighbors) and uses O(n*d) scratch space. It draws no RNG and keeps
    caller-selected row/optimizer ownership. This estimates a bounded pair
    graph, not a nearest-neighbor or complete Coulomb interaction.

    The repulsive gradient is -delta/(distance²+softening²)^(3/2), with the
    averaging and 1/initial_radius factors. Exact collisions have finite zero
    repulsive force; no arbitrary direction or random jitter is injected.
    """
    def __init__(self, *, center=None, radius=None, weight=1., softening=.1,
                 confinement=.35, neighbors=8):
        super().__init__()
        for name, value, strict in (("weight", weight, False), ("softening", softening, True),
                                    ("confinement", confinement, True)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (value <= 0 if strict else value < 0)):
                raise ValueError(f"{name} must be finite and {'positive' if strict else 'nonnegative'}")
        if type(neighbors) is not int or not 1 <= neighbors <= 64:
            raise ValueError("neighbors must be an integer in [1, 64]")
        if (center is None) != (radius is None):
            raise ValueError("Coulomb reference requires center and radius together")
        if center is not None and (center.ndim != 1 or radius.ndim != 0
                or not torch.isfinite(center).all() or not torch.isfinite(radius)
                or radius < 0):
            raise ValueError("invalid Coulomb reference geometry")
        self.register_buffer("center", center)
        self.register_buffer("radius", radius)
        self.weight = float(weight)
        self.softening, self.confinement = float(softening), float(confinement)
        self.neighbors = neighbors

    def forward(self, rows):
        if self.weight == 0:
            return rows[:0].sum()
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if self.center is None:
            raise ValueError("Coulomb regularizer requires fixed initial reference geometry")
        if (rows.shape[1] != len(self.center) or rows.device != self.center.device
                or rows.dtype != self.center.dtype):
            raise ValueError("Coulomb rows must match reference dimension, device and dtype")
        if not len(rows):
            return rows.sum()
        q = (rows - self.center) / self.radius.clamp_min(1e-6)
        confinement = .5 * self.confinement * q.square().sum(1).mean()
        repulsion = rows.new_zeros(())
        n = len(rows)
        if n > 1:
            k = min(self.neighbors, n - 1)
            for j in range(k):
                offset = 1 + (j * (n - 1)) // k
                delta = q - q.roll(offset, 0)
                repulsion = repulsion + (delta.square().sum(1) + self.softening**2).rsqrt().mean()
            repulsion = repulsion / k
        return self.weight * (repulsion + confinement)
