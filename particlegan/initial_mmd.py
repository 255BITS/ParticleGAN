"""Bounded distribution tether to initialized raw location tables; no RNG draws."""
from __future__ import annotations

import math

import torch
from torch import nn


def _panel(rows, limit=256):
    """Equally spaced midpoint indices, without replacement or random draws."""
    if len(rows) <= limit:
        return rows
    indices = ((2 * torch.arange(limit, device=rows.device) + 1) * len(rows)) // (2 * limit)
    return rows[indices]


def _kernel_mean(left, right, scale):
    # Three fixed dimensionless squared bandwidths. Each pair block is at most
    # 64 x 64; current/reference panels are at most 256 rows each.
    result = left[:0].sum()
    for a in left.split(64):
        for b in right.split(64):
            # Exact broadcast differences (not _stats.pairwise_sq_dist, whose
            # matmul cdist kernel rounds differently).
            distance = (a[:, None] - b[None, :]).square().sum(-1)
            for bandwidth in (.25, 1., 4.):
                width = scale * bandwidth
                result = result + (width / (width + distance)).sum()
    return result / (3 * len(left) * len(right))


class InitialMMDRegularizer(nn.Module):
    """Biased, nonnegative IMQ MMD against a fixed initial reference panel.

    Both empirical panels include self-pairs. This estimates distance to the
    *finite reference panel*, not an exact initial distribution. Panels use
    deterministic midpoint thinning and preserve all rows when N <= 256.
    Bandwidths use the initial full-table population covariance trace, floored
    at 1e-4 for degenerate explicit fixtures. Particle identity is not anchored.
    """

    def __init__(self, reference, *, weight=1., scale=None):
        super().__init__()
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        self.weight = float(weight)
        self._validate(reference)
        with torch.no_grad():
            if not torch.isfinite(reference).all():
                raise ValueError("initial MMD reference locations must be finite")
            reference_scale = ((reference - reference.mean(0)).square().mean(0).sum()
                               if scale is None else reference.new_tensor(scale))
            if reference_scale.ndim or not torch.isfinite(reference_scale) or reference_scale < 0:
                raise ValueError("initial MMD scale must be a finite nonnegative scalar")
            reference_scale = reference_scale.clamp_min(1e-4)
            panel = _panel(reference).detach().clone()
            self.register_buffer("reference", panel)
            self.register_buffer("scale", reference_scale.detach().clone())
            self.register_buffer("reference_kernel", _kernel_mean(panel, panel, self.scale))
        object.__setattr__(self, "_reference_owner", None)

    @staticmethod
    def _validate(rows):
        if (not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.shape[1] == 0
                or not rows.is_floating_point() or rows.is_complex() or len(rows) == 0):
            raise ValueError("initial MMD locations must be nonempty real floating [rows, dimensions]")

    @classmethod
    def from_priors(cls, priors, *, weight=1.):
        """Capture after initialization; store the original reference in prior state.

        A joint host concatenates its initialized tables, then takes one panel.
        The first table carries this shared reference in its checkpoint. Loading
        prior state restores it rather than recapturing the trained locations.
        """
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        priors = tuple(priors) if isinstance(priors, (tuple, list)) else (priors,)
        if not priors or any(not hasattr(prior, "z") for prior in priors):
            raise ValueError("initial_mmd requires initialized prior table(s)")
        owner = priors[0]
        names = ("_initial_mmd_reference", "_initial_mmd_scale", "_initial_mmd_reference_kernel")
        present = [hasattr(owner, name) for name in names]
        if any(present) and not all(present):
            raise ValueError("incomplete initial MMD reference state")
        if not all(present):
            rows = torch.cat([prior.z.detach() for prior in priors])
            module = cls(rows, weight=weight)
            for name, buffer in zip(names, (module.reference, module.scale, module.reference_kernel)):
                owner.register_buffer(name, buffer)
        else:
            # Do not recompute the reference kernel from current trained rows.
            module = cls.__new__(cls)
            nn.Module.__init__(module)
            module.weight = float(weight)
        object.__setattr__(module, "_reference_owner", owner)
        # Access owner buffers at forward time so prior.to(...) and checkpoint
        # restoration cannot leave stale reference tensors in the callable.
        for name in ("reference", "scale", "reference_kernel"):
            if name in module._buffers:
                del module._buffers[name]
        return module

    def forward(self, rows):
        if self.weight == 0:
            return rows[:0].sum()
        self._validate(rows)
        owner = self._reference_owner
        reference, scale, reference_kernel = ((self.reference, self.scale, self.reference_kernel)
            if owner is None else (owner._initial_mmd_reference, owner._initial_mmd_scale,
                                   owner._initial_mmd_reference_kernel))
        if (rows.shape[1] != reference.shape[1] or rows.device != reference.device
                or rows.dtype != reference.dtype):
            raise ValueError("current and initial MMD locations must share dimensions, device and dtype")
        panel = _panel(rows)
        distance = (_kernel_mean(panel, panel, scale) + reference_kernel
                    - 2 * _kernel_mean(panel, reference, scale))
        # The biased estimator is nonnegative analytically; clip cancellation
        # roundoff only, rather than use a signed unbiased minibatch estimator.
        return self.weight * distance.clamp_min(0)
