"""Scale-free Sinkhorn coupling shared by the BISPF engine, loss and critic penalty.

Callers look ``sinkhorn_coupling`` up on this module at call time, so one
patch of ``particlegan._bispf.transport.sinkhorn_coupling`` (profilers,
instrumentation) reaches every BISPF component.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .._stats import median_offdiag, pairwise_sq_dist


@torch.no_grad()
def sinkhorn_coupling(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scale-free doubly-stochastic Sinkhorn coupling Pi.

    Returns ``(Pi, med_sq_dist, residual, data_scale)``.
    """
    b_r = x_real.shape[0]
    b_f = x_fake.shape[0]
    x_r = x_real.detach().reshape(b_r, -1)
    x_f = x_fake.detach().reshape(b_f, -1)

    sq_dist = pairwise_sq_dist(x_r, x_f)
    med_sq_dist = torch.median(sq_dist).clamp_min(eps)
    C = sq_dist / med_sq_dist

    log_K = -C / sinkhorn_eps
    u = torch.zeros(b_r, device=x_r.device, dtype=x_r.dtype)
    v = torch.zeros(b_f, device=x_r.device, dtype=x_r.dtype)
    log_inv_br = -math.log(max(1, b_r))
    log_inv_bf = -math.log(max(1, b_f))

    for _ in range(sinkhorn_iters):
        u = log_inv_br - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_inv_bf - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

    log_Pi = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    Pi = torch.exp(log_Pi)
    Pi = Pi / (Pi.sum(dim=1, keepdim=True).clamp_min(eps) * b_r)

    # Scale-free symmetric nearest-neighbor residual with zero entropic blur floor
    min_r = sq_dist.min(dim=1).values.mean()
    min_f = sq_dist.min(dim=0).values.mean()
    delta_nn = torch.sqrt(0.5 * (min_r + min_f) / med_sq_dist)
    data_scale = med_sq_dist.sqrt()
    return Pi, med_sq_dist, delta_nn, data_scale


@torch.no_grad()
def _cloud_scale_factor(x_real: torch.Tensor, x_fake: torch.Tensor, data_scale: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return coordinate unit scale when both batches are non-degenerate clouds, else 1.0."""
    b_f = x_fake.shape[0]
    if b_f < 2:
        return data_scale.new_ones(())
    x_f = x_fake.detach().reshape(b_f, -1)
    f_med = median_offdiag(pairwise_sq_dist(x_f))  # b_f >= 2: the off-diagonal is nonempty
    if float(f_med.item()) <= 1e-6:
        # Point-mass probe (e.g. x_f = 0 in capture-radius tests): preserve raw Euclidean units.
        return data_scale.new_ones(())
    return data_scale.clamp_min(eps)
