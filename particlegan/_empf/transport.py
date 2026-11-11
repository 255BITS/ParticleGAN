"""Mass-weighted entropic optimal transport between e-means prototypes and generated samples."""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .._stats import pairwise_sq_dist


def mass_weighted_sinkhorn(
    prototypes: torch.Tensor,
    weights: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale-free log-domain Sinkhorn coupling with prototype row masses ``weights``.

    Row marginals equal the normalized e-means sufficient-count shares
    ``w_k = N_k / sum_j N_j``; column marginals are uniform ``1 / B_f``.
    Returns ``(Pi, norm_cost, residual, data_scale)`` with ``residual`` in
    dimensionless linear root-mean-square nearest-neighbor units.
    """
    c_flat = prototypes.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    k_proto, b_f = c_flat.shape[0], f_flat.shape[0]

    w = weights.detach().reshape(-1).to(device=c_flat.device, dtype=torch.float32).clamp_min(1e-12)
    w = w / w.sum().clamp_min(1e-12)

    raw_cost = pairwise_sq_dist(c_flat, f_flat)
    with torch.no_grad():
        scale = raw_cost.detach().median().clamp_min(eps_scale)
        data_scale = scale.sqrt()

    norm_cost = raw_cost / scale
    log_a = torch.log(w)
    log_b = -math.log(max(1, b_f))
    log_kernel = -norm_cost / float(sinkhorn_eps)

    u = torch.zeros(k_proto, device=c_flat.device, dtype=torch.float32)
    v = torch.zeros(b_f, device=c_flat.device, dtype=torch.float32)
    for _ in range(int(sinkhorn_iters)):
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

    pi = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
    pi = pi * (w.unsqueeze(1) / pi.sum(dim=1, keepdim=True).clamp_min(1e-12))

    with torch.no_grad():
        min_p2f = (w * norm_cost.detach().min(dim=1).values).sum()
        min_f2p = norm_cost.detach().min(dim=0).values.mean()
        residual = torch.sqrt((0.5 * (min_p2f + min_f2p)).clamp_min(0.0))

    return (
        pi.to(dtype=x_fake.dtype),
        norm_cost.to(dtype=x_fake.dtype),
        residual.to(dtype=x_fake.dtype),
        data_scale.to(dtype=x_fake.dtype),
    )


@torch.no_grad()
def conditional_fiber_residual(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    eps_scale: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale-free residual for row-aligned conditioned pairs ``(x_real[j], x_fake[j] | cond[j])``."""
    r_flat = x_real.detach().flatten(1).to(dtype=torch.float32)
    f_flat = x_fake.detach().flatten(1).to(dtype=torch.float32)
    raw_cost = pairwise_sq_dist(r_flat, f_flat)
    scale = raw_cost.median().clamp_min(eps_scale)
    data_scale = scale.sqrt()
    if r_flat.shape[0] == f_flat.shape[0]:
        fiber_sq = (r_flat - f_flat).square().sum(dim=1)
        mean_sq = (fiber_sq / scale).mean()
    else:
        mean_sq = 0.5 * (raw_cost.min(dim=1).values.mean() + raw_cost.min(dim=0).values.mean()) / scale
    residual = torch.sqrt(mean_sq.clamp_min(0.0))
    return residual.to(dtype=x_fake.dtype), data_scale.to(dtype=x_fake.dtype)
