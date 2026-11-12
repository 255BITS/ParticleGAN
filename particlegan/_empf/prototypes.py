"""Online e-means sufficient-statistics prototype bank for EMPF.

Adapts the online sufficient-statistics EMA and proportional-split dead-unit
revival of e-means (Bessa et al., arXiv:2611.09501) to continuous adversarial
transport without dataset-size or epoch-horizon clocks.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from .._stats import pairwise_sq_dist

# Golden-ratio and R2 plastic-constant low-discrepancy increments.
_PHI1 = 0.6180339887498949
_PHI2 = 0.7548776662466927


def emeans_effective_alpha(
    batch_size: int,
    num_prototypes: int,
    *,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.35,
) -> float:
    """Clock-free (K, B) hit-rate EMA rate inspired by e-means scaling.

    In e-means, optimal EMA update rates grow sublinearly with the expected
    per-batch cluster hit count ``B / K``. Dropping dataset-size and horizon
    clocks yields ``alpha = clip(alpha_base * (B / K) ** hit_exponent,
    alpha_min, alpha_max)``.
    """
    if batch_size <= 0 or num_prototypes <= 0:
        raise ValueError("batch_size and num_prototypes must be positive")
    ratio = float(batch_size) / float(num_prototypes)
    scaled = float(alpha_base) * (ratio ** float(hit_exponent))
    return float(min(alpha_max, max(alpha_min, scaled)))


def _stratified_initial_prototypes(x: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic stratified selection of ``k`` rows from ``x`` (zero RNG)."""
    b_r, dim = x.shape
    if k >= b_r:
        return x.clone()
    idx_dim = torch.arange(1, dim + 1, device=x.device, dtype=x.dtype)
    proj_dir = torch.remainder(idx_dim * _PHI1, 1.0) - 0.5
    proj_norm = proj_dir.norm().clamp_min(1e-8)
    scores = x @ (proj_dir / proj_norm)
    order = torch.argsort(scores, stable=True)
    slots = (torch.arange(k, device=x.device, dtype=torch.long) * b_r + (b_r // 2)) // k
    slots = slots.clamp(0, b_r - 1)
    return x[order[slots]].clone()


def _within_batch_knn_sq(x: torch.Tensor, k_local: int = 2) -> torch.Tensor:
    """Squared distance from each row of ``x`` to its ``k_local``-th neighbor in ``x``."""
    b_r = x.shape[0]
    if b_r <= 1:
        return torch.zeros(b_r, device=x.device, dtype=x.dtype)
    k_eff = max(1, min(int(k_local), b_r - 1))
    d2_xx = pairwise_sq_dist(x)
    eye = torch.eye(b_r, device=x.device, dtype=torch.bool)
    d2_xx = d2_xx.masked_fill(eye, float("inf"))
    return torch.topk(d2_xx, k=k_eff, dim=1, largest=False).values[:, -1]


@torch.no_grad()
def update_emeans_prototypes(
    x_real: torch.Tensor,
    proto_sum: torch.Tensor | None,
    proto_count: torch.Tensor | None,
    revival_phase: float,
    revivals: int,
    *,
    max_prototypes: int = 256,
    alpha_base: float = 0.08,
    hit_exponent: float = 0.45,
    alpha_min: float = 0.03,
    alpha_max: float = 0.35,
    revival_tau: float = 0.05,
    split_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Update online (S, N) e-means prototypes and return ``(C, w, S, N, phase, revivals)``."""
    x = x_real.detach().flatten(1).to(dtype=torch.float32)
    b_r, dim = x.shape
    if b_r == 0 or dim == 0:
        raise ValueError("x_real must have positive batch size and feature dimension")

    if (
        proto_sum is None
        or proto_count is None
        or proto_sum.ndim != 2
        or proto_count.ndim != 1
        or proto_sum.shape[1] != dim
        or proto_sum.shape[0] != proto_count.shape[0]
        or proto_sum.shape[0] == 0
    ):
        k = min(int(max_prototypes), b_r)
        c_init = _stratified_initial_prototypes(x, k)
        init_count = float(b_r) / float(k)
        new_count = torch.full((k,), init_count, device=x.device, dtype=torch.float32)
        new_sum = c_init * init_count
        weights = torch.full((k,), 1.0 / float(k), device=x.device, dtype=x_real.dtype)
        return c_init.to(dtype=x_real.dtype), weights, new_sum, new_count, float(revival_phase), int(revivals)

    cur_sum = proto_sum.to(device=x.device, dtype=torch.float32)
    cur_count = proto_count.to(device=x.device, dtype=torch.float32)
    k = cur_sum.shape[0]
    centroids_prev = cur_sum / cur_count.clamp_min(1e-8).unsqueeze(1)

    # Online deterministic farthest-point bank growth when batch_size < true mode count M
    # (small-batch multi-mode regime 32 <= b_r < 100 in 2D..4D): appends genuinely uncovered
    # populated modes up to max_prototypes without RNG while leaving b_r >= 100 banks fixed.
    nn_x_d2: torch.Tensor | None = None
    if 32 <= b_r < 100 and 2 <= dim <= 4 and 2 <= k < int(max_prototypes):
        d2_init = pairwise_sq_dist(x, centroids_prev)
        d2_min_grow = d2_init.min(dim=1).values
        nn_x_d2 = _within_batch_knn_sq(x, k_local=1)
        covered_d2 = d2_min_grow.quantile(0.25).clamp_min(1e-10)
        grow_thresh = torch.maximum(25.0 * nn_x_d2.clamp_min(1e-10), 25.0 * covered_d2)
        if bool((d2_min_grow > grow_thresh).any()):
            max_add = min(int(max_prototypes) - k, max(1, b_r // 4))
            mean_prev = cur_count.mean().clamp_min(1e-6)
            added_c = []
            for _ in range(max_add):
                eligible = d2_min_grow > grow_thresh
                if not bool(eligible.any()):
                    break
                masked_scores = torch.where(eligible, d2_min_grow, x.new_full((b_r,), -1.0))
                best_idx = int(masked_scores.argmax().item())
                new_pt = x[best_idx]
                added_c.append(new_pt)
                d2_min_grow = torch.minimum(d2_min_grow, (x - new_pt.unsqueeze(0)).square().sum(dim=1))
            if added_c:
                extra_c = torch.stack(added_c, dim=0)
                extra_n = torch.full((extra_c.shape[0],), float(mean_prev.item()), device=x.device, dtype=torch.float32)
                cur_sum = torch.cat([cur_sum, extra_c * extra_n.unsqueeze(1)], dim=0)
                cur_count = torch.cat([cur_count, extra_n], dim=0)
                k = cur_sum.shape[0]
                centroids_prev = cur_sum / cur_count.clamp_min(1e-8).unsqueeze(1)

    d2 = pairwise_sq_dist(x, centroids_prev)
    d2_min, assign = d2.min(dim=1)

    batch_count = torch.zeros(k, device=x.device, dtype=torch.float32)
    batch_count.scatter_add_(0, assign, torch.ones(b_r, device=x.device, dtype=torch.float32))
    batch_sum = torch.zeros((k, dim), device=x.device, dtype=torch.float32)
    batch_sum.index_add_(0, assign, x)

    alpha = emeans_effective_alpha(
        b_r,
        k,
        alpha_base=alpha_base,
        hit_exponent=hit_exponent,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    # Clock-free innovation & starvation shock boost: stays exact zero
    # (alpha_eff == alpha) on stationary distributions, while accelerating
    # prototype tracking and dead-unit turnover immediately when target support shifts.
    r2p_rms = float(torch.sqrt(d2_min.mean() / d2.median().clamp_min(1e-12)).item())
    shock_r2p = max(0.0, min(1.0, (r2p_rms - 0.30) / 0.35))
    shock_zero = 0.0
    if k >= 16 and b_r >= 16:
        cur_w = cur_count / cur_count.sum().clamp_min(1e-8)
        expected_zero = float(torch.exp(-float(b_r) * cur_w).mean().item())
        actual_zero = float((batch_count == 0).to(dtype=torch.float32).mean().item())
        shock_zero = max(0.0, min(1.0, (actual_zero - expected_zero - 0.22) / 0.25))
    shock = max(shock_r2p, shock_zero)
    alpha_eff = float(min(0.50, alpha + (0.50 - alpha) * (shock * shock)))

    new_sum = (1.0 - alpha_eff) * cur_sum + alpha_eff * batch_sum
    new_count = (1.0 - alpha_eff) * cur_count + alpha_eff * batch_count

    mean_count = new_count.mean()
    dead_mask = new_count < (float(revival_tau) * mean_count)
    phase = float(revival_phase)
    total_revivals = int(revivals)

    if bool(dead_mask.any()) and k >= 2:
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1).tolist()
        coord_std = x.std(dim=0, unbiased=False) + 1e-6
        dim_idx = torch.arange(1, dim + 1, device=x.device, dtype=torch.float32)

        for offset, d_idx in enumerate(dead_indices):
            donor_weights = (new_count - float(revival_tau) * mean_count).clamp_min(0.0)
            donor_weights[d_idx] = 0.0
            total_donor = float(donor_weights.sum().item())
            if total_donor <= 1e-12:
                donor_weights = new_count.clamp_min(1e-12).clone()
                donor_weights[d_idx] = 0.0
                total_donor = float(donor_weights.sum().item())
            if total_donor <= 1e-12:
                break
            donor_probs = donor_weights / total_donor
            donor_cdf = torch.cumsum(donor_probs, dim=0)
            q = (phase + (offset + 1) * _PHI2) % 1.0
            q_tensor = torch.tensor([q], device=x.device, dtype=torch.float32)
            p_idx = int(torch.searchsorted(donor_cdf, q_tensor).item())
            p_idx = max(0, min(k - 1, p_idx))
            if p_idx == d_idx:
                p_idx = (d_idx + 1) % k

            c_donor = new_sum[p_idx] / new_count[p_idx].clamp_min(1e-8)
            # Split donor + dead sufficient counts equally around donor centroid to conserve exact total mass.
            half_total = 0.5 * (new_count[p_idx] + new_count[d_idx])
            angles = dim_idx * ((phase + (offset + 1) * _PHI1) * (2.0 * math.pi))
            direction = torch.cos(angles)
            direction = direction / direction.norm().clamp_min(1e-8)
            delta = float(split_eps) * coord_std * direction
            new_count[p_idx] = half_total
            new_count[d_idx] = half_total
            new_sum[p_idx] = half_total * (c_donor - delta)
            new_sum[d_idx] = half_total * (c_donor + delta)

            total_revivals += 1

        phase = (phase + _PHI1) % 1.0

    centroids = (new_sum / new_count.clamp_min(1e-8).unsqueeze(1)).to(dtype=x_real.dtype)
    weights = (new_count / new_count.sum().clamp_min(1e-8)).to(dtype=x_real.dtype)
    return centroids, weights, new_sum, new_count, phase, total_revivals
