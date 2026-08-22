"""
Distributional metrics for the 2D Gaussian-grid benchmark.

`mode_coverage` (lib/toy_models.py) answers "did the generator find the modes",
which is the coarse question. These functions answer the finer ones a controlled
comparison of gradient penalties needs:

  * how far is the model distribution from the data distribution
    (`sliced_w1`, cheap and GPU-resident; `exact_w1_w2`, exact but O(n^2 log n)),
  * how well-calibrated is the density it puts on each mode (`mixture_nll`),
  * how balanced is its mass across the 100 modes
    (`mode_recall_and_hists`).

Everything takes samples as (N, 2) tensors on any device. The two torch-only
metrics stay on the input device; the OT one moves to numpy float64.
"""

from typing import Dict, Optional, Tuple

import math

import numpy as np
import torch


# =========================
#  Grid helpers
# =========================

def _grid_centers(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    grid_scale: float = 1.0,
) -> torch.Tensor:
    """The 100 mixture centers as a (100, 2) tensor, in row-major (x, y) order."""
    coords = (torch.arange(10, device=device, dtype=dtype) - 4.5) * grid_scale
    cx, cy = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([cx.flatten(), cy.flatten()], dim=1)


# =========================
#  Sliced Wasserstein-1
# =========================

def sliced_w1(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 128,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Sliced Wasserstein-1 distance between two point clouds.

    Projects both clouds onto `n_proj` random unit directions, computes the 1D
    W1 distance per direction as the mean absolute difference of the sorted
    projections, and averages over directions. Pure torch, so it runs on the
    GPU and never leaves the device until the final scalar.

    The two clouds must have the same number of points (the 1D W1 shortcut
    "sort both, take mean |difference|" is only valid for equal-size uniform
    empirical measures).

    Args:
        x, y: (N, 2) samples.
        n_proj: number of random projection directions.
        seed: if given, projections are drawn from a fresh generator with this
            seed, making the estimate reproducible. Ignored if `generator` is
            passed.
        generator: explicit torch.Generator (must live on `x`'s device).

    Returns:
        The sliced W1 estimate as a python float.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"sliced_w1 needs equal sample counts, got {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    device = x.device
    dim = x.shape[1]

    if generator is None and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    with torch.no_grad():
        if generator is None:
            proj = torch.randn(dim, n_proj, device=device, dtype=x.dtype)
        else:
            proj = torch.randn(
                dim, n_proj, device=device, dtype=x.dtype, generator=generator
            )
        proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-12)

        px, _ = (x @ proj).sort(dim=0)  # (N, n_proj)
        py, _ = (y.to(x.dtype) @ proj).sort(dim=0)
        return float((px - py).abs().mean())


# =========================
#  Exact optimal transport
# =========================

def exact_w1_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int = 4096,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """
    Exact Wasserstein-1 and Wasserstein-2 via the POT network-simplex solver.

    Both clouds are (uniformly, without replacement) subsampled to at most
    `max_points` points, given uniform weights, and matched exactly:

        w1 = emd2(a, b, C)          with C_ij = ||x_i - y_j||
        w2 = sqrt(emd2(a, b, C^2))  with C_ij = ||x_i - y_j||^2

    O(n^2) memory in the cost matrix and roughly O(n^3 log n) time, hence the
    subsample cap. Everything is converted to numpy float64 before the solve,
    because the simplex is sensitive to float32 cost ties.

    Returns:
        (w1, w2) as python floats.
    """
    import ot  # imported lazily: POT is only needed for the final eval

    rng = np.random.default_rng(seed)

    def _prep(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().to("cpu", torch.float64).numpy()
        if arr.shape[0] > max_points:
            sel = rng.choice(arr.shape[0], size=max_points, replace=False)
            arr = arr[sel]
        return np.ascontiguousarray(arr)

    xa = _prep(x)
    ya = _prep(y)

    a = np.full(xa.shape[0], 1.0 / xa.shape[0], dtype=np.float64)
    b = np.full(ya.shape[0], 1.0 / ya.shape[0], dtype=np.float64)

    m2 = ot.dist(xa, ya, metric="sqeuclidean")  # squared Euclidean cost
    m1 = np.sqrt(m2)

    w1 = float(ot.emd2(a, b, m1, numItermax=1_000_000))
    w2_sq = float(ot.emd2(a, b, m2, numItermax=1_000_000))
    return w1, math.sqrt(max(w2_sq, 0.0))


# =========================
#  Analytic likelihood
# =========================

def mixture_nll(
    x: torch.Tensor,
    std: float = 0.03,
    grid_scale: float = 1.0,
) -> float:
    """
    Mean negative log-likelihood of `x` under the analytic 100-Gaussian mixture.

    The density is an equal-weight mixture of isotropic 2D Gaussians:

        p(x) = sum_k (1/100) * N(x; mu_k, std^2 I)

    so, with d = 2,

        log p(x) = logsumexp_k [ -||x - mu_k||^2 / (2 std^2) ] - log 100
                   - (d/2) log(2 pi std^2)

    Lower is better; the entropy floor for perfect samples is roughly
    d/2 * (1 + log(2 pi std^2)) + log 100 nats.

    Returns:
        The mean NLL in nats, as a python float.
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)

        d = xf.shape[1]
        var = float(std) ** 2
        sq = torch.cdist(xf, centers).pow(2)  # (N, 100)

        log_comp = -sq / (2.0 * var)
        log_p = (
            torch.logsumexp(log_comp, dim=1)
            - math.log(centers.shape[0])
            - 0.5 * d * math.log(2.0 * math.pi * var)
        )
        return float(-log_p.mean())


# =========================
#  Mode balance
# =========================

def mode_recall_and_hists(
    x: torch.Tensor,
    n_expected_frac: float = 0.5,
    std: float = 0.03,
    grid_scale: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Mode recall and mass-balance divergences from a nearest-center assignment.

    Every sample is assigned to its nearest of the 100 grid centers (no
    distance threshold: this measures *balance*, not sharpness, which
    `mode_coverage` and `mixture_nll` already cover). From the resulting
    100-cell histogram:

      - mode_recall: fraction of the 100 modes whose count is at least
        `n_expected_frac * N / 100`, i.e. that hold at least that share of the
        mass a uniform model would give them. Recall 1.0 means no mode is
        starved.
      - hist_kl: KL(empirical || uniform) in nats. 0 for perfect balance,
        log(100) ~ 4.605 for total collapse onto one mode.
      - hist_js: Jensen-Shannon divergence between the same two histograms, in
        nats. Bounded by log 2, so it stays informative when KL saturates.

    `eps` smooths both histograms before the logs.

    Returns:
        {'mode_recall': float, 'hist_kl': float, 'hist_js': float}
    """
    with torch.no_grad():
        xf = x.detach().to(torch.float64)
        centers = _grid_centers(xf.device, torch.float64, grid_scale)
        k = centers.shape[0]

        nearest = torch.cdist(xf, centers).argmin(dim=1)
        counts = torch.bincount(nearest, minlength=k).to(torch.float64)

        n = float(xf.shape[0])
        threshold = n_expected_frac * n / k
        mode_recall = float((counts >= threshold).to(torch.float64).mean())

        p = counts / counts.sum().clamp_min(eps) + eps
        p = p / p.sum()
        q = torch.full_like(p, 1.0 / k)

        hist_kl = float((p * (p / q).log()).sum())

        m = 0.5 * (p + q)
        hist_js = float(
            0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum()
        )

    return {
        "mode_recall": mode_recall,
        "hist_kl": hist_kl,
        "hist_js": hist_js,
    }
