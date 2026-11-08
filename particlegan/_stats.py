"""Shared private numerical primitives.

Each helper is the single implementation of a computation that several
modules performed with identical operations. Every helper keeps the exact
operation order, dtype and device of the call sites it replaced, so the
trajectories of the recipes that use it are unchanged bit for bit. Call
sites whose numerics differ (another distance kernel, accumulation dtype or
formula) keep their own code and say why next to it.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import torch

# Added to the diagonal of a squared-distance matrix so a row is never its own
# nearest neighbour.
SELF_SQ_DISTANCE_FILL = 1e9


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """Squared Euclidean distances ``cdist(a, b) ** 2`` (``b`` defaults to ``a``).

    Uses torch's default ``cdist`` compute mode (matrix-multiply kernel for
    larger inputs), then squares. ``kinetic_transport_local_loss`` keeps its
    own exact (no-matmul) ``cdist`` and infinite-diagonal ``kthvalue`` width,
    because they give different numerics (and its bytes are source-pinned by
    Forge task cards).
    """
    return torch.cdist(a, a if b is None else b, p=2.0).pow(2)


def knn_excluding_self(
    distances: torch.Tensor, k: int, fill: float = SELF_SQ_DISTANCE_FILL
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(values, indices)`` of the ``k`` smallest entries of each row, ignoring the diagonal.

    ``distances`` is a square matrix of distances or squared distances. The
    diagonal is excluded by adding ``fill`` to it, so ``fill`` must exceed every
    off-diagonal entry.
    """
    eye = torch.eye(len(distances), device=distances.device, dtype=distances.dtype) * fill
    return torch.topk(distances + eye, k=k, dim=1, largest=False)


def median_offdiag(distances: torch.Tensor) -> torch.Tensor:
    """Lower median (``torch.median``) of the off-diagonal entries of a square matrix."""
    off = ~torch.eye(len(distances), dtype=torch.bool, device=distances.device)
    return torch.median(distances[off])


def knn_sq_and_median_bandwidth(z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``k`` nearest squared distances of each row of ``z`` and the median squared bandwidth.

    The bandwidth is the median off-diagonal squared distance, computed without
    gradient and floored at 1e-12. The kNN distances keep their gradient.
    """
    sq_dist = pairwise_sq_dist(z)
    with torch.no_grad():
        r_med_sq = median_offdiag(sq_dist).clamp_min(1e-12)
    knn_sq_dist, _ = knn_excluding_self(sq_dist, k)
    return knn_sq_dist, r_med_sq


def bh_rejections(sorted_p: torch.Tensor, q: float) -> int:
    """Benjamini-Hochberg step-up: the number of rejected hypotheses at level ``q``.

    ``sorted_p`` holds all ``m`` p-values in ascending order. The result ``j``
    is the largest rank with ``p_(j) <= j q / m``, or 0 when no rank passes.
    The threshold is evaluated as ``(ranks * q) / m`` in the p-value dtype.
    """
    m = len(sorted_p)
    ranks = torch.arange(1, m + 1, device=sorted_p.device, dtype=sorted_p.dtype)
    passed = (sorted_p <= ranks * q / m).nonzero()
    return int(passed[-1]) + 1 if len(passed) else 0


def group_means(values: torch.Tensor, groups: torch.Tensor, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float64 per-group means of ``values`` rows and the group row counts.

    Empty groups have mean 0 and count 0.
    """
    counts = torch.bincount(groups, minlength=count)
    sums = torch.zeros(count, values.shape[1], dtype=torch.float64, device=values.device)
    sums.index_add_(0, groups, values.double())
    return sums / counts.clamp_min(1)[:, None], counts


def adam_moment_states(optimizer, group) -> Iterator[Tuple[torch.Tensor, dict]]:
    """``(parameter, state)`` for parameters of ``group`` with a gradient and Adam second moments."""
    for parameter in group["params"]:
        state = optimizer.state.get(parameter)
        if parameter.grad is None or not state or "exp_avg_sq" not in state:
            continue
        yield parameter, state


def adam_surprise_ratio(grad: torch.Tensor, second: torch.Tensor, beta2: float, step) -> torch.Tensor:
    """RMS gradient over the RMS bias-corrected Adam second moment, as a 0-d tensor.

    ``sqrt(mean(g^2)) / sqrt(mean(v) / (1 - beta2 ** step))``, with the
    corrected second moment floored at 1e-30. ``second`` is ``exp_avg_sq`` (or
    ``max_exp_avg_sq``) and ``step`` the Adam step count (number or tensor).
    """
    vhat = second.mean() / (1.0 - beta2 ** step)
    return grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
