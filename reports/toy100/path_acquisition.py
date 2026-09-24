"""Path-crossing acquisition direction on a 2D critic field.

Rays are fixed. A sample is redirected only when a ray dips and then reaches
a higher score in a region farther than ``EMPTY_RADIUS`` from every support
particle, and that sample belongs to the support particle nearest the hit.
A zero direction leaves the local gradient unchanged. Mode centers are not read.
"""
import math

import torch


RADII = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
N_ANGLES = 8
EMPTY_RADIUS = 1.0


@torch.no_grad()
def path_crossing_directions(score_fn, points, support, radii=RADII,
                             empty_radius=EMPTY_RADIUS):
    """Return unit directions ``(N, 2)``. Zeros mean do not redirect."""
    points = points.detach()
    support = support.detach()
    if points.ndim != 2 or points.shape[-1] != 2 or support.ndim != 2 or support.shape[-1] != 2:
        raise ValueError("path crossing expects 2D points and support")
    if support.shape[0] == 0:
        return torch.zeros_like(points)
    angles = torch.arange(N_ANGLES, dtype=points.dtype) * (2 * math.pi / N_ANGLES)
    rays = torch.stack((angles.cos(), angles.sin()), 1)
    radius = torch.tensor(radii, dtype=points.dtype)
    offsets = rays[:, None, :] * radius[None, :, None]
    probes = points[:, None, None, :] + offsets
    n, n_angles, n_radii, _ = probes.shape
    base = score_fn(points).reshape(-1)
    probed = score_fn(probes.reshape(-1, 2)).reshape(n, n_angles, n_radii)
    nearest = torch.cdist(probes.reshape(-1, 2), support).min(dim=1).values
    empty = nearest.reshape(n, n_angles, n_radii) > empty_radius
    owner = torch.cdist(points, support).argmin(dim=1)
    out = torch.zeros_like(points)
    for i in range(n):
        best = None
        best_score = base[i]
        for angle in range(n_angles):
            row = probed[i, angle]
            for k in range(n_radii):
                if not bool(empty[i, angle, k]) or row[k] <= best_score:
                    continue
                if k == 0 or row[:k].min() >= base[i]:
                    continue
                best_score = row[k]
                best = probes[i, angle, k]
        if best is None:
            continue
        if int(torch.cdist(best[None], support).argmin()) != int(owner[i]):
            continue
        delta = best - points[i]
        norm = delta.norm()
        if float(norm) <= 1e-8:
            continue
        out[i] = delta / norm
    return out


def redirect_loss_grad(grad, direction):
    """Point Adam's output step along ``direction`` without changing its norm.

    Adam moves opposite ``grad``. Replacement happens only when that motion
    points against ``direction`` and ``grad`` is nonzero, so a zero local
    gradient stays at rest.
    """
    if grad.shape != direction.shape:
        return grad
    out = grad.clone()
    norms = grad.norm(dim=-1)
    oppose = (grad * direction).sum(dim=-1) > 0
    take = oppose & (norms > 0) & (direction.norm(dim=-1) > 0)
    out[take] = -direction[take] * norms[take, None]
    return out
