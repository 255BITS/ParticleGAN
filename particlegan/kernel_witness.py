"""Stateless, real-scaled full pairwise Cauchy-kernel mean discrepancy.

This is a raw-output MMD, not a learned critic or a finite moment frame. The
three bandwidths and target are detached; only generated samples receive force.
For fixed positive bandwidths the Cauchy kernel is a positive mixture of
Gaussian kernels with full spectral support, hence characteristic on R^d.
Minibatch bandwidth adaptation and neural optimization confer no convergence
guarantee. The biased empirical mean embedding includes every self diagonal.
"""
import math

import torch


def squared_distances(left, right):
    return (left[:, None, :] - right[None, :, :]).square().sum(-1)


def kernel_witness_loss(fake, real, *, return_parts=False):
    """Full pairwise MMD² at local, intermediate and global real-data scales.

    Local squared scale is the median sqrt(n)-th other-neighbor distance.
    Global squared scale is total real variance. Clamp the local scale to
    [S/(16*n²), S]; an exactly atomic target uses S=1 explicitly. No draws,
    labels, stored targets or checkpoint state are introduced.
    """
    if (fake.ndim != 2 or real.ndim != 2 or fake.shape[1] != real.shape[1]
            or min(len(fake), len(real)) < 2 or fake.shape[1] < 1):
        raise ValueError("kernel witness requires matching vector batches with at least two rows")
    if (not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device):
        raise ValueError("kernel witness batches must share a floating dtype and device")
    real = real.detach()
    with torch.no_grad():
        rr = squared_distances(real, real)
        global_scale = (real - real.mean(0)).square().sum(-1).mean()
        global_scale = torch.where(global_scale > 0, global_scale, global_scale.new_ones(()))
        other = rr.clone()
        other.fill_diagonal_(float("inf"))
        rank = min(len(real) - 1, max(1, math.isqrt(len(real))))
        local_scale = other.kthvalue(rank, dim=1).values.median()
        local_scale = local_scale.clamp(min=global_scale / (16 * len(real)**2), max=global_scale)
        scales = torch.stack((local_scale, (local_scale * global_scale).sqrt(), global_scale))
    qq = squared_distances(fake, fake)
    pq = squared_distances(real, fake)
    repulsion = torch.stack([(s / (s + qq)).mean() for s in scales]).mean()
    attraction = -2 * torch.stack([(s / (s + pq)).mean() for s in scales]).mean()
    target = torch.stack([(s / (s + rr)).mean() for s in scales]).mean()
    loss = repulsion + attraction + target
    parts = dict(repulsion=repulsion, attraction=attraction, target=target,
                 local_scale=local_scale, global_scale=global_scale)
    return (loss, parts) if return_parts else loss
