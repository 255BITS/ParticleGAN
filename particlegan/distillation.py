"""Label-free finite partition-of-unity mass, location and shape witness.

Fit normalization only on the real batch, then freeze it for differentiation.
No RNG or prior/sampling mutation. Finite moments are not characteristic.
"""
import torch


def distillation_loss(fake, real, *, cells=8, return_parts=False):
    """Distill same-batch soft-cell occupancy and conditional moments.

    Farthest-first anchors start at real[0]. Real variance floors (1/16 pooled
    variance / n**2) and 1/n mass floors bound normalization. Zero-variance laws use
    unit pooled scale. All real-derived quantities are detached.
    """
    if type(cells) is not int or cells < 1:
        raise ValueError("cells must be a positive integer")
    if (fake.ndim != 2 or real.ndim != 2 or fake.shape != real.shape
            or not len(real) or not fake.shape[1] or not fake.is_floating_point()
            or fake.dtype != real.dtype or fake.device != real.device):
        raise ValueError("distillation requires matching nonempty floating vector batches")
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError("distillation requires finite batches")
    n, d = real.shape
    k = min(cells, n)
    with torch.no_grad():
        real = real.detach()
        pooled = (real - real.mean(0)).square().sum(1).mean()
        pooled = torch.where(pooled > torch.finfo(real.dtype).tiny, pooled, pooled.new_ones(()))
        selected = [0]
        distance = (real - real[0]).square().sum(1)
        for _ in range(1, k):
            index = int(distance.argmax())
            selected.append(index)
            distance = torch.minimum(distance, (real - real[index]).square().sum(1))
        anchors = real[selected]
        temperature = pooled / (4 * k)
        r = torch.softmax(-(real[:, None] - anchors).square().sum(2) / temperature, dim=1)
        mass = r.mean(0)
        denominator = mass.clamp_min(1 / n)
        counts = (n * mass.clamp_min(torch.finfo(real.dtype).tiny))
        mean = torch.einsum("nk,nd->kd", r, real) / counts[:, None]
        centered = real[:, None] - mean
        variance = (r * centered.square().sum(2)).sum(0) / counts
        scale2 = variance.clamp_min(pooled / (16 * n * n))
        u = centered / scale2.sqrt()[None, :, None]
        covariance = torch.einsum("nk,nki,nkj->kij", r, u, u) / counts[:, None, None]
        # Measured corrections make identical empirical batches exactly zero,
        # including float32 summation error in real centering.
        real_first = torch.einsum("nk,nkd->kd", r, u) / n
        real_second = (torch.einsum("nk,nki,nkj->kij", r, u, u) / n
                       - mass[:, None, None] * covariance)
    s = torch.softmax(-(fake[:, None] - anchors).square().sum(2) / temperature, dim=1)
    v = (fake[:, None] - mean) / scale2.sqrt()[None, :, None]
    fake_mass = s.mean(0)
    first = torch.einsum("nk,nkd->kd", s, v) / n - real_first
    second = (torch.einsum("nk,nki,nkj->kij", s, v, v) / n
              - fake_mass[:, None, None] * covariance - real_second)
    parts = {
        "mass": ((fake_mass - mass).square() / denominator).sum(),
        "location": (first.square().sum(1) / (denominator * d)).sum(),
        "shape": (second.square().sum((1, 2)) / (denominator * d * d)).sum(),
    }
    loss = sum(parts.values())
    return (loss, parts) if return_parts else loss
