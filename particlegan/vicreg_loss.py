import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLikeLoss(nn.Module):
    """
    VICReg-inspired regularization for latent particles.

    Encourages:
    1. Variance: Standard deviation of each dimension >= target_std (hinge loss).
    2. Covariance: Decorrelation between dimensions.

    This allows the distribution to have arbitrary topology (holes, clusters)
    unlike Epps-Pulley which forces a Gaussian shape on the batch.
    """
    def __init__(self, target_std=1.0, eps=1e-4):
        super().__init__()
        self.target_std = target_std
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] == 0:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if len(z) < 2:
            return z.sum() * 0.0

        # 1. Variance Loss: Force std to be close to 1.0 (hinge-style)
        # Penalize only if std < target_std to prevent collapse,
        # but allow expansion (holes).
        std_z = torch.sqrt(z.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.target_std - std_z))

        # 2. Covariance Loss: Decorrelate dimensions
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)

        # Off-diagonal elements should be 0
        d = z.shape[1]
        if d > 1:
            off_diag = cov.flatten()[:-1].view(d-1, d+1)[:, 1:].flatten()
            cov_loss = off_diag.pow(2).sum() / d
        else:
            cov_loss = z.new_zeros(())

        return std_loss + cov_loss


class ParticleRegularizer(VICRegLikeLoss):
    """Weighted VICReg regularization on the particle rows supplied by the caller."""

    def __init__(self, target_std=1.0, eps=1e-4, weight=1.0):
        import math
        super().__init__(target_std=target_std, eps=eps)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        if not math.isfinite(target_std) or target_std < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("target_std must be nonnegative and eps positive, both finite")
        self.weight = float(weight)

    def forward(self, z):
        return self.weight * super().forward(z)
