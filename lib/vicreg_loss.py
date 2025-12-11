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
        # z: (Batch, Dim)
        
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
            cov_loss = torch.tensor(0.0, device=z.device)

        return std_loss + cov_loss
