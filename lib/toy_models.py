"""
Toy models and data for the 2D Gaussian-grid benchmark.

The example and experiment trainer import the same generator, discriminator,
dataset sampler and coverage metric from this module.
"""

from typing import Tuple

import torch
import torch.nn as nn

from lib.particle_prior import ParticlePrior


# =========================
#  Simple MLP G / D
# =========================

class SimpleMLPGenerator(nn.Module):
    """
    Very small MLP generator: z -> x in R^2.

    Strong enough for the toy problem but still minimal and CPU-friendly.
    """

    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleMLPDiscriminator(nn.Module):
    """
    Simple MLP discriminator: x in R^2 -> scalar score.

    fourier=K appends sin/cos features at frequencies pi * 2^i (i < K) per
    input dimension. MLPs are spectrally biased toward low frequencies, so
    without this D cannot resolve the sigma=0.03 mode structure until very
    late in training and sample sharpness stalls.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        fourier: int = 2,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        dim = in_dim + (2 * fourier * in_dim if fourier > 0 else 0)
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs  # (B, in_dim, K)
            h = torch.cat([h, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        # Return shape (B,) for convenience.
        return self.net(h).squeeze(-1)


# =========================
#  100 Gaussians dataset
# =========================

def sample_100gaussians(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator = None,
    grid_scale: float = 1.0,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Sample from a 100-Gaussian mixture:
      - Centers on a 10x10 grid at coordinates:
            { -4.5, -3.5, ..., 4.5 } * grid_scale
      - Isotropic Gaussian noise with `std`.

    This is intentionally dense and low-variance to stress-test mode coverage.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if generator is None:
        idx_x = torch.randint(0, 10, (batch_size,), device=device)
        idx_y = torch.randint(0, 10, (batch_size,), device=device)
    else:
        idx_x = torch.randint(0, 10, (batch_size,), device=device, generator=generator)
        idx_y = torch.randint(0, 10, (batch_size,), device=device, generator=generator)

    # Map indices 0..9 to coordinates -4.5..4.5
    centers_x = (idx_x - 4.5) * grid_scale
    centers_y = (idx_y - 4.5) * grid_scale

    centers = torch.stack(
        (centers_x, centers_y),
        dim=1,
    ).to(device=device, dtype=torch.float32)

    if generator is None:
        noise = torch.randn(batch_size, 2, device=device) * std
    else:
        noise = torch.randn(batch_size, 2, device=device, generator=generator) * std

    return centers + noise


# =========================
#  Metrics
# =========================

@torch.no_grad()
def mode_coverage(
    generator: nn.Module,
    prior: ParticlePrior,
    device: torch.device,
    n_eval: int = 20000,
    std: float = 0.03,
    min_count: int = 10,
    sample_generator: torch.Generator = None,
) -> Tuple[int, float]:
    """
    Coverage metrics on samples from the selected prior:
      - modes: number of grid centers with >= min_count "high quality" samples
        (within 3 sigma of the center),
      - hq_frac: fraction of samples that are high quality.
    """
    was_training = generator.training
    generator.eval()
    z, _ = prior.sample(n_eval, generator=sample_generator)
    fake = generator(z)
    coords = torch.arange(10, device=device, dtype=torch.float32) - 4.5
    cx, cy = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([cx.flatten(), cy.flatten()], dim=1)
    dists = torch.cdist(fake, centers)
    mind, nearest = dists.min(dim=1)
    hq = mind <= 3 * std
    counts = torch.bincount(nearest[hq], minlength=100)
    generator.train(was_training)
    return int((counts >= min_count).sum().item()), hq.float().mean().item()
