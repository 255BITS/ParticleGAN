"""Optional benchmark-local noise wrappers for 100-Gaussian models.

These wrappers do not use target centers or component assignments. The public
``GANTrainer`` and ``Recipe`` remain responsible for the ordinary GAN updates.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnableOutputScale(nn.Module):
    """One positive output-noise standard deviation learned with the generator."""

    def __init__(self, initial_std: float):
        super().__init__()
        if (isinstance(initial_std, bool) or not math.isfinite(initial_std)
                or initial_std <= 0):
            raise ValueError("learnable output noise requires a positive finite initial std")
        self.initial_std = float(initial_std)
        self.raw_scale = nn.Parameter(torch.tensor(
            initial_std + math.log(-math.expm1(-initial_std)),
        ))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_scale)


class OutputNoise(nn.Module):
    """Add fresh isotropic output noise in both training and sampling."""

    def __init__(self, model: nn.Module, std: float, learnable: bool = False):
        super().__init__()
        if not math.isfinite(std) or std < 0:
            raise ValueError("output noise std must be finite and nonnegative")
        if type(learnable) is not bool:
            raise ValueError("learnable must be a boolean")
        self.model = model
        self.std = float(std)
        self.initial_std = float(std)
        self.output_scale = LearnableOutputScale(std) if learnable else None
        if self.output_scale is not None:
            reference = next(model.parameters(), None)
            if reference is None:
                reference = next(model.buffers(), None)
            if reference is not None:
                dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
                self.output_scale.to(device=reference.device, dtype=dtype)

    def effective_std(self) -> float | torch.Tensor:
        """Apply the common warmup amplitude to the learned base scale."""
        if self.output_scale is None:
            return self.std
        return self.output_scale() * (self.std / self.initial_std)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        prediction = self.model(latent)
        if self.std == 0:
            return prediction
        # GANTrainer.sample forks the global RNG, so evaluation leaves this
        # training noise stream untouched and fixed-seed frames replay exactly.
        return prediction + self.effective_std() * torch.randn_like(prediction)


class InputNoise(nn.Module):
    """Add noise to each discriminator input from its own training stream."""

    def __init__(self, model: nn.Module, *, seed: int, device: torch.device):
        super().__init__()
        self.model = model
        self.sigma = 0.0
        self.noise_stream = torch.Generator(device=device).manual_seed(seed)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.sigma:
            noise = torch.randn(
                points.shape, generator=self.noise_stream,
                device=points.device, dtype=points.dtype,
            )
            points = points + self.sigma * noise
        return self.model(points)


def linear_input_noise(
    peak: float, completed_steps: int, total_steps: int, end_fraction: float,
) -> float:
    """Peak at the first update, linearly reaching zero by ``end_fraction``."""
    if not math.isfinite(peak) or peak < 0:
        raise ValueError("input noise std must be finite and nonnegative")
    if (type(completed_steps) is not int or completed_steps < 0
            or type(total_steps) is not int or total_steps <= 0):
        raise ValueError("invalid input noise step count")
    if not math.isfinite(end_fraction) or not 0 < end_fraction <= 1:
        raise ValueError("input noise anneal end must be in (0, 1]")
    return float(peak * max(0.0, 1.0 - completed_steps / (total_steps * end_fraction)))


def linear_output_noise(
    peak: float, completed_steps: int, total_steps: int, warmup_fraction: float = 0.0,
) -> float:
    """Rise from zero to ``peak`` over a fraction of completed updates.

    A zero warmup retains the original constant-noise recipe. The first
    update of a positive warmup runs with zero output noise.
    """
    if isinstance(peak, bool) or not math.isfinite(peak) or peak < 0:
        raise ValueError("output noise std must be finite and nonnegative")
    if (type(completed_steps) is not int or completed_steps < 0
            or type(total_steps) is not int or total_steps <= 0):
        raise ValueError("invalid output noise step count")
    if (isinstance(warmup_fraction, bool) or not math.isfinite(warmup_fraction)
            or not 0 <= warmup_fraction <= 1):
        raise ValueError("output noise warmup must be a finite fraction in [0, 1]")
    if warmup_fraction == 0:
        return float(peak)
    return float(peak * min(1.0, completed_steps / (warmup_fraction * total_steps)))
