"""Sliced isotropic Gaussian characteristic-function regularization.

Adapted mathematically from LeJEPA's MINIMAL.md at c293d291ca87cd4fddee9d3fffe4e914c7272052.
Projection draws use a private checkpointed CPU stream, independent of model,
data, prior sampling and evaluation streams on both CPU and CUDA.
"""
import math

import torch
from torch import nn


class SIGReg(nn.Module):
    """Weighted SIGReg on the caller's actual selected location rows.

    The unweighted statistic is N times the mean projected, windowed squared
    characteristic-function error against N(0, 1). Rows are neither centered
    nor standardized: mean, scale and higher moments all affect the objective.
    Retain this module and save its state_dict for stochastic continuation.
    """

    def __init__(self, *, weight=1., projections=32, knots=17):
        super().__init__()
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        if type(projections) is not int or projections < 1:
            raise ValueError("projections must be a positive integer")
        if type(knots) is not int or knots < 2:
            raise ValueError("knots must be an integer >= 2")
        self.weight, self.projections, self.knots = float(weight), projections, knots
        # A named, private stream with a fixed seed is not a task/seed search axis.
        self.register_buffer("projection_rng_state", torch.Generator(device="cpu").manual_seed(0).get_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        value = state_dict.get(prefix + "projection_rng_state")
        if value is not None:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("invalid SIGReg projection RNG state")
            try:
                torch.Generator(device="cpu").set_state(value.cpu())
            except RuntimeError as error:
                raise ValueError("invalid SIGReg projection RNG state") from error
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, rows):
        if rows.ndim != 2 or rows.shape[1] == 0 or not rows.is_floating_point():
            raise ValueError("particles must have floating shape [batch, positive latent dimension]")
        if self.weight == 0 or len(rows) == 0:
            return rows[:0].sum()
        stream = torch.Generator(device="cpu").set_state(self.projection_rng_state.cpu())
        # Sampling on CPU keeps this additional stream portable across devices.
        directions = torch.randn(rows.shape[1], self.projections, generator=stream, dtype=torch.float32, device="cpu")
        directions = directions / directions.norm(dim=0).clamp_min(torch.finfo(directions.dtype).tiny)
        self.projection_rng_state.copy_(stream.get_state().to(self.projection_rng_state.device))
        return self.weight * sigreg_statistic(rows, directions.to(rows), knots=self.knots)


def sigreg_statistic(rows, directions, *, knots=17):
    """Upstream quadrature with supplied unit projection columns; no RNG draws."""
    if rows.ndim != 2 or directions.ndim != 2 or directions.shape[0] != rows.shape[1]:
        raise ValueError("rows and projection dimensions must agree")
    if len(rows) == 0 or directions.shape[1] == 0:
        return rows[:0].sum()
    if type(knots) is not int or knots < 2:
        raise ValueError("knots must be an integer >= 2")
    # Evaluate trigonometry in float32 under mixed precision, float64 otherwise.
    dtype = torch.float64 if rows.dtype == torch.float64 else torch.float32
    t = torch.linspace(0, 3, knots, device=rows.device, dtype=dtype)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=rows.device, dtype=dtype)
    weights[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2)
    projected = rows.to(dtype) @ directions.to(device=rows.device, dtype=dtype)
    x_t = projected.unsqueeze(-1) * t
    error = (x_t.cos().mean(0) - phi).square() + x_t.sin().mean(0).square()
    return len(rows) * (error @ (weights * phi)).mean()
