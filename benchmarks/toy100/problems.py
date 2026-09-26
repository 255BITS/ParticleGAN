"""Public, unlabelled samplers for the frozen 100-Gaussian task family.

Training code should only call ``sample_real``. The component geometry is
exposed through ``evaluation_geometry`` for the independent evaluator and
plots; it must not be passed to the model, prior, loss, or optimizer.
"""

from __future__ import annotations

import math

import torch

import particlegan.sample_stream as sample_stream


PROBLEM_NAMES = ("grid100", "rotated100", "staggered100")
N_MODES = 100
DATA_STD = 0.03


def _centers(problem_name: str, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    if problem_name not in PROBLEM_NAMES:
        raise ValueError(f"unknown toy100 problem {problem_name!r}; choose from {PROBLEM_NAMES}")
    coords = torch.arange(10, device=device, dtype=dtype) - 4.5
    row, col = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack((row.flatten(), col.flatten()), dim=1)
    if problem_name == "rotated100":
        angle = math.radians(25.0)
        c, s = math.cos(angle), math.sin(angle)
        rotation = centers.new_tensor(((c, -s), (s, c)))
        centers = centers @ rotation.T
    elif problem_name == "staggered100":
        # Adjacent rows shift by half a cell, while row spacing contracts. The
        # nearest-center gap remains about one unit, over 30 target sigmas.
        centers[:, 1] += torch.where(
            (torch.arange(100, device=device) // 10) % 2 == 0,
            centers.new_tensor(-0.25), centers.new_tensor(0.25),
        )
        centers[:, 0] *= 0.85
    return centers


def evaluation_geometry(
    problem_name: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float]:
    """Return evaluator/plot-only centers and common isotropic standard deviation."""
    return _centers(problem_name, device=device, dtype=dtype), DATA_STD


def sample_real(
    problem_name: str,
    n: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw independent, unlabelled (n, 2) samples with equal mode weights."""
    if n <= 0:
        raise ValueError("sample count must be positive")
    centers = _centers(problem_name, device=device, dtype=torch.float32)
    if sample_stream.replacing():
        indices, noise = sample_stream.index_and_normal(
            "data", n, N_MODES, 2, device=centers.device, dtype=centers.dtype,
        )
    else:
        indices = torch.randint(N_MODES, (n,), device=device, generator=generator)
        noise = torch.randn(n, 2, device=device, generator=generator)
    return centers[indices] + DATA_STD * noise
