"""Explicit Frobenius convolution directions, distinct from polar DualNorm.

Each channel matrix follows M / hypot(||M||_F, smoothing). Its operator norm
is at most one, but its singular directions receive smaller steps than polar
normalization. The optimizer applies this rule only to large CUDA kernels.
"""
import math

import torch


def frobenius_direction(matrices, *, smoothing):
    """Normalize a batch of channel matrices without overflowing finite inputs.

    Half precision arithmetic is promoted to float32. Rescaling both the
    matrix and the absolute smoothing parameter preserves the intended law.
    Zero matrices produce zero even when smoothing is also zero.
    """
    if not isinstance(matrices, torch.Tensor) or matrices.ndim < 2 or not matrices.is_floating_point():
        raise ValueError("frobenius_direction requires floating-point matrices")
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Frobenius smoothing must be finite and nonnegative")
    if not matrices.numel():
        return torch.zeros_like(matrices)
    dtype = (torch.float64 if matrices.dtype == torch.float64
             or smoothing > torch.finfo(torch.float32).max else torch.float32)
    work = matrices.to(dtype)
    scale = work.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(smoothing)
    scale = scale.clamp_min(torch.finfo(work.dtype).tiny)
    normalized = work / scale
    denominator = torch.hypot(normalized.norm(dim=(-2, -1), keepdim=True),
                              smoothing / scale)
    result = normalized / denominator.clamp_min(torch.finfo(work.dtype).tiny)
    return result.to(matrices.dtype)
