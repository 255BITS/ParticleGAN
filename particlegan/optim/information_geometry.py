"""Instantaneous fractional matrix preconditioning for a bounded BCAP study.

The rule is orthogonally equivariant, not a Fisher natural gradient. It uses
the same exact SVD and numerical rank policy as DualNorm. Per-offset kernel
matrices keep convolution work bounded by channel dimensions.
"""
import torch


def information_geometry_spectral_half(matrix, *, smoothing=0.):
    """Keep the largest smoothed-polar weight; attenuate weaker directions.

    For G=U diag(s) V^T, let h_i=hypot(s_i, smoothing). The update is
    U diag(s_i / sqrt(h_i h_max)) V^T. With negligible damping this is
    sqrt(s_i/s_max), rather than polar's unit weights. This equals a
    normalized, one-sided fourth-root Gram preconditioner. No RNG/history
    or task information is consumed, and zero singular directions stay zero.
    """
    import math
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("spectral smoothing must be finite and nonnegative")
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("spectral-half requires a floating-point matrix")
    value = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    if 0 in value.shape:
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(value, full_matrices=False)
    threshold = max(value.shape) * torch.finfo(value.dtype).eps * singular[0]
    scale = torch.hypot(singular, torch.full_like(singular, smoothing))
    # Clamp only the zero denominator. Rank policy masks its numerator.
    denominator = (scale * scale[0]).sqrt().clamp_min(torch.finfo(value.dtype).tiny)
    weights = singular / denominator * (singular > threshold)
    return ((left * weights) @ right).to(dtype=matrix.dtype)
