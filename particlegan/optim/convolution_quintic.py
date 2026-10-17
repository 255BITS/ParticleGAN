"""Bounded five-step convolution directions inspired by Muon's polynomial.

This is a different optimizer from exact or converged smoothed polar updates.
It keeps the absolute smoothing parameter through an augmented matrix, uses
float32 polynomial arithmetic, and retains no iteration state or random stream.
"""
import math

import torch


SEMANTIC_VERSION = "per_offset_quintic_hybrid_v1"
COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ITERATIONS = 5
OUTPUT_SCALE = 1.21


def quintic_direction(matrices, *, smoothing):
    """Return a bounded matrix filter, preserving leading batch dimensions.

    Orient M so its smaller channel dimension is last. Normalize [M; λI] by
    sqrt(||M||F² + λ²), a spectral upper bound rather than its Frobenius norm.
    Apply p(s)=3.4445s-4.775s³+2.0315s⁵ five times to its singular values,
    then return the top block divided by 1.21. The scalar interval
    [0, 1.2023686052] is invariant, giving a unit operator bound in exact
    arithmetic after this rescaling. Floating-point bounds need tolerance.

    Unlike native DualNorm, this filter has no hard numerical-rank cutoff,
    exact-SVD claim, or convergence stopping rule. The smoothing parameter is
    fixed in the input's absolute units. Max-absolute pre-scaling protects
    finite extreme inputs; float64 inputs or smoothing outside float32's range
    are scaled before conversion. All five polynomial steps use float32.
    """
    if not isinstance(matrices, torch.Tensor) or matrices.ndim < 2 or not matrices.is_floating_point():
        raise ValueError("quintic_direction requires floating-point matrices")
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("quintic smoothing must be finite and nonnegative")
    if not matrices.numel():
        return torch.zeros_like(matrices)
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("quintic directions require finite matrix entries")
    original_shape, original_dtype = matrices.shape, matrices.dtype
    # Disable the caller's AMP casting locally; the declared filter uses FP32,
    # including when the surrounding model runs under autocast.
    with torch.autocast(device_type=matrices.device.type, enabled=False):
        scaling_dtype = (torch.float64 if original_dtype == torch.float64
                         or smoothing > torch.finfo(torch.float32).max else torch.float32)
        value = matrices.reshape(-1, *original_shape[-2:]).to(dtype=scaling_dtype)
        transposed = value.shape[-2] < value.shape[-1]
        if transposed:
            value = value.mT
        minimum = max(smoothing, torch.finfo(scaling_dtype).tiny)
        scale = value.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(minimum)
        scaled = (value / scale).float()
        regularizer = (smoothing / scale).float()
        normalizer = (scaled.square().sum(dim=(-2, -1), keepdim=True)
                      + regularizer.square()).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
        identity = torch.eye(value.shape[-1], device=value.device, dtype=torch.float32)
        result, lower = scaled / normalizer, regularizer * identity / normalizer
        a, b, c = COEFFICIENTS
        for _ in range(ITERATIONS):
            gram = result.mT @ result + lower.mT @ lower
            correction = a * identity + b * gram + c * (gram @ gram)
            result, lower = result @ correction, lower @ correction
        result = result / OUTPUT_SCALE
        if transposed:
            result = result.mT
        return result.reshape(original_shape).to(dtype=original_dtype)
