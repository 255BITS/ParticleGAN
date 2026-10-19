"""Opt-in FP64 Gram polar map with a guard on the factors actually used.

This is a numerical profile, not a bitwise replacement for the default SVD.
The fixed-margin error model is not outward-rounded interval arithmetic.
"""
import math

import torch


GRAM_DENSE_PROFILE = "gram_fp64_guarded_v1"


class GramGuardError(RuntimeError):
    """The direction is unqualified; discard/restore the complete update state.

    Earlier optimizer leaves, clocks, moments or protected state may already
    have changed. No transactional rollback or automatic solver fallback occurs.
    """


def _guard_factors(z, gram, eigen, vectors, cutoff_constant):
    """Screen the SAME device/factors before their direction can be applied.

    Weyl's sufficient mask separation condition is evaluated with the frozen
    8x floating-point error estimate. Small negative roundoff tails are allowed;
    significant negatives and ambiguous retained/dropped modes are rejected.
    """
    rows, columns = z.shape
    unit = torch.finfo(torch.float64).eps / 2
    gram_rounding = (rows * unit / (1 - rows * unit) + 2 * unit + unit * unit)
    gram_rounding *= (z.abs().mT @ z.abs()).norm()
    identity = torch.eye(columns, device=vectors.device, dtype=torch.float64)
    orthogonality = (vectors.mT @ vectors - identity).norm()
    residual = (gram - (vectors * eigen) @ vectors.mT).norm()
    eta = float(orthogonality)
    if not math.isfinite(eta) or eta >= 1:
        raise GramGuardError("Gram eigenvectors fail the orthogonality error model")
    distance = eta / (1 + math.sqrt(1 - eta))
    correction = eigen.abs().amax() * distance * (math.sqrt(1 + eta) + 1)
    diagnostic_roundoff = columns * unit / (1 - columns * unit) * gram.norm()
    error = 8 * (gram_rounding + residual + correction + diagnostic_roundoff)
    if not bool(torch.isfinite(error)):
        raise GramGuardError("nonfinite Gram error estimate")
    if bool(eigen[0] < -error):
        raise GramGuardError("significantly negative Gram eigenvalue")
    largest = eigen[-1].clamp_min(0)
    constant_squared = cutoff_constant ** 2
    safe_keep = eigen - error > constant_squared * (largest + error)
    safe_drop = eigen + error <= constant_squared * (largest - error).clamp_min(0)
    if not bool((safe_keep | safe_drop).all()):
        raise GramGuardError("ambiguous Gram rank cutoff; no solver fallback")


def guarded_gram_polar(matrix, *, smoothing=.001):
    """Return the versioned, guarded FP32 smoothed-polar direction.

    Accepts strided finite FP32 CPU/CUDA matrices without an autograd graph.
    The qualified profile fixes absolute damping to .001 and uses the ORIGINAL
    matrix dimensions and FP32 epsilon for the strict rank cutoff. Every solve
    promotes BEFORE forming the Gram matrix; no rank truncation, normalization,
    K/V reduction, global setting change or random draw is introduced.

    Numerical guard failures raise :class:`GramGuardError`. Technical solver
    exceptions propagate. Neither path silently retries with another solver.
    An optimizer caller must discard or restore its full affected training state.
    """
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 2
            or matrix.dtype != torch.float32 or matrix.layout != torch.strided
            or matrix.device.type not in ("cpu", "cuda") or matrix.requires_grad):
        raise ValueError("guarded Gram requires a strided FP32 CPU/CUDA matrix without autograd")
    if type(smoothing) not in (int, float) or smoothing != .001:
        raise ValueError("gram_fp64_guarded_v1 requires fixed smoothing .001")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("guarded Gram requires finite input")
    if 0 in matrix.shape:
        return torch.zeros_like(matrix)
    version = matrix._version
    transposed = matrix.shape[0] < matrix.shape[1]
    work = matrix.mT if transposed else matrix
    value = work.double()
    scale = value.abs().amax()
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    z = value / safe_scale
    product = z.mT @ z
    eigen, vectors = torch.linalg.eigh(product, UPLO="L")
    # eigh consumes this lower triangle, not an averaged, different matrix.
    gram = torch.tril(product) + torch.tril(product, diagonal=-1).mT
    if not bool(torch.isfinite(eigen).all()) or not bool(torch.isfinite(vectors).all()):
        raise GramGuardError("nonfinite Gram eigendecomposition")
    constant = max(matrix.shape) * torch.finfo(torch.float32).eps
    keep = eigen > constant * constant * eigen[-1].clamp_min(0)
    scaled_lambda = safe_scale.new_tensor(smoothing) / safe_scale
    if not bool(torch.isfinite(scaled_lambda)):
        raise GramGuardError("unrepresentable scaled absolute smoothing")
    inverse = torch.zeros_like(eigen)
    inverse[keep] = torch.hypot(eigen[keep].sqrt(), scaled_lambda).reciprocal()
    direction = ((z @ vectors) * inverse) @ vectors.mT
    output = direction.to(dtype=torch.float32)
    if not bool(torch.isfinite(direction).all()) or not bool(torch.isfinite(output).all()):
        raise GramGuardError("nonfinite Gram direction")
    _guard_factors(z, gram, eigen, vectors, constant)
    if matrix._version != version:
        raise GramGuardError("input changed during the guarded solve")
    return output.mT.contiguous() if transposed else output
