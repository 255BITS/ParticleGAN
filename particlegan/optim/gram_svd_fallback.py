"""Explicit opt-in FP64 SVD with diagnostic-only rank uncertainty.

Returns a direction BEFORE any caller leaf write. Earlier optimizer leaves and
clocks are not rolled back on failure. Full owning state must be discarded or
restored after HOLD. Error estimates use the unchanged 8x standard floating-point model.
Rank uncertainty is diagnostic only: the strict computed SVD mask is applied,
including dropping computed exact ties. This is not rank certification.
Other numerical guards and technical errors remain fatal.
"""
from dataclasses import dataclass
import hashlib
import math
import sys
import time

import torch
from .gram_polar import GramGuardError, guarded_gram_polar as _original_guard

VERSION = "gram_fp64_svd_diagnostic_v2"
AMBIGUOUS_MESSAGE = "ambiguous Gram rank cutoff; no solver fallback"


@dataclass
class Counters:
    calls: int = 0
    gram_passes: int = 0
    rank_ambiguities: int = 0
    svd_attempts: int = 0
    fallback_passes: int = 0
    uncertified_svd_rank_decisions: int = 0
    fallback_holds: int = 0
    propagated_other_errors: int = 0
    primary_wall_seconds: float = 0.
    fallback_wall_seconds: float = 0.
    CPU_input_copy_and_hash_seconds: float = 0.
    CPU64_analysis_seconds: float = 0.
    GPU64_analysis_seconds: float = 0.
    return_transfer_host_seconds: float = 0.
    last_fallback: dict | None = None


@dataclass(frozen=True)
class Decision:
    direction: torch.Tensor
    path: str
    diagnostics: dict


class SVDHoldError(GramGuardError):
    """CPU64 analysis remains unresolved; no direction is returned/applied."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _gamma(count):
    product = count * (torch.finfo(torch.float64).eps / 2)
    if product >= 1:
        raise SVDHoldError("CPU64 diagnostic dimensions exceed the error model", {})
    return product / (1-product)


def _factor_guard(z, u, singular, vh):
    """Bound an orthogonal-factor SVD of original input using THESE factors.

    Reconstruction accounts for U*S rounding and matrix multiplication. Factor
    orthogonality defects bound distances to nearby isometries. Division and
    diagnostic-reduction roundoff are included. Weyl then screens the original
    epsilon32 strict cutoff on singular values, without squaring conditioning.
    """
    rows, columns = z.shape; width = len(singular); unit = torch.finfo(torch.float64).eps / 2
    identity = torch.eye(width, dtype=torch.float64, device=z.device)
    observed_left = float((u.T @ u-identity).norm())
    observed_right = float((vh @ vh.T-identity).norm())
    # Include roundoff of the diagnostic orthogonality products themselves.
    left_product_roundoff = _gamma(rows)*float((u.abs().T @ u.abs()).norm())
    right_product_roundoff = _gamma(columns)*float((vh.abs() @ vh.abs().T).norm())
    eta_left = observed_left + left_product_roundoff
    eta_right = observed_right + right_product_roundoff
    reconstruction = float((z-(u*singular) @ vh).norm())
    diagnostics = dict(left_orthogonality_frobenius=observed_left, right_orthogonality_frobenius=observed_right,
        left_orthogonality_model_bound=eta_left, right_orthogonality_model_bound=eta_right,
        reconstruction_frobenius=reconstruction, error_model="8x standard FP64 model; not interval certificate",
        same_factors_for_guard_and_output=True, strict_greater_than=True,
        original_fp32_cutoff_constant=max(rows, columns)*torch.finfo(torch.float32).eps)
    if (not all(math.isfinite(value) for value in (eta_left, eta_right, reconstruction))
            or max(eta_left, eta_right) >= 1):
        raise SVDHoldError("CPU64 SVD orthogonality/reconstruction is unresolved", diagnostics)
    # Fatal factor-validity checks are independent of rank certification.
    # A factor's own measured defect must not pay for its acceptance budget.
    left_orthogonality_budget = 8*(left_product_roundoff
        + _gamma(width*width)*(observed_left+left_product_roundoff))
    right_orthogonality_budget = 8*(right_product_roundoff
        + _gamma(width*width)*(observed_right+right_product_roundoff))
    diagnostics.update(left_orthogonality_guard_budget=left_orthogonality_budget,
                       right_orthogonality_guard_budget=right_orthogonality_budget)
    if (observed_left > left_orthogonality_budget
            or observed_right > right_orthogonality_budget):
        raise SVDHoldError("SVD orthogonality exceeds the FP64 error model", diagnostics)
    left_distance = eta_left / (1+math.sqrt(1-eta_left))
    right_distance = eta_right / (1+math.sqrt(1-eta_right))
    correction = float(singular[0]) * (left_distance*math.sqrt(1+eta_right)+right_distance)
    norm = float(z.norm())
    division_roundoff = unit/(1-unit) * norm
    reconstruction_abs = float(((u.abs()*singular.abs()) @ vh.abs()).norm())
    reconstruction_roundoff = (_gamma(width)+unit+_gamma(width)*unit) * reconstruction_abs
    diagnostic_roundoff = (_gamma(z.numel())+_gamma(max(rows, columns))) * (norm+reconstruction)
    error = 8*(reconstruction+correction+division_roundoff+reconstruction_roundoff+diagnostic_roundoff)
    constant = diagnostics["original_fp32_cutoff_constant"]; largest = float(singular[0])
    safe_keep = singular-error > constant*(largest+error)
    safe_drop = singular+error <= constant*max(largest-error, 0.)
    ambiguous = ~(safe_keep | safe_drop)
    diagnostics.update(normalized_singular_error_estimate=error, fixed_error_margin=8,
        normalized_tau=constant*largest, normalized_cutoff_gap=float((singular-constant*largest).abs().min()),
        mask_ambiguous_count=int(ambiguous.sum()), reconstruction_roundoff_estimate=reconstruction_roundoff,
        diagnostic_roundoff_estimate=diagnostic_roundoff, factor_correction_estimate=correction)
    if not math.isfinite(error):
        raise SVDHoldError("nonfinite SVD error model; HOLD before write", diagnostics)
    reconstruction_budget = 8*(division_roundoff+reconstruction_roundoff+diagnostic_roundoff)
    diagnostics["reconstruction_guard_budget"] = reconstruction_budget
    if reconstruction > reconstruction_budget:
        raise SVDHoldError("SVD reconstruction exceeds the FP64 error model", diagnostics)
    diagnostics.update(rank_decision_policy="public_strict_computed_svd",
        rank_certified=not bool(ambiguous.any()),
        uncertified_svd_rank_decision=bool(ambiguous.any()))
    return diagnostics



def _sync_selected(device):
    # This is used ONLY after original rank ambiguity. Never change global state.
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def _svd_factors(z):
    # Exact measured CUDA QR driver. Technical errors propagate; never retry.
    if z.device.type == "cuda":
        return torch.linalg.svd(z, full_matrices=False, driver="gesvd")
    return torch.linalg.svd(z, full_matrices=False)


def _device_svd(matrix, smoothing, version, counters=None):
    counters = counters if counters is not None else Counters()
    diagnostics = {"shape": list(matrix.shape), "device": str(matrix.device),
                   "driver": "gesvd" if matrix.device.type == "cuda" else "native_cpu"}
    counters.last_fallback = _summary(diagnostics, "pending")
    tick = time.monotonic()
    try:
        _sync_selected(matrix.device)
        original = matrix.detach().to(device="cpu", copy=True).contiguous()
        diagnostics["input_sha256"] = hashlib.sha256(
            memoryview(original.reshape(-1).view(torch.uint8).numpy())).hexdigest()
    finally:
        elapsed = time.monotonic()-tick
        counters.CPU_input_copy_and_hash_seconds += elapsed
        diagnostics["CPU_input_copy_and_hash_seconds"] = elapsed
    tick = time.monotonic()
    try:
        # CPU snapshot is for the raw-input hash only on CUDA; solver stays local.
        value = matrix.double() if matrix.device.type == "cuda" else original.double()
        scale = value.abs().amax()
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        z = value/safe_scale
        u, singular, vh = _svd_factors(z)
        if (not bool(torch.isfinite(u).all() & torch.isfinite(singular).all() & torch.isfinite(vh).all())
                or bool((singular < 0).any()) or bool((singular[:-1] < singular[1:]).any())):
            raise SVDHoldError("FP64 SVD factors are nonfinite/negative/unsorted", diagnostics)
        try:
            diagnostics.update(_factor_guard(z, u, singular, vh))
        except SVDHoldError as error:
            diagnostics.update(error.diagnostics)
            error.diagnostics = diagnostics
            raise
        cutoff = max(matrix.shape)*torch.finfo(torch.float32).eps*singular[0]
        keep = singular > cutoff
        weights = torch.zeros_like(singular)
        scaled_lambda = safe_scale.new_tensor(smoothing)/safe_scale
        if not bool(torch.isfinite(scaled_lambda)):
            raise SVDHoldError("unrepresentable scaled absolute smoothing", diagnostics)
        weights[keep] = singular[keep]/torch.hypot(singular[keep], scaled_lambda)
        direction64 = (u*weights) @ vh
        if not bool(torch.isfinite(direction64).all()):
            raise SVDHoldError("FP64 weighted reconstruction is nonfinite", diagnostics)
        direction32 = direction64.to(dtype=torch.float32)  # Sole output floating dtype cast.
        if not bool(torch.isfinite(direction32).all()):
            raise SVDHoldError("FP32 weighted reconstruction is nonfinite", diagnostics)
        diagnostics.update(original_tau=float(cutoff*safe_scale),
            retained_rank=int(keep.sum()), retained_mask=keep.tolist(), lambda_absolute=smoothing,
            output_float32_casts=1, no_parameter_write=True)
    finally:
        if sys.exc_info()[0] is None:
            _sync_selected(matrix.device)
        elapsed = time.monotonic()-tick
        if matrix.device.type == "cuda":
            counters.GPU64_analysis_seconds += elapsed
        else:
            counters.CPU64_analysis_seconds += elapsed
        diagnostics["device_analysis_seconds"] = elapsed
        # Also retained if a technical exception propagates unchanged.
        counters.last_fallback = _summary(diagnostics, "pending")
    tick = time.monotonic()
    try:
        result = direction32.to(device=matrix.device)
        # Complete the selected device's transfer before ending host wall timing.
        _sync_selected(matrix.device)
    finally:
        elapsed = time.monotonic()-tick
        counters.return_transfer_host_seconds += elapsed
        diagnostics["return_transfer_host_seconds"] = elapsed
    if matrix._version != version:
        raise SVDHoldError("input changed during fallback; HOLD before write", diagnostics)
    return Decision(result, "gpu_fp64_gesvd_fallback" if matrix.device.type == "cuda"
                    else "cpu_fp64_svd_fallback", diagnostics)


def _summary(diagnostics, status):
    return {"raw_input_sha256": diagnostics.get("input_sha256"),
        "shape": diagnostics.get("shape"), "retained_rank": diagnostics.get("retained_rank"),
        "normalized_cutoff_gap": diagnostics.get("normalized_cutoff_gap"),
        "normalized_singular_error_estimate": diagnostics.get("normalized_singular_error_estimate"),
        "mask_ambiguous_count": diagnostics.get("mask_ambiguous_count"),
        "path": "gpu_fp64_gesvd_fallback" if str(diagnostics.get("device", "")).startswith("cuda")
                else "cpu_fp64_svd_fallback", "device": diagnostics.get("device"),
        "driver": diagnostics.get("driver"),
        "rank_decision_policy": diagnostics.get("rank_decision_policy"),
        "rank_certified": diagnostics.get("rank_certified"),
        "uncertified_svd_rank_decision": diagnostics.get("uncertified_svd_rank_decision"),
        "status": status}


def solve(matrix, *, smoothing=.001, counters=None):
    """Call ORIGINAL public guard, catching only its exact rank-ambiguity error.

    A returned Decision is permission for a proposed caller's next leaf write,
    NOT policy/training adoption. No leaf/optimizer write occurs in this module.
    Caller-owned counters are the only intentional mutable reporting object.
    """
    if counters is None:
        counters = Counters()
    if not isinstance(counters, Counters):
        raise ValueError("optional Counters reporting object required")
    counters.calls += 1
    version = matrix._version if isinstance(matrix, torch.Tensor) else None
    primary_tick = time.monotonic()
    try:
        result = _original_guard(matrix, smoothing=smoothing)
    except GramGuardError as error:
        if type(error) is not GramGuardError or str(error) != AMBIGUOUS_MESSAGE:
            counters.propagated_other_errors += 1
            raise
        counters.rank_ambiguities += 1
    except Exception:
        counters.propagated_other_errors += 1
        raise
    else:
        counters.gram_passes += 1
        return Decision(result, "original_guarded_gram", dict(version=VERSION, no_parameter_write=True))
    finally:
        counters.primary_wall_seconds += time.monotonic()-primary_tick
    # Original public input validation already passed before this exact message.
    if version is None or matrix._version != version:
        counters.fallback_holds += 1
        raise SVDHoldError("input changed during original guard; HOLD before fallback/write", {})
    counters.svd_attempts += 1; tick = time.monotonic()
    try:
        decision = _device_svd(matrix, smoothing, version, counters)
    except SVDHoldError as error:
        counters.last_fallback = _summary(error.diagnostics, "HOLD")
        counters.fallback_holds += 1
        raise
    except Exception:
        if counters.last_fallback is not None:
            counters.last_fallback["status"] = "technical_error"
        counters.propagated_other_errors += 1
        raise
    else:
        counters.last_fallback = _summary(decision.diagnostics, "PASS")
        counters.fallback_passes += 1
        counters.uncertified_svd_rank_decisions += int(
            decision.diagnostics.get("uncertified_svd_rank_decision", False))
        decision.diagnostics["version"] = VERSION
        return decision
    finally:
        counters.fallback_wall_seconds += time.monotonic()-tick


def svd_fallback_polar(matrix, *, smoothing=.001, counters=None):
    """Return FP32 smoothed polar via original Gram or device-local FP64 SVD.

    Only the exact original ambiguous-rank error permits fallback. All other
    guards and technical errors propagate. Rank uncertainty is diagnostic only;
    the strict computed SVD mask is always used. Other failures occur before
    this leaf's write; prior optimizer leaves and clocks may have advanced. Diagnostics are caller-owned, nonnumerical, and nonserialized.
    CUDA uses explicit gesvd and never retries on CPU. The diagnostic input
    hash copies FP32 bytes to CPU; no CUDA solver factors are transferred there.
    Fallback wall time includes the hash and selected-device synchronization.
    """
    return solve(matrix, smoothing=smoothing, counters=counters).direction
