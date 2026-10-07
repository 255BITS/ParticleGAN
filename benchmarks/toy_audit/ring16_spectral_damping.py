"""CUDA-only smooth spectral damping for the Ring16 diagnostic.

This helper is experiment-scoped. The public trainer's optimizer invokes it
only when the prospective intervention runner activates the declared schedule.
It does not modify the production optimizer or any Recipe defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def polar(matrix: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return U diag(s / hypot(s, tau)) Vh without changing the input dtype.

    tau = max(rows, columns) * float32 epsilon * max(s). This numerical-rank
    scale is fixed across matrix sizes and roles, rather than tuned to Ring16.
    No random draw is made; ``generator`` shares the runner's mechanism API.
    """
    if matrix.ndim != 2 or matrix.dtype != torch.float32 or matrix.device.type != "cuda":
        raise ValueError("Ring16 damping requires a CUDA float32 matrix; no conversion or CPU fallback")
    if min(matrix.shape) == 0:
        raise ValueError("Ring16 damping requires a nonempty matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Ring16 damping requires finite gradients")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    scale = singular[0]
    if bool(scale == 0):
        return torch.zeros_like(matrix)
    tau = max(matrix.shape) * torch.finfo(torch.float32).eps * scale
    # hypot avoids overflow/underflow from explicitly squaring s or tau.
    weights = singular / torch.hypot(singular, tau)
    return (left * weights.unsqueeze(0)) @ right


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@reproducible_execution
@torch.no_grad()
def probe(raw: Path, protocol: dict, *, device: str = "cuda:0") -> dict:
    """Compare the two archived 401 gradients on CUDA, without model updates."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("saved-gradient damping probe requires CUDA; no CPU fallback")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    pairs, arms = [], {}
    for arm in ("live", "restart"):
        path = raw / arm / "trace.pt"
        expected = protocol["saved_gradient_probe"]["input_sha256"][arm]
        if _sha256(path) != expected:
            raise ValueError("saved gradient trace differs: " + arm)
        trace = torch.load(path, map_location="cpu", weights_only=True)[0]
        if trace["step"] != 401:
            raise ValueError("saved gradient is not the declared update401")
        row = trace["polar"][1]
        matrix = row["input"].to(device=device)
        if matrix.shape != (64, 64):
            raise ValueError("saved gradient is not the hidden critic64x64 matrix")
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        original = left @ right
        damped, repeat = polar(matrix, None), polar(matrix, None)
        tau = 64 * torch.finfo(torch.float32).eps * singular[0]
        arms[arm] = {
            "trace_sha256": expected,
            "singular_max_float32": float(singular[0]),
            "tau": float(tau),
            "weak_directions": int((singular <= tau).sum()),
            "original_matches_captured": torch.equal(original, row["output"].to(device=device)),
            "same_input_damping_repeat_exact": torch.equal(damped, repeat),
            "finite": bool(torch.isfinite(damped).all()),
        }
        pairs.append((matrix, original, damped))
    live, restart = pairs
    original_difference = (live[1] - restart[1]).norm() / live[1].norm()
    damped_difference = (live[2] - restart[2]).norm() / live[2].norm()
    ratio = damped_difference / original_difference
    unchanged_rng = (torch.equal(cpu_rng, torch.get_rng_state())
                     and torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
    checks = {
        "finite": all(arm["finite"] for arm in arms.values()),
        "same_input_repeat_exact": all(arm["same_input_damping_repeat_exact"] for arm in arms.values()),
        "original_factors_match_captured": all(arm["original_matches_captured"] for arm in arms.values()),
        "at_least_tenfold_discrepancy_reduction": bool(ratio <= .1),
        "ambient_rng_unchanged": unchanged_rng,
    }
    pair = {
        "raw_max_difference": float((live[0] - restart[0]).abs().max()),
        "raw_relative_frobenius_difference": float((live[0] - restart[0]).norm() / live[0].norm()),
        "original_relative_frobenius_difference": float(original_difference),
        "damped_relative_frobenius_difference": float(damped_difference),
        "damped_over_original_discrepancy": float(ratio),
        "damped_max_difference": float((live[2] - restart[2]).abs().max()),
    }
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    checks["within_30_second_allowance"] = elapsed <= protocol["saved_gradient_probe"]["timeout_seconds"]
    return {
        "schema_version": 1, "scope": "fixed_saved_gradient_cuda_damping_analysis",
        "device": device, "seed": 0, "training_updates": 0, "sampling_draws": 0,
        "qualification_input": False, "arms": arms, "checks": checks,
        "mechanistic_pass": all(checks.values()),
        "pair": pair,
        "elapsed_seconds": elapsed,
        "source_sha256": _sha256(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Hydrated PR331 raw root containing live/trace.pt and restart/trace.pt")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "reports/forge/ring16-damping/protocol.json")
    parser.add_argument("--output", type=Path, required=True, help="New, exclusive local artifact directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # Reject unsupported hosts before creating a scientific attempt directory.
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU fallback and no scientific attempt was started")
    protocol = json.loads(args.protocol.read_text())
    for path, digest in protocol["bindings"].items():
        if _sha256(ROOT / path) != digest:
            raise ValueError("frozen declaration changed: " + path)
    args.output.mkdir(parents=True, exist_ok=False)
    result = probe(args.raw, protocol, device=args.device)
    result["protocol_sha256"] = _sha256(args.protocol)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["mechanistic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
