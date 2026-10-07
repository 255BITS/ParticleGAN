"""Experiment-only CUDA float32 spectral truncation and saved-gradient probe.

The public optimizer's matrix direction becomes U diag(s > tau) Vh, with
tau = max(rows, columns) * float32_epsilon * s_max. Scheduling belongs to
``ring16_interventions``; this module does not change production defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sources import runtime_manifest
from experiments.forge.state import state_digest
from .reproducibility import reproducible_execution


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/ring16-truncation/protocol.json"
TRACE_HASHES = {
    "live": "7735822645b4d7d03fd4d5952c316e41b83123d67fcba984bb64b8b05bf43145",
    "restart": "87975f57050de221f1c09d4146a0416390e7d3511a50eab6a255f14314ed235d",
}


def _require_matrix(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.dtype != torch.float32:
        raise ValueError("The declared truncation cohort requires a float32 matrix")
    if matrix.device.type != "cuda":
        raise ValueError("CUDA is required; there is no CPU SVD fallback")
    if 0 in matrix.shape:
        raise ValueError("The declared matrix must have nonzero dimensions")


@torch.no_grad()
def polar(matrix: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Truncate numerically null directions without casting or consuming RNG."""
    _require_matrix(matrix)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("The matrix direction must be finite")
    if not bool(torch.count_nonzero(matrix)):
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(matrix.shape) * torch.finfo(torch.float32).eps * singular[0]
    return (left * (singular > threshold)) @ right


def _saved_gradient(path: Path, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    if file_hash(path) != TRACE_HASHES[arm]:
        raise ValueError("The trace must be the original PR331 " + arm + " trace")
    trace = torch.load(path, map_location="cpu", weights_only=True)
    rows = [point for point in trace if point["step"] == 401]
    if len(rows) != 1:
        raise ValueError("Expected exactly one captured update 401")
    matrix = rows[0]["optimizers"]["D"]["gradients"]["net.2.weight"]
    if matrix.shape != (64, 64) or matrix.dtype != torch.float32:
        raise ValueError("Expected the unchanged hidden critic gradient")
    captured = rows[0]["polar"][1]
    if not torch.equal(matrix, captured["input"]):
        raise ValueError("The captured polar input must be the hidden critic gradient")
    return matrix, captured["output"]


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    return {
        "max_absolute_difference": float((left - right).abs().max()),
        "relative_frobenius_difference": float((left - right).norm() / left.norm()),
    }


def _declaration(protocol_path: Path) -> dict:
    from .ring16_interventions import declaration
    protocol = declaration(protocol_path)
    if protocol["mechanism_module"] != "benchmarks.toy_audit.ring16_spectral_truncation":
        raise ValueError("The protocol must declare this truncation mechanism")
    if protocol["saved_matrix_probe"]["trace_sha256"] != TRACE_HASHES:
        raise ValueError("The saved-gradient source binding changed")
    return protocol


def probe(live_trace: Path, restart_trace: Path, *, device: str,
          protocol_path: Path = PROTOCOL) -> dict:
    """Compare one fixed saved pair on CUDA; no trainer or sampling calls."""
    started = time.monotonic()
    protocol = _declaration(protocol_path)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("A CUDA device is required; no CPU algebra substitutes")
    result = _cuda_probe(live_trace, restart_trace, device=device)
    result["protocol_sha256"] = file_hash(protocol_path)
    result["trace_execution_sources"] = protocol["saved_matrix_probe"]["trace_execution_sources"]
    result["elapsed_seconds"] = time.monotonic() - started
    result["within_30_second_allowance"] = result["elapsed_seconds"] <= 30
    if not result["within_30_second_allowance"]:
        result["mechanistic_falsifier"]["pass"] = False
    return result


@reproducible_execution
@torch.no_grad()
def _cuda_probe(live_trace: Path, restart_trace: Path, *, device: str) -> dict:
    started = time.monotonic()
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state(device).clone()
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": "fixed_saved401_truncation_cuda_algebra",
        "device": str(torch.device(device)),
        "runtime": runtime_manifest(),
        "cuda_model": torch.cuda.get_device_name(device),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "training_updates": 0,
        "sampling_draws": 0,
        "rng_draws": 0,
        "qualification_input": False,
        "threshold_rule": "max(rows,cols)*finfo(float32).eps*smax",
        "dtype_conversion": False,
        "source_sha256": file_hash(Path(__file__)),
        "arms": {},
    }
    pairs = []
    for arm, path in (("live", live_trace), ("restart", restart_trace)):
        saved, captured = _saved_gradient(path, arm)
        matrix = saved.to(device=device)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        full = left @ right
        threshold = max(matrix.shape) * torch.finfo(torch.float32).eps * singular[0]
        truncated = polar(matrix)
        full_repeat_left, _, full_repeat_right = torch.linalg.svd(matrix, full_matrices=False)
        truncated_repeat = polar(matrix)
        result["arms"][arm] = {
            "trace_sha256": file_hash(path),
            "input_digest": state_digest(saved),
            "threshold": float(threshold),
            "discarded_directions": int((singular <= threshold).sum()),
            "retained_directions": int((singular > threshold).sum()),
            "finite": bool(torch.isfinite(truncated).all()),
            "full_polar_matches_captured": torch.equal(full, captured.to(device=device)),
            "same_input_full_repeat_exact": torch.equal(full, full_repeat_left @ full_repeat_right),
            "same_input_truncation_repeat_exact": torch.equal(truncated, truncated_repeat),
        }
        pairs.append((matrix, full, truncated))
    live, restart = pairs
    result["raw_gradient"] = _difference(live[0], restart[0])
    result["full_polar"] = _difference(live[1], restart[1])
    result["truncated_polar"] = _difference(live[2], restart[2])
    full_difference = result["full_polar"]["relative_frobenius_difference"]
    truncated_difference = result["truncated_polar"]["relative_frobenius_difference"]
    repeated = all(
        arm["finite"] and arm["same_input_full_repeat_exact"]
        and arm["same_input_truncation_repeat_exact"] and arm["full_polar_matches_captured"]
        for arm in result["arms"].values()
    )
    result["ambient_rng_preserved"] = {
        "cpu": torch.equal(initial_cpu_rng, torch.get_rng_state()),
        "cuda": torch.equal(initial_cuda_rng, torch.cuda.get_rng_state(device)),
    }
    rng_preserved = all(result["ambient_rng_preserved"].values())
    result["mechanistic_falsifier"] = {
        "required_reduction_factor": 10,
        "pass": repeated and rng_preserved and truncated_difference <= full_difference / 10,
        "criterion": "finite, exact repeats, captured full-factor parity, unchanged RNGs, and >=10x reduction in pair relative Frobenius difference",
        "quality_gate": False,
    }
    torch.cuda.synchronize(device)
    result["cuda_algebra_elapsed_seconds"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe"])
    parser.add_argument("--live-trace", type=Path, required=True)
    parser.add_argument("--restart-trace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _declaration(args.protocol.resolve())
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required; no algebra attempt or output directory created")
    # Exclusive outputs prevent silently replacing a completed probe receipt.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = probe(args.live_trace, args.restart_trace, device=args.device,
                       protocol_path=args.protocol.resolve())
    except Exception as exc:
        atomic_json(args.output / "receipt.json", {
            "scope": "fixed_saved401_truncation_cuda_algebra", "status": "INCOMPLETE",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "device": args.device, "training_updates": 0, "sampling_draws": 0,
            "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": file_hash(args.protocol.resolve()), "qualification_input": False,
        })
        raise
    atomic_json(args.output / "receipt.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if not result["mechanistic_falsifier"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
