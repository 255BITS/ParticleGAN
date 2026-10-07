"""Experiment-only random signs on weak singular directions of polar updates.

Matrix inputs remain CUDA float32. Only the supplied isolated CUDA generator
is consumed. Scheduling and full public-API training belong to the shared
``ring16_interventions`` runner; package defaults are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.rng import NamedStreams
from experiments.forge.sources import runtime_manifest
from experiments.forge.state import state_digest
from .reproducibility import reproducible_execution


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/ring16-sign-flips/protocol.json"
TRACE_HASHES = {
    "live": "7735822645b4d7d03fd4d5952c316e41b83123d67fcba984bb64b8b05bf43145",
    "restart": "87975f57050de221f1c09d4146a0416390e7d3511a50eab6a255f14314ed235d",
}


def _require_matrix(matrix: torch.Tensor, generator: torch.Generator) -> None:
    if matrix.ndim != 2 or matrix.dtype != torch.float32 or 0 in matrix.shape:
        raise ValueError("The declared sign-flip cohort requires a nonempty float32 matrix")
    if matrix.device.type != "cuda":
        raise ValueError("CUDA is required; there is no CPU SVD fallback")
    if not isinstance(generator, torch.Generator) or generator.device != matrix.device:
        raise ValueError("Pass the isolated CUDA generator on the matrix device")


@torch.no_grad()
def _direction(matrix: torch.Tensor, generator: torch.Generator) -> tuple:
    _require_matrix(matrix, generator)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("The matrix direction must be finite")
    if not bool(torch.count_nonzero(matrix)):
        return torch.zeros_like(matrix), None
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(matrix.shape) * torch.finfo(torch.float32).eps * singular[0]
    weak = singular <= threshold
    signs = torch.ones_like(singular)
    count = int(weak.sum())
    if count:
        coins = torch.rand((count,), dtype=torch.float32, device=matrix.device, generator=generator)
        signs[weak] = torch.where(coins < .5, -torch.ones_like(coins), torch.ones_like(coins))
        result = (left * signs) @ right
    else:
        result = left @ right
    return result, {"singular": singular, "threshold": threshold, "weak": weak, "signs": signs}


@torch.no_grad()
def polar(matrix: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Flip each weak spectral sign with probability .5; keep strong signs +1."""
    return _direction(matrix, generator)[0]


def _saved_gradient(path: Path, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    if file_hash(path) != TRACE_HASHES[arm]:
        raise ValueError("The trace must be the original PR331 " + arm + " trace")
    trace = torch.load(path, map_location="cpu", weights_only=True)
    rows = [point for point in trace if point["step"] == 401]
    if len(rows) != 1:
        raise ValueError("Expected exactly one captured update 401")
    matrix = rows[0]["optimizers"]["D"]["gradients"]["net.2.weight"]
    captured = rows[0]["polar"][1]
    if matrix.shape != (64, 64) or matrix.dtype != torch.float32 or not torch.equal(matrix, captured["input"]):
        raise ValueError("Expected the unchanged hidden critic gradient and captured polar input")
    return matrix, captured["output"]


def _declaration(protocol_path: Path) -> dict:
    from .ring16_interventions import declaration
    protocol = declaration(protocol_path)
    if protocol["mechanism_module"] != "benchmarks.toy_audit.ring16_small_sign_flips":
        raise ValueError("The protocol must declare this sign-flip mechanism")
    if protocol["saved_matrix_probe"]["trace_sha256"] != TRACE_HASHES:
        raise ValueError("The saved-gradient source binding changed")
    return protocol


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    return {"max_absolute_difference": float((left - right).abs().max()),
            "relative_frobenius_difference": float((left - right).norm() / left.norm())}


@reproducible_execution
@torch.no_grad()
def _cuda_probe(live_trace: Path, restart_trace: Path, *, device: str,
                checkpoint_path: Path) -> tuple[dict, dict]:
    initial_cpu = torch.get_rng_state().clone()
    initial_cuda = torch.cuda.get_rng_state(device).clone()
    streams = NamedStreams(0, device=device)
    rng = streams.generator("noise", component="ring16_intervention", purpose="weak_directions")
    initial_perturbation = rng.get_state().clone()
    checkpoints = {"initial": streams.state_dict()}
    torch.save(checkpoints, checkpoint_path)

    def observed_direction(matrix, label):
        try:
            return _direction(matrix, rng)
        finally:
            # Preserve the actual consumed state even when an operation fails.
            checkpoints[label] = streams.state_dict()
            torch.save(checkpoints, checkpoint_path)
    result = {
        "schema_version": 1, "status": "COMPLETE",
        "scope": "fixed_saved401_weak_sign_flip_cuda_algebra", "device": str(torch.device(device)),
        "runtime": runtime_manifest(), "cuda_model": torch.cuda.get_device_name(device),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_runtime": torch.version.cuda,
        "training_updates": 0, "sampling_draws": 0, "qualification_input": False,
        "threshold_rule": "s <= max(rows,cols)*finfo(float32).eps*smax",
        "flip_probability": .5, "dtype_conversion": False,
        "source_sha256": file_hash(Path(__file__)), "stream_manifest": streams.manifest(), "arms": {},
    }
    pairs = []
    for arm, path in (("live", live_trace), ("restart", restart_trace)):
        saved, captured = _saved_gradient(path, arm)
        matrix = saved.to(device=device)
        left, _, right = torch.linalg.svd(matrix, full_matrices=False)
        full = left @ right
        # Reset only the separate perturbation stream; ambient/data/prior RNGs
        # are untouched. Equal coins do not imply equal weak spectral bases.
        rng.set_state(initial_perturbation)
        paired_start_exact = torch.equal(rng.get_state(), initial_perturbation)
        flipped, spectral = observed_direction(matrix, arm + "_after_first")
        after = rng.get_state().clone()
        rng.set_state(initial_perturbation)
        repeat_start_exact = torch.equal(rng.get_state(), initial_perturbation)
        repeat, _ = observed_direction(matrix, arm + "_after_repeat")
        weak, signs, singular = spectral["weak"], spectral["signs"], spectral["singular"]
        result["arms"][arm] = {
            "trace_sha256": file_hash(path), "input_digest": state_digest(saved),
            "weak_count": int(weak.sum()), "exact_zero_count": int((singular == 0).sum()),
            "flipped_count": int((signs[weak] < 0).sum()), "threshold": float(spectral["threshold"]),
            "finite": bool(torch.isfinite(flipped).all()),
            "strong_signs_all_plus_one": bool((signs[~weak] == 1).all()),
            "weak_signs_are_plus_or_minus_one": bool(((signs[weak] == 1) | (signs[weak] == -1)).all()),
            "same_input_same_rng_repeat_exact": torch.equal(flipped, repeat),
            "same_input_rng_progress_exact": torch.equal(after, rng.get_state()),
            "paired_perturbation_start_exact": paired_start_exact,
            "repeat_perturbation_start_exact": repeat_start_exact,
            "full_polar_matches_captured": torch.equal(full, captured.to(device=device)),
            "change_from_ordinary_direction": _difference(full, flipped),
            "weak_signs": signs[weak].cpu().tolist(),
        }
        pairs.append((matrix, full, flipped))
    live, restart = pairs
    result["raw_gradient"] = _difference(live[0], restart[0])
    result["full_polar"] = _difference(live[1], restart[1])
    result["same_draw_sign_flipped_pair"] = _difference(live[2], restart[2])
    result["paired_inputs_restore_same_perturbation_state"] = all(
        arm["paired_perturbation_start_exact"] for arm in result["arms"].values())
    # Explicit separate algebra fixtures exercise no-draw guards. They never
    # replace the learned initialization or the archived gradient pair.
    fixture_results = {}
    for name, matrix in (("zero", torch.zeros((2, 2), device=device, dtype=torch.float32)),
                         ("identity_no_weak", torch.eye(2, device=device, dtype=torch.float32))):
        before = rng.get_state().clone()
        factor, _ = observed_direction(matrix, "fixture_" + name)
        fixture_results[name] = {"scope": "fixed_algebra_fixture_separate_cohort",
                                 "rng_unchanged": torch.equal(before, rng.get_state()),
                                 "factor_exact": torch.equal(factor, matrix)}
    result["fixtures"] = fixture_results
    result["ambient_rng_preserved"] = {
        "cpu": torch.equal(initial_cpu, torch.get_rng_state()),
        "cuda": torch.equal(initial_cuda, torch.cuda.get_rng_state(device)),
    }
    result["consumed_stream_keys"] = list(streams.state_dict()["states"])
    result["checkpoint_labels"] = list(checkpoints)
    valid = all(arm["finite"] and arm["strong_signs_all_plus_one"]
                and arm["weak_signs_are_plus_or_minus_one"] and arm["same_input_same_rng_repeat_exact"]
                and arm["same_input_rng_progress_exact"] and arm["paired_perturbation_start_exact"]
                and arm["repeat_perturbation_start_exact"] for arm in result["arms"].values())
    valid = valid and all(result["ambient_rng_preserved"].values())
    valid = valid and all(row["rng_unchanged"] and row["factor_exact"] for row in fixture_results.values())
    result["perturbation_gate"] = {
        "pass": valid, "quality_gate": False,
        "criterion": "finite factors; exact same-input/same-RNG repeats; strong signs+1; only checkpointed perturbation stream consumed; ambient RNG unchanged; no-draw zero/no-weak guards",
        "mismatch_reduction_required": False,
    }
    torch.cuda.synchronize(device)
    return result, checkpoints


def probe(live_trace: Path, restart_trace: Path, *, device: str, checkpoint_path: Path,
          protocol_path: Path = PROTOCOL) -> dict:
    started = time.monotonic()
    protocol = _declaration(protocol_path)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("A CUDA device is required; no CPU algebra substitutes")
    result, checkpoints = _cuda_probe(live_trace, restart_trace, device=device,
                                     checkpoint_path=checkpoint_path)
    torch.save(checkpoints, checkpoint_path)
    result["rng_checkpoint_sha256"] = file_hash(checkpoint_path)
    result["all_consumed_stream_states_checkpointed"] = all(
        set(snapshot["states"]) == set(result["consumed_stream_keys"]) for snapshot in checkpoints.values())
    if not result["all_consumed_stream_states_checkpointed"]:
        result["perturbation_gate"]["pass"] = False
    result["protocol_sha256"] = file_hash(protocol_path)
    result["trace_execution_sources"] = protocol["saved_matrix_probe"]["trace_execution_sources"]
    result["elapsed_seconds"] = time.monotonic() - started
    result["within_30_second_allowance"] = result["elapsed_seconds"] <= 30
    if not result["within_30_second_allowance"]:
        result["perturbation_gate"]["pass"] = False
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
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = probe(args.live_trace, args.restart_trace, device=args.device,
                       protocol_path=args.protocol.resolve(), checkpoint_path=args.output / "rng-checkpoints.pt")
    except Exception as exc:
        atomic_json(args.output / "receipt.json", {
            "scope": "fixed_saved401_weak_sign_flip_cuda_algebra", "status": "INCOMPLETE",
            "error": {"type": type(exc).__name__, "message": str(exc)}, "device": args.device,
            "training_updates": 0, "sampling_draws": 0, "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": file_hash(args.protocol.resolve()), "qualification_input": False,
        })
        raise
    atomic_json(args.output / "receipt.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if not result["perturbation_gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
