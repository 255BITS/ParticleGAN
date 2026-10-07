"""CUDA-only weak-subspace gradient noise for a prospective Ring16 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from particlegan.optim.dualnorm import polar_factor as _ordinary_polar
from experiments.forge.rng import NamedStreams
from experiments.forge.state import state_digest
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _perturbed_input(matrix: torch.Tensor, generator: torch.Generator | None):
    """Perturb only the numerical weak singular subspace in exact arithmetic.

    Float32 reconstruction can round other entries as well. Never change the
    gradient dtype, use ambient randomness, or mutate the original gradient.
    """
    if matrix.ndim != 2 or matrix.dtype != torch.float32 or matrix.device.type != "cuda":
        raise ValueError("Ring16 noise requires a CUDA float32 matrix; no conversion or CPU fallback")
    if min(matrix.shape) == 0 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("Ring16 noise requires a nonempty finite gradient matrix")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    scale = singular[0]
    tau = max(matrix.shape) * torch.finfo(torch.float32).eps * scale
    if bool(scale == 0):
        return matrix, {"zero": True, "weak_directions": 0, "tau": tau,
                        "sigma_max": scale, "noise_std": scale, "noise_drawn": False}
    weak = singular <= tau
    k = int(weak.sum())
    std = torch.finfo(torch.float32).eps * scale / math.sqrt(k) if k else scale * 0
    metadata = {"zero": False, "weak_directions": k, "tau": tau,
                "sigma_max": scale, "noise_std": std, "noise_drawn": bool(k)}
    if not k:
        return matrix, metadata
    if (generator is None or torch.device(generator.device) != matrix.device
            or torch.device(generator.device).type != "cuda"):
        raise ValueError("weak-subspace noise requires the passed isolated CUDA generator")
    noise = torch.randn((k, k), device=matrix.device, dtype=matrix.dtype, generator=generator) * std
    perturbation = left[:, weak] @ noise @ right[weak, :]
    return matrix + perturbation, metadata


@torch.no_grad()
def polar(matrix: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply rounding-scale weak-subspace noise, then the unchanged polar rule."""
    perturbed, metadata = _perturbed_input(matrix, generator)
    return torch.zeros_like(matrix) if metadata["zero"] else _ordinary_polar(perturbed)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clone_streams(state: dict, device: str) -> NamedStreams:
    streams = NamedStreams(0, device=device)
    streams.load_state_dict(state)
    return streams


@reproducible_execution
@torch.no_grad()
def probe(raw: Path, protocol: dict, *, device: str = "cuda:0"):
    """One saved-gradient check with cloned, fully checkpointed noise streams."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("saved-gradient noise probe requires CUDA; no CPU fallback")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    canonical = NamedStreams(0, device=device)
    canonical.generator("noise", component="ring16_intervention", purpose="weak_directions")
    initial = canonical.state_dict()
    retained_states = {"initial": initial}
    pairs, arms, draws = [], {}, 0
    for arm in ("live", "restart"):
        path = raw / arm / "trace.pt"
        expected = protocol["saved_gradient_probe"]["input_sha256"][arm]
        if _sha256(path) != expected:
            raise ValueError("saved gradient trace differs: " + arm)
        source = json.loads((raw / arm / "source.json").read_text())
        execution = protocol["saved_gradient_probe"]["training_execution_sources"][arm]
        if any(source[k] != execution[k] for k in ("origin_commit", "digest")):
            raise ValueError("saved gradient execution provenance differs: " + arm)
        trace = torch.load(path, map_location="cpu", weights_only=True)[0]
        if trace["step"] != 401:
            raise ValueError("saved gradient is not the declared update401")
        row = trace["polar"][1]
        matrix = row["input"].to(device=device)
        if matrix.shape != (64, 64):
            raise ValueError("saved gradient is not the hidden critic64x64 matrix")
        working, repeat_streams = _clone_streams(initial, device), _clone_streams(initial, device)
        generator = working.generator("noise", component="ring16_intervention", purpose="weak_directions")
        repeat_rng = repeat_streams.generator("noise", component="ring16_intervention", purpose="weak_directions")
        perturbed, metadata = _perturbed_input(matrix, generator)
        original, candidate = _ordinary_polar(matrix), _ordinary_polar(perturbed)
        repeat = polar(matrix, repeat_rng)
        draws += 2 * int(metadata["noise_drawn"])
        relative = (perturbed - matrix).norm() / matrix.norm()
        after, repeat_after = working.state_dict(), repeat_streams.state_dict()
        retained_states[arm] = {"after": after, "repeat_after": repeat_after}
        arms[arm] = {
            "trace_sha256": expected, "training_execution_source": execution,
            "weak_directions": metadata["weak_directions"], "noise_drawn": metadata["noise_drawn"],
            "tau": float(metadata["tau"]), "noise_std": float(metadata["noise_std"]),
            "actual_relative_perturbation": float(relative),
            "actual_max_perturbation": float((perturbed - matrix).abs().max()),
            "actual_perturbation_nonzero": bool((perturbed != matrix).any()),
            "relative_factor_change": float((candidate - original).norm() / original.norm()),
            "original_matches_captured": torch.equal(original, row["output"].to(device=device)),
            "same_input_same_noise_repeat_exact": torch.equal(candidate, repeat),
            "repeated_noise_stream_after_exact": state_digest(after) == state_digest(repeat_after),
            "noise_stream_after_digest": state_digest(after),
            "finite": bool(torch.isfinite(perturbed).all() and torch.isfinite(candidate).all()),
        }
        pairs.append((matrix, original, candidate))
    live, restart = pairs
    pair = {
        "raw_relative_frobenius_difference": float((live[0] - restart[0]).norm() / live[0].norm()),
        "ordinary_relative_frobenius_difference": float((live[1] - restart[1]).norm() / live[1].norm()),
        "noisy_relative_frobenius_difference": float((live[2] - restart[2]).norm() / live[2].norm()),
        "noisy_max_difference": float((live[2] - restart[2]).abs().max()),
    }
    checks = {
        "finite": all(a["finite"] for a in arms.values()),
        "actual_relative_perturbation_at_most_1e_minus_5": all(a["actual_relative_perturbation"] <= 1e-5 for a in arms.values()),
        "same_input_same_noise_repeat_exact": all(a["same_input_same_noise_repeat_exact"] for a in arms.values()),
        "repeated_noise_stream_after_exact": all(a["repeated_noise_stream_after_exact"] for a in arms.values()),
        "original_factors_match_captured": all(a["original_matches_captured"] for a in arms.values()),
        "ambient_rng_unchanged": (torch.equal(cpu_rng, torch.get_rng_state())
                                  and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))),
        "canonical_template_stream_unchanged": state_digest(canonical.state_dict()) == state_digest(initial),
    }
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    checks["within_30_second_allowance"] = elapsed <= protocol["saved_gradient_probe"]["timeout_seconds"]
    return {
        "schema_version": 1, "scope": "fixed_saved_gradient_cuda_weak_noise_analysis",
        "device": device, "seed": 0, "gpu_model": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)), "cuda_runtime": torch.version.cuda,
        "training_updates": 0, "sampling_draws": 0, "perturbation_tensor_draws": draws,
        "qualification_input": False, "arms": arms, "checks": checks,
        "implementation_gate_pass": all(checks.values()), "pair": pair,
        "elapsed_seconds": elapsed, "source_sha256": _sha256(Path(__file__)),
        "noise_initial_state_digest": state_digest(initial),
        "noise_checkpoint_artifact": "noise-stream-states.pt",
        "limitations": ["Paired gradient noise shares the initial cloned stream state; weak SVD bases can differ.",
                        "The sensitivity measurement has no improvement gate or convergence implication."],
    }, retained_states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/forge/ring16-noise/protocol.json")
    parser.add_argument("--output", type=Path, required=True, help="New exclusive local artifact directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU fallback and no scientific attempt was started")
    protocol = json.loads(args.protocol.read_text())
    for path, digest in protocol["bindings"].items():
        if _sha256(ROOT / path) != digest:
            raise ValueError("frozen declaration changed: " + path)
    args.output.mkdir(parents=True, exist_ok=False)
    result, states = probe(args.raw, protocol, device=args.device)
    checkpoint = args.output / "noise-stream-states.pt"
    torch.save(states, checkpoint)
    result["noise_checkpoint_sha256"] = _sha256(checkpoint)
    result["protocol_sha256"] = _sha256(args.protocol)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["implementation_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
