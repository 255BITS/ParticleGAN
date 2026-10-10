#!/usr/bin/env python3
"""Compare retained short-profile packets, including actual pre-change source."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


def differences(left, right, path=""):
    rows = []
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return [{"path": path, "kind": "tensor_contract"}]
        if not torch.equal(left, right):
            rows.append({"path": path, "kind": "tensor", "max_abs": float(
                (left.detach().cpu().double() - right.detach().cpu().double()).abs().max())})
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            rows.append({"path": path, "kind": "keys", "left_only": sorted(left.keys() - right.keys()),
                         "right_only": sorted(right.keys() - left.keys())})
        for key in sorted(left.keys() & right.keys(), key=str):
            rows.extend(differences(left[key], right[key], f"{path}/{key}"))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [{"path": path, "kind": "length"}]
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(differences(a, b, f"{path}/{index}"))
    elif left != right:
        rows.append({"path": path, "kind": "value", "left": left, "right": right})
    return rows


def remove_declared_backend(value):
    """Project only the two explicit optional configuration locations."""
    if isinstance(value, dict):
        if value.get("optimizer_svd_backend") == "cpu":
            value.pop("optimizer_svd_backend")
        if "dualnorm" in value:
            assert value["dualnorm"].get("svd_backend", "native") == "cpu"
            value["dualnorm"].pop("svd_backend")
        for item in value.values():
            remove_declared_backend(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            remove_declared_backend(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root
    def load(name):
        return torch.load(root / name / "state.pt", map_location="cpu", weights_only=False)
    old, native, diagnostic_cpu, active_cpu = map(load,
        ("baseline", "implemented-native", "cpu-svd", "implemented-cpu"))
    native_diff = differences(old, native)
    projected_cpu = deepcopy(active_cpu)
    remove_declared_backend(projected_cpu)
    cpu_implementation_diff = differences(diagnostic_cpu, projected_cpu)
    # These four earlier independent profiles did not align ambient RNG starts.
    # Retain that difference explicitly; all consumed named streams are checked.
    ambient_paths = {"/fixture/api_state/cpu_rng", "/fixture/api_state/cuda_rng"}
    ambient = {"native": [r for r in native_diff if r["path"] in ambient_paths],
               "cpu": [r for r in cpu_implementation_diff if r["path"] in ambient_paths]}
    native_diff = [r for r in native_diff if r["path"] not in ambient_paths]
    cpu_implementation_diff = [r for r in cpu_implementation_diff if r["path"] not in ambient_paths]
    numerical_delta = differences(native, projected_cpu)
    numerical_delta = [r for r in numerical_delta if r["path"] not in ambient_paths]
    rng_diff = differences(native["streams"], active_cpu["streams"])
    data_rng_diff = differences(native["fixture"]["data_generator"], active_cpu["fixture"]["data_generator"])
    receipt = dict(scope="software_diagnostic_only", qualification_input=False,
        baseline_source_commit="79f7ddb512f0bc1e1457257e85c97adbe6d11bd1", matched_completed_updates=256,
        native_consumed_state_exact=not native_diff,
        ambient_rng_differences=ambient,
        ambient_scope="Original exploratory processes had unmatched ambient initialization; independent replay.json checks aligned whole-state parity.",
        cpu_backend_matches_exploratory_tensor_path=not cpu_implementation_diff,
        declared_projection=["Recipe.optimizer_svd_backend", "dualnorm.svd_backend"],
        named_streams_exact_across_numerical_modes=not rng_diff,
        data_rng_exact_across_numerical_modes=not data_rng_diff,
        native_differences=native_diff, implementation_differences=cpu_implementation_diff,
        numerical_delta_different_tensor_leaves=sum(r["kind"] == "tensor" for r in numerical_delta),
        numerical_delta_max_abs=max((r.get("max_abs", 0.) for r in numerical_delta), default=0.),
        numerical_delta_examples=numerical_delta[:12],
        packets={name: hashlib.sha256((root / name / "state.pt").read_bytes()).hexdigest()
                 for name in ("baseline", "implemented-native", "cpu-svd", "implemented-cpu")})
    (root / "parity.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    assert not native_diff and not cpu_implementation_diff and not rng_diff and not data_rng_diff
    assert receipt["numerical_delta_different_tensor_leaves"] > 0, "CUDA backend delta must be declared"


if __name__ == "__main__":
    main()
