"""CPU metadata/bitwise arithmetic on retained tensors; no neural execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest


def bits_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                        b.contiguous().reshape(-1).view(torch.uint8)))


def difference(a, b):
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("ULP diagnostic requires the retained float32 tensors")
    def ordered(tensor):
        raw = tensor.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(raw < 0, -raw - 1, raw + 2 ** 31)
    delta = (a.double() - b.double()).abs()
    ulps = (ordered(a) - ordered(b)).abs()
    return {"bit_exact": bits_equal(a, b), "dtype": str(a.dtype),
            "elements": a.numel(), "changed_elements": int((ulps != 0).sum()),
            "max_absolute_difference": float(delta.max()),
            "relative_frobenius_difference": (float(delta.norm() / a.double().norm())
                                               if bool(a.count_nonzero()) else
                                               0.0 if bits_equal(a, b) else None),
            "max_ulp_distance": int(ulps.max())}


def analyze(root):
    files = {f"{arm}/{name}": root / arm / name
             for arm in ("live", "restart") for name in ("trace.pt", "prefix-state.pt")
             if (root / arm / name).exists()}
    files["restart/before-state.pt"] = root / "restart/before-state.pt"
    live, restart = [torch.load(root / arm / "trace.pt", weights_only=True, map_location="cpu")[0]
                     for arm in ("live", "restart")]
    prefix = torch.load(root / "live/prefix-state.pt", weights_only=True, map_location="cpu")
    restored = torch.load(root / "restart/before-state.pt", weights_only=True, map_location="cpu")
    assert live["step"] == restart["step"] == 401
    assert state_digest(prefix) == state_digest(restored)
    before = {name: bits_equal(live["optimizers"]["D"]["before"][name], value)
              for name, value in restart["optimizers"]["D"]["before"].items()}
    grads = {name: difference(live["optimizers"]["D"]["gradients"][name], value)
             for name, value in restart["optimizers"]["D"]["gradients"].items()}
    tensors = [value for values in prefix["trainer"]["models"].values()
               for value in values.values() if isinstance(value, torch.Tensor)]
    assert all(t.dtype == torch.float32 for t in tensors)
    return {"schema_version": 1, "scope": "saved_tensor_metadata_and_bitwise_math",
            "qualification_input": False, "new_neural_updates": 0, "new_model_calls": 0,
            "cpu_neural_execution": False, "observed_update": 401,
            "inputs": {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                       for name, path in files.items()},
            "full_serialized_prefix_bit_exact": True, "prefix_state_digest": state_digest(prefix),
            "critic_weights_before_update": before,
            "real_batch_bit_exact": bits_equal(live["real"], restart["real"]),
            "first_six_forward_outputs_bit_exact": [bits_equal(a["output"], b["output"])
                                                     for a, b in zip(live["forwards"][:6], restart["forwards"][:6])],
            "critic_gradient_differences": grads,
            "model_tensor_count": len(tensors), "model_tensor_dtypes": ["torch.float32"],
            "saved_float32_float64_float32_roundtrip_bit_exact": all(
                bits_equal(t, t.double().float()) for t in tensors),
            "saved_model_state_contains_nonfinite": any(not bool(torch.isfinite(t).all()) for t in tensors),
            "interpretation": "No observed precision loss before backward. The differing gradients are computed float32 outputs; ULP differences do not identify the arithmetic ordering or prove a conversion caused them.",
            "gpu_roundtrip_verified_by_this_analysis": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    atomic_json(args.output, result)
    print(json.dumps({"event": "saved_analysis_complete", "output": str(args.output),
                      "model_tensors": result["model_tensor_count"], "new_neural_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
