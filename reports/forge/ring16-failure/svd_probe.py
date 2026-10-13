"""Reproduce the saved-gradient CUDA algebra probe; no training or sampling.

This source records the operations used in the original inline probe. --check
compares them with its retained receipt without replacing that receipt.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.state import state_digest


@reproducible_execution
def probe(raw, *, device="cuda:0"):
    if torch.device(device).type != "cuda":
        raise ValueError("CUDA is required")
    result = {"device": device, "scope": "fixed_saved_gradient_cuda_svd_analysis",
              "sampling_draws": 0, "training_updates": 0}
    pairs = []
    for arm in ("live", "restart"):
        row = torch.load(raw / arm / "trace.pt", map_location="cpu", weights_only=True)[0]["polar"][1]
        value = row["input"].to(device)
        singular = torch.linalg.svdvals(value.double())
        u, s, vh = torch.linalg.svd(value, full_matrices=False)
        factor = u @ vh
        u2, _, vh2 = torch.linalg.svd(value, full_matrices=False)
        captured = row["output"].to(device)
        result[arm] = {
            "singular_max": float(singular[0]), "singular_min": float(singular[-1]),
            "float64_singular_values": singular.cpu().tolist(), "float32_singular_values": s.cpu().tolist(),
            "below_1e6_relative": int((singular < singular[0] * 1e-6).sum()),
            "below_default_float32_rank_threshold": int((singular < singular[0] * 64 * torch.finfo(torch.float32).eps).sum()),
            "input_sha256": state_digest(row["input"]),
            "same_input_svd_repeat_exact": torch.equal(factor, u2 @ vh2),
            "recomputed_factor_matches_captured": torch.equal(factor, captured),
            "captured_factor_max_difference": float((factor-captured).abs().max())}
        # Original receipt computes pair norms from saved CPU tensors. SVD
        # and the same-input repeat above remain on CUDA.
        pairs.append((row["input"], factor.cpu()))
    a, b = pairs
    result["pair"] = {
        "input_max_difference": float((a[0]-b[0]).abs().max()),
        "input_relative_frobenius_difference": float((a[0]-b[0]).norm() / a[0].norm()),
        "factor_max_difference": float((a[1]-b[1]).abs().max()),
        "factor_relative_frobenius_difference": float((a[1]-b[1]).norm() / a[1].norm())}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "runs/api/ring16-restart-diagnostic-v1")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = probe(args.raw, device="cuda:0")
    if args.check:
        if result != json.loads((args.raw / "svd-probe.json").read_text()):
            raise ValueError("Saved algebra receipt did not reproduce exactly")
        print(json.dumps({"saved_algebra_receipt_exact": True, "device": "cuda:0", "training_updates": 0}))
    else:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
