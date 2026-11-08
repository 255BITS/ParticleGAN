"""Replay the historical algebra microbenchmarks on saved training gradients.

This loads existing local artifacts and launches no training. Outputs contain
one final timing summary per matrix batch and algebra prototype. The public
optimizer adds validation/dispatch, so this is not its end-to-end speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(matrix):
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    weights = singular / torch.hypot(singular, singular.new_tensor(.001))
    weights *= singular > max(matrix.shape[-2:]) * torch.finfo(matrix.dtype).eps * singular[..., :1]
    return (left * weights.unsqueeze(-2)) @ right


def masked_eigh64(matrix):
    scale = matrix.abs().amax((-2, -1), keepdim=True).clamp_min(.001)
    value = matrix.double() / scale.double()
    eigenvalues, vectors = torch.linalg.eigh(value.mT @ value)
    cutoff = (max(matrix.shape[-2:]) * torch.finfo(matrix.dtype).eps
              * eigenvalues[..., -1:].clamp_min(0).sqrt())
    inverse = torch.where(
        eigenvalues > cutoff.square(),
        torch.rsqrt(eigenvalues.clamp_min(0) + (.001 / scale.double()).flatten()[:, None].square()), 0)
    return (((value @ vectors) * inverse.unsqueeze(-2)) @ vectors.mT).to(matrix.dtype)


def quintic_prototype(matrix):
    identity = torch.eye(matrix.shape[-1], device=matrix.device)
    scale = matrix.abs().amax((-2, -1), keepdim=True).clamp_min(.001)
    value, regularizer = matrix / scale, .001 / scale
    normalizer = (value.square().sum((-2, -1), keepdim=True) + regularizer.square()).sqrt()
    upper, lower = value / normalizer, regularizer * identity / normalizer
    for _ in range(5):
        gram = upper.mT @ upper + lower.mT @ lower
        correction = 3.4445 * identity - 4.775 * gram + 2.0315 * (gram @ gram)
        upper, lower = upper @ correction, lower @ correction
    return upper / 1.21


def frobenius_prototype(matrix):
    scale = matrix.abs().amax((-2, -1), keepdim=True).clamp_min(.001)
    value = matrix / scale
    return value / (value.square().sum((-2, -1), keepdim=True) + (.001 / scale).square()).sqrt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True, type=Path,
                        help="Existing guard-replay-v1 directory or byte-exact archive extraction")
    parser.add_argument("--receipt", type=Path, default=Path(__file__).with_name("kernel-audit.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path, help="Local final-summary JSON")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    receipt = json.loads(args.receipt.read_text())
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows = []
    for artifact in receipt["input_artifacts"]:
        path = args.matrix_root / artifact["name"]
        if file_hash(path) != artifact["sha256"]:
            raise ValueError("saved matrix artifact bytes changed: " + str(path))
        saved = torch.load(path, weights_only=True, map_location="cpu")
        matrix = saved["matrices"].to(device)
        if matrix.dtype != torch.float32 or saved["smoothing"] != .001:
            raise ValueError("the historical prototype contract is FP32 with smoothing .001")
        reference = native(matrix)
        sync()
        for name, function in (("masked_eigh64", masked_eigh64),
                               ("muon5_augmented_bound1_21", quintic_prototype),
                               ("frobenius_smoothed", frobenius_prototype)):
            function(matrix)
            sync()
            times = []
            for _ in range(args.repeats):
                tick = time.perf_counter()
                result = function(matrix)
                sync()
                times.append((time.perf_counter() - tick) * 1000)
            row = dict(file=artifact["name"], shape=list(matrix.shape), method=name,
                       median_ms=statistics.median(times),
                       relative_error_vs_native=((result-reference).norm()/reference.norm()).item(),
                       max_error_vs_native=(result-reference).abs().max().item(),
                       first_matrix_operator_norm=torch.linalg.svdvals(result[0].double())[0].item(),
                       finite=torch.isfinite(result).all().item())
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    result = dict(schema="bispf_kernel_replay_v1", scope=receipt["scope"],
                  qualification_input=False, default_adoption=False,
                  prototype_source_sha256=file_hash(__file__), torch=str(torch.__version__),
                  cuda=torch.version.cuda, device=str(device), repeats=args.repeats,
                  input_artifacts=receipt["input_artifacts"], measurements=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
