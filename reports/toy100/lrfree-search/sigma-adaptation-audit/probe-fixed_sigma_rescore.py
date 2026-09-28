"""Diagnostic paired-noise fixed-sigma rescore of full-sigma terminal clouds.

This reads saved generated clouds and the frozen evaluator; it does not change
training, the model, or protocol thresholds. The same noise realization at
each observation is scaled from that observation's original output sigma.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "runs/h2-prior-couple-sigma/grid100"
FIXTURE_PATH = Path("/ml2/hypergan/lrfree-20260926/harness/tasks/native100_fixture.json")
FIXTURE = json.loads(FIXTURE_PATH.read_text())
FROZEN = Path(FIXTURE["frozen_repo"])
for relative, expected in FIXTURE["host_source_sha256"].items():
    actual = hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"frozen host source changed: {relative}")
sys.path.insert(0, str(FROZEN))
from benchmarks.toy100.accuracy import evaluate_accuracy  # noqa: E402
from benchmarks.toy100.metrics import evaluate_samples  # noqa: E402

SIGMAS = tuple(float(x) for x in os.environ.get("FIXED_SIGMAS", "0.015,0.017,0.019").split(","))
STEPS = (6000, 6250, 6500, 6750, 7000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = {}
    for line in (RUN / "native-noisy/events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "eval" and row["model"] == "live" and row["step"] in STEPS:
            original[row["step"]] = row["output_sigma"]
    assert sorted(original) == list(STEPS)
    original["holdout"] = original[7000]
    rows = []
    inputs = {}
    for split in (*STEPS, "holdout"):
        relative = (f"quality_checks/step_{split:06d}.npz" if isinstance(split, int)
                    else "holdout_samples.npz")
        clean_path = RUN / "native-clean" / relative
        noisy_path = RUN / "native-noisy" / relative
        inputs[f"clean/{relative}"] = sha(clean_path)
        inputs[f"noisy/{relative}"] = sha(noisy_path)
        with np.load(clean_path) as ca, np.load(noisy_path) as na:
            for model in ("live", "ema"):
                clean, noisy = ca[model], na[model]
                delta = noisy - clean
                for sigma in SIGMAS:
                    cloud = (clean + np.float32(sigma / original[split]) * delta).astype(np.float32)
                    coverage = evaluate_samples(torch.from_numpy(cloud), "grid100")
                    acc = evaluate_accuracy(cloud, "grid100", gate_metrics=coverage)
                    rows.append(dict(split=split, model=model, sigma=sigma,
                                     passed=acc["passed"], accuracy_pass=acc["accuracy_pass"],
                                     frozen_pass=acc["frozen_pass"], precision=coverage["precision"],
                                     center_rms_sigma=acc["center_rms_sigma"],
                                     abs_cov_trace_bias=acc["abs_cov_trace_bias"],
                                     radial_ks=acc["radial_ks"], mass_tv=acc["mass_tv"],
                                     min_cov_eig_ratio=coverage["min_cov_eig_ratio"],
                                     max_cov_eig_ratio=coverage["max_cov_eig_ratio"]))
    summary = []
    for model in ("live", "ema"):
        for sigma in SIGMAS:
            selected = [r for r in rows if r["model"] == model and r["sigma"] == sigma]
            summary.append(dict(model=model, sigma=sigma, all_six_pass=all(r["passed"] for r in selected),
                                final_check_passes=sum(r["passed"] for r in selected if r["split"] != "holdout"),
                                holdout_pass=next(r["passed"] for r in selected if r["split"] == "holdout")))
    result = dict(method="clean + float32(target_sigma/original_sigma_at_observation) * (noisy-clean)",
                  diagnostic_only=True, frozen_repo=str(FROZEN), fixture_sha256=sha(FIXTURE_PATH),
                  original_sigma=original, sigmas=SIGMAS, steps=STEPS, input_sha256=inputs,
                  summary=summary, rows=rows)
    (HERE / os.environ.get("FIXED_REPORT", "fixed_sigma_rescore.json")).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(summary=summary, live_rows=[r for r in rows if r["model"] == "live"]), indent=2))


if __name__ == "__main__":
    main()
