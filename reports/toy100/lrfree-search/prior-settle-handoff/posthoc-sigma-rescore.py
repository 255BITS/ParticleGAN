"""Read-only paired-noise sensitivity of the H2 prior handoff native clouds."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs/h2-prior-handoff"
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

TASKS = ("grid100", "staggered100")
SIGMAS = (0.02, 0.021, 0.022)
STEPS = (6000, 6250, 6500, 6750, 7000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    inputs = {}
    old_sigmas = {}
    for task in TASKS:
        run = RUNS / task
        old = {}
        for line in (run / "native-noisy/events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "eval" and event["model"] == "live" and event["step"] in STEPS:
                old[event["step"]] = event["output_sigma"]
        assert sorted(old) == list(STEPS)
        old["holdout"] = old[7000]
        old_sigmas[task] = old
        for split in (*STEPS, "holdout"):
            relative = (f"quality_checks/step_{split:06d}.npz" if isinstance(split, int)
                        else "holdout_samples.npz")
            clean_path = run / "native-clean" / relative
            noisy_path = run / "native-noisy" / relative
            inputs[f"{task}/clean/{relative}"] = sha(clean_path)
            inputs[f"{task}/noisy/{relative}"] = sha(noisy_path)
            with np.load(clean_path) as ca, np.load(noisy_path) as na:
                for model in ("live", "ema"):
                    clean, noisy = ca[model], na[model]
                    delta = noisy-clean
                    for sigma in SIGMAS:
                        cloud = (noisy if sigma == 0.02 else
                                 (clean + np.float32(sigma/old[split])*delta).astype(np.float32))
                        coverage = evaluate_samples(torch.from_numpy(cloud), task)
                        acc = evaluate_accuracy(cloud, task, gate_metrics=coverage)
                        rows.append(dict(task=task, split=split, model=model, sigma=sigma,
                                         passed=acc["passed"], frozen_pass=acc["frozen_pass"],
                                         accuracy_pass=acc["accuracy_pass"],
                                         precision=coverage["precision"],
                                         min_hq_mode_mass=coverage["min_hq_mode_mass"],
                                         max_mode_mass=coverage["max_mode_mass"],
                                         min_cov_eig_ratio=coverage["min_cov_eig_ratio"],
                                         max_cov_eig_ratio=coverage["max_cov_eig_ratio"],
                                         mass_tv=acc["mass_tv"],
                                         center_rms_sigma=acc["center_rms_sigma"],
                                         abs_cov_trace_bias=acc["abs_cov_trace_bias"],
                                         radial_ks=acc["radial_ks"]))
    summary = []
    for task in TASKS:
        for model in ("live", "ema"):
            for sigma in SIGMAS:
                selected = [r for r in rows if r["task"] == task and r["model"] == model and r["sigma"] == sigma]
                finals = [r for r in selected if r["split"] != "holdout"]
                holdout = next(r for r in selected if r["split"] == "holdout")
                summary.append(dict(task=task, model=model, sigma=sigma,
                                    final_check_passes=sum(r["passed"] for r in finals),
                                    holdout_pass=holdout["passed"],
                                    all_six_pass=all(r["passed"] for r in selected),
                                    worst_precision=min(r["precision"] for r in selected),
                                    worst_radial_ks=max(r["radial_ks"] for r in selected),
                                    worst_abs_cov_trace_bias=max(r["abs_cov_trace_bias"] for r in selected),
                                    worst_center_rms_sigma=max(r["center_rms_sigma"] for r in selected)))
    result = dict(method="original noisy at .02; otherwise clean + float32(target_sigma/original_sigma_at_observation)*(noisy-clean)",
                  diagnostic_only=True, frozen_repo=str(FROZEN), fixture_sha256=sha(FIXTURE_PATH),
                  original_sigma=old_sigmas, tasks=TASKS, sigmas=SIGMAS, steps=STEPS,
                  input_sha256=inputs, summary=summary, rows=rows)
    (HERE / "handoff_sigma_rescore.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
