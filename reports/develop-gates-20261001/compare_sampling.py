"""Diagnose the CI sampling-law change without retraining or rewriting evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.toy100.accuracy_gate import HOLDOUT_SEED_OFFSETS
from benchmarks.toy100.problems import PROBLEM_NAMES


def score(points, problem):
    result = evaluate_accuracy(points, problem)
    fields = ("precision", "mass_tv", "center_rms_sigma", "cov_trace_bias",
              "radial_ks", "frozen_pass", "passed")
    return {key: result[key] for key in fields}


def compare(path, problem, sigma, noise_seed):
    with np.load(path, allow_pickle=False) as archive:
        clean = torch.from_numpy(archive["live"].copy())
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(noise_seed),
                        dtype=clean.dtype)
    return {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "draws": len(clean), "noise_seed": noise_seed,
            "clean": score(clean, problem),
            "training_noise": score(clean + sigma * noise, problem)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CI toy100 artifact directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"kind": "paired_saved_draw_diagnostic", "diagnostic_only": True,
              "scientific_qualification": False, "new_training_updates": 0,
              "source_run": "https://github.com/255BITS/ParticleGAN/actions/runs/36829031177",
              "source_head": "fa2c378d5ea4dd8f66eabeb17973818a174732ef",
              "torch": str(torch.__version__), "cases": {}}
    for problem in PROBLEM_NAMES:
        root = args.input / problem
        summary = json.loads((root / "summary.json").read_text())
        config = summary["config"]
        assert summary["eval_output_noise"] == "clean"
        assert not config.get("output_noise_learnable", False)
        seed, sigma = config["seed"], config["output_noise_std"]
        checks = [compare(root / "quality_checks" / f"step_{step:06d}.npz",
                          problem, sigma, seed + 402)
                  for step in summary["accuracy"]["check_steps"]]
        holdout = compare(root / "holdout_samples.npz", problem, sigma,
                          seed + HOLDOUT_SEED_OFFSETS["noise"])
        report["cases"][problem] = {
            "sigma": sigma, "checks": checks, "holdout": holdout,
            "passing_terminal_checks": {
                law: sum(check[law]["passed"] for check in checks)
                for law in ("clean", "training_noise")}}
        print(json.dumps({"problem": problem,
                          "terminal": report["cases"][problem]["passing_terminal_checks"],
                          "holdout": {law: holdout[law] for law in ("clean", "training_noise")}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
