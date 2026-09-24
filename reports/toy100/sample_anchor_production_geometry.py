"""Offline first-bank grouping check for the frozen 100-Gaussian sampler.

This reads production configuration and sampler code, then draws exactly the
first training-data minibatch for each declared problem with the native seed.
Evaluator centers are used afterward only to audit inferred groups.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100.problems import PROBLEM_NAMES, evaluation_geometry, sample_real
from reports.toy100.sample_group_anchor import mst_groups


SOURCES = (
    "reports/toy100/sample_anchor_production_geometry.py",
    "reports/toy100/sample_group_anchor.py",
    "benchmarks/toy100/problems.py",
    "benchmarks/toy100/train.py",
    "configs/toy100/constraints_simple_regularization.json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / SOURCES[-1]).read_text())
    assert config["device"] == "cpu"
    assert config["batch_size"] == 2048 and config["num_particles"] == 20000
    assert config["seed"] == 1234
    torch.set_num_threads(1)
    cases = []
    for problem in PROBLEM_NAMES:
        stream = torch.Generator(device="cpu").manual_seed(config["seed"])
        real = sample_real(problem, config["batch_size"], generator=stream)
        centers, _ = evaluation_geometry(problem)
        diagnostic_labels = torch.cdist(real, centers).argmin(1)
        counts = torch.bincount(diagnostic_labels, minlength=len(centers))
        groups, details = mst_groups(real)
        purity = 0
        for members in details["member_indices"]:
            purity += int(torch.bincount(diagnostic_labels[members], minlength=len(centers)).max())
        cases.append(dict(problem=problem,
            first_real_bank_sha256=hashlib.sha256(real.numpy().tobytes()).hexdigest(),
            inferred_groups=len(groups), diagnostic_target_groups=len(centers),
            diagnostic_observed_groups=int((counts > 0).sum()),
            diagnostic_omitted_groups=int((counts == 0).sum()),
            diagnostic_min_count=int(counts.min()), diagnostic_max_count=int(counts.max()),
            diagnostic_group_purity=purity / len(real),
            largest_within_mst_edge=details["largest_within_edge"],
            smallest_between_mst_edge=details["smallest_between_edge"],
            largest_additive_gap=details["largest_additive_gap"],
            second_largest_gap=details["second_largest_gap"],
            data_rng_after_first_bank_sha256=hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest()))
    receipt = dict(scope="first native minibatch per fixed production task; no training",
        seed=config["seed"], batch_size=config["batch_size"],
        learned_particles=config["num_particles"],
        noise_std=config["observation_sigma"],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(json.dumps({key: case[key] for key in
                          ("problem", "inferred_groups", "diagnostic_observed_groups",
                           "diagnostic_omitted_groups", "diagnostic_group_purity")}))


if __name__ == "__main__":
    main()
