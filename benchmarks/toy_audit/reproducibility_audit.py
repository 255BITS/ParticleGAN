"""Bounded public-API software proof; regeneration alone launches no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from experiments.forge.contracts import atomic_json, atomic_text
from . import api_contract, api_run
from .reproducibility import DEFAULT_SEED, VERSION

REPORT = Path("reports/forge/reproducibility")
UPDATES = 8


def run(output, report):
    """Twenty-four CPU updates, one seed, no distribution qualification."""
    torch.set_num_threads(1)
    case = api_contract.discover()["api-vector-two-broad"]
    receipts = [api_run.run_case(case, output / name, recipe_name=recipe, steps=UPDATES,
                                eval_samples=128, frames=3, wall_cap_seconds=20.)
                for name, recipe in (("k3p", "k3p"), ("k3p-repeat", "k3p"), ("ka2", "ka2"))]
    if any(row["status"] != "COMPLETE" or row["completed_updates"] != UPDATES for row in receipts):
        raise ValueError("all bounded public-API executions must complete")
    first, repeat, alternative = [row["reproducibility"] for row in receipts]
    states = [torch.load(output / name / "final-state.pt", map_location="cpu", weights_only=True)
              for name in ("k3p", "k3p-repeat")]
    from experiments.forge.state import state_digest
    checks = {
        "repeat_initial_state_mismatches": int(first["initial_state_sha256"] != repeat["initial_state_sha256"]),
        "repeat_final_state_mismatches": int(state_digest(states[0]) != state_digest(states[1])),
        "cross_trainer_initial_model_mismatches": sum(first["initial_models"][key] != alternative["initial_models"][key]
                                                     for key in first["initial_models"]),
        "batch_sequence_mismatches": sum(first["batch_sequence_sha256"] != row["batch_sequence_sha256"]
                                         for row in (repeat, alternative)),
    }
    result = {"schema": VERSION, "scope": "bounded_public_api_reproducibility_software_control",
              "seed": DEFAULT_SEED, "updates_per_run": UPDATES,
              "budget": {"runs": 3, "cpu_threads": 1, "timeout_seconds_per_run": 20},
              "case": case, "bounds": dict.fromkeys(checks, 0), "metrics": checks,
              "passed": all(value == 0 for value in checks.values()), "qualification_credit": False,
              "runs": [{"recipe": row["recipe"], "source": row["source"], "runtime": row["runtime"],
                        "reproducibility": row["reproducibility"], "quality_verdict": row["verdict"],
                        "default_protocol_complete": row["default_protocol_complete"],
                        "artifacts": row["artifacts"]} for row in receipts]}
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(report / "final-metrics.json", result)
    for name in ("k3p", "ka2"):
        shutil.copyfile(output / name / "goal.gif", report / (name + ".gif"))
    return result


def render(result):
    rows = ["# Public-API reproducibility audit", "",
            "New comparisons use seed `0`, the repository initializer, fixed task conditions and isolated data streams. "
            "This bounded software control compares the same vector task under K3P and KA2, and repeats K3P with the same seed.", "",
            f"Budget: three CPU runs, {result['updates_per_run']} updates each, one Torch thread, 20 seconds per run. "
            "Each numerical bound was zero mismatches before execution. No full distribution-quality qualification is claimed.", "",
            "| Check | Mismatches | Bound | Result |", "| --- | ---: | ---: | --- |"]
    rows.extend(f"| {name.replace('_', ' ')} | {value} | 0 | {'PASS' if value == 0 else 'FAIL'} |"
                for name, value in result["metrics"].items())
    rows += ["", f"Reproducibility verdict: **{'PASS' if result['passed'] else 'FAIL'}**. "
             "The short runs remain FAIL for their unchanged full training-quality gates.", "",
             "Actual target/output training GIFs: [K3P](k3p.gif), [KA2](ka2.gif). "
             "[Final metrics and source/runtime/artifact hashes](final-metrics.json) bind the measured execution. "
             "Raw states, arrays and progress receipts remain in the local run directory.", "",
             "Use one complete global trainer configuration across the task ladder; hold each task's architecture, "
             "target law, batch size, initialization, prior, sampling and budget fixed across candidates. "
             "Fixed identity/zero or stored-weight controls and named initialization diagnostics remain explicit separate cohorts. "
             "Archived results retain their original sources and are not regraded. "
             "The [current solution leaderboard](../technique-inventory.md) retains its recorded qualifications; "
             "this control selects no winning trainer.", "",
             "Regenerate this readout without training:", "",
             "```sh", "python -m benchmarks.toy_audit.reproducibility_audit", "```", "",
             "Reproduce the bounded public-API proof in a fresh local directory:", "",
             "```sh", "python -u -m benchmarks.toy_audit.reproducibility_audit --run --output runs/reproducibility-proof", "```", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/reproducibility-proof"))
    args = parser.parse_args()
    result = run(args.output, REPORT) if args.run else json.loads((REPORT / "final-metrics.json").read_text())
    atomic_text(REPORT / "README.md", render(result))
    print(json.dumps({"passed": result["passed"], "metrics": result["metrics"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
