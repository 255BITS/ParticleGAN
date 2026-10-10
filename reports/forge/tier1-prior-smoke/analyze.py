"""Summarize saved prior-grid evidence without training or new model draws."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.forge.contracts import atomic_json, file_hash


def analyze(raw, published):
    report = json.loads((published / "results.json").read_text())
    rows = []
    for result in report["runs"]:
        directory = raw / result["arm"] / result["task"]
        if file_hash(directory / "receipt.json") != result["raw_receipt_sha256"]:
            raise ValueError("original receipt changed")
        curve = json.loads((directory / "curve.json").read_text())
        row = dict(arm=result["arm"], task=result["task"], particles=result["particles"],
                   full_verdict=result["full_verdict"], smoke_verdict=result["smoke_verdict"],
                   full_passing_checks=sum(point["full_pass"] for point in curve),
                   smoke_passing_checks=sum(point["smoke_pass"] for point in curve),
                   full_terminal_suffix=result["full_terminal_suffix"],
                   smoke_terminal_suffix=result["smoke_terminal_suffix"],
                   prior_g_row_update_opportunities_expected=result["completed_updates"] * (1 - (1 - 1 / result["particles"]) ** 128),
                   gif=result["gif"], final_metrics=result["final_metrics"])
        if result["task"].startswith("gaussian"):
            row.update(terminal_max_ks=max(point["metrics"]["cdf_ks"] for point in curve[-5:]),
                       terminal_mean_error_range=[min(point["metrics"]["mean_error_sigma"] for point in curve[-5:]), max(point["metrics"]["mean_error_sigma"] for point in curve[-5:])],
                       terminal_std_ratio_range=[min(point["metrics"]["std_ratio"] for point in curve[-5:]), max(point["metrics"]["std_ratio"] for point in curve[-5:])])
        else:
            row.update(locations_per_target_component=result["particles"] / 16,
                       components_above_covariance_bound=sum(value > .85 for value in result["final_metrics"]["component_covariance_errors"]),
                       samples_outside_three_sigma=4096 - sum(result["final_metrics"]["hq_component_counts"]))
        rows.append(row)
    total_seconds = sum(result["elapsed_seconds"] for result in report["runs"])
    atomic_json(published / "analysis.json", dict(
        schema_version=1, qualification_input=False, results_sha256=file_hash(published / "results.json"),
        full_passes=sum(row["full_verdict"] == "PASS" for row in rows),
        smoke_projection_passes=sum(row["smoke_verdict"] == "PASS" for row in rows),
        total_cells=len(rows), rows=rows,
        measured_training_evaluation_checkpoint_seconds=total_seconds,
        cost_scope="Per-trial timer after model construction, source capture and initial checkpoint; includes initial evaluation, training, scoring and final GPU synchronization. Final serialization, process startup, controls and media rendering excluded. No speed ranking.",
        exposure_scope="Analytic expected G minibatches containing each row, assuming uniform independent index draws. This is not a measured optimizer count or a proposed rate adjustment.",
        training_updates_added=0, model_sampling_draws_added=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--published", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.raw, args.published)
