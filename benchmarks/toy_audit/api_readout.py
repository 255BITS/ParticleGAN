"""Produce a sorted, source-bound readout of existing API toy attempts.

This launches no training, sampling, rescoring or renderer. Historical definition
scores remain separate from new model outcomes and narrowed variant claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import api_contract, api_publish, api_reframe, api_run


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def readout(archive, output):
    archive, output = Path(archive), Path(output)
    definitions = api_contract.discover()
    paths = {path.parent.name: path for path in archive.glob("*/receipt.json")}
    if set(paths) != set(definitions):
        raise ValueError("readout requires exactly one retained attempt per registered variant")
    historical = api_publish.read(api_contract.ROOT / "reports/toy_audit/improvements.json")
    ratings = {row["id"]: row["followup_rating"] for row in historical["rows"]}
    catalog = api_publish.read(api_contract.ROOT / "reports/toy_audit/catalog.json")
    ratings.update({row["id"]: row["rating"] for row in catalog["cases"] if row["id"] not in ratings})
    rows, totals = [], Counter()
    for name, path in sorted(paths.items()):
        receipt = api_publish.read(path)
        if receipt["status"] == "COMPLETE":
            receipt = api_publish.verify_run(path.parent)
        else:
            receipt, _, _, _ = api_reframe.verify_failure_view(path.parent)
        case = definitions[name]
        if receipt["case"] != api_run.json_value(case):
            raise ValueError(f"{name}: changed registered definition")
        metric_steps = receipt["protocol"]["metric_evaluation_steps"]
        terminal_count = receipt["protocol"]["terminal_observations"]
        records = [record for record in receipt["observations"]
                   if record["step"] > 0 and record["step"] in metric_steps]
        terminal = records[-terminal_count:]
        failed_checks = [{"step": record["step"], "failed_bounds": record["failed_bounds"]}
                         for record in terminal if not record["passed"]]
        endpoint = receipt["observations"][-1]
        source_scores = {legacy: ratings[legacy] for legacy in case["legacy_ids"] if legacy in ratings}
        row = {"id": name, "legacy_ids": case["legacy_ids"], "source_definition_scores": source_scores,
               "goal": case["goal"], "scope": case["scope"], "provider": case["provider"],
               "execution_status": receipt["status"], "verdict": receipt["verdict"],
               "completed_updates": receipt["completed_updates"], "default_updates": case["default_steps"],
               "final_observation_passed": endpoint["passed"], "failed_bounds": receipt["failed_bounds"],
               "failed_terminal_checks": failed_checks, "elapsed_seconds": receipt["elapsed_seconds"],
               "raw_receipt_sha256": api_run.file_hash(path), "source_commit": receipt["source"]["commit"],
               "gif": f"media/{name}{'-failure' if receipt['status'] == 'ERROR' else ''}.gif"}
        rows.append(row)
        totals[f"{receipt['status']}/{receipt['verdict']}"] += 1
    rows.sort(key=lambda row: (-max(row["source_definition_scores"].values(), default=0),
                              row["verdict"] != "PASS", row["execution_status"] != "COMPLETE", row["id"]))
    endpoint_only = [row["id"] for row in rows if row["execution_status"] == "COMPLETE"
                     and row["final_observation_passed"] and row["verdict"] == "FAIL"]
    data = {"schema": "particlegan_api_toy_readout_v1", "training_or_rescoring": False,
            "historical_definition_scores_changed": False, "variant_definition_scores_assigned": False,
            "original_archive": str(archive.resolve()), "outcomes": dict(sorted(totals.items())),
            "registered_variants": len(rows), "completed_updates": sum(row["completed_updates"] for row in rows),
            "sum_per_case_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "endpoint_pass_default_fail": endpoint_only, "cases": rows}
    output.mkdir(parents=True, exist_ok=True)
    api_run.write_json(output / "readout.json", data)
    lines = ["# Sorted API toy results", "",
             "All registered attempts are retained. Sort: historical question definition score",
             "descending, then default PASS, completed FAIL, execution ERROR, and case ID.",
             "The /5 score belongs to the original question; it is **not a new rating for**",
             "**a narrowed API variant**, a convergence score, or a count of passing gates.", "",
             f"**{totals['COMPLETE/PASS']} PASS · {totals['COMPLETE/FAIL']} completed FAIL · "
             f"{totals['ERROR/FAIL']} ERROR/FAIL · {len(rows)} actual goal GIFs.**", "",
             "[Full readout](RUN_REPORT.md) · [Numerical/provenance receipts](readout.json) ·",
             "[Exact definitions and frozen gates](cases.json) · [Gallery](GALLERY.md)", "",
             "A useful negative control can have a FAIL model outcome. Each failed bound is",
             "observed evidence; it does not by itself identify the optimizer's causal mechanism.",
             "An endpoint PASS cannot replace the required terminal window or completed execution.", "",
             "| Source score | API variant / GIF | What it verifies | Outcome | Rejected bounds / failed terminal checks |",
             "|---|---|---|---|---|"]
    for row in rows:
        scores = ", ".join(f"{name}: {score}/5" for name, score in row["source_definition_scores"].items()) or "Unrated source"
        outcome = f"{row['execution_status']} / {row['verdict']}"
        if row["final_observation_passed"] and row["verdict"] == "FAIL":
            outcome += " (last metric PASS)"
        failures = list(row["failed_bounds"])
        failures.extend(f"update {record['step']}: {', '.join(record['failed_bounds'])}"
                        for record in row["failed_terminal_checks"])
        reason = "; ".join(failures) or "All declared terminal checks pass"
        lines.append(f"| {_text(scores)} | [{row['id']}]({row['gif']}) | {_text(row['goal'])} | "
                     f"{_text(outcome)} | {_text(reason)} |")
    (output / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = readout(args.runs, args.output)
    print(result["outcomes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
