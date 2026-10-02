"""Publish verified goal GIFs and compact receipts from API toy runs.

This reads existing actual observations; it launches no training or rescoring.
Checkpoints, numeric observation arrays and progress logs remain in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from . import api_contract, api_run


def read(path):
    return json.loads(Path(path).read_text())


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _flags(record, name):
    if type(record.get("passed")) is not bool:
        raise ValueError(f"{name}: passed must be binary")
    failures = record.get("failed_bounds")
    if not isinstance(failures, list) or any(not isinstance(value, str) or not value.strip() for value in failures):
        raise ValueError(f"{name}: failed_bounds must identify rejected bounds")
    if record["passed"] != (not failures):
        raise ValueError(f"{name}: binary pass/failed_bounds inconsistency")


def _verify_grade(receipt):
    """Rederive protocol completion/terminal grade, without model rescoring."""
    case = api_contract.validate_case(receipt["case"])
    protocol = receipt["protocol"]
    updates = _integer(protocol["updates"], "protocol updates")
    samples = _integer(protocol["evaluation_samples"], "evaluation samples")
    metric_count = api_contract.metric_observations(case)
    terminal_count = case.get("terminal_observations", 5)
    for key, expected in (("default_updates", case["default_steps"]),
                          ("default_evaluation_samples", case["eval_samples"]),
                          ("metric_observations", metric_count),
                          ("terminal_observations", terminal_count)):
        if _integer(protocol[key], key) != expected:
            raise ValueError(f"protocol {key} differs from registered case")
    frames = _integer(protocol["media_frames"], "requested media frames")
    metric_steps = api_contract.evaluation_steps(updates, min(updates, metric_count) + 1)
    media_steps = api_contract.evaluation_steps(updates, frames)
    union_steps = sorted(set(metric_steps) | set(media_steps))
    for key, expected in (("metric_evaluation_steps", metric_steps), ("media_steps", media_steps),
                          ("evaluation_steps", union_steps)):
        actual = protocol[key]
        if not isinstance(actual, list) or any(type(step) is not int for step in actual) or actual != expected:
            raise ValueError(f"protocol {key} differs from exact frozen step coverage")
    if _integer(receipt["completed_updates"], "completed updates") != updates:
        raise ValueError("completion does not match executed update budget")
    observations = receipt["observations"]
    if not isinstance(observations, list) or len(observations) < 2:
        raise ValueError("training media requires initial and post-update observations")
    actual_steps = [record.get("step") for record in observations]
    if any(type(step) is not int for step in actual_steps) or actual_steps != union_steps:
        raise ValueError("missing, duplicate or out-of-order observation step coverage")
    for record in observations:
        _flags(record, f"observation{record['step']}")
        if not isinstance(record.get("metrics"), dict) or not record["metrics"]:
            raise ValueError("observation must retain its measured metrics")
        if not isinstance(record.get("views"), list) or not record["views"]:
            raise ValueError("observation must retain actual goal views")
    metric_records = [record for record in observations if record["step"] > 0 and record["step"] in metric_steps]
    terminal = metric_records[-terminal_count:]
    final = observations[-1]
    sustained = len(terminal) == terminal_count and all(record["passed"] for record in terminal)
    full = (updates >= case["default_steps"] and samples >= case["eval_samples"]
            and len(metric_records) >= metric_count)
    expected_failures = list(final["failed_bounds"])
    if not full:
        expected_failures.append("default budget, evaluation draw count or metric cadence not completed")
    if not sustained:
        expected_failures.append(f"last {terminal_count} post-update metric observations do not all pass")
    passed = full and sustained
    for key, expected in (("metric_passed", final["passed"]), ("sustained_metric_passed", sustained),
                          ("default_protocol_complete", full), ("passed", passed)):
        if type(receipt.get(key)) is not bool or receipt[key] != expected:
            raise ValueError(f"{key} differs from rederived protocol grade")
    _flags(receipt, "receipt")
    if receipt["failed_bounds"] != expected_failures:
        raise ValueError("receipt failed_bounds differ from rederived protocol grade")
    if receipt.get("verdict") != ("PASS" if passed else "FAIL"):
        raise ValueError("verdict differs from rederived protocol grade")
    if _integer(receipt["gif_frames"], "retained GIF frames", minimum=2) != len(media_steps):
        raise ValueError("goal GIF count differs from declared media steps")
    return media_steps


def verify_run(path):
    path = Path(path)
    receipt = read(path / "receipt.json")
    if receipt["status"] != "COMPLETE" or receipt["source_unchanged"] is not True:
        raise ValueError(f"{path}: unsuccessful API execution cannot publish as completed media")
    for name in ("goal.gif", "observations.npz", "final-state.pt"):
        expected = receipt["artifacts"][name]
        artifact = path / name
        if artifact.stat().st_size != expected["bytes"] or api_run.file_hash(artifact) != expected["sha256"]:
            raise ValueError(f"{path}: {name} identity mismatch")
    media_steps = _verify_grade(receipt)
    observations = receipt["observations"]
    with Image.open(path / "goal.gif") as gif:
        if gif.n_frames != len(media_steps) or gif.n_frames != receipt["gif_frames"]:
            raise ValueError("goal GIF lost actual observed states")
    expected_keys = {f"step{observation['step']}_view{index}_{role}"
                     for observation in observations
                     for index in range(len(observation["views"]))
                     for role in ("target", "samples")}
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("numeric media observations differ from receipt")
        for name in arrays.files:
            values = arrays[name]
            if not values.size or not np.issubdtype(values.dtype, np.number):
                raise ValueError("missing actual numeric goal observations")
            if name.endswith("_target") and not np.isfinite(values).all():
                raise ValueError("nonfinite reference goal")
    return receipt


def publish(archives, output):
    output = Path(output)
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    cases = api_contract.discover()
    rows, seen, manifests = [], set(), {}
    for archive in archives:
        archive = Path(archive)
        for summary in read(archive / "summary.json")["cases"]:
            name = summary["id"]
            if name in seen:
                raise ValueError(f"ambiguous repeated case receipt: {name}")
            seen.add(name)
            receipt_path = archive / name / "receipt.json"
            receipt = verify_run(receipt_path.parent)
            definition = api_run.json_value(cases[name])
            if receipt["case"] != definition:
                raise ValueError(f"{name}: executed definition differs from registered variant")
            source = receipt["source"]
            source_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            # Different imported dependency sets remain separate source receipts.
            manifests[source_key] = source
            target = media / f"{name}.gif"
            if target.exists() and api_run.file_hash(target) != receipt["artifacts"]["goal.gif"]["sha256"]:
                raise ValueError(f"refusing to replace earlier media for {name}")
            if not target.exists():
                shutil.copyfile(receipt_path.parent / "goal.gif", target)
            final = receipt["observations"][-1]
            rows.append({"id": name, "legacy_ids": definition["legacy_ids"],
                         "goal": definition["goal"], "scope": definition["scope"],
                         "sampling": definition["sampling"], "thresholds": definition["thresholds"],
                         "api_components": receipt["api_components"], "recipe": receipt["recipe"],
                         "source_identity": source_key, "runtime": receipt["runtime"],
                         "protocol": receipt["protocol"], "completed_updates": receipt["completed_updates"],
                         "metric_passed": receipt["metric_passed"],
                         "sustained_metric_passed": receipt["sustained_metric_passed"],
                         "default_protocol_complete": receipt["default_protocol_complete"],
                         "verdict": receipt["verdict"], "failed_bounds": receipt["failed_bounds"],
                         "final_metrics": final["metrics"], "gif": f"media/{name}.gif",
                         "gif_sha256": receipt["artifacts"]["goal.gif"]["sha256"],
                         "frames": receipt["gif_frames"], "seed": receipt["seed"],
                         "raw_receipt": str(receipt_path), "raw_receipt_sha256": api_run.file_hash(receipt_path),
                         "raw_artifacts": receipt["artifacts"]})
    rows.sort(key=lambda row: row["id"])
    ledger = api_run.inventory(cases)
    ledger["published_actual_api_runs"] = len(rows)
    ledger["missing_api_media"] = sorted(set(cases) - seen)
    api_run.write_json(output / "cases.json", ledger)
    api_run.write_json(output / "runs.json", {
        "schema": "particlegan_api_toy_media_v1", "cases": rows,
        "source_identities": manifests, "training_or_rescoring_by_publication": False,
        "historical_receipts_changed": False,
        "api_execution_complete": len(rows),
        "default_protocol_complete": sum(row["default_protocol_complete"] for row in rows),
        "default_test_passes": sum(row["verdict"] == "PASS" for row in rows),
        "instantaneous_metric_passes": sum(row["metric_passed"] for row in rows)})
    lines = ["# API toy goal gallery", "",
             "These GIFs compare the declared target/behavior with actual public-API outputs.",
             "Each frame is a retained observation; the update, instantaneous metric and budget",
             "are visible. Short runs demonstrate executable tests and goal views; their default",
             "training-budget verdict remains FAIL. Historical trained evidence is unchanged.", "",
             f"Registered variants: **{len(cases)}**. Verified actual-state GIFs: **{len(rows)}**.",
             f"Original questions without a runnable API variant: **{len(ledger['coverage']['missing'])}**.", "",
             "[Executable definitions and exact gates](cases.json) · [Compact run receipts](runs.json)", "",
             "| API variant / goal GIF | Question | Executed / default updates | Final instantaneous metric | Default-budget test |",
             "|---|---|---:|---|---|"]
    for row in rows:
        goal = row["goal"].replace("|", "\\|").replace("\n", " ")
        instant = "PASS" if row["metric_passed"] else "FAIL"
        lines.append(f"| [{row['id']}]({row['gif']}) | {goal} | {row['completed_updates']} / "
                     f"{cases[row['id']]['default_steps']} | {instant} | {row['verdict']} |")
    (output / "GALLERY.md").write_text("\n".join(lines) + "\n")
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ledger = publish(args.runs, args.output)
    print(json.dumps({"variants": ledger["coverage"]["api_variants"],
                      "missing_questions": ledger["coverage"]["missing"],
                      "missing_media": ledger["missing_api_media"]}))
    return 0 if not ledger["coverage"]["missing"] and not ledger["missing_api_media"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
