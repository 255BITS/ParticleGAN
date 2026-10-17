"""Publish verified goal GIFs and compact receipts from API toy runs.

This reads existing actual observations; it launches no training or rescoring.
Checkpoints, numeric observation arrays and progress logs remain in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
        if record["passed"]:
            for name, value in record["metrics"].items():
                try:
                    finite_numeric = isinstance(value, (int, float, bool)) and math.isfinite(value)
                except OverflowError:
                    finite_numeric = False
                if not finite_numeric:
                    raise ValueError(f"passing observation metric {name} must be finite numeric")
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
    passing_sample_keys = {f"step{observation['step']}_view{index}_samples"
                           for observation in observations if observation["passed"]
                           for index in range(len(observation["views"]))}
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("numeric media observations differ from receipt")
        for name in arrays.files:
            values = arrays[name]
            if not values.size or not np.issubdtype(values.dtype, np.number):
                raise ValueError("missing actual numeric goal observations")
            if name.endswith("_target") and not np.isfinite(values).all():
                raise ValueError("nonfinite reference goal")
            if name in passing_sample_keys and not np.isfinite(values).all():
                raise ValueError("passing observation contains nonfinite actual samples")
    return receipt


def _verify_renderer_annotations(review, receipt, media_steps, display):
    renderer = review.get("renderer_source")
    files = renderer.get("files_sha256") if isinstance(renderer, dict) else None
    required_files = {"benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_reframe.py",
                      "benchmarks/toy_audit/api_contract.py"}
    if (not isinstance(renderer, dict) or not isinstance(renderer.get("commit"), str)
            or not renderer["commit"] or not isinstance(files, dict) or set(files) != required_files
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in files.values())):
        raise ValueError("media review must identify its separate renderer source")
    annotations = review.get("annotations")
    if (not isinstance(annotations, dict)
            or annotations.get("default_verdict_displayed") != display
            or annotations.get("numeric_observations_changed") is not False
            or not isinstance(annotations.get("goal_annotations"), list)):
        raise ValueError("media review annotations change the displayed grade or observations")
    observations = {record["step"]: record for record in receipt["observations"]}
    for annotation in annotations["goal_annotations"]:
        if (not isinstance(annotation, dict) or type(annotation.get("step")) is not int
                or annotation["step"] not in media_steps or type(annotation.get("view")) is not int
                or not 0 <= annotation["view"] < len(observations[annotation["step"]]["views"])):
            raise ValueError("media review annotation references an uncaptured view")


def _verify_media_review(path, receipt, review_path):
    """Bind a separate render to verified raw evidence; never replace its grade."""
    path, review_path = Path(path), Path(review_path)
    review = read(review_path / "media-review.json")
    if review.get("schema") != "particlegan_api_toy_media_review_v1":
        raise ValueError("unsupported media review schema")
    raw_receipt = path / "receipt.json"
    declared_path = review.get("raw_receipt")
    if (not isinstance(declared_path, str) or not Path(declared_path).is_absolute()
            or Path(declared_path).resolve() != raw_receipt.resolve()):
        raise ValueError("media review references a different raw receipt")
    if review.get("raw_receipt_sha256") != api_run.file_hash(raw_receipt):
        raise ValueError("media review raw receipt identity mismatch")
    for key, expected in (("case_id", receipt["case"]["id"]),
                          ("raw_artifacts", receipt["artifacts"]),
                          ("training_source_commit", receipt["source"]["commit"]),
                          ("verdict", receipt["verdict"]),
                          ("failed_bounds", receipt["failed_bounds"]),
                          ("final_metrics", receipt["observations"][-1]["metrics"])):
        if review.get(key) != expected:
            raise ValueError(f"media review {key} differs from raw evidence")
    for key in ("metric_passed", "sustained_metric_passed", "default_protocol_complete"):
        if type(review.get(key)) is not bool or review[key] != receipt[key]:
            raise ValueError(f"media review {key} differs from raw evidence")
    for key, expected in (("training_or_rescoring", False), ("raw_files_unchanged", True)):
        if review.get(key) is not expected:
            raise ValueError(f"media review {key} must be {expected}")
    media_steps = review.get("media_steps")
    if (not isinstance(media_steps, list) or any(type(step) is not int for step in media_steps)
            or media_steps != receipt["protocol"]["media_steps"]):
        raise ValueError("media review changes the original media steps")
    _verify_renderer_annotations(review, receipt, media_steps, receipt["verdict"])
    gif = review.get("reviewed_gif")
    if not isinstance(gif, dict) or gif.get("file") != "reviewed-goal.gif":
        raise ValueError("media review must name its separate goal GIF")
    frames = _integer(gif.get("frames"), "reviewed GIF frames", minimum=2)
    if frames != len(media_steps) or frames != receipt["gif_frames"]:
        raise ValueError("reviewed GIF count differs from original media steps")
    expected_bytes = _integer(gif.get("bytes"), "reviewed GIF bytes")
    artifact = review_path / "reviewed-goal.gif"
    if artifact.stat().st_size != expected_bytes or api_run.file_hash(artifact) != gif.get("sha256"):
        raise ValueError("reviewed GIF identity mismatch")
    with Image.open(artifact) as decoded:
        if decoded.n_frames != frames:
            raise ValueError("reviewed GIF lost actual observed states")
    return review


def verify_media_review(path, review_path):
    """Verify original completion and a supplemental render without rescoring."""
    return _verify_media_review(path, verify_run(path), review_path)


def _verify_failure_media_review(path, review_path):
    # api_reframe uses the strict normal verifier; defer this import to avoid
    # a module-initialization cycle while retaining its independent source gate.
    from . import api_reframe
    path, review_path = Path(path), Path(review_path)
    receipt, available, steps, display = api_reframe.verify_failure_view(path)
    review = read(review_path / "failure-review.json")
    if review.get("schema") != "particlegan_api_toy_failure_media_review_v1":
        raise ValueError("unsupported failure media review schema")
    raw_path = review.get("raw_receipt")
    if (not isinstance(raw_path, str) or not Path(raw_path).is_absolute()
            or Path(raw_path).resolve() != (path / "receipt.json").resolve()):
        raise ValueError("failure media review references a different raw receipt")
    artifacts = {name: identity for name, identity in available.items() if name != "receipt.json"}
    expected = {"case_id": receipt["case"]["id"],
                "raw_receipt_sha256": available["receipt.json"]["sha256"],
                "raw_artifacts": receipt["artifacts"], "available_raw_artifacts": artifacts,
                "artifact_attestations": {name: "original_receipt" if name in receipt["artifacts"] else "review_time_only"
                                          for name in artifacts},
                "training_source": receipt["source"], "recipe": receipt["recipe"],
                "original_execution_status": "ERROR", "verdict": "FAIL",
                "original_failed_bounds": receipt["failed_bounds"],
                "original_numeric_flags": {key: receipt.get(key) for key in
                                           ("metric_passed", "sustained_metric_passed", "default_protocol_complete")},
                "completed_updates": receipt["completed_updates"], "planned_protocol": receipt["protocol"],
                "original_media_steps": receipt["protocol"]["media_steps"], "media_steps": steps,
                "terminal_frame_added": steps[-1] not in receipt["protocol"]["media_steps"],
                "final_metrics": receipt["observations"][-1]["metrics"],
                "observations": receipt["observations"], "displayed_verdict": display,
                "qualification_upgrade": False, "training_or_rescoring": False, "raw_files_unchanged": True}
    # Canonical JSON equality binds types too: True must not become 1 and an
    # original missing numeric flag must remain null rather than an invented PASS.
    for key, value in expected.items():
        if (key not in review or json.dumps(review[key], sort_keys=True, allow_nan=False)
                != json.dumps(value, sort_keys=True, allow_nan=False)):
            raise ValueError(f"failure media review {key} differs from retained evidence")
    _verify_renderer_annotations(review, receipt, steps, display)
    gif = review.get("reviewed_gif")
    if not isinstance(gif, dict) or gif.get("file") != "failure-goal.gif":
        raise ValueError("failure review must name its separate goal GIF")
    if _integer(gif.get("frames"), "failure GIF frames", minimum=2) != len(steps):
        raise ValueError("failure GIF count differs from captured media steps")
    artifact = review_path / "failure-goal.gif"
    if (artifact.stat().st_size != _integer(gif.get("bytes"), "failure GIF bytes")
            or api_run.file_hash(artifact) != gif.get("sha256")):
        raise ValueError("failure GIF identity mismatch")
    with Image.open(artifact) as decoded:
        if decoded.n_frames != len(steps):
            raise ValueError("failure GIF lost actual captured states")
    return receipt, review


def verify_failure_media_review(path, review_path):
    """Verify one of five frozen ERROR supplements without granting completion."""
    return _verify_failure_media_review(path, review_path)[1]


def _copy_media(source, target, identity):
    if target.exists() and api_run.file_hash(target) != identity["sha256"]:
        raise ValueError(f"refusing to replace earlier media for {target.stem}")
    if not target.exists():
        shutil.copyfile(source, target)


def publish(archives, output, *, media_review=None, failure_media_review=None):
    output = Path(output)
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    cases = api_contract.discover()
    rows, seen, manifests, renderer_manifests = [], set(), {}, {}
    failed_rows, failed_manifests, failed_renderers = [], {}, {}
    for archive in archives:
        archive = Path(archive)
        for summary in read(archive / "summary.json")["cases"]:
            name = summary["id"]
            if name in seen:
                raise ValueError(f"ambiguous repeated case receipt: {name}")
            seen.add(name)
            receipt_path = archive / name / "receipt.json"
            raw = read(receipt_path)
            if raw.get("status") == "ERROR" and failure_media_review is not None:
                failure_path = Path(failure_media_review) / name
                receipt, review = _verify_failure_media_review(receipt_path.parent, failure_path)
                definition = api_run.json_value(cases[name])
                if receipt["case"] != definition:
                    raise ValueError(f"{name}: failed definition differs from registered variant")
                source_key = hashlib.sha256(json.dumps(receipt["source"], sort_keys=True).encode()).hexdigest()
                renderer_key = hashlib.sha256(json.dumps(review["renderer_source"], sort_keys=True).encode()).hexdigest()
                failed_manifests[source_key] = receipt["source"]
                failed_renderers[renderer_key] = review["renderer_source"]
                target = media / f"{name}-failure.gif"
                _copy_media(failure_path / "failure-goal.gif", target, review["reviewed_gif"])
                failed_rows.append({"id": name, "legacy_ids": definition["legacy_ids"],
                    "goal": definition["goal"], "scope": definition["scope"],
                    "sampling": definition["sampling"], "thresholds": definition["thresholds"],
                    "api_components": receipt["api_components"], "recipe": receipt["recipe"],
                    "source_identity": source_key, "runtime": receipt["runtime"], "seed": receipt["seed"],
                    "original_execution_status": "ERROR", "verdict": "FAIL", "passed": False,
                    "qualification_upgrade": False, "qualified_api_execution_complete": False,
                    "qualified_default_protocol_complete": False, "qualified_default_pass": False,
                    "original_failed_bounds": receipt["failed_bounds"],
                    "original_numeric_flags": review["original_numeric_flags"],
                    "completed_updates": receipt["completed_updates"], "planned_protocol": receipt["protocol"],
                    "observation_count": len(receipt["observations"]),
                    "final_observed_metric_passed": receipt["observations"][-1]["passed"],
                    "final_metrics": review["final_metrics"], "displayed_verdict": review["displayed_verdict"],
                    "original_media_steps": review["original_media_steps"], "media_steps": review["media_steps"],
                    "terminal_frame_added": review["terminal_frame_added"],
                    "gif": f"media/{target.name}", "gif_sha256": review["reviewed_gif"]["sha256"],
                    "frames": review["reviewed_gif"]["frames"],
                    "raw_receipt": str(receipt_path), "raw_receipt_sha256": review["raw_receipt_sha256"],
                    "raw_artifacts": review["raw_artifacts"], "available_raw_artifacts": review["available_raw_artifacts"],
                    "artifact_attestations": review["artifact_attestations"],
                    "media_review": {"raw_sidecar": str(failure_path / "failure-review.json"),
                        "raw_sidecar_sha256": api_run.file_hash(failure_path / "failure-review.json"),
                        "renderer_source_identity": renderer_key, "annotations": review["annotations"]}})
                continue
            receipt = verify_run(receipt_path.parent)
            review_path = Path(media_review) / name if media_review is not None else None
            review = (_verify_media_review(receipt_path.parent, receipt, review_path)
                      if review_path is not None else None)
            definition = api_run.json_value(cases[name])
            if receipt["case"] != definition:
                raise ValueError(f"{name}: executed definition differs from registered variant")
            source = receipt["source"]
            source_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            # Different imported dependency sets remain separate source receipts.
            manifests[source_key] = source
            gif_source = receipt_path.parent / "goal.gif"
            gif_identity = receipt["artifacts"]["goal.gif"]
            media_provenance = None
            if review is not None:
                gif_source = review_path / "reviewed-goal.gif"
                gif_identity = review["reviewed_gif"]
                renderer_key = hashlib.sha256(json.dumps(review["renderer_source"], sort_keys=True).encode()).hexdigest()
                renderer_manifests[renderer_key] = review["renderer_source"]
                media_provenance = {"raw_sidecar": str(review_path / "media-review.json"),
                                    "raw_sidecar_sha256": api_run.file_hash(review_path / "media-review.json"),
                                    "renderer_source_identity": renderer_key,
                                    "training_source_commit": review["training_source_commit"],
                                    "annotations": review["annotations"]}
            target = media / f"{name}.gif"
            _copy_media(gif_source, target, gif_identity)
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
                         "gif_sha256": gif_identity["sha256"],
                         "frames": receipt["gif_frames"], "seed": receipt["seed"],
                         "raw_receipt": str(receipt_path), "raw_receipt_sha256": api_run.file_hash(receipt_path),
                         "raw_artifacts": receipt["artifacts"]})
            if media_provenance is not None:
                rows[-1]["media_review"] = media_provenance
    rows.sort(key=lambda row: row["id"])
    failed_rows.sort(key=lambda row: row["id"])
    ledger = api_run.inventory(cases)
    ledger["published_actual_api_runs"] = len(rows)
    ledger["published_failed_api_attempts"] = len(failed_rows)
    ledger["published_actual_goal_views"] = len(rows) + len(failed_rows)
    ledger["api_execution_complete"] = len(rows)
    ledger["default_protocol_complete"] = sum(row["default_protocol_complete"] for row in rows)
    ledger["default_test_passes"] = sum(row["verdict"] == "PASS" for row in rows)
    ledger["missing_api_media"] = sorted(set(cases) - seen)
    api_run.write_json(output / "cases.json", ledger)
    api_run.write_json(output / "runs.json", {
        "schema": "particlegan_api_toy_media_v1", "cases": rows,
        "source_identities": manifests, "training_or_rescoring_by_publication": False,
        "media_review_used": media_review is not None,
        "media_renderer_source_identities": renderer_manifests,
        "historical_receipts_changed": False,
        "api_execution_complete": len(rows),
        "default_protocol_complete": sum(row["default_protocol_complete"] for row in rows),
        "default_test_passes": sum(row["verdict"] == "PASS" for row in rows),
        "instantaneous_metric_passes": sum(row["metric_passed"] for row in rows)})
    if failure_media_review is not None:
        api_run.write_json(output / "failed-runs.json", {
            "schema": "particlegan_api_toy_failure_media_v1", "cases": failed_rows,
            "source_identities": failed_manifests, "media_renderer_source_identities": failed_renderers,
            "training_or_rescoring_by_publication": False, "historical_receipts_changed": False,
            "api_execution_complete": 0, "default_protocol_complete": 0, "default_test_passes": 0,
            "qualification_upgrades": 0, "actual_goal_views": len(failed_rows)})
    lines = ["# API toy goal gallery", "",
             "These GIFs compare the declared target/behavior with actual public-API outputs.",
             "Each frame is a retained observation; the update, instantaneous metric and budget",
             "are visible. Short runs demonstrate executable tests and goal views; their default",
             "training-budget verdict remains FAIL. Historical trained evidence is unchanged.", "",
             f"Registered variants: **{len(cases)}**. Verified actual-state GIFs: **{len(rows) + len(failed_rows)}**.",
             f"Completed executions: **{len(rows)}**; failed-attempt views: **{len(failed_rows)}**. "
             f"Completed default protocols: **{ledger['default_protocol_complete']}**; "
             f"default passes: **{ledger['default_test_passes']}**.",
             f"Original questions without a runnable API variant: **{len(ledger['coverage']['missing'])}**.", "",
             "[Executable definitions and exact gates](cases.json) · [Compact run receipts](runs.json)", "",
             "| API variant / goal GIF | Question | Executed / default updates | Final instantaneous metric | Default-budget test |",
             "|---|---|---:|---|---|"]
    if media_review is not None:
        lines[8:8] = ["The displayed GIFs use a separate reviewed renderer. Original receipts,",
                      "numeric observations, training sources and test verdicts remain unchanged.", ""]
    for row in rows:
        goal = row["goal"].replace("|", "\\|").replace("\n", " ")
        instant = "PASS" if row["metric_passed"] else "FAIL"
        lines.append(f"| [{row['id']}]({row['gif']}) | {goal} | {row['completed_updates']} / "
                     f"{cases[row['id']]['default_steps']} | {instant} | {row['verdict']} |")
    if failed_rows:
        lines.extend(["", "These [failed-attempt views](failed-runs.json) retain the original ERROR/FAIL.",
                      "They illustrate captured training states and grant no execution completion or default pass.",
                      "A numeric export-error PASS remains separate from qualification; failed prerequisites leave continuation unattempted.", "",
                      "| Failed API variant / goal GIF | Question | Captured / planned updates | Original result |",
                      "|---|---|---:|---|"])
        for row in failed_rows:
            goal = row["goal"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| [{row['id']}]({row['gif']}) | {goal} | {row['completed_updates']} / "
                         f"{row['planned_protocol']['updates']} | ERROR / {row['displayed_verdict']} |")
    (output / "GALLERY.md").write_text("\n".join(lines) + "\n")
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-review", type=Path,
                        help="separate reviewed-render archive; original execution and grades remain required")
    parser.add_argument("--failure-media-review", type=Path,
                        help="explicit unqualified views for the five named frozen ERROR/FAIL attempts")
    args = parser.parse_args(argv)
    options = {"media_review": args.media_review}
    if args.failure_media_review is not None:
        options["failure_media_review"] = args.failure_media_review
    ledger = publish(args.runs, args.output, **options)
    print(json.dumps({"variants": ledger["coverage"]["api_variants"],
                      "missing_questions": ledger["coverage"]["missing"],
                      "missing_media": ledger["missing_api_media"]}))
    return 0 if not ledger["coverage"]["missing"] and not ledger["missing_api_media"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
