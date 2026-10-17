"""Publish certified ordinary outcomes and saved training outputs; never train."""
from collections import Counter
from dataclasses import asdict
import argparse
import json
import math
import operator
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import file_hash, stable_hash
from reports.forge.regenerate_technique_inventory import _evaluator_summary

METRIC_RENDERER = ROOT / "reports/forge/family-wide-word-repairs/render.py"
OPS = {"==": operator.eq, ">=": operator.ge, "<=": operator.le,
       ">": operator.gt, "<": operator.lt}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": file_hash(path)}


def binding(row):
    return row if "recipe" in row else row["applied"]


def evaluator_projection(evaluator):
    # Final metric-bound rows are compact; per-check trajectories are archived.
    return (_evaluator_summary(evaluator) if any(isinstance(value, list) and key != "metrics"
            for key, value in evaluator.items()) else evaluator)


def render_outputs(local, request, task, row, output):
    """Render retained scored tensors with the established API GIF renderer."""
    import torch
    from PIL import Image
    from benchmarks.toy_audit.api_run import render_gif
    from benchmarks.toy_audit.api_reframe import renderer_source

    descriptor = row["evidence"]["saved_observer_outputs"]
    path = local / descriptor["path"]
    assert path.resolve().is_relative_to(local.resolve())
    assert path.stat().st_size == descriptor["bytes"] and file_hash(path) == descriptor["sha256"]
    retained = torch.load(path, map_location="cpu", weights_only=True)
    observations = row["evidence"]["observations"]
    assert len(retained) == len(observations) == descriptor["observation_count"] == 24
    assert [record["step"] for record in retained] == [point["step"] for point in observations]
    assert descriptor["optimizer_updates_added"] == descriptor["sampling_draws_added"] == 0
    records = []
    for record, point in zip(retained, observations):
        metrics = {key: value for key, value in point.items() if key != "step"}
        if task["adapter"] == "word_joint":
            normalized = dict(record["metrics"])
            normalized["reconstruction_exact"] = int(normalized["reconstruction_exact"])
            assert stable_hash(normalized) == stable_hash(metrics)
            views = record["views"]
        else:
            assert task["id"] == "ring16_acquisition" and record["samples"].shape == (4096, 2)
            metrics = {name: point[name] for name, _, _ in task["evaluation"]["thresholds"]}
            # Declared centers are a deterministic reference, not sampled data.
            views = [{"kind": "scatter", "title": "Ring centers and actual scored samples",
                "target": torch.tensor(task["execution"]["host_definition"]["means"]),
                "samples": record["samples"],
                "caption": "Gray: 16 declared target centers. Red: all 4096 clean/live samples already scored at this update. Numerical bounds determine quality and covariance."}]
        assert all(torch.isfinite(torch.as_tensor(view[role])).all()
                   for view in views for role in ("target", "samples"))
        failures = [f"{name} {op} {bound}" for name, op, bound in task["evaluation"]["thresholds"]
                    if not OPS[op](metrics[name], bound)]
        records.append({"step": point["step"], "metrics": metrics,
                        "passed": not failures, "failed_bounds": failures, "views": views})
    count = min(9, len(records))
    indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    selected = [records[index] for index in indices]
    case = {"id": task["id"], "goal": task["description"],
            "default_steps": task["execution"]["steps"], "sampling": row["evidence"]["sampling_law"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations = render_gif(case, selected, output, full_budget=True,
        requested_steps=task["execution"]["steps"], final_verdict=row["gate_status"])
    with Image.open(output) as gif:
        assert gif.n_frames == len(indices)
    write(output.with_suffix(".json"), {"schema_version": 1,
        "kind": "actual_training_saved_observer_outputs_gif", "candidate": request["candidate"]["id"],
        "candidate_revision": request["candidate_revision"], "task": task["id"],
        "thresholds": task["evaluation"]["thresholds"], "observation_count": len(observations),
        "selected_observation_indices": indices, "updates": [records[i]["step"] for i in indices],
        "observations_sha256": stable_hash(observations), "retained_outputs": identity(path),
        "saved_observer_outputs": descriptor, "gif": identity(output), "renderer": renderer_source(),
        "publication_source": identity(Path(__file__)), "annotations": annotations,
        "raw_result": identity(local / "raw-result.json"), "resolved_request": identity(local / "request.json"),
        "optimizer_updates": 0, "sampling_draws": 0,
        "scope": "Already scored post-update outputs only; nine displayed frames retain all 24 numerical checks. Ring reference uses declared centers, with no invented reference samples. Publication does not regrade or qualify a candidate."})


def certified_row(task_row, trial, plan):
    directory = ROOT / "reports/forge/attempts" / task_row["attempt_id"]
    envelope, result, certificate = [read(directory / name) for name in
                                     ("request.json", "result.json", "evidence.json")]
    request, job = envelope["request"], envelope["job"]
    assert certificate["result_hash"] == stable_hash(result)
    assert certificate["source"] == request["source"] and certificate["runtime"] == request["runtime"]
    assert result["attempt_id"] == directory.name
    assert request["candidate"]["id"] == trial["candidate_id"]
    assert result["candidate_revision"] == request["candidate_revision"] == trial["candidate_revision"]
    assert request["source"]["digest"] == trial["source_digest"] == plan["source_digest"]
    assert stable_hash(request["source"]["files"]) == request["source"]["digest"]
    # Verify the report uses the same executed code before resolving metadata.
    for relative, expected in request["source"]["files"].items():
        assert file_hash(ROOT / relative) == expected, relative
    task = request["tasks"][task_row["task"]]
    row = next(value for value in result["task_results"] if value["task_id"] == task["id"])
    assert row["compatibility_key"] == task_row["compatibility_key"] == job["compatibility_key"]
    assert row["gate_status"] == task_row["gate_status"] and row["raw_status"] == "completed"
    assert row["evaluator_result"]["convergence"]["observations"] == task["evaluation"]["observations"] == 24
    observations = row["evidence"]["observations"]
    assert len(observations) == 24
    expected_steps = [math.ceil(index * task["execution"]["steps"] / 24) for index in range(1, 25)]
    assert [point["step"] for point in observations] == expected_steps
    assert row["evidence"]["guards"]["all_finite"]
    assert row["evidence"]["guards"]["unintended_rng_deviations"] == 0
    context = task_formulation_context(request["candidate"], task, request["protocol"], device="cpu", root=ROOT)
    effective = binding(row)
    expected = asdict(context.recipe) if "applied" in row else context.recipe.to_dict()
    assert stable_hash(effective["recipe"]) == stable_hash(expected)
    assert effective["prior"] == context.prior_config and effective["initializer"] == context.initializer
    assert effective["rng"]["seed"] == request["protocol"]["seed"] == plan["seed"]
    local = Path(certificate["local_artifact_root"]).resolve()
    assert local.is_relative_to(Path(request["queue_root"]).resolve())
    assert Path(request["queue_root"]).resolve().is_relative_to(ROOT / "runs/forge")
    raw, grading = read(local / "raw-result.json"), read(local / "graded-result.json")
    assert grading["raw_hash"] == stable_hash(raw) and grading["source_digest"] == request["source"]["digest"]
    assert result["raw"]["grading"] == grading
    assert (local / "request.json").read_bytes() == (directory / "request.json").read_bytes()
    assert (local / "result.json").read_bytes() == (directory / "result.json").read_bytes()
    return directory, local, request, task, row, certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=REPORT / "plans.json")
    parser.add_argument("--output-subdirectory", default="", help="Additive round directory within this goal report")
    args = parser.parse_args()
    report = (REPORT / args.output_subdirectory).resolve()
    assert report.is_relative_to(REPORT)
    plan = read(args.plans)
    study_path = ROOT / f"reports/forge/configuration-search/{plan['study']}.json"
    study = read(study_path)
    assert study["selection"]["all_trials_terminal"] and study["tuning_through_tier"] == plan["through_tier"] == 1
    assert len(study["trials"]) == plan["configuration_count"]
    prepared = {trial["candidate"]: trial for trial in plan["trials"]}
    candidates, statuses, source_commits = [], Counter(), set()
    measured_wall = 0.
    for trial in study["trials"]:
        assert trial["candidate_revision"] == prepared[trial["candidate_id"]]["candidate_revision"]
        assert trial["settings"] == prepared[trial["candidate_id"]]["settings"]
        tasks = []
        assert len(trial["tasks"]) == 26 and len({task["task"] for task in trial["tasks"]}) == 26
        for task_row in trial["tasks"]:
            status = task_row["gate_status"]
            statuses[status] += 1
            compact = {"task_id": task_row["task"], "tier": task_row["qualification_tier"],
                       "importance": task_row["importance"], "status": status}
            if status == "UNKNOWN":
                compact["reason"] = ("outside declared Tier 1 execution scope" if task_row["qualification_tier"] > 1
                                     else "earlier required prerequisite failed")
                tasks.append(compact)
                continue
            assert status in {"PASS", "FAIL"} and task_row["qualification_tier"] == 1
            directory, local, request, task, row, certificate = certified_row(task_row, trial, plan)
            source_commits.add(request["source"]["origin_commit"])
            media = report / "media" / f"{trial['configuration_id'][:12]}-{task['id']}.gif"
            if task["adapter"] in {"transfer_vector", "word_joint"}:
                render_outputs(local, request, task, row, media)
            else:
                subprocess.run([sys.executable, str(METRIC_RENDERER), str(local),
                                "--output", str(media.relative_to(ROOT))], cwd=ROOT, check=True)
            wall = row["cost"]["wall_seconds"]
            assert math.isfinite(wall) and wall >= 0
            measured_wall += wall
            effective = binding(row)
            compact.update(metrics=row["metrics"], evaluator_result=evaluator_projection(row["evaluator_result"]),
                guards=row["evidence"]["guards"], execution_steps=task["execution"]["steps"],
                thresholds=task["evaluation"]["thresholds"], effective_recipe=effective["recipe"],
                prior=effective["prior"], initializer=effective["initializer"], initialization=effective["initialization"],
                field_ownership=effective["field_ownership"], sampling_law=row["evidence"]["sampling_law"],
                rng_manifest_sha256=stable_hash(effective["rng"]), host=row["evidence"].get("host"),
                training_schedules=row.get("applied", {}).get("training_schedules"),
                optimizer_group_bindings=row.get("applied", {}).get("optimizer_group_bindings"),
                attempt_id=directory.name, request_id=request["request_id"], compatibility_key=row["compatibility_key"],
                source_commit=request["source"]["origin_commit"], charged_wall_seconds=wall,
                durable_certificate={"valid": True, "result_stable_hash": certificate["result_hash"],
                    "artifacts": [identity(directory / name) for name in ("request.json", "result.json", "evidence.json")]},
                media=str(media.with_suffix(".json").relative_to(ROOT)))
            tasks.append(compact)
        declaration = read(ROOT / prepared[trial["candidate_id"]]["declaration"])
        contract = declaration["decision_contract"]
        def contract_result(condition):
            observed = next(task for task in tasks if task["task_id"] == condition["task_id"])
            value = observed.get("metrics", {}).get(condition["metric"])
            return None if value is None else OPS[condition["op"]](value, condition["threshold"])
        prediction_supported = contract_result(contract["prediction"])
        falsifier_triggered = contract_result(contract["falsifier"])
        receipt = {"schema_version": 1, "scope": "ordinary_global_numeric_configuration",
            "candidate_id": trial["candidate_id"], "candidate_revision": trial["candidate_revision"],
            "configuration_id": trial["configuration_id"], "family": "k3p", "settings": trial["settings"],
            "global_recipe_overrides": trial["recipe_overrides"], "source_digest": trial["source_digest"],
            "source_commits": sorted({task["source_commit"] for task in tasks if "source_commit" in task}),
            "runtime_cohort": trial["runtime_cohort"], "tasks": tasks, "qualification": trial["qualification"],
            "prediction_supported": prediction_supported,
            "falsifier_triggered": falsifier_triggered,
            "prior_word_qualification_reuse": False, "default_adoption": False,
            "charged_wall_seconds": trial["cost"]["new_paid_wall_seconds"], "search_report": identity(study_path)}
        receipt_path = report / "receipts" / f"{trial['configuration_id']}.json"
        write(receipt_path, receipt)
        candidates.append({key: receipt[key] for key in ("candidate_id", "candidate_revision", "configuration_id",
            "settings", "qualification", "prediction_supported", "falsifier_triggered", "charged_wall_seconds")} |
            {"receipt": str(receipt_path.relative_to(ROOT))})
    charged = sum(candidate["charged_wall_seconds"] for candidate in candidates)
    assert math.isclose(charged, measured_wall, abs_tol=1e-6) and charged <= plan["campaign_ceiling_seconds"]
    summary = {"schema_version": 1, "study": plan["study"], "scope": plan["scope"],
        "source_commits": sorted(source_commits), "source_digest": plan["source_digest"],
        "configuration_count": len(candidates), "candidates": candidates,
        "measured_task_count": statuses["PASS"] + statuses["FAIL"], "measured_pass": statuses["PASS"],
        "measured_fail": statuses["FAIL"], "unknown_count": statuses["UNKNOWN"], "task_cells": sum(statuses.values()),
        "tier1_qualified_candidate_ids": sorted(candidate["candidate_id"] for candidate in candidates
            if candidate["qualification"]["qualified_tier"] >= 1),
        "charged_wall_seconds": charged, "candidate_ceiling_seconds": plan["candidate_ceiling_seconds"],
        "campaign_ceiling_seconds": plan["campaign_ceiling_seconds"], "through_tier": 1,
        "required_tier1_count": 5, "full_view_required_count": 26,
        "historical_word_qualification_reuse": False, "default_adoption": False,
        "configured_standard_eligibility": "All five required ordinary Tier 1 tasks PASS for one complete global recipe. Higher tiers remain UNKNOWN; this provisional screen never adopts public defaults.",
        "single_current_leaderboard": "reports/forge/technique-inventory.md",
        "research_readout": "reports/forge/k3p-global-tier1-v3/README.md",
        "search_report": identity(study_path), "publication_optimizer_updates": 0, "publication_sampling_draws": 0}
    write(report / "summary.json", summary)
    write(report / "media/receipts.json", {"schema_version": 1, "receipts": [identity(path)
          for path in sorted((report / "media").glob("*.json")) if path.name != "receipts.json"],
          "optimizer_updates": 0, "sampling_draws": 0})
    print(json.dumps({key: summary[key] for key in ("configuration_count", "measured_task_count",
        "measured_pass", "measured_fail", "unknown_count", "charged_wall_seconds")}))


if __name__ == "__main__":
    main()
