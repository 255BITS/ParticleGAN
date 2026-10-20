"""Publish completed ordinary BISPF quality receipts and saved observations only."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt, _scored_outputs, render
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = "bispf-optimized-quintic-v1-quality"
REQUEST = "2387a25eca8bf513ae66f76b"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
DIGEST = "c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213"
READOUT = "readout-10ed0ff3f18514315a760f4a"
ARCHIVED = "readout-c2d279f7d47ac009ee80d771"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pointer(path, root=None):
    return dict(path=path.relative_to(root).as_posix() if root else str(path), sha256=file_hash(path))


def transient_media(task, row, local, output):
    """Use the existing public word renderer on five already scored states."""
    from benchmarks.toy_audit.api_run import render_gif

    records, inputs = _scored_outputs(task, row["evidence"], local)
    selected_steps = [834, 15001, 15835, 16668, 20001]
    selected = [r for r in records if r["step"] in selected_steps]
    require([r["step"] for r in selected] == selected_steps, "Missing saved word transient state")
    require(selected[2]["passed"] is False and all(r["passed"] for i, r in enumerate(selected) if i != 2),
            "Word transient/recovery labels differ from the recorded observations")
    case = dict(id=task["id"], goal=task.get("description", task["id"]),
                default_steps=task["execution"]["steps"], sampling=row["evidence"]["sampling_law"])
    render_gif(case, selected, output, full_budget=True, requested_steps=20001, final_verdict=row["gate_status"])
    receipt = dict(kind="actual_training_saved_observations_gif", task_id=task["id"],
                   recorded_grade=row["gate_status"], selected_steps=selected_steps,
                   selection="earliest confirmed acquisition, pre-failure, recorded failure, recovery, endpoint",
                   source_inputs=inputs, gif_sha256=file_hash(output), optimizer_updates_added=0,
                   sampling_draws_added=0, qualification_input=False,
                   renderer="benchmarks.toy_audit.api_run:render_gif")
    atomic_json(output.with_suffix(".json"), receipt)
    return receipt


def export_media(durable, request, result, local, output):
    cached = []
    for row in result["task_results"]:
        gif = output / (row["task_id"] + ".gif")
        sidecar = gif.with_suffix(".json")
        if not gif.is_file() or not sidecar.is_file():
            break
        receipt = read_json(sidecar)
        require(receipt["task_id"] == row["task_id"] and receipt["recorded_grade"] == row["gate_status"]
                and receipt["gif_sha256"] == file_hash(gif), "Existing saved media changed")
        require(all(file_hash(Path(p)) == h for p, h in receipt["source_inputs"].items()),
                "Saved media inputs changed")
        cached.append(receipt)
    if len(cached) == len(result["task_results"]):
        return cached
    if all(r["evidence"].get("dense") and not r["evidence"].get("observations")
           for r in result["task_results"]):
        # Same display bridge as the public attempt worker: exact saved dense
        # training measurements are the available ring endurance media.
        return [render(request["tasks"][row["task_id"]],
                       {**row, "evidence": {**row["evidence"], "observations": row["evidence"]["dense"]}},
                       local, output / (row["task_id"] + ".gif")) for row in result["task_results"]]
    try:
        return export_attempt(durable, output)
    except (ValueError, TypeError) as error:
        # Some public transfer hosts retain additional initial/display frames.
        # Select the already scored records; preserve the original descriptor.
        import torch
        from benchmarks.toy_audit.api_run import render_gif

        receipts = []
        for row in result["task_results"]:
            evidence = row["evidence"]
            descriptor = evidence["saved_observer_outputs"]
            source = local / descriptor["path"]
            require(file_hash(source) == descriptor["sha256"], "Saved views changed")
            records = torch.load(source, map_location="cpu", weights_only=True)
            require(all("views" in r for r in records), "Unsupported saved-view schedule mismatch")
            by_step = {r["step"]: r for r in records}
            observations = evidence["observations"]
            selected = [by_step[p["step"]] for p in observations]
            require(all(r["metrics"] == {k: v for k, v in p.items() if k != "step"}
                        for r, p in zip(selected, observations)), "Saved view metrics differ")
            task = request["tasks"][row["task_id"]]
            case = dict(id=task["id"], goal=task.get("description", task["id"]),
                        default_steps=task["execution"]["steps"], sampling=evidence["sampling_law"])
            gif = output / (task["id"] + ".gif")
            gif.parent.mkdir(parents=True, exist_ok=True)
            display = [{**r, "metrics": {k: v for k, v in r["metrics"].items()
                                        if type(v) in (int, float, bool)}} for r in selected]
            render_gif(case, display, gif, full_budget=True,
                       requested_steps=task["execution"]["steps"], final_verdict=row["gate_status"])
            receipt = dict(kind="actual_training_saved_observations_gif", task_id=task["id"],
                           recorded_grade=row["gate_status"], observation_count=len(selected),
                           source_inputs={str(source): file_hash(source)}, gif_sha256=file_hash(gif),
                           optimizer_updates_added=0, sampling_draws_added=0, qualification_input=False,
                           observations_sha256=stable_hash(observations),
                           selection="All scored scheduled records from the original saved training views",
                           metric_labels="Scalar display labels only; original saved metrics and views are unchanged",
                           retained_records=len(records),
                           adapter_note="Strict exporter rejected extra initial/display records: " + str(error),
                           renderer="benchmarks.toy_audit.api_run:render_gif")
            atomic_json(gif.with_suffix(".json"), receipt)
            receipts.append(receipt)
        return receipts


def publish(root, queue_root):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    output = root / "reports/forge/bispf-optimized"
    state = read_json(queue_root / "queue/state.json")
    campaign = state["campaigns"][CAMPAIGN]
    jobs = [j for j in state["jobs"].values() if (j.get("cost_owner") or {}).get("campaign") == CAMPAIGN]
    require(len(jobs) == 29 and all(j["status"] == "terminal" for j in jobs), "Quality jobs are incomplete")
    require(all(len(j["attempts"]) == 1 for j in jobs), "Unexpected paid retry")
    require(campaign["reserved_seconds"] == 0 and campaign["spent_seconds"] <= 46620, "Quality budget incomplete")
    request = read_json(queue_root / "queue/requests" / (REQUEST + ".json"))
    require(request["source"]["origin_commit"] == SOURCE and request["source"]["digest"] == DIGEST,
            "Quality source differs")
    require(request["protocol"]["seed"] == 0 and len(request["tasks"]) == 30, "Quality task/seed differs")
    record_path = root / "reports/forge/records" / (READOUT + ".json")
    record = read_json(record_path)
    require(record["lifecycle"] == "concluded" and record["qualification_input"] is False
            and record.get("full_readout_sha256"), "Conclude and compact the quality readout first")
    require(file_hash(Path(record["full_readout_archive"]["path"])) == record["full_readout_sha256"],
            "Original complete readout archive changed")
    archived_path = root / "reports/forge/records" / (ARCHIVED + ".json")
    archived = read_json(archived_path)
    old = {r["task_id"]: r for r in archived["task_results"]}
    positives = {n for n, r in old.items() if r["gate_status"] == "PASS"}
    require(len(old) == 30 and len(positives) == 19, "Incumbent comparison denominator differs")
    scheduling_path = queue_root / "bispf-optimized/quality-scheduling.json"
    scheduling = read_json(scheduling_path)
    require(scheduling["request"] == REQUEST and scheduling["source_commit"] == SOURCE
            and scheduling["task_cancellations"] == scheduling["paid_retries"] == 0,
            "Scheduling provenance differs")
    guard_path = root / "reports/forge/gaussian-smoke-inventory/export_media.py"
    spec = importlib.util.spec_from_file_location("bispf_saved_media_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    summaries, rows, actual, media = [], [], {}, []
    media_root = output / "quality-media"
    with guard.forbid_live_execution():
        for job in sorted(jobs, key=lambda j: j["definition"]["task_id"]):
            attempt = job["attempts"][0]["attempt_id"]
            durable = root / "reports/forge/attempts" / attempt
            compact = project_receipt(root, attempt)
            envelope, result, certificate = [read_json(durable / (n + ".json"))
                                              for n in ("request", "result", "evidence")]
            frozen = envelope.get("request", envelope)
            require(frozen["request_id"] == REQUEST and frozen["source"] == request["source"]
                    and frozen["candidate_revision"] == request["candidate_revision"], "Mixed quality cohort")
            local = Path(certificate["local_artifact_root"])
            exports = export_media(durable, frozen, result, local, media_root)
            summaries.append(compact)
            for row in result["task_results"]:
                name = row["task_id"]
                require(name not in actual and row["gate_status"] in {"PASS", "FAIL"}, "Invalid quality cell")
                actual[name] = row
                projected = next(r for r in compact["task_results"] if r["task_id"] == name)
                gif = media_root / (name + ".gif")
                receipt = next(r for r in exports if r["task_id"] == name)
                require(receipt["optimizer_updates_added"] == receipt["sampling_draws_added"] == 0,
                        "Media added execution")
                item = dict(task=name, attempt=attempt, status=row["gate_status"],
                            path=gif.relative_to(root).as_posix(), sha256=file_hash(gif),
                            receipt=pointer(gif.with_suffix(".json"), root))
                media.append(item)
                rows.append(dict(task=name, attempt=attempt, status=row["gate_status"],
                                 metrics=projected["metrics"], evaluator_summary=projected["evaluator_summary"],
                                 final_metrics_sha256=stable_hash(row["metrics"]), cost=projected["cost"],
                                 sampling=projected["sampling"],
                                 incumbent_status=old[name]["gate_status"],
                                 archived_final_metrics_equal=row["metrics"] == old[name]["metrics"]))
                if name == "five_word_joint_smoke":
                    transient_media(frozen["tasks"][name], row, local,
                                    media_root / "five_word_joint_smoke-transient.gif")
                print(f"published {name}: {row['gate_status']} ({attempt})", flush=True)
    require(set(actual) == set(request["tasks"]) == set(old), "Quality denominator changed")
    regressions = sorted(n for n in positives if actual[n]["gate_status"] != "PASS")
    require(not regressions, "An incumbent positive regressed")
    assignments = request["view"]["assignments"]
    counted = {a["task"] for a in assignments if a["importance"] == "required"}
    require(all(actual[n]["gate_status"] == "PASS" for n in counted), "A counted gate failed")
    attempts = {j["attempts"][0]["attempt_id"] for j in jobs}
    charges = [c for c in state["charges"] if c["attempt_id"] in attempts]
    require(len(charges) == 29 and math.isclose(sum(c["seconds"] for c in charges),
                                             campaign["spent_seconds"], abs_tol=1e-8), "Cost accounting differs")
    smoke, hold = actual["five_word_joint_smoke"], actual["five_word_joint_hold"]
    transient_primary = next(p for p in smoke["evidence"]["observations"] if p["step"] == 15835)
    transient_confirmation = next(p["metrics"] for p in smoke["evidence"]["confirmations"] if p["step"] == 15835)
    ring = actual["ring_hold"]["metrics"]
    document = dict(schema="bispf_optimized_ordinary_quality_readout_v1", candidate=request["candidate"]["id"],
                    candidate_revision=request["candidate_revision"], source_commit=SOURCE, source_digest=DIGEST,
                    request=REQUEST, study=CAMPAIGN, seed=0, qualification_input=False,
                    scope="Display summary of the current independently graded ordinary cohort; original envelopes own qualification",
                    global_recipe_sha256=stable_hash(request["candidate"]["resolved_recipe"]),
                    declared_global_delta={"optimizer_convolution_rule": ["polar", "quintic"]},
                    counts=dict(Counter(r["gate_status"] for r in actual.values())),
                    incumbent_passes_preserved=19, incumbent_pass_regressions=regressions,
                    counted_tasks=len(counted), counted_passes=len(counted), full_task_denominator=30,
                    execution_groups=29, archived_incumbent=pointer(archived_path, root),
                    archived_final_metric_matches=sum(r["archived_final_metrics_equal"] for r in rows),
                    archived_comparison_scope="Historical final metrics only; no archived evidence used to qualify this new cohort",
                    budget=dict(reservation_seconds=46620, spent_seconds=campaign["spent_seconds"],
                                reserved_seconds=0, attempts=29, paid_retries=0, task_cancellations=0),
                    scheduling=scheduling, scheduling_receipt=pointer(scheduling_path),
                    timing_scope="Quality wall times may include GPU sharing and measure cost, not comparative performance",
                    word_smoke=dict(executed_updates=20001, confirmed_checks=23, observations=24,
                                    transient_failed_step=15835, endpoint_passed=True, earliest_confirmed_step=834,
                                    transient_primary=transient_primary, transient_confirmation=transient_confirmation,
                                    exact_archived_final_metrics_match=smoke["metrics"] == old["five_word_joint_smoke"]["metrics"],
                                    gate="Any independently confirmed scheduled acquisition plus full declared execution"),
                    word_hold=dict(executed_updates=4000, checkpoint_selection="earliest_confirmed_passing_state",
                                   prefix_step=834, cumulative_endpoint=4834, confirmed_checks=25, observations=25,
                                   gate="Every confirmation check throughout the exact own-state continuation"),
                    ring=dict(maximum_update_allowance=7500, actual_updates=actual["ring_hold"]["cost"]["completed_steps"],
                              convergence_step=ring["converged_step"], hold_checks=ring["hold_checks"],
                              extension_checks=actual["ring_extension"]["metrics"]["extension_checks"],
                              explanation="The fixed first-convergence stop rule consumes 1782+1200+300=3282 updates; 7500 is a maximum"),
                    concluded_readout=pointer(record_path, root), raw_readout_archive=record["full_readout_archive"],
                    rows=sorted(rows, key=lambda r: r["task"]), attempt_summaries=summaries, media=media,
                    transient_media=pointer(media_root / "five_word_joint_smoke-transient.gif", root),
                    optimizer_updates_added_by_publication=0, sampling_draws_added_by_publication=0,
                    publisher_sha256=file_hash(Path(__file__)))
    atomic_json(output / "quality.json", document)
    print(f"published {len(actual)} task verdicts, {len(media)} task GIFs; {campaign['spent_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.queue_root)
