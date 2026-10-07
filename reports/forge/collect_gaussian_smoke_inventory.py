"""Collect a quiescent Gaussian inventory from certified saved JSON, without grading.

Only compact readout/receipt/recall files are emitted. Original traces and model
states remain in the local archive; no queue, numerical source or board changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.execution_policy import group_blockers
from experiments.forge.sources import verify_snapshot
from reports.forge.regenerate_technique_inventory import project_receipt, _scalars, _evaluator_summary

DEFAULT_ROUND = "gaussian-smoke-inventory-v4"
PUBLICATION = Path("reports/forge/gaussian-smoke-inventory")
TERMINAL = {"blocked", "completed", "concluded"}


def _inactive(state, campaign_id, *, partial_cut=False):
    campaign = state["campaigns"].get(campaign_id)
    if campaign is None or campaign.get("paused") or campaign.get("reserved_seconds") != 0:
        raise ValueError("campaign is active, paused, absent, or still has reserved work")
    entries = [entry for entry in state["submissions"].values()
               if entry["request"].get("campaign_id") == campaign_id]
    allowed = TERMINAL | {"queued", "cancelled"} if partial_cut else TERMINAL
    if not entries or any(entry["status"] not in allowed for entry in entries):
        raise ValueError("campaign still has active or cancelled submissions; refuse finalization")
    if any(job["status"] == "running" for job in state["jobs"].values()
           if (job.get("cost_owner") or {}).get("campaign") == campaign_id
           or any(name in state["submissions"] and state["submissions"][name]["request"].get("campaign_id") == campaign_id
                  for name in job.get("subscribers", []))):
        raise ValueError("campaign still has a running worker")


def _coordinator_active(queue_root):
    path = Path(queue_root) / "coordinator.lock"
    if not path.exists():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def _source(source, *, origin, digest):
    if (source.get("origin_commit") != origin or source.get("digest") != digest
            or not isinstance(source.get("files"), dict) or stable_hash(source["files"]) != digest):
        raise ValueError("receipt/request has a different or invalid frozen source identity")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def saved_state_certificate(row, task, *, allow_missing=False):
    """Validate certified state bytes and declared format; never deserialize them.

    Tensor contents are the frozen producer's contract. This saved-data audit
    verifies the bound file certificate, not a new neural restore or evaluation.
    """
    evidence = row.get("evidence", {})
    tree = evidence.get("artifact_root")
    manifest = evidence.get("artifact_manifest")
    if bool(tree) != bool(manifest):
        raise ValueError("saved evaluator tree has an incomplete artifact certificate")
    if tree:
        verify_artifacts(tree, manifest)
    descriptor = evidence.get("provenance_checkpoint")
    if descriptor is not None:
        if (not isinstance(descriptor, dict) or type(descriptor.get("schema_version")) is not int
                or descriptor["schema_version"] != 1 or descriptor.get("purpose") != "provenance_only"
                or descriptor.get("prerequisite_eligible") is not False
                or type(descriptor.get("optimizer_updates_added")) is not int or descriptor["optimizer_updates_added"] != 0
                or type(descriptor.get("sampling_draws_added")) is not int or descriptor["sampling_draws_added"] != 0
                or type(descriptor.get("completed_steps")) is not int or descriptor["completed_steps"] < 0
                or not _sha256(descriptor.get("state_sha256"))):
            raise ValueError("invalid provenance-only checkpoint contract or added scientific work")
        checkpoint_root, checkpoint_manifest = descriptor.get("artifact_root"), descriptor.get("artifact_manifest")
        if not isinstance(checkpoint_root, str) or not checkpoint_root or not isinstance(checkpoint_manifest, dict):
            raise ValueError("provenance checkpoint lacks its certified artifact tree")
        verify_artifacts(checkpoint_root, checkpoint_manifest)
        files = checkpoint_manifest["files"]
        path = descriptor.get("path")
        if (len(files) != 1 or not isinstance(path, str) or path not in files or files[path]["sha256"] != descriptor.get("sha256")
                or type(descriptor.get("bytes")) is not int or files[path]["size"] != descriptor["bytes"]):
            raise ValueError("provenance checkpoint file identity differs from its strict complete manifest")
        if tree:
            first, second = Path(tree).resolve(), Path(checkpoint_root).resolve()
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("provenance and evaluator certificate trees overlap")
        keys, hashes = descriptor.get("named_stream_keys"), descriptor.get("named_stream_state_sha256")
        rng = row.get("rng", row.get("applied", {}).get("rng", {}))
        bindings = rng.get("bindings", {})
        if (not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys)
                or keys != sorted(set(keys)) or not isinstance(hashes, dict) or set(keys) != set(hashes)
                or not isinstance(bindings, dict) or set(keys) != set(bindings)
                or any(not _sha256(value) for value in hashes.values())):
            raise ValueError("provenance checkpoint does not bind every consumed named stream")
        completed = row.get("cost", {}).get("completed_steps")
        if completed is not None and descriptor["completed_steps"] != completed:
            raise ValueError("provenance checkpoint update count differs from measured completed steps")
        components = task.get("adapter") == "transfer_behavior" and task.get("execution", {}).get("host") != "mode_hold"
        payload_format = ("component_models_role_values_optimizers_named_streams_v1" if components else
                          "joint_word_fixture_policy_named_streams_v1" if task.get("adapter") == "word_joint" else
                          "public_formulation_context_v1")
        return {"status": "certified", "format": "provenance_only_v1", "artifact_root": checkpoint_root,
                "path": path, "sha256": descriptor["sha256"], "bytes": descriptor["bytes"],
                "artifact_manifest_sha256": checkpoint_manifest["sha256"],
                "state_sha256": descriptor["state_sha256"], "completed_steps": descriptor["completed_steps"],
                "declared_payload_format": payload_format, "completion_budget_witness": False,
                "completed_steps_semantics": "minimum_actual_active_role_optimizer_count" if components else "actual_public_completed_updates",
                "named_stream_count": len(keys), "named_stream_bindings_sha256": stable_hash(bindings),
                "named_stream_states_sha256": stable_hash(hashes), "prerequisite_eligible": False,
                "validation": "Strict saved-byte manifest and stream metadata only; no checkpoint loading or neural restore."}
    # These original formats already serialize complete public contexts and RNG.
    if task.get("adapter") == "clockfree_audit" and tree:
        if not {"initial.pt", "comparisons.pt"}.issubset(manifest["files"]):
            raise ValueError("clockfree certificate lacks its full branch-state proof")
        return {"status": "certified", "format": "clockfree_full_context_branches_v1", "artifact_root": tree,
                "artifact_manifest_sha256": manifest["sha256"],
                "state_files": {name: manifest["files"][name] for name in ("initial.pt", "comparisons.pt")},
                "validation": "Strict saved-byte full-context branch manifest; no checkpoint loading or new parity evaluation."}
    if task.get("evaluation", {}).get("kind") in {"gaussian_smoke", "gaussian_stability"} and tree:
        checkpoint = evidence.get("checkpoint", {})
        name = checkpoint.get("path")
        if (name != "state.pt" or name not in manifest["files"]
                or manifest["files"][name]["sha256"] != checkpoint.get("sha256")
                or not _sha256(checkpoint.get("state_sha256"))):
            raise ValueError("Gaussian full-context checkpoint is not bound to its complete evaluator tree")
        return {"status": "certified", "format": "gaussian_full_context_v1", "artifact_root": tree,
                "artifact_manifest_sha256": manifest["sha256"], "path": name,
                "sha256": checkpoint["sha256"], "bytes": manifest["files"][name]["size"],
                "state_sha256": checkpoint["state_sha256"], "completed_steps": evidence.get("completed_steps"),
                "validation": "Strict saved-byte full public context manifest; no checkpoint loading or new restore."}
    if allow_missing:
        return {"status": "uncertified", "reason": "Original source has no receipt-bound full model/prior/optimizer and consumed-stream checkpoint; interrupted source evidence only."}
    raise ValueError("completed task lacks a certified complete model/prior/optimizer and consumed-stream checkpoint")


def validate_attempt(root, resolved, result, *, origin, digest):
    """Validate administrative and saved evaluator certificates, never regrade."""
    request, job, worker = resolved["request"], resolved["job"], resolved["worker"]
    _source(request["source"], origin=origin, digest=digest)
    if (result["candidate_revision"] != request["candidate_revision"]
            or result["attempt_id"] != worker["attempt"] or result.get("retry_of") or resolved.get("retry_of")
            or result["raw"].get("token") != worker["token"]):
        raise ValueError("attempt identity/token mismatch or forbidden scientific retry")
    owner = result.get("cost_owner", {})
    if (resolved.get("request_id") != request["request_id"]
            or owner.get("request") != request["request_id"]
            or owner.get("campaign") != request["campaign_id"]
            or owner.get("revision") != request["candidate_revision"]):
        raise ValueError("attempt changed its exact request or paid cost owner")
    slot = str(worker["device"])
    resources = job["resources"]
    if (slot not in {"0", "1"} or request.get("execution_backend") != "cuda"
            or resources.get("backend") != "cuda" or resources.get("allow_cpu") is not False
            or resources.get("gpus") != 1 or job.get("science", {}).get("seed") != 0
            or request.get("protocol", {}).get("seed") != 0 or request.get("rng", {}).get("seed") != 0
            or job.get("science", {}).get("compute", {}).get("backend") != "cuda"):
        raise ValueError("actual worker or declared execution permits non-CUDA/seed drift")
    raw = result["raw"]
    if raw.get("telemetry", {}).get("interval", {}).get("device") != worker["device"]:
        raise ValueError("supervised actual device disagrees with its worker")
    if not isinstance(raw.get("finished_at"), str) or not raw["finished_at"]:
        raise ValueError("attempt lacks its durable worker completion timestamp")
    members = set(job.get("task_ids", [job["task_id"]]))
    rows = result["task_results"]
    if len(rows) != len(members) or {row["task_id"] for row in rows} != members:
        raise ValueError("attempt changed its complete execution-group task membership")
    grades = raw.get("grading", {})
    if raw["attempt_status"] == "completed":
        if (request.get("requires_independent_grading") is not True
                or grades.get("source_digest") != digest
                or grades.get("raw_hash") != stable_hash(raw.get("result", {}))):
            raise ValueError("completed attempt lacks its exact frozen evaluator certificate")
        for row in rows:
            grade = grades.get("grades", {}).get(row["task_id"])
            if not isinstance(grade, dict) or any(row.get(key) != value for key, value in grade.items()):
                raise ValueError("stored task result differs from its independent evaluator certificate")
    for row in rows:
        if (row.get("compatibility_key") != job["compatibility_key"]
                or str(row.get("cost", {}).get("device")) != slot
                or (row.get("device") is not None and re.fullmatch(r"cuda(?::[0-9]+)?", str(row["device"])) is None)):
            raise ValueError("task result changed its source-bound key or actual CUDA device")
        allowed_paths = {"public_trainer", "public_components"} | ({None} if raw["attempt_status"] != "completed" else set())
        if row.get("execution_path") not in allowed_paths:
            raise ValueError("task result did not execute through the public API")
        member_raw = raw.get("result", {}).get("task_results", {}).get(row["task_id"], raw.get("result", {}))
        if any(row.get(key) != member_raw[key] for key in
               ("evidence", "recipe", "prior", "initializer", "initialization", "rng", "applied", "execution_path", "device")
               if key in member_raw):
            raise ValueError("task's state/recipe/RNG evidence differs from the evaluator-certified producer output")
    # Checkpoint parents retain the same candidate/source and their exact receipt.
    for name, parent in resolved.get("prerequisites", {}).items():
        directory = Path(root) / "reports/forge/attempts" / parent["attempt_id"]
        previous = read_json(directory / "result.json")
        parent_request = read_json(directory / "request.json")["request"]
        _source(parent_request["source"], origin=origin, digest=digest)
        matches = [row for row in previous["task_results"] if row["task_id"] == name]
        if (parent["result_hash"] != stable_hash(previous) or len(matches) != 1
                or parent["result"] != matches[0] or parent["candidate_revision"] != request["candidate_revision"]
                or previous["candidate_revision"] != request["candidate_revision"]
                or parent["compatibility_key"] != matches[0]["compatibility_key"]):
            raise ValueError("checkpoint prerequisite splices another candidate/source or changes its saved state receipt")


def _compact_task(row, attempt_id, request, job, state_certificate):
    task = request["tasks"][row["task_id"]]
    evaluator = row.get("evaluator_result", {})
    applied = row.get("applied", {})
    recipe = row.get("recipe", applied.get("recipe"))
    guards = row.get("evidence", {}).get("guards", {})
    return {"task_id": row["task_id"], "gate_status": row["gate_status"], "raw_status": row["raw_status"],
            "attempt_id": attempt_id, "compatibility_key": job["compatibility_key"],
            "metrics": _scalars(row.get("metrics", {})), "reason": row.get("reason", ""),
            "evaluator_summary": _evaluator_summary(evaluator),
            "failed_bounds": [_scalars(bound) for bound in evaluator.get("metrics", []) if bound.get("status") != "PASS"],
            "cost": _scalars(row.get("cost", {})), "reported_task_device": row.get("device"),
            "physical_cuda_slot": str(row["cost"]["device"]), "execution_path": row.get("execution_path"),
            "initializer": task["execution"].get("initializer"), "prior": deepcopy(task["execution"].get("prior")),
            "observed_initializer": row.get("initializer", applied.get("initializer")),
            "observed_prior": _scalars(row.get("prior", applied.get("prior", {}))),
            "resolved_recipe_sha256": stable_hash(recipe) if recipe is not None else None,
            "declared_steps": task["execution"].get("steps"), "saved_state_certificate": state_certificate,
            "sampling": {key: row.get("evidence", {}).get(key) for key in
                         ("sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")},
            "guard_summary": {key: _scalars(guards[key]) for key in
                              ("all_finite", "hooks_exercised", "optimizer_updates", "unintended_rng_deviations") if key in guards},
            "mechanism_audit_sha256": stable_hash(guards["mechanism_audit"]) if "mechanism_audit" in guards else None,
            "execution_sha256": job["science"]["execution"].get(row["task_id"]),
            "evaluation_sha256": job["science"]["evaluation"].get(row["task_id"])}


def collect_saved(root, queue_root, state, round_definition, preparation, launch,
                  *, origin, digest, roster_size=52, partial_cut=False):
    """Read-only collector. Explicit fixture identities are only used by tests."""
    root, queue_root = Path(root), Path(queue_root)
    campaign_id = round_definition["id"]
    _inactive(state, campaign_id, partial_cut=partial_cut)
    roster = round_definition["candidate_ids"]
    prepared = {row["candidate_id"]: row for row in preparation["candidates"]}
    if (len(roster) != roster_size or len(set(roster)) != roster_size or set(roster) != set(prepared)
            or len(preparation["candidates"]) != roster_size
            or preparation.get("candidate_count") != roster_size
            or preparation.get("round_sha256") != stable_hash(round_definition)
            or preparation.get("round") != campaign_id or preparation.get("stage") != "enqueue"
            or preparation.get("source_digest") != digest or preparation.get("through_tier") != 2
            or round_definition.get("scientific_retries") != 0 or round_definition.get("seed") != 0
            or round_definition.get("required_denominator_by_tier") != [6, 20, 2]
            or preparation.get("execution_backend") != "cuda" or round_definition.get("execution_backend") != "cuda"
            or launch.get("source_commit") != origin or launch.get("source_digest") != digest
            or launch.get("campaign") != campaign_id
            or launch.get("seed") != 0 or launch.get("through_tier") != 2
            or launch.get("workers_per_gpu") != 1 or launch.get("devices") != ["0", "1"]):
        raise ValueError("frozen roster, preparation, campaign launch or protocol drift")
    campaign_entries = [entry for entry in state["submissions"].values()
                        if entry["request"].get("campaign_id") == campaign_id]
    entries = {entry["request"]["candidate"]["id"]: entry for entry in campaign_entries}
    admitted = {name for name, row in prepared.items() if row.get("request_id") is not None}
    if (set(entries) != admitted or len(entries) != len(campaign_entries) or len(entries) != launch["requests"]
            or state["campaigns"][campaign_id]["definition"] != preparation["campaign"]):
        raise ValueError("saved queue admissions differ from the complete prepared roster")
    views = {stable_hash(entry["request"]["view"]) for entry in entries.values()}
    if len(views) != 1:
        raise ValueError("campaign requests changed the frozen view")
    view = next(iter(entries.values()))["request"]["view"]
    required = {str(tier): [a["task"] for a in view["assignments"]
                          if a["importance"] == "required" and a["qualification_tier"] == tier] for tier in (1, 2, 3)}
    if ([len(required[str(tier)]) for tier in (1, 2, 3)] != [6, 20, 2]
            or view["revision"] != round_definition["view_revision"] or view["id"] != round_definition["view"]):
        raise ValueError("required task denominator/view drift")
    attempts, rows, devices, missing_t1, missing_t2, uncertified = {}, [], Counter(), [], [], []
    for name in sorted(roster):
        declaration = prepared[name]
        if declaration["source_digest"] != digest:
            raise ValueError("prepared candidate belongs to another source")
        observed, request, entry = {}, None, entries.get(name)
        if entry is not None:
            request = entry["request"]
            _source(request["source"], origin=origin, digest=digest)
            if (request["candidate_revision"] != declaration["candidate_revision"]
                    or request["request_id"] != declaration["request_id"]
                    or stable_hash({key: value for key, value in request.items() if key != "request_id"})[:24] != request["request_id"]
                    or request.get("policy_fingerprint") != stable_hash(view)
                    or request.get("execution_policy") != {"schema_version": 1, "mode": "complete_current_tier"}):
                raise ValueError("saved request differs from its exact preparation/scientific protocol")
            saved_request = read_json(queue_root / "queue/requests" / (request["request_id"] + ".json"))
            if saved_request != request:
                raise ValueError("queue request file changed")
            for definition in request["jobs"]:
                job = state["jobs"][definition["compatibility_key"]]
                if job["definition"] != definition or len(job.get("attempts", [])) > 1 or job.get("retry_of"):
                    raise ValueError("job definition drift or forbidden scientific retry")
                result = job.get("result")
                if result is None:
                    if job.get("attempts"):
                        raise ValueError("attempt has not reached its durable terminal receipt")
                    continue
                attempt_id = result["attempt_id"]
                if job["status"] != "terminal" or len(job["attempts"]) != 1 or job["attempts"][0]["attempt_id"] != attempt_id:
                    raise ValueError("terminal result is absent from its exact attempt history")
                if attempt_id not in attempts:
                    summary = project_receipt(root, attempt_id)
                    directory = root / "reports/forge/attempts" / attempt_id
                    resolved = read_json(directory / "request.json")
                    durable = read_json(directory / "result.json")
                    if durable != result or resolved["request"] != request or resolved["job"] != definition:
                        raise ValueError("durable originals differ from the queue's complete result/request")
                    validate_attempt(root, resolved, durable, origin=origin, digest=digest)
                    elapsed = durable["raw"]["elapsed_seconds"]
                    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("invalid actual paid wall time")
                    attempts[attempt_id] = {"attempt_id": attempt_id, "candidate_id": name,
                                           "request_id": request["request_id"], "compatibility_key": definition["compatibility_key"],
                                           "physical_cuda_slot": str(resolved["worker"]["device"]),
                                           "wall_seconds": elapsed, "raw_status": durable["raw"]["attempt_status"],
                                           "canonical_result_hash": summary["provenance"]["canonical_result_hash"],
                                           "original_files": summary["provenance"]["original_files"]}
                    devices[str(resolved["worker"]["device"])] += 1
                for task_result in result["task_results"]:
                    if task_result["task_id"] in observed:
                        raise ValueError("candidate pools duplicate task cells")
                    if result["raw"]["attempt_status"] == "completed":
                        certificate = saved_state_certificate(task_result, request["tasks"][task_result["task_id"]],
                                                              allow_missing=partial_cut)
                    else:
                        certificate = {"status": "execution_incomplete", "reason": "No complete trained-state claim follows from an interrupted/error attempt."}
                    if certificate["status"] == "uncertified":
                        uncertified.append({"candidate_id": name, "task_id": task_result["task_id"], "attempt_id": attempt_id,
                                            "reason": certificate["reason"]})
                    observed[task_result["task_id"]] = _compact_task(task_result, attempt_id, request, definition, certificate)
        statuses = {task: cell["gate_status"] for task, cell in observed.items()}
        unlocked = all(statuses.get(task) == "PASS" for task in required["1"])
        tier_summaries = {}
        for tier, tasks in required.items():
            states = Counter(statuses.get(task, "BLOCKED" if request is None else "NOT_RUN") for task in tasks)
            tier_summaries[tier] = {"passed": states["PASS"], "total": len(tasks), "counts": dict(sorted(states.items())),
                                   "observed": sum(task in observed for task in tasks)}
        eligible_by_tier, missing_by_tier = {1: [], 2: []}, {1: [], 2: []}
        if request is not None:
            tier_for = {a["task"]: a["qualification_tier"] for a in view["assignments"]}
            for definition in request["jobs"]:
                members = set(definition.get("task_ids", [definition["task_id"]]))
                for tier in (1, 2):
                    if (not members & set(required[str(tier)]) or any(tier_for[task] != tier for task in members)
                            or tier == 2 and not unlocked or group_blockers(request, definition)):
                        continue
                    dependencies = {d["task"] if isinstance(d, dict) else d for task in members
                                    for d in request["tasks"][task].get("dependencies", [])} - members
                    if any(statuses.get(task) != "PASS" for task in dependencies):
                        continue
                    key = definition["compatibility_key"]
                    eligible_by_tier[tier].append(key)
                    if state["jobs"][key].get("result") is None:
                        missing_by_tier[tier].append({"candidate_id": name, "task_ids": sorted(members), "compatibility_key": key})
        missing_t1.extend(missing_by_tier[1]); missing_t2.extend(missing_by_tier[2])
        rows.append({"candidate_id": name, "candidate_revision": declaration["candidate_revision"],
                     "declaration_kind": declaration["declaration_kind"], "study_id": declaration.get("study_id"),
                     "admission_status": declaration["submission_status"], "submission_blockers": declaration["submission_blockers"],
                     "queue_status": entry["status"] if entry else None, "queue_reason": entry.get("reason") if entry else None,
                     "source_bound_status": "measured_exact_source" if observed else "declared_unexecuted",
                     "request_id": request["request_id"] if request else None, "tiers": tier_summaries,
                     "qualified_tier_from_recorded_cells": 2 if unlocked and tier_summaries["2"]["passed"] == 20 else 1 if unlocked else 0,
                     "tier2_unlocked": unlocked, "eligible_required_job_counts": {str(t): len(eligible_by_tier[t]) for t in (1, 2)},
                     "missing_eligible_jobs": {str(t): missing_by_tier[t] for t in (1, 2)},
                     "attempt_ids": sorted({cell["attempt_id"] for cell in observed.values()}),
                     "tasks": [observed[task] for task in sorted(observed)],
                     "owned_paid_seconds": sum(attempt["wall_seconds"] for attempt in attempts.values() if attempt["candidate_id"] == name)})
    charges = [charge for charge in state["charges"] if charge["owner"]["campaign"] == campaign_id]
    if len(charges) != len(attempts) or {charge["attempt_id"] for charge in charges} != set(attempts):
        raise ValueError("campaign charge ledger differs from its complete certified attempts")
    for charge in charges:
        attempt = attempts[charge["attempt_id"]]
        if (charge["seconds"] != attempt["wall_seconds"] or charge["owner"]["request"] != attempt["request_id"]
                or charge["owner"]["revision"] != prepared[attempt["candidate_id"]]["candidate_revision"]):
            raise ValueError("actual charge changed its cost owner/time")
    paid = sum(attempt["wall_seconds"] for attempt in attempts.values())
    if not math.isclose(paid, state["campaigns"][campaign_id]["spent_seconds"], rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("campaign paid time disagrees with unique original attempts")
    return {"schema_version": 1, "readout_version": "gaussian-smoke-inventory-saved-v1", "round": campaign_id,
            "publication_scope": "closed_partial_cut" if partial_cut else "quiescent_campaign_readout",
            "finalized": not partial_cut,
            "scope": "Whole candidate rows under one exact executed source; saved certificates only, no task pooling or new grading.",
            "qualification_input": False, "qualification_reuse": False, "default_adoption": False,
            "source_commit": origin, "source_digest": digest, "view": view["id"], "view_revision": view["revision"],
            "policy_fingerprint": stable_hash(view), "seed": 0, "through_tier": 2,
            "roster_count": len(rows), "admitted_count": len(entries), "required_tasks_by_tier": required,
            "required_denominator_by_tier": [6, 20, 2], "admission_status_counts": dict(sorted(Counter(row["admission_status"] for row in rows).items())),
            "candidate_rows": rows, "actual_cuda_slots": dict(sorted(devices.items())), "all_actual_attempts_cuda": True,
            "attempt_count": len(attempts), "scientific_retries": 0, "attempt_receipts": [attempts[key] for key in sorted(attempts)],
            "saved_state_audit": {"all_completed_tasks_have_certified_full_state": not uncertified,
                                  "uncertified_completed_cells": uncertified,
                                  "validation_scope": "Receipt-bound strict file/manifest identities and declared state formats only; no torch.load, restore, sampling or grading."},
            "tier2_unlocked_candidates": [row["candidate_id"] for row in rows if row["tier2_unlocked"]],
            "all_runnable_tier1_jobs_ran": not missing_t1, "all_newly_eligible_tier2_jobs_ran": not missing_t2,
            "missing_runnable_tier1_jobs": missing_t1, "missing_newly_eligible_tier2_jobs": missing_t2,
            "cost": {"unique_campaign_paid_seconds": paid,
                     "retained_prior_campaign_cost_claims": {key: value for key, value in launch.items()
                                                             if key.startswith("retained_") and key.endswith("spent_seconds")},
                     "bounded_combined_cost": deepcopy(round_definition.get("bounded_combined_cost")),
                     "reserved_seconds": 0, "campaign_ceiling_seconds": preparation["campaign"]["budget_seconds"]},
            "source_repair_context": deepcopy(round_definition.get("software_predecessor", round_definition.get("repair_of"))),
            "source_reuse": "Earlier failed/refused sources retain their own evidence and cost; no earlier gates enter these rows."}


def collect(root, queue_root, round_path, preparation_path=None, *, campaign_id=None,
            origin=None, digest=None, partial_cut=False, archive_receipt_path=None):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    if _coordinator_active(queue_root):
        raise ValueError("campaign coordinator is active; refuse finalization")
    state_path = queue_root / "queue/state.json"
    state_sha256 = file_hash(state_path)
    state = read_json(state_path)
    round_path = Path(round_path)
    round_path = round_path if round_path.is_absolute() else root / round_path
    round_definition = read_json(round_path)
    round_id = round_definition["id"]
    if campaign_id is not None and campaign_id != round_id:
        raise ValueError("requested campaign differs from the frozen round")
    _inactive(state, round_id, partial_cut=partial_cut)
    launch_path = queue_root / "launch-receipt.json"
    launch = read_json(launch_path)
    origin, digest = origin or launch["source_commit"], digest or launch["source_digest"]
    frozen_round = subprocess.check_output(["git", "show", origin + ":configs/forge/rounds/" + round_id + ".json"], cwd=root)
    if hashlib.sha256(frozen_round).hexdigest() != file_hash(round_path):
        raise ValueError("round definition differs from its actual executed Git commit")
    preparation_path = Path(preparation_path) if preparation_path else queue_root / round_id / "preparation-enqueue.json"
    preparation_path = preparation_path if preparation_path.is_absolute() else root / preparation_path
    result = collect_saved(root, queue_root, state, round_definition, read_json(preparation_path), launch,
                           origin=origin, digest=digest, partial_cut=partial_cut)
    interruption_path = queue_root / "interruption-receipt.json"
    if partial_cut:
        interruption = read_json(interruption_path)
        if (interruption.get("campaign_id") != round_id or interruption.get("source_origin_commit") != origin
                or interruption.get("source_digest") != digest or interruption.get("active_workers") != 0
                or interruption.get("reserved_seconds") != 0 or interruption.get("scientific_retries") != 0
                or interruption.get("old_results_are_new_source_credit") is not False
                or interruption.get("numerical_verdicts_unchanged") is not True
                or interruption.get("paid_seconds") != state["campaigns"][round_id]["spent_seconds"]):
            raise ValueError("partial evidence needs its exact inactive source-bound interruption receipt")
        result["interruption"] = {**_scalars(interruption), "receipt_sha256": file_hash(interruption_path)}
    sources = [entry["request"]["source"] for entry in state["submissions"].values()
               if entry["request"].get("campaign_id") == round_id]
    if sources:
        verify_snapshot(Path(sources[0]["snapshot_path"]), sources[0])
    if _coordinator_active(queue_root) or file_hash(state_path) != state_sha256:
        raise ValueError("queue changed during collection; no final artifacts written")
    result["provenance"] = {"collector_sha256": file_hash(Path(__file__)),
                            "inputs": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in
                                       (("queue_state", state_path), ("round", round_path), ("preparation", preparation_path), ("launch", launch_path))}}
    if partial_cut:
        result["provenance"]["inputs"]["interruption"] = {"path": str(interruption_path), "sha256": file_hash(interruption_path)}
    if archive_receipt_path is not None:
        archive_receipt_path = Path(archive_receipt_path)
        archive_receipt_path = archive_receipt_path if archive_receipt_path.is_absolute() else root / archive_receipt_path
        certificate = read_json(archive_receipt_path)
        archive = root / certificate["archive"]
        if (certificate.get("campaign") != round_id or certificate.get("source_origin_commit") != origin
                or certificate.get("source_digest") != digest or not _sha256(certificate.get("sha256"))
                or not _sha256(certificate.get("members_digest"))
                or type(certificate.get("bytes")) is not int or certificate["bytes"] != archive.stat().st_size
                or certificate["sha256"] != file_hash(archive)
                or set(certificate.get("attempts", [])) != {attempt["attempt_id"] for attempt in result["attempt_receipts"]}
                or len(certificate.get("attempts", [])) != result["attempt_count"]
                or type(certificate.get("original_files")) is not int or certificate["original_files"] < result["attempt_count"] * 3):
            raise ValueError("archive receipt differs from the exact source/attempt cohort or retained archive bytes")
        if partial_cut:
            # The readout adds its own file hash; the original stop receipt does not.
            original = {key: value for key, value in result["interruption"].items() if key != "receipt_sha256"}
            if (certificate.get("new_source_credit") is not False or certificate.get("qualification_input") is not False
                    or certificate.get("interruption") != original):
                raise ValueError("interrupted archive changes its original stop/verdict qualification limits")
        relative = archive_receipt_path.relative_to(root).as_posix()
        result["archive"] = {"receipt_path": relative, "receipt_sha256": file_hash(archive_receipt_path),
                             **{key: certificate[key] for key in ("archive", "sha256", "bytes", "original_files", "members_digest")},
                             "validation": "Archive byte hash/size and exact source/attempt membership checked; file-count/member digest are retained root certificate claims."}
        result["provenance"]["inputs"]["archive_receipt"] = {"path": relative, "sha256": file_hash(archive_receipt_path)}
    if _coordinator_active(queue_root) or file_hash(state_path) != state_sha256:
        raise ValueError("queue changed before readout finalization; no final artifacts written")
    return result


def recall_record(result, readout_sha256, publication_prefix=PUBLICATION):
    unlocked = result["tier2_unlocked_candidates"]
    return {"schema_version": 1, "record_id": result["round"] + "-readout", "record_type": "scientific",
            "candidate_id": result["round"], "candidate_revision": result["provenance"]["inputs"]["round"]["sha256"],
            "goal": "discriminator_stability", "evidence_scope": "source_bound_campaign_readout",
            "lifecycle": "concluded" if result["finalized"] else "closed_partial_cut",
            "qualification_input": False, "qualification_reuse": False,
            "hypothesis": "A confirmed can-it-pass Gaussian Tier 1 gate can unlock ordinary Tier 2 while preserving the separate continuous-stability question.",
            "mechanism_class": "task_gate_reform_and_source_repair",
            "changed_factors": ["Separate Gaussian acquisition smoke from Tier 2 continuous stability.",
                                "New ordinary Gaussian task IDs bind a learned MoG sigma 0.1; archived Gaussian task identities retain their original priors and verdicts.",
                                "Repair the public adapter's execution allowance without changing its original schedule horizon.",
                                "Retain complete public model, prior, optimizer/controller and consumed-stream provenance separately from checkpoint prerequisite eligibility."],
            "conclusion": (f"The exact 52-candidate CUDA roster admits {result['admitted_count']} whole candidates. "
                           f"{len(unlocked)} pass all six required Tier 1 tasks and unlock Tier 2. "
                           f"All newly eligible Tier 2 jobs ran: {result['all_newly_eligible_tier2_jobs_ran']}. "
                           f"{result['attempt_count']} certified attempts cost {result['cost']['unique_campaign_paid_seconds']:.6f} seconds; "
                           "no scientific retries or cross-source task pooling. "
                           + ("This source was interrupted; it retains missing checkpoint limitations and does not establish campaign completion. " if not result["finalized"] else "")
                           + "The linked readout preserves every candidate, refusal and measured scalar metric."),
            "next_action": "Use the single regenerated technique inventory for whole-family selection; retain blocked and failed cells. No calibration, default promotion or new tuning follows from this readout.",
            "source": {"path": str(Path(publication_prefix) / "readout.json"), "sha256": readout_sha256},
            "provenance": {"source_commit": result["source_commit"], "source_digest": result["source_digest"],
                           "attempt_count": result["attempt_count"], "seed": 0, "scientific_retries": 0,
                           "all_completed_tasks_have_certified_full_state": result["saved_state_audit"]["all_completed_tasks_have_certified_full_state"],
                           "readout_collector_sha256": result["provenance"]["collector_sha256"]},
            "task_results": [],
            "task_results_note": "Campaign-level recall only. Candidate-specific metrics stay in the linked whole-row readout with candidate/revision/source identity; no aggregate scientific task cell or synthetic family qualification."}


def markdown(result, publication_prefix=PUBLICATION):
    measured = [row for row in result["candidate_rows"] if row["attempt_ids"]]
    failures = Counter(task["task_id"] for row in measured for task in row["tasks"]
                       if task["gate_status"] != "PASS" and task["task_id"] in result["required_tasks_by_tier"]["1"])
    lines = [f"# Gaussian smoke inventory: exact {result['round']} readout", "",
             f"The frozen roster retains all **52 candidates**, including **{result['admitted_count']} admitted** recipes and every blocked/refused declaration. "
             "Each recipe keeps its own six Tier 1, twenty Tier 2 and two Tier 3 required cells. This report does not select or rank recipes.", "",
             f"**{len(result['tier2_unlocked_candidates'])} whole recipes pass all six Tier 1 gates.** "
             f"All runnable Tier 1 peers ran: **{result['all_runnable_tier1_jobs_ran']}**. "
             f"All newly eligible Tier 2 jobs ran: **{result['all_newly_eligible_tier2_jobs_ran']}**. Tier 3 is outside this campaign's cap.", "",
             f"The **{result['attempt_count']} unique certified attempts** cost **{result['cost']['unique_campaign_paid_seconds']:.3f} seconds**. "
             "All actual workers used CUDA; process-local `cuda:0` can correspond to either physical GPU because the worker limits visible devices. "
             "No scientific retries, new seeds or cross-source gate pooling occurred.", "",
             "Observed required Tier 1 non-passes by task: " + (", ".join(f"{task}: {count}" for task, count in sorted(failures.items())) or "none") + ".", "",
             "Newly eligible recipes: " + (", ".join(f"`{name}`" for name in result["tier2_unlocked_candidates"]) or "none") + ".", "",
             f"Executed source `{result['source_commit']}`, digest `{result['source_digest']}`, protocol seed 0. "
             "Earlier source refusal/error cohorts and their paid cost remain separate; their gates do not fill these rows.", "",
             "All completed task states have receipt-bound complete-state certificates: "
             f"**{result['saved_state_audit']['all_completed_tasks_have_certified_full_state']}**. "
             f"Uncertified completed cells: **{len(result['saved_state_audit']['uncertified_completed_cells'])}**. "
             "This collection checks strict artifact manifests, file hashes and declared state formats from the frozen producer. It does not load models or repeat evaluation.", "",
             "[Every whole candidate, numerical metric, unknown cell count and eligibility audit](readout.json) · "
             "[Compact file receipt](receipt.json). Original request/evidence/result files, stdout, curves and tensors stay in the artifact archive.", "",
             "Use the single regenerated technique inventory for family selection. A Tier 1 smoke pass establishes acquisition under its declared bounds; "
             "continuous stability retains its separate Tier 2 gate. No calibration or default-adoption claim follows.", "",
             "Reproduce the saved-data collection after the campaign coordinator has exited:", "", "```sh",
             "/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \\",
             f"  --queue-root runs/forge/{result['round']} \\",
             f"  --round configs/forge/rounds/{result['round']}.json \\",
             f"  --source-commit {result['source_commit']} \\",
             f"  --source-digest {result['source_digest']} \\",
             f"  --output runs/software/{result['round']}-readout" + (" --allow-interrupted" if not result["finalized"] else ""), "```", ""]
    if not result["finalized"]:
        lines[2:2] = ["This is an explicitly closed partial source cut, not a completed campaign. "
                      "Missing runnable work and checkpoint limitations remain visible; these gates cannot fill another source cohort.", ""]
    if result.get("archive"):
        line = next(i for i, value in enumerate(lines) if value.startswith("  --output "))
        lines[line:line + 1] = ["  --archive-receipt " + result["archive"]["receipt_path"] + " \\", lines[line]]
    if result.get("archive"):
        archive = result["archive"]
        relative = os.path.relpath(archive["receipt_path"], publication_prefix)
        lines.extend([f"The [root archive receipt]({relative}) retains {archive['original_files']} original files in "
                      f"`{archive['archive']}` ({archive['bytes']} bytes, SHA-256 `{archive['sha256']}`). "
                      "The collector checks the archive bytes and exact attempt cohort; the root receipt supplies its member digest.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--round", dest="round_path", type=Path, default=Path("configs/forge/rounds") / (DEFAULT_ROUND + ".json"))
    parser.add_argument("--campaign", help="optional explicit campaign ID; must equal the round ID")
    parser.add_argument("--source-commit", help="optional explicit executed origin; otherwise read the saved launch receipt")
    parser.add_argument("--source-digest", help="optional explicit frozen digest; otherwise read the saved launch receipt")
    parser.add_argument("--allow-interrupted", "--partial-cut", dest="partial_cut", action="store_true",
                        help="archive an explicitly closed interrupted source with its stop receipt; never finalize or qualify it")
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--archive-receipt", type=Path, help="optional root-owned exact source archive receipt; validate and link, never duplicate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-prefix", type=Path, default=PUBLICATION)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    queue_root = args.queue_root if args.queue_root.is_absolute() else root / args.queue_root
    try:
        result = collect(root, queue_root, args.round_path, args.preparation, campaign_id=args.campaign,
                         origin=args.source_commit, digest=args.source_digest, partial_cut=args.partial_cut,
                         archive_receipt_path=args.archive_receipt)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    output = args.output if args.output.is_absolute() else root / args.output
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    readout_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    report = markdown(result, args.publication_prefix)
    record = recall_record(result, readout_sha256, args.publication_prefix)
    atomic_text(output / "readout.json", payload)
    atomic_text(output / "README.md", report)
    record_path = output / "records" / (result["round"] + "-readout.json")
    atomic_json(record_path, record)
    atomic_json(output / "receipt.json", {"schema_version": 1, "qualification_input": False,
                                         "source_commit": result["source_commit"], "source_digest": result["source_digest"],
                                         "readout_sha256": readout_sha256,
                                         "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
                                         "recall_record_sha256": file_hash(record_path),
                                         "collector_sha256": file_hash(Path(__file__)),
                                         "original_attempt_count": result["attempt_count"], "training_launched": False})
    print(json.dumps({"output": str(output.resolve()), "roster_count": result["roster_count"],
                      "attempt_count": result["attempt_count"], "tier2_unlocked": result["tier2_unlocked_candidates"],
                      "all_newly_eligible_tier2_jobs_ran": result["all_newly_eligible_tier2_jobs_ran"],
                      "finalized": result["finalized"],
                      "uncertified_completed_cells": len(result["saved_state_audit"]["uncertified_completed_cells"]),
                      "training_launched": False}, sort_keys=True))


if __name__ == "__main__":
    main()
