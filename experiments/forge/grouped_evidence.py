"""Reuse an already measured uninterrupted ring group, never resume its state.

The isolated attempt worker imports this orchestration module while its frozen
source supplies the grader. Provenance-only checkpoints remain ineligible for
fresh training. Original parent and refused-child receipts are never rewritten.
"""
from copy import deepcopy
from pathlib import Path

from .artifacts import verify_artifacts
from .contracts import file_hash, read_json, stable_hash
from .sources import verify_snapshot
from .views import grade_result, task_execution_fingerprint, task_evaluation_fingerprint


def _identity(job):
    return [job[key] for key in ("source_digest", "candidate_revision", "task",
                                "execution_sha256", "evaluation_sha256")]


def _policy():
    """Scoped-cohort helpers from the frozen source; older sources have none."""
    try:
        from .tier1_policy import is_policy_task, validate
    except ImportError:
        return (lambda task: False), None
    return is_policy_task, validate


def _schedule_free(card):
    # Checked locally: an older frozen source may predate the shared helper.
    return card.get("execution", {}).get("policy_contract", {}).get("schedule") == "preserve_resolved_recipe_none"


def _law(card):
    """Execution/evaluation law without a scoped variant's own parent identity."""
    is_policy_task, validate = _policy()
    execution, evaluation = deepcopy(card["execution"]), deepcopy(card["evaluation"])
    if is_policy_task(card):
        original = validate(card)
        execution.pop("policy_parent_definition")
        evaluation.get("sources", {}).pop("configs/forge/tasks/" + original["id"] + ".json")
    return execution, evaluation


def _check_group(producer, consumer):
    a, b = producer["execution"], consumer["execution"]
    if (producer["adapter"] != consumer["adapter"] or producer["adapter"] != "ring_endurance"
            or producer["evaluation"]["kind"] != "ring_hold"
            or consumer["evaluation"]["kind"] != "ring_extension"
            or not a.get("execution_group") or a.get("execution_group") != b.get("execution_group")
            or a.get("uninterrupted") is not True or b.get("uninterrupted") is not True
            or b.get("continuation_of") != producer["id"]
            or {"task": producer["id"], "kind": "checkpoint"} not in consumer.get("dependencies", [])):
        raise ValueError("not a declared uninterrupted ring execution group")
    (a, producer_evaluation), (execution, evaluation) = _law(producer), _law(consumer)
    execution.pop("continuation_of", None)
    extension_steps = execution.pop("extension_steps", None)
    evaluation = {**evaluation, "kind": "ring_hold"}
    if (execution != a or evaluation != producer_evaluation
            or extension_steps != producer_evaluation["extension_steps"]):
        raise ValueError("grouped consumer changes the producer law, budget or evaluator")


def grade_saved_group(parent, child):
    """Validate both frozen bindings and regrade the complete producer curve."""
    for job in (parent, child):
        if stable_hash(_identity(job)) != job["key"]:
            raise ValueError("grouped evidence job identity changed")
    if (parent["status"] != "PASS" or parent.get("origin") != "new_attempt"
            or any(parent[key] != child[key] for key in
                   ("family", "source_digest", "candidate_revision", "request_sha256"))):
        raise ValueError("grouped evidence requires the same passing producer and frozen request")
    request_path = Path(parent["request_path"])
    if (request_path != Path(child["request_path"])
            or file_hash(request_path) != parent["request_sha256"]):
        raise ValueError("grouped evidence request bytes changed")
    request = read_json(request_path)
    if (request["source"]["digest"] != parent["source_digest"]
            or request["candidate_revision"] != parent["candidate_revision"]):
        raise ValueError("grouped evidence source or recipe differs from request")
    source = Path(parent["source_root"])
    verify_snapshot(source, request["source"])
    cards = [request["tasks"][job["task"]] for job in (parent, child)]
    for job, card in zip((parent, child), cards):
        if (card["id"] != job["task"] or task_execution_fingerprint(card) != job["execution_sha256"]
                or task_evaluation_fingerprint(card) != job["evaluation_sha256"]):
            raise ValueError("grouped evidence task differs from frozen binding")
        for relative, digest in card["evaluation"].get("sources", {}).items():
            if file_hash(source / relative) != digest:
                raise ValueError("grouped evaluator source changed: " + relative)
    producer, consumer = cards
    a = producer["execution"]
    is_policy_task, validate = _policy()
    if is_policy_task(producer) or is_policy_task(consumer):
        # Scoped variants must share one cohort, and their unchanged original
        # parents must form the same declared uninterrupted group.
        if producer.get("task_cohort") != consumer.get("task_cohort"):
            raise ValueError("grouped evidence cannot pool policy cohorts")
        _check_group(validate(producer), validate(consumer))
    _check_group(producer, consumer)
    groups = [group for group in request["jobs"]
              if {producer["id"], consumer["id"]} <= set(group.get("task_ids", []))]
    if len(groups) != 1:
        raise ValueError("both tasks must belong to one frozen execution job")
    group = groups[0]
    if stable_hash(group["science"]) != group["compatibility_key"]:
        raise ValueError("grouped execution identity changed")
    for job in (parent, child):
        if (group["science"]["execution"][job["task"]] != job["execution_sha256"]
                or group["science"]["evaluation"][job["task"]] != job["evaluation_sha256"]):
            raise ValueError("grouped execution job task binding differs")
    directory = Path(parent["directory"])
    original_spec = read_json(directory / "request.json")
    for key in ("key", "source_digest", "candidate_revision", "task", "request_sha256"):
        if original_spec[key] != parent[key]:
            raise ValueError("producer attempt differs from its indexed request")
    result_path, raw_path = directory / "result.json", directory / "raw.json"
    if file_hash(result_path) != parent["result_sha256"]:
        raise ValueError("producer result bytes changed")
    result = read_json(result_path)
    if result["status"] != "PASS" or file_hash(raw_path) != result["raw_sha256"]:
        raise ValueError("producer is not a verified passing raw receipt")
    raw = read_json(raw_path)
    evidence = raw["evidence"]
    continuity = evidence.get("continuity", {})
    checkpoint = evidence.get("provenance_checkpoint", {})
    if (continuity.get("mode") != "uninterrupted" or not continuity.get("run_id")
            or continuity.get("resume_count") != 0
            or continuity.get("original_schedule_horizon") != (
                None if is_policy_task(producer) and _schedule_free(producer) else a["original_schedule_horizon"])
            or continuity.get("max_total_steps") != a["max_total_steps"]):
        raise ValueError("grouped evidence lacks the uninterrupted producer identity")
    verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
    state_path = Path(checkpoint["artifact_root"]) / checkpoint["path"]
    if (checkpoint["path"] not in checkpoint["artifact_manifest"]["files"]
            or file_hash(state_path) != checkpoint["sha256"]
            or checkpoint.get("completed_steps") != raw["cost"]["completed_steps"]
            or checkpoint["completed_steps"] != evidence["dense"][-1]["step"]):
        raise ValueError("grouped provenance does not bind the complete measured curve")
    parent_grade = grade_result(producer, raw)
    if parent_grade["status"] != "PASS":
        raise ValueError("producer's saved observations do not pass their original gate")
    grade = grade_result(consumer, raw)
    if grade["status"] not in {"PASS", "FAIL"}:
        raise ValueError("saved group does not contain a complete extension: " + str(grade))
    return {
        "schema": "forge_grouped_evidence_v1", "qualification_input": False, "qualification_reuse": False,
        "source_digest": parent["source_digest"], "candidate_revision": parent["candidate_revision"],
        "task": child["task"], "task_key": child["key"],
        "execution_group": a["execution_group"], "compatibility_key": group["compatibility_key"],
        "producer": {key: parent[key] for key in
                     ("key", "task", "attempt_id", "result_sha256", "source_digest", "candidate_revision")},
        "request_sha256": parent["request_sha256"], "raw_sha256": file_hash(raw_path),
        "original_attempt": str(result_path), "continuity": continuity,
        "artifact_manifest_sha256": checkpoint["artifact_manifest"]["sha256"],
        "grader_source_sha256": file_hash(source / "experiments/forge/views.py"),
        "grade": grade, "parent_grade": parent_grade, "runtime": result["runtime"],
        "producer_updates": raw["cost"]["completed_steps"],
        "optimizer_updates_added": 0, "sampling_draws_added": 0,
    }
