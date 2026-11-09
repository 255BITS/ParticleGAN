"""Prospective execution witnesses and exact certified coverage exports.

Missing observations stay missing. These helpers neither grade evidence nor
repair a receipt from a task name, and never update historical publications.
"""
from __future__ import annotations

import json
from copy import deepcopy

from .contracts import canonical, stable_hash
from .initialization import task_initializer
from .sampling import FIELDS, validate_declaration
from .sources import compute_profile, runtime_manifest
from .views import task_evaluation_fingerprint, task_execution_fingerprint


def execution_environment(torch, device):
    """Observe the executing process, including actual deterministic settings."""
    device = torch.device(device)
    model = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    compute = compute_profile(device.type, model, threads=torch.get_num_threads())
    if compute.get("availability") == "unavailable":
        raise ValueError("compute hardware observation unavailable")
    compute.update(deterministic=torch.are_deterministic_algorithms_enabled(),
                   tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
                   autograd_multithreading_enabled=torch.autograd.is_multithreading_enabled())
    return {"runtime": runtime_manifest(), "compute": compute}


def observed_contract(request, job, name, raw, environment):
    """Bind actual applied facts to a complete frozen contract, or explain gaps.

    The resolver is used only to *compare* a recipe. The witness always carries
    the executed adapter's recipe, initializer, RNG seed and sampling receipt.
    Constructing a CPU context creates private streams, never models or draws
    from any executing stream.
    """
    errors = []
    task = request.get("tasks", {}).get(name, {})
    science = job.get("science", {})
    applied = raw.get("applied", raw)
    evidence = raw.get("evidence", {})
    if not all(isinstance(value, dict) for value in (task, science, applied, evidence, environment)):
        return None, ["malformed_observation_inputs"]
    recipe = applied.get("recipe_serialized", applied.get("recipe"))
    if not isinstance(recipe, dict) or not recipe:
        errors.append("missing_applied_recipe")
    elif ("recipe_serialized" in applied and "recipe" in applied
          and stable_hash(applied["recipe_serialized"]) != stable_hash(applied["recipe"])):
        errors.append("conflicting_applied_recipes")
    protocol = request.get("protocol", {})
    if (type(protocol.get("seed")) is not int or applied.get("rng", {}).get("seed") != protocol.get("seed")
            or science.get("protocol") != protocol or science.get("seed") != protocol.get("seed")):
        errors.append("protocol_seed_mismatch_or_missing")
    try:
        initializer = task_initializer(task, request.get("candidate", {}))
        if (applied.get("initializer") != initializer
                or science.get("task_initializers", {}).get(name) != initializer):
            errors.append("initializer_mismatch_or_missing")
    except (KeyError, TypeError, ValueError):
        errors.append("unresolved_initializer")
    sampling = {key: evidence.get(key) for key in FIELDS}
    sampling["scoring_weights"] = evidence.get("executed_scoring_weights")
    try:
        expected = validate_declaration(task, required=True)
        expected["scoring_weights"] = task["evaluation"].get("scoring_weights", "live")
        if sampling != expected:
            errors.append("sampling_or_weights_mismatch_or_missing")
    except (KeyError, TypeError, ValueError):
        errors.append("unresolved_sampling")
    if (not environment.get("runtime") or environment["runtime"] != request.get("runtime")
            or environment["runtime"] != science.get("runtime")):
        errors.append("runtime_mismatch_or_missing")
    if not environment.get("compute") or environment["compute"] != science.get("compute"):
        errors.append("compute_mismatch_or_missing")
    if (not task.get("execution") or not task.get("evaluation")
            or science.get("execution", {}).get(name) != task_execution_fingerprint(task)
            or science.get("evaluation", {}).get(name) != task_evaluation_fingerprint(task)
            or job.get("compatibility_key") != stable_hash(science)):
        errors.append("frozen_contract_mismatch_or_missing")
    if not errors:
        try:
            from .api import task_formulation_context
            expected_recipe = task_formulation_context(request["candidate"], task, protocol, device="cpu").recipe.to_dict()
            if stable_hash(recipe) != stable_hash(expected_recipe):
                errors.append("resolved_recipe_mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"unresolved_recipe: {type(exc).__name__}")
    if errors:
        return None, errors
    return json.loads(canonical({"schema_version": 1, "resolved_recipe": recipe,
                     "initializer": applied["initializer"], "protocol": protocol,
                     "sampling": sampling, "runtime": environment["runtime"], "compute": environment["compute"],
                     "execution": task["execution"], "evaluation": task["evaluation"]})), []


def attach_observations(request, job, result, environment):
    """Add optional observations without altering metrics, costs or grades."""
    result = deepcopy(result)
    for name in job.get("task_ids", [job["task_id"]]):
        member = result.get("task_results", {}).get(name) if "task_results" in result else result
        if not isinstance(member, dict):
            continue
        # An adapter cannot pre-attest the worker environment.
        member.pop("observed_contract", None)
        try:
            witness, errors = observed_contract(request, job, name, member, environment)
        except Exception as exc:  # noqa: BLE001 -- optional provenance cannot change the numerical outcome
            witness, errors = None, [f"observation_unavailable: {type(exc).__name__}: {exc}"]
        if witness is not None:
            member["observed_contract"] = witness
            member.pop("observed_contract_gaps", None)
        else:
            member["observed_contract_gaps"] = errors
    return result


def coverage_export(resolved, result, certificate, *, family=None):
    """Export exact rows from a certified attempt; no aggregate or regrading.

    ``family`` is an optional display label supplied by a publisher. It never
    enters the scientific witness. Consumers still verify and independently
    grade the original request/result/evidence triplet.
    """
    request = resolved["request"]
    digest = stable_hash(result)
    if (certificate.get("result_hash") != digest or certificate.get("source") != request["source"]
            or certificate.get("runtime") != request.get("runtime")
            or result.get("candidate_revision") != request["candidate_revision"]
            or result.get("attempt_id") != resolved["worker"]["attempt"]
            or result.get("retry_of") != resolved.get("retry_of")):
        raise ValueError("invalid producer certificate")
    raw = result.get("raw", {})
    raw_result = raw.get("result", {})
    records, gaps = [], {}
    for row in result.get("task_results", []):
        name = row["task_id"]
        member = raw_result.get("task_results", {}).get(name, raw_result)
        witness = row.get("observed_contract")
        if (raw.get("attempt_status") != "completed" or row.get("raw_status") != "completed"
                or row.get("gate_status") not in {"PASS", "FAIL"}):
            gaps[name] = ["non_numerical_or_incomplete_result"]
            continue
        if not witness or witness != member.get("observed_contract"):
            gaps[name] = member.get("observed_contract_gaps", ["missing_or_conflicting_observed_contract"])
            continue
        jobs = [job for job in request["jobs"] if name in job.get("task_ids", [job["task_id"]])]
        if len(jobs) != 1 or row.get("compatibility_key") != jobs[0].get("compatibility_key"):
            gaps[name] = ["producer_job_mismatch"]
            continue
        observed, errors = observed_contract(request, jobs[0], name, member,
            {key: witness.get(key) for key in ("runtime", "compute")})
        if errors or observed != witness:
            gaps[name] = errors or ["conflicting_observed_contract"]
            continue
        if request.get("requires_independent_grading"):
            grading = raw.get("grading", {})
            grade = grading.get("grades", {}).get(name, {})
            if (grading.get("raw_hash") != stable_hash(raw_result)
                    or grading.get("source_digest") != request["source"]["digest"]
                    or grade.get("gate_status", grade.get("status")) != row["gate_status"]):
                gaps[name] = ["invalid_frozen_grading_certificate"]
                continue
        task = request["tasks"][name]
        record = {"candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
                  "source_digest": request["source"]["digest"], "task": name,
                  "execution_sha256": task_execution_fingerprint(task),
                  "evaluation_sha256": task_evaluation_fingerprint(task), "status": row["gate_status"],
                  "attempted": True, "diagnostic": request.get("view", {}).get("evidence_scope") == "research_diagnostic",
                  "runtime": witness["runtime"], "observed_contract": deepcopy(witness),
                  "producer": {"attempt_id": result["attempt_id"], "result_hash": digest}}
        record["key"] = stable_hash([record[field] for field in
            ("source_digest", "candidate_revision", "task", "execution_sha256", "evaluation_sha256")])
        if family is not None:
            record["family"] = family
        records.append(record)
    return {"schema": "forge_observed_coverage_v1", "qualification_input": False,
            "qualification_reuse": False, "records": records, "gaps": gaps}
