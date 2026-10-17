"""Forge binding and independent completion grading for image performance.

These diagnostic tasks measure the public trainer's complete update path. Their
PASS means a finite measured budget completed, and supplies no image-quality or
default-adoption evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time

from .contracts import atomic_json, file_hash, stable_hash


ADAPTER = "dcgan_performance_v1"
KIND = "image_training_performance"
SAMPLING_LAW = "training_update_measurement"


def is_performance_task(task):
    return task.get("adapter") == ADAPTER and task.get("evaluation", {}).get("kind") == KIND


def validate_task(task):
    if not is_performance_task(task):
        raise ValueError("performance task requires its registered adapter and evaluator")
    execution = task["execution"]
    spec = execution["performance"]
    required = {"image_size", "batch_size", "base_channels", "max_channels", "latent_dim", "n_particles",
                "warmup_updates", "measured_updates", "seed", "prior", "max_seconds", "profile_updates"}
    if set(spec) != required:
        raise ValueError(f"performance fields must be exactly {sorted(required)}")
    for key in required - {"prior", "max_seconds"}:
        if type(spec[key]) is not int or spec[key] < (0 if key in {"warmup_updates", "seed", "profile_updates"} else 1):
            raise ValueError(f"performance.{key} must be a valid integer")
    if spec["seed"] != 0:
        raise ValueError("image performance uses protocol seed 0; seed-only experiments are unsupported")
    if spec["image_size"] < 16 or spec["image_size"] & (spec["image_size"] - 1):
        raise ValueError("performance image_size must be a power of two >= 16")
    if spec["profile_updates"] > spec["measured_updates"]:
        raise ValueError("profile_updates cannot exceed the measured budget")
    if (type(spec["max_seconds"]) not in (int, float) or not math.isfinite(spec["max_seconds"])
            or not 0 < spec["max_seconds"] < task["resources"]["timeout_seconds"]):
        raise ValueError("performance wall budget must fit strictly inside the worker timeout")
    if execution["steps"] != spec["warmup_updates"] + spec["measured_updates"]:
        raise ValueError("performance task steps differ from the complete warmup and measurement budget")
    expected = {"batch_size": spec["batch_size"], "z_dim": spec["latent_dim"], "num_particles": spec["n_particles"]}
    if execution["resources"] != expected:
        raise ValueError("performance spec differs from task-owned Recipe resources")
    if spec["prior"] != execution["prior"] or execution.get("prior_contract") != "recipe_owned_v1":
        raise ValueError("performance tasks preserve the declared recipe-owned initial prior")
    from .priors import task_prior
    task_prior(task)
    if execution.get("initializer") != "deterministic_orthogonal":
        raise ValueError("performance tasks use the public deterministic initializer")
    evaluation = task["evaluation"]
    if (evaluation.get("evaluator") != "experiments.forge.scaling_adapter:grade"
            or evaluation.get("training_feedback") is not False
            or evaluation.get("qualification_input") is not False):
        raise ValueError("performance is an explicit nonqualifying diagnostic with no training feedback")


def preflight(task, candidate):
    from .api import task_formulation_context
    from .scaling_benchmark import recipe_blockers
    try:
        validate_task(task)
        context = task_formulation_context(candidate, task, device="cpu")
        return recipe_blockers(context.recipe, task["execution"]["performance"])
    except (KeyError, TypeError, ValueError) as error:
        return getattr(error, "blockers", [str(error)])
    return []


def run(request, task, output, device):
    from .api import task_formulation_context
    from .sampling import executed_receipt
    from .scaling_benchmark import run_benchmark
    started = time.monotonic()
    validate_task(task)
    context = task_formulation_context(request["candidate"], task, request["protocol"], device=device)
    summary = run_benchmark(context.recipe, deepcopy(task["execution"]["performance"]), Path(output), device)
    summary.update(candidate_id=request["candidate"]["id"], task_id=task["id"],
                   forge_binding={"candidate_revision": request.get("candidate_revision"),
                                  "candidate_sha256": stable_hash(request["candidate"]),
                                  "request_id": request.get("request_id"),
                                  "source_digest": request.get("source", {}).get("digest"),
                                  "task_sha256": stable_hash(task)})
    atomic_json(Path(output) / "result.json", summary)
    evidence = deepcopy(summary)
    if summary.get("artifacts", {}).get("trace"):
        evidence["artifacts"]["updates"] = {
            "path": str((Path(output) / summary["artifacts"]["trace"]).resolve()),
            "sha256": summary.get("artifact_sha256", {}).get("trace")}
    evidence.update(executed_receipt(SAMPLING_LAW, eval_output_noise="not_applied_to_measurement"))
    evidence["scoring_weights"] = "live"
    # The core records actual prior trainability, independently of declarations.
    applied = {**context.receipt(), **summary.get("api", {})}
    result = {**summary, "evidence": evidence, "applied": applied,
              "qualification_input": False, "qualification_reuse": False,
              "cost": {"wall_seconds": time.monotonic() - started}}
    atomic_json(Path(output) / "adapter-receipt.json", result)
    return result


def _verdict(status, reason, **extra):
    return {"status": status, "gate_status": status, "reasons": [reason],
            "qualification_input": False, "qualification_reuse": False, **extra}


def grade(task, evidence):
    """Recompute budget/finite/timing checks from the hashed update trace."""
    validate_task(task)
    spec = task["execution"]["performance"]
    artifacts = evidence.get("artifacts", {})
    trace = artifacts.get("updates")
    if not isinstance(trace, dict) or not trace.get("path") or not trace.get("sha256"):
        return _verdict("INCOMPLETE", "missing hashed per-update timing evidence")
    path = Path(trace["path"])
    if file_hash(path) != trace["sha256"]:
        return _verdict("INVALID", "per-update evidence differs from its frozen artifact hash")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
        return _verdict("INVALID", "update trace must contain each completed update exactly once in order")
    for index, row in enumerate(rows):
        expected = "warmup" if index < spec["warmup_updates"] else "measured"
        duration = row.get("duration_ms")
        if row.get("phase") != expected or type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0:
            return _verdict("INVALID", "timing evidence has an invalid phase or duration")
    measured = rows[spec["warmup_updates"]:]
    validity = evidence.get("validity", {})
    finite = all(row.get("finite_losses") is True and isinstance(row.get("losses"), dict)
                 and all(type(value) in (float, int) and math.isfinite(value)
                         for value in row["losses"].values()) for row in rows)
    if not finite or validity.get("finite_model_state") is False:
        return _verdict("FAIL", "the measured public trainer produced nonfinite losses or model state",
                        benchmark_status=evidence.get("benchmark_status"))
    if len(rows) != task["execution"]["steps"] or len(measured) != spec["measured_updates"]:
        return _verdict("INCOMPLETE", "the complete warmup and measured update budget did not finish",
                        benchmark_status=evidence.get("benchmark_status"), completed_updates=len(rows))
    if validity.get("finite_model_state") is not True:
        return _verdict("INCOMPLETE", "missing actual finite model-state audit")
    if any(validity.get(key) is not True for key in ("changed_generator", "changed_discriminator")):
        return _verdict("FAIL", "both learned networks must change during actual public-API training")
    values = [row["duration_ms"] for row in measured]
    metrics = deepcopy(evidence.get("metrics", {}))
    metrics.update(completed_updates=len(rows), measured_updates=len(measured), finite_update_fraction=1.,
                   update_ms_median=statistics.median(values),
                   images_per_second=1000. * spec["batch_size"] / statistics.mean(values))
    return _verdict("PASS", "complete finite public-trainer timing budget; diagnostic performance evidence only",
                    metrics=metrics, benchmark_status=evidence.get("benchmark_status"))
