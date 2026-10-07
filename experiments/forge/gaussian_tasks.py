"""Scalar acquisition smoke and own-checkpoint continuous-learning gates.

Task variants own architecture and prior conditions. This module never changes
candidate recipes or substitutes a fixed/target-informed initialization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import torch

from .contracts import atomic_json, file_hash
from .state import state_digest

SMOKE_KIND = "gaussian_smoke"
STABILITY_KIND = "gaussian_stability"
THRESHOLDS = [["sample_count", ">=", 4096], ["finite_fraction", "==", 1.],
              ["mean_error_sigma", "<=", .2], ["std_ratio", ">=", .8],
              ["std_ratio", "<=", 1.2], ["cdf_ks", "<=", .05]]


def schedule(start, stop):
    """Keep the original 24 observations per 1,000-update spacing."""
    return [base + math.ceil(i * 1000 / 24) for base in range(start, stop, 1000)
            for i in range(1, 25)]


def validate_task(task):
    evaluation, execution = task["evaluation"], task["execution"]
    if task["adapter"] != "transfer_vector" or evaluation["kind"] not in {SMOKE_KIND, STABILITY_KIND}:
        raise ValueError("scalar smoke/stability needs its declared vector adapter")
    expected = dict(thresholds=THRESHOLDS, observations=24, eval_samples=4096,
                    scoring_weights="live", sample_evaluator="benchmarks.toy_audit.gaussian1d_quality:score_samples",
                    evaluator="experiments.forge.gaussian_tasks:grade")
    for name, value in expected.items():
        if evaluation.get(name) != value:
            raise ValueError(f"{task['id']}: Gaussian {name} is fixed by its evaluator")
    if execution.get("prior") != dict(kind="mog", sigma=.1, standardize=False, learnable=True):
        raise ValueError("Gaussian smoke/stability requires learned256MoG sigma .1 without standardization")
    spec = execution["host_definition"]
    if (spec["means"] != [[2.]] or spec["covariances"] != [[[.25]]] or spec["masses"] != [1.]
            or spec["kind"] != "gaussian_mixture" or spec["batch"] != 128 or spec["z_dim"] != 2
            or spec["particles"] != 256):
        raise ValueError("Gaussian task target/resources differ from its declared smoke cohort")
    if evaluation["kind"] == SMOKE_KIND:
        if execution["steps"] != 1000 or execution.get("produces_state") is not True:
            raise ValueError("Gaussian smoke completes all 1,000 updates and produces its own checkpoint")
        if evaluation.get("confirmation") != "independent_draw_same_state":
            raise ValueError("Gaussian smoke requires independent same-state confirmation")
    else:
        fixed = dict(stationary_end=4000, shift_end=6000, shift_mean=3., reacquisition_deadline=5000,
                     minimum_stable_checks=5, stationary_checks=72, shift_checks=48, shift_hold_checks=24)
        if execution["steps"] != 6000 or execution.get("preserve_prefix_steps") != 1000:
            raise ValueError("Gaussian stability must continue its own 1,000-update smoke checkpoint to 6,000")
        if execution.get("original_schedule_horizon") != 1000:
            raise ValueError("Gaussian stability must preserve the smoke schedule horizon")
        for name, value in fixed.items():
            if evaluation.get(name) != value:
                raise ValueError(f"Gaussian stability {name} is fixed by its evaluator")
        if task["dependencies"] != [{"task": execution.get("continuation_of"), "kind": "checkpoint"}]:
            raise ValueError("Gaussian stability requires its own declared smoke checkpoint")


def bounds(metrics):
    failed = []
    for name, op, bound in THRESHOLDS:
        value = metrics.get(name)
        finite = type(value) in (int, float) and math.isfinite(value)
        passed = finite and ((value >= bound) if op == ">=" else (value <= bound) if op == "<=" else value == bound)
        if not passed:
            failed.append(name + " " + op + " " + str(bound))
    return failed


def _verdict(status, reason, **details):
    return dict(status=status, gate_status=status, reasons=[reason], **details)


def _rows(value, expected):
    if not isinstance(value, list) or not value:
        return _verdict("INCOMPLETE", "missing Gaussian observation curve")
    steps = [row.get("step") if isinstance(row, dict) else None for row in value]
    if any(type(step) is not int for step in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
        return _verdict("INVALID", "Gaussian observation steps must be unique increasing integers")
    if steps != expected:
        return _verdict("INCOMPLETE", "all exact Gaussian observation checkpoints are required")
    for row in value:
        for name, _, _ in THRESHOLDS:
            if name not in row:
                return _verdict("INCOMPLETE", "missing Gaussian metric " + name)
    return None


def grade(task, evidence):
    """Recompute gates from the complete curve; never trust PASS stamps."""
    validate_task(task)
    kind = task["evaluation"]["kind"]
    stop = 1000 if kind == SMOKE_KIND else 6000
    if evidence.get("completed_steps") != stop:
        return _verdict("INCOMPLETE", "Gaussian execution did not complete its declared budget")
    rows = evidence.get("observations")
    missing = _rows(rows, schedule(0, 1000) if kind == SMOKE_KIND else schedule(1000, 4000) + schedule(4000, 6000))
    if missing:
        return missing
    # Nonfinite outputs always fail, even if another scheduled state passed.
    if any(row["finite_fraction"] != 1. for row in rows):
        return _verdict("FAIL", "nonfinite Gaussian output")
    for row in rows:
        if any(type(row[name]) not in (int, float) or not math.isfinite(row[name]) for name, _, _ in THRESHOLDS):
            return _verdict("INVALID", "missing finite Gaussian metric")
    if kind == SMOKE_KIND:
        confirmations = evidence.get("confirmations")
        if not isinstance(confirmations, list):
            return _verdict("INCOMPLETE", "missing independent confirmation evidence")
        if any(not isinstance(row, dict) for row in confirmations):
            return _verdict("INVALID", "Gaussian confirmation rows must be objects")
        if [row.get("step") for row in confirmations] != schedule(0, 1000):
            return _verdict("INCOMPLETE", "all 24 independent confirmation draws are required")
        indexed = {row["step"]: row for row in rows}
        seen, hits = set(), []
        for confirm in confirmations:
            step = confirm.get("step")
            if step not in indexed or step in seen:
                return _verdict("INVALID", "confirmation must refer to a unique scheduled state")
            seen.add(step)
            if (confirm.get("primary_state_sha256") != confirm.get("confirmed_state_sha256")
                    or not isinstance(confirm.get("primary_state_sha256"), str)
                    or len(confirm["primary_state_sha256"]) != 64
                    or confirm.get("independent_stream") != "eval/live/smoke_confirmation"
                    or confirm.get("training_state_unchanged") is not True):
                return _verdict("INVALID", "confirmation is not an independent draw at the same training state")
            if not isinstance(confirm.get("metrics"), dict):
                return _verdict("INVALID", "Gaussian confirmation metrics must be an object")
            if confirm["metrics"].get("finite_fraction") != 1.:
                return _verdict("FAIL", "nonfinite Gaussian confirmation output")
            if not bounds(indexed[step]) and not bounds(confirm.get("metrics", {})):
                hits.append(step)
        return _verdict("PASS" if hits else "FAIL", "any scheduled full pass with independent same-state confirmation",
                        metrics=rows[-1], evaluator_result=dict(confirmed_steps=hits, first_confirmed_step=min(hits) if hits else None,
                        passing_observations=sum(not bounds(row) for row in rows), observations=len(rows)))
    stationary = rows[:72]
    shifted = rows[72:]
    before_deadline = shifted[:24]
    reacquired = len(before_deadline) >= 5 and all(not bounds(row) for row in before_deadline[-5:])
    held = all(not bounds(row) for row in shifted[24:])
    stationary_ok = all(not bounds(row) for row in stationary)
    frozen = evidence.get("frozen_observations")
    missing = _rows(frozen, schedule(4000, 6000))
    if missing:
        return missing
    continuity = evidence.get("continuity", {})
    if (continuity.get("prefix_steps") != 1000 or continuity.get("restored_exactly") is not True
            or continuity.get("history_reset") is not False or continuity.get("frozen_completed_steps") != 4000
            or continuity.get("matched_frozen_draws") is not True):
        return _verdict("INVALID", "Gaussian stability requires exact own-state continuation and matched frozen control")
    passed = stationary_ok and reacquired and held
    return _verdict("PASS" if passed else "FAIL", "stationary hold, deadline reacquisition and shifted hold",
                    metrics=rows[-1], evaluator_result=dict(stationary_passes=sum(not bounds(row) for row in stationary),
                    stationary_checks=72, reacquisition_status="PASS" if reacquired else "FAIL",
                    shift_hold_passes=sum(not bounds(row) for row in shifted[24:]), shift_hold_checks=24,
                    frozen_passes=sum(not bounds(row) for row in frozen), frozen_checks=48))


def build(request, task, device, *, max_steps=1000):
    """Shared public constructor; no task-specific trainer recipes."""
    from .api import task_formulation_context
    from .vectorprofiles import resolve_vector_spec, build_vector_models
    validate_task(task)
    spec = resolve_vector_spec(task)
    context = task_formulation_context(request["candidate"], task, request.get("protocol"), device=device)
    generator, discriminator = build_vector_models(context, spec)
    return context, context.build_trainer(generator, discriminator, max_steps=max_steps), spec


def training_digest(context):
    state = deepcopy(context.state_dict())
    state.pop("streams")
    state["trainer"].pop("streams", None)
    return state_digest(state)


def _restore_parent(request, task, prerequisites, context, *, diagnostic=False):
    from .artifacts import verify_artifacts
    parent_id = task["execution"]["continuation_of"]
    parent = (prerequisites or {}).get(parent_id)
    expected = next((job["compatibility_key"] for job in request.get("jobs", [])
                     if parent_id in job.get("task_ids", [job["task_id"]])), None)
    if (not parent or parent.get("candidate_revision") != request.get("candidate_revision")
            or not expected or parent.get("compatibility_key") != expected
            or (parent.get("result", {}).get("gate_status") != "PASS" and not diagnostic)):
        raise ValueError("Gaussian stability needs this candidate's compatible passing own smoke checkpoint")
    parent_task = request.get("tasks", {}).get(parent_id)
    if parent_task is None or parent_task["execution"]["host_definition"] != task["execution"]["host_definition"]:
        raise ValueError("Gaussian continuation architecture/target must match its own smoke task")
    if parent_task["execution"]["prior"] != task["execution"]["prior"]:
        raise ValueError("Gaussian continuation prior differs from own smoke task")
    old = parent["result"]["evidence"]
    verify_artifacts(old["artifact_root"], old["artifact_manifest"])
    checkpoint = old["checkpoint"]
    if checkpoint["path"] not in old["artifact_manifest"]["files"]:
        raise ValueError("Gaussian parent checkpoint is not certified by its artifact manifest")
    path = Path(old["artifact_root"]) / checkpoint["path"]
    if file_hash(path) != checkpoint["sha256"]:
        raise ValueError("Gaussian parent checkpoint bytes differ")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["trainer"]["completed_steps"] != 1000 or state_digest(saved) != checkpoint["state_sha256"]:
        raise ValueError("Gaussian checkpoint does not represent its declared exact prefix")
    context.load_state_dict(saved)
    if state_digest(context.state_dict()) != state_digest(saved):
        raise ValueError("Gaussian checkpoint restore changed state")
    return dict(prefix_steps=1000, restored_exactly=True, history_reset=False,
                parent_checkpoint_sha256=file_hash(path), parent_state_sha256=state_digest(saved),
                parent_compatibility_key=expected, parent_candidate_revision=parent["candidate_revision"],
                diagnostic_continuation=diagnostic, parent_smoke_status=parent["result"]["gate_status"])


def run_gaussian(request, task, output, device, *, prerequisites=None, diagnostic=False):
    """One public update loop shared by ordinary tasks and architecture studies."""
    from .adapters import _Run, _event, _host_receipt
    from benchmarks.toy_audit.gaussian1d_quality import sample_target, score_samples
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Gaussian smoke/stability requires CUDA; no CPU fallback")
    output = Path(output)
    validate_task(task)
    kind = task["evaluation"]["kind"]
    context, trainer, spec = build(request, task, device)
    host = _host_receipt(spec, None, trainer.G, trainer.D)
    continuity = None
    if kind == STABILITY_KIND:
        continuity = _restore_parent(request, task, prerequisites, context, diagnostic=diagnostic)
        trainer.extend_execution(6000)
    run = _Run(context, trainer, output, task)
    artifact_output = output / "evaluator"
    artifact_output.mkdir(exist_ok=False)
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    confirmation_stream = context.streams.generator("eval", component="live", purpose="smoke_confirmation")
    rows, confirmations, records, frozen_rows = [], [], [], []
    frozen_context = frozen = None
    checks = set(schedule(0, 1000) if kind == SMOKE_KIND else schedule(1000, 4000) + schedule(4000, 6000))
    digests = {phase: hashlib.sha256() for phase in ("stationary", "shift")}
    torch.save(context.state_dict(), artifact_output / "initial-state.pt")

    def observe(step):
        before = training_digest(context)
        points = run.sample(4096).detach().cpu()
        metrics = score_samples(points, spec, step)
        row = dict(step=step, **metrics)
        snapshot = dict(step=step, samples=points, metrics=metrics, training_state_sha256=before)
        if kind == SMOKE_KIND:
            confirmed = trainer.sample(4096, generator=confirmation_stream, output_noise=False).detach().cpu()
            after = training_digest(context)
            confirm = dict(step=step, metrics=score_samples(confirmed, spec, step), primary_state_sha256=before,
                           confirmed_state_sha256=after, independent_stream="eval/live/smoke_confirmation",
                           training_state_unchanged=before == after)
            confirmations.append(confirm)
            snapshot["confirmation_samples"] = confirmed
            snapshot["confirmation"] = confirm
        if frozen is not None:
            stream = frozen_context.streams.generator("eval", component="live", purpose="samples")
            frozen_points = frozen.sample(4096, generator=stream, output_noise=False).detach().cpu()
            active_stream = context.streams.generator("eval", component="live", purpose="samples")
            if not torch.equal(stream.get_state(), active_stream.get_state()):
                raise ValueError("Gaussian frozen and active evaluation draws differ")
            frozen_metrics = score_samples(frozen_points, spec, step)
            frozen_rows.append(dict(step=step, **frozen_metrics))
            snapshot.update(frozen_samples=frozen_points, frozen_metrics=frozen_metrics)
        if training_digest(context) != before:
            raise ValueError("Gaussian observation changed training state")
        records.append(snapshot)
        _event("observation", task=task["id"], step=step, metrics=metrics, full_pass=not bounds(metrics))
        return row

    with context.streams.preserve():
        run.evaluate(lambda: observe(trainer.completed_steps))
    confirmations.clear()
    began = time.monotonic()
    start = trainer.completed_steps
    for step in range(start + 1, task["execution"]["steps"] + 1):
        if kind == STABILITY_KIND and step == 4001:
            saved = context.state_dict()
            torch.save(saved, artifact_output / "pre-shift-state.pt")
            frozen_context, frozen, _ = build(request, task, device, max_steps=6000)
            frozen_context.load_state_dict(saved)
            if state_digest(frozen_context.state_dict()) != state_digest(saved):
                raise ValueError("Gaussian frozen restore changed its own pre-shift state")
            spec["means"] = [[3.]]
        phase = "shift" if step > 4000 else "stationary"
        real = sample_target(spec, context.recipe.batch_size, data, step - 1)
        digests[phase].update(real.numpy().tobytes())
        run.step(real.to(device))
        if step in checks:
            rows.append(run.evaluate(lambda: observe(step)))
        if time.monotonic() - began > task["resources"]["timeout_seconds"]:
            raise TimeoutError("Gaussian declared task allowance exceeded")
    torch.cuda.synchronize(device)
    state = context.state_dict()
    torch.save(state, artifact_output / "state.pt")
    torch.save(records, artifact_output / "observed-samples.pt")
    if frozen_context is not None:
        torch.save(frozen_context.state_dict(), artifact_output / "frozen-state.pt")
        continuity.update(frozen_completed_steps=frozen.completed_steps, matched_frozen_draws=True)
    evidence = dict(completed_steps=trainer.completed_steps, observations=rows, confirmations=confirmations,
                    live=rows[-1], host=host, frozen_observations=frozen_rows, continuity=continuity,
                    data_sha256={phase: digest.hexdigest() for phase, digest in digests.items()},
                    artifact_root=str(artifact_output.resolve()), checkpoint=dict(path="state.pt", sha256=file_hash(artifact_output / "state.pt"),
                    state_sha256=state_digest(state)), diagnostic=diagnostic,
                    saved_observer_outputs=dict(path="observed-samples.pt", sha256=file_hash(artifact_output / "observed-samples.pt")))
    result = run.receipt(evidence, save_state=False)
    result["gaussian_grade"] = grade(task, result["evidence"])
    atomic_json(output / "adapter-receipt.json", result)
    return result
