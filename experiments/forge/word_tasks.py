"""Confirmed acquisition and strict own-checkpoint retention of five words.

Both questions use WordFixture's public G/E/joint-D loop. Historical sustained
acquisition remains a different task and is never regraded by this evaluator.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from .artifacts import manifest_artifacts, save_provenance_checkpoint, verify_artifacts
from .contracts import atomic_json, file_hash
from .sampling import JOINT_WORDS_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer

SMOKE_KIND, HOLD_KIND = "word_smoke", "word_hold"
SMOKE_STEPS, HOLD_STEPS, HORIZON, OBSERVATIONS = 20001, 4000, 20000, 24
CONFIRMATION_STREAM = "eval/live/word_confirmation"
RECONSTRUCTION_STREAM = "eval/live/word_reconstruction_confirmation"


def schedule(updates):
    return [math.ceil(i * updates / OBSERVATIONS) for i in range(1, OBSERVATIONS + 1)]


def validate_task(task):
    from .word_adapter import THRESHOLDS
    execution, evaluation = task["execution"], task["evaluation"]
    kind = evaluation["kind"]
    if task["adapter"] != "word_joint" or kind not in {SMOKE_KIND, HOLD_KIND}:
        raise ValueError("word ladder requires the public joint word host")
    fixed = dict(thresholds=THRESHOLDS, eval_samples=1024, observations=24,
        scoring_weights="live", evaluator="experiments.forge.word_tasks:grade",
        confirmation="independent_draw_same_state")
    if any(evaluation.get(key) != value for key, value in fixed.items()):
        raise ValueError("word ladder requires unchanged full bounds and 24 independent paired checks")
    if execution.get("original_schedule_horizon") != HORIZON:
        raise ValueError("word continuation preserves the original 20,000-update schedule horizon")
    if kind == SMOKE_KIND:
        if execution.get("steps") != SMOKE_STEPS or execution.get("produces_state") is not True or task["dependencies"]:
            raise ValueError("word smoke finishes 20,001 updates and saves its earliest confirmed own state")
        if execution.get("checkpoint_selection") != "earliest_confirmed_passing_state":
            raise ValueError("word smoke cannot select a later or final checkpoint")
    else:
        if (execution.get("steps") != HOLD_STEPS or execution.get("produces_state") is not False
                or execution.get("continuation_of") != "five_word_joint_smoke"
                or task["dependencies"] != [{"task": "five_word_joint_smoke", "kind": "checkpoint"}]):
            raise ValueError("word hold continues its own confirmed smoke state for exactly 4,000 updates")
        if evaluation.get("hold_policy") != "all_scheduled_checks_including_restored_state":
            raise ValueError("word hold requires every declared check, including the restored state")


def bounds(metrics):
    from .word_adapter import THRESHOLDS
    failed = []
    for name, op, threshold in THRESHOLDS:
        value = metrics.get(name)
        finite = type(value) in (int, float) and math.isfinite(value)
        passed = finite and (value >= threshold if op == ">=" else value <= threshold if op == "<=" else value == threshold)
        if not passed:
            failed.append(name)
    return failed


def _verdict(status, reason, **details):
    return dict(status=status, gate_status=status, reasons=[reason], **details)


def _validate_confirmation(row, step):
    if not isinstance(row, dict) or row.get("step") != step:
        raise ValueError("word confirmation must refer to its scheduled state")
    identity = row.get("primary_state_sha256")
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
            or identity != row.get("confirmed_state_sha256")
            or row.get("training_state_unchanged") is not True
            or row.get("independent_stream") != CONFIRMATION_STREAM
            or row.get("reconstruction_stream") != RECONSTRUCTION_STREAM
            or not isinstance(row.get("metrics"), dict)):
        raise ValueError("word confirmation requires independent same-state clean generation and paired reconstruction")


def grade(task, evidence):
    """Require the whole budget; reconstruct acquisition/hold from actual curves."""
    validate_task(task)
    from .priors import validate_prior_policy
    prior_grade = validate_prior_policy(task, evidence)
    if prior_grade is not None:
        return _verdict(prior_grade["status"], prior_grade["reason"])
    kind = task["evaluation"]["kind"]
    total = SMOKE_STEPS if kind == SMOKE_KIND else HOLD_STEPS
    if evidence.get("executed_updates") != total:
        return _verdict("INCOMPLETE", "word execution did not finish the full declared update budget")
    prefix = 0 if kind == SMOKE_KIND else evidence.get("continuity", {}).get("prefix_steps")
    if type(prefix) is not int or not 0 <= prefix <= SMOKE_STEPS:
        return _verdict("INVALID", "word continuation lacks a valid acquisition step")
    if kind == HOLD_KIND and prefix not in schedule(SMOKE_STEPS):
        return _verdict("INVALID", "word continuation must begin at a scheduled acquisition state")
    if evidence.get("completed_steps") != prefix + total:
        return _verdict("INCOMPLETE", "word completed-step labels differ from executed updates")
    counts = evidence.get("guards", {}).get("optimizer_updates", {})
    from .priors import expected_prior_updates
    if any(type(counts.get(role)) is not int or counts[role] != (
            expected_prior_updates(task, evidence, prefix + total) if role == "prior" else prefix + total)
           for role in ("generator", "encoder", "prior", "discriminator")):
        return _verdict("INCOMPLETE", "all four actual optimizer histories must cover every cumulative update")
    rows, confirmations = evidence.get("observations"), evidence.get("confirmations")
    expected = ([prefix] if kind == HOLD_KIND else []) + [prefix + offset for offset in schedule(total)]
    if (not isinstance(rows, list) or not isinstance(confirmations, list)
            or [row.get("step") for row in rows if isinstance(row, dict)] != expected
            or [row.get("step") for row in confirmations if isinstance(row, dict)] != expected
            or len(rows) != len(expected) or len(confirmations) != len(expected)):
        return _verdict("INCOMPLETE", "every exact primary and independent confirmation checkpoint is required")
    from .word_adapter import THRESHOLDS
    for row, confirm in zip(rows, confirmations):
        try:
            _validate_confirmation(confirm, row["step"])
        except ValueError as error:
            return _verdict("INVALID", str(error))
        for metrics in (row, confirm["metrics"]):
            if any(type(metrics.get(name)) not in (int, float) or not math.isfinite(metrics[name]) for name, _, _ in THRESHOLDS):
                return _verdict("INVALID", "word observations require all finite numerical metrics")
    hits = [row["step"] for row, confirm in zip(rows, confirmations)
            if not bounds(row) and not bounds(confirm["metrics"])]
    details = dict(confirmed_steps=hits, first_confirmed_step=hits[0] if hits else None,
        passing_observations=sum(not bounds(row) for row in rows),
        confirmed_checks=len(hits), observations=len(rows), endpoint_passed=not bounds(rows[-1]))
    if kind == SMOKE_KIND:
        checkpoint = evidence.get("checkpoint")
        if hits and (not isinstance(checkpoint, dict) or checkpoint.get("completed_steps") != hits[0]
                or checkpoint.get("training_state_sha256") != confirmations[expected.index(hits[0])]["primary_state_sha256"]
                or checkpoint.get("selection") != "earliest_confirmed_passing_state"):
            return _verdict("INVALID", "smoke checkpoint must preserve the earliest confirmed passing training state")
        return _verdict("PASS" if hits else "FAIL", "any scheduled full joint pass plus independent same-state confirmation",
                        metrics=rows[-1], evaluator_result=details)
    continuity = evidence.get("continuity", {})
    if (continuity.get("restored_exactly") is not True or continuity.get("history_reset") is not False
            or continuity.get("parent_smoke_status") != "PASS"
            or continuity.get("parent_confirmed_step") != prefix
            or continuity.get("same_recipe_prior_architecture") is not True):
        return _verdict("INVALID", "word hold requires exact compatible own-state continuation")
    return _verdict("PASS" if len(hits) == len(expected) else "FAIL",
                    "every joint generation/inverse check throughout the fixed continuation window",
                    metrics=rows[-1], evaluator_result=details)


def restore_parent(request, task, prerequisites, fixture, context):
    from .views import grade_result, task_fingerprint
    parent_id = task["execution"]["continuation_of"]
    parent = (prerequisites or {}).get(parent_id)
    expected = next((job["compatibility_key"] for job in request.get("jobs", [])
                     if parent_id in job.get("task_ids", [job["task_id"]])), None)
    if (not parent or parent.get("candidate_revision") != request.get("candidate_revision")
            or not expected or parent.get("compatibility_key") != expected):
        raise ValueError("word hold needs this candidate's compatible own smoke checkpoint")
    parent_task = request.get("tasks", {}).get(parent_id)
    if parent_task is None or grade_result(parent_task, parent["result"])["gate_status"] != "PASS":
        raise ValueError("word hold requires a completely measured passing smoke producer")
    for field in ("host_definition", "prior", "initializer", "original_schedule_horizon"):
        if parent_task["execution"][field] != task["execution"][field]:
            raise ValueError("word hold changed task-owned " + field)
    evidence = parent["result"]["evidence"]
    root, checkpoint = Path(evidence["artifact_root"]), evidence["checkpoint"]
    verify_artifacts(root, evidence["artifact_manifest"])
    if checkpoint["path"] not in evidence["artifact_manifest"]["files"]:
        raise ValueError("word checkpoint is not certified by its complete artifact manifest")
    path = root / checkpoint["path"]
    if file_hash(path) != checkpoint["sha256"]:
        raise ValueError("word acquisition checkpoint bytes differ")
    saved = torch.load(path, weights_only=True, map_location=context.device)
    prefix = checkpoint["completed_steps"]
    current = fixture.state_dict()
    frozen_fixture = saved["fixture"]
    for field in ("version", "case", "seed", "recipe", "api_components"):
        if frozen_fixture.get(field) != current[field]:
            raise ValueError("word fixture checkpoint changed static field " + field)
    if frozen_fixture.get("max_steps") != SMOKE_STEPS or prefix not in schedule(SMOKE_STEPS):
        raise ValueError("word acquisition checkpoint has an invalid external cap or prefix")
    context.streams.validate_state_dict(saved["streams"])
    if saved["streams"]["manifest"] != context.streams.manifest():
        raise ValueError("word continuation changed named RNG bindings")
    def named_state(family, component, purpose):
        keys = [key for key, binding in saved["streams"]["manifest"]["bindings"].items()
                if (binding["family"], binding["component"], binding["purpose"]) == (family, component, purpose)]
        if len(keys) != 1:
            raise ValueError("word checkpoint is missing a uniquely named consumed stream")
        return saved["streams"]["states"][keys[0]].cpu()
    if not torch.equal(frozen_fixture["data_generator"].cpu(), named_state("data", "target", "training")):
        raise ValueError("word fixture and named data stream disagree")
    policy_streams = dict(latent_generator=("prior", "latent", "indices"),
        penalty_generator=("noise", "penalty", "training"), eval_generator=("eval", "sampler", "samples"),
        noise_generator=("noise", "generator", "output"))
    for key, binding in policy_streams.items():
        if not torch.equal(frozen_fixture["api_state"]["streams"][key].cpu(), named_state(*binding)):
            raise ValueError("word policy and named stream disagree: " + key)
    optimizers = frozen_fixture["api_state"]["optimizers"]
    if len(optimizers) != 2:
        raise ValueError("word checkpoint must retain both optimizer histories")
    for optimizer in optimizers:
        counters = [value["step"] for value in optimizer["state"].values() if "step" in value]
        if not counters or any(float(value) != prefix for value in counters):
            raise ValueError("word checkpoint optimizer histories differ from its acquired step")
    if (state_digest(saved) != checkpoint["state_sha256"] or saved["fixture"]["api_state"]["completed_steps"] != prefix
            or saved["parent_task_fingerprint"] != task_fingerprint(parent_task)
            or saved["candidate_revision"] != request["candidate_revision"]
            or saved["fixture"]["recipe"] != fixture.recipe.to_dict()
            or saved["fixture"]["seed"] != request["protocol"]["seed"]):
        raise ValueError("word saved state differs from its recipe/task/candidate/prefix binding")
    context.streams.load_state_dict(saved["streams"])
    fixture.policy.load_state_dict(saved["fixture"]["api_state"])
    fixture.restore_component_transport(saved["fixture"].get("component_transport"))
    fixture.data_generator.set_state(saved["fixture"]["data_generator"].cpu())
    fixture.max_steps = saved["fixture"]["max_steps"]
    if state_digest(fixture.state_dict()) != state_digest(saved["fixture"]):
        raise ValueError("word fixture restore changed model/optimizer/policy/data state")
    if state_digest(context.streams.state_dict()) != state_digest(saved["streams"]):
        raise ValueError("word restore changed a named stream")
    return dict(prefix_steps=prefix, parent_confirmed_step=prefix, restored_exactly=True,
        history_reset=False, same_recipe_prior_architecture=True, parent_smoke_status="PASS",
        parent_checkpoint_sha256=file_hash(path), parent_state_sha256=state_digest(saved),
        parent_compatibility_key=expected, parent_candidate_revision=parent["candidate_revision"])


def run_ladder(request, task, output, device, *, context, execution_limit=None,
               capture_media=False, prerequisites=None, retain_scored_outputs=True):
    from benchmarks.toy_audit.api_images import WordFixture
    from .adapters import _event, _finite_tree, _host_receipt, _save_observer_outputs
    from .mechanisms import MechanismAudit, mechanism_blockers
    from .views import task_fingerprint
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("word acquisition/hold requires CUDA; no CPU fallback")
    updates = task["execution"]["steps"] if execution_limit is None else execution_limit
    if type(updates) is not int or not 1 <= updates <= task["execution"]["steps"]:
        raise ValueError("word execution limit must fit the declared allowance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = WordFixture(device=device, seed=request["protocol"]["seed"], recipe_name=None,
                          max_steps=SMOKE_STEPS, components=context)
    host = _host_receipt(task["execution"]["host_definition"], None, fixture.G, fixture.D)
    host["models"]["encoder"] = {"class": type(fixture.E).__module__ + "." + type(fixture.E).__qualname__,
        "parameters": sum(p.numel() for p in fixture.E.parameters()),
        "parameter_shapes": {name: list(p.shape) for name, p in fixture.E.named_parameters()},
        "initial_state_sha256": state_digest(fixture.E.state_dict())}
    host["initial_prior_sha256"] = state_digest(fixture.prior.state_dict())
    generated = context.streams.generator("eval", component="live", purpose="generated_words")
    reconstructed = context.streams.generator("eval", component="live", purpose="paired_reconstruction")
    confirmed = context.streams.generator("eval", component="live", purpose="word_confirmation")
    confirmed_reconstruction = context.streams.generator("eval", component="live", purpose="word_reconstruction_confirmation")
    continuity = None
    if task["evaluation"]["kind"] == HOLD_KIND:
        continuity = restore_parent(request, task, prerequisites, fixture, context)
    prefix = fixture.completed_steps
    fixture.max_steps = prefix + updates
    audit = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
    timing, observations, confirmations, records, rng_audits = PhaseTimer(), [], [], [], []
    checks = set(schedule(updates))
    finite, checkpoint = True, None
    started = time.monotonic()

    def observe(offset):
        nonlocal finite, checkpoint
        step = prefix + offset
        before = context.streams.audit()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(context.device).clone()
        training_before = state_digest(fixture.state_dict())
        with timing.measure("evaluation"):
            observed = fixture.observe(n=1024, generator=generated, reconstruction_generator=reconstructed)
            confirmation = fixture.observe(n=1024, generator=confirmed, reconstruction_generator=confirmed_reconstruction)
        training_after = state_digest(fixture.state_dict())
        for result in (observed, confirmation):
            result["metrics"]["reconstruction_exact"] = int(result["metrics"]["reconstruction_exact"])
            if any(result["metrics"][key] for key in ("served_averaged", "policy_latent_perturbation", "output_noise_added")):
                raise ValueError("word ladder departed from declared clean/live sampling")
            finite = finite and _finite_tree(result["metrics"])
        allowed = [key for key, binding in context.streams.manifest()["bindings"].items() if binding["family"] == "eval"]
        rng_audit = context.streams.compare(before, context.streams.audit(), allowed=allowed)
        rng_audit["unintended_rng_deviations"] += int(not torch.equal(cpu_rng, torch.get_rng_state()))
        rng_audit["unintended_rng_deviations"] += int(not torch.equal(cuda_rng, torch.cuda.get_rng_state(context.device)))
        rng_audits.append(rng_audit)
        row = dict(step=step, **observed["metrics"])
        confirm = dict(step=step, metrics=deepcopy(confirmation["metrics"]),
            primary_state_sha256=training_before, confirmed_state_sha256=training_after,
            training_state_unchanged=training_before == training_after,
            independent_stream=CONFIRMATION_STREAM, reconstruction_stream=RECONSTRUCTION_STREAM)
        if offset or task["evaluation"]["kind"] == HOLD_KIND:
            observations.append(row)
            confirmations.append(confirm)
        if capture_media or retain_scored_outputs:
            records.append(deepcopy(dict(step=step, **observed, confirmation=confirm,
                                         confirmation_views=confirmation["views"])))
        hit = not bounds(row) and not bounds(confirm["metrics"])
        if offset and task["evaluation"]["kind"] == SMOKE_KIND and checkpoint is None and hit:
            saved = dict(fixture=fixture.state_dict(), streams=context.streams.state_dict(),
                         parent_task_fingerprint=task_fingerprint(task), candidate_revision=request["candidate_revision"])
            root = output / "acquisition"
            root.mkdir(exist_ok=False)
            path = root / "acquired-state.pt"
            torch.save(saved, path)
            checkpoint = dict(path=path.name, sha256=file_hash(path), state_sha256=state_digest(saved),
                training_state_sha256=training_after, completed_steps=step, selection="earliest_confirmed_passing_state")
        _event("observation", task=task["id"], step=step, continuation_offset=offset,
               metrics=row, confirmation_metrics=confirm["metrics"], confirmed_pass=hit)

    if task["evaluation"]["kind"] == HOLD_KIND:
        observe(0)
    elif capture_media:
        with context.streams.preserve():
            observe(0)
    for offset in range(1, updates + 1):
        with timing.measure("training_updates"):
            update = fixture.step()
        finite = finite and _finite_tree(update)
        audit.observe_penalty(fixture.penalty.last_stats)
        if not finite:
            raise FloatingPointError("nonfinite joint word public-API update")
        if offset in checks:
            observe(offset)
        if time.monotonic() - started > task["resources"]["timeout_seconds"]:
            raise TimeoutError("word declared task allowance exceeded")
    torch.cuda.synchronize(device)
    state = fixture.state_dict()
    finite = finite and _finite_tree(state)
    def count(optimizer, parameters):
        values = [int(optimizer.state[p]["step"]) for p in parameters if "step" in optimizer.state.get(p, {})]
        return min(values) if values else 0
    counts = {role: count(opt, module.parameters()) for role, opt, module in (
        ("generator", fixture.opt_g, fixture.G), ("encoder", fixture.opt_g, fixture.E),
        ("prior", fixture.opt_g, fixture.prior), ("discriminator", fixture.opt_d, fixture.D))}
    mechanisms = audit.receipt()
    guards = dict(all_finite=finite, optimizer_updates=counts,
        hooks_exercised=fixture.opt_d.record.observed_steps == fixture.completed_steps and not mechanism_blockers(mechanisms),
        mechanism_audit=mechanisms, unintended_rng_deviations=sum(r["unintended_rng_deviations"] for r in rng_audits))
    evidence = dict(observations=observations, confirmations=confirmations, live=observations[-1], host=host,
        scoring_weights="live", completed_steps=fixture.completed_steps, executed_updates=updates,
        guards=guards, rng_audits=rng_audits, continuity=continuity,
        **executed_receipt(JOINT_WORDS_CLEAN, eval_output_noise="clean"))
    from .priors import recipe_owned_prior, prior_policy_receipt
    if recipe_owned_prior(task):
        evidence["prior_policy"] = prior_policy_receipt(context.recipe, fixture.prior)
    if fixture.transport is not None:
        evidence["component_transport"] = fixture.transport.state_dict()
    if checkpoint is not None:
        evidence.update(checkpoint=checkpoint, artifact_root=str((output / "acquisition").resolve()),
                        artifact_manifest=manifest_artifacts(output / "acquisition"))
    if retain_scored_outputs:
        evidence["saved_observer_outputs"] = _save_observer_outputs(output, "observed-records.pt", records,
                                                                   kind="scored_word_ladder_records_v1")
    final = dict(fixture=state, streams=context.streams.state_dict())
    torch.save(final, output / "state.pt")
    evidence["provenance_checkpoint"] = save_provenance_checkpoint(output, final, completed_steps=fixture.completed_steps)
    receipt = {**context.receipt(), "api_components": list(fixture.api_components), "evidence": evidence,
        "cost": dict(optimizer_updates=counts, completed_steps=fixture.completed_steps, new_optimizer_updates=updates,
                     adapter_loop_seconds=time.monotonic() - started, phase_timing=timing.snapshot()),
        "scope": "ordinary_full_task" if updates == task["execution"]["steps"] else "integration_demo_only",
        "declared_task_updates": task["execution"]["steps"], "execution_limit": updates}
    atomic_json(output / "adapter-receipt.json", receipt)
    return (receipt, records) if capture_media else receipt
