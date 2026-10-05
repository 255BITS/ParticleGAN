"""Forge binding of the retained joint BiGAN host, not a second trainer.

WordFixture owns the existing G/E/joint-D update and score_words. Forge injects
the candidate recipe, finite prior and named streams, then records the same
24-point sustained-quality evidence used by the shared transfer grader.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from .api import CapabilityError, task_formulation_context
from .contracts import atomic_json, file_hash
from .sampling import JOINT_WORDS_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer


RESOURCES = {"num_particles": 5, "z_dim": 2, "batch_size": 256}
THRESHOLDS = [["sample_count", ">=", 1024], ["quality_fraction", ">=", .95],
              ["modes", "==", 5], ["mass_tv", "<=", .10],
              ["reconstruction_exact", "==", 1],
              ["minimum_reconstruction_token_probability", ">=", .90]]


def validate_word_task(task, *, root=None):
    from benchmarks.toy_audit.api_images import WORDS, WORD_CHARS, WORD_LENGTH, word_oracle_controls
    execution, evaluation = task["execution"], task["evaluation"]
    expected = {"words": list(WORDS), "characters": WORD_CHARS, "length": WORD_LENGTH,
                "generator_widths": [2, 64, 128, 168], "encoder_widths": [168, 128, 64, 2],
                "joint_critic_widths": [170, 256, 128, 1], "resources": RESOURCES}
    if execution.get("host_definition") != expected:
        raise ValueError("word host definition differs from the shared retained fixture")
    if execution.get("execution_path") != "public_components":
        raise ValueError("joint word host requires its explicit public_components execution path")
    if (execution.get("steps") != 20001 or execution.get("original_schedule_horizon") != 20000
            or execution.get("produces_state") is not False):
        raise ValueError("word task requires 20,001 updates / 20,000-update schedule; no continuation claim")
    prior = execution.get("prior", {})
    if any(prior.get(key) != value for key, value in
           {"kind": "particle_cloud", "sigma": 0., "standardize": False, "learnable": True}.items()):
        raise ValueError("joint word task requires its explicit five-row learned particle-cloud exception")
    if not isinstance(prior.get("exception_reason"), str) or not prior["exception_reason"].strip():
        raise ValueError("finite-word particle-cloud exception needs its reason")
    if evaluation.get("thresholds") != THRESHOLDS or evaluation.get("eval_samples") != 1024:
        raise ValueError("word numerical bounds differ from the retained shared scorer")
    for name, control in word_oracle_controls().items():
        if control["passed"] != control["expected_pass"]:
            raise ValueError(f"word scorer control failed: {name}")
    if root is not None:
        for path, expected_hash in evaluation["sources"].items():
            if file_hash(Path(root) / path) != expected_hash:
                raise ValueError(f"word source binding differs: {path}")


def word_context(request, task, device, *, root=None):
    validate_word_task(task, root=root)
    context = task_formulation_context(request["candidate"], task, request.get("protocol"), device=device, root=root)
    if context.initializer != "deterministic_orthogonal":
        raise CapabilityError(["joint word host requires named deterministic orthogonal initialization"])
    if context.recipe.model != "gan" or context.recipe.conditioning != "scalar" or context.recipe.encoder_mode != "none":
        raise CapabilityError(["joint word fixture binds its own encoder and joint adversarial objective; recipe objective incompatible"])
    return context


def word_preflight(task, candidate, *, root=None):
    try:
        word_context({"candidate": candidate, "protocol": {"seed": 0}}, task, "cpu", root=root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return error.blockers if isinstance(error, CapabilityError) else [str(error)]
    return []


def run_word(request, task, output, device, *, execution_limit=None, capture_media=False,
             retain_scored_outputs=True):
    """Bounded API execution. A reduced cap is only an explicit integration demo."""
    from benchmarks.toy_audit.api_images import WordFixture
    from .adapters import _event, _finite_tree, _host_receipt, _save_observer_outputs
    from .mechanisms import MechanismAudit, mechanism_blockers

    context = word_context(request, task, device)
    steps = task["execution"]["steps"] if execution_limit is None else execution_limit
    if type(steps) is not int or not 1 <= steps <= task["execution"]["steps"]:
        raise ValueError("word execution limit must fit the preregistered task budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = WordFixture(device=device, seed=request["protocol"]["seed"], recipe_name=None,
                          max_steps=steps, components=context)
    host = _host_receipt(task["execution"]["host_definition"], None, fixture.G, fixture.D)
    host["models"]["encoder"] = {"class": type(fixture.E).__module__ + "." + type(fixture.E).__qualname__,
        "parameters": sum(parameter.numel() for parameter in fixture.E.parameters()),
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in fixture.E.named_parameters()},
        "initial_state_sha256": state_digest(fixture.E.state_dict())}
    host["initial_prior_sha256"] = state_digest(fixture.prior.state_dict())
    audit = MechanismAudit(context.recipe, fixture.opt_d, [fixture.opt_g])
    timing, observations, records, rng_audits = PhaseTimer(), [], [], []
    generated = context.streams.generator("eval", component="live", purpose="generated_words")
    reconstructed = context.streams.generator("eval", component="live", purpose="paired_reconstruction")
    # Media adds the genuine initial state; it cannot remove any metric check.
    checks = {math.ceil(index * steps / 24) for index in range(1, 25)}
    from .optimizer_diagnostics import attach
    optimizer_diagnostics = attach({"G": fixture.opt_g, "D": fixture.opt_d}, fixture.D,
        output, checks, prior=fixture.prior)
    started, finite = time.monotonic(), True
    for step in range(steps + 1):
        if step:
            with timing.measure("training_updates"):
                update = fixture.step()
            finite = finite and _finite_tree(update)
            audit.observe_penalty(fixture.penalty.last_stats)
            if not finite:
                raise FloatingPointError("nonfinite joint word public-API update")
        if step not in checks and not (capture_media and step == 0):
            continue
        before = context.streams.audit()
        global_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(context.device).clone() if context.device.type == "cuda" else None
        with timing.measure("evaluation"):
            observed = fixture.observe(n=task["evaluation"]["eval_samples"],
                                       generator=generated, reconstruction_generator=reconstructed)
        metrics = deepcopy(observed["metrics"])
        metrics["reconstruction_exact"] = int(metrics["reconstruction_exact"])
        if metrics["served_averaged"] or metrics["policy_latent_perturbation"] or metrics["output_noise_added"]:
            raise CapabilityError(["joint word observation departed from declared clean/live sampling"])
        finite = finite and _finite_tree(metrics)
        allowed = [key for key, binding in context.streams.manifest()["bindings"].items()
                   if binding["family"] == "eval"]
        rng_audit = context.streams.compare(before, context.streams.audit(), allowed=allowed)
        rng_audit["unintended_rng_deviations"] += int(not torch.equal(global_before, torch.get_rng_state()))
        if cuda_before is not None:
            rng_audit["unintended_rng_deviations"] += int(not torch.equal(cuda_before, torch.cuda.get_rng_state(context.device)))
        rng_audits.append(rng_audit)
        row = {"step": step, **metrics}
        if step:
            observations.append(row)
        if capture_media or retain_scored_outputs:
            records.append(deepcopy({"step": step, **observed}))
        _event("observation", task=task["id"], step=step, metrics=metrics,
               metric_passed=observed["passed"], failed_bounds=observed["failed_bounds"])

    state = fixture.state_dict()
    finite = finite and _finite_tree(state)
    def count(optimizer, parameters):
        values = [int(optimizer.state[parameter]["step"]) for parameter in parameters
                  if "step" in optimizer.state.get(parameter, {})]
        return min(values) if values else 0
    counts = {role: count(optimizer, module.parameters()) for role, optimizer, module in
              (("generator", fixture.opt_g, fixture.G), ("encoder", fixture.opt_g, fixture.E),
               ("prior", fixture.opt_g, fixture.prior), ("discriminator", fixture.opt_d, fixture.D))}
    mechanisms = audit.receipt()
    guards = {"all_finite": finite, "optimizer_updates": counts,
              "hooks_exercised": fixture.opt_d.record.observed_steps == fixture.completed_steps and not mechanism_blockers(mechanisms),
              "mechanism_audit": mechanisms,
              "unintended_rng_deviations": sum(row["unintended_rng_deviations"] for row in rng_audits)}
    receipt = {**context.receipt(), "api_components": list(fixture.api_components),
        "evidence": {"observations": observations, "live": observations[-1], "host": host,
            "scoring_weights": "live", "guards": guards, "rng_audits": rng_audits,
            **executed_receipt(JOINT_WORDS_CLEAN, eval_output_noise="clean")},
        "cost": {"optimizer_updates": counts, "completed_steps": fixture.completed_steps,
                 "adapter_loop_seconds": time.monotonic() - started, "phase_timing": timing.snapshot()},
        "scope": "ordinary_full_task" if steps == task["execution"]["steps"] else "integration_demo_only",
        "declared_task_updates": task["execution"]["steps"], "execution_limit": steps}
    if optimizer_diagnostics is not None:
        receipt["evidence"]["optimizer_diagnostics"] = optimizer_diagnostics.receipt()
    if retain_scored_outputs:
        receipt["evidence"]["saved_observer_outputs"] = _save_observer_outputs(
            output, "observed-records.pt", [record for record in records if record["step"]],
            kind="scored_word_records_v1")
    # Durable bulk state remains local; it is not a continuation qualification.
    torch.save({"fixture": state, "streams": context.streams.state_dict()}, output / "state.pt")
    atomic_json(output / "adapter-receipt.json", receipt)
    return (receipt, records) if capture_media else receipt
