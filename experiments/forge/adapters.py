"""Task data/models/evidence around the single public GANTrainer update path.

Historical host optimizer constants are provenance, not a second formulation.
Adapters copy only resource/architecture/data definitions and call the frozen
evaluators. Their effective recipe and sampling law travel with every receipt.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .api import (CapabilityError, task_formulation_context, task_recipe_resources,
                  task_policy_blockers)
from .artifacts import manifest_artifacts, verify_artifacts
from .contracts import atomic_json, file_hash, read_json, stable_hash
from .initialization import task_initializer
from .mechanisms import MechanismAudit, mechanism_blockers
from .priors import task_prior
from .sampling import ENUMERATED_PRIOR_CLEAN, PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import state_digest
from .telemetry import PhaseTimer, normalize_adapter_costs
from .taskrecipes import bind_task_candidate


def _event(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True, allow_nan=False), flush=True)


def _context(request, task, device, resources):
    declared = task_recipe_resources(task)
    if declared and resources != declared:
        raise CapabilityError([f"{task['id']}: adapter resources differ from experiment-owned task resources"])
    return task_formulation_context(request["candidate"], task, request.get("protocol"), device=device)


def adapter_preflight(task, candidate, *, root=None):
    """Report unsupported task/host bindings before reserving training compute."""
    from .noisy_prior_tier1 import is_noisy_task as recognized_noisy
    if recognized_noisy(task):
        from .noisy_prior_adapters import blockers as original_noisy_blockers
        reasons = original_noisy_blockers(task, candidate, root=root)
        if reasons:
            return reasons
    try:
        task_prior(task)
        task_initializer(task, candidate)
        candidate = bind_task_candidate(candidate, task)
    except ValueError as error:
        return [str(error)]
    adapter = task["adapter"]
    from .atlas_existing_mog import is_candidate as existing_mog_candidate, blockers as existing_mog_blockers
    existing_mog = existing_mog_candidate(candidate)
    if existing_mog:
        reasons = existing_mog_blockers(task, candidate, root=root)
        if reasons:
            return reasons
    from .noisy_prior_tier1 import is_noisy_task, validate as validate_noisy_task
    noisy_task = is_noisy_task(task)
    if noisy_task:
        from .noisy_prior_adapters import blockers as noisy_blockers
        reasons = noisy_blockers(task, candidate, root=root)
        if reasons:
            return reasons
    if task.get("task_cohort") == "tier1_policy_selected_cloud_v1":
        from .tier1_policy import validate
        try:
            validate(task, root=root)
        except (ValueError, KeyError, OSError) as error:
            return [str(error)]
        blockers = task_policy_blockers(task, candidate)
        if blockers:
            return blockers
        if adapter == "transfer_behavior":
            from .policy_behavior_adapters import behavior_preflight
            return behavior_preflight(task, candidate)
        try:
            task_formulation_context(candidate, task, device="cpu", root=root)
        except (ValueError, KeyError) as error:
            return error.blockers if isinstance(error, CapabilityError) else [str(error)]
        return []
    if adapter == "word_joint":
        from .word_adapter import word_preflight
        return word_preflight(task, candidate, root=root)
    supported = {"transfer_behavior", "transfer_vector", "transfer_image", "native100",
                 "native100_continuation", "ring_endurance", "clockfree_audit", "paired_adaptation"}
    if adapter not in supported:
        return [f"no public adapter for {adapter}; implement its declared capability before training"]
    policy_blockers = task_policy_blockers(task, candidate)
    if policy_blockers:
        return policy_blockers
    if not noisy_task and not existing_mog and adapter == "transfer_behavior" and task["execution"].get("host") != "mode_hold":
        from .behavior_adapters import behavior_preflight
        blockers = behavior_preflight(task, candidate)
        if blockers:
            return blockers
    blockers = []
    from .paired_sampling import paired_sampling_blockers
    blockers.extend(paired_sampling_blockers(task))
    if adapter in {"native100", "native100_continuation"}:
        from .nativeprofiles import native_profile_blockers
        blockers.extend(native_profile_blockers(task, candidate, root=root))
        if blockers:
            return blockers
    if adapter == "transfer_vector":
        scorer = task["evaluation"].get("sample_evaluator")
        if scorer not in (None, "benchmarks.toy_audit.ring16_quality:score_samples",
                          "benchmarks.toy_audit.gaussian1d_quality:score_samples"):
            blockers.append(f"unsupported vector sample evaluator: {scorer}")
        from .vectorprofiles import vector_profile_blockers
        blockers.extend(vector_profile_blockers(task, root=root))
        spec = task["execution"].get("host_definition", {})
        scalar = spec.get("kind") == "gaussian_mixture" and spec.get("means") and len(spec["means"][0]) == 1
        if scalar and scorer != "benchmarks.toy_audit.gaussian1d_quality:score_samples":
            blockers.append("one-dimensional vector host requires the explicit Gaussian scalar scorer")
        if scorer == "benchmarks.toy_audit.gaussian1d_quality:score_samples":
            from benchmarks.toy_audit.gaussian1d_quality import score_samples
            try:
                score_samples(torch.zeros(1, 1), spec)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                blockers.append(f"unsupported scalar target: {error}")
        if blockers:
            return blockers
    if adapter == "transfer_image":
        from .imageprofiles import image_profile_blockers
        blockers.extend(image_profile_blockers(task, root=root))
        if blockers:
            return blockers
    if adapter == "clockfree_audit" and task["execution"].get("clock_audit_scope") != "measure_known_dependencies":
        from .clockfree import source_audit
        recipe = candidate.get("resolved_recipe")
        if recipe and "lr_floor" in recipe:
            blockers.extend(source_audit(recipe, candidate.get("extensions", {}))["unexplained_clock_dependencies"])
    # Planning, preflight and execution share the exact public binding path.
    try:
        task_formulation_context(candidate, task, device="cpu", root=root)
    except CapabilityError as error:
        blockers.extend(error.blockers)
    except ValueError as error:
        blockers.append(str(error))
    return blockers


def _models(context, spec):
    from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
    g = context.construct(lambda: SimpleMLPGenerator(context.recipe.z_dim, spec["hidden"], spec["layers"], 2), component="generator")
    d = context.construct(lambda: SimpleMLPDiscriminator(2, spec.get("d_hidden", spec["hidden"]),
                                                      spec.get("d_layers", spec["layers"]), spec["fourier"]), component="discriminator")
    return g.to(context.device), d.to(context.device)


def _host_receipt(spec, profile, generator, discriminator):
    return {"definition": deepcopy(spec), "profile": deepcopy(profile),
        "models": {name: {"class": type(model).__module__ + "." + type(model).__qualname__,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "parameter_shapes": {key: list(p.shape) for key, p in model.named_parameters()},
                          "initial_state_sha256": state_digest(model.state_dict())}
                   for name, model in (("generator", generator), ("discriminator", discriminator))}}


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class _Run:
    """Shared progress, state, measured optimizer counts and RNG audit."""
    def __init__(self, context, trainer, output, task, *, sampling_law=PUBLIC_PRIOR_CLEAN):
        self.context, self.trainer = context, trainer
        self.sampling_policy = executed_receipt(sampling_law, eval_output_noise="clean")
        self.output, self.task = Path(output), task
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.finite = True
        self.rng_audits = []
        self.last_update = {}
        self.policy_audit = None
        from .noisy_prior_tier1 import is_noisy_task
        self.noisy_task = is_noisy_task(task)
        self.existing_mog_task = getattr(context, "_existing_mog", False)
        self.policy_purity, self.policy_observations = [], []
        if context.policy_task is not None:
            from .policy_adapters import PolicyLifecycleAudit
            self.policy_audit = PolicyLifecycleAudit(trainer.policy)
            self.sampling_policy = executed_receipt(task["evaluation"]["sampling_law"],
                eval_output_noise=task["evaluation"]["eval_output_noise"])
        if self.existing_mog_task:
            from .atlas_existing_mog import initialize_vector_receipts
            initialize_vector_receipts(context, trainer, self.output, task)
        if self.noisy_task:
            from .noisy_prior_adapters import initialize_vector_receipts
            initialize_vector_receipts(context, trainer, self.output, task)
        self.mechanism_audit = MechanismAudit(context.recipe, trainer.opt_d, [trainer.opt_g])
        self.timing = PhaseTimer(synchronize=(lambda: torch.cuda.synchronize(context.device))
                                if context.device.type == "cuda" else None)
        # One scoring law covers live/EMA observations and evaluator-owned
        # holdouts. Clean removes training output noise, never the MoG kernel.
        original_sample = trainer.sample
        def measured_sample(*args, **kwargs):
            if kwargs.get("output_noise", False) is not False:
                raise ValueError("Forge scalar scoring requires clean samples without output noise")
            kwargs["output_noise"] = False
            with self.timing.measure("sampling"):
                return original_sample(*args, **kwargs)
        trainer.sample = measured_sample

    def step(self, real):
        before = self.trainer.completed_steps
        with self.timing.measure("training_updates"):
            self.last_update = self.trainer.step(real, collect_stats=True)
        if self.existing_mog_task:
            from .atlas_existing_mog import restore_live_owner
            restore_live_owner(self.trainer)
        if self.noisy_task:
            from .noisy_prior_adapters import restore_live_owner
            restore_live_owner(self.trainer)
        self.mechanism_audit.observe_penalty(self.last_update.get("penalty_stats", {}))
        if self.trainer.completed_steps != before + 1:
            raise RuntimeError("public trainer did not complete one update")
        self.finite = self.finite and _finite_tree(self.last_update)
        if not self.finite:
            raise FloatingPointError("nonfinite public training loss")

    def evaluate(self, function):
        policy_before = None
        if self.policy_audit is not None:
            from .policy_adapters import evaluation_state, typed_state_digest
            if self.existing_mog_task:
                from .atlas_existing_mog import live_evaluation_state
                policy_before = typed_state_digest(live_evaluation_state(self.context, self.trainer))
            elif self.noisy_task:
                from .noisy_prior_adapters import live_evaluation_state
                policy_before = typed_state_digest(live_evaluation_state(self.context, self.trainer))
            else:
                policy_before = typed_state_digest(evaluation_state(self.context.state_dict()))
        before = self.context.streams.audit()
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state(self.context.device).clone() if self.context.device.type == "cuda" else None
        with self.timing.measure("evaluation"):
            result = function()
        after = self.context.streams.audit()
        bindings = self.context.streams.manifest()["bindings"]
        allowed = [key for key, value in bindings.items() if value["family"] == "eval"]
        audit = self.context.streams.compare(before, after, allowed=allowed)
        if not torch.equal(cpu, torch.get_rng_state()):
            audit["unintended_rng_deviations"] += 1
            audit["unintended_streams"].append("global_cpu")
        if cuda is not None and not torch.equal(cuda, torch.cuda.get_rng_state(self.context.device)):
            audit["unintended_rng_deviations"] += 1
            audit["unintended_streams"].append("global_cuda")
        self.rng_audits.append(audit)
        if policy_before is not None:
            if self.existing_mog_task:
                from .atlas_existing_mog import observation_receipt, live_evaluation_state
                after_digest = typed_state_digest(live_evaluation_state(self.context, self.trainer))
            elif self.noisy_task:
                from .noisy_prior_adapters import observation_receipt, live_evaluation_state
                after_digest = typed_state_digest(live_evaluation_state(self.context, self.trainer))
            else:
                from .policy_adapters import observation_receipt
                after_digest = typed_state_digest(evaluation_state(self.context.state_dict()))
            pure = policy_before == after_digest
            self.policy_purity.append({"completed_steps": self.trainer.completed_steps,
                "before_sha256": policy_before, "after_sha256": after_digest, "pure": pure})
            self.policy_observations.append(observation_receipt(self.task, self.trainer.policy))
            if not pure:
                raise RuntimeError("selected-policy observation changed public training state")
        self.finite = self.finite and _finite_tree(result)
        return result

    def sample(self, n, *, ema=False):
        stream = self.context.streams.generator("eval", component="ema" if ema else "live", purpose="samples")
        return self.trainer.sample(n, ema=ema, generator=stream)

    def receipt(self, evidence, *, save_state=True):
        trainer = self.trainer
        state = self.context.state_dict()
        if self.policy_audit is not None:
            from .policy_adapters import finite_policy_state
            self.finite = self.finite and finite_policy_state(state)
        else:
            self.finite = self.finite and _finite_tree(state)
        # Count actual Adam state steps, rather than echoing the task budget.
        def updates(optimizer, parameters):
            observed = [int(optimizer.state[p]["step"]) for p in parameters
                        if p in optimizer.state and "step" in optimizer.state[p]]
            return min(observed) if observed else 0
        counts = {"generator": updates(trainer.opt_g, trainer.G.parameters()),
                  "discriminator": updates(trainer.opt_d, trainer.D.parameters()),
                  "prior": updates(trainer.opt_g, trainer.prior.parameters())}
        a2 = trainer.prior_mechanisms["a2"]
        mechanisms = self.mechanism_audit.receipt()
        hooks = trainer.opt_d.record.observed_steps == trainer.completed_steps and (
            not a2["requested"] or (a2["enabled"] and trainer.latent_history is not None)) and not mechanism_blockers(mechanisms)
        guards = {"all_finite": self.finite, "optimizer_updates": counts,
                  "hooks_exercised": hooks, "mechanism_audit": mechanisms,
                  "unintended_rng_deviations": sum(a["unintended_rng_deviations"] for a in self.rng_audits)}
        evidence = {**evidence, "guards": guards, "rng_audits": self.rng_audits,
                    **self.sampling_policy}
        if self.policy_audit is not None:
            from .policy_adapters import controls_receipt
            controls = controls_receipt(trainer.policy, trainer.completed_steps)
            if self.existing_mog_task:
                from .atlas_existing_mog import COHORT, vector_receipt
                controls["cohort"] = COHORT
                evidence["existing_mog717"] = vector_receipt(self.context, trainer, self.task)
            else:
                controls["cohort"] = self.task["task_cohort"]
            evidence.update(scoring_weights="live" if self.noisy_task or self.existing_mog_task else "state_selected", policy_controls=controls,
                policy_purity=self.policy_purity, policy_observations=self.policy_observations)
            guards["hooks_exercised"] = guards["hooks_exercised"] and controls["implementation_observed"]
        if evidence.get("artifact_root"):
            evidence["artifact_manifest"] = manifest_artifacts(evidence["artifact_root"])
            evidence["artifact_portability"] = {
                "storage": "local", "identity": "relative_paths_sizes_sha256",
                "requires_bulk_artifacts": True,
                "relocation": "copy the complete artifact tree and verify the unchanged manifest"}
        if save_state:
            torch.save(state, self.output / "state.pt")
        receipt = {"evidence": evidence, "cost": {"optimizer_updates": counts,
                   "completed_steps": trainer.completed_steps, "adapter_loop_seconds": time.monotonic() - self.started,
                   "phase_timing": self.timing.snapshot()},
                   **self.context.receipt()}
        atomic_json(self.output / "adapter-receipt.json", receipt)
        return receipt


def _checkpoints(task):
    return sorted({math.ceil(i * task["execution"]["steps"] / 24) for i in range(1, 25)})


def _save_observer_outputs(output, filename, records, *, kind):
    """Archive existing scored draws; serialization adds no observations."""
    path = Path(output) / filename
    torch.save(records, path)
    return {"path": filename, "sha256": file_hash(path), "bytes": path.stat().st_size,
            "observation_count": len(records), "kind": kind,
            "optimizer_updates_added": 0, "sampling_draws_added": 0}


def _vector(request, task, output, device, *, retain_scored_outputs=True):
    from benchmarks.transfer_suite.vector_tasks import sample_target, score_samples
    scorer = task["evaluation"].get("sample_evaluator")
    if scorer is not None:
        if scorer == "benchmarks.toy_audit.gaussian1d_quality:score_samples":
            from benchmarks.toy_audit.gaussian1d_quality import sample_target, score_samples
        elif scorer == "benchmarks.toy_audit.ring16_quality:score_samples":
            from benchmarks.toy_audit.ring16_quality import score_samples
        else:
            raise CapabilityError([f"unsupported vector sample evaluator: {scorer}"])
    from .vectorprofiles import build_vector_models, resolve_vector_spec
    spec = resolve_vector_spec(task)
    context = _context(request, task, device, {"num_particles": spec["particles"],
                       "z_dim": spec["z_dim"], "batch_size": spec["batch"]})
    from .atlas_existing_mog import admitted_vector_context
    admitted_vector_context(request, task, context, output, device)
    g, d = build_vector_models(context, spec)
    trainer = context.build_trainer(g, d)
    host_receipt = _host_receipt(spec, task["execution"].get("vector_profile"), g, d)
    spec["thresholds"] = task["evaluation"]["thresholds"]
    run = _Run(context, trainer, output, task)
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    observations, records = [], []
    checkpoints = set(_checkpoints(task))
    def evaluate():
        samples = run.sample(4096).cpu()
        if retain_scored_outputs:
            records.append({"step": step, "samples": samples.detach().clone()})
        return score_samples(samples, spec, step)
    for step in range(1, task["execution"]["steps"] + 1):
        run.step(sample_target(spec, context.recipe.batch_size, data, step - 1).to(device))
        if step in checkpoints:
            row = run.evaluate(evaluate)
            observations.append({"step": step, **row})
            _event("observation", task=task["id"], step=step, metrics=row)
    evidence = {"observations": observations, "live": observations[-1], "host": host_receipt}
    if retain_scored_outputs:
        evidence["saved_observer_outputs"] = _save_observer_outputs(
            output, "observed-samples.pt", records, kind="scored_vector_samples_v1")
    return run.receipt(evidence,
                       save_state=task["execution"].get("produces_state", False))


def _image(request, task, output, device):
    from benchmarks.transfer_suite.image_tasks import templates, image_metrics
    from .imageprofiles import build_image_models, resolve_image_spec
    spec = resolve_image_spec(task)
    context = _context(request, task, device, {"num_particles": spec["particles"],
                       "z_dim": spec["z_dim"], "batch_size": spec["batch_size"]})
    g, d = build_image_models(context, spec)
    trainer = context.build_trainer(g, d)
    run = _Run(context, trainer, output, task, sampling_law=ENUMERATED_PRIOR_CLEAN)
    host_receipt = _host_receipt(spec, task["execution"].get("image_profile"), g, d)
    centers = templates(spec).to(device)
    data = context.streams.generator("data", component="target", purpose="training")
    def evaluate():
        generated = trainer.sample(context.recipe.num_particles, fixed_first_n=True,
            generator=context.streams.generator("eval", component="live", purpose="enumerated_samples"))
        return image_metrics(generated, centers, task["evaluation"]["measurement"])
    observations = []
    checkpoints = set(_checkpoints(task))
    for step in range(1, task["execution"]["steps"] + 1):
        indices = torch.randint(len(centers), (context.recipe.batch_size,), device=device, generator=data)
        # Preserve the reference image host's clipped noisy-template law.
        real = (centers[indices] + spec["noise_std"] * torch.randn(
            (context.recipe.batch_size, 1, 8, 8), device=device, generator=data)).clamp(0., 1.)
        run.step(real)
        if step in checkpoints:
            row = run.evaluate(evaluate)
            observations.append({"step": step, **row})
            _event("observation", task=task["id"], step=step, metrics=row)
    return run.receipt({"observations": observations, "live": observations[-1], "host": host_receipt},
                       save_state=task["execution"].get("produces_state", False))


def _ring(request, task, output, device, *, endurance=False):
    from benchmarks.locked_shared.mode_hold import ring_means, sample_ring, diversity, SIGMA
    from benchmarks.legacy.locked_shared import LOCKED_SHARED
    from .views import _convergence_class
    context = _context(request, task, device, {"num_particles": LOCKED_SHARED.n_particles, "z_dim": 4, "batch_size": 128})
    trainer = context.build_trainer(*_models(context, {"hidden": 96, "layers": 3, "fourier": 3}),
                                    max_steps=task["execution"]["steps"])
    run = _Run(context, trainer, output, task)
    means = ring_means()
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    observations, dense = [], []
    checkpoints = set(_checkpoints(task))
    evaluation = task["evaluation"]
    gate = _convergence_class()(confirmation=evaluation["confirmation_checks"],
        settling_budget=evaluation["settling_budget"], hold_budget=evaluation["hold_budget"],
        start_step=evaluation["start_step"]) if endurance else None
    extension = 0
    for step in range(1, task["execution"]["steps"] + 1):
        run.step(sample_ring(means, context.recipe.batch_size, SIGMA, data).to(device))
        if (endurance and step > gate.start_step) or (not endurance and step in checkpoints):
            row = run.evaluate(lambda: diversity(run.sample(4096).cpu(), means))
            point = {"step": step, **row}
            if endurance:
                dense.append(point)
                if not gate.done:
                    gate.observe(point)
                else:
                    extension += 1
                if step % 100 == 0 or gate.done:
                    _event("observation", task=task["id"], step=step, metrics=row, gate=gate.status)
                if gate.done and (gate.status != "PASS" or extension >= evaluation["extension_steps"]):
                    break
            else:
                observations.append(point)
                _event("observation", task=task["id"], step=step, metrics=row)
    if endurance:
        evidence = {"dense": dense, "convergence": gate.summary(), "continuity": {
            "mode": "uninterrupted", "run_id": stable_hash({"output": str(Path(output).resolve()),
            "candidate": request.get("candidate_revision"), "task": task["id"]}),
            "original_schedule_horizon": context.recipe.total_steps,
            "max_total_steps": trainer.max_steps, "resume_count": 0}}
    else:
        evidence = {"observations": observations, "live": observations[-1]}
    return run.receipt(evidence)


def _native(request, task, output, device, *, prerequisites=None):
    from .paired_sampling import paired_sampling_blockers
    blockers = paired_sampling_blockers(task)
    if blockers:
        raise CapabilityError(blockers)
    from benchmarks.toy100.problems import sample_real
    from benchmarks.toy100.metrics import evaluate_samples
    from benchmarks.toy100.accuracy_evidence import AccuracyEvidence
    from benchmarks.toy100.train import evaluation_steps
    from .nativeprofiles import build_native_models, resolve_native_spec, validate_native_continuation
    problem = task["execution"]["problem"]
    evaluation = task["evaluation"]
    spec = resolve_native_spec(task)
    if task["execution"].get("continuation_of"):
        validate_native_continuation(request["tasks"][task["execution"]["continuation_of"]], task)
    context = _context(request, task, device, spec["resources"] if spec else task["execution"]["resources"])
    g, d = build_native_models(context, spec) if spec else _models(context, task["execution"]["model"])
    trainer = context.build_trainer(g, d,
        max_steps=context.recipe.total_steps if task["execution"].get("preserve_prefix_steps") else task["execution"]["steps"])
    host_receipt = _host_receipt(spec, task["execution"]["native_profile"], g, d) if spec else None
    run = _Run(context, trainer, output, task)
    artifact_root = Path(output) / "native100"
    directory = artifact_root / problem
    prefix = None
    reference = None
    elapsed_offset = 0.0
    prefix_steps = task["execution"].get("preserve_prefix_steps", 0)
    if prefix_steps:
        parent = (prerequisites or {}).get(task["execution"]["continuation_of"])
        if not parent or parent.get("candidate_revision") != request.get("candidate_revision"):
            raise CapabilityError(["continuation requires this candidate's certified prerequisite attempt"])
        expected = next(job["compatibility_key"] for job in request["jobs"]
                        if task["execution"]["continuation_of"] in job.get("task_ids", [job["task_id"]]))
        if parent["compatibility_key"] != expected or parent["result"]["gate_status"] != "PASS":
            raise CapabilityError(["continuation prerequisite is not a compatible passing own-state result"])
        old = parent["result"]["evidence"]
        verify_artifacts(old["artifact_root"], old["artifact_manifest"])
        if old["checkpoint"]["path"] not in old["artifact_manifest"]["files"]:
            raise ValueError("prerequisite checkpoint is not certified by its artifact manifest")
        retained = artifact_root / "prefix"
        shutil.copytree(old["artifact_root"], retained)
        verify_artifacts(retained, old["artifact_manifest"])
        checkpoint = retained / old["checkpoint"]["path"]
        if file_hash(checkpoint) != old["checkpoint"]["sha256"]:
            raise ValueError("continuation checkpoint bytes differ from prerequisite")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["trainer"]["completed_steps"] != prefix_steps or state_digest(state) != old["checkpoint"]["state_sha256"]:
            raise ValueError("continuation checkpoint does not represent the exact original prefix")
        context.load_state_dict(state)
        restored = context.state_dict()
        if state_digest(restored) != state_digest(state):
            raise ValueError("public restore changed prefix state")
        torch.save(restored, artifact_root / "resume-restored.pt")
        prefix = {"steps": prefix_steps, "reference_sha256": state_digest(state),
                  "continued_sha256": state_digest(restored), "prerequisite": parent,
                  "artifact_manifest": old["artifact_manifest"], "checkpoint": old["checkpoint"]}
        trainer.extend_execution(task["execution"]["steps"])
        shutil.copytree(retained / problem, directory)
        prefix_events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        elapsed_offset = max(event["elapsed"] for event in prefix_events if event.get("event") == "eval")
        with np.load(directory / "quality_checks" / f"step_{prefix_steps:06d}.npz", allow_pickle=False) as archive:
            reference = torch.from_numpy(archive["target"].copy()).to(device)
    (directory / "snapshots").mkdir(parents=True, exist_ok=True)
    config = {"problem": problem, "steps": task["execution"]["steps"], "device": str(device),
              "seed": request["protocol"]["seed"], "eval_interval": evaluation["eval_interval"],
              "early_eval_steps": evaluation["early_eval_steps"], "eval_samples": evaluation["eval_samples"],
              "snapshot_samples": min(4096, evaluation["eval_samples"]),
              "eval_output_noise": "clean",
              "forge_recipe": context.recipe.to_dict(), "forge_prior": context.prior_config}
    atomic_json(directory / "config.json", config)
    steps = evaluation_steps(config["steps"], config["eval_interval"], config["early_eval_steps"])
    if reference is None:
        reference = sample_real(problem, config["eval_samples"], device=device,
            generator=context.streams.generator("eval", component="target", purpose="reference"))
    accuracy = AccuracyEvidence(config, directory, steps, reference)
    from .paired_sampling import FIELD, PairedOutputNoiseEvidence
    paired = (PairedOutputNoiseEvidence(context, config, artifact_root / "paired-output-noise", steps, reference)
              if FIELD in evaluation else None)
    data = context.streams.generator("data", component="target", purpose="training")
    final_metrics, final_arrays = {}, {}
    def evaluate(step):
        arrays, scores = {}, {}
        for model in ("live", "ema"):
            draw = run.sample(config["eval_samples"], ema=model == "ema")
            metrics = evaluate_samples(draw, problem)
            fidelity = accuracy.observe(step, model, draw, metrics)
            event = {"event": "eval", "model": model, "step": step,
                     "elapsed": elapsed_offset + time.monotonic() - run.started, "metrics": metrics, "accuracy": fidelity}
            with (directory / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            if paired is not None:
                paired.observe(step, model, draw, event["elapsed"])
            arrays[model] = draw.detach().cpu().numpy()
            scores[model] = metrics
            if model == "live":
                _event("observation", task=task["id"], step=step, metrics=metrics, accuracy=fidelity)
        arrays["target"] = reference.cpu().numpy()
        np.savez_compressed(directory / "snapshots" / f"step_{step:06d}.npz",
                            **{key: value[:config["snapshot_samples"]] for key, value in arrays.items()})
        return scores, arrays
    if not prefix_steps:
        final_metrics, final_arrays = run.evaluate(lambda: evaluate(0))
    for step in range(prefix_steps + 1, config["steps"] + 1):
        run.step(sample_real(problem, context.recipe.batch_size, device=device, generator=data))
        if step in steps:
            final_metrics, final_arrays = run.evaluate(lambda: evaluate(step))
    np.savez_compressed(directory / "final_samples.npz", **final_arrays)
    declaration, holdout = run.evaluate(lambda: accuracy.finish(trainer))
    summary = {"status": "complete", "problem": problem, "config": config,
               "eval_output_noise": "clean",
               "budget_steps": config["steps"], "completed_steps": trainer.completed_steps,
               "eval_steps": steps, "snapshot_steps": steps, "final_samples_file": "final_samples.npz",
               "final": final_metrics, "accuracy": declaration, "holdout": holdout}
    atomic_json(directory / "summary.json", summary)
    paired_receipt = run.evaluate(lambda: paired.finish(summary, directory)) if paired is not None else None
    saved = context.state_dict()
    state_path = directory / "training-state.pt"
    torch.save(saved, state_path)
    evidence = {"artifact_root": str(artifact_root.resolve()), "problem": problem,
                "checkpoint": {"path": str(state_path.relative_to(artifact_root)),
                    "sha256": file_hash(state_path), "state_sha256": state_digest(saved)},
                "artifact_schema": "native100_coverage_accuracy_v1",
                "holdout_rng": "frozen_accuracy_evidence_seed_offsets_1601_1602_1603"}
    if host_receipt:
        evidence["host"] = host_receipt
    if paired_receipt is not None:
        evidence["paired_sampling_diagnostic"] = paired_receipt
    if prefix:
        evidence["prefix_parity"] = prefix
    return run.receipt(evidence, save_state=False)


def _dispatch_task(request: dict, job: dict, output_dir: Path, device: str) -> dict:
    """Dispatch frozen task definitions; unsupported capabilities fail before work."""
    task = request["tasks"][job["task_id"]]
    adapter = task["adapter"]
    from .noisy_prior_tier1 import is_noisy_task
    if is_noisy_task(task):
        from .noisy_prior_adapters import blockers as noisy_blockers
        reasons = noisy_blockers(task, request["candidate"], root=request["source"]["snapshot_path"])
        if reasons:
            raise CapabilityError(reasons)
        if task["execution"].get("host") == "ae_gan_hold":
            from .atlas_noisy_ae import run_behavior as run_noisy_ae
            return run_noisy_ae(request, task, output_dir, device)
    from .atlas_existing_mog import is_candidate as existing_mog_candidate, blockers as existing_mog_blockers
    if existing_mog_candidate(request["candidate"]):
        reasons = existing_mog_blockers(task, request["candidate"], root=request["source"]["snapshot_path"])
        if reasons:
            raise CapabilityError(reasons)
        if task["id"] == "ae_gan_hold":
            from .atlas_existing_mog_ae import run_behavior as run_existing_mog_ae
            return run_existing_mog_ae(request, task, output_dir, device)
    if adapter == "word_joint":
        from .word_adapter import run_word
        return run_word(request, task, output_dir, device)
    if adapter == "transfer_behavior":
        from .atlas_two_pole import supports, run_behavior as run_atlas_two_pole
        if supports(task, request["candidate"]):
            result = run_atlas_two_pole(request, task, output_dir, device)
            if existing_mog_candidate(request["candidate"]):
                from .atlas_existing_mog import direct_control_receipt
                result["evidence"]["existing_mog717"] = direct_control_receipt(task, result)
            return result
        if task.get("task_cohort") == "tier1_policy_selected_cloud_v1":
            from .policy_behavior_adapters import run_behavior
            return run_behavior(request, task, output_dir, device)
        if task["execution"].get("host") == "mode_hold":
            return _ring(request, task, output_dir, device)
        from .behavior_adapters import run_behavior
        return run_behavior(request, task, output_dir, device)
    if adapter == "transfer_vector":
        return _vector(request, task, output_dir, device)
    if adapter == "transfer_image":
        return _image(request, task, output_dir, device)
    if adapter == "native100":
        return _native(request, task, output_dir, device)
    if adapter == "native100_continuation":
        return _native(request, task, output_dir, device, prerequisites=job.get("prerequisites"))
    if adapter == "clockfree_audit":
        from .clockfree import run_clockfree
        return run_clockfree(request, task, output_dir, device)
    if adapter == "paired_adaptation":
        from .adaptation import run_adaptation
        return run_adaptation(request, task, output_dir, device)
    if adapter == "ring_endurance":
        raw = _ring(request, task, output_dir, device, endurance=True)
        ids = job.get("task_ids", [task["id"]])
        return {"task_results": {name: deepcopy(raw) for name in ids}, "cost": raw["cost"]} if len(ids) > 1 else raw
    raise CapabilityError([f"no public adapter for {adapter}; implement and validate the declared capability before training"])


def run_task(request: dict, job: dict, output_dir: Path, device: str) -> dict:
    result = normalize_adapter_costs(_dispatch_task(request, job, output_dir, device))
    # Normalize the common diagnostic receipt too. The certified bulk evidence
    # lives in separate artifact roots and is not modified by timing annotation.
    receipt = Path(output_dir) / "adapter-receipt.json"
    if receipt.is_file():
        atomic_json(receipt, normalize_adapter_costs(read_json(receipt)))
    return result
