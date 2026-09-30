"""Measured clock perturbations on the public trainer, with saved state proof.

Passing a short probe is only one eligibility check. The declared long
continuations remain required to expose state-dependent delayed failures.
"""
from copy import deepcopy
from pathlib import Path
import time

import torch

from .artifacts import manifest_artifacts, verify_artifacts
from .contracts import atomic_json, file_hash, stable_hash
from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import require_consistent_rng, require_optimizer_steps, state_digest


ALLOWED_STATE = ["parameters_and_ema", "optimizer_moments_and_bias_correction",
                 "observed_gradients_and_row_visit_history", "named_training_rng"]


def source_audit(recipe, extensions):
    """Conservative declaration for the shipped scalar API; unknown hooks block."""
    dependencies = []
    if recipe["lr_floor"] != 1 or recipe["network_lr_floor"] not in (None, 1):
        dependencies.append("learning-rate annealing depends on completed steps and horizon")
    if recipe["input_noise_std"]:
        dependencies.append("input-noise annealing depends on completed steps and horizon")
    if recipe["output_noise_std"] and recipe["output_noise_warmup"]:
        dependencies.append("output-noise warmup depends on completed steps and horizon")
    if recipe["reg_coeff"] and recipe["reg_every"] != 1:
        dependencies.append("lazy critic penalty uses a periodic update counter")
    if recipe["d_guard_ratio"] and recipe["d_guard_min_steps"]:
        dependencies.append("critic guard releases at a fixed minimum update count")
    if extensions:
        dependencies.append("additional formulation bindings need an explicit clock/state audit")
    root = Path(__file__).resolve().parents[2]
    files = ["particlegan/training.py", "particlegan/recipes.py", "particlegan/k3p.py",
             "particlegan/grad_regularizers.py", "experiments/forge/clockfree.py",
             "experiments/forge/api.py", "experiments/forge/rng.py"]
    return {"source_sha256": {name: file_hash(root / name) for name in files},
            "allowed_state": ALLOWED_STATE, "unexplained_clock_dependencies": dependencies,
            "scope": "shipped public scalar trainer and explicitly listed state; no arbitrary-extension certification"}


def learning_state(state):
    """Exclude only external clocks/configuration and independent eval streams."""
    trainer = state["trainer"]
    stream_names = state["streams"]["manifest"]["bindings"]
    named = {key: value for key, value in state["streams"]["states"].items()
             if stream_names[key]["family"] != "eval"}
    return {"trainer": {key: value for key, value in trainer.items()
                        if key not in {"recipe", "completed_steps", "max_steps", "streams"}},
            "training_rng": named}


def _comparisons(proof):
    initial = learning_state(proof["initial"])
    common = {"permitted_state_sha256": state_digest(initial),
              "rng_state_sha256": state_digest(initial["training_rng"])}
    reference = state_digest([learning_state(s) for s in proof["trajectories"]["reference"]])
    return [{"condition": name, **common, "reference_sha256": reference,
             "changed_sha256": state_digest([learning_state(s) for s in proof["trajectories"][name]])}
            for name in ("step_label", "horizon", "evaluation_cadence", "restart")]


def run_clockfree(request, task, output, device):
    from .adapters import _context, _models, _event

    spec = task["execution"]
    probe_steps, warmup = spec["probe_steps"], spec["warmup_steps"]
    offset = spec["step_label_offset"]
    if any(type(n) is not int or n < 1 for n in (probe_steps, warmup, offset, spec["perturbed_horizon"])):
        raise ValueError("clock probes need positive declared steps, offset and horizon")
    maximum = offset + warmup + probe_steps + 1
    started = time.monotonic()

    def construct():
        context = _context(request, task, device, spec["resources"])
        trainer = context.build_trainer(*_models(context, spec["model"]), max_steps=maximum)
        # Bind every probe stream before the checkpoint, including eval-only names.
        data = context.streams.generator("data", component="clock_probe", purpose="training")
        context.streams.generator("eval", component="clock_probe", purpose="samples")
        return context, trainer, data

    def update(context, trainer, data):
        real = torch.randn(context.recipe.batch_size, 2, device=device, generator=data)
        trainer.step(real)

    context, trainer, data = construct()
    for _ in range(warmup):
        update(context, trainer, data)
    initial = context.state_dict()
    proof = {"schema_version": 1, "execution": deepcopy(spec), "initial": initial,
             "recipe": context.recipe.to_dict(), "extensions": context.extension_values,
             "trajectories": {}, "branch_initial": {}}
    directory = Path(output) / "clockfree-proof"
    directory.mkdir(parents=True)
    torch.save(initial, directory / "initial.pt")
    for name in ("reference", "step_label", "horizon", "evaluation_cadence", "restart"):
        if name != "reference":
            context, trainer, data = construct()
            restored = (torch.load(directory / "initial.pt", map_location=device, weights_only=True)
                        if name == "restart" else deepcopy(initial))
            context.load_state_dict(restored)
        proof["branch_initial"][name] = context.state_dict()
        if name == "step_label":
            trainer.completed_steps += offset
        elif name == "horizon":
            context.recipe = context.recipe.replace(total_steps=spec["perturbed_horizon"])
            trainer.recipe = context.recipe
        trajectory = []
        for _ in range(probe_steps):
            if name == "evaluation_cadence":
                generator = context.streams.generator("eval", component="clock_probe", purpose="samples")
                trainer.sample(spec["evaluation_samples"], generator=generator, output_noise=False)
            update(context, trainer, data)
            trajectory.append(context.state_dict())
        proof["trajectories"][name] = trajectory
        _event("clock_probe", task=task["id"], condition=name, updates=probe_steps)
    torch.save(proof, directory / "comparisons.pt")
    evidence = {**executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
                "comparisons": _comparisons(proof),
                "source_audit": source_audit(proof["recipe"], proof["extensions"]),
                "artifact_root": str(directory.resolve()), "artifact_manifest": manifest_artifacts(directory),
                "artifact_portability": {"storage": "local", "requires_bulk_artifacts": True}}
    result = {"evidence": evidence, "cost": {"training_seconds": time.monotonic() - started,
              "completed_updates": warmup + 5 * probe_steps}, "execution_path": "public_trainer",
              "recipe": proof["recipe"]}
    atomic_json(Path(output) / "adapter-receipt.json", result)
    return result


def verify_probe(task, evidence):
    """Recompute state comparisons from certified tensors in the frozen evaluator."""
    root = Path(evidence["artifact_root"])
    verify_artifacts(root, evidence["artifact_manifest"])
    proof = torch.load(root / "comparisons.pt", map_location="cpu", weights_only=True)
    if proof.get("schema_version") != 1 or proof.get("execution") != task["execution"]:
        raise ValueError("clock probe execution differs from its task")
    names = {"reference", "step_label", "horizon", "evaluation_cadence", "restart"}
    if set(proof["trajectories"]) != names or set(proof["branch_initial"]) != names:
        raise ValueError("clock probe must preserve every comparison branch")
    initial_hash = state_digest(proof["initial"])
    if proof["initial"]["trainer"]["completed_steps"] != task["execution"]["warmup_steps"]:
        raise ValueError("clock initial state is not at the declared warmup")
    require_optimizer_steps(proof["initial"], task["execution"]["warmup_steps"])
    require_consistent_rng(proof["initial"])
    if (proof["recipe"] != proof["initial"]["recipe"]
            or proof["recipe"] != proof["initial"]["trainer"]["recipe"]
            or proof["extensions"] != proof["initial"]["extensions"]):
        raise ValueError("clock source audit formulation differs from the measured initial state")
    if initial_hash != state_digest(torch.load(root / "initial.pt", map_location="cpu", weights_only=True)):
        raise ValueError("serialized restart differs from the measured initial state")
    for name in names:
        if state_digest(proof["branch_initial"][name]) != initial_hash:
            raise ValueError("clock comparison did not start from identical own state")
        trajectory = proof["trajectories"][name]
        if len(trajectory) != task["execution"]["probe_steps"]:
            raise ValueError("clock comparison lacks its complete common prefix")
        offset = task["execution"]["step_label_offset"] if name == "step_label" else 0
        for index, state in enumerate(trajectory, 1):
            require_optimizer_steps(state, task["execution"]["warmup_steps"] + index)
            require_consistent_rng(state)
            if state["trainer"]["completed_steps"] != task["execution"]["warmup_steps"] + offset + index:
                raise ValueError("clock perturbation labels do not match the declared probe")
            horizon = task["execution"]["perturbed_horizon"] if name == "horizon" else task["execution"]["original_schedule_horizon"]
            expected_recipe = {**proof["recipe"], "total_steps": horizon}
            if (state["recipe"] != expected_recipe or state["trainer"]["recipe"] != expected_recipe
                    or state["extensions"] != proof["extensions"]
                    or state["prior"] != proof["initial"]["prior"]
                    or state["initializer"] != proof["initial"]["initializer"]
                    or state["initialization"] != proof["initial"]["initialization"]):
                raise ValueError("clock perturbation changed formulation beyond its declared horizon")
    comparisons = _comparisons(proof)
    audit = source_audit(proof["recipe"], proof["extensions"])
    if stable_hash(comparisons) != stable_hash(evidence.get("comparisons")) or audit != evidence.get("source_audit"):
        raise ValueError("clock verdict declarations disagree with the saved state or source audit")
    verify_artifacts(root, evidence["artifact_manifest"])
    return comparisons, audit
