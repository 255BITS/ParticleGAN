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
    if recipe.get("continuous_policy") is not None:
        dependencies.append("continuous policy lifecycle needs a separate reviewed clock/state audit")
    elif recipe["lr_floor"] != 1 or recipe["network_lr_floor"] not in (None, 1):
        dependencies.append("learning-rate annealing depends on completed steps and horizon")
    if recipe["input_noise_std"]:
        dependencies.append("input-noise annealing depends on completed steps and horizon")
    if recipe["output_noise_std"] and recipe["output_noise_warmup"]:
        dependencies.append("output-noise warmup depends on completed steps and horizon")
    if recipe["reg_coeff"] and recipe["reg_every"] != 1:
        dependencies.append("lazy critic penalty uses a periodic update counter")
    if recipe["reg_coeff"] and recipe.get("reg_arm") is None and recipe.get("critic_formulation", "ka2") == "ka2":
        dependencies.append("KA2 switches from pure A to blended penalty at call 800")
    if recipe["d_guard_ratio"] and recipe["d_guard_min_steps"]:
        dependencies.append("critic guard releases at a fixed minimum update count")
    if extensions:
        dependencies.append("additional formulation bindings need an explicit clock/state audit")
    root = Path(__file__).resolve().parents[2]
    files = ["particlegan/training.py", "particlegan/recipes.py", "particlegan/k3p.py", "particlegan/ka2.py",
             "particlegan/policy.py", "particlegan/continuous.py",
             "particlegan/grad_regularizers.py", "experiments/forge/clockfree.py",
             "experiments/forge/api.py", "experiments/forge/rng.py"]
    return {"source_sha256": {name: file_hash(root / name) for name in files},
            "allowed_state": ALLOWED_STATE, "unexplained_clock_dependencies": dependencies,
            "scope": "shipped public scalar trainer and explicitly listed state; no arbitrary-extension certification"}


# --- Proposed audit revision (needs user/reviewer sign-off; opt-in per task) ---
# A task opts in with evaluation.audit_rule == MEASURED_V2. Every task without
# that field keeps the v1 rule above unchanged. See
# reports/forge/dig2-20261016/clock-audit-measured.md for the written rule.
MEASURED_V2 = "measured_parity_v2"
COUNTER_POLICIES = ("measured", "declared")
# Class S: step/horizon schedule declarations. A passing measurement overrides
# one only when the probe had power to see a step/horizon-driven version of it.
STEP_SCHEDULE_ENTRIES = {
    "learning-rate annealing depends on completed steps and horizon": "learning_rate",
    "input-noise annealing depends on completed steps and horizon": "input_noise",
    "output-noise warmup depends on completed steps and horizon": "output_noise",
}
PROGRESS_COUNTER_ENTRY = "progress schedule advances by a fixed internal update count (progress_schedule='updates')"
# Class C: fixed internal counts. Clock-free by measurement when the counter is
# saved learning state; admitted only under internal_counter_policy="measured".
COUNTER_ENTRIES = {
    "critic guard releases at a fixed minimum update count": "optimizer_step",
    "KA2 switches from pure A to blended penalty at call 800": "ka2_calls",
    PROGRESS_COUNTER_ENTRY: "progress_observed",
}


def v2_declarations(recipe, extensions):
    """The v1 table plus the internal-counter progress mode v1 does not know about."""
    entries = list(source_audit(recipe, extensions)["unexplained_clock_dependencies"])
    if recipe.get("progress_schedule") == "updates":
        entries.append(PROGRESS_COUNTER_ENTRY)
    return entries


def _step_schedule_values(recipe, completed, horizon):
    """Counterfactual step/horizon-driven schedule values for the recipe's shapes."""
    from particlegan import Recipe
    from particlegan.policy import input_noise_std, output_noise_std
    from particlegan.recipes import learning_rate_scales
    options = {key: (tuple(value) if isinstance(value, list) else value) for key, value in recipe.items()}
    options.update(total_steps=horizon, progress_schedule=None)
    stepped = Recipe(**options)
    network, prior = learning_rate_scales(completed, stepped)
    return {"learning_rate": (round(network, 12), round(prior, 12)),
            "input_noise": round(input_noise_std(stepped, completed), 12),
            "output_noise": round(output_noise_std(stepped, completed), 12)}


def power_witness(recipe, execution):
    """Which step-schedule components the declared probe could have detected.

    For each probe update the schedule is read with ``completed_steps`` before
    the update. Under a step/horizon-driven schedule the step_label branch
    would read ``completed + offset`` and the horizon branch would read the
    perturbed horizon. A component has power when either branch would see a
    different value than the reference at some probe update.
    """
    warmup, probe = execution["warmup_steps"], execution["probe_steps"]
    offset, horizon = execution["step_label_offset"], execution["perturbed_horizon"]
    reference_horizon = recipe["total_steps"]
    if type(reference_horizon) is not int:
        return {name: False for name in STEP_SCHEDULE_ENTRIES.values()}
    power = {name: False for name in STEP_SCHEDULE_ENTRIES.values()}
    for index in range(probe):
        completed = warmup + index
        reference = _step_schedule_values(recipe, completed, reference_horizon)
        for changed in (_step_schedule_values(recipe, completed + offset, reference_horizon),
                        _step_schedule_values(recipe, completed, horizon)):
            for name in power:
                power[name] = power[name] or changed[name] != reference[name]
    return power


def counter_witness(initial_state, execution):
    """Which fixed internal counters are saved learning state at the declared warmup."""
    warmup = execution["warmup_steps"]
    trainer = learning_state(initial_state)["trainer"]
    calls = [entry.get("regularizer", {}).get("record", {}).get("calls")
             for entry in trainer.get("optimizers", []) if isinstance(entry, dict)
             and isinstance(entry.get("regularizer"), dict)]
    progress = trainer.get("progress")
    return {
        # verify_probe already requires every optimizer step count == warmup.
        "optimizer_step": True,
        "ka2_calls": any(type(value) is int and value == warmup for value in calls),
        "progress_observed": isinstance(progress, dict) and progress.get("observed") == warmup,
    }


def measured_v2_decision(proof, execution, comparisons, counter_policy):
    """Decide the proposed measured-parity rule from a verified proof.

    1. Any differing measured comparison is FAIL; nothing overrides it.
    2. Every v2 declaration is a pre-screen entry. Class S entries are
       overridden by the passing measurement when the power witness shows the
       probe could have seen a step/horizon-driven version of that component.
       Class C entries are overridden only under counter_policy "measured" and
       only when the counter is found in saved learning state. Anything else
       (continuous policies, extensions, lazy-penalty counters, S entries
       without power) stands.
    3. PASS when nothing stands.
    """
    if counter_policy not in COUNTER_POLICIES:
        raise ValueError("internal_counter_policy must be measured or declared")
    differing = [row["condition"] for row in comparisons if row["reference_sha256"] != row["changed_sha256"]]
    declarations = v2_declarations(proof["recipe"], proof["extensions"])
    power = power_witness(proof["recipe"], execution)
    counters = counter_witness(proof["initial"], execution)
    overridden, standing = [], []
    for entry in declarations:
        component = STEP_SCHEDULE_ENTRIES.get(entry)
        counter = COUNTER_ENTRIES.get(entry)
        if not differing and component is not None and power[component]:
            overridden.append({"entry": entry, "class": "step_schedule", "witness": f"probe power on {component}"})
        elif not differing and counter is not None and counter_policy == "measured" and counters[counter]:
            overridden.append({"entry": entry, "class": "internal_counter", "witness": f"saved counter {counter}"})
        else:
            standing.append(entry)
    return {"rule": MEASURED_V2, "internal_counter_policy": counter_policy,
            "differing_conditions": differing, "declarations": declarations,
            "overridden": overridden, "standing": standing,
            "power_witness": power, "counter_witness": counters,
            "verdict": "FAIL" if differing or standing else "PASS",
            "sign_off": "required: proposed rule, not adopted"}


def learning_state(state):
    """Exclude only external clocks/configuration and independent eval streams."""
    trainer = state["trainer"]
    stream_names = state["streams"]["manifest"]["bindings"]
    named = {key: value for key, value in state["streams"]["states"].items()
             if stream_names[key]["family"] != "eval"}
    return {"trainer": {key: value for key, value in trainer.items()
                        if key not in {"recipe", "completed_steps", "max_steps", "streams"}},
            "training_rng": named}


def _proof_digest(value, proof):
    if proof["recipe"].get("continuous_policy") is not None:
        from .policy_adapters import typed_state_digest
        return typed_state_digest(value)
    return state_digest(value)


def _comparisons(proof):
    digest = lambda value: _proof_digest(value, proof)
    initial = learning_state(proof["initial"])
    common = {"permitted_state_sha256": digest(initial),
              "rng_state_sha256": digest(initial["training_rng"])}
    reference = digest([learning_state(s) for s in proof["trajectories"]["reference"]])
    return [{"condition": name, **common, "reference_sha256": reference,
             "changed_sha256": digest([learning_state(s) for s in proof["trajectories"][name]])}
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
            if context.recipe.total_steps is None:
                # Public continuous Recipes reject a training horizon. Their
                # only legal horizon is the independent external execution cap.
                trainer.max_steps = spec["perturbed_horizon"]
            else:
                context.recipe = context.recipe.replace(total_steps=spec["perturbed_horizon"])
                trainer.recipe = context.recipe
                trainer.policy.recipe = context.recipe
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
    law = (task["evaluation"]["sampling_law"] if context.policy_task is not None else PUBLIC_PRIOR_CLEAN)
    evidence = {**executed_receipt(law, eval_output_noise="clean"),
                "scoring_weights": task["evaluation"].get("scoring_weights", "live"),
                "comparisons": _comparisons(proof),
                "source_audit": source_audit(proof["recipe"], proof["extensions"]),
                "artifact_root": str(directory.resolve()), "artifact_manifest": manifest_artifacts(directory),
                "artifact_portability": {"storage": "local", "requires_bulk_artifacts": True}}
    result = {"evidence": evidence, "cost": {"training_seconds": time.monotonic() - started,
              "completed_updates": warmup + 5 * probe_steps}, "execution_path": "public_trainer",
              "recipe": proof["recipe"]}
    atomic_json(Path(output) / "adapter-receipt.json", result)
    return result


def verify_probe(task, evidence, *, with_proof=False):
    """Recompute state comparisons from certified tensors in the frozen evaluator."""
    root = Path(evidence["artifact_root"])
    verify_artifacts(root, evidence["artifact_manifest"])
    proof = torch.load(root / "comparisons.pt", map_location="cpu", weights_only=True)
    if proof.get("schema_version") != 1 or proof.get("execution") != task["execution"]:
        raise ValueError("clock probe execution differs from its task")
    names = {"reference", "step_label", "horizon", "evaluation_cadence", "restart"}
    if set(proof["trajectories"]) != names or set(proof["branch_initial"]) != names:
        raise ValueError("clock probe must preserve every comparison branch")
    digest = lambda value: _proof_digest(value, proof)
    if proof["recipe"].get("continuous_policy") is not None:
        from .policy_adapters import finite_policy_state
        states = [proof["initial"], *proof["branch_initial"].values(),
                  *(state for trajectory in proof["trajectories"].values() for state in trajectory)]
        if not all(finite_policy_state(state) for state in states):
            raise ValueError("clock proof contains nonfinite learned public policy state")
    initial_hash = digest(proof["initial"])
    if proof["initial"]["trainer"]["completed_steps"] != task["execution"]["warmup_steps"]:
        raise ValueError("clock initial state is not at the declared warmup")
    require_optimizer_steps(proof["initial"], task["execution"]["warmup_steps"])
    require_consistent_rng(proof["initial"])
    if (proof["recipe"] != proof["initial"]["recipe"]
            or proof["recipe"] != proof["initial"]["trainer"]["recipe"]
            or proof["extensions"] != proof["initial"]["extensions"]):
        raise ValueError("clock source audit formulation differs from the measured initial state")
    if initial_hash != digest(torch.load(root / "initial.pt", map_location="cpu", weights_only=True)):
        raise ValueError("serialized restart differs from the measured initial state")
    for name in names:
        if digest(proof["branch_initial"][name]) != initial_hash:
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
            continuous = proof["recipe"].get("continuous_policy") is not None
            horizon = (task["execution"]["perturbed_horizon"] if name == "horizon" and not continuous
                       else proof["recipe"]["total_steps"])
            if continuous:
                maximum = (task["execution"]["perturbed_horizon"] if name == "horizon"
                           else proof["initial"]["trainer"]["max_steps"])
                if state["trainer"].get("max_steps") != maximum:
                    raise ValueError("continuous clock probe external horizon differs")
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
    return (comparisons, audit, proof) if with_proof else (comparisons, audit)
