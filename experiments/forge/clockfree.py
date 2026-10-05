"""Measured clock perturbations on the public trainer, with saved state proof.

Passing a short probe is only one eligibility check. The declared long
continuations remain required to expose state-dependent delayed failures.
"""
from copy import deepcopy
from contextlib import contextmanager
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .artifacts import manifest_artifacts, verify_artifacts
from .contracts import atomic_json, file_hash, stable_hash
from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
from .state import require_consistent_rng, require_optimizer_steps, require_same_formulation, state_digest


ALLOWED_STATE = ["parameters_and_ema", "optimizer_moments_and_bias_correction",
                 "observed_gradients_and_row_visit_history", "named_training_rng"]
SCHEDULE_CLOCKS = {"learning_rate": "trainer_completed_steps",
                   "input_output_noise": "trainer_completed_steps", "beta2": "trainer_completed_steps",
                   "critic_coefficient": "critic_optimizer_observed_steps",
                   "critic_guard": "per_parameter_adam_steps"}


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
    if recipe.get("beta2_end") is not None and any(
            recipe["beta2_end"] != betas[1] for betas in (recipe["betas"], recipe.get("prior_betas") or recipe["betas"])):
        dependencies.append("Adam beta2 cosine depends on completed steps and horizon")
    if recipe.get("reg_coeff_end") is not None and recipe["reg_coeff_end"] != recipe["reg_coeff"]:
        dependencies.append("critic coefficient cosine depends on optimizer updates and horizon")
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
             "particlegan/recipe_schedules.py",
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
    scheduled = task["evaluation"]["kind"] == "schedule_contract"
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
    if scheduled:
        from .api import CapabilityError
        blockers = schedule_blockers(context.recipe.to_dict(), context.extension_values)
        if blockers:
            raise CapabilityError(blockers)
    for _ in range(warmup):
        update(context, trainer, data)
    initial = context.state_dict()
    proof = {"schema_version": 1, "execution": deepcopy(spec), "initial": initial,
             "recipe": context.recipe.to_dict(), "extensions": context.extension_values,
             "trajectories": {}, "branch_initial": {}}
    if scheduled:
        proof.update(schedule_observations={}, schedule_oracles={})
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
                if hasattr(trainer.penalty, "recipe"):
                    trainer.penalty.recipe = context.recipe
        trajectory = []
        controls = []
        for _ in range(probe_steps):
            if name == "evaluation_cadence":
                generator = context.streams.generator("eval", component="clock_probe", purpose="samples")
                trainer.sample(spec["evaluation_samples"], generator=generator, output_noise=False)
            if scheduled:
                with _observe_guard(trainer) as guard:
                    update(context, trainer, data)
                controls.append(_observed_controls(trainer, guard))
            else:
                update(context, trainer, data)
            trajectory.append(context.state_dict())
        proof["trajectories"][name] = trajectory
        if scheduled:
            proof["schedule_observations"][name] = controls
        _event("clock_probe", task=task["id"], condition=name, updates=probe_steps)
    if scheduled:
        # Replay through the same public trainer with its external clocks held
        # at reference values. Only independently predicted scheduled controls
        # are injected. Any other step/horizon effect therefore breaks parity.
        for name in ("reference", "step_label", "horizon"):
            context, trainer, data = construct()
            context.load_state_dict(deepcopy(initial))
            trajectory = []
            for index in range(probe_steps):
                expected = expected_controls(proof["recipe"], initial, spec, name, index)
                with _oracle_controls(trainer, expected):
                    update(context, trainer, data)
                trajectory.append(context.state_dict())
            proof["schedule_oracles"][name] = trajectory
            _event("schedule_oracle", task=task["id"], condition=name, updates=probe_steps)
    torch.save(proof, directory / "comparisons.pt")
    law = (task["evaluation"]["sampling_law"] if context.policy_task is not None else PUBLIC_PRIOR_CLEAN)
    evidence = {**executed_receipt(law, eval_output_noise="clean"),
                "scoring_weights": task["evaluation"].get("scoring_weights", "live"),
                "comparisons": _comparisons(proof),
                "source_audit": source_audit(proof["recipe"], proof["extensions"]),
                "artifact_root": str(directory.resolve()), "artifact_manifest": manifest_artifacts(directory),
                "artifact_portability": {"storage": "local", "requires_bulk_artifacts": True}}
    if scheduled:
        evidence["schedule_contract"] = schedule_metrics(task, proof)
    result = {"evidence": evidence, "cost": {"training_seconds": time.monotonic() - started,
              "completed_updates": warmup + (8 if scheduled else 5) * probe_steps}, "execution_path": "public_trainer",
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
    return comparisons, audit


def schedule_blockers(recipe, extensions):
    """The operational audit covers reviewed scheduled scalar mechanisms only."""
    audit = source_audit(recipe, extensions)
    supported = ("learning-rate annealing", "input-noise annealing", "output-noise warmup",
                 "Adam beta2 cosine", "critic coefficient cosine", "critic guard releases")
    return [entry for entry in audit["unexplained_clock_dependencies"]
            if not entry.startswith(supported)]


def expected_controls(recipe, initial, spec, condition, index):
    """Independent equations: never call the trainer's schedule functions.

    LR/noise/beta2 consume completed_steps. The penalty factory refreshes its
    coefficient at call time from the real critic optimizer counter. Guard
    release likewise consumes Adam history, never the external step label.
    """
    updates = spec["warmup_steps"] + index
    label = updates + (spec["step_label_offset"] if condition == "step_label" else 0)
    horizon = spec["perturbed_horizon"] if condition == "horizon" else recipe["total_steps"]

    def decay(total, floor):
        progress = min(1., max(0., (label - recipe["lr_anneal_start"] * total) /
                                 ((1 - recipe["lr_anneal_start"]) * total)))
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))

    def cosine(start, end, fraction, clock):
        progress = min(1., clock / (fraction * horizon))
        return float(end + (start - end) * .5 * (1 + math.cos(math.pi * progress)))

    cap = recipe.get("network_lr_horizon_cap")
    network_horizon = horizon if cap is None else min(horizon, cap)
    network_floor = recipe["network_lr_floor"]
    network = decay(network_horizon, recipe["lr_floor"] if network_floor is None else network_floor)
    prior = decay(horizon, recipe["lr_floor"])
    roles = initial["trainer"]["policy"]["roles"]
    optimizers = initial["trainer"]["optimizers"]
    rates, betas = [], []
    for groups, base_rates, role_names in zip(optimizers, initial["trainer"]["initial_lrs"], roles):
        rates.append([rate * (prior if role == "table" else network)
                      for rate, role in zip(base_rates, role_names)])
        row = []
        for group in groups["param_groups"]:
            beta = group.get("_recipe_initial_betas", group["betas"])
            row.append([beta[0], beta[1] if recipe.get("beta2_end") is None else cosine(
                beta[1], recipe["beta2_end"], recipe.get("beta2_anneal_end", .2), label)])
        betas.append(row)
    output = recipe["output_noise_std"]
    if recipe["output_noise_warmup"]:
        output *= min(1., label / (recipe["output_noise_warmup"] * horizon))
    coefficient = recipe["reg_coeff"]
    if recipe.get("reg_coeff_end") is not None:
        coefficient = cosine(coefficient, recipe["reg_coeff_end"],
                             recipe.get("reg_coeff_anneal_end", .2), updates)
    return {"rates": rates, "betas": betas, "network_scale": network, "prior_scale": prior,
            "input_sigma": float(recipe["input_noise_std"] * max(
                0., 1 - label / (recipe["input_noise_anneal_end"] * horizon))),
            "output_sigma": float(output), "coefficient": coefficient}


@contextmanager
def _oracle_controls(trainer, expected):
    """Override only schedule boundaries, without implementing an update loop."""
    def apply(steps, recipe, optimizers, penalty=None):
        for optimizer, betas in zip(optimizers, expected["betas"]):
            for group, beta in zip(optimizer.param_groups, betas):
                group["betas"] = tuple(beta)
        if penalty is not None:
            penalty.regularizer.coeff = expected["coefficient"]

    def penalty_schedule(steps, recipe, penalty):
        penalty.regularizer.coeff = expected["coefficient"]

    with patch.object(trainer.policy, "schedule", lambda *args: (
            expected["network_scale"], expected["prior_scale"])), \
            patch("particlegan.policy.input_noise_std", lambda *args: expected["input_sigma"]), \
            patch("particlegan.policy.output_noise_std", lambda *args: expected["output_sigma"]), \
            patch("particlegan.recipe_schedules.apply_training_schedules", apply), \
            patch("particlegan.recipe_schedules.apply_penalty_schedule", penalty_schedule):
        yield


@contextmanager
def _observe_guard(trainer):
    """Save the actual guard's gradients and Adam history, without altering it."""
    rows = []
    guard = getattr(trainer.opt_d, "guard", None)
    if guard is None:
        yield rows
        return
    original = guard.apply_

    def observe(optimizer):
        parameters = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter)
                if parameter.grad is None or not state:
                    continue
                second_name = "max_exp_avg_sq" if group.get("amsgrad", False) else "exp_avg_sq"
                parameters.append(parameter)
                rows.append({"before": parameter.grad.detach().clone(),
                             "second": state[second_name].detach().clone(),
                             "step": state["step"].detach().clone(), "beta2": group["betas"][1],
                             "ratio": guard.ratio, "min_steps": guard.min_steps})
        result = original(optimizer)
        for row, parameter in zip(rows, parameters):
            row["after"] = parameter.grad.detach().clone()
        return result

    with patch.object(guard, "apply_", observe):
        yield rows


def _observed_controls(trainer, guard):
    return {"rates": [[float(group["lr"]) for group in optimizer.param_groups]
                      for optimizer in trainer.policy.optimizers],
            "betas": [[list(group["betas"]) for group in optimizer.param_groups]
                      for optimizer in trainer.policy.optimizers],
            "input_sigma": float(trainer._noisy_D.std), "output_sigma": trainer.last_output_sigma,
            "coefficient": float(trainer.penalty.regularizer.coeff), "guard": guard}


def schedule_metrics(task, proof):
    """Recompute numerical control errors and public state replay comparisons."""
    names = {"reference", "step_label", "horizon", "evaluation_cadence", "restart"}
    if set(proof.get("schedule_observations", {})) != names:
        raise ValueError("every schedule branch needs actual consumed control observations")
    oracle_names = {"reference", "step_label", "horizon"}
    if set(proof.get("schedule_oracles", {})) != oracle_names:
        raise ValueError("all independent public-trainer schedule replays are required")
    spec, initial, recipe = task["execution"], proof["initial"], proof["recipe"]
    errors, guard_errors, guard_checks, verified_releases = [], [], 0, 0
    for name in sorted(names):
        observations = proof["schedule_observations"][name]
        if len(observations) != spec["probe_steps"]:
            raise ValueError("schedule observation prefix is incomplete")
        previous = initial
        for index, (actual, state) in enumerate(zip(observations, proof["trajectories"][name])):
            expected = expected_controls(recipe, initial, spec, name, index)
            for key in ("rates", "betas"):
                if len(actual[key]) != len(expected[key]):
                    raise ValueError("schedule optimizer observation topology differs")
                for observed, predicted, optimizer in zip(actual[key], expected[key], state["trainer"]["optimizers"]):
                    if len(observed) != len(predicted):
                        raise ValueError("schedule parameter group observation topology differs")
                    saved = [list(group["betas"]) if key == "betas" else group["lr"]
                             for group in optimizer["param_groups"]]
                    if observed != saved:
                        raise ValueError("consumed optimizer controls disagree with saved public state")
                    errors.extend((torch.tensor(observed, dtype=torch.float64) -
                                   torch.tensor(predicted, dtype=torch.float64)).abs().flatten().tolist())
            for key in ("input_sigma", "output_sigma", "coefficient"):
                errors.append(abs(actual[key] - expected[key]))
            if actual["output_sigma"] != state["trainer"]["policy"]["last_output_sigma"]:
                raise ValueError("consumed output noise differs from saved public state")
            guard_rows = actual["guard"]
            guarded = recipe.get("optimizer_family", "recipe") != "adam" and recipe["d_guard_ratio"] > 0
            prior_d = previous["trainer"]["optimizers"][1]
            history = [(group, prior_d["state"][key]) for group in prior_d["param_groups"]
                       for key in group["params"] if key in prior_d["state"]]
            if guarded and len(guard_rows) != len(history):
                raise ValueError("critic guard observations must cover every actual parameter history")
            if not guarded and guard_rows:
                raise ValueError("disabled guard produced observations")
            clipped_parameters = 0
            for row, (group, history_row) in zip(guard_rows, history):
                if (row["ratio"] != recipe["d_guard_ratio"] or row["min_steps"] != recipe["d_guard_min_steps"]
                        or row["step"].item() != spec["warmup_steps"] + index
                        or row["beta2"] != actual["betas"][1][0][1]):
                    raise ValueError("guard release settings differ from recipe or actual Adam history")
                second_name = "max_exp_avg_sq" if group.get("amsgrad", False) else "exp_avg_sq"
                if not torch.equal(row["second"], history_row[second_name]):
                    raise ValueError("guard second moment differs from saved optimizer history")
                variance = row["second"].mean() / (1 - row["beta2"] ** row["step"])
                ratio = row["before"].square().mean().sqrt() / variance.clamp_min(1e-30).sqrt()
                clips = (row["step"] >= row["min_steps"]) & (ratio > row["ratio"])
                scale = torch.where(clips,
                                    row["ratio"] / ratio, torch.ones_like(ratio))
                desired = row["before"] * scale
                guard_errors.append(float((desired - row["after"]).abs().max() /
                                          desired.abs().max().clamp_min(1e-30)))
                guard_checks += 1
                clipped_parameters += int(clips)
                verified_releases += int(row["step"].item() >= row["min_steps"])
            if guarded:
                before_count = prior_d["regularizer"]["guard"]["clipped_tensors"]
                after_count = state["trainer"]["optimizers"][1]["regularizer"]["guard"]["clipped_tensors"]
                guard_errors.append(float(abs(after_count - before_count - clipped_parameters)))
            previous = state
    mismatches = 0
    for name in sorted(oracle_names):
        trajectory = proof["schedule_oracles"][name]
        if len(trajectory) != spec["probe_steps"]:
            raise ValueError("independent schedule replay prefix is incomplete")
        for index, state in enumerate(trajectory, 1):
            require_optimizer_steps(state, spec["warmup_steps"] + index)
            require_same_formulation(initial, state)
            if state["recipe"] != recipe or state["trainer"]["recipe"] != recipe:
                raise ValueError("independent schedule replay changed its normalized recipe")
            if state["trainer"]["completed_steps"] != spec["warmup_steps"] + index:
                raise ValueError("independent schedule replay did not normalize its external clock")
            mismatches += int(state_digest(learning_state(state)) != state_digest(
                learning_state(proof["trajectories"][name][index - 1])))
    comparisons = _comparisons(proof)
    parity_failures = sum(row["reference_sha256"] != row["changed_sha256"]
                          for row in comparisons if row["condition"] in {"restart", "evaluation_cadence"})
    values = errors + guard_errors
    if not all(math.isfinite(value) for value in values):
        raise ValueError("schedule or guard observations contain nonfinite values")
    return {"maximum_schedule_error": max(errors, default=0.),
            "maximum_guard_relative_error": max(guard_errors, default=0.),
            "schedule_replay_state_mismatches": mismatches, "restart_cadence_failures": parity_failures,
            "schedule_observations": len(names) * spec["probe_steps"],
            "guard_parameter_checks": guard_checks, "guard_released_parameter_checks": verified_releases,
            "clockfree_claim": False}


def verify_schedule_probe(task, evidence):
    comparisons, audit = verify_probe(task, evidence)
    root = Path(evidence["artifact_root"])
    proof = torch.load(root / "comparisons.pt", map_location="cpu", weights_only=True)
    metrics = schedule_metrics(task, proof)
    if metrics != evidence.get("schedule_contract"):
        raise ValueError("schedule contract declarations disagree with saved actual training controls")
    verify_artifacts(root, evidence["artifact_manifest"])
    return comparisons, audit, metrics, schedule_blockers(proof["recipe"], proof["extensions"])
