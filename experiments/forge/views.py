"""Read-only task grading and versioned qualification views.

Receipts supplied here must already be filtered by the coordinator's candidate,
seed, source, and runtime compatibility key. A trainer's PASS string is never
evidence: graders reconstruct verdicts from curves or the existing file audits.
No function in this module launches training, rewrites receipts, or enqueues work.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from .sampling import grade_sampling, validate_declaration
from .priors import task_prior
from .initialization import task_initializer


ROOT = Path(__file__).resolve().parents[2]
IMPORTANCES = {"required", "ranking", "diagnostic"}
GATE_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INVALID", "NOT_RUN", "BLOCKED"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def task_execution_fingerprint(task: dict) -> str:
    """Scientific execution identity deliberately excludes view policy."""
    binding = {key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")}
    if task.get("task_cohort") is not None:
        binding.update(task_cohort=task["task_cohort"], policy_parent=task.get("policy_parent"))
    return _hash(binding)


def task_evaluation_fingerprint(task: dict) -> str:
    return _hash(task["evaluation"])


def task_fingerprint(task: dict) -> str:
    return _hash({"execution": task_execution_fingerprint(task),
                  "evaluation": task_evaluation_fingerprint(task)})


def view_fingerprint(view: dict) -> str:
    return _hash(view)


def _positive_int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cost(receipt):
    cost = receipt.get("cost", {})
    value = cost.get("wall_seconds", cost.get("seconds", 0)) if isinstance(cost, dict) else 0
    return value if _finite(value) and value >= 0 else 0


def _dependencies(task):
    result = []
    for entry in task.get("dependencies", []):
        if not isinstance(entry, dict) or entry.get("kind") not in {"gate", "checkpoint"}:
            raise ValueError(f"{task['id']}: dependency needs task and gate/checkpoint kind")
        if not isinstance(entry.get("task"), str) or not entry["task"]:
            raise ValueError(f"{task['id']}: dependency task must be named")
        result.append(entry)
    if len({entry["task"] for entry in result}) != len(result):
        raise ValueError(f"{task['id']}: duplicate dependency")
    return result


def _validate_measurement_contract(task):
    """Repeated evaluator constants are declarations, never ignored overrides."""
    validate_declaration(task)
    evaluation = task["evaluation"]
    kind = evaluation["kind"]
    fixed = {"scoring_weights": "live"}
    if task.get("task_cohort") is not None:
        from .policy_cohorts import validate_policy_observation
        fixed["scoring_weights"] = validate_policy_observation(task)["weight_selector"]
    if kind == "transfer_sustained":
        from benchmarks.locked_shared.observation import OBSERVATIONS, MIN_STABLE_CHECKS
        fixed.update(evaluator="benchmarks.transfer_suite.protocol:test_verdict",
                     observations=OBSERVATIONS, minimum_stable_checks=MIN_STABLE_CHECKS)
    elif kind == "native_accuracy":
        from benchmarks.toy100.accuracy import LIMITS
        from benchmarks.toy100.accuracy_gate import HOLDOUT_N
        from benchmarks.toy100.gate import MIN_STABLE_CHECKS
        from benchmarks.toy100.metrics import REQUIREMENTS
        fixed.update(evaluator="benchmarks.toy100.accuracy_gate:evaluate_suite",
                     coverage_evaluator="benchmarks.toy100.gate:evaluate_suite",
                     minimum_stable_checks=MIN_STABLE_CHECKS, holdout_samples=HOLDOUT_N,
                     coverage_thresholds=REQUIREMENTS, accuracy_limits=LIMITS)
    elif kind in {"ring_hold", "ring_extension"}:
        declaration = _convergence_class()().declaration()
        fixed.update(evaluator="reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py:ConvergenceGate",
                     thresholds=[["modes", "==", declaration["modes"]],
                                 ["hq", ">=", declaration["min_hq"]], ["hq", "<=", 1.]])
    elif kind == "paired_adaptation":
        from benchmarks.toy100.continuous_probe import RECOVERY_DEADLINE
        fixed.update(evaluator="benchmarks.toy100.continuous_probe:match_frozen_control",
                     recovery_deadline=RECOVERY_DEADLINE, diagnostic_every=10,
                     stationary_checks=5, minimum_frozen_passing=0)
        # The complete late window is recomputed from these consumed fields.
        fixed["deadline_checks"] = (evaluation["steps"] - evaluation["shift_step"] - RECOVERY_DEADLINE) // 10 + 1
    elif kind == "clockfree_parity":
        fixed.update(evaluator="experiments.forge.views:_clockfree", training_feedback=False)
        conditions = evaluation.get("conditions")
        if not isinstance(conditions, list) or sorted(conditions) != sorted(("step_label", "horizon", "evaluation_cadence", "restart")):
            raise ValueError(f"{task['id']}: all four fixed clock-free comparisons are required")
    else:
        return
    for key, expected in fixed.items():
        if key in evaluation and _hash(evaluation[key]) != _hash(expected):
            raise ValueError(f"{task['id']}: evaluation.{key} is fixed by the referenced evaluator; unsupported override")


def _validate_task(task):
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    required = {"schema_version", "id", "adapter", "execution", "evaluation",
                "resources", "requires_capabilities", "dependencies"}
    if required - task.keys():
        raise ValueError(f"task lacks {sorted(required - task.keys())}")
    if "qualification_tier" in task:
        raise ValueError("qualification_tier belongs only in view assignments")
    _positive_int(task["schema_version"], "task schema_version")
    for key in ("id", "adapter"):
        if not isinstance(task[key], str) or not task[key]:
            raise ValueError(f"task {key} must be nonempty")
    for key in ("execution", "evaluation", "resources"):
        if not isinstance(task[key], dict):
            raise ValueError(f"task {key} must be an object")
    if not isinstance(task["evaluation"].get("kind"), str):
        raise ValueError("task evaluation.kind is required")
    if not isinstance(task["dependencies"], list):
        raise ValueError("task dependencies must be a list")
    caps = task["requires_capabilities"]
    if not isinstance(caps, list) or any(not isinstance(x, str) or not x for x in caps):
        raise ValueError("requires_capabilities must be a list of names")
    _dependencies(task)
    task_prior(task)
    task_initializer(task)
    _validate_measurement_contract(task)


def load_tasks(root: Path | str) -> dict:
    tasks = {}
    for path in sorted((Path(root) / "configs/forge/tasks").glob("*.json")):
        task = _read(path)
        _validate_task(task)
        if task["id"] in tasks:
            raise ValueError(f"duplicate task id {task['id']}")
        tasks[task["id"]] = task
    if not tasks:
        raise ValueError("no Forge tasks found")
    return tasks


def load_view(root: Path | str, view_id: str) -> dict:
    if not isinstance(view_id, str) or Path(view_id).name != view_id or view_id in {"", ".", ".."}:
        raise ValueError("view id must be a filename-safe name")
    view = _read(Path(root) / "configs/forge/views" / f"{view_id}.json")
    if view.get("id") != view_id:
        raise ValueError("view filename and id differ")
    validate_view(view, load_tasks(root))
    return view


def validate_view(view: dict, tasks: dict) -> None:
    if not isinstance(view, dict):
        raise ValueError("view must be an object")
    for key in ("id", "goal"):
        if not isinstance(view.get(key), str) or not view[key]:
            raise ValueError(f"view needs {key}")
    _positive_int(view.get("schema_version"), "view schema_version")
    _positive_int(view.get("revision"), "view revision")
    assignments = view.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("view needs nonempty assignments")
    if not isinstance(view.get("eligibility", {}), dict):
        raise ValueError("view eligibility must be an object")
    index = {}
    required_tiers = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("assignment must be an object")
        name = assignment.get("task")
        if not isinstance(name, str) or name not in tasks:
            raise ValueError(f"missing task reference {name!r}")
        if name in index:
            raise ValueError(f"duplicate assignment {name}")
        _validate_task(tasks[name])
        if tasks[name]["id"] != name:
            raise ValueError("task index and id disagree")
        tier = assignment.get("qualification_tier")
        if type(tier) is not int or tier not in (1, 2, 3):
            raise ValueError("qualification_tier must be 1, 2, or 3")
        if assignment.get("importance") not in IMPORTANCES:
            raise ValueError("unknown task importance")
        _positive_int(assignment.get("order"), "assignment order", minimum=0)
        index[name] = assignment
        if assignment["importance"] == "required":
            required_tiers.add(tier)
    highest = max(a["qualification_tier"] for a in assignments)
    diagnostic = view.get("evidence_scope") == "calibration_diagnostic"
    if diagnostic and any(a["importance"] != "diagnostic" or a["qualification_tier"] != 1 for a in assignments):
        raise ValueError("calibration diagnostic views require only diagnostic tasks in Tier 1")
    if not diagnostic and required_tiers != set(range(1, highest + 1)):
        raise ValueError("empty required tier: qualification tiers must be contiguous from Tier 1")
    execution_groups = {}
    for name, assignment in index.items():
        execution = tasks[name]["execution"]
        if execution.get("uninterrupted") and execution.get("execution_group"):
            group = execution["execution_group"]
            tier = assignment["qualification_tier"]
            if group in execution_groups and execution_groups[group] != tier:
                raise ValueError(f"retier uninterrupted execution group {group} together; it cannot cross tier caps")
            execution_groups[group] = tier
    visiting, visited = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"dependency cycle involving {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in _dependencies(tasks[name]):
            parent = dep["task"]
            if parent not in index:
                raise ValueError(f"{name}: dependency {parent} missing from view")
            if index[parent]["qualification_tier"] > index[name]["qualification_tier"]:
                raise ValueError(f"{name}: dependency {parent} is in a later tier")
            if index[parent]["importance"] != "required" and index[name]["importance"] == "required":
                raise ValueError("a required task cannot depend on a nonrequired task")
            if dep["kind"] == "checkpoint":
                if not tasks[parent]["execution"].get("produces_state"):
                    raise ValueError(f"{name}: checkpoint dependency does not produce state")
                if tasks[name]["execution"].get("continuation_of") != parent:
                    raise ValueError(f"{name}: continuation_of must name its checkpoint dependency")
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    for name in index:
        visit(name)


def _verdict(status, reason, **details):
    return {"status": status, "gate_status": status, "reasons": [reason], **details}


def _curve(points, requirements, expected=None):
    if not isinstance(points, list) or not points:
        raise ValueError("missing observation curve")
    steps = []
    for point in points:
        if not isinstance(point, dict) or type(point.get("step")) is not int or point["step"] < 0:
            raise ValueError("observation steps must be nonnegative integers")
        steps.append(point["step"])
        for name, _, _ in requirements:
            if name not in point or not _finite(point[name]):
                raise ValueError(f"missing or nonfinite observed metric {name}")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("observation steps must be unique and strictly increasing")
    if expected is not None and steps != list(expected):
        raise ValueError("observation schedule is incomplete or differs from the task")
    return points


def _guards(task, evidence):
    expected = task["evaluation"].get("guards", {})
    if not expected:
        return None
    guards = evidence.get("guards")
    if not isinstance(guards, dict):
        return _verdict("INCOMPLETE", "missing finite-state/update/mechanism guards")
    if expected.get("finite_state") and guards.get("all_finite") is not True:
        return _verdict("FAIL" if guards.get("all_finite") is False else "INCOMPLETE",
                        "finite-state guard did not pass")
    if not isinstance(guards.get("optimizer_updates", {}), dict):
        return _verdict("INCOMPLETE", "optimizer update counts must be an object")
    for role in expected.get("optimizer_roles", []):
        count = guards.get("optimizer_updates", {}).get(role)
        if type(count) is not int or count < 0:
            return _verdict("INCOMPLETE", f"missing valid {role} optimizer update count")
        if count == 0:
            return _verdict("FAIL", f"{role} did not perform an intended optimizer update")
        if expected.get("exact_optimizer_updates") and count != task["execution"]["steps"]:
            return _verdict("INCOMPLETE", f"{role} optimizer updates do not complete the declared task budget")
    if expected.get("mechanism_exercised"):
        from .mechanisms import mechanism_blockers
        reasons = mechanism_blockers(guards.get("mechanism_audit"))
        if guards.get("hooks_exercised") is not True or reasons:
            return _verdict("BLOCKED", "; ".join(reasons) or "required mechanism activation has not been demonstrated")
    if expected.get("rng_isolation"):
        deviations = guards.get("unintended_rng_deviations")
        if type(deviations) is not int or deviations < 0:
            return _verdict("INCOMPLETE", "missing RNG deviation audit")
        if deviations:
            return _verdict("INVALID", "unintended RNG stream deviations invalidate comparison")
    return None


def _named_policy_layout(task):
    """Actual groups/count labels of the source-bound component producers.

    A packed basis bank is itself the generator; it has one table group and
    cannot attest a nonexistent separate generator group.  Optimizer-step
    counts still measure every original generator parameter in that group.
    """
    cohort = task["task_cohort"]
    host = task["execution"]["host"]
    roles = ["generator", "table", "noise"]
    counts = ("generator", "table", "discriminator")
    models = ["critic", "generator", "router"]
    if cohort == "conditional_policy_selected_cloud_v1":
        arcs = host in {"trajectory", "residual_student"}
        callback = "complete_arc_forward" if arcs else "complete_basis_forward"
        module = "conditional_policy_adapters"
        if arcs:
            models.append("prior")
        else:
            roles.remove("generator")
    elif cohort == "routed_policy_selected_cloud_v1":
        callback, module = "complete_slot_forward", "routed_policy_adapters"
    elif cohort == "multibank_policy_v1":
        callback, module = "complete_polar_forward", "multibank_policy_adapters"
        counts = ("generator", "prior", "discriminator")
        models.append("prior")
    elif cohort == "ae_routed_policy_v1":
        callback, module = "complete_ae_forward", "ae_routed_policy_adapters"
        counts = ("generator", "encoder", "prior", "discriminator")
        roles.insert(1, "encoder")
        models.extend(("encoder", "prior"))
    elif cohort in {"word_joint_policy_min11_v1", "word_joint_policy_min11_rates_v1"}:
        callback, module = "joint_generation", "word_joint_policy_adapters"
        counts = ("generator", "encoder", "prior", "discriminator")
        roles.insert(1, "encoder")
        models = ["critic", "encoder", "generator", "prior"]
    else:
        raise ValueError("no declared component-owner layout for this named policy cohort")
    return {"counts": counts, "groups": [roles, ["critic"]], "models": sorted(models),
            "callback": {"callable": True, "module": "experiments.forge." + module,
                         "qualname": callback}}


def _routed_policy_guards(task, evidence, contract, steps):
    """Check actual conditional owners without claiming independent row credit."""
    controls = evidence["policy_controls"]
    if (type(controls.get("schema_version")) is not int or controls["schema_version"] != 1
            or controls.get("family") != contract["family"]
            or controls.get("row_policy") != contract["row_policy"]
            or controls.get("independent_atlas_qualification") is not False):
        return _verdict("INVALID", "named routed controls borrowed another family or independent-row credit")
    layout = _named_policy_layout(task)
    if "roles" not in controls:
        return _verdict("INCOMPLETE", "missing actual named optimizer group owners")
    if controls["roles"] != layout["groups"]:
        return _verdict("INVALID", "actual optimizer groups omit or substitute a named model/table owner")
    rates = controls.get("effective_group_lrs")
    if (not isinstance(rates, list) or len(rates) != len(layout["groups"])
            or any(not isinstance(row, list) or len(row) != len(group)
                   or any(not _finite(value) or value < 0 for value in row)
                   for row, group in zip(rates, layout["groups"]))):
        return _verdict("INCOMPLETE", "missing finite actual rates for every named optimizer group")
    if controls["execution"]["floating_dtypes"] != ["torch.float32"]:
        return _verdict("INVALID", "named routed host did not preserve its original FP32 tensor law")
    direct = evidence.get("guards", {}).get("mechanism_audit", {}).get("mechanisms", {}).get("direct_particle_gain")
    if not isinstance(direct, dict):
        return _verdict("INCOMPLETE", "missing routed direct-response inapplicability evidence")
    if (direct.get("applicable_to_routed_table") is not False
            or direct.get("host_activation_credit") is not False
            or direct.get("requested") is not True or direct.get("enabled") is not False
            or any(type(direct.get(key)) is not int or direct[key] != 0
                   for key in ("calls", "eligible", "applied"))):
        return _verdict("INVALID", "conditional transport cannot claim independent direct-particle response")
    damping = evidence["guards"]["mechanism_audit"]["mechanisms"].get("a2")
    if (not isinstance(damping, dict) or damping.get("requested") is not True
            or damping.get("enabled") is not True or type(damping.get("calls")) is not int
            or damping["calls"] != steps):
        return _verdict("BLOCKED", "named table lacks its actual requested latent-damping owner/history")
    from .mechanisms import mechanism_blockers
    reasons = mechanism_blockers(evidence["guards"].get("mechanism_audit"))
    if evidence["guards"].get("hooks_exercised") is not True or reasons:
        return _verdict("BLOCKED", "; ".join(reasons) or "named requested mechanisms lack actual host/probe evidence")
    semantics = contract.get("table_optimizer_semantics")
    if semantics is not None and controls.get("table_optimizer_semantics") != semantics:
        return _verdict("INVALID", "actual routed table optimizer semantics differ from their named contract")
    owner = controls.get("routed_owner")
    if not isinstance(owner, dict):
        return _verdict("INCOMPLETE", "missing actual guarded complete-forward routed owner")
    if (type(owner.get("schema_version")) is not int or owner["schema_version"] != 1
            or owner.get("owner") != "particlegan.routing.RoutedRowControl"
            or _hash(owner.get("model_forward")) != _hash(layout["callback"])
            or owner.get("model_roles") != layout["models"]
            or owner.get("table_matches_policy") is not True
            or owner.get("averaged_table_matches_policy") is not True):
        return _verdict("INVALID", "routed owner omits the complete named callback/models or selected table")
    config = owner.get("config")
    if not isinstance(config, dict):
        return _verdict("INCOMPLETE", "missing actual routed configuration")
    routing = contract["routing"]
    fixed = {"model_forward": True, "sites": routing["sites"],
             "log_mass_key": "log_mass", "row_buffers": routing["row_buffers"],
             "row_parameters": [], "routed_geometry": "mass_atoms_v1",
             "output_error_guard": True, "max_context_harm": 0.,
             "max_output_error_increase": 0., "max_output_context_harm": 0.,
             # All four named v1 producers use these original public defaults.
             # A weaker evidence resource or altered transport is a new law.
             "probe_budget": 8, "reservoir_size": 64, "min_observations": 8,
             "min_effect": 1e-6, "improvement_margin": 1e-8,
             "persistence_threshold": .75, "split_scale": .1, "candidate_budget": 4}
    if (set(config) != set(fixed) or any(_hash(config[key]) != _hash(value) for key, value in fixed.items())):
        return _verdict("INVALID", "actual routed function/mass/row law or protected-context harm bound changed")
    for name in ("fit_fill", "guard_fill"):
        if type(owner.get(name)) is not int or not config["min_observations"] <= owner[name] <= config["reservoir_size"]:
            return _verdict("INCOMPLETE", "missing nonempty original fit and protected guard contexts")
    clock = owner.get("probe_clock")
    if (not isinstance(clock, dict) or type(clock.get("observed_updates")) is not int
            or clock["observed_updates"] != steps or type(clock.get("last_probe_update")) is not int
            or not 0 <= clock["last_probe_update"] <= steps):
        return _verdict("INVALID", "routed context observations do not cover the complete public lifecycle")
    resource = task["execution"]["resources"]
    shape = [resource["num_particles"], resource["z_dim"]]
    ownership = owner.get("row_ownership")
    if (owner.get("table_shape") != shape or not isinstance(ownership, dict)
            or set(ownership) != {"table", "router.log_mass", *("router." + name for name in routing["row_buffers"])}):
        return _verdict("INVALID", "routed table/row-buffer ownership differs from the original named host")
    for name, value in ownership.items():
        expected = {"shape": shape if name == "table" else (
            [shape[0]] if name == "router.log_mass" else [shape[0], 1]),
            "dtype": "torch.int64" if name not in {"table", "router.log_mass"} else "torch.float32",
            "parameter": name == "table", "optimizer": 0 if name == "table" else None}
        if not isinstance(value, dict) or _hash(value) != _hash(expected):
            return _verdict("INVALID", "routed row owner dtype/shape/optimizer differs from its source-bound role")
    if (owner.get("state_digest_kind") != "typed_policy_state_v1"
            or not _digest(owner.get("state_sha256"))):
        return _verdict("INCOMPLETE", "actual routed contexts/controller state lack a typed identity")
    diagnostics = controls.get("diagnostics")
    controller = diagnostics.get("controller") if isinstance(diagnostics, dict) else None
    if (not isinstance(controller, dict) or controller.get("variant") != "dv12"
            or type(controller.get("updates")) is not int or controller["updates"] != steps):
        return _verdict("BLOCKED", "named routed execution lacks actual DV12 controller/latent applications")
    applications = controller.get("latent_applications")
    fields = {"radius_min", "radius_mean", "radius_max", "perturbation_rms", "clipped_fraction"}
    if (not isinstance(applications, list) or not 1 <= len(applications) <= 2
            or any(not isinstance(row, dict) or set(row) != fields
                   or any(not _finite(value) or value < 0 for value in row.values())
                   or row["radius_min"] > row["radius_max"]
                   or row["clipped_fraction"] > 1 for row in applications)):
        return _verdict("BLOCKED", "named DV12 proof lacks its actual bounded last-two application measurements")
    for name in ("stationarity_lr", "row_evidence", "birth_death", "surprise", "reopen_guard"):
        if not isinstance(diagnostics.get(name), dict):
            return _verdict("INCOMPLETE", "missing actual named policy owner diagnostics: " + name)
    row_updates = diagnostics["row_evidence"].get("updates")
    if (type(controls.get("row_evidence_observations")) is not int
            or controls["row_evidence_observations"] != steps
            or type(row_updates) is not int or row_updates != steps):
        return _verdict("INVALID", "named paired-row gradient observations do not cover the actual updates")
    rows = diagnostics["birth_death"].get("rows")
    if (not isinstance(rows, dict) or rows.get("law") != "conditional_paired_diagnostic"
            or rows.get("counters") != diagnostics["row_evidence"]):
        return _verdict("INVALID", "named routed evidence cannot borrow independent atom decisions")
    settle = diagnostics["stationarity_lr"]
    group_keys = {f"{'g' if index == 0 else 'd'}{j}"
                  for index, group in enumerate(layout["groups"]) for j in range(len(group))}
    if (set(settle) != group_keys or any(not isinstance(value, dict)
            or not _finite(value.get("s")) or value["s"] < 0 for value in settle.values())):
        return _verdict("INCOMPLETE", "stationarity owner diagnostics omit an actual named optimizer group")
    if (type(diagnostics["surprise"].get("fires")) is not int or diagnostics["surprise"]["fires"] < 0
            or type(diagnostics["surprise"].get("armed")) is not bool
            or type(diagnostics["reopen_guard"].get("epoch_rebases")) is not int
            or diagnostics["reopen_guard"]["epoch_rebases"] < 0):
        return _verdict("INCOMPLETE", "actual surprise/reopen owners lack their source-defined state counters")
    counters = owner.get("counters")
    if not isinstance(counters, dict) or not counters or any(type(v) is not int or v < 0 for v in counters.values()):
        return _verdict("INCOMPLETE", "missing typed actual routed counters; structural moves are not presumed")
    if diagnostics["birth_death"].get("counters") != counters:
        return _verdict("INVALID", "routed structural owner diagnostics disagree with their actual counters")
    return None


def _word_joint_policy_guards(task, evidence, contract, steps):
    """Require the distinct min11 independent joint cloud and its free encoder.

    Its same-code atom and words-only noise are source-bound adaptations, not
    routed guards or evidence for the original five-row/independent Atlas task.
    """
    controls = evidence["policy_controls"]
    if task["task_cohort"] == "word_joint_policy_min11_rates_v1":
        from .word_joint_rate_policy_contracts import validate_binding_receipt
        try:
            validate_binding_receipt(controls.get("word_rate_binding"), task)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return _verdict("INVALID", str(error))
    if (type(controls.get("schema_version")) is not int or controls["schema_version"] != 1
            or controls.get("family") != contract["family"]
            or controls.get("independent_atlas_qualification") is not False
            or controls.get("routed_owner") is not None):
        return _verdict("INVALID", "min11 joint controls borrowed another family or routed/Atlas credit")
    fixed = {"actual_prior_rows": 11, "canonical_target_words": 5,
             "joint_atom_code": "same_effective_code", "output_noise_coordinates": "words168_only"}
    if not fixed.keys() <= controls.keys():
        return _verdict("INCOMPLETE", "missing actual min11 resource, joint-code or word-noise owners")
    if (any(type(controls[key]) is not type(value) or controls[key] != value
            for key, value in fixed.items())
            or _hash(controls.get("resource_adaptation")) != _hash(contract["resource_adaptation"])):
        return _verdict("INVALID", "min11 joint resource, code/noise law or adaptation provenance differs")
    birth = controls.get("actual_birth_death")
    actual = {"rows": 11, "neighbours": 5, "isolation": True, "reference_half": 6}
    if not isinstance(birth, dict) or set(birth) != set(actual):
        return _verdict("INCOMPLETE", "missing the actual eligible eleven-row isolated birth/death population")
    if any(type(birth[key]) is not type(value) or birth[key] != value for key, value in actual.items()):
        return _verdict("INVALID", "min11 birth/death population cannot borrow five/six-row or disabled-isolation evidence")
    layout = _named_policy_layout(task)
    if "roles" not in controls:
        return _verdict("INCOMPLETE", "missing actual word generator/encoder/table/noise optimizer owners")
    if controls["roles"] != layout["groups"]:
        return _verdict("INVALID", "word optimizer groups omit the free encoder or substitute another row law")
    rates = controls.get("effective_group_lrs")
    if (not isinstance(rates, list) or len(rates) != len(layout["groups"])
            or any(not isinstance(row, list) or len(row) != len(group)
                   or any(not _finite(value) or value < 0 for value in row)
                   for row, group in zip(rates, layout["groups"]))):
        return _verdict("INCOMPLETE", "missing finite actual rates for every word optimizer group")
    if controls["execution"]["floating_dtypes"] != ["torch.float32"]:
        return _verdict("INVALID", "word host did not preserve its original FP32 tensor law")
    host = evidence.get("host")
    if not isinstance(host, dict):
        return _verdict("INCOMPLETE", "missing actual original-word host/objective/resource evidence")
    expected_host = {"family": contract["family"], "task_cohort": task["task_cohort"],
        "actual_resources": task["execution"]["resources"],
        "canonical_words": task["execution"]["host_definition"]["words"],
        "resource_adaptation": contract["resource_adaptation"], "objective": contract["objective"],
        "reconstruction_training_loss": False, "original_capacity_or_qualification_credit": False}
    if any(key not in host for key in expected_host):
        return _verdict("INCOMPLETE", "word host lacks its original objective, free-inverse scope or resource provenance")
    if any(type(host[key]) is not type(value) or _hash(host[key]) != _hash(value)
           for key, value in expected_host.items()):
        return _verdict("INVALID", "word host changed the objective, target, row resources or claimed borrowed credit")
    mechanisms = evidence.get("guards", {}).get("mechanism_audit", {}).get("mechanisms", {})
    direct = mechanisms.get("direct_particle_gain")
    if (not isinstance(direct, dict) or direct.get("applicable_to_direct_output") is not False
            or direct.get("host_activation_credit") is not False
            or direct.get("requested") is not True or direct.get("enabled") is not False
            or any(type(direct.get(key)) is not int or direct[key] != 0
                   for key in ("calls", "eligible", "applied"))):
        return _verdict("INVALID", "latent joint rows cannot claim direct word-output particle response")
    damping = mechanisms.get("a2")
    if (not isinstance(damping, dict) or damping.get("requested") is not True
            or damping.get("enabled") is not True or type(damping.get("calls")) is not int
            or damping["calls"] != steps):
        return _verdict("BLOCKED", "word prior lacks its actual requested latent-damping owner/history")
    from .mechanisms import mechanism_blockers
    reasons = mechanism_blockers(evidence["guards"].get("mechanism_audit"))
    if evidence["guards"].get("hooks_exercised") is not True or reasons:
        return _verdict("BLOCKED", "; ".join(reasons) or "word requested mechanisms lack actual host/probe evidence")
    diagnostics = controls.get("diagnostics")
    controller = diagnostics.get("controller") if isinstance(diagnostics, dict) else None
    if (not isinstance(controller, dict) or controller.get("variant") != "dv12"
            or type(controller.get("updates")) is not int or controller["updates"] != steps):
        return _verdict("BLOCKED", "word execution lacks the actual DV12 joint-cloud controller history")
    applications = controller.get("latent_applications")
    fields = {"radius_min", "radius_mean", "radius_max", "perturbation_rms", "clipped_fraction"}
    if (not isinstance(applications, list) or not 1 <= len(applications) <= 2
            or any(not isinstance(row, dict) or set(row) != fields
                   or any(not _finite(value) or value < 0 for value in row.values())
                   or row["radius_min"] > row["radius_max"] or row["clipped_fraction"] > 1
                   for row in applications)):
        return _verdict("BLOCKED", "word DV12 proof lacks actual bounded last-two application measurements")
    for name in ("stationarity_lr", "row_evidence", "birth_death", "surprise", "reopen_guard"):
        if not isinstance(diagnostics.get(name), dict):
            return _verdict("INCOMPLETE", "missing actual word policy diagnostics: " + name)
    updates = diagnostics["row_evidence"].get("updates")
    if (type(controls.get("row_evidence_observations")) is not int
            or controls["row_evidence_observations"] != steps or type(updates) is not int or updates != steps):
        return _verdict("INVALID", "word joint row evidence does not cover the actual optimizer history")
    if type(diagnostics["birth_death"].get("k")) is not int or diagnostics["birth_death"]["k"] != 5:
        return _verdict("INVALID", "word diagnostics differ from the actual eligible independent birth/death owner")
    settle = diagnostics["stationarity_lr"]
    if (set(settle) != {"g0", "g1", "g2", "g3", "d0"}
            or any(not isinstance(row, dict) or not _finite(row.get("s")) or row["s"] < 0
                   for row in settle.values())):
        return _verdict("INCOMPLETE", "word stationarity diagnostics omit an actual generator/encoder/prior/noise/critic group")
    if (type(diagnostics["surprise"].get("fires")) is not int or diagnostics["surprise"]["fires"] < 0
            or type(diagnostics["surprise"].get("armed")) is not bool
            or type(diagnostics["reopen_guard"].get("epoch_rebases")) is not int
            or diagnostics["reopen_guard"]["epoch_rebases"] < 0):
        return _verdict("INCOMPLETE", "actual word surprise/reopen owners lack source-defined state counters")
    return None


def _named_policy_artifacts(task, evidence, measured_steps):
    """Verify retained bytes only; never deserialize a model or rescore arrays."""
    checkpoint = evidence.get("checkpoint")
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {
            "path", "sha256", "state_sha256", "digest_kind"}
            or checkpoint.get("digest_kind") != "typed_policy_state_v1"
            or not _digest(checkpoint.get("sha256")) or not _digest(checkpoint.get("state_sha256"))):
        return _verdict("INCOMPLETE", "named component host requires its complete typed checkpoint identity")
    artifact = evidence.get("artifact_root")
    manifest = evidence.get("artifact_manifest")
    if not isinstance(artifact, str) or not artifact or not isinstance(manifest, dict):
        return _verdict("INCOMPLETE", "named component checkpoint and observations need retained artifact identities")
    files = manifest.get("files")
    if not isinstance(files, dict):
        return _verdict("INCOMPLETE", "named component artifact manifest is incomplete")
    row = files.get("state.pt")
    if (checkpoint["path"] != "state.pt" or not isinstance(row, dict)
            or row.get("sha256") != checkpoint["sha256"]
            or type(row.get("size")) is not int or row["size"] <= 0):
        return _verdict("INVALID", "named checkpoint bytes are not bound to the retained artifact manifest")
    required = {f"observations/step_{step:06d}.npz" for step in measured_steps}
    if not required <= files.keys():
        return _verdict("INCOMPLETE", "original selected goal views are missing from retained observation checkpoints")
    if any(not isinstance(files[name], dict) or type(files[name].get("size")) is not int
           or files[name]["size"] <= 0 for name in required):
        return _verdict("INVALID", "retained selected goal views cannot be empty artifact placeholders")
    from .artifacts import verify_artifacts
    try:
        verify_artifacts(Path(artifact), manifest)
    except (FileNotFoundError, OSError) as error:
        return _verdict("INCOMPLETE", "retained named policy artifacts are unavailable: " + str(error))
    except (KeyError, TypeError, ValueError) as error:
        return _verdict("INVALID", "retained named policy artifact identity changed: " + str(error))
    return None


def _policy_guards(task, evidence):
    """Require observed lifecycle and pure reads for prospective policy tasks."""
    if task.get("task_cohort") is None:
        return None
    from .policy_cohorts import validate_policy_task
    contract = validate_policy_task(task)
    controls = evidence.get("policy_controls")
    if not isinstance(controls, dict):
        return _verdict("INCOMPLETE", "missing actual public policy lifecycle/control evidence")
    if (controls.get("cohort") != task["task_cohort"]
            or controls.get("row_semantics") != contract["row_semantics"]):
        return _verdict("INVALID", "public policy cohort or row semantics differ from the frozen task")
    execution = controls.get("execution")
    if (not isinstance(execution, dict) or not isinstance(execution.get("model_devices"), list)
            or not execution["model_devices"]
            or not isinstance(execution.get("floating_dtypes"), list) or not execution["floating_dtypes"]
            or type(execution.get("autocast_enabled")) is not bool):
        return _verdict("INCOMPLETE", "missing observed model device, dtype and precision context")
    if (any(not isinstance(value, str) or value.split(":", 1)[0] != task["execution"]["device"]
            for value in execution["model_devices"])
            or any(not isinstance(value, str) or not value.startswith("torch.float")
                   for value in execution["floating_dtypes"])
            or execution["autocast_enabled"]):
        return _verdict("INVALID", "observed policy device or autocast differs from this frozen GPU task")
    requested, enabled = controls.get("requested"), controls.get("enabled")
    if (not isinstance(requested, dict) or not requested or not isinstance(enabled, dict)
            or set(requested) != set(enabled)
            or any(type(value) is not bool for value in (*requested.values(), *enabled.values()))):
        return _verdict("INCOMPLETE", "missing typed requested/enabled policy-owner evidence")
    required_controls = {"continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
                         "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard"}
    if set(requested) != required_controls or not all(requested.values()):
        return _verdict("INVALID", "requested owners differ from this declared Atlas policy mechanism")
    if (controls.get("requested_owners_bound") is not True
            or any(value and not enabled[name] for name, value in requested.items())):
        return _verdict("BLOCKED", "a requested policy mechanism has no actual enabled owner")
    lifecycle = controls.get("lifecycle", {})
    if not isinstance(lifecycle, dict) or "owner" not in lifecycle:
        return _verdict("INCOMPLETE", "missing actual public policy lifecycle owner")
    steps = controls.get("completed_steps")
    start = lifecycle.get("start_completed_steps")
    if (type(steps) is not int or not 0 < steps <= task["execution"]["steps"]
            or type(start) is not int or not 0 <= start < steps):
        return _verdict("INCOMPLETE", "missing valid observed public policy update interval")
    if start != task["execution"].get("preserve_prefix_steps", 0):
        return _verdict("INCOMPLETE", "policy lifecycle does not cover this task's complete owned update interval")
    kind = task["evaluation"]["kind"]
    if kind in {"transfer_sustained", "native_accuracy"} and steps != task["execution"]["steps"]:
        return _verdict("INCOMPLETE", "public policy did not complete the original full task budget")
    from .policy_adapters import HOOKS
    calls = lifecycle.get("calls")
    expected = steps - start
    if (not isinstance(calls, dict) or set(calls) != set(HOOKS)
            or any(type(value) is not int or value != expected for value in calls.values())
            or lifecycle.get("owner") != "particlegan.UpdatePolicy"
            or lifecycle.get("end_completed_steps") != steps
            or lifecycle.get("observed_updates") != expected
            or lifecycle.get("pending") != [] or lifecycle.get("order_errors") != 0
            or lifecycle.get("last_order") != list(HOOKS)
            or lifecycle.get("complete") is not True
            or controls.get("implementation_observed") is not True):
        return _verdict("INVALID", "ordered public policy hooks do not establish the complete update interval")
    guards = evidence.get("guards")
    if not isinstance(guards, dict) or type(guards.get("all_finite")) is not bool:
        return _verdict("INCOMPLETE", "missing observed finite policy-state guard")
    if not guards["all_finite"]:
        return _verdict("FAIL", "observed public policy state is nonfinite")
    deviations = guards.get("unintended_rng_deviations")
    if type(deviations) is not int or deviations < 0:
        return _verdict("INCOMPLETE", "missing observed global and named training RNG audit")
    if deviations:
        return _verdict("INVALID", "unintended RNG stream deviations invalidate policy comparison")
    updates = guards.get("optimizer_updates")
    routed = contract["row_semantics"] == "conditional"
    word = task["task_cohort"] in {"word_joint_policy_min11_v1", "word_joint_policy_min11_rates_v1"}
    named = routed or word
    roles = (_named_policy_layout(task)["counts"] if named else (
        ("prior", "discriminator") if task["execution"].get("host") == "two_pole" else
        ("generator", "discriminator", "prior")))
    for role in roles:
        count = updates.get(role) if isinstance(updates, dict) else None
        if type(count) is not int or count < 0:
            return _verdict("INCOMPLETE", f"missing actual {role} optimizer update count")
        if count == 0:
            return _verdict("FAIL", f"{role} did not perform an intended policy optimizer update")
        if count != steps:
            return _verdict("INVALID", f"{role} optimizer state disagrees with the completed policy updates")
    if routed:
        problem = _routed_policy_guards(task, evidence, contract, steps)
        if problem is not None:
            return problem
    elif word:
        problem = _word_joint_policy_guards(task, evidence, contract, steps)
        if problem is not None:
            return problem
    purity = evidence.get("policy_purity")
    if not isinstance(purity, list) or not purity:
        return _verdict("INCOMPLETE", "missing policy state and RNG measurement-purity evidence")
    for observed in purity:
        if (not isinstance(observed, dict) or observed.get("digest_kind") != "typed_policy_state_v1"
                or not _digest(observed.get("before_sha256"))
                or not _digest(observed.get("after_sha256"))):
            return _verdict("INCOMPLETE", "measurement purity lacks exact typed policy-state identities")
        if (observed.get("pure") is not True
                or observed["before_sha256"] != observed["after_sha256"]):
            return _verdict("INVALID", "measurement changed policy, optimizer, model or training RNG state")
        if named:
            prefix = "global" if task["task_cohort"] == "ae_routed_policy_v1" else "global_rng"
            before, after = observed.get(prefix + "_before_sha256"), observed.get(prefix + "_after_sha256")
            if not _digest(before) or not _digest(after):
                return _verdict("INCOMPLETE", "named policy measurement lacks a separate global RNG audit")
            if before != after:
                return _verdict("INVALID", "named policy measurement changed global RNG state")
    observations = evidence.get("policy_observations")
    if not isinstance(observations, list) or len(observations) != len(purity):
        return _verdict("INCOMPLETE", "policy measurement declarations and purity observations are incomplete")
    if named:
        audits = evidence.get("rng_audits")
        if not isinstance(audits, list) or len(audits) != len(observations):
            return _verdict("INCOMPLETE", "named policy measurements lack their full named-stream audit")
        for audit in audits:
            if (not isinstance(audit, dict) or type(audit.get("unintended_rng_deviations")) is not int
                    or audit["unintended_rng_deviations"] != 0 or audit.get("unintended_streams") != []
                    or not isinstance(audit.get("changed_streams"), list)):
                return _verdict("INVALID", "named policy observation consumed an unintended training RNG stream")
            for key in audit["changed_streams"]:
                try:
                    binding = json.loads(key)
                except (TypeError, ValueError):
                    return _verdict("INVALID", "named policy stream audit contains a malformed stream identity")
                if (not isinstance(binding, list) or len(binding) != 4
                        or any(not isinstance(value, str) or not value for value in binding)
                        or binding[0] != "eval" or binding[3].split(":", 1)[0] != task["execution"]["device"]):
                    return _verdict("INVALID", "only isolated evaluation streams may change during a policy observation")
    if evidence.get("policy_observation") != observations[-1]:
        return _verdict("INVALID", "final policy measurement differs from its retained observation")
    if "served_source" not in controls:
        return _verdict("INCOMPLETE", "missing actual final policy serving source")
    if controls["served_source"] != observations[-1].get("selected_source"):
        return _verdict("INVALID", "final policy serving source differs from its scored measurement")
    measured_steps = []
    from .sampling import grade_sampling
    for observed, audit in zip(observations, purity):
        observed_step = observed.get("completed_steps") if isinstance(observed, dict) else None
        minimum_step = 0 if kind == "native_accuracy" else 1
        if type(observed_step) is not int or type(audit.get("completed_steps")) is not int:
            return _verdict("INCOMPLETE", "measurement is not bound to its actual policy update and purity audit")
        if not minimum_step <= observed_step <= steps or audit["completed_steps"] != observed_step:
            return _verdict("INVALID", "measurement and purity clocks disagree with actual policy updates")
        measured_steps.append(observed_step)
        problem = grade_sampling(task, {**evidence, "policy_observation": observed})
        if problem is not None:
            return _verdict(problem["status"], problem["reason"])
    if measured_steps[-1] != steps or any(b < a for a, b in zip(measured_steps, measured_steps[1:])):
        return _verdict("INVALID", "retained policy measurements do not establish chronological final-state evidence")
    if kind == "transfer_sustained":
        expected_steps = [math.ceil(i * task["execution"]["steps"] / 24) for i in range(1, 25)]
        if measured_steps != expected_steps:
            status = "INCOMPLETE" if len(measured_steps) < len(expected_steps) else "INVALID"
            return _verdict(status, "all original transfer checkpoints need observed pure policy measurements")
    elif kind == "native_accuracy":
        from benchmarks.toy100.train import evaluation_steps
        evaluation = task["evaluation"]
        expected_steps = evaluation_steps(task["execution"]["steps"],
                                          evaluation["eval_interval"], evaluation["early_eval_steps"])
        if start:
            expected_steps = [step for step in expected_steps if step > start]
        if not set(expected_steps) <= set(measured_steps):
            return _verdict("INCOMPLETE", "all original native checkpoints need observed pure policy measurements")
    elif kind in {"ring_hold", "ring_extension"}:
        dense = evidence.get("dense")
        if (not isinstance(dense, list) or not dense
                or not {point.get("step") for point in dense if isinstance(point, dict)} <= set(measured_steps)
                or dense[-1].get("step") != steps):
            return _verdict("INCOMPLETE", "ring measurements lack their complete uninterrupted policy interval")
    if named:
        return _named_policy_artifacts(task, evidence, measured_steps)
    return None


def _transfer(task, evidence):
    from benchmarks.transfer_suite.protocol import test_verdict

    evaluation = task["evaluation"]
    spec = {"steps": task["execution"]["steps"], "thresholds": evaluation["thresholds"]}
    expected = [math.ceil(i * spec["steps"] / 24) for i in range(1, 25)]
    points = evidence.get("observations", evidence.get("curve"))
    try:
        _curve(points, spec["thresholds"])
    except ValueError as exc:
        return _verdict("INVALID" if points else "INCOMPLETE", str(exc))
    if [p["step"] for p in points] != expected:
        return _verdict("INCOMPLETE", "all 24 declared observation checkpoints are required")
    live = evidence.get("live")
    if not isinstance(live, dict) or any(not _finite(live.get(key)) for key, _, _ in spec["thresholds"]):
        return _verdict("INCOMPLETE", "missing finite final live metrics")
    grade = test_verdict(spec, {"observations": points, "live": live})
    return _verdict(grade["status"], "recomputed complete live curve and terminal suffix",
                    metrics=live, evaluator_result=grade)


def _native(task, evidence):
    from benchmarks.toy100.accuracy_gate import evaluate_suite
    from .artifacts import verify_artifacts

    artifact = evidence.get("artifact_root")
    if not isinstance(artifact, str) or not artifact:
        return _verdict("INCOMPLETE", "native coverage/accuracy needs saved samples and artifact_root")
    manifest = evidence.get("artifact_manifest")
    if manifest is None:
        return _verdict("INCOMPLETE", "native evaluator inputs require a certified artifact manifest")
    path = Path(artifact)
    verify_artifacts(path, manifest)
    problem = task["execution"]["problem"]
    config = _read(path / problem / "config.json")
    if config.get("steps") != task["execution"]["steps"]:
        return _verdict("INVALID", "native artifact budget differs from declared task")
    if config.get("eval_interval", 250) != task["evaluation"]["eval_interval"]:
        return _verdict("INVALID", "native artifact observation cadence differs from declared task")
    if config.get("eval_samples") != task["evaluation"]["eval_samples"]:
        return _verdict("INVALID", "native evaluation sample count differs from declared task")
    if config.get("early_eval_steps") != task["evaluation"]["early_eval_steps"]:
        return _verdict("INVALID", "native early observation schedule differs from declared task")
    if task["execution"].get("preserve_prefix_steps"):
        prefix = evidence.get("prefix_parity", {})
        if prefix.get("steps") != task["execution"]["preserve_prefix_steps"] or not _digest(prefix.get("reference_sha256")):
            return _verdict("INCOMPLETE", "continuation requires bound original-prefix parity evidence")
        if prefix.get("continued_sha256") != prefix["reference_sha256"]:
            return _verdict("FAIL", "continuation changed the original training prefix")
        from .continuation import verify_native_prefix
        verify_native_prefix(task, evidence)
    grade = evaluate_suite(path, problem=problem, write=False)["problems"][problem]
    verify_artifacts(path, manifest)
    return _verdict(grade["status"], grade["reason"],
                    metrics=grade.get("holdout_metrics", {}), evaluator_result=grade)


def _convergence_class():
    name = "_forge_original_convergence_gate"
    if name not in sys.modules:
        path = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].ConvergenceGate


def _ring(task, evidence):
    evaluation = task["evaluation"]
    points = evidence.get("dense", evidence.get("observations"))
    if not points:
        return _verdict("INCOMPLETE", "dense first-convergence and hold observations are required")
    _curve(points, [["modes", ">=", 8], ["hq", ">=", .9]])
    if any(type(p["modes"]) is not int or not 0 <= p["modes"] <= 8 or not 0 <= p["hq"] <= 1 for p in points):
        return _verdict("INVALID", "ring metrics have invalid mode counts or quality bounds")
    _curve(points, [], range(evaluation["start_step"] + 1, points[-1]["step"] + 1))
    if points[-1]["step"] > evaluation["max_total_steps"]:
        return _verdict("INVALID", "ring observations exceed the frozen total budget")
    gate = _convergence_class()(confirmation=evaluation["confirmation_checks"],
                               settling_budget=evaluation["settling_budget"],
                               hold_budget=evaluation["hold_budget"],
                               start_step=evaluation["start_step"])
    for point in points:
        if gate.done:
            break
        gate.observe(point)
    if gate.status in {"NOT_CONVERGED", "POST_CONVERGENCE_FAIL"}:
        return _verdict("FAIL", "first qualifying window failed acquisition or uninterrupted hold",
                        metrics=gate.summary())
    if gate.status != "PASS":
        return _verdict("INCOMPLETE", "first-convergence hold has not completed", metrics=gate.summary())
    if evaluation["kind"] == "ring_extension":
        extension = [p for p in points if p["step"] > gate.last_step]
        if len(extension) != evaluation["extension_steps"]:
            return _verdict("INCOMPLETE", "the complete immediately following extension is required")
        continuity = evidence.get("continuity", {})
        if continuity.get("mode") != "uninterrupted" or not continuity.get("run_id"):
            return _verdict("BLOCKED", "extension requires bound uninterrupted own-state provenance")
        passing = sum(p["modes"] == 8 and .9 <= p["hq"] <= 1 for p in extension)
        return _verdict("PASS" if passing == len(extension) else "FAIL",
                        "recomputed every extension observation without selecting another window",
                        metrics={**gate.summary(), "extension_checks": len(extension),
                                 "extension_passing": passing})
    return _verdict("PASS", "recomputed first-convergence hold", metrics=gate.summary())


def _adaptation(task, evidence):
    from benchmarks.toy100.continuous_probe import _window, match_frozen_control

    active, frozen = deepcopy(evidence.get("active")), deepcopy(evidence.get("frozen"))
    if not isinstance(active, dict) or not isinstance(frozen, dict):
        return _verdict("INCOMPLETE", "matched active and frozen raw control evidence is required")
    if not evidence.get("artifact_root") or not evidence.get("artifact_manifest"):
        return _verdict("INCOMPLETE", "paired adaptation requires certified own-state and control artifacts")
    from .adaptation import verify_pair_artifacts
    verify_pair_artifacts(evidence)
    evaluation = task["evaluation"]
    required = ("config_sha256", "source_sha256", "runtime", "mode", "noise_horizon",
                "diagnostic_every", "dense_after", "dense_until", "shift", "shift_pair", "optimizer_final")
    for run in (active, frozen):
        if any(key not in run or (run[key] is None and key not in {"dense_after", "dense_until"}) for key in required):
            return _verdict("INCOMPLETE", "paired control lacks complete provenance/optimizer state")
        if run.get("steps") != evaluation["steps"] or run.get("shift_step") != evaluation["shift_step"]:
            return _verdict("INVALID", "shift timing differs from task")
        points = run.get("diagnostic")
        _curve(points, [["modes", ">=", 8], ["hq", ">=", .9]])
        if any(type(p["modes"]) is not int or not 0 <= p["modes"] <= 8 or not 0 <= p["hq"] <= 1 for p in points):
            return _verdict("INVALID", "invalid ring mode count or quality bounds")
        if run["diagnostic_every"] != evaluation["diagnostic_every"]:
            return _verdict("INVALID", "diagnostic observation cadence differs from protocol")
        if [p["step"] for p in points] != list(range(10, evaluation["steps"] + 1, 10)):
            return _verdict("INCOMPLETE", "paired adaptation requires the complete declared diagnostic curve")
        at_shift = run["shift_pair"].get("optimizer_at_shift", [])
        if not at_shift or any(r.get("updates") != evaluation["shift_step"] for r in at_shift):
            return _verdict("INVALID", "optimizer state at the target shift is not preserved")
        by_step = {p["step"]: p for p in points}
        stationary = list(range(1000, 1201, 50))
        continued = list(range(1210, evaluation["shift_step"] + 1, 10))
        recovery = list(range(evaluation["shift_step"] + 10, evaluation["steps"] + 1, 10))
        if any(step not in by_step for step in stationary + continued + recovery):
            return _verdict("INCOMPLETE", "complete stationary, hold and recovery windows are required")
        run["stationary"] = _window([by_step[s] for s in stationary])
        run["continued_hold"] = _window([by_step[s] for s in continued])
        deadline = evaluation["shift_step"] + evaluation["recovery_deadline"]
        late = _window([by_step[s] for s in recovery if s >= deadline])
        run["shift_recovery"] = {"deadline_window": late, "deadline_pass": late["pass_all"]}
    for run in (active, frozen):
        final = run["optimizer_final"]
        expected = evaluation["shift_step"] if run.get("freeze_after_shift") else evaluation["steps"]
        if not isinstance(final, list) or not final or any(r.get("updates") != expected for r in final):
            return _verdict("INVALID", "optimizer updates do not match declared active/frozen continuation")
    grade = match_frozen_control(active, frozen)
    return _verdict(grade["status"], "recomputed live recovery and matched frozen negative control",
                    metrics={"live_deadline": active["shift_recovery"]["deadline_window"],
                             "frozen_deadline": frozen["shift_recovery"]["deadline_window"]})


def _clockfree(task, evidence):
    if not evidence.get("artifact_root") or not evidence.get("artifact_manifest"):
        return _verdict("INCOMPLETE", "clock comparisons require certified saved state artifacts")
    from .clockfree import verify_probe
    comparisons, audit = verify_probe(task, evidence)
    if not isinstance(comparisons, list) or not isinstance(audit, dict):
        return _verdict("INCOMPLETE", "state/horizon parity comparisons and source audit are required")
    conditions = task["evaluation"]["conditions"]
    names = [r.get("condition") for r in comparisons if isinstance(r, dict)]
    if sorted(names) != sorted(conditions):
        return _verdict("INCOMPLETE", "each declared clock/evaluation/restart comparison is required exactly once")
    for row in comparisons:
        for key in ("permitted_state_sha256", "rng_state_sha256", "reference_sha256", "changed_sha256"):
            value = row.get(key)
            if not _digest(value):
                return _verdict("INCOMPLETE", "parity comparisons require state and output digests")
        if row["reference_sha256"] != row["changed_sha256"]:
            return _verdict("FAIL", f"{row['condition']} changed the update or common-prefix state")
    hashes = audit.get("source_sha256")
    dependencies = audit.get("unexplained_clock_dependencies")
    if (not isinstance(hashes, dict) or not hashes or not all(_digest(v) for v in hashes.values())
            or not isinstance(audit.get("allowed_state"), list) or not audit["allowed_state"]
            or not isinstance(dependencies, list)):
        return _verdict("INCOMPLETE", "source audit must bind code and explain permitted state/counters")
    if audit["unexplained_clock_dependencies"]:
        return _verdict("BLOCKED", "unexplained clock dependencies prevent clock-free eligibility")
    return _verdict("PASS", "declared state/horizon/cadence/restart comparisons agree; source audit bound",
                    metrics={"parity_comparisons": len(comparisons)})


def grade_result(task: dict, result: dict | None) -> dict:
    """Independently grade compatible raw evidence, without changing it."""
    try:
        _validate_measurement_contract(task)
    except OSError as exc:
        return _verdict("BLOCKED", "frozen task/source identity is unavailable: " + str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        return _verdict("INVALID", str(exc))
    if result is None:
        return _verdict("NOT_RUN", "no compatible result")
    if not isinstance(result, dict):
        return _verdict("INVALID", "receipt must be an object")
    applicability = result.get("applicability", {})
    if isinstance(applicability, dict) and applicability.get("status") in {"unsupported", "unknown"}:
        return _verdict("BLOCKED", applicability.get("reason") or "host applicability has not been demonstrated",
                        raw_status=result.get("gate_status", result.get("status")))
    raw = result.get("gate_status", result.get("status"))
    if raw == "BLOCKED":
        return _verdict("BLOCKED", result.get("reason", "execution blocked"))
    if raw in {"ERROR", "error", "timeout", "cancelled"} or result.get("error"):
        return _verdict("INCOMPLETE", result.get("error") or f"execution ended with {raw}", raw_status=raw)
    if result.get("gate_status") in {"INCOMPLETE", "INVALID"}:
        return _verdict(result["gate_status"], result.get("reason") or "execution did not produce valid complete evidence")
    evidence = result.get("evidence", result.get("metrics"))
    if not isinstance(evidence, dict):
        return _verdict("INCOMPLETE", "missing raw evaluator evidence; status stamps cannot qualify")
    if evidence.get("scoring_weights", "live") != task["evaluation"].get("scoring_weights", "live"):
        return _verdict("INVALID", "live/EMA scoring policies differ")
    sampling = grade_sampling(task, evidence)
    if sampling is not None:
        return _verdict(sampling["status"], sampling["reason"])
    try:
        policy = _policy_guards(task, evidence)
    except OSError as exc:
        return _verdict("BLOCKED", "frozen policy/source identity is unavailable: " + str(exc))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return _verdict("INVALID", "malformed policy-owner evidence: " + str(exc))
    if policy is not None:
        return policy
    guard = _guards(task, evidence)
    if guard is not None:
        return guard
    graders = {"transfer_sustained": _transfer, "native_accuracy": _native,
               "ring_hold": _ring, "ring_extension": _ring,
               "paired_adaptation": _adaptation, "clockfree_parity": _clockfree}
    grader = graders.get(task["evaluation"]["kind"])
    if grader is None:
        return _verdict("BLOCKED", f"unsupported evaluator {task['evaluation']['kind']}")
    try:
        grade = grader(task, evidence)
        # Normalized metrics must not replace the raw curve when the input used
        # the metrics fallback. The queue can safely merge this grade into it.
        if "evidence" not in result:
            grade["evidence"] = deepcopy(evidence)
        return grade
    except FileNotFoundError as exc:
        return _verdict("INCOMPLETE", f"required saved evidence unavailable: {exc}")
    except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        return _verdict("INVALID", str(exc))


def _candidate_eligibility(view, candidate):
    rules = view.get("eligibility", {})
    required = rules.get("requires_capabilities", [])
    claims = rules.get("claim_contract", {})
    if not required and not claims:
        return []
    if not isinstance(candidate, dict):
        return ["candidate capability/claim metadata is required for this view"]
    available = candidate.get("capabilities", candidate.get("provides_capabilities", []))
    reasons = [f"missing capability {name}" for name in required
               if (available.get(name) is not True if isinstance(available, dict) else name not in available)]
    actual = candidate.get("claim_contract", {})
    reasons.extend(f"claim_contract.{name} must equal {value!r}"
                   for name, value in claims.items() if actual.get(name) != value)
    return reasons


def qualify(view: dict, tasks: dict, results: list[dict], *, candidate: dict | None = None) -> dict:
    """Reduce one candidate's compatible receipts; tier attainment is computed."""
    validate_view(view, tasks)
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    indexed = {}
    for receipt in results:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("task_id"), str):
            raise ValueError("each receipt needs task_id")
        indexed.setdefault(receipt["task_id"], []).append(receipt)
    assignments = sorted(view["assignments"], key=lambda a: (a["qualification_tier"], a["order"], a["task"]))
    rows, selected = {}, {}
    for assignment in assignments:
        name = assignment["task"]
        receipts = indexed.get(name, [])
        grades = [grade_result(tasks[name], receipt) for receipt in receipts]
        if not grades:
            grade = grade_result(tasks[name], None)
        elif len({_hash(grade) for grade in grades}) > 1:
            grade = _verdict("INVALID", "conflicting compatible attempts require explicit resolution")
        else:
            grade = grades[0]
            selected[name] = receipts[0]
        rows[name] = {"task_id": name, "qualification_tier": assignment["qualification_tier"],
                      "importance": assignment["importance"], **grade,
                      "cost": sum(_cost(r) for r in receipts)}
        rows[name].pop("evidence", None)
    # Retiering may place children before parents in presentation order. Resolve
    # prerequisite statuses in dependency order before exposing any PASS rows.
    ordered, visited = [], set()
    def dependency_order(name):
        if name in visited:
            return
        for dep in _dependencies(tasks[name]):
            dependency_order(dep["task"])
        visited.add(name)
        ordered.append(name)
    for name in rows:
        dependency_order(name)
    for name in ordered:
        row = rows[name]
        if row["status"] != "PASS":
            continue
        for dep in _dependencies(tasks[name]):
            if rows[dep["task"]]["status"] != "PASS":
                row.update(_verdict("BLOCKED", f"dependency {dep['task']} has not passed"))
                continue
            if dep["kind"] == "checkpoint":
                child = selected.get(name, {}).get("evidence", {})
                parent = selected.get(dep["task"], {}).get("evidence", {})
                a, b = child.get("continuity", {}), parent.get("continuity", {})
                uninterrupted = a.get("mode") == b.get("mode") == "uninterrupted" and a.get("run_id") and a.get("run_id") == b.get("run_id")
                native_resume = False
                if tasks[name]["adapter"] == "native100_continuation":
                    # grade_result has already verified actual prefix/restore
                    # artifacts. Bind that proof to this selected parent too.
                    proof = child.get("prefix_parity", {})
                    prerequisite = proof.get("prerequisite", {})
                    producer = selected.get(dep["task"], {})
                    native_resume = bool(parent.get("checkpoint") and
                        proof.get("checkpoint") == parent["checkpoint"] and
                        proof.get("artifact_manifest") == parent.get("artifact_manifest") and
                        prerequisite.get("compatibility_key") == producer.get("compatibility_key") and
                        (not producer.get("_attempt_id") or prerequisite.get("attempt_id") == producer["_attempt_id"]))
                if not uninterrupted and not native_resume:
                    reason = ("fresh-process checkpoint restoration has no supported proof contract; uninterrupted continuation required"
                              if a.get("mode") == "verified_checkpoint" else
                              "checkpoint continuation is not bound to the producer's own state")
                    row.update(_verdict("BLOCKED", reason))
    if view.get("evidence_scope") == "calibration_diagnostic":
        return {"view": view["id"], "view_revision": view["revision"], "policy_fingerprint": view_fingerprint(view),
                "status": "DIAGNOSTIC", "qualified_tier": 0, "eligible": False,
                "evidence_scope": "calibration_diagnostic", "current_qualification_reuse": False,
                "diagnostic_complete": all(row["status"] in {"PASS", "FAIL"} for row in rows.values()),
                "tiers": [], "tasks": list(rows.values()),
                "task_statuses": {name: row["status"] for name, row in rows.items()},
                "next_task": None, "next_tasks": [], "blockers": [], "required_passed": 0, "required_total": 0,
                "cost_seconds": sum(row["cost"] for row in rows.values())}
    eligibility = _candidate_eligibility(view, candidate)
    tiers, qualified_tier, next_tasks, blockers = [], 0, [], []
    stopped = False
    for tier in sorted({a["qualification_tier"] for a in assignments}):
        subset = [rows[a["task"]] for a in assignments if a["qualification_tier"] == tier]
        required = [row for row in subset if row["importance"] == "required"]
        counts = dict(Counter(row["status"] for row in required))
        all_pass = all(row["status"] == "PASS" for row in required)
        status = "PASS" if all_pass else next((s for s in ("FAIL", "INVALID", "BLOCKED", "INCOMPLETE", "NOT_RUN") if counts.get(s)), "INCOMPLETE")
        tiers.append({"qualification_tier": tier, "status": status,
                      "required_passed": counts.get("PASS", 0), "required_total": len(required),
                      "counts": counts, "diagnostic_total": sum(r["importance"] == "diagnostic" for r in subset)})
        if stopped:
            continue
        if all_pass and not eligibility:
            qualified_tier = tier
            continue
        stopped = True
        blockers = [{"task_id": row["task_id"], "status": row["status"], "reasons": row["reasons"]}
                    for row in required if row["status"] != "PASS"]
        hard_stop = eligibility or any(row["status"] in {"FAIL", "INVALID", "BLOCKED", "INCOMPLETE"} for row in required)
        if not hard_stop:
            next_tasks = [row["task_id"] for row in required if row["status"] == "NOT_RUN"
                          and all(rows[dep["task"]]["status"] == "PASS" for dep in _dependencies(tasks[row["task_id"]]))]
    if eligibility:
        blockers = [{"task_id": None, "status": "BLOCKED", "reasons": eligibility}, *blockers]
    status = ("PASS" if qualified_tier == len(tiers) else "BLOCKED" if eligibility else
              next((t["status"] for t in tiers if t["qualification_tier"] > qualified_tier), "INCOMPLETE"))
    if status == "NOT_RUN":
        status = "INCOMPLETE"
    return {"view": view["id"], "view_revision": view["revision"], "policy_fingerprint": view_fingerprint(view),
            "status": status, "qualified_tier": qualified_tier, "eligible": status == "PASS",
            "tiers": tiers, "tasks": list(rows.values()), "task_statuses": {name: row["status"] for name, row in rows.items()},
            "next_task": next_tasks[0] if next_tasks else None, "next_tasks": next_tasks,
            "blockers": blockers, "required_passed": sum(row["status"] == "PASS" for row in rows.values() if row["importance"] == "required"),
            "required_total": sum(row["importance"] == "required" for row in rows.values()),
            "cost_seconds": sum(row["cost"] for row in rows.values())}
