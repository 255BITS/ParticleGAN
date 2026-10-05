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
DIAGNOSTIC_SCOPES = frozenset({"calibration_diagnostic", "research_diagnostic"})


def diagnostic_evidence_scope(request: dict) -> str | None:
    """Keep diagnostic evidence nonqualifying across saved-request reducers."""
    markers = {request.get("view", {}).get("evidence_scope")}
    markers.update(job.get("science", {}).get("evidence_use") for job in request.get("jobs", []))
    if request.get("calibration_lane") or "calibration_diagnostic" in markers:
        return "calibration_diagnostic"
    return "research_diagnostic" if "research_diagnostic" in markers else None


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def task_execution_fingerprint(task: dict) -> str:
    """Scientific execution identity deliberately excludes view policy."""
    return _hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


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
    if task.get("task_cohort") == "tier1_policy_selected_cloud_v1":
        from .tier1_policy import validate
        validate(task)
        fixed["scoring_weights"] = "state_selected"
    if kind == "transfer_sustained":
        from benchmarks.locked_shared.observation import OBSERVATIONS, MIN_STABLE_CHECKS
        fixed.update(evaluator="benchmarks.transfer_suite.protocol:test_verdict",
                     observations=OBSERVATIONS, minimum_stable_checks=MIN_STABLE_CHECKS)
    elif kind == "transfer_budget_diagnostic":
        from .vector_budget_diagnostics import validate
        validate(task)
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
    elif kind in {"clockfree_parity", "schedule_contract"}:
        fixed.update(evaluator=("experiments.forge.views:_clockfree" if kind == "clockfree_parity"
                                else "experiments.forge.views:_schedule_contract"), training_feedback=False)
        conditions = evaluation.get("conditions")
        if not isinstance(conditions, list) or sorted(conditions) != sorted(("step_label", "horizon", "evaluation_cadence", "restart")):
            raise ValueError(f"{task['id']}: all four fixed clock-free comparisons are required")
        if kind == "schedule_contract":
            from .clockfree import SCHEDULE_CLOCKS
            fixed.update(schedule_tolerance=1e-12, guard_relative_tolerance=1e-6, clockfree_claim=False)
            execution = task["execution"]
            if execution.get("schedule_clocks") != SCHEDULE_CLOCKS:
                raise ValueError(f"{task['id']}: schedule clocks differ from the independently audited laws")
            if execution.get("steps") != execution.get("warmup_steps", 0) + 8 * execution.get("probe_steps", 0):
                raise ValueError(f"{task['id']}: budget must include all five probes and three schedule oracle replays")
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
    from .tier1_policy import load_variants
    for name, task in load_variants(root, tasks).items():
        _validate_task(task)
        tasks[name] = task
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
    diagnostic = view.get("evidence_scope") in DIAGNOSTIC_SCOPES
    if diagnostic and any(a["importance"] != "diagnostic" or a["qualification_tier"] != 1 for a in assignments):
        raise ValueError("diagnostic views require only diagnostic tasks in Tier 1")
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
        if task["execution"].get("clock_audit_scope") == "measure_known_dependencies":
            return _verdict("FAIL", "; ".join(audit["unexplained_clock_dependencies"]),
                            metrics={"clock_dependency_count": len(audit["unexplained_clock_dependencies"]), "parity_comparisons": len(comparisons)})
        return _verdict("BLOCKED", "unexplained clock dependencies prevent clock-free eligibility")
    return _verdict("PASS", "declared state/horizon/cadence/restart comparisons agree; source audit bound",
                    metrics={"parity_comparisons": len(comparisons)})


def _schedule_contract(task, evidence):
    if not evidence.get("artifact_root") or not evidence.get("artifact_manifest"):
        return _verdict("INCOMPLETE", "schedule contract requires saved actual controls and public states")
    from .clockfree import verify_schedule_probe
    comparisons, audit, metrics, blockers = verify_schedule_probe(task, evidence)
    if blockers:
        return _verdict("BLOCKED", "; ".join(blockers), metrics=metrics)
    if (metrics["maximum_schedule_error"] > 1e-12 or metrics["maximum_guard_relative_error"] > 1e-6
            or metrics["schedule_replay_state_mismatches"] or metrics["restart_cadence_failures"]):
        return _verdict("FAIL", "actual controls, normalized schedule replay or exact cadence/restart differ", metrics=metrics)
    return _verdict("PASS", "independent scheduled controls and normalized public replay agree; exact cadence/restart parity; no clock-free claim",
                    metrics=metrics)


def grade_result(task: dict, result: dict | None) -> dict:
    """Independently grade compatible raw evidence, without changing it."""
    try:
        _validate_measurement_contract(task)
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
    if task.get("task_cohort") == "tier1_policy_selected_cloud_v1" and task["adapter"] != "clockfree_audit":
        from .tier1_policy import validate_evidence
        policy_grade = validate_evidence(task, evidence)
        if policy_grade is not None:
            return _verdict(policy_grade["status"], policy_grade["reason"])
    guard = _guards(task, evidence)
    if guard is not None:
        return guard
    from .vector_budget_diagnostics import grade as budget_grade
    graders = {"transfer_sustained": _transfer, "transfer_budget_diagnostic": budget_grade,
               "native_accuracy": _native,
               "ring_hold": _ring, "ring_extension": _ring,
               "paired_adaptation": _adaptation, "clockfree_parity": _clockfree,
               "schedule_contract": _schedule_contract}
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
    if view.get("evidence_scope") in DIAGNOSTIC_SCOPES:
        return {"view": view["id"], "view_revision": view["revision"], "policy_fingerprint": view_fingerprint(view),
                "status": "DIAGNOSTIC", "qualified_tier": 0, "eligible": False,
                "evidence_scope": view["evidence_scope"], "current_qualification_reuse": False,
                "qualification_input": False, "qualification_reuse": False,
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
