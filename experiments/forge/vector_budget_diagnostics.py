"""Longer vector acquisition with the exact original schedule and observations.

These tasks answer a budget question only. Their numerical thresholds remain
those of the original task, and their separate view grants no qualification.
"""
import math
from pathlib import Path

from .contracts import file_hash, read_json, stable_hash
from .state import state_digest


KIND = "transfer_budget_diagnostic"
EVALUATOR = "experiments.forge.vector_budget_diagnostics:grade"
ROOT = Path(__file__).resolve().parents[2]


def validate(task):
    """Return the explicit schedule, rejecting ignored or contradictory fields."""
    execution, evaluation = task["execution"], task["evaluation"]
    declaration = execution.get("budget_diagnostic")
    if declaration is None and evaluation["kind"] != KIND:
        return None
    if (task["adapter"] != "transfer_vector" or evaluation["kind"] != KIND
            or not isinstance(declaration, dict)
            or set(declaration) != {"schema_version", "kind", "original_task", "prefix_steps"}
            or type(declaration["schema_version"]) is not int or declaration["schema_version"] != 1
            or declaration["kind"] != "preserve_vector_prefix_v1"):
        raise ValueError("unsupported schedule-preserving vector budget diagnostic")
    original = declaration["original_task"]
    if not isinstance(original, str) or Path(original).name != original or original in {"", ".", ".."}:
        raise ValueError("budget diagnostic requires a filename-safe original_task")
    prefix, limit = declaration["prefix_steps"], execution["steps"]
    if type(prefix) is not int or prefix < 24 or type(limit) is not int or limit <= prefix:
        raise ValueError("budget diagnostic requires at least 24 original updates and a larger execution budget")
    if (type(execution.get("original_schedule_horizon")) is not int
            or execution["original_schedule_horizon"] != prefix
            or execution["host_definition"]["steps"] != prefix):
        raise ValueError("budget diagnostic must preserve the original host and schedule horizon")
    prefix_checks = sorted({math.ceil(i * prefix / 24) for i in range(1, 25)})
    declared = evaluation.get("observation_steps")
    if not isinstance(declared, list) or len(declared) != 29 or declared[:24] != prefix_checks:
        raise ValueError("budget diagnostic must retain all 24 original observation checkpoints plus five terminal checks")
    terminal = declared[24:]
    if (any(type(step) is not int or step <= prefix or step > limit for step in terminal)
            or any(b <= a for a, b in zip(terminal, terminal[1:])) or terminal[-1] != limit):
        raise ValueError("budget diagnostic requires five increasing later checks ending at its execution limit")
    fixed = {"evaluator": EVALUATOR, "observations": 29, "minimum_stable_checks": 5,
             "scoring_weights": "live"}
    if any(evaluation.get(key) != value for key, value in fixed.items()):
        raise ValueError("budget diagnostic evaluator, observation count and five-check live gate are fixed")
    return {"original_task": original, "prefix_steps": prefix, "execution_steps": limit,
            "original_schedule_horizon": prefix, "prefix_observation_steps": prefix_checks,
            "terminal_observation_steps": terminal, "observation_steps": declared}


def validate_original(task, *, root=None):
    """Freeze the host, data, initialization, prior and bounds against its parent."""
    contract = validate(task)
    if contract is None:
        return
    root = ROOT if root is None else Path(root)
    relative = f"configs/forge/tasks/{contract['original_task']}.json"
    path = root / relative
    original = read_json(path)
    if task["evaluation"].get("sources", {}).get(relative) != file_hash(path):
        raise ValueError("budget diagnostic original task source changed; declare a new diagnostic revision")
    if original["execution"]["steps"] != contract["prefix_steps"]:
        raise ValueError("budget diagnostic prefix differs from its original task budget")
    for key in ("host", "host_source", "host_definition", "prior", "initializer", "protocol"):
        if task["execution"].get(key) != original["execution"].get(key):
            raise ValueError(f"budget diagnostic changed original execution.{key}")
    for key in ("thresholds", "sample_evaluator", "gate_policy", "sampling_law", "eval_output_noise"):
        if task["evaluation"].get(key) != original["evaluation"].get(key):
            raise ValueError(f"budget diagnostic changed original evaluation.{key}")


def prefix_receipt(context, observations, records):
    """Hash the original full learned state and every stream after prefix scoring.

    Only the external execution cap is excluded: it is intentionally larger.
    Keep the recipe horizon, update labels, optimizer histories and eval RNG.
    """
    state = context.state_dict()
    state["trainer"].pop("max_steps", None)
    return {"completed_steps": context._trainer.completed_steps,
            "context_except_execution_cap_sha256": state_digest(state),
            "named_rng_sha256": state_digest(state["streams"]),
            "observations_sha256": stable_hash(observations),
            "scored_samples_sha256": state_digest(records)}


def grade(task, evidence):
    """Recompute the unrelaxed gate at all five new terminal observations."""
    from benchmarks.locked_shared import baseline
    from benchmarks.locked_shared.observation import sustained
    from .views import _curve, _verdict

    contract = validate(task)
    points = evidence.get("observations")
    if not points:
        return _verdict("INCOMPLETE", "missing schedule-preserving budget observations")
    _curve(points, task["evaluation"]["thresholds"])
    if [point["step"] for point in points] != contract["observation_steps"]:
        return _verdict("INCOMPLETE", "all 24 original and five diagnostic terminal checks are required")
    proof = evidence.get("budget_diagnostic", {})
    if any(proof.get(key) != contract[key] for key in contract):
        return _verdict("INVALID", "executed budget/schedule contract differs from its declaration")
    if (proof.get("trainer_execution_limit") != contract["execution_steps"]
            or proof.get("completed_steps") != contract["execution_steps"]
            or "recipe_schedule_horizon" not in proof
            or proof["recipe_schedule_horizon"] not in (None, contract["original_schedule_horizon"])):
        return _verdict("INVALID", "actual trainer duration or recipe schedule differs from its declared budget")
    prefix = proof.get("prefix", {})
    expected_prefix = points[:24]
    if (prefix.get("completed_steps") != contract["prefix_steps"]
            or prefix.get("observations_sha256") != stable_hash(expected_prefix)):
        return _verdict("INVALID", "original-prefix evidence differs from retained numerical observations")
    for key in ("context_except_execution_cap_sha256", "named_rng_sha256", "scored_samples_sha256"):
        value = prefix.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return _verdict("INCOMPLETE", "missing original-prefix state/RNG/sample identity")
    live = evidence.get("live")
    if live != points[-1]:
        return _verdict("INVALID", "final live metrics differ from the final recorded diagnostic observation")
    thresholds = task["evaluation"]["thresholds"]
    convergence = sustained(points, thresholds, expected_steps=contract["observation_steps"])
    prefix_result = sustained(expected_prefix, thresholds, expected_steps=contract["prefix_observation_steps"])
    terminal_cells = [baseline.score_metrics(point, thresholds) for point in points[-5:]]
    passed = all(cell["status"] == "PASS" for cells in terminal_cells for cell in cells)
    return _verdict("PASS" if passed else "FAIL", "recomputed five late checks with the original numerical bounds",
                    metrics=live, evaluator_result={"convergence": convergence, "terminal_metrics": terminal_cells},
                    original_prefix_result=prefix_result)
