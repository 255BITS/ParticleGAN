"""Bounded later training evidence, separate from immutable deadline grades.

This reporting capability consumes certified adapter receipts; it schedules no
work and does not replace any original grader or passing-parent dependency.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import math
import operator

from .contracts import read_json, stable_hash
from .family_reports import evaluation_rows

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
       ">=": operator.ge, "==": operator.eq}
NONTRAINING = {"clockfree_parity", "api_parity", "source_audit",
               "parameter_parity", "api_contract"}
KINDS = {"gaussian_smoke", "word_smoke", "word_hold", "gaussian_stability",
         "native_accuracy", "transfer_sustained", "ring_hold",
         "ring_extension"}
SCHEMAS = {"public_context_v1", "joint_word_v1"}


def applicability(task, original):
    evaluation = task["evaluation"]
    if evaluation["kind"] in NONTRAINING:
        return {"status": "NOT_APPLICABLE", "reason": "Nontraining gate has no later training convergence."}
    if task["adapter"] == "transfer_behavior":
        return {"status": "BLOCKED", "reason": "Behavior host has provenance state but no supported restore/resume contract."}
    checkpoint = original.get("checkpoint", {})
    if checkpoint.get("schema") not in SCHEMAS:
        return {"status": "BLOCKED", "reason": "Certified resumable checkpoint schema is unavailable or unsupported."}
    if evaluation["kind"] not in KINDS or not evaluation_rows(evaluation):
        return {"status": "BLOCKED", "reason": "No frozen numerical extension receipt adapter is defined."}
    return {"status": "SUPPORTED", "reason": "Reporting and exact prefix verification supported; no extension executor is installed by this capability."}


def phase_windows(task, scope):
    evaluation = task["evaluation"]
    kind = evaluation["kind"]
    if kind in {"gaussian_smoke", "word_smoke"}:
        if scope != "confirmed_acquisition":
            raise ValueError("smoke confirms acquisition at one state, not sustained convergence")
        return {"confirmed_acquisition": 1}
    if scope != "eventual_recovery_window":
        raise ValueError("later windows must be separately named eventual_recovery_window")
    if kind == "gaussian_stability":
        return {"reacquisition": evaluation["minimum_stable_checks"],
                "shift_hold": evaluation["shift_hold_checks"]}
    if kind == "native_accuracy":
        return {"terminal_coverage": evaluation["minimum_stable_checks"], "independent_holdout": 1}
    if kind in {"ring_hold", "ring_extension"}:
        windows = {"confirmed_acquisition": evaluation["confirmation_checks"],
                   "first_later_hold": evaluation["hold_budget"]}
        if kind == "ring_extension":
            windows["endurance_extension"] = evaluation["extension_steps"]
        return windows
    return {"extended_hold": evaluation["observations"]}


def phase_bounds(task, windows):
    bounds = evaluation_rows(task["evaluation"])
    if task["evaluation"]["kind"] == "native_accuracy":
        return {"terminal_coverage": [row for row in bounds if row[0].startswith("coverage.")],
                "independent_holdout": [row for row in bounds if row[0].startswith("accuracy.")]}
    return {phase: bounds for phase in windows}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("receipt timestamp must include timezone")
    return parsed


def plan_extension(task, original, *, maximum_extra_updates, maximum_wall_seconds,
                   observation_extra_steps, registered_at, scope="eventual_recovery_window"):
    state = applicability(task, original)
    if state["status"] != "SUPPORTED":
        return {"schema_version": 1, "applicability": state,
                "original_result": deepcopy(original), "policy": None}
    timestamp(registered_at)
    if (type(maximum_extra_updates) is not int or maximum_extra_updates <= 0
            or type(maximum_wall_seconds) not in (int, float)
            or not math.isfinite(maximum_wall_seconds) or maximum_wall_seconds <= 0):
        raise ValueError("extension needs finite positive update/wall caps")
    if (original["gate_status"] not in {"PASS", "FAIL", "INCOMPLETE"}
            or type(original["executed_updates"]) is not int or original["executed_updates"] < 0
            or type(original["checkpoint"]["step"]) is not int or original["checkpoint"]["step"] < 0):
        raise ValueError("original grade/checkpoint/work clocks are inconsistent")
    if not original.get("observation_steps"):
        raise ValueError("preserve the original observation positions explicitly")
    windows = phase_windows(task, scope)
    by_phase = observation_extra_steps
    if not isinstance(by_phase, dict) and len(windows) == 1:
        by_phase = {next(iter(windows)): list(by_phase)}
    if not isinstance(by_phase, dict) or set(by_phase) != set(windows):
        raise ValueError("preregister each required phase suffix explicitly")
    schedule = []
    previous_end = 0
    for phase, count in windows.items():
        steps = list(by_phase[phase])
        if (len(steps) < count or steps != sorted(set(steps))
                or any(type(n) is not int or not 0 < n <= maximum_extra_updates for n in steps)):
            raise ValueError("phase schedule needs its full window, unique ordered integer steps within cap")
        if phase in {"shift_hold", "extended_hold", "first_later_hold", "endurance_extension"} and len(steps) != count:
            raise ValueError("the predeclared hold requires every observation, not a selected later suffix")
        interval = task["evaluation"].get("eval_interval")
        if interval and any((original["checkpoint"]["step"] + n) % interval for n in steps):
            raise ValueError("extension observation cadence differs from original interval")
        if task["evaluation"]["kind"] in {"ring_hold", "ring_extension"}:
            if steps != list(range(previous_end + 1, steps[-1] + 1)):
                raise ValueError("ring acquisition/hold/extension must retain every dense update check")
        if task["evaluation"]["kind"] == "native_accuracy":
            if phase == "independent_holdout" and steps != [previous_end]:
                raise ValueError("independent holdout must certify the terminal coverage checkpoint")
        elif steps[0] <= previous_end:
            raise ValueError("later phases must follow the preceding phase without retrospective selection")
        previous_end = steps[-1]
        schedule.extend({"phase": phase, "extra_step": n} for n in steps)
    schedule.sort(key=lambda row: (row["extra_step"], row["phase"]))
    policy = dict(schema_version=1, kind="failure_eligible_training_extension_v1",
                  scope=scope, registered_at=registered_at, task_id=task["id"],
                  original_result_sha256=stable_hash(original),
                  execution_sha256=stable_hash(task["execution"]),
                  evaluation_sha256=stable_hash(task["evaluation"]),
                  original_schedule_horizon=original["schedule_horizon"],
                  checkpoint=deepcopy(original["checkpoint"]),
                  original_executed_updates=original["executed_updates"],
                  original_execution_group_ids=deepcopy(original.get("execution_group_ids", [])),
                  original_observation_steps=deepcopy(original["observation_steps"]),
                  append_observation_schedule=schedule,
                  maximum_extra_updates=maximum_extra_updates,
                  maximum_wall_seconds=maximum_wall_seconds,
                  required_windows=windows, numerical_bounds_by_phase=phase_bounds(task, windows),
                  independent_same_state_confirmation_required=task["evaluation"].get("confirmation") == "independent_draw_same_state",
                  independent_holdout_samples=task["evaluation"].get("holdout_samples"),
                  retain_original_phase_evidence=task["evaluation"]["kind"] in {"ring_hold", "ring_extension"})
    policy.update({key: original[key] for key in ("attempt_id", "candidate_revision", "source_digest", "protocol_sha256")})
    return {"schema_version": 1, "applicability": state,
            "original_result": deepcopy(original), "policy": policy}


def unique_attempt_cost(attempts):
    """Charge each execution group once, including interrupted and retry work."""
    seen = {}
    updates, seconds = 0, 0.0
    if not attempts:
        raise ValueError("measured execution receipts are required")
    for attempt in attempts:
        key = attempt["execution_attempt_id"]
        if not isinstance(key, str) or not key:
            raise ValueError("execution group identity is required")
        if key in seen:
            if seen[key] != attempt:
                raise ValueError("shared execution group has conflicting cost receipts")
            continue
        if (type(attempt["executed_updates"]) is not int or attempt["executed_updates"] < 0
                or type(attempt["wall_seconds"]) not in (int, float)
                or not math.isfinite(attempt["wall_seconds"]) or attempt["wall_seconds"] < 0):
            raise ValueError("actual attempt work/cost must be measured and nonnegative")
        seen[key] = attempt
        updates += attempt["executed_updates"]
        seconds += attempt["wall_seconds"]
    return updates, seconds


def numerical_pass(values, bounds):
    for name, op, limit in bounds:
        value = values.get(name)
        allowed = (bool,) if type(limit) is bool else (int, float)
        if (type(value) not in allowed or not math.isfinite(value)
                or op not in OPS or not OPS[op](value, limit)):
            return False
    return bool(bounds)


def evaluate_extension(task, original, policy, continuation):
    # Regenerate the policy from immutable inputs to reject edited criteria,
    # windows, clocks and schedules even when a replacement hash is supplied.
    by_phase = {phase: [row["extra_step"] for row in policy["append_observation_schedule"]
                        if row["phase"] == phase] for phase in policy["required_windows"]}
    expected_policy = plan_extension(task, original,
        maximum_extra_updates=policy["maximum_extra_updates"], maximum_wall_seconds=policy["maximum_wall_seconds"],
        observation_extra_steps=by_phase, registered_at=policy["registered_at"], scope=policy["scope"])["policy"]
    if stable_hash(expected_policy) != stable_hash(policy):
        raise ValueError("immutable original execution/result/criteria or preregistered policy differs")
    if continuation["policy_sha256"] != stable_hash(policy):
        raise ValueError("continuation is not bound to the preregistered extension policy")
    if timestamp(continuation["started_at"]) <= timestamp(policy["registered_at"]):
        raise ValueError("extension was not preregistered before execution")
    for key in ("attempt_id", "candidate_revision", "source_digest", "protocol_sha256"):
        actual = continuation["parent_attempt_id"] if key == "attempt_id" else continuation[key]
        if actual != policy[key]:
            raise ValueError("extension source/candidate/protocol/parent binding differs")
    proof = continuation["prefix_verification"]
    if (continuation["parent_checkpoint"] != policy["checkpoint"]
            or proof.get("policy_sha256") != stable_hash(policy)
            or proof.get("restored_exactly") is not True or proof.get("history_reset") is not False
            or proof.get("checkpoint_step") != policy["checkpoint"]["step"]
            or proof.get("checkpoint_sha256") != policy["checkpoint"]["sha256"]
            or proof.get("checkpoint_state_sha256") != policy["checkpoint"]["state_sha256"]
            or proof.get("restored_state_sha256") != policy["checkpoint"]["state_sha256"]
            or continuation["schedule_horizon"] != policy["original_schedule_horizon"]):
        raise ValueError("extension lacks exact checkpoint/optimizer/RNG/horizon continuity")
    if any(attempt["execution_attempt_id"] in policy["original_execution_group_ids"] for attempt in continuation["attempts"]):
        raise ValueError("original shared execution work cannot be charged again as extension work")
    cost_updates, cost_seconds = unique_attempt_cost(continuation["attempts"])
    if cost_updates > policy["maximum_extra_updates"] or cost_seconds > policy["maximum_wall_seconds"]:
        raise ValueError("all interrupted/retry work counts against the bounded extension")
    if type(continuation["completed_steps"]) is not int:
        raise ValueError("accepted update clock must be an integer")
    extra = continuation["completed_steps"] - policy["checkpoint"]["step"]
    if not 0 <= extra <= cost_updates:
        raise ValueError("checkpoint-relative clock differs from charged work")
    observations = continuation["observations"]
    actual_schedule = [{"phase": row["phase"], "extra_step": row["extra_step"]} for row in observations]
    expected_schedule = [row for row in policy["append_observation_schedule"] if row["extra_step"] <= extra]
    if actual_schedule != expected_schedule:
        raise ValueError("missing/reordered/duplicate/off-cadence/beyond-cap evidence")
    if continuation["preserved_original_observation_steps"] != policy["original_observation_steps"]:
        raise ValueError("original observation positions must be preserved, not recomputed")
    if (policy["retain_original_phase_evidence"]
            and continuation.get("original_phase_evidence_sha256") != stable_hash(original.get("phase_evidence"))):
        raise ValueError("original first-hold/shift/frozen evidence must remain immutable")
    final = continuation["final_checkpoint"]
    if (final["step"] != continuation["completed_steps"] or not final.get("sha256")
            or not final.get("state_sha256") or continuation.get("final_state_verified") is not True):
        raise ValueError("final checkpoint must certify the actual end state")
    phases, all_sustained = {}, True
    first_hits, confirmations = [], []
    for phase, count in policy["required_windows"].items():
        points = [row for row in observations if row["phase"] == phase]
        passes = []
        for row in points:
            if (row["evaluation_sha256"] != policy["evaluation_sha256"]
                    or row.get("independent_evaluator") is not True or row.get("guards_passed") is not True
                    or not row.get("state_sha256") or not row.get("artifact_sha256")):
                raise ValueError("original independent evaluator, execution guards and certified samples are required")
            passing = numerical_pass(row["values"], policy["numerical_bounds_by_phase"][phase])
            if policy["independent_same_state_confirmation_required"]:
                passing = (passing and row.get("independent_same_state_confirmation") is True
                           and row.get("confirmation_state_sha256") == row["state_sha256"]
                           and numerical_pass(row.get("confirmation_values", {}), policy["numerical_bounds_by_phase"][phase]))
            if phase == "independent_holdout":
                passing = passing and row.get("sample_count") == policy["independent_holdout_samples"]
            passes.append(passing)
        first = next((point["extra_step"] for point, passed in zip(points, passes) if passed), None)
        suffix = 0
        for passed in reversed(passes):
            if not passed:
                break
            suffix += 1
        sustained = next((points[i]["extra_step"] for i in range(count - 1, len(points))
                          if all(passes[i - count + 1:i + 1])), None)
        terminal = bool(passes and passes[-1])
        qualified = terminal and suffix >= count
        phases[phase] = dict(first_hit_extra_step=first, first_sustained_extra_step=sustained,
                            terminal_passing_checks=suffix, required_checks=count,
                            end_state="PASS" if terminal else "FAIL" if points else "UNMEASURED",
                            qualified_later=qualified, unit="observation_checks")
        all_sustained = all_sustained and qualified
        if first is not None:
            first_hits.append(first)
        if sustained is not None:
            confirmations.append(sustained)
    complete = (continuation["status"] == "completed"
                and extra == policy["append_observation_schedule"][-1]["extra_step"])
    if complete and any(row["state_sha256"] != final["state_sha256"] for row in observations if row["extra_step"] == extra):
        raise ValueError("end-state observation does not match final checkpoint")
    status = "PASS" if complete and all_sustained else "FAIL" if complete else "INCOMPLETE"
    result = dict(schema_version=1, original_grade=original["gate_status"],
                  original_result_sha256=policy["original_result_sha256"], scope=policy["scope"], extended_status=status,
                  extra_training_updates=extra, extra_executed_updates_including_retries=cost_updates,
                  checkpoint_step=policy["checkpoint"]["step"], absolute_end_step=continuation["completed_steps"],
                  absolute_first_hit_step=policy["checkpoint"]["step"] + min(first_hits) if first_hits else None,
                  absolute_first_confirmation_step=policy["checkpoint"]["step"] + max(confirmations) if len(confirmations) == len(phases) else None,
                  total_charged_training_updates=policy["original_executed_updates"] + cost_updates,
                  measured_extra_wall_seconds=cost_seconds, phases=phases,
                  original_qualification_changed=False, benchmark_tasks_passed_original=int(original["gate_status"] == "PASS"),
                  continuation_provenance={key: deepcopy(continuation[key]) for key in
                    ("parent_attempt_id", "parent_checkpoint", "final_checkpoint", "prefix_verification",
                     "candidate_revision", "source_digest", "protocol_sha256", "schedule_horizon", "policy_sha256")})
    return {"original_result": deepcopy(original), "extended_result": result}


def render_status(wrapper):
    original = wrapper["original_result"]["gate_status"]
    result = wrapper.get("extended_result")
    if not result:
        return original + " (original deadline; extended result unmeasured)"
    return (f"{original} (original deadline); {result['extended_status']} after "
            f"+{result['extra_training_updates']:,} training updates ({result['scope']}); "
            "end/window status retained separately")


def summarize_row(row):
    state = applicability(row["task"], row["original"])
    if state["status"] != "SUPPORTED":
        return {"original_result": deepcopy(row["original"]), "applicability": state,
                "extended_result": None}
    return evaluate_extension(row["task"], row["original"], row["policy"], row["continuation"])


def board_rows(root):
    path = Path(root) / "reports/forge/extended-result-board.json"
    if not path.exists():
        return []
    data = read_json(path)
    lines = ["## Bounded extended results", "",
             "Original task grades and totals remain immutable. Later windows are separate; first hits are not sustained qualification.", "",
             "| Task / source | Original deadline | Extended result | First hit / sustained confirmation | End windows | Checkpoint / absolute end | Accepted / charged extra updates | Extra measured wall seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in data["rows"]:
        wrapper = summarize_row(row)
        result = wrapper["extended_result"]
        label = f"{row['task']['id']} / `{row['original']['source_digest'][:12]}`"
        if result is None:
            state = wrapper["applicability"]
            lines.append(f"| {label} | {row['original']['gate_status']} | {state['status']}: {state['reason']} | — | — | — | 0 / 0 | 0 |")
            continue
        windows = "; ".join(f"{name}: {phase['end_state']}, {phase['terminal_passing_checks']}/{phase['required_checks']} observation checks"
                            for name, phase in result["phases"].items())
        lines.append(f"| {label} | {row['original']['gate_status']} | {render_status(wrapper)} | "
                     f"{result['absolute_first_hit_step']} / {result['absolute_first_confirmation_step']} absolute updates | {windows} | "
                     f"{result['checkpoint_step']} / {result['absolute_end_step']} updates | "
                     f"{result['extra_training_updates']} / {result['extra_executed_updates_including_retries']} updates | "
                     f"{result['measured_extra_wall_seconds']:.3f} |")
    return lines + [""]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="Normalized certified adapter receipt containing task/original/policy/continuation")
    parser.add_argument("--output", type=Path, help="Write separate result wrapper; never write into original receipt")
    args = parser.parse_args()
    if args.output and (args.output.resolve() == args.receipt.resolve()
                        or (args.output.exists() and args.output.samefile(args.receipt))):
        parser.error("the original receipt is immutable; choose a separate output")
    wrapper = summarize_row(read_json(args.receipt))
    payload = json.dumps(wrapper, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
