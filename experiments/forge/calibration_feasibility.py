"""Read-only feasibility of frozen calibration criteria; never adoption credit.

Unknown scientific decisions are possible completions, not measured outcomes.
This necessary-condition check can prove that a profile cannot be accepted; it
cannot establish that an unmeasured completion will pass or authorize training.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import math

from .contracts import file_hash, identifier, read_json


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def decision_feasibility(matrix: list[dict], criteria: dict) -> dict:
    """Solve possible joint decision counts without changing a saved decision.

    Every eventual complete lineage has one of four smoke/reference outcomes.
    Dynamic programming over their counts avoids a Cartesian enumeration of
    individual completions. Diagnostic tasks are deliberately never inspected.
    """
    for key in ("minimum_paired_lineages", "minimum_reference_positives", "minimum_reference_negatives"):
        if type(criteria.get(key)) is not int or criteria[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("minimum_paired_fraction", "maximum_false_accept_fraction", "maximum_false_reject_fraction"):
        if not 0 <= _number(criteria.get(key), key) <= 1:
            raise ValueError(f"{key} must be at most one")
    for key in ("maximum_smoke_wall_seconds", "maximum_smoke_to_reference_wall_ratio"):
        if _number(criteria.get(key), key) == 0:
            raise ValueError(f"{key} must be positive")
    if not matrix or len({row["id"] for row in matrix}) != len(matrix):
        raise ValueError("feasibility needs nonempty distinct lineage identities")
    cohorts = {row.get("cohort_sha256") for row in matrix}
    if len(cohorts) != 1:
        raise ValueError("calibration feasibility cannot pool scientific cohorts")

    outcomes = {("PASS", "PASS"): 0, ("PASS", "FAIL"): 1,
                ("FAIL", "FAIL"): 2, ("FAIL", "PASS"): 3}
    options, reasons, unknowns, missing_cost = [], [], 0, []
    for row in matrix:
        choices = []
        for part in ("smoke", "reference"):
            decision = row[part]["decision"]
            if decision not in {"PASS", "FAIL", "UNKNOWN"}:
                raise ValueError("only scientific PASS/FAIL or explicit UNKNOWN decisions may be completed")
            unknowns += decision == "UNKNOWN"
            choices.append(("PASS", "FAIL") if decision == "UNKNOWN" else (decision,))
            cost = row[part].get("cost", {})
            actual = cost.get("wall_seconds")
            if actual is None:
                missing_cost.append({"lineage": row["id"], "part": part})
            else:
                _number(actual, part + " wall seconds")
            lower_bound = cost.get("known_wall_seconds", actual)
            if lower_bound is not None:
                _number(lower_bound, part + " known wall seconds")
            if part == "smoke" and lower_bound is not None and lower_bound > criteria["maximum_smoke_wall_seconds"]:
                reasons.append(f"{row['id']}: already-paid smoke cost exceeds the frozen smoke ceiling")
        smoke = row["smoke"].get("cost", {}).get("wall_seconds")
        reference = row["reference"].get("cost", {}).get("wall_seconds")
        if smoke is not None and reference is not None:
            if reference == 0:
                reasons.append(f"{row['id']}: complete reference cost is zero; the required cost ratio is undefined")
            elif smoke / reference > criteria["maximum_smoke_to_reference_wall_ratio"]:
                reasons.append(f"{row['id']}: complete measured smoke/reference cost ratio exceeds its frozen ceiling")
        options.append({outcomes[pair] for pair in product(*choices)})

    count = len(matrix)
    if count < criteria["minimum_paired_lineages"]:
        reasons.append("declared lineage count is below the frozen paired-lineage minimum")
    if criteria["minimum_reference_positives"] + criteria["minimum_reference_negatives"] > count:
        reasons.append("declared lineage count cannot supply both frozen reference minima")
    # A count tuple is (true_accept, false_accept, true_reject, false_reject).
    states = {(0, 0, 0, 0)}
    for index, choices in enumerate(options):
        following, remaining = set(), count - index - 1
        for state in states:
            for outcome in choices:
                updated = list(state)
                updated[outcome] += 1
                ta, fa, tr, fr = updated
                # Even an optimistic assignment of all remaining lineages
                # cannot dilute these fixed errors below the frozen maxima.
                if fr > criteria["maximum_false_reject_fraction"] * (ta + fr + remaining):
                    continue
                if fa > criteria["maximum_false_accept_fraction"] * (tr + fa + remaining):
                    continue
                following.add(tuple(updated))
        states = following
    viable = [state for state in states
              if state[0] + state[3] >= criteria["minimum_reference_positives"]
              and state[1] + state[2] >= criteria["minimum_reference_negatives"]
              and state[3] / (state[0] + state[3]) <= criteria["maximum_false_reject_fraction"]
              and state[1] / (state[1] + state[2]) <= criteria["maximum_false_accept_fraction"]]
    if not viable:
        reasons.append("no completion of the UNKNOWN decisions meets the frozen reference minima and error limits")
    return {"status": "INFEASIBLE" if reasons else "POSSIBLE",
            "reasons": reasons, "lineages": count, "unknown_decisions": unknowns,
            "missing_cost_vectors": missing_cost,
            "possible_completion_counts": (dict(zip(("true_accept", "false_accept", "true_reject", "false_reject"),
                                                      min(viable))) if viable and not reasons else None),
            "completion_scope": "Hypothetical complete decisions only; unknowns are not measured PASS or FAIL.",
            "cost_scope": "Missing costs remain unavailable; possible decision counts do not establish cost acceptance.",
            "qualification_input": False, "default_adoption": False, "training_authorized": False}


def _bound_published(root, profile, config, config_path, criteria, criteria_path):
    """Retain an original published matrix's frozen validation semantics.

    Older diagnostic cards can predate newer host schema requirements. Applying
    today's validator would erase their recorded failures into UNKNOWN. Read
    their exactly bound matrix for this necessary-condition check, with original
    byte availability checked separately. This never grants qualification.
    """
    path = root / "reports/forge/calibration" / (profile + ".json")
    if not path.is_file():
        return None
    saved = read_json(path)
    if (saved.get("profile") != profile or saved.get("evidence_scope") != "current"
            or saved.get("profile_sha256") != file_hash(config_path)
            or saved.get("criteria_sha256") != file_hash(criteria_path)
            or saved.get("criteria") != criteria or len(saved.get("cohorts", [])) != 1
            or any(saved["cohorts"][0].get(key) != config["cohort"][key] for key in ("sha256", "identity"))):
        return None
    from .calibration import _decision
    expected = [(r["id"], r["candidate_id"], r["candidate_revision"]) for r in config["lineages"]]
    if [(r["id"], r["candidate_id"], r["candidate_revision"]) for r in saved["matrix"]] != expected:
        raise ValueError("published calibration matrix differs from its frozen lineage denominator")
    issues = []
    for row in saved["matrix"]:
        if row.get("cohort_sha256") != config["cohort"]["sha256"]:
            raise ValueError("published calibration matrix changes scientific cohort")
        for part in ("smoke", "reference"):
            statuses = row[part]["task_statuses"]
            if set(statuses) != set(config[part + "_tasks"]):
                raise ValueError("published calibration decisions change the frozen required tasks")
            decision = _decision({name: {"gate_status": status} for name, status in statuses.items()}, list(statuses))
            if decision["decision"] != row[part]["decision"]:
                raise ValueError("published calibration decision contradicts its recorded task outcomes")
        for entry in row["inputs"]:
            identity = identifier(entry["attempt_id"], "attempt")
            for name, digest in entry.get("files", {}).items():
                if name not in {"request.json", "result.json", "evidence.json"}:
                    raise ValueError("published calibration names an unsupported original receipt")
                original = root / "reports/forge/attempts" / identity / name
                if not original.is_file() or file_hash(original) != digest:
                    issues.append({"attempt_id": identity, "file": name,
                                   "reason": "original receipt missing or byte hash differs; hydrate its exact archived source"})
    return saved, path, issues


def preflight(root: Path, profile: str) -> dict:
    """Inspect bound recorded decisions without changing their frozen meaning.

    No files, saved reports, verdicts, queues, or registrations are rewritten.
    Bound archived decisions remain advisory with explicit receipt-availability
    issues. Unpublished current profiles use the existing original reducer.
    Neither path can authorize adoption or supply qualification credit.
    """
    from .calibration import _current_profile, _evaluate_current
    root = Path(root).resolve()
    identifier(profile, "calibration profile")
    config_path = root / "configs/forge/calibration" / (profile + ".json")
    config = read_json(config_path)
    criteria_path = config_path.parent / identifier(config["criteria"], "criteria file")
    criteria = read_json(criteria_path)
    _current_profile(config, criteria)
    published = _bound_published(root, profile, config, config_path, criteria, criteria_path)
    if published:
        evaluated, published_path, availability = published
    else:
        evaluated = _evaluate_current(root, profile, config, config_path, criteria, criteria_path)
        published_path, availability = None, []
    feasibility = decision_feasibility(evaluated["matrix"], criteria)
    issues = evaluated["receipt_issues"] + availability + [
        {"lineage": row["id"], "conflicts": row["conflicts"]}
        for row in evaluated["matrix"] if row["conflicts"]]
    return {"schema_version": 1, "profile": profile,
            "profile_sha256": file_hash(config_path), "criteria_sha256": file_hash(criteria_path),
            "cohort_sha256": config["cohort"]["sha256"], "feasibility": feasibility,
            "status": "BLOCKED" if issues else feasibility["status"],
            "decision_source": ({"scope": "bound_published_report", "path": str(published_path.relative_to(root)),
                                 "sha256": file_hash(published_path), "qualification_input": False}
                                if published_path else {"scope": "current_original_reducer", "qualification_input": False}),
            "observed_criteria_status": evaluated["adoption"], "complete": evaluated["complete"],
            "receipt_issues": issues,
            "matrix_decisions": [{"lineage": row["id"], "smoke": row["smoke"]["decision"],
                                  "reference": row["reference"]["decision"]} for row in evaluated["matrix"]],
            "next_action": ("Stop adoption-oriented filling of this exact profile; retain its evidence and preregister a justified successor."
                            if feasibility["status"] == "INFEASIBLE" else
                            "Resolve missing evidence and costs in an explicitly bounded registration; POSSIBLE is not calibration acceptance."),
            "qualification_input": False, "default_adoption": False, "training_authorized": False}
