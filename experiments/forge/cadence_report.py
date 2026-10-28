"""Pure, source-bound comparisons of a Forge regularizer-cadence search.

This is a readout, not a qualification input or another technique leaderboard.
It consumes the frozen search summary without reading receipts or starting jobs.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math


LEADERBOARD = "reports/forge/technique-inventory.md"
_IDENTITY = ("source_digest", "protocol_hash", "runtime_cohort")
_ROLES = ("required", "held", "diagnostic")
_MEASURED = {"PASS", "FAIL"}


def _safe(value):
    """Keep the projection strict JSON, including optional instrumented data."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return deepcopy(value)


def _scalars(metrics, prefix=""):
    values, unavailable = {}, []
    if not isinstance(metrics, dict):
        return {}, [prefix or "metrics"]
    for name, value in sorted(metrics.items()):
        key = f"{prefix}.{name}" if prefix else str(name)
        if name == "noise_floor":  # reporting-only oracle annotation, not a compared metric
            continue
        if isinstance(value, dict):
            nested, missing = _scalars(value, key)
            values.update(nested)
            unavailable.extend(missing)
        elif type(value) in (int, float):
            if math.isfinite(value):
                values[key] = value
            else:
                unavailable.append(key)
    return values, unavailable


def _rows(trial):
    rows, duplicates = {}, set()
    for row in trial.get("tasks", []):
        task = row.get("task")
        if not isinstance(task, str) or not task:
            continue
        if task in rows:
            duplicates.add(task)
        rows[task] = row
    return rows, duplicates


def _assignments(summary, source_assignments):
    if source_assignments is not None:
        if isinstance(source_assignments, dict):
            rows = [{"task": name, **assignment} for name, assignment in source_assignments.items()]
        else:
            rows = list(source_assignments)
    else:
        rows = [row for trial in summary["trials"] for row in trial.get("tasks", [])]
    assignments, conflicts = {}, set()
    through = summary.get("tuning_through_tier", math.inf)
    for row in rows:
        name, tier, role = row.get("task"), row.get("qualification_tier"), row.get("importance")
        if not isinstance(name, str) or type(tier) is not int or tier < 1 or role not in _ROLES:
            raise ValueError("cadence report requires task, qualification_tier and importance")
        if source_assignments is None and tier > through:
            continue
        assignment = {"task": name, "tier": tier, "importance": role}
        if name in assignments and assignments[name] != assignment:
            conflicts.add(name)
        assignments.setdefault(name, assignment)
    return sorted(assignments.values(), key=lambda a: (a["tier"], a["task"])), conflicts


def _cohort_reasons(summary, control, trial):
    reasons = []
    for key in _IDENTITY:
        baseline = control.get(key, summary.get(key))
        candidate = trial.get(key, summary.get(key))
        declared = summary.get(key)
        if baseline in (None, "", {}) or candidate in (None, "", {}):
            reasons.append(f"missing {key}")
        elif baseline != candidate:
            reasons.append(f"unmatched {key}")
        if declared not in (None, "", {}) and (baseline != declared or candidate != declared):
            reasons.append(f"{key} differs from frozen summary")
    for key in ("settings", "resolved_recipe"):
        left, right = control.get(key), trial.get(key)
        if left is None and right is None:
            continue
        if not isinstance(left, dict) or not isinstance(right, dict):
            reasons.append(f"missing {key}")
        elif ({k: v for k, v in left.items() if k not in {"reg_every", "name"}}
              != {k: v for k, v in right.items() if k not in {"reg_every", "name"}}):
            reasons.append(f"unmatched non-cadence {key}")
    return reasons


def _pair(summary, control, trial, assignment, control_rows, rows, duplicate_tasks, conflicts):
    name = assignment["task"]
    baseline, candidate = control_rows.get(name), rows.get(name)
    reasons = _cohort_reasons(summary, control, trial)
    if name in duplicate_tasks:
        reasons.append("duplicate task cell")
    if name in conflicts:
        reasons.append("conflicting task assignment")
    for label, row in (("control", baseline), ("candidate", candidate)):
        if row is None:
            reasons.append(f"missing {label} task cell")
        elif row.get("gate_status") not in _MEASURED:
            reasons.append(f"{label} gate is {row.get('gate_status', 'UNKNOWN')}")
    if baseline and candidate:
        binding = baseline.get("comparison_binding")
        if not binding or not candidate.get("comparison_binding"):
            reasons.append("missing task comparison_binding")
        elif binding != candidate["comparison_binding"]:
            reasons.append("unmatched task comparison_binding")
    control_status = baseline.get("gate_status", "UNKNOWN") if baseline else "UNAVAILABLE"
    candidate_status = candidate.get("gate_status", "UNKNOWN") if candidate else "UNAVAILABLE"
    left, bad_left = _scalars((baseline or {}).get("metrics", {}))
    right, bad_right = _scalars((candidate or {}).get("metrics", {}))
    if baseline and candidate and not (set(left) & set(right)):
        reasons.append("no paired finite scalar metrics")
    unavailable_metrics = sorted(set(left) ^ set(right) | set(bad_left) | set(bad_right))
    deltas = []
    if not reasons:
        for metric in sorted(set(left) & set(right)):
            delta = right[metric] - left[metric]
            if math.isfinite(delta):
                deltas.append({"metric": metric, "control": left[metric], "candidate": right[metric],
                               "delta": delta})
            else:
                unavailable_metrics.append(metric)
    matched = not reasons
    return {**assignment, "status": "AVAILABLE" if matched else "UNAVAILABLE",
            "control_status": control_status, "candidate_status": candidate_status,
            "regression": matched and control_status == "PASS" and candidate_status == "FAIL",
            "repair": matched and control_status == "FAIL" and candidate_status == "PASS",
            "unavailable_reasons": sorted(set(reasons)), "metric_deltas": deltas,
            "unavailable_metrics": sorted(set(unavailable_metrics)),
            "control_attempt_id": (baseline or {}).get("attempt_id"),
            "candidate_attempt_id": (candidate or {}).get("attempt_id"),
            "cadence": _safe((candidate or {}).get("cadence"))}


def build_report(summary, *, control_value=1, source_assignments=None):
    """Compare whole ``reg_every`` recipes from one frozen search summary.

    Original assignments may be supplied for a separately registered coverage
    lane. Its required cells remain descriptive and grant no qualification.
    Missing or incompatible evidence is unavailable rather than a numerical loss.
    """
    trials = summary.get("trials", [])
    controls = [t for t in trials if t.get("settings", {}).get("reg_every") == control_value]
    if len(controls) != 1:
        raise ValueError("cadence report requires exactly one reg_every control")
    if any(type(t.get("settings", {}).get("reg_every")) is not int
           or t["settings"]["reg_every"] < 1 for t in trials):
        raise ValueError("cadence report requires positive integer settings.reg_every")
    values = [t["settings"]["reg_every"] for t in trials]
    if len(set(values)) != len(values):
        raise ValueError("cadence report requires one global recipe per reg_every value")
    assignments, conflicts = _assignments(summary, source_assignments)
    control = controls[0]
    control_rows, control_duplicates = _rows(control)
    scope = summary.get("evidence_scope", "ordinary")
    diagnostic = source_assignments is not None or "diagnostic" in str(scope).lower()
    projected = []
    for trial in sorted(trials, key=lambda t: t["settings"]["reg_every"]):
        rows, duplicates = _rows(trial)
        pairs = [_pair(summary, control, trial, a, control_rows, rows,
                       duplicates | control_duplicates, conflicts) for a in assignments]
        tiers = {}
        for tier in sorted({a["tier"] for a in assignments}):
            roles = {}
            for role in _ROLES:
                cells = [a for a in assignments if a["tier"] == tier and a["importance"] == role]
                statuses = Counter(rows.get(a["task"], {}).get("gate_status", "UNAVAILABLE") for a in cells)
                roles[role] = {"denominator": len(cells), "pass": statuses.get("PASS", 0),
                               "fail": statuses.get("FAIL", 0),
                               "unavailable": sum(n for status, n in statuses.items() if status not in _MEASURED),
                               "statuses": dict(sorted(statuses.items()))}
            tiers[str(tier)] = roles
        actual_counts = dict(sorted(Counter(row.get("gate_status", "UNKNOWN")
                                            for row in trial.get("tasks", [])).items()))
        counts_consistent = "counts" not in trial or trial["counts"] == actual_counts
        unavailable = [p["task"] for p in pairs if p["status"] == "UNAVAILABLE"]
        regressions = [p["task"] for p in pairs if p["regression"]]
        complete = bool(pairs) and not unavailable and counts_consistent and not any(p["unavailable_metrics"] for p in pairs)
        required = [p for p in pairs if p["importance"] == "required"]
        full_view_qualified = (not diagnostic and bool(required) and complete
                               and all(p["candidate_status"] == "PASS" for p in required)
                               and trial.get("qualification", {}).get("full_view_qualified") is True)
        projected.append({"candidate_id": trial.get("candidate_id"), "configuration_id": trial.get("configuration_id"),
                          "value": trial["settings"]["reg_every"], "tiers": tiers, "pairs": pairs,
                          "regressions": regressions, "repairs": [p["task"] for p in pairs if p["repair"]],
                          "unavailable": unavailable, "missing_cells": [a["task"] for a in assignments if a["task"] not in rows],
                          "counts_consistent": counts_consistent, "recomputed_counts": actual_counts,
                          "complete": complete, "pass_preserving": complete and not regressions,
                          "full_view_qualified": full_view_qualified, "qualification_credit": False,
                          "cost": _safe(trial.get("cost", {})), "cadence": _safe(trial.get("cadence")),
                          "provenance": _safe({key: trial.get(key, summary.get(key)) for key in _IDENTITY}),
                          "attempt_ids": _safe(trial.get("attempt_ids", [])),
                          "logs": _safe(trial.get("logs"))})
    variants = [t for t in projected if t["value"] != control_value]
    preserving = [t["candidate_id"] for t in variants if t["pass_preserving"]]
    complete = bool(variants) and all(t["complete"] for t in projected)
    recommendation = {"status": "incomplete" if not complete else
                      ("pass preserving candidates" if preserving else "no pass preserving variant"),
                      "candidate_ids": preserving,
                      "reason": ("Finish or repair unavailable matched evidence before selecting a cadence." if not complete else
                                 "These complete variants retain every measured control PASS; inspect individual metric deltas."
                                 if preserving else "Every complete variant loses at least one control PASS.")}
    return {"schema_version": 1, "study_id": summary.get("study_id"), "axis": "reg_every",
            "control_value": control_value, "control_candidate_id": control.get("candidate_id"),
            "evidence_scope": _safe(scope), "diagnostic_scope": diagnostic, "qualification_credit": False,
            "qualification_note": "Comparison readout only; diagnostic coverage grants no ordinary qualification.",
            "complete": complete, "trials": projected, "recommendation": recommendation,
            "provenance": _safe({key: summary.get(key) for key in _IDENTITY}),
            "cost": _safe(summary.get("cost", {})), "logs": _safe(summary.get("logs")),
            "leaderboard": LEADERBOARD,
            "limitations": ["Fixed-seed measured pass preservation is not statistical noninferiority.",
                            "Raw metric deltas have no inferred common scale or improvement direction.",
                            "Charged worker wall time is cost, not a speed measurement."]}


def _cell(value):
    if value is None:
        return "UNAVAILABLE"
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("|", "&#124;").replace("`", "&#96;").replace("\r", " ").replace("\n", " "))


def _number(value):
    return _cell(f"{value:.8g}" if type(value) is float else value)


def render_markdown(report):
    """Render a compact readout with unaggregated paired scalar metrics."""
    lines = [f"Cadence study {_cell(report.get('study_id'))} compares reg_every against {_cell(report['control_value'])}.",
             "", "[The current technique inventory](../technique-inventory.md) remains the single leaderboard.",
             "This comparison grants no qualification; diagnostic coverage supplies no ordinary qualification credit.",
             "", "| Cadence | Tier | Required PASS / total | Held PASS / total | Diagnostic PASS / total | Lost PASS | Unavailable |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for trial in report["trials"]:
        for tier, roles in trial["tiers"].items():
            counts = [f"{roles[role]['pass']}/{roles[role]['denominator']}" for role in _ROLES]
            losses = sum(p["regression"] for p in trial["pairs"] if str(p["tier"]) == tier)
            missing = sum(p["status"] == "UNAVAILABLE" for p in trial["pairs"] if str(p["tier"]) == tier)
            lines.append("| " + " | ".join(map(_cell, [trial["value"], tier, *counts, losses, missing])) + " |")
    lines.extend(["", "| Cadence | Task | Role | Control → candidate gate | Metric | Control | Candidate | Delta |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for trial in report["trials"]:
        if trial["value"] == report["control_value"]:
            continue
        for pair in trial["pairs"]:
            prefix = [trial["value"], pair["task"], pair["importance"],
                      f"{pair['control_status']} → {pair['candidate_status']}"]
            for metric in pair["metric_deltas"]:
                lines.append("| " + " | ".join([*map(_cell, prefix), _cell(metric["metric"]),
                             _number(metric["control"]), _number(metric["candidate"]), _number(metric["delta"])]) + " |")
            if pair["status"] == "UNAVAILABLE" or pair["unavailable_metrics"] or not pair["metric_deltas"]:
                detail = "; ".join(pair["unavailable_reasons"] +
                                   (["missing/nonfinite metrics: " + ", ".join(pair["unavailable_metrics"])]
                                    if pair["unavailable_metrics"] else [])) or "no paired finite scalar metrics"
                lines.append("| " + " | ".join(map(_cell, [*prefix, f"UNAVAILABLE: {detail}", None, None, None])) + " |")
    rec = report["recommendation"]
    lines.extend(["", f"Recommendation: {_cell(rec['status'])}. {_cell(rec['reason'])}"])
    if rec["candidate_ids"]:
        lines.append("Measured pass preserving candidates: " + ", ".join(map(_cell, rec["candidate_ids"])) + ".")
    lines.extend(["", "Cadence counters describe local audit windows, which may overlap across task rows; do not sum them.",
                  "Observed penalty calls can differ from controller applications when a host observes only some calls.",
                  "", "| Cadence | Task | Critic updates in window | Controller applications | Observed applied / skipped | First / last application update | Thermal min / max / last |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for trial in report["trials"]:
        for pair in trial["pairs"]:
            cadence = pair.get("cadence") or {}
            optimizer, observations = cadence.get("optimizer", {}), cadence.get("observations", {})
            thermal = cadence.get("ranges", {}).get("s_thermal", {})
            applied_skipped = " / ".join(_number(observations.get(k)) for k in ("applied", "skipped"))
            first_last = " / ".join(_number(cadence.get(k)) for k in ("first_application_update", "last_application_update"))
            thermal_values = " / ".join(_number(thermal.get(k)) for k in ("min", "max", "last"))
            fields = [trial["value"], pair["task"], optimizer.get("observed_steps_in_window"),
                      optimizer.get("applied_record_calls_in_window")]
            lines.append("| " + " | ".join([*map(_cell, fields), applied_skipped, first_last, thermal_values]) + " |")
    lines.extend(["", "| Cadence | Evidence worker seconds | New paid worker seconds |",
                  "| --- | --- | --- |"])
    for trial in report["trials"]:
        cost = trial["cost"]
        lines.append("| " + " | ".join([_cell(trial["value"]),
                     _number(cost.get("evidence_wall_seconds", cost.get("wall_seconds"))),
                     _number(cost.get("new_paid_wall_seconds"))]) + " |")
    lines.extend(["", f"Frozen source: {_cell(report['provenance'].get('source_digest'))}.",
                  f"Frozen protocol: {_cell(report['provenance'].get('protocol_hash'))}.",
                  "Runtime and remaining provenance are retained in the companion JSON; worker wall time supplies no speed ranking.",
                  "Fixed-seed pass preservation supplies no statistical noninferiority claim.",
                  f"Logs: {_cell(report.get('logs'))}.", ""])
    return "\n".join(lines)
