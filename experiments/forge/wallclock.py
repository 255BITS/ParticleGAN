"""Separate wall-clock limit hits from recipe verdicts (opt-in reporting and limits).

A wall-clock limit (`resources.timeout_seconds`) is a safety cap on shared
hardware, not part of a gate. A run killed by it was never graded, so it is
reported as NOT MEASURED (wall-clock): kept out of fail totals and out of paired
recipe-versus-baseline deltas, together with any task it blocks downstream.

Two opt-in mechanisms live here. Nothing changes unless a caller uses them.

* ``outcome`` / ``tally`` / ``paired_delta`` classify saved attempt records.
* ``effective_limit`` is the contention-aware limit. At the declared limit the
  supervisor measures seconds per update so far, divides by a solo reference for
  the task, and extends the limit by that slowdown factor, capped at a hard
  ceiling (default 3x). No measured progress means no extension: a run that has
  not logged a single observation is a recipe/runtime defect, not contention.

This module has no package imports so that ``attempt_worker.py`` can load it by
file path while a frozen source snapshot supplies every scientific module.

    python -m experiments.forge.wallclock report [--reruns PATH] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

NOT_MEASURED = "NOT_MEASURED_WALLCLOCK"
LABEL = "NOT MEASURED (wall-clock)"
VERDICTS = {"PASS", "FAIL"}
DEFAULT_CEILING = 3.0
RERUNS = Path("reports/forge/wallclock-reruns.json")
REGISTRY = Path("reports/forge/attempt-coverage.json")


def is_wallclock_hit(record):
    """TIMEOUT, or INCOMPLETE whose raw terminal status is a timeout at or past the limit."""
    status = record.get("status")
    if status == "TIMEOUT":
        return True
    if status != "INCOMPLETE":
        return False
    raw = record.get("raw_terminal_status")
    seconds, limit = record.get("seconds"), record.get("timeout_seconds")
    return raw == "timeout" and (seconds is None or limit is None or seconds >= limit)


def outcome(record):
    return NOT_MEASURED if is_wallclock_hit(record) else record.get("status")


def blocked_by_wallclock(record):
    """A BLOCKED verdict whose only cause is an ungraded (TIMEOUT/INCOMPLETE) checkpoint parent."""
    reason = str(record.get("reason", ""))
    return record.get("status") == "BLOCKED" and "checkpoint parent" in reason and (
        "TIMEOUT" in reason or "INCOMPLETE" in reason)


def tally(records):
    """Outcome counts; wall-clock hits and their downstream blocks are never FAIL."""
    counts = Counter()
    for record in records:
        value = outcome(record)
        if value != NOT_MEASURED and blocked_by_wallclock(record):
            value = NOT_MEASURED + "_DOWNSTREAM"
        counts[value] += 1
    return {"passed": counts.get("PASS", 0), "failed": counts.get("FAIL", 0),
            "not_measured_wallclock": counts.get(NOT_MEASURED, 0),
            "not_measured_downstream": counts.get(NOT_MEASURED + "_DOWNSTREAM", 0),
            "counts": dict(sorted(counts.items()))}


def paired_delta(recipe, baseline):
    """PASS-count delta over tasks where both arms have a real verdict.

    ``recipe`` and ``baseline`` map task -> attempt record (or a bare status
    string). Tasks where either arm is NOT MEASURED, blocked by an ungraded
    parent, or otherwise lacks a PASS/FAIL verdict are excluded and listed.
    """
    def status(value):
        record = value if isinstance(value, dict) else {"status": value}
        if outcome(record) == NOT_MEASURED or blocked_by_wallclock(record):
            return NOT_MEASURED
        return record.get("status")
    compared, excluded = [], {}
    for task in sorted(set(recipe) | set(baseline)):
        left, right = status(recipe.get(task)), status(baseline.get(task))
        if left in VERDICTS and right in VERDICTS:
            compared.append((task, left, right))
        else:
            excluded[task] = {"recipe": left, "baseline": right}
    delta = sum((l == "PASS") - (r == "PASS") for _, l, r in compared)
    return {"delta": delta, "compared": len(compared), "excluded": excluded,
            "tasks": {t: {"recipe": l, "baseline": r} for t, l, r in compared}}


def effective_limit(declared, elapsed, progressed_updates, solo_seconds_per_update, ceiling_factor=DEFAULT_CEILING):
    """Contention-aware limit decided once, at the declared limit.

    Returns ``(limit_seconds, slowdown_factor)``. The factor is measured seconds
    per update divided by the solo reference; the limit is
    ``declared * clamp(factor, 1, ceiling_factor)``.
    """
    if not (declared > 0 and ceiling_factor >= 1):
        raise ValueError("declared limit must be positive and the ceiling at least 1")
    if not progressed_updates or progressed_updates <= 0 or not solo_seconds_per_update or solo_seconds_per_update <= 0:
        return float(declared), None
    factor = (elapsed / progressed_updates) / solo_seconds_per_update
    if not math.isfinite(factor):
        return float(declared), None
    return float(declared) * min(max(factor, 1.0), float(ceiling_factor)), factor


def logged_progress(log_path):
    """Largest training step in saved observation lines of a run log (optionally timestamp-prefixed)."""
    best = 0
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        text = line.rsplit("\t", 1)[-1].strip()
        if not text.startswith("{"):
            continue
        try:
            event = json.loads(text)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "observation" and isinstance(event.get("step"), (int, float)):
            best = max(best, int(event["step"]))
    return best


def policy_for(policy, task):
    """Resolve an opt-in policy document to this task's ``(solo_seconds_per_update, ceiling)`` or None."""
    if not policy:
        return None
    reference = policy.get("solo_seconds_per_update", {}).get(task)
    if reference is None:
        return None
    return float(reference), float(policy.get("ceiling_factor", DEFAULT_CEILING))


def load_reruns(root):
    """Validated rerun overlay: real verdicts that supersede matching wall-clock hits (by exact attempt key)."""
    path = Path(root) / RERUNS
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    if (document.get("schema") != "forge_wallclock_reruns_v1" or document.get("qualification_input") is not False
            or document.get("formulations_modified") is not False):
        raise ValueError("wall-clock reruns must remain unchanged-formulation diagnostics")
    reruns = {}
    for record in document["records"]:
        if record["key"] in reruns:
            raise ValueError("duplicate wall-clock rerun key")
        reruns[record["key"]] = record
    return reruns


def apply_reruns(records, reruns):
    """Replace a wall-clock hit with its same-key rerun only when the rerun has a real verdict.

    A task BLOCKED only because its checkpoint parent was a wall-clock hit is
    treated the same way: its rerun continues from the rerun parent's checkpoint.
    """
    out = []
    for record in records:
        rerun = reruns.get(record.get("key"))
        # A retained receipt may keep its raw terminal status only in its evidence
        # file; the overlay then quotes it (``original_hit``) for the same check.
        hit = rerun and (is_wallclock_hit({**record, **rerun.get("original_hit", {}), "status": record.get("status")})
                         or blocked_by_wallclock(record))
        if hit and rerun["status"] in VERDICTS:
            merged = {**record, "status": rerun["status"], "reason": rerun.get("reason", ""),
                      "seconds": rerun.get("seconds"), "updates": rerun.get("updates"),
                      "wallclock_rerun": {"attempt": rerun["attempt"], "evidence": str(RERUNS),
                                          "superseded_status": record["status"]}}
            out.append(merged)
        else:
            out.append(record)
    return out


def report(root, output_dir):
    root = Path(root)
    records = json.loads((root / REGISTRY).read_text())["records"]
    reruns = load_reruns(root)
    records = apply_reruns(records, reruns)
    by_recipe = {}
    for record in records:
        by_recipe.setdefault(record.get("label", record["candidate_id"]), []).append(record)
    rows = {label: tally(items) for label, items in sorted(by_recipe.items())}
    for record in records:
        quoted = (reruns.get(record.get("key")) or {}).get("original_hit")
        if quoted and not record.get("wallclock_rerun"):
            record.update({k: v for k, v in quoted.items() if k != "evidence"})
    hits = [r for r in records if is_wallclock_hit(r)]
    value = {"schema": "forge_wallclock_outcomes_v1", "qualification_input": False,
             "label": LABEL, "registry": str(REGISTRY), "reruns": str(RERUNS) if reruns else None,
             "summary": tally(records), "recipes": rows,
             "not_measured": [{k: r.get(k) for k in ("label", "task", "status", "seconds", "timeout_seconds", "key")}
                              for r in sorted(hits, key=lambda r: (r.get("label", ""), r["task"]))],
             "rerun_verdicts": [{k: r.get(k) for k in ("label", "task", "status", "seconds", "wallclock_rerun")}
                                for r in records if r.get("wallclock_rerun")]}
    output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wallclock-outcomes.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    s = value["summary"]
    lines = ["# Wall-clock outcomes", "",
             f"A run killed by its task's wall-clock limit was never graded. It is reported as **{LABEL}**, "
             "not FAIL, and is excluded (with any task it blocks downstream) from paired recipe-versus-baseline deltas. "
             "A same-key solo rerun under the one-job-per-GPU lease replaces the hit only when it produced a real "
             "PASS/FAIL verdict under the unchanged declared limit.", "",
             f"Totals: {s['passed']} PASS, {s['failed']} FAIL, {s['not_measured_wallclock']} not measured (wall-clock), "
             f"{s['not_measured_downstream']} blocked by an unmeasured parent.", "",
             "| Recipe | PASS | FAIL | Not measured (wall-clock) | Blocked by unmeasured parent |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label, row in rows.items():
        if row["not_measured_wallclock"] or row["not_measured_downstream"] or any(
                r.get("wallclock_rerun") for r in by_recipe[label]):
            lines.append(f"| {label} | {row['passed']} | {row['failed']} | {row['not_measured_wallclock']} | "
                         f"{row['not_measured_downstream']} |")
    if value["rerun_verdicts"]:
        lines += ["", "## Real verdicts from solo reruns", "", "| Recipe | Task | Verdict | Seconds | Superseded |",
                  "| --- | --- | --- | ---: | --- |"]
        for r in value["rerun_verdicts"]:
            lines.append(f"| {r['label']} | {r['task']} | {r['status']} | {r['seconds']:.1f} | "
                         f"{r['wallclock_rerun']['superseded_status']} |")
    (output_dir / "WALLCLOCK_OUTCOMES.md").write_text("\n".join(lines) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("report",))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/forge"))
    args = parser.parse_args()
    value = report(args.root, args.output_dir)
    print(json.dumps(value["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
