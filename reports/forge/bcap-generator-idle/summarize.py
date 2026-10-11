"""Saved-only audit and leaderboard for the generator idle comparison.

Reads original queue receipts (and the separate off-ladder diagnostic summary);
constructs no model and runs no update. Writes audit.json and the README leaderboard block.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import importlib.util

_spec = importlib.util.spec_from_file_location("prior_search_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from experiments.forge.contracts import atomic_json, atomic_text, read_json, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.queue import Queue

REPORT = ROOT / "reports/forge/bcap-generator-idle"
TARGETS = {"gaussian1d_stability": ("mean_error_sigma", "std_ratio", "cdf_ks"), "mode_hold": ("modes", "hq")}


def fmt(value):
    return base.fmt(value)


def idle_row(row):
    mechanisms = row.get("evidence", {}).get("guards", {}).get("mechanism_audit", {}).get("mechanisms", {})
    return mechanisms.get("generator_idle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    progress = read_json(args.archive / "progress.json")
    spec = read_json(REPORT / "spec.json")
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    state = queue.inspect()
    originals, _ = _attempts(ROOT)
    attempts = {a["attempt_id"]: a for a in originals}
    arms = [("control", progress["control_request"])] + [("idle1.0", rid) for rid in progress["search_requests"]]
    cells, checks = {}, []
    for arm, request_id in arms:
        entry = state["submissions"][request_id]
        request = entry["request"]
        checks.append(dict(arm=arm, check="source digest matches registration",
                           passed=request["source"]["digest"] == spec["source_digest"] == progress["source_digest"]))
        recipe = request["candidate"]["resolved_recipe"]
        checks.append(dict(arm=arm, check="declared trainer delta only",
                           passed=recipe.get("generator_idle_se") == (None if arm == "control" else 1.0)))
        rows = {r["task_id"]: r for r in base.validated_submission_rows(state, entry, attempts)}
        tier1_failed = any(r.get("gate_status") != "PASS" for name, r in rows.items()
                           if next(a for a in request["view"]["assignments"] if a["task"] == name)["qualification_tier"] == 1
                           and next(a for a in request["view"]["assignments"] if a["task"] == name)["importance"] == "required")
        for assignment in request["view"]["assignments"]:
            if assignment["qualification_tier"] > 2:
                continue
            name, row = assignment["task"], rows.get(assignment["task"])
            if row is None:
                status = "BLOCKED" if tier1_failed and assignment["qualification_tier"] == 2 else "UNMEASURED"
                cells[(arm, name)] = dict(tier=assignment["qualification_tier"], status=status,
                                          required=assignment["importance"] == "required")
                continue
            idle = idle_row(row)
            checks.append(dict(arm=arm, task=name, check="valid original receipt",
                               passed=bool(row["receipt_binding"]["valid_receipt"])))
            if row.get("evidence", {}).get("guards", {}).get("mechanism_audit") is not None:
                checks.append(dict(arm=arm, task=name, check="idle row present only when enabled",
                                   passed=(idle is None) == (arm == "control")))
            cells[(arm, name)] = dict(tier=assignment["qualification_tier"], status=row["gate_status"],
                                      required=assignment["importance"] == "required",
                                      reason=row.get("reason"), metrics=row.get("metrics", {}),
                                      idle_steps=None if idle is None else idle["applied"],
                                      decisions=None if idle is None else idle["calls"],
                                      attempt_id=row["receipt_binding"]["attempt_id"])
    diagnostic = None
    path = args.archive / "diagnostic" / "summary.json"
    if path.exists():
        diagnostic = read_json(path)
    tasks = [a["task"] for a in state["submissions"][progress["control_request"]]["request"]["view"]["assignments"]
             if a["qualification_tier"] <= 2]
    lines = ["| Task | Tier | Control | idle1.0 | idle1.0 idle steps / G decisions |", "| --- | ---: | --- | --- | ---: |"]
    unreached = [n for n in tasks if cells[("control", n)]["status"] == cells[("idle1.0", n)]["status"] == "BLOCKED"]
    for name in tasks:
        if name in unreached:
            continue
        c, t = cells[("control", name)], cells[("idle1.0", name)]
        ratio = (f"{t['idle_steps']}/{t['decisions']} ({t['idle_steps'] / t['decisions']:.1%})"
                 if t.get("decisions") else "—")
        lines.append(f"| {name} | {c['tier']} | {c['status']} | {t['status']} | {ratio} |")
    if unreached:
        lines.append(f"| {len(unreached)} Tier 2 tasks ({', '.join(unreached)}) | 2 | BLOCKED | BLOCKED | — |")
    totals = {arm: {tier: sum(cells[(arm, n)]["status"] == "PASS" for n in tasks if cells[(arm, n)]["tier"] == tier and cells[(arm, n)]["required"])
                    for tier in (1, 2)} for arm in ("control", "idle1.0")}
    lines += ["", f"Required passes: control {totals['control'][1]}/6 Tier 1, {totals['control'][2]}/21 Tier 2; "
              f"idle1.0 {totals['idle1.0'][1]}/6 Tier 1, {totals['idle1.0'][2]}/21 Tier 2 "
              "(the clock audit is an optional diagnostic and is listed but not counted)."]
    if diagnostic:
        lines += ["", "Off-ladder target diagnostic (separate cohort, no qualification credit):", "",
                  "| Arm | Task | Diagnostic verdict | Sustained gate | Target metrics (final) | Idle steps / G decisions |",
                  "| --- | --- | --- | --- | --- | ---: |"]
        for row in diagnostic["rows"]:
            shown = ", ".join(f"{k} {fmt(row['metrics'].get(k))}" for k in TARGETS[row["task_id"]])
            ratio = (f"{row['idle_steps']}/{row['decisions']} ({row['idle_steps'] / row['decisions']:.1%})"
                     if row.get("decisions") else "—")
            e = row["evaluator"]
            sustained = (f"stationary {e.get('stationary_passes')}/72, reacquisition {e.get('reacquisition_status')}, "
                         f"shifted hold {e.get('shift_hold_passes')}/24" if row["task_id"] == "gaussian1d_stability" else
                         f"terminal suffix {e.get('convergence', {}).get('passing_suffix')}/>=5")
            lines.append(f"| {row['arm']} | {row['task_id']} | {row['gate_status']} | {sustained} | {shown} | {ratio} |")
    readme = (REPORT / "README.md").read_text()
    start, end = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"
    head, rest = readme.split(start)
    atomic_text(REPORT / "README.md", head + start + "\n" + "\n".join(lines) + "\n" + end + rest.split(end)[1])
    audit = dict(schema_version=1, scope="saved_receipts_only", source_commit=progress["source_commit"],
                 source_digest=progress["source_digest"], spec_sha256=stable_hash(spec),
                 checks=checks, all_checks_pass=all(c["passed"] for c in checks),
                 cells=[dict(arm=arm, task_id=name, **cell) for (arm, name), cell in sorted(cells.items())],
                 diagnostic=diagnostic and dict(cohort=diagnostic["cohort"], qualification_credit=False,
                                                rows=[{k: r[k] for k in ("arm", "task_id", "gate_status", "metrics",
                                                       "idle_steps", "decisions", "source_digest")}
                                                      for r in diagnostic["rows"]]))
    atomic_json(REPORT / "audit.json", audit)
    print("\n".join(lines))
    print("all_checks_pass", audit["all_checks_pass"])


if __name__ == "__main__":
    main()
