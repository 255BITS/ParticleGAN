"""Display the frozen round-one R1/R2 report without training or regrading.

Refresh the original study with report_search first. This projection is neither
qualification input nor a new leaderboard; every scalar retains its source row.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
STUDY = "r1r2-modern-family-round1-v1"
SOURCE = "9f89fa7cc7af552ef7e41405fab415abbd00acf3d3c6e4af8e4435fb2423142e"
COMMIT = "05a2b3c021155fa72471b2293c3fd6d41d1e58a0"
DENOMINATORS = {1: 3, 2: 19, 3: 2}

if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import stable_hash


def project(report, *, original_sha256, original_path):
    if (report.get("study_id") != STUDY or report.get("tuning_through_tier") != 3
            or report.get("source_digest") != SOURCE or len(report.get("trials", [])) != 8):
        raise ValueError("readout requires the exact frozen eight-config round-one study")
    trials = []
    roster = None
    for trial in sorted(report["trials"], key=lambda r: r["configuration_id"]):
        if trial.get("source_digest") != SOURCE:
            raise ValueError("mixed scientific sources cannot enter this readout")
        tasks = trial["tasks"]
        required = [t for t in tasks if t["importance"] == "required"]
        if dict(Counter(t["qualification_tier"] for t in required)) != DENOMINATORS:
            raise ValueError("required denominators differ from frozen 3/19/2 contract")
        task_roster = [(t["qualification_tier"], t["order"], t["task"]) for t in required]
        if roster is not None and task_roster != roster:
            raise ValueError("all complete configurations must receive the same task roster")
        roster = task_roster
        observed = [deepcopy(t) for t in tasks if t["gate_status"] != "UNKNOWN"]
        for item in observed:
            for key in ("blockers", "reusable", "shared_pending", "permitted_by_tier_cap"):
                item.pop(key, None)
        trials.append({
            "candidate_id": trial["candidate_id"],
            "configuration_id": trial["configuration_id"],
            "candidate_revision": trial["candidate_revision"],
            "runtime_cohort": deepcopy(trial["runtime_cohort"]),
            "settings": deepcopy(trial["settings"]),
            "status": trial["status"],
            "submission_status": trial["submission_status"],
            "submission_reason": trial.get("submission_reason"),
            "qualification": deepcopy(trial["qualification"]),
            "cost": deepcopy(trial["cost"]),
            "receipt_bindings": deepcopy(trial["receipt_bindings"]),
            "attempt_history": deepcopy(trial["attempt_history"]),
            "observed_tasks": observed,
            "unknown_tasks": [t["task"] for t in tasks if t["gate_status"] == "UNKNOWN"],
        })
    output = dict(schema_version=1, kind="frozen_source_display_projection", study_id=STUDY,
                  qualification_input=False, qualification_reuse=False, regrading=False,
                  training_updates=0, source_digest=SOURCE, executed_source_commit=COMMIT,
                  original_report=dict(path=original_path, sha256=original_sha256),
                  full_required_denominators=DENOMINATORS, required_task_roster=roster,
                  representation_status="UNRESOLVED_COMPLETE_SUITE",
                  selection=deepcopy(report["selection"]), speed_selection=deepcopy(report["speed_selection"]),
                  cost=deepcopy(report["cost"]), trials=trials)
    # Report scalar projections must never conceal NaN or infinity.
    json.dumps(output, allow_nan=False)
    return output


def attach_recorded_details(data, root):
    """Copy exact original evaluator/contract evidence, never recompute a gate."""
    for trial in data["trials"]:
        proofs = {row["attempt_id"]: row for row in trial["receipt_bindings"]}
        for item in trial["observed_tasks"]:
            attempt_id = item.get("attempt_id")
            if not attempt_id:
                continue
            directory = root / "reports/forge/attempts" / attempt_id
            path = directory / "result.json"
            payload = path.read_bytes()
            result = json.loads(payload)
            proof = proofs[attempt_id]
            if proof.get("valid_receipt") is not True or stable_hash(result) != proof["result_hash"]:
                raise ValueError("original evaluator detail lacks its exact certified result binding")
            matches = [row for row in result["task_results"] if row["task_id"] == item["task"]]
            if (len(matches) != 1 or matches[0]["gate_status"] != item["gate_status"]
                    or matches[0]["metrics"] != item["metrics"]):
                raise ValueError("original task details differ from the frozen report projection")
            row = matches[0]
            item["recorded_evaluator_result"] = deepcopy(row["evaluator_result"])
            item["recorded_claim_contract"] = deepcopy(row["claim_contract"])
            item["actual_device"] = deepcopy(row["device"])
            item["original_file_sha256"] = {
                name: hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest()
                for name in ("request", "evidence", "result")}
    json.dumps(data, allow_nan=False)


def failure_detail(item):
    recorded = item.get("recorded_evaluator_result", {})
    bounds = [f"{row['metric']}={row['value']:.6g} ({row['op']}{row['threshold']})"
              for row in recorded.get("metrics", []) if row["status"] != "PASS"]
    convergence = recorded.get("convergence", {})
    suffix = convergence.get("passing_suffix")
    needed = convergence.get("minimum_stable_checks")
    if suffix is not None and needed is not None and suffix < needed:
        bounds.append(f"terminal suffix {suffix}/{needed}")
    return "; ".join(bounds) or item.get("reason") or "see exact original metrics"


def markdown(data):
    lines = ["# R1/R2 frozen round-one readout", "",
             "All eight complete configurations retain the **3 smoke / 19 quality / 2 endurance** "
             "denominators. This is a display of the certified study report, with no training or "
             "independent qualification supplied by this projection.", "",
             "| Configuration | Critic multiplier / cosine start / floor | Smoke | Quality | Endurance | Status | First observed non-pass | Paid seconds |",
             "| --- | --- | ---: | ---: | ---: | --- | --- | ---: |"]
    for trial in data["trials"]:
        settings = trial["settings"]
        cells = {row["tier"]: f"{row['required_passed']}/{row['required_total']}"
                 for row in trial["qualification"]["tiers"]}
        failures = [row for row in trial["observed_tasks"] if row["gate_status"] != "PASS"]
        why = (f"{failures[0]['task']}: {failures[0]['gate_status']} — {failure_detail(failures[0])}"
               if failures else "No recorded non-pass" if trial["observed_tasks"] else "Unmeasured")
        why = why.replace("|", "\\|").replace("\n", " ")
        values = [f"`{trial['configuration_id'][:12]}`",
                  f"{settings['d_lr_mult']} / {settings['lr_anneal_start']} / {settings['lr_floor']}",
                  cells[1], cells[2], cells[3], trial["status"], why,
                  round(trial["cost"].get("new_paid_wall_seconds", 0), 3)]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    selection = data["selection"]
    lines += ["", f"Selection: **{selection['selection_kind']}**; all trials terminal: "
              f"**{selection['all_trials_terminal']}**; qualified: **{selection['qualified']}**.", "",
              "UNKNOWN remains unmeasured. A failed or stopped prerequisite does not turn later "
              "tasks into passes. Complete-suite Q1 remains unresolved; the separate supported "
              "smoke and trajectory parameter constructions give zero ordinary qualification credit.", "",
              "No fastest-convergence ranking, robustness claim or default promotion follows from "
              "this provisional screen. The paid cost records execution; it is not acquisition time.", "",
              f"Frozen scientific digest `{SOURCE}`; executed commit `{COMMIT}`. "
              "Later source changes do not qualify this cohort as current.", "",
              "[Every observed scalar metric, verdict, original receipt binding and unknown task](r1r2-execution-readout.json) · "
              "[Original configuration-search report](../configuration-search/r1r2-modern-family-round1-v1.json)", "",
              f"Original report SHA256 `{data['original_report']['sha256']}`.", ""]
    if selection.get("all_trials_terminal"):
        winner = next((row for row in data["trials"]
                       if row["candidate_id"] == selection.get("selected_candidate_id")), None)
        if winner:
            passed_trajectory = next((row for row in winner["observed_tasks"]
                                      if row["task"] == "trajectory" and row["gate_status"] == "PASS"), None)
            if passed_trajectory:
                lines += [f"The best observed configuration `{winner['configuration_id'][:12]}` "
                          f"passes trajectory at identity MSE **{passed_trajectory['metrics']['identity_mse']:.9g}**. "
                          "The same shared configuration then fails the residual correspondence question: "
                          "the raw endpoint has no correct pads and every selected pad is wrong. "
                          "The numerical gate detects this identity failure; it does not identify a critic, "
                          "optimizer or conditioning cause by itself.", "",
                          "Within this declared eight-point screen, earlier strong decay with the full-rate "
                          "critic is the only combination to pass trajectory. This finite observation supports "
                          "that configuration as a research candidate; it does not establish robustness or "
                          "justify mixing it with another configuration's later task passes.", ""]
        actual_devices = Counter(str(task.get("actual_device", "unrecorded"))
                                 for row in data["trials"] for task in row["observed_tasks"])
        lines += ["Actual completed task devices: " + ", ".join(f"{name} {count}" for name, count in sorted(actual_devices.items())) +
                  ". The study's declared later CUDA resources do not turn these behavioral executions into GPU measurements.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / f"reports/forge/configuration-search/{STUDY}.json"
    payload = path.read_bytes()
    data = project(json.loads(payload), original_sha256=hashlib.sha256(payload).hexdigest(),
                   original_path=path.relative_to(root).as_posix())
    attach_recorded_details(data, root)
    directory = root / "reports/forge/family-winner-round1"
    (directory / "r1r2-execution-readout.json").write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (directory / "R1R2_READOUT.md").write_text(markdown(data))
    print(json.dumps({"observed_tasks": sum(len(r["observed_tasks"]) for r in data["trials"]),
                      "selection": data["selection"]["selection_kind"], "training_updates": 0}))


if __name__ == "__main__":
    main()
