"""Compact receipts and leaderboard for the 2026-10-25 matched prior ablation.

Reads the shared Forge queue (events, frozen requests, graded results) and the
plan blockers captured before enqueue; writes ``receipts.json`` and the
generated tables in the report README. Launches no training.

    python -m experiments.prior_ablation_report \
        --queue-root /path/to/runs/forge --report reports/prior-ablation-2026-10-25
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CAMPAIGN = "prior-ablation-20261025"
FIXED_LIST = Path("configs/forge/leaderboards/technique-inventory-full-suite-2026-10-25.json")
# (tag, studies in precedence order, candidate, label). A later study's attempt
# replaces an earlier one only for the same task (the A3 repair request).
ARMS = [
    ("A1", ("prior-ablation-a1-learned",), "bispf-optimized-quintic-v1", "BISPF default (learned table)"),
    ("A2", ("prior-ablation-a2-frozen",), "bispf-prior-frozen-v1", "Frozen table"),
    ("A3", ("prior-ablation-a3-fresh-gaussian", "prior-ablation-a3-fresh-gaussian-r2"),
     "bispf-prior-fresh-gaussian-v1", "Fresh N(0, I) latent"),
    ("A4", ("prior-ablation-a4-pure-global",), "bispf-pure-global-v1", "Shape/step gates off"),
]
# Tests whose "prior" is not a sampled latent: the table is the optimized
# sample, embedding or residual control itself, so freezing it removes the
# trained object rather than ablating a latent prior.
NON_LATENT = {"two_pole", "unused_token_hold", "unipolar", "mid_scale_identity", "trajectory",
              "residual_student", "cover_leftover", "five_word_joint_smoke", "five_word_joint_hold"}
START, END = "<!-- prior-ablation tables: start -->", "<!-- prior-ablation tables: end -->"


def _gates(task):
    evaluation = task.get("evaluation", {})
    gates = [tuple(item) for item in evaluation.get("thresholds", []) if len(item) == 3]
    for name, value in (evaluation.get("accuracy_limits") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gates.append((name, "<=", value))
    return gates


STRUCTURAL_GATES = {"sample_count", "finite_fraction", "all_finite", "valid_n", "min_samples"}


def _margin(metrics, gates):
    """Smallest relative signed distance to a declared quality threshold (negative fails).

    Structural gates (sample counts, finiteness) are skipped: they say nothing about quality.
    """
    best = None
    for name, op, threshold in gates:
        if name in STRUCTURAL_GATES:
            continue
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isinstance(threshold, (int, float)):
            continue
        scale = abs(threshold) or 1.
        if op in (">=", ">"):
            margin = (value - threshold) / scale
        elif op in ("<=", "<"):
            margin = (threshold - value) / scale
        else:
            margin = -abs(value - threshold) / scale
        if best is None or margin < best[1]:
            best = (name, margin, value, op, threshold)
    return None if best is None else {"gate": best[0], "relative_margin": round(best[1], 4),
                                      "value": best[2], "op": best[3], "threshold": best[4]}


def _floor(metrics):
    """Oracle noise-floor margins (oracle-noise-floor-v1) recorded with the graded vector metrics."""
    floor = metrics.get("noise_floor")
    if not isinstance(floor, dict) or not isinstance(floor.get("metrics"), dict):
        return None
    return {"version": floor.get("version"), "draws": floor.get("draws"),
            "eval_draw_contract": floor.get("eval_draw_contract"),
            "gates": {name: {key: gate.get(key) for key in ("margin_oracle_std", "oracle_mean", "oracle_std",
                                                             "oracle_pass_fraction")}
                      for name, gate in floor["metrics"].items() if isinstance(gate, dict)}}


def collect(queue_root: Path, report: Path) -> dict:
    tests = json.loads(FIXED_LIST.read_text())["tests"]
    blockers = json.loads((report / "plan-blockers.json").read_text())
    requests = {}
    for path in sorted((queue_root / "queue" / "requests").glob("*.json")):
        request = json.loads(path.read_text())
        if request.get("campaign_id") == CAMPAIGN:
            requests[(request.get("study") or {}).get("id")] = request
    by_id = {request["request_id"]: study for study, request in requests.items()}
    attempt_study, completed = {}, {}
    for line in (queue_root / "events.jsonl").open():
        if f'"{CAMPAIGN}"' not in line:
            continue
        event = json.loads(line)
        if event.get("event") == "claimed" and event.get("request") in by_id:
            attempt_study[event["attempt"]] = by_id[event["request"]]
        if event.get("event") == "completed" and event.get("attempt") in attempt_study:
            completed[(attempt_study[event["attempt"]], event["task"])] = event
    arms = {}
    for tag, studies, candidate, label in ARMS:
        request = requests.get(studies[-1], {})
        cells = {}
        for test in tests:
            task = json.loads(Path(f"configs/forge/tasks/{test}.json").read_text())
            event = None
            for name in studies:
                found = completed.get((name, test))
                if found is None and test == "ring_extension":
                    found = completed.get((name, "ring_hold"))
                event = found or event
            cell = {"status": "NOT_MEASURED", "applicability": "non_latent_table" if test in NON_LATENT else "latent"}
            if event is not None and test in (event.get("verdicts") or {}):
                attempt = queue_root / CAMPAIGN / event["attempt"]
                graded = {}
                if (attempt / "graded-result.json").is_file():
                    graded = json.loads((attempt / "graded-result.json").read_text())
                grade = graded.get("grades", {}).get(test, {})
                metrics = grade.get("metrics") or (event.get("metrics") or {}).get(test) or {}
                metrics = metrics if isinstance(metrics, dict) else {}
                cell.update(status=event["verdicts"][test], attempt=event["attempt"],
                            wall_seconds=round((event.get("cost") or {}).get("wall_seconds") or 0., 1),
                            reasons=grade.get("reasons") or [(event.get("reason") or {}).get(test)],
                            graded_raw_hash=graded.get("raw_hash"),
                            final_margin=_margin(metrics, _gates(task)),
                            noise_floor=_floor(metrics))
            elif test in blockers.get(tag, {}):
                cell.update(status="BLOCKED", reason=blockers[tag][test])
            cells[test] = cell
        counts = {}
        for cell in cells.values():
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
        arms[tag] = {"label": label, "studies": list(studies), "candidate": candidate,
                     "request_ids": [requests.get(name, {}).get("request_id") for name in studies], "candidate_revision": request.get("candidate_revision"),
                     "source_digest": (request.get("source") or {}).get("digest"),
                     "counts": counts, "cells": cells,
                     "wall_seconds": round(sum(c.get("wall_seconds", 0.) for c in cells.values()), 1)}
    return {"schema": "prior_ablation_receipts_v1", "campaign": CAMPAIGN, "fixed_list": str(FIXED_LIST),
            "fixed_list_sha256": hashlib.sha256(FIXED_LIST.read_bytes()).hexdigest(), "arms": arms}


def _short(status):
    return {"PASS": "PASS", "FAIL": "**FAIL**", "BLOCKED": "blocked", "NOT_MEASURED": "—"}.get(status, status.lower())


def render(receipts: dict) -> str:
    arms = receipts["arms"]
    tags = list(arms)
    lines = [START, "", "### Leaderboard (fixed denominator: 49 tests)", "",
             "| Arm | Declared delta | PASS | FAIL | Blocked | Not measured | Other | GPU seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for tag in tags:
        arm = arms[tag]
        c = arm["counts"]
        other = sum(v for k, v in c.items() if k not in ("PASS", "FAIL", "BLOCKED", "NOT_MEASURED"))
        lines.append(f"| {tag} | {arm['label']} | {c.get('PASS', 0)} | {c.get('FAIL', 0)} | {c.get('BLOCKED', 0)} | "
                     f"{c.get('NOT_MEASURED', 0)} | {other} | {arm['wall_seconds']:.0f} |")
    lines += ["", "### Per-test outcomes and final-state margins", "",
              "Margin: the smallest relative signed distance to a declared threshold at the graded final state "
              "(negative = that gate fails there). Tasks graded on confirmed earlier passes can PASS with a negative "
              "final margin. σ: the vector oracle noise-floor margin of that gate in oracle-std units, where recorded.",
              "", "| Test | Prior role | " + " | ".join(tags) + " |", "| --- | --- | " + " | ".join("---" for _ in tags) + " |"]
    for test in arms[tags[0]]["cells"]:
        row = []
        for tag in tags:
            cell = arms[tag]["cells"][test]
            text = _short(cell["status"])
            margin = cell.get("final_margin")
            if margin:
                text += f" {margin['relative_margin']:+.2f} `{margin['gate']}`"
                floor = (cell.get("noise_floor") or {}).get("gates", {}).get(margin["gate"])
                if floor and isinstance(floor.get("margin_oracle_std"), (int, float)):
                    text += f" ({floor['margin_oracle_std']:+.1f}σ)"
            row.append(text)
        role = "table is the trained object" if arms[tags[0]]["cells"][test]["applicability"] == "non_latent_table" \
            else "sampled latent"
        lines.append(f"| `{test}` | {role} | " + " | ".join(row) + " |")
    lines += ["", END]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=Path("runs/forge"))
    parser.add_argument("--report", type=Path, default=Path("reports/prior-ablation-2026-10-25"))
    args = parser.parse_args()
    receipts = collect(args.queue_root, args.report)
    (args.report / "receipts.json").write_text(json.dumps(receipts, indent=1, sort_keys=True) + "\n")
    readme = args.report / "README.md"
    tables = render(receipts)
    text = readme.read_text() if readme.is_file() else START + "\n" + END + "\n"
    head, rest = text.split(START, 1)
    readme.write_text(head + tables + rest.split(END, 1)[1])
    for tag, arm in receipts["arms"].items():
        print(tag, arm["counts"], arm["wall_seconds"])


if __name__ == "__main__":
    main()
