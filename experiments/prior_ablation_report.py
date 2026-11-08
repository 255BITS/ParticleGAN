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
# Ordinary qualification view whose roles (counted, held, diagnostic) are shown
# per test. The fixed denominator counts held tests regardless.
ORDINARY_VIEW = Path("configs/forge/views/discriminator_stability.json")
START, END = "<!-- prior-ablation tables: start -->", "<!-- prior-ablation tables: end -->"
CIFAR_START, CIFAR_END = "<!-- prior-ablation cifar: start -->", "<!-- prior-ablation cifar: end -->"
# CIFAR-10 particle DDGAN arms (configs/prior_ablation_cifar/*.yaml): (tag, run, label).
CIFAR_ARMS = [("C1", "ddgan_learned", "Learned table (host default)"),
              ("C2", "ddgan_frozen", "Frozen table"),
              ("C3", "ddgan_fresh_gaussian", "Fresh N(0, I) latent")]


def _gates(task):
    evaluation = task.get("evaluation", {})
    gates = [tuple(item) for item in evaluation.get("thresholds", []) if len(item) == 3]
    for name, value in (evaluation.get("accuracy_limits") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gates.append((name, "<=", value))
    for name, value in (evaluation.get("coverage_thresholds") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and name[:4] in ("min_", "max_"):
            gates.append((name[4:], ">=" if name.startswith("min_") else "<=", value))
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
            if test in blockers.get(tag, {}) and found is None:
                # ``found`` is the last study's completion. The arm's final plan
                # blocks this task, so an earlier superseded request's (crashed)
                # attempt must not mask BLOCKED as INCOMPLETE.
                event = None
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
        for test, cell in cells.items():
            if cell["status"] != "NOT_MEASURED":
                continue
            task = json.loads(Path(f"configs/forge/tasks/{test}.json").read_text())
            parents = [d.get("task") if isinstance(d, dict) else d for d in task.get("dependencies", [])]
            if test == "ring_extension":
                parents = ["ring16_acquisition"]
            failed = [name for name in parents if cells.get(name, {}).get("status", "PASS") != "PASS"]
            if failed:
                cell["reason"] = "prerequisite not PASS: " + ", ".join(
                    f"{name} ({cells[name]['status']})" for name in failed)
        counts = {}
        for cell in cells.values():
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
        arms[tag] = {"label": label, "studies": list(studies), "candidate": candidate,
                     "request_ids": [requests.get(name, {}).get("request_id") for name in studies], "candidate_revision": request.get("candidate_revision"),
                     "source_digest": (request.get("source") or {}).get("digest"),
                     "counts": counts, "cells": cells,
                     "wall_seconds": round(sum(c.get("wall_seconds", 0.) for c in cells.values()), 1)}
    view = json.loads(ORDINARY_VIEW.read_text())
    roles = {item["task"]: item.get("importance") for item in view.get("assignments", [])}
    for arm in arms.values():
        for test, cell in arm["cells"].items():
            cell["ordinary_view_role"] = roles.get(test, "not in view")
    return {"schema": "prior_ablation_receipts_v1", "campaign": CAMPAIGN, "fixed_list": str(FIXED_LIST),
            "ordinary_view": {"id": view.get("id"), "revision": view.get("revision")},
            "fixed_list_sha256": hashlib.sha256(FIXED_LIST.read_bytes()).hexdigest(), "arms": arms}


def collect_cifar(cifar_root: Path) -> dict:
    """Compact receipts for the CIFAR arms: final FID50k, FID5k curve, evaluation noise floor, provenance."""
    arms = {}
    for tag, run, label in CIFAR_ARMS:
        out = cifar_root / run
        arm = {"label": label, "run": run, "config": f"configs/prior_ablation_cifar/{run}.yaml", "status": "NOT_MEASURED"}
        if (out / "summary.json").is_file():
            summary = json.loads((out / "summary.json").read_text())
            curve = {}
            for line in (out / "metrics.jsonl").open():
                row = json.loads(line)
                if row.get("generation"):
                    curve[str(row["step"])] = round(row["generation"]["fid5k"], 3)
            final = summary["final"]
            sources = json.loads((out / "provenance.json").read_text())["sources"]
            arm.update(status="COMPLETE", fid50k=round(final["fid"], 3), fid5k_curve=curve,
                       feature_variance_ratio=round(final["feature_variance_ratio"], 4),
                       steps=final["step"], train_seconds=round(summary["train_seconds"], 1),
                       total_seconds=round(summary["total_seconds"], 1),
                       prior_rms_movement=round(summary["prior_rms_movement"], 5),
                       initialization_sha256=summary["metadata"]["initialization_sha256"],
                       rng_sha256=summary["rng_sha256"], gpu=summary["metadata"]["gpu"],
                       sources_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                       config_prior={k: summary["config"][k] for k in ("prior_update", "prior_code_law", "seed", "steps")})
            # Interrupted earlier attempts (same config and seed) give a GPU-nondeterminism floor.
            for previous in sorted(cifar_root.glob(run + ".attempt*")):
                if previous.is_dir() and (previous / "metrics.jsonl").is_file():
                    rerun = {}
                    for line in (previous / "metrics.jsonl").open():
                        row = json.loads(line)
                        if row.get("generation"):
                            rerun[str(row["step"])] = round(row["generation"]["fid5k"], 3)
                    arm.setdefault("interrupted_attempts", {})[previous.name] = {
                        "fid5k_curve": rerun, "abs_delta_vs_final_run": {
                            step: round(abs(value - curve[step]), 3) for step, value in rerun.items() if step in curve}}
            if (out / "fid_floor.json").is_file():
                floor = json.loads((out / "fid_floor.json").read_text())
                arm["fid50k_eval_floor"] = {"draws": len(floor["rows"]), "values": [round(r["fid"], 3) for r in floor["rows"]],
                                            "mean": round(floor["mean"], 3), "std": round(floor["std"], 3)}
        arms[tag] = arm
    return {"protocol": "lib.cifar_metrics.PROTOCOL (torch-fidelity Inception-v3 2048, CIFAR-10 train 50k real)",
            "arms": arms}


def render_cifar(cifar: dict) -> str:
    arms = cifar["arms"]
    lines = [CIFAR_START, "", "| Arm | Prior | FID50k ↓ | Eval floor (std, n) | FID5k @2500 / 5000 / 7500 / 10000 | "
             "Feature var. ratio | Prior RMS move | Train s |", "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for tag, arm in arms.items():
        if arm["status"] != "COMPLETE":
            lines.append(f"| {tag} | {arm['label']} | not measured | | | | | |")
            continue
        floor = arm.get("fid50k_eval_floor")
        floor_text = f"{floor['std']:.2f} (n={floor['draws']})" if floor else "—"
        curve = " / ".join(f"{arm['fid5k_curve'].get(str(s), float('nan')):.1f}" for s in (2500, 5000, 7500, 10000))
        lines.append(f"| {tag} | {arm['label']} | {arm['fid50k']:.2f} | {floor_text} | {curve} | "
                     f"{arm['feature_variance_ratio']:.3f} | {arm['prior_rms_movement']:.4f} | {arm['train_seconds']:.0f} |")
    deltas = [d for arm in arms.values() for attempt in arm.get("interrupted_attempts", {}).values()
              for d in attempt["abs_delta_vs_final_run"].values()]
    if deltas:
        lines += ["", f"Training-nondeterminism floor: interrupted same-config attempts differ from the final runs by "
                  f"{min(deltas):.1f}–{max(deltas):.1f} FID5k at matched checkpoints (n={len(deltas)})."]
    lines += ["", CIFAR_END]
    return "\n".join(lines)


def _short(status):
    return {"PASS": "PASS", "FAIL": "**FAIL**", "BLOCKED": "blocked", "NOT_MEASURED": "not run"}.get(
        status, status.lower())


def render(receipts: dict) -> str:
    arms = receipts["arms"]
    view = receipts["ordinary_view"]
    tags = list(arms)
    lines = [START, "", "### Leaderboard (fixed denominator: 49 tests)", "",
             "| Arm | Declared delta | PASS | FAIL | Blocked | Not measured | Incomplete | GPU seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for tag in tags:
        arm = arms[tag]
        c = arm["counts"]
        other = sum(v for k, v in c.items() if k not in ("PASS", "FAIL", "BLOCKED", "NOT_MEASURED"))
        lines.append(f"| {tag} | {arm['label']} | {c.get('PASS', 0)} | {c.get('FAIL', 0)} | {c.get('BLOCKED', 0)} | "
                     f"{c.get('NOT_MEASURED', 0)} | {other} | {arm['wall_seconds']:.0f} |")
    lines.append(f"| A5 | `ot_target_column_normalize=True` | 0 | 0 | 0 | {len(arms[tags[0]]['cells'])} | 0 | 0 |")
    lines += ["", "A5 is not measured on any test: the flag exists only on the standalone `BISPFEngine`, "
              "which no recipe uses (see Method)."]
    graded = ("PASS", "FAIL")
    matched = [test for test in arms[tags[0]]["cells"]
               if all(arms[tag]["cells"][test]["status"] in graded for tag in tags[:3])]
    lines += ["", f"### Matched sampled-latent subset ({len(matched)} tests graded in A1, A2 and A3)", "",
              "The only tests where all three prior arms produced a PASS/FAIL grade: " +
              ", ".join(f"`{test}`" for test in matched) + ".", "",
              "| Arm | PASS | FAIL | Passing tests |", "| --- | ---: | ---: | --- |"]
    for tag in tags:
        cells = arms[tag]["cells"]
        passed = [test for test in matched if cells[test]["status"] == "PASS"]
        failed = [test for test in matched if cells[test]["status"] == "FAIL"]
        lines.append(f"| {tag} | {len(passed)} | {len(failed)} | " + ", ".join(f"`{t}`" for t in passed) + " |")
    lines += ["", "### Per-test outcomes and final-state margins", "",
              "Margin: the smallest relative signed distance to a declared threshold at the graded final state "
              "(negative = that gate fails there). Tasks graded on confirmed earlier passes can PASS with a negative "
              "final margin. σ: the vector oracle noise-floor margin of that gate in oracle-std units, where recorded. "
              "Only the `vector_*` gates record a noise floor; the other margins, including the +0.00 mode-count "
              "margins of `ring16_acquisition` and `mode_hold`, have none.",
              "", f"Ordinary view: the test's role in `{view['id']}` revision {view['revision']}. A *held* test is "
              "excluded from qualification totals there but still counts in this fixed denominator; a held FAIL "
              "is still a FAIL here.",
              "", "| Test | Prior role | Ordinary view | " + " | ".join(tags) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in tags) + " |"]
    for test in arms[tags[0]]["cells"]:
        row = []
        for tag in tags:
            cell = arms[tag]["cells"][test]
            text = _short(cell["status"])
            if cell["status"] == "NOT_MEASURED" and cell.get("reason", "").startswith("prerequisite"):
                text = "not run (prereq.)"
            margin = cell.get("final_margin")
            if margin:
                text += f" {margin['relative_margin']:+.2f} `{margin['gate']}`"
                floor = (cell.get("noise_floor") or {}).get("gates", {}).get(margin["gate"])
                if floor and isinstance(floor.get("margin_oracle_std"), (int, float)):
                    text += f" ({floor['margin_oracle_std']:+.1f}σ)"
            row.append(text)
        role = "table is the trained object" if arms[tags[0]]["cells"][test]["applicability"] == "non_latent_table" \
            else "sampled latent"
        view_role = arms[tags[0]]["cells"][test].get("ordinary_view_role")
        view_role = {"required": "counted"}.get(view_role, view_role)
        lines.append(f"| `{test}` | {role} | {view_role} | " + " | ".join(row) + " |")
    lines += ["", END]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=Path("runs/forge"))
    parser.add_argument("--report", type=Path, default=Path("reports/prior-ablation-2026-10-25"))
    parser.add_argument("--cifar-root", type=Path, default=Path("/nvme/dev/particlegan/prior-ablation-runs/cifar"))
    args = parser.parse_args()
    receipts = collect(args.queue_root, args.report)
    receipts["cifar"] = collect_cifar(args.cifar_root)
    (args.report / "receipts.json").write_text(json.dumps(receipts, indent=1, sort_keys=True) + "\n")
    readme = args.report / "README.md"
    tables = render(receipts)
    text = readme.read_text() if readme.is_file() else START + "\n" + END + "\n"
    head, rest = text.split(START, 1)
    text = head + tables + rest.split(END, 1)[1]
    if CIFAR_START in text:
        head, rest = text.split(CIFAR_START, 1)
        text = head + render_cifar(receipts["cifar"]) + rest.split(CIFAR_END, 1)[1]
    readme.write_text(text)
    for tag, arm in receipts["arms"].items():
        print(tag, arm["counts"], arm["wall_seconds"])
    for tag, arm in receipts["cifar"]["arms"].items():
        print(tag, arm["status"], arm.get("fid50k"), arm.get("fid50k_eval_floor"))


if __name__ == "__main__":
    main()
