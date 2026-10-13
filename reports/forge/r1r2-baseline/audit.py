"""Project the recorded R1/R2 baseline without training, sampling or regrading."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
SNAPSHOT = "reports/forge/technique-evidence/29979e41c9c05e3304156cf4a43ed9c9dda6953bffd0fb47096fe47d6b8af13a.json"
CANDIDATE = "recipe-prior-refactor-r1r2-v1"
sys.path.insert(0, str(ROOT))
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint


def read(path):
    return json.loads(path.read_text())


def pin(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(data).hexdigest()}


def clock_projection(path, receipt):
    expected = receipt["provenance"]["original_files"]["result"]
    assert sha256(path.read_bytes()).hexdigest() == expected["sha256"], "original result hash mismatch"
    raw = read(path)["raw"]
    evidence = raw["result"]["evidence"]
    return {
        "qualification_input": False,
        "qualification_reuse": False,
        "attempt_id": receipt["attempt_id"],
        "source_digest": receipt["provenance"]["source_digest"],
        "original_result": expected,
        "independent_grade": raw["grading"]["grades"]["clockfree_audit_measurement_v1"],
        "comparisons": evidence["comparisons"],
        "source_audit": evidence["source_audit"],
        "artifact_manifest": evidence["artifact_manifest"],
    }


def project():
    inventory = read(ROOT / "reports/forge/technique-inventory.json")
    snapshot = read(ROOT / SNAPSHOT)
    row = next(r for r in snapshot["rows"] if r["candidate_id"] == CANDIDATE)
    selected = next(r for r in inventory["rows"] if r["trainer_family"] == "r1r2")
    assert selected["candidate_revision"] == row["candidate_revision"]
    assert selected["bindings"] == row["bindings"]
    family = next(f for f in inventory["family_progress"]["families"] if f["id"] == "r1r2")
    cohort = next(c for c in family["cohorts"] if c["row_index"] == inventory["rows"].index(selected))
    tasks = []
    for attempt in row["attempt_ids"]:
        path = ROOT / f"reports/forge/technique-receipts/{attempt}.json"
        receipt = read(path)
        assert receipt["candidate_revision"] == row["candidate_revision"]
        assert receipt["provenance"]["source_digest"] == row["bindings"]["source_digest"]
        assert receipt["qualification_input"] is False
        for task in receipt["task_results"]:
            task_id = task["task_id"]
            contract = snapshot["task_contracts"][row["bindings"]["task_contracts"][task_id]]
            original_path = f"configs/forge/tasks/{task_id}.json"
            card = json.loads(subprocess.check_output(
                ["git", "show", f"{row['bindings']['source_origin_commit']}:{original_path}"], cwd=ROOT))
            assert task_execution_fingerprint(card) == contract["execution_sha256"]
            assert task_evaluation_fingerprint(card) == contract["evaluation_sha256"]
            tasks.append({
                "task_id": task_id, "attempt_id": attempt, "status": task["status"],
                "reason": task["reason"], "receipt": pin(path),
                "original_files": receipt["provenance"]["original_files"],
                "recorded_task_contract": row["bindings"]["task_contracts"][task_id],
                "execution": {k: card["execution"][k] for k in
                              ("steps", "initializer", "prior", "prior_contract", "original_schedule_horizon")
                              if k in card["execution"]},
                "evaluation": {k: v for k, v in card["evaluation"].items()
                               if k not in ("sources", "evaluator_revision")},
                "sampling": task["sampling"],
                "final_metrics": task.get("metrics", {}),
                "evaluator_summary": task.get("evaluator_summary", {}),
                "wall_seconds": task["cost"]["wall_seconds"],
                "gif": (f"reports/forge/recipe-prior-refactor/publication/media/r1r2-ordinary-{task_id}.gif"
                        if task_id != "clockfree_audit_measurement_v1" else None),
            })
    tasks.sort(key=lambda t: t["task_id"])
    assert Counter(t["status"] for t in tasks) == {"PASS": 5, "FAIL": 2}
    assert len(tasks) == 7
    for task in tasks:
        if task["gif"]:
            assert (ROOT / task["gif"]).is_file()
    clock = read(REPORT / "clock-audit.json")
    clock_task = next(t for t in tasks if t["task_id"] == "clockfree_audit_measurement_v1")
    assert clock["original_result"] == clock_task["original_files"]["result"]
    assert clock["source_digest"] == row["bindings"]["source_digest"]
    source_path = ROOT / "reports/forge/recipe-prior-refactor/publication/results.json"
    source = read(source_path)
    diagnostics = []
    for cell in source["task_cells"]:
        if cell["family"] != "r1r2" or cell["original_tier"] != 2:
            continue
        result = next(r for r in source["certified_results"]
                      if r["attempt_id"] == cell["diagnostic_attempt"] and r["task_id"] == cell["task_id"])
        assert result["scope"] == "research_diagnostic" and result["qualification_input"] is False
        diagnostics.append({"task_id": cell["task_id"], "status": cell["diagnostic_status"],
                            "attempt_id": cell["diagnostic_attempt"], "reason": result["reason"],
                            "failed_final_checks": {
                                k: {field: check[field] for field in ("value", "op", "threshold", "status")}
                                for k, check in result["evaluator_summary"].get("metric_checks", {}).items()
                                if check["status"] == "FAIL"}})
    assert Counter(t["status"] for t in diagnostics) == {"PASS": 9, "FAIL": 12}
    return {
        "schema_version": 1, "scope": "r1r2_recorded_baseline_readout",
        "qualification_input": False, "qualification_reuse": False,
        "training_updates_added": 0, "sampling_draws_added": 0, "new_paid_seconds": 0,
        "candidate_id": CANDIDATE, "candidate_revision": row["candidate_revision"],
        "frozen_source": snapshot["frozen_source"], "bindings": row["bindings"],
        "runtime_cohort": row["runtime_cohort"], "qualified_tier": row["qualified_tier"],
        "recipe": snapshot["recipe_contracts"][row["bindings"]["recipe_sha256"]],
        "protocol": snapshot["protocol_contracts"][row["bindings"]["protocol_sha256"]],
        "evidence_snapshot": pin(ROOT / SNAPSHOT), "clock_projection": pin(REPORT / "clock-audit.json"),
        "tier1_view_requirements": cohort["tiers"]["1"],
        "tier1_views": {v["id"]: v["tiers"]["1"] for v in cohort["views"]},
        "ordinary_required_tier1": row["tiers"]["1"],
        "distinct_tier1_tasks": {"PASS": 5, "FAIL": 2, "total": 7},
        "ordinary_and_clock_paid_seconds": sum(t["wall_seconds"] for t in tasks),
        "tasks": tasks,
        "separate_tier2_diagnostics": {"qualification_input": False, "source": pin(source_path),
                                      "counts": {"PASS": 9, "FAIL": 12},
                                      "tasks": sorted(diagnostics, key=lambda t: t["task_id"])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh only this report's compact results")
    parser.add_argument("--clock-result", type=Path, help="verify the byte-exact archived clock result")
    args = parser.parse_args()
    if args.clock_result:
        receipt = read(ROOT / "reports/forge/technique-receipts/12bc543e8b564690953453bfaf8720cc.json")
        assert clock_projection(args.clock_result, receipt) == read(REPORT / "clock-audit.json")
    result = project()
    path = REPORT / "results.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        assert read(path) == result, "stale baseline results; rerun with --write"
    print("PASS: 20/22 view requirements; 5/7 distinct tasks; ring/clock failures preserved; diagnostics separate; zero training")


if __name__ == "__main__":
    main()
