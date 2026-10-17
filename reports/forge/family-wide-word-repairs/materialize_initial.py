"""Publish compact outcomes of the three frozen ordinary global candidates.

Read exact durable certificates; do not train, sample, regrade or transplant
historical task-only word evidence into ordinary qualification.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import stable_hash


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    expected = ["k3p-global-repair-v1", "ka2-global-repair-v1", "r1r2-global-repair-v1"]
    receipts = {}
    for directory in (ROOT / "reports/forge/attempts").iterdir():
        if not (directory / "request.json").exists():
            continue
        envelope = json.loads((directory / "request.json").read_text())
        request = envelope["request"]
        name = request["candidate"]["id"]
        if name not in expected:
            continue
        result = json.loads((directory / "result.json").read_text())
        certificate = json.loads((directory / "evidence.json").read_text())
        assert certificate["result_hash"] == stable_hash(result)
        assert certificate["source"] == request["source"]
        assert certificate["runtime"] == request["runtime"]
        assert result["candidate_revision"] == request["candidate_revision"]
        assert request["source"]["origin_commit"] == "9d089ca777e647c6d3b6ad56529ac27f171fdcb6"
        assert name not in receipts, "No repeat or cross-cohort pooling permitted"
        rows = result["task_results"]
        assert len(rows) == 1 and rows[0]["task_id"] == "two_pole"
        row = rows[0]
        assert row["raw_status"] == "completed" and row["gate_status"] == "FAIL"
        assert row["evaluator_result"]["convergence"]["observations"] == 24
        assert row["evidence"]["guards"]["optimizer_updates"] == {"discriminator": 80, "prior": 80}
        receipt = {"schema_version": 1, "scope": "ordinary_global_candidate",
            "candidate_id": name, "candidate_revision": request["candidate_revision"],
            "attempt_id": directory.name, "request_id": request["request_id"],
            "source_commit": request["source"]["origin_commit"],
            "source_digest": request["source"]["digest"], "runtime": request["runtime"],
            "compute": envelope["job"]["science"]["compute"],
            "global_recipe_overrides": request["candidate"]["recipe_overrides"],
            "measured_task": {"task_id": row["task_id"], "gate_status": row["gate_status"],
                "metrics": row["metrics"], "evaluator_result": row["evaluator_result"],
                "guards": row["evidence"]["guards"],
                "training_schedules": row["applied"]["training_schedules"],
                "optimizer_group_bindings": row["applied"]["optimizer_group_bindings"],
                "compatibility_key": row["compatibility_key"]},
            "tasks": [{"task_id": a["task"], "tier": a["qualification_tier"],
                "status": row["gate_status"] if a["task"] == "two_pole" else "UNKNOWN",
                "reason": "measured ordinary first gate" if a["task"] == "two_pole" else
                    "gated by measured two_pole failure; no task-only witness reuse"}
                for a in request["view"]["assignments"]],
            "charged_wall_seconds": row["cost"]["wall_seconds"],
            "prediction": "two_pole.mean_abs >= 0.3", "prediction_supported": False,
            "falsifier_triggered": True, "ordinary_word_measured": False,
            "qualified_tier": 0, "prior_word_qualification_reuse": False,
            "durable_certificate": {"result_stable_hash": certificate["result_hash"],
                "valid": True, "artifacts": [identity(directory / p) for p in
                    ("request.json", "result.json", "evidence.json")]},
            "media": f"reports/forge/family-wide-word-repairs/media/{name}-two_pole.json"}
        receipts[name] = receipt
    assert set(receipts) == set(expected), "All three declared candidates must finish"
    for name, receipt in receipts.items():
        write(ROOT / f"reports/forge/family-wide-word-repairs/initial/{name}.json", receipt)
    summary = {"schema_version": 1, "round": "family-wide-word-repairs-v1",
        "scope": "ordinary_global_family_candidates", "source_commit": receipts[expected[0]]["source_commit"],
        "source_digest": receipts[expected[0]]["source_digest"],
        "candidates": [{"candidate_id": name, "attempt_id": receipts[name]["attempt_id"],
            "metrics": receipts[name]["measured_task"]["metrics"], "qualified_tier": 0,
            "measured_fail": 1, "gated_unknown": 25,
            "receipt": f"reports/forge/family-wide-word-repairs/initial/{name}.json"} for name in expected],
        "charged_wall_seconds": sum(r["charged_wall_seconds"] for r in receipts.values()),
        "candidate_ceiling_seconds": 48900, "campaign_ceiling_seconds": 146700,
        "prediction_supported": False, "falsifier_triggered_for_all_candidates": True,
        "decision": "All complete candidates failed the unchanged 80-update movement prerequisite. This rejects the whole candidates and leaves 75 later cells unmeasured; it neither establishes a word failure nor qualifies historical word witnesses. Retain historical family incumbents. A separately bounded global coupled-rate search tests the observed short-horizon rate tradeoff."}
    write(ROOT / "reports/forge/family-wide-word-repairs/initial-summary.json", summary)
    print(json.dumps({"candidates": 3, "fail": 3, "gated_unknown": 75,
                      "charged_wall_seconds": summary["charged_wall_seconds"]}))


if __name__ == "__main__":
    main()
