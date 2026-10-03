"""Compact exact ordinary search receipts and render saved observations only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import file_hash, stable_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": file_hash(path)}


def main():
    plan = json.loads((ROOT / "reports/forge/family-wide-word-repairs/rates-plans.json").read_text())
    receipts, measured, paid = [], 0, 0
    for family in plan["families"]:
        study_path = ROOT / f"reports/forge/configuration-search/{family['study']}.json"
        study = json.loads(study_path.read_text())
        assert study["selection"]["all_trials_terminal"]
        for trial in study["trials"]:
            label = f"{family['family']}-lr{trial['settings']['lr']:.6g}"
            tasks = []
            for task in trial["tasks"]:
                if task["gate_status"] == "UNKNOWN":
                    tasks.append({"task_id": task["task"], "tier": task["qualification_tier"],
                        "status": "UNKNOWN", "reason": "gated by this complete candidate's earlier required failure"})
                    continue
                assert task["gate_status"] in {"PASS", "FAIL"}
                directory = ROOT / "reports/forge/attempts" / task["attempt_id"]
                envelope = json.loads((directory / "request.json").read_text())
                request = envelope["request"]
                result = json.loads((directory / "result.json").read_text())
                certificate = json.loads((directory / "evidence.json").read_text())
                assert certificate["result_hash"] == stable_hash(result)
                assert certificate["source"] == request["source"]
                assert certificate["runtime"] == request["runtime"]
                assert result["candidate_revision"] == trial["candidate_revision"] == request["candidate_revision"]
                assert request["source"]["origin_commit"] == "36ed5655d9e6cfd7f92626cc4ca8c18e07edcb45"
                row = next(r for r in result["task_results"] if r["task_id"] == task["task"])
                assert row["compatibility_key"] == task["compatibility_key"]
                assert row["raw_status"] == "completed" and row["gate_status"] == task["gate_status"]
                assert row["evaluator_result"]["convergence"]["observations"] == 24
                assert row["evidence"]["guards"]["all_finite"]
                assert row["evidence"]["guards"]["unintended_rng_deviations"] == 0
                local = Path(certificate["local_artifact_root"])
                media = ROOT / f"reports/forge/family-wide-word-repairs/media/{label}-{task['task']}.gif"
                subprocess.run(["/home/martyn/dev/ParticleGAN/.venv/bin/python",
                    str(ROOT / "reports/forge/family-wide-word-repairs/render.py"),
                    str(local), "--output", str(media.relative_to(ROOT))], cwd=ROOT, check=True)
                tasks.append({"task_id": task["task"], "tier": task["qualification_tier"],
                    "status": row["gate_status"], "metrics": row["metrics"],
                    "evaluator_result": row["evaluator_result"], "guards": row["evidence"]["guards"],
                    "execution_steps": request["tasks"][task["task"]]["execution"]["steps"],
                    "thresholds": request["tasks"][task["task"]]["evaluation"]["thresholds"],
                    "effective_recipe": row.get("recipe", row.get("applied", {}).get("recipe")),
                    "training_schedules": row.get("applied", {}).get("training_schedules"),
                    "optimizer_group_bindings": row.get("applied", {}).get("optimizer_group_bindings"),
                    "attempt_id": task["attempt_id"], "request_id": request["request_id"],
                    "compatibility_key": row["compatibility_key"],
                    "charged_wall_seconds": row["cost"]["wall_seconds"],
                    "durable_certificate": {"valid": True, "result_stable_hash": certificate["result_hash"],
                        "artifacts": [identity(directory / p) for p in ("request.json", "result.json", "evidence.json")]},
                    "media": str(media.with_suffix(".json").relative_to(ROOT))})
                measured += 1
                paid += row["cost"]["wall_seconds"]
            first = next(t for t in tasks if t["task_id"] == "two_pole")
            receipt = {"schema_version": 1, "scope": "ordinary_global_numeric_configuration",
                "candidate_id": trial["candidate_id"], "candidate_revision": trial["candidate_revision"],
                "configuration_id": trial["configuration_id"], "family": family["family"],
                "settings": trial["settings"], "global_recipe_overrides": trial["recipe_overrides"],
                "source_commit": "36ed5655d9e6cfd7f92626cc4ca8c18e07edcb45",
                "source_digest": trial["source_digest"], "runtime_cohort": trial["runtime_cohort"],
                "tasks": tasks, "required_passes_by_tier": [sum(t["tier"] == i and t["status"] == "PASS" for t in tasks) for i in (1,2,3)],
                "unknown_count": sum(t["status"] == "UNKNOWN" for t in tasks),
                "prediction_supported": first["metrics"]["mean_abs"] >= .3,
                "falsifier_triggered": first["metrics"]["mean_abs"] < .3,
                "ordinary_word_measured": False, "qualified_tier": 0,
                "prior_word_qualification_reuse": False, "default_adoption": False,
                "charged_wall_seconds": trial["cost"]["new_paid_wall_seconds"],
                "search_report": identity(study_path)}
            path = ROOT / f"reports/forge/family-wide-word-repairs/rates/{label}.json"
            write(path, receipt)
            receipts.append({"candidate_id": trial["candidate_id"], "family": family["family"],
                "settings": trial["settings"], "prediction_supported": receipt["prediction_supported"],
                "required_passes_by_tier": receipt["required_passes_by_tier"],
                "unknown_count": receipt["unknown_count"], "receipt": str(path.relative_to(ROOT)),
                "charged_wall_seconds": receipt["charged_wall_seconds"]})
    assert len(receipts) == 6 and measured == 9
    summary = {"schema_version": 1, "round": "family-wide-word-repair-rates-v1",
        "source_commit": "36ed5655d9e6cfd7f92626cc4ca8c18e07edcb45",
        "source_digest": plan["source_digest"], "configuration_count": 6,
        "candidates": receipts, "measured_task_count": measured,
        "measured_pass": 3, "measured_fail": 6,
        "unknown_count": sum(r["unknown_count"] for r in receipts),
        "charged_wall_seconds": paid, "campaign_ceiling_seconds": 293400,
        "grouped_job_ceiling_seconds": 271800, "ordinary_word_measured": 0,
        "default_adoption": False, "new_configured_family_standards": 0,
        "decision": "The bounded coupled-rate prediction was supported only for R1/R2 LR .0085; that complete configuration passed three Tier1 tasks then failed ring16 quality. The other five configurations failed movement. All six full candidates remain unqualified, with word and later cells UNKNOWN. Preserve historical family incumbents and exact independent outcomes; no automatic paid continuation."}
    write(ROOT / "reports/forge/family-wide-word-repairs/rates-summary.json", summary)
    print(json.dumps({"configurations": 6, "measured": measured,
        "gated_unknown": summary["unknown_count"], "charged_wall_seconds": paid}))


if __name__ == "__main__":
    main()
