"""Read saved receipts; do not train, sample, regrade, or change declarations."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.forge.decision_contracts import evaluate
from experiments.forge.queue import Queue

SOURCE = "730b77d2b4e1e6ec34f9aed7a5d061874e70eaebb5af0ff839cff1781b849a8d"
COMMIT = "72da7275034422bc171f9b9b1955c9b150cda045"


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load_study(plan_name, summary_name):
    plan = json.loads((OUT / plan_name).read_text())
    summary = json.loads((OUT / summary_name).read_text())
    assert summary["source_digest"] == plan["source_digest"] == SOURCE
    assert summary["source_commits"] == [COMMIT]
    state = Queue(ROOT / "runs/forge" / plan["study"] / "queue", on_completion=None).inspect()
    grouped = {}
    for job in state["jobs"].values():
        result = job.get("result") or {}
        if not result.get("task_results"):
            continue
        request = state["submissions"][result["cost_owner"]["request"]]["request"]
        assert request["source"]["digest"] == SOURCE
        assert request["source"]["origin_commit"] == COMMIT
        bucket = grouped.setdefault(request["candidate"]["id"], {"request": request, "rows": [], "jobs": {}})
        bucket["rows"].extend(result["task_results"])
        for row in result["task_results"]:
            assert row["gate_status"] in {"PASS", "FAIL"}
            bucket["jobs"][row["task_id"]] = job
    rows = []
    raw = {}
    for candidate in summary["candidates"]:
        bucket = grouped[candidate["candidate_id"]]
        request = bucket["request"]
        receipt_path = ROOT / candidate["receipt"]
        receipt = json.loads(receipt_path.read_text())
        assert receipt["candidate_revision"] == request["candidate_revision"]
        assert receipt["global_recipe_overrides"] == request["candidate"]["recipe_overrides"]
        decision = evaluate(request, bucket["rows"])
        assert decision["binding_errors"] == decision["duplicate_task_ids"] == []
        assert decision["outcome"] == "incomplete"
        assert decision["observed"]["prediction"]["satisfied"] is False
        assert decision["observed"]["falsifier"]["satisfied"] is True
        tasks = {}
        for task in receipt["tasks"]:
            if task["status"] not in {"PASS", "FAIL"}:
                continue
            raw_task = next(t for t in bucket["rows"] if t["task_id"] == task["task_id"])
            job = bucket["jobs"][task["task_id"]]
            attempt_path = Path(next(a["path"] for a in job["attempts"] if a["attempt_id"] == task["attempt_id"]))
            guards = task["guards"]
            assert guards["all_finite"] and guards["unintended_rng_deviations"] == 0
            assert task["metrics"] == raw_task["metrics"]
            tasks[task["task_id"]] = {
                "status": task["status"], "attempt_id": task["attempt_id"],
                "metrics": {k: v for k, v in task["metrics"].items() if type(v) in (int, float, bool, str)},
                "passing_suffix": task["evaluator_result"]["convergence"]["passing_suffix"],
                "execution_steps": task["execution_steps"],
                "sampling_law": task["sampling_law"],
                "initializer": task["initializer"], "initialization_sha256": stable_hash(task["initialization"]),
                "prior_sha256": stable_hash(task["prior"]), "rng_manifest_sha256": task["rng_manifest_sha256"],
                "resource_sha256": stable_hash(job["definition"]["resources"]),
                "task_execution_binding": job["definition"]["science"]["execution"][task["task_id"]],
                "task_evaluation_binding": job["definition"]["science"]["evaluation"][task["task_id"]],
                "task_budget_seconds": job["definition"]["budget_seconds"],
                "all_finite": True, "unintended_rng_deviations": 0,
                "optimizer_updates": guards["optimizer_updates"],
                "raw_artifacts": [artifact(attempt_path / name) for name in ("request.json", "raw-result.json", "graded-result.json", "result.json")]
                    + ([artifact(attempt_path / "adapter-receipt.json")] if (attempt_path / "adapter-receipt.json").exists() else []),
            }
        row = {"candidate_id": candidate["candidate_id"], "candidate_revision": candidate["candidate_revision"],
               "settings": candidate["settings"], "receipt": artifact(receipt_path),
               "tier1_statuses": receipt["qualification"]["tiers"][0]["statuses"],
               "decision_review": decision, "tasks": tasks}
        raw[row["candidate_id"]] = {"receipt": receipt, **bucket}
        rows.append(row)
    return plan, summary, rows, raw


def main():
    ring_plan, ring_summary, ring_rows, ring_raw = load_study("plans.json", "summary.json")
    direct_plan, direct_summary, direct_rows, direct_raw = load_study("plans-direct-moments.json", "direct-moments/summary.json")
    ring_analysis_path = OUT / "current-ring-analysis.json"
    ring_analysis = json.loads(ring_analysis_path.read_text())
    assert ring_analysis["source_digest"] == SOURCE
    by_config = {r["candidate_id"].removeprefix("k3p--"): r for r in ring_rows}
    contrasts = []
    invariants = ("sampling_law", "initializer", "initialization_sha256", "prior_sha256", "rng_manifest_sha256", "resource_sha256", "task_execution_binding", "task_evaluation_binding", "task_budget_seconds", "execution_steps")
    for contrast in ring_analysis["matched_numeric_contrasts"]:
        low = by_config[contrast["low_configuration"]]
        high = by_config[contrast["high_configuration"]]
        recipes = [ring_raw[r["candidate_id"]]["receipt"]["global_recipe_overrides"] for r in (low, high)]
        diff = sorted(k for k in recipes[0] if recipes[0][k] != recipes[1][k])
        assert diff == [contrast["axis"]]
        lt, ht = [r["tasks"]["ring16_acquisition"] for r in (low, high)]
        assert all(lt[k] == ht[k] for k in invariants)
        deltas = {k: ht["metrics"][k] - lt["metrics"][k] for k in contrast["high_minus_low_final_metrics"]}
        assert deltas == contrast["high_minus_low_final_metrics"]
        contrasts.append({**contrast, "only_global_recipe_difference": diff,
                          "matched_control_fields": list(invariants),
                          "low_receipt_sha256": low["receipt"]["sha256"], "high_receipt_sha256": high["receipt"]["sha256"]})
    assert len(contrasts) == 12
    direct_rows.sort(key=lambda r: r["settings"]["lr"])
    for row in direct_rows:
        bucket = direct_raw[row["candidate_id"]]
        task = bucket["rows"][0]
        applied = task["applied"]
        generator = next(g for g in applied["optimizer_group_bindings"] if g["representation"] == "direct_sample_coordinates")
        assert generator["base_betas"] == generator["direct_particle_response"]["step_betas"] == [0.0, 0.999]
        assert generator["prior_lr_mult_consumed"] is generator["prior_betas_consumed"] is False
        assert applied["recipe"]["reg_coeff"] == 170 and applied["recipe"]["d_lr_mult"] == 1.5
        assert applied["noise"]["input_std"] == applied["noise"]["output_std"] == 0
        observations = task["evidence"]["observations"]
        peak = max(observations, key=lambda o: o["mean_abs"])
        assert max(o["grad_med"] for o in observations) <= 1.0
        row["direct_observation"] = {
            "base_particle_lr": generator["base_lr"], "base_critic_lr": 1.5 * generator["base_lr"],
            "actual_step_betas": generator["direct_particle_response"]["step_betas"],
            "gain_range": generator["direct_particle_response"]["lr_gain_range"],
            "prior_lr_mult_consumed": False, "prior_betas_consumed": False,
            "peak_observed_movement": peak["mean_abs"], "peak_step": peak["step"],
            "first_observed_critic_median": observations[0]["grad_med"],
            "final_observed_critic_median": observations[-1]["grad_med"],
            "all_observed_critic_medians_pass": True, "passing_observations": 0,
            "direct_gain_applications": task["evidence"]["guards"]["mechanism_audit"]["mechanisms"]["direct_particle_gain"]["applied"],
        }
    direct_reference = direct_rows[0]["tasks"]["two_pole"]
    assert all(all(r["tasks"]["two_pole"][k] == direct_reference[k] for k in invariants) for r in direct_rows)
    direct_contrasts = []
    for low, high in zip(direct_rows, direct_rows[1:]):
        recipes = [direct_raw[r["candidate_id"]]["receipt"]["global_recipe_overrides"] for r in (low, high)]
        diff = sorted(k for k in recipes[0] if recipes[0][k] != recipes[1][k])
        assert diff == ["lr", "prior_lr_mult"]
        direct_contrasts.append({"low_candidate": low["candidate_id"], "high_candidate": high["candidate_id"],
            "declared_recipe_differences": diff, "nominal_latent_prior_lr": 0.0012,
            "consumed_change": "particle and critic absolute rates increase together; latent prior multiplier is unconsumed on this direct host",
            "mean_abs_delta": high["tasks"]["two_pole"]["metrics"]["mean_abs"] - low["tasks"]["two_pole"]["metrics"]["mean_abs"],
            "low_receipt_sha256": low["receipt"]["sha256"], "high_receipt_sha256": high["receipt"]["sha256"]})
    result = {
        "schema_version": 1, "id": "k3p-global-tier1-v3-matched-diagnostics-v1",
        "scope": "recorded_global_grid_contrasts_and_decision_review", "qualification_input": False,
        "source_digest": SOURCE, "source_commit": COMMIT,
        "reproducer": artifact(Path(__file__)),
        "inputs": [artifact(OUT / name) for name in ("plans.json", "plans-direct-moments.json", "summary.json", "direct-moments/summary.json", "current-ring-analysis.json", "direct-rate-analysis.json")],
        "recorded_cost_seconds": ring_summary["charged_wall_seconds"] + direct_summary["charged_wall_seconds"],
        "declared_combined_ceiling_seconds": 25200,
        "combined_cells": {"PASS": 24, "FAIL": 12, "UNKNOWN": 276},
        "new_tier1_qualified_candidates": [], "new_word_measurements": 0,
        "ring_candidates": ring_rows, "ring_matched_contrasts": contrasts,
        "direct_candidates": direct_rows, "direct_matched_contrasts": direct_contrasts,
        "validation": {"frozen_source_commit_and_digest_match_all_requests": True,
            "compact_and_raw_final_metrics_match": True, "actual_direct_beta2_999_consumed": True,
            "all_matched_cohort_control_bindings_equal": True,
            "all_twelve_numeric_falsifiers_observed": True,
            "all_twelve_formal_contract_reviews_incomplete_from_first_fail_unknowns": True},
        "interpretation": [
            "All eight noisy low-coefficient candidates pass three tasks then fail ring quality; all four clean coefficient170 candidates fail direct movement. Words remain UNKNOWN for all twelve.",
            "Within this fixed cohort, D multiplier1 improves ring HQ and full covariance over.5 in all four matched pairs. Coefficient1 improves full covariance over.5 in all four pairs, sometimes lowering HQ. Prior multiplier1 improves HQ in all four pairs, with one slightly worse full covariance. These finite observations do not establish a general monotonic law.",
            "The lowest ring full covariance error is1.4863335117697716 at coefficient1/D1/prior1, still above the required.85. The saved-output forensic analysis verifies substantial served tails and residual core errors; trimming tails would change the gate.",
            "The .999 direct second moment is actually installed. Its analytic low-rate bound permits a pass but does not promise one: movement atLR.0006 is.03008131869137287, far below.3. Raising the coupled rates increases movement to.1193917915225029 without passing; this does not isolate particle LR from critic LR or identify a single restoring-force cause.",
            "The final direct critic medians are.0032-.0047 and pass their upper-bound gate. The remaining low movement, despite actual gain applications, supports investigation of the learned force and regularization balance rather than an unconsumed beta control. It is not proof of an impossible formulation.",
            "Decision prediction and falsifier metrics are measured, with prediction false and falsifier true in all twelve. Formal decision outcome remains incomplete because the five-task scope contains ordinary prerequisite-gated UNKNOWN rows; this neither changes the FAIL gates nor authorizes additional execution.",
            "The historical word-only coefficient170 receipt remains separate motivation. No current whole-family candidate has demonstrated all five passes, and no word-only receipt is promoted as family qualification."
        ],
        "recommendations": [
            "Publish the terminal bounded studies and retain the current family selection; adopt none of these twelve as a Tier1 standard.",
            "Use the saved ring output analysis and compact bindings to target the unresolved full-mass shape problem before declaring another search. The current artifacts cannot attribute tails to specific prior rows or generator derivatives because ring checkpoints were not retained.",
            "Treat the clean coefficient170 movement failures as a measured tradeoff, not a blanket word-formulation failure. No further round, new stabilization technique, seed variant, task-specific override or gate relaxation is declared here."
        ],
        "training_updates_added": 0, "sampling_draws_added": 0,
    }
    destination = OUT / "diagnosis-current.json"
    destination.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": artifact(destination), "ring_pairs": len(contrasts), "direct_pairs": len(direct_contrasts), "cost_seconds": result["recorded_cost_seconds"]}))


if __name__ == "__main__":
    main()
