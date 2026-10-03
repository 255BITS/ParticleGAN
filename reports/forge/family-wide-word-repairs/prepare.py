"""Freeze/review three ordinary global candidates; no training or enqueue."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.planning import resolve_idea, plan_summary
from experiments.forge.word_adapter import word_context


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    selected = json.loads((ROOT / "configs/forge/selections/word-joint-task-v1.json").read_text())
    plans = []
    for row in selected["recipes"]:
        name = f"{row['family']}-global-repair-v1"
        path = ROOT / f"configs/forge/ideas/{name}.json"
        idea = json.loads(path.read_text())
        # Draft mode recomputes the exact expected bindings after all shared
        # execution sources are stable. It does not alter any public recipe.
        idea["decision_contract"]["status"] = "draft"
        write(path, idea)
        request = resolve_idea(ROOT, name, through_tier=3, execution_backend="cuda",
                               cuda_model="NVIDIA RTX A6000")
        expected = request["decision_review"]["expected"]
        contract = idea["decision_contract"]
        contract.update(status="ready", prior_evidence=[{
            "path": row["receipt"]["path"], "sha256": row["receipt"]["sha256"],
            "selector": [], "identity": {"id": row["arm"], "candidate_revision":
                json.loads((ROOT / row["receipt"]["path"]).read_text())["candidate_revision"]},
            "use": "motivation_only"}],
            candidate_binding_sha256=expected["candidate_binding_sha256"],
            substantive_delta=expected["substantive_delta"],
            prediction={"task_id": "two_pole", "metric": "mean_abs", "op": ">=",
                        "threshold": 0.3, "phase": "final"},
            falsifier={"task_id": "two_pole", "metric": "mean_abs", "op": "<",
                       "threshold": 0.3, "phase": "final"},
            competing_explanation="The lower word-compatible global rates may be too small for the frozen 80-update movement screen. A failure rejects this complete ordinary candidate; prior word passes cannot fill its gated cells or justify per-task tuning.")
        contract["control"].update(binding_sha256=expected["control_binding_sha256"],
                                   task_map=expected["task_map"])
        contract["scope"].update(view="discriminator_stability", through_tier=3,
            task_ids=expected["task_ids"], max_rounds=1, candidate_budget_seconds=48900,
            campaign_budget_seconds=146700,
            **{key: expected[key] for key in ("protocol_sha256", "source_digest", "execution_backend",
                                             "runtime_cohort_sha256", "jobs_sha256")})
        write(path, idea)
        request = resolve_idea(ROOT, name, through_tier=3, execution_backend="cuda",
                               cuda_model="NVIDIA RTX A6000")
        assert not request["preflight_blockers"], request["preflight_blockers"]
        assert not any(task["preflight_blockers"] for task in request["tasks"].values())
        effective = word_context(request, request["tasks"]["five_word_joint_acquisition"], "cpu", root=ROOT).recipe
        assert json.loads(json.dumps(asdict(effective))) == row["recipe"]
        write(ROOT / f"runs/forge/family-wide-word-repairs-v1/{name}-ready-request.json", request)
        summary = plan_summary(request, include_ownership=False)
        plans.append({"candidate": name, "candidate_revision": request["candidate_revision"],
            "preparation_checkout_commit": request["source"]["origin_commit"], "source_digest": request["source"]["digest"],
            "idea_sha256": file_hash(path), "decision_status": request["decision_review"]["status"],
            "word_effective_recipe_matches_ancestor": True, "word_witness_qualification_reuse": False,
            "global_recipe_overrides": request["candidate"]["recipe_overrides"],
            "runtime": request["runtime"], "compute_profiles": request["compute_profiles"],
            "protocol_sha256": stable_hash(request["protocol"]),
            "grouped_job_reservation_seconds": summary["worst_case_seconds"],
            "tasks": [{**task, "execution_fingerprint": request["jobs"][next(i for i,j in
                enumerate(request["jobs"]) if task["task"] in j["task_ids"])]["science"]["execution"][task["task"]],
                "evaluation_fingerprint": request["jobs"][next(i for i,j in enumerate(request["jobs"])
                if task["task"] in j["task_ids"])]["science"]["evaluation"][task["task"]]}
                for task in summary["tasks"]]})
    assert len({plan["source_digest"] for plan in plans}) == 1, "Shared execution source changed during preparation"
    write(ROOT / "reports/forge/family-wide-word-repairs/plans.json", {
        "schema_version": 1, "scope": "ordinary_global_family_candidates",
        "view": "discriminator_stability", "through_tier": 3, "candidates": plans,
        "candidate_reservation_seconds": 48900, "campaign_reservation_seconds": 146700,
        "actual_total_grouped_job_reservation_seconds": sum(p["grouped_job_reservation_seconds"] for p in plans),
        "old_task_only_word_receipts_reused_for_qualification": 0,
        "decision": "Evaluate each whole global candidate under ordinary ordered prerequisites. Preserve every gated later cell as unmeasured; no per-task recipe selection, tuning, or automatic continuation."})
    print(json.dumps({"ready_candidates": len(plans), "task_bindings": sum(len(p["tasks"]) for p in plans),
                      "grouped_job_seconds": sum(p["grouped_job_reservation_seconds"] for p in plans)}))


if __name__ == "__main__":
    main()
