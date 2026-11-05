"""Write the ring16 6,400-update task revision, its diagnostic view and one study per selected family.

Reproduction source for dig2 lead ``ring16-cap``. Run from the repository root:

    PYTHONPATH=. python reports/forge/dig2-20261016/ring16-cap-make.py

The original ``ring16_acquisition`` declaration is not modified; the new task is a
separate identity that names it as its predecessor. Only the update cap, the
task-owned schedule horizon, the observation count (to keep the original cadence
of one check every 16.67 updates), the description and the wall-clock allowance
change. Prior, host, initializer, sampling law, scorer, thresholds and the
5-check terminal suffix rule are copied unchanged.
"""
import json
from copy import deepcopy
from pathlib import Path

from experiments.forge.contracts import file_hash

OLD = Path("configs/forge/tasks/ring16_acquisition.json")
NEW_ID = "ring16_acquisition_6400"
CAP = 6400
OLD_CAP = 1600
OLD_OBS = 96

old = json.loads(OLD.read_text())
assert old["execution"]["steps"] == OLD_CAP and old["evaluation"]["observations"] == OLD_OBS
task = deepcopy(old)
task["id"] = NEW_ID
task["execution"]["steps"] = CAP
task["execution"]["host_definition"]["steps"] = CAP
# Each recipe's own step schedules (LR cosine, noise anneals, coefficient ramps) are
# declared over the full new cap, as in ring16-schedule-20261011 ("full-budget schedules").
task["execution"]["original_schedule_horizon"] = CAP
# Same cadence as the original (1,600 / 96 = 16.67 updates per check); 96 of the 384
# checkpoints are exactly the 96-check schedule ceil(i * 6400 / 96).
task["evaluation"]["observations"] = OLD_OBS * CAP // OLD_CAP
task["resources"]["timeout_seconds"] = 1200
task["description"] = (
    "Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; "
    "require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread, "
    "sustained over the final 5 checks (one check every 16.67 updates) of 6400 updates. Each recipe's own "
    "step schedule is declared over the full 6400 updates. No extended hold phase. Revision 2 of "
    "ring16_acquisition (1600 updates, schedule horizon 400), which is preserved unchanged.")
task["task_revision"] = {
    "schema_version": 1,
    "revision": 2,
    "revision_of": old["id"],
    "previous_task": str(OLD),
    "previous_task_sha256": file_hash(OLD),
    "changed_fields": {
        "execution.steps": [OLD_CAP, CAP],
        "execution.host_definition.steps": [OLD_CAP, CAP],
        "execution.original_schedule_horizon": [old["execution"]["original_schedule_horizon"], CAP],
        "evaluation.observations": [OLD_OBS, OLD_OBS * CAP // OLD_CAP],
        "resources.timeout_seconds": [old["resources"]["timeout_seconds"], 1200],
    },
    "unchanged": "prior, host architecture, initializer, target, sampling law, scorer, thresholds, "
                 "minimum_stable_checks=5 terminal suffix, observation cadence (16.67 updates)",
    "reason": "User-authorised ring16 cap change (dig2 ring16-cap, 2026-10-16). ring16-schedule-20261011 "
              "showed K3P, KA2 and R1/R2 pass the unchanged gate at 6400 updates with full-budget schedules "
              "and fail at the 1600/3200/4800 cuts; the 1600-update cap was set around BCAP's passing run (#316).",
    "history": "The original ring16_acquisition identity, its receipts and every published verdict are unchanged.",
}
task["research_artifacts"] = {**old.get("research_artifacts", {}),
                              "previous_revision": str(OLD),
                              "cap_evidence": "reports/forge/dig2-20261016/ring16-cap.md"}
Path(f"configs/forge/tasks/{NEW_ID}.json").write_text(json.dumps(task, indent=2) + "\n")

VIEW = "dig2-ring16-cap-diagnostic-v1"
view = {
    "assignments": [
        {"importance": "diagnostic", "order": 0, "qualification_tier": 1, "task": NEW_ID},
        {"importance": "diagnostic", "order": 1, "qualification_tier": 1, "task": "ring16_acquisition"},
    ],
    "calibration": {"adoption_blocker": "Research diagnostic for the user-authorised ring16 cap change; "
                                        "no ordinary qualification follows from this view.",
                    "status": "provisional"},
    "eligibility": {},
    "evidence_scope": "research_diagnostic",
    "goal": "discriminator_stability",
    "id": VIEW,
    "policy_change_reason": ("Every runnable selected recipe on ring16 at the new 6400-update cap (full-budget "
                             "schedules) and, as the matched control on the same frozen source, the original "
                             "1600-update task."),
    "ranking": {"compare_compatible_cohorts": True, "cost_separate": True,
                "policy": "qualified_tier_only_with_raw_metrics"},
    "revision": 1,
    "schema_version": 1,
}
Path(f"configs/forge/views/{VIEW}.json").write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")

FAMILIES = {
    "recipe-prior-refactor-bcap-pure-v1": "BCAP (selected DualNorm recipe)",
    "bispf-v1": "BISPF",
    "recipe-prior-refactor-k3p-v1": "K3P",
    "recipe-prior-refactor-ka2-v1": "KA2",
    "recipe-prior-refactor-r1r2-v1": "R1/R2",
    "recipe-prior-refactor-release07-gan-v3-v1": "GAN v3 release 0.7",
}
for cid, label in FAMILIES.items():
    sid = f"dig2-ring16-cap-{cid}-study"
    study = {
        "campaign": {"budget_seconds": 10800, "candidate_budget_seconds": 1800, "id": "dig2-ring16-cap-v1"},
        "candidate": cid,
        "competing_explanation": ("A longer run can also expose late collapse: constant-rate recipes (BCAP, BISPF) "
                                  "that acquire by update ~700 may lose covariance or spread before update 6400."),
        "control": {"candidate_id": ("bispf-v1" if cid == "recipe-prior-refactor-bcap-pure-v1"
                                     else "recipe-prior-refactor-bcap-pure-v1"), "task_map": {}},
        "falsifier": {"metric": "component_covariance_error", "op": ">", "phase": "final",
                      "task_id": NEW_ID, "threshold": 0.85},
        "hypothesis": (f"{label} with its unchanged selected recipe passes the unchanged ring16 gate at a "
                       "6400-update cap with its own schedule over the full cap."),
        "id": sid,
        "max_rounds": 1,
        "prediction": {"metric": "component_covariance_error", "op": "<=", "phase": "final",
                       "task_id": NEW_ID, "threshold": 0.85},
        "prior_evidence": [{"identity": {"record_id": "ring16-schedule-20261011-readout"},
                            "path": "reports/forge/dig2-20261016/ring16-cap-motivation.json",
                            "selector": [], "use": "motivation_only"}],
        "schema_version": 1,
        "scope": {"cuda_model": "NVIDIA RTX A6000", "execution_backend": "cuda", "through_tier": 1, "view": VIEW},
        "status": "ready",
        "terminal_rules": {"falsified": "stop_revision", "incomplete": "request_missing_evidence",
                           "inconclusive": "stop_and_readout", "prediction_observed": "review_saved_diagnostics"},
    }
    Path(f"configs/forge/studies/{sid}.json").write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    print(sid)
