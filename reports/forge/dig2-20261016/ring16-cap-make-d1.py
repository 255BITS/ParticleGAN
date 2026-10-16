"""Conditional diagnostic D1 (preregistered rule 6): K3P/KA2 with the G/D LR anneal over the full cap.

    PYTHONPATH=. python reports/forge/dig2-20261016/ring16-cap-make-d1.py

Writes two diagnostic idea cards (selected K3P/KA2 recipe with network_lr_horizon_cap=None, all
other fields unchanged), a one-task diagnostic view and four studies: each diagnostic card and,
as its matched control on the same frozen source, the unchanged selected card.
Not selected-recipe evidence; never enters the recount.
"""
import json
from pathlib import Path

VIEW = "dig2-ring16-cap-d1-diagnostic-v1"
CAMPAIGN = "dig2-ring16-cap-d1-v1"
ARMS = {"recipe-prior-refactor-k3p-v1": "k3p-r16cap-net-horizon-full-v1",
        "recipe-prior-refactor-ka2-v1": "ka2-r16cap-net-horizon-full-v1"}

for parent, cid in ARMS.items():
    card = json.loads(Path(f"configs/forge/ideas/{parent}.json").read_text())
    assert card["recipe_overrides"]["network_lr_horizon_cap"] == 1600
    card["id"] = cid
    card["parent"] = parent
    card["recipe_overrides"]["network_lr_horizon_cap"] = None
    card["changed_factors"] = ["Diagnostic only: network_lr_horizon_cap 1600 -> None, so the G/D cosine LR "
                               "anneal spans the task-owned schedule horizon (6400 on ring16_acquisition_6400) "
                               "instead of min(horizon, 1600). Every other selected-recipe field is unchanged."]
    card["mechanism_rationale"] = ("dig2 ring16-cap preregistered conditional diagnostic D1: the selected recipe "
                                   "failed ring16_acquisition_6400 on min-eigen spread after its G/D LR reached "
                                   "its 0.01 floor at update 1600; test whether the fixed 1600-update network "
                                   "anneal cap is what blocks the 6400-update pass.")
    card["guide"] = "reports/forge/dig2-20261016/ring16-cap.md"
    Path(f"configs/forge/ideas/{cid}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")

view = {
    "assignments": [{"importance": "diagnostic", "order": 0, "qualification_tier": 1,
                     "task": "ring16_acquisition_6400"}],
    "calibration": {"adoption_blocker": "Conditional research diagnostic; no qualification.", "status": "provisional"},
    "eligibility": {}, "evidence_scope": "research_diagnostic", "goal": "discriminator_stability", "id": VIEW,
    "policy_change_reason": "K3P/KA2 selected recipe vs the same recipe with the G/D anneal over the full 6400 cap.",
    "ranking": {"compare_compatible_cohorts": True, "cost_separate": True,
                "policy": "qualified_tier_only_with_raw_metrics"},
    "revision": 1, "schema_version": 1,
}
Path(f"configs/forge/views/{VIEW}.json").write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")

for parent, diag in ARMS.items():
    for cid, control in ((diag, parent), (parent, diag)):
        sid = f"dig2-ring16-cap-d1-{cid}-study"
        study = {
            "campaign": {"budget_seconds": 7200, "candidate_budget_seconds": 1500, "id": CAMPAIGN},
            "candidate": cid,
            "competing_explanation": ("The recipe-owned prior refactor, not the network anneal cap, may explain why "
                                      "K3P/KA2 no longer pass at 6400 updates."),
            "control": {"candidate_id": control, "task_map": {}},
            "falsifier": {"metric": "component_min_eigen_ratio", "op": "<", "phase": "final",
                          "task_id": "ring16_acquisition_6400", "threshold": 0.15},
            "hypothesis": ("With the G/D LR anneal over the full 6400-update cap, the recipe keeps local spread "
                           "(min eigen ratio >= 0.15) and passes ring16_acquisition_6400."
                           if cid == diag else "Matched control: unchanged selected recipe on the D1 source."),
            "id": sid, "max_rounds": 1,
            "prediction": {"metric": "component_min_eigen_ratio", "op": ">=", "phase": "final",
                           "task_id": "ring16_acquisition_6400", "threshold": 0.15},
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
