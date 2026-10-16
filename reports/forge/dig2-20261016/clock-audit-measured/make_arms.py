"""Write dig2 clock-audit-measured idea cards, view and studies (reproduction source).

Run from the repo root: PYTHONPATH=. python reports/forge/dig2-20261016/clock-audit-measured/make_arms.py
"""
import json
from pathlib import Path

from experiments.forge.api import resolve_public_recipe
from experiments.forge.clockfree import v2_declarations

RATE_6400 = 1 / 6400  # internal counter reaches p=1 at update 6,400 (horizon encoded in a recipe constant)
COMMON = {"d_guard_min_steps": 0, "network_lr_horizon_cap": None}
ARMS = {
    "step": dict(
        ov={},
        cf=["Matched step-schedule control: shipped cosine LR / input-noise / output-noise schedules over the "
            "task-owned 6,400-update horizon (network LR horizon cap removed so G/D anneal over the full run)",
            "Critic spike guard active from the first update (d_guard_min_steps=0), as in every arm"],
        mr="Positive ring16 control at 6,400 updates; reads the step label and horizon, so it should fail the "
           "measured clock audit."),
    "const": dict(
        ov={"lr_floor": 1.0, "network_lr_floor": 1.0, "input_noise_std": 0.0, "output_noise_warmup": 0.0},
        cf=["Constant learning rates (no cosine anneal), no input-noise anneal, constant output noise",
            "Critic spike guard active from the first update (d_guard_min_steps=0)"],
        mr="Clock-free non-annealing control: separates the effect of annealing from the longer cap."),
    "count6400": dict(
        ov={"progress_schedule": "updates", "progress_rate": RATE_6400},
        cf=["Schedules driven by an internal progress counter advancing 1/6400 per update (no gate), "
            "replacing completed_steps/total_steps; same shapes as the step control",
            "Critic spike guard active from the first update (d_guard_min_steps=0)"],
        mr="Internal-counter (D-style) anneal over a 6,400-update duration: reads no step label or horizon, "
           "but its duration is a recipe constant chosen to equal this cap."),
    "gated": dict(
        ov={"progress_schedule": "critic_order", "progress_rate": 0.001, "progress_target": 0.75,
            "progress_ema": 0.98},
        cf=["Schedules driven by critic-order progress (round-1 C hyperparameters unchanged: at most 0.001 per "
            "update, zero when the critic orders 75% of pairs)",
            "Critic spike guard active from the first update (d_guard_min_steps=0)"],
        mr="State-driven anneal with no fixed duration: the horizon-free candidate under a strict counter policy."),
}
FAMILIES = {"k3p": {"critic_formulation": "k3p", "name": "k3p"},
            "ka2": {"critic_formulation": "ka2", "name": "ka2"}}
VIEW = "dig2-clock-audit-measured-v1"
CAMPAIGN = "dig2-clock-audit-measured-v1"

template = json.loads(Path("configs/forge/ideas/k3p-hc-progress-v1.json").read_text())
cards = []
for family, base in FAMILIES.items():
    for arm, spec in ARMS.items():
        cid = f"{family}-cad-{arm}-v1"
        card = dict(template, id=cid, parent=family, prior_art=[family], changed_factors=spec["cf"],
                    mechanism_rationale=spec["mr"], recipe_overrides={**base, **COMMON, **spec["ov"]})
        Path(f"configs/forge/ideas/{cid}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
        print(cid, v2_declarations(resolve_public_recipe(card).to_dict(), {}))
        cards.append((family, arm, cid))

view = {
    "assignments": [
        {"importance": "diagnostic", "order": 0, "qualification_tier": 1, "task": "ring16_acquisition_6400-clockdig"},
        {"importance": "diagnostic", "order": 1, "qualification_tier": 1, "task": "clockfree_audit_measurement_v1"},
        {"importance": "diagnostic", "order": 2, "qualification_tier": 1, "task": "clockfree_audit_measured_v2-clockdig"},
    ],
    "calibration": {"adoption_blocker": "Research diagnostic: a scoped 6,400-update ring16 variant and a PROPOSED "
                    "measured clock audit revision; counted tasks, gates and results are unchanged.",
                    "status": "provisional"},
    "eligibility": {}, "evidence_scope": "research_diagnostic", "goal": "discriminator_stability", "id": VIEW,
    "policy_change_reason": "Can an annealing family pass ring16 at a 6,400 cap and a measured clock audit? "
                            "Matched step / constant / internal-counter / critic-gated arms for K3P and KA2.",
    "ranking": {"compare_compatible_cohorts": True, "cost_separate": True,
                "policy": "qualified_tier_only_with_raw_metrics"},
    "revision": 1, "schema_version": 1,
}
Path(f"configs/forge/views/{VIEW}.json").write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")

for family, arm, cid in cards:
    control = f"{family}-cad-step-v1" if arm != "step" else f"{family}-cad-const-v1"
    sid = f"dig2-cad-{cid}-study"
    study = {
        "campaign": {"budget_seconds": 9600, "candidate_budget_seconds": 1200, "id": CAMPAIGN},
        "candidate": cid,
        "competing_explanation": ("At 6,400 updates the longer training at high rate, not annealing, may be what "
                                  "passes ring16; the constant arm tests this."),
        "control": {"candidate_id": control, "task_map": {}},
        "falsifier": {"metric": "component_covariance_error", "op": ">", "phase": "final",
                      "task_id": "ring16_acquisition_6400-clockdig", "threshold": 0.85},
        "hypothesis": f"{arm} arm of {family}: " + ARMS[arm]["mr"],
        "id": sid, "max_rounds": 1,
        "prediction": {"metric": "component_covariance_error", "op": "<=", "phase": "final",
                       "task_id": "ring16_acquisition_6400-clockdig", "threshold": 0.85},
        "prior_evidence": [{"identity": {"record_id": "ring16-schedule-20261011-readout"},
                            "path": "reports/forge/dig-20261015/horizon-clock-motivation.json",
                            "selector": [], "use": "motivation_only"}],
        "schema_version": 1,
        "scope": {"cuda_model": "NVIDIA RTX A6000", "execution_backend": "cuda", "through_tier": 1, "view": VIEW},
        "status": "ready",
        "terminal_rules": {"falsified": "stop_revision", "incomplete": "request_missing_evidence",
                           "inconclusive": "stop_and_readout", "prediction_observed": "review_saved_diagnostics"},
    }
    Path(f"configs/forge/studies/{sid}.json").write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    print(sid)
