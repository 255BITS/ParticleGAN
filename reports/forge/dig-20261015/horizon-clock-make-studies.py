"""Write the horizon-clock studies (one per arm, one shared campaign). Reproduction source."""
import json
from pathlib import Path

ARMS = {
    "k3p-hc-step-v1": "Step-schedule control: shipped K3P; on ring16 the task-owned 400-update schedule horizon finishes annealing by update 400.",
    "k3p-hc-constant-v1": "Constant-rate clock-free K3P control.",
    "k3p-hc-progress-v1": "Candidate: critic-order progress drives K3P's annealing shape without the step label or horizon.",
    "k3p-hc-progress-ungated-v1": "Diagnostic: internal-counter progress (anneal inside the cap without a gate).",
}
for cid, hypothesis in ARMS.items():
    study = {
        "campaign": {"budget_seconds": 7200, "candidate_budget_seconds": 1800, "id": "dig-horizon-clock-v1"},
        "candidate": cid,
        "competing_explanation": ("K3P may need far more than 1,600 updates of training at high rate (the 6,400 full "
                                  "schedule still failed at its 1,600/3,200/4,800 cuts), so no schedule shape inside "
                                  "the cap can reproduce the long-horizon pass."),
        "control": {"candidate_id": "k3p-hc-step-v1" if cid != "k3p-hc-step-v1" else "k3p-hc-constant-v1", "task_map": {}},
        "falsifier": {"metric": "component_covariance_error", "op": ">", "phase": "final",
                      "task_id": "ring16_acquisition", "threshold": 0.85},
        "hypothesis": hypothesis,
        "id": f"dig-horizon-clock-{cid}-study",
        "max_rounds": 1,
        "prediction": {"metric": "component_covariance_error", "op": "<=", "phase": "final",
                       "task_id": "ring16_acquisition", "threshold": 0.85},
        "prior_evidence": [{"identity": {"record_id": "ring16-schedule-20261011-readout"},
                            "path": "reports/forge/dig-20261015/horizon-clock-motivation.json",
                            "selector": [], "use": "motivation_only"}],
        "schema_version": 1,
        "scope": {"cuda_model": "NVIDIA RTX A6000", "execution_backend": "cuda", "through_tier": 1,
                  "view": "dig-horizon-clock-diagnostic-v1"},
        "status": "ready",
        "terminal_rules": {"falsified": "stop_revision", "incomplete": "request_missing_evidence",
                           "inconclusive": "stop_and_readout", "prediction_observed": "review_saved_diagnostics"},
    }
    Path(f"configs/forge/studies/dig-horizon-clock-{cid}-study.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n")
    print(f"dig-horizon-clock-{cid}-study")
