"""Prepare eight global K3P rate/penalty recipes; no training or enqueue."""
from prepare_common import prepare


if __name__ == "__main__":
    prepare(study="k3p-global-tier1-v3", base="k3p-global-input-noise-v1",
        grid={"lr": [.006375], "d_lr_mult": [.5, 1.0],
              "prior_lr_mult": [.5, 1.0], "reg_coeff": [.5, 1.0]},
        count=8, plans_name="plans.json", scope="bounded_global_k3p_ring_quality_controls",
        hypothesis=(
            "Within the existing input-noisy/output-clean full-horizon K3P "
            "recipe, slower critic rates and faster learned-prior transport "
            "may reduce ring within-mode covariance error. Positive critic "
            "coefficients .5/1 test this tradeoff while each complete global "
            "recipe must pass all five ordinary Tier 1 tasks."),
        rationale=(
            "The previous coefficient1/D1.5/nominal-prior-LR.0012 candidates "
            "passed the first three tasks but failed ring covariance and "
            "high-quality fraction. The historical 4/5 incumbent instead "
            "used D1/prior1 at LR.006375 (nominal prior LR.006375). Its "
            "1600-step network cap is inactive on the 400-update ring, so "
            "cap removal cannot explain that ring difference. Output noise "
            "and source cohorts also differ; this is no single-cause proof. "
            "Saved scored outputs show broad component cores as well as tails: "
            "recentering alone barely changes quality, and the four-sigma "
            "core covariance error still exceeds .85. "
            "The finite grid holds base LR.006375, existing input noise.5, "
            "output noise0, full schedule, moments and all mechanism flags "
            "fixed. D.5/1 and prior.5/1 test declared role balance; coefficients "
            ".5/1 remain positive. Words remain an ordinary required task; "
            "the historical word win supplies no transferable qualification."),
        evidence_paths=[
            "reports/forge/k3p-global-tier1-v2/input-noise/summary.json",
            "reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json",
            "reports/forge/k3p-global-tier1-v3/ring-analysis.json"],
        prediction={"task_id": "ring16_acquisition", "metric": "component_covariance_error",
            "op": "<=", "threshold": .85, "phase": "final"},
        falsifier={"task_id": "ring16_acquisition", "metric": "component_covariance_error",
            "op": ">", "threshold": .85, "phase": "final"},
        competing_explanation=(
            "Faster prior transport or reduced critic regularization can still "
            "lose movement, mass balance, ring quality or the word bijection. "
            "A final covariance bound alone does not establish the complete "
            "sustained ring gate. If an earlier prerequisite fails, the ring "
            "prediction is unmeasured; no word witness may fill that candidate's "
            "UNKNOWN cells. A finite negative grid does not prove impossibility."))
