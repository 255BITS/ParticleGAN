"""Prepare four global K3P direct-moment recipes; no training or enqueue."""
from prepare_common import prepare


if __name__ == "__main__":
    prepare(study="k3p-global-direct-moments-tier1-v1", base="k3p-global-repair-v1",
        grid={"coupled_rates": [{"lr": lr, "prior_lr_mult": .0012 / lr}
                               for lr in (.0006, .0012, .002125, .00425)],
              "direct_particle_betas": [[0.0, .999]]},
        count=4, plans_name="plans-direct-moments.json",
        scope="bounded_global_k3p_existing_direct_moment_controls",
        hypothesis=(
            "The existing direct-particle beta2=.9 response imposes a finite "
            "80-update displacement upper bound below .3 at base LR.0006. "
            "Changing its already-public positive beta2 to .999 loosens that "
            "bound and may permit one global clean K3P recipe, retaining "
            "coefficient170 and nominal latent-prior LR.0012, to pass all "
            "five ordinary Tier 1 tasks."),
        rationale=(
            "The historical coefficient170 word recipe passed all24 checks at "
            "LR.0006/D1.5/prior2 with clean outputs and a full horizon. Its "
            "global transfer failed movement; higher base rates with the "
            "original direct beta2 also failed. This finite grid retains the "
            "same penalty kernel/coefficient170, D1.5, zero training noise, "
            "full schedule and other mechanisms. Global base LR varies over "
            ".0006/.0012/.002125/.00425, paired to preserve nominal latent "
            "prior LR.0012. The fixed direct betas[0,.999] use an existing "
            "public optimizer setting consumed only by direct-coordinate "
            "groups. No task-ID override, new response formula or stabilization "
            "mechanism is introduced. LR.0006 retains the old word-active "
            "recipe, but its historical pass cannot fill new ordinary cells. "
            "A looser theoretical movement bound does not predict actual "
            "acceleration; persistent second moments can slow decaying gradients. "
            "Low shared rates and strong penalty may still fail ring400."),
        evidence_paths=[
            "reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json",
            "reports/forge/family-wide-word-repairs/initial/k3p-global-repair-v1.json",
            "reports/forge/k3p-global-tier1-v3/direct-rate-analysis.json"],
        prediction={"task_id": "two_pole", "metric": "mean_abs",
            "op": ">=", "threshold": .3, "phase": "final"},
        falsifier={"task_id": "two_pole", "metric": "mean_abs",
            "op": "<", "threshold": .3, "phase": "final"},
        competing_explanation=(
            "The upper bound is necessary eligibility, not a predicted "
            "trajectory: changing beta2 can retain an early large denominator "
            "and reduce movement when gradients decay. A movement pass does "
            "not establish low-rate ring acquisition or word transfer. All "
            "five sustained ordinary gates must pass from the same candidate; "
            "no historical word receipt or different task recipe supplies credit."))
