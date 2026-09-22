"""Predeclared development stresses; tiers never change after seeing results.

Reference failures are evidence about solvability, not reasons to promote a
diagnostic to a blocker or silently discard a ranking task. The historical
nine behavioral tasks remain the only eligibility blockers.
"""
from copy import deepcopy
import math


PROTOCOL = {
    "version": "transfer-stress-v2", "seed": 0, "observations": 24,
    "minimum_stable_checks": 5,
    "references": ["fixed_cosine", "fixed_constant"],
    "reference_attempts_per_task": 2,
    "selection": "Live numerical behavior; EMA separate. New tasks rank or diagnose, never block eligibility.",
    "unsolved_reference": "NOT DEMONSTRATED; preserve the predeclared tier and report all attempts.",
    "correction": "v2 adds minimum normalized component covariance eigenvalue >= .15; v1 source and executed results are preserved, and recorded curves are re-scored explicitly.",
}

THRESHOLDS = [
    ["sw1_normalized", "<=", .18], ["mass_tv", "<=", .15],
    ["hq", ">=", .85], ["component_covariance_error", "<=", .85],
    ["component_min_eigen_ratio", ">=", .15],
]

_BASE = {
    "kind": "gaussian_mixture", "identifiable": True,
    "means": [[3 * math.cos(i * math.pi / 4), 3 * math.sin(i * math.pi / 4)] for i in range(8)],
    "covariances": [[[.12 ** 2, 0.], [0., .12 ** 2]] for _ in range(8)],
    "masses": [.125] * 8,
    "hidden": 64, "layers": 2, "d_hidden": 64, "d_layers": 2,
    "fourier": 2, "z_dim": 4, "particles": 256, "batch": 128,
    "lr": .001, "d_lr_mult": 1.5, "prior_lr_mult": 10.,
    "prior_reg": .05, "betas": [0., .999], "ema_decay": .995,
    "reg_arm": "b_cap", "reg_coeff": 3., "reg_kappa": 1.25,
    "steps": 1200, "thresholds": THRESHOLDS,
}


def _spec(name, family, importance_reason, limitations, *, tier="ranking", **changes):
    return {**deepcopy(_BASE), "name": name, "family": family,
            "split": "development", "tier": tier,
            "importance_reason": importance_reason, "limitations": limitations,
            **deepcopy(changes)}


TASKS = [
    _spec("stress_fast_critic", "learning_rate_imbalance",
          "A critic learning twice as fast is a realistic optimizer imbalance; maintaining distribution quality matters for routine tuning.",
          "Changes only the critic learning rate; it does not test additional discriminator updates or minibatch changes.",
          d_lr_mult=3.),
    _spec("stress_slow_critic", "learning_rate_imbalance",
          "A critic learning half as fast tests whether feedback remains useful when generator transport can outrun density estimation.",
          "The critic retains adequate architecture; failure within the fixed budget is not proof the target is unsolvable.",
          d_lr_mult=.75),
    _spec("stress_small_batch", "minibatch_noise",
          "Batch 64 is an ordinary memory-limited setting; feedback should tolerate its noisier gradient observations.",
          "The update count stays fixed, so this arm sees fewer training examples; report that compute/data difference rather than attributing everything to noise.",
          batch=64),
    _spec("stress_large_critic", "capacity_imbalance",
          "Doubling critic width is a plausible architecture choice that tests controller transfer across gradient scale and capacity.",
          "A wider critic costs more per update; use wall time alongside update-normalized convergence.",
          d_hidden=128),
    _spec("stress_long_horizon", "training_horizon",
          "A doubled training horizon checks persistence of convergence and delayed collapse beyond the usual training budget.",
          "There are still only 24 fixed observations, now 100 steps apart; stability between observations is not certified.",
          steps=2400),
    _spec("stress_r1_r2", "gradient_penalty_formulation",
          "A supported R1+R2 penalty tests whether a controller transfers across common discriminator regularization objectives.",
          "Coefficient .1 is a predeclared reference alternative, not an equal-strength equivalence to the cap penalty or a search over coefficients.",
          reg_arm="a_r1r2", reg_coeff=.1),
    _spec("stress_weak_critic", "architecture_limit",
          "A deliberately narrow, shallow critic without Fourier features diagnoses an architecture bottleneck; this artificial weakness is not a selection requirement.",
          "Failure can arise because the critic cannot resolve narrow target modes. It supplies diagnostic evidence only, even if another method succeeds.",
          tier="diagnostic", d_hidden=16, d_layers=1, fourier=0),
    _spec("stress_overlapping_data", "data_ambiguity",
          "Broad overlapping components diagnose sensitivity to ambiguous mixture labels. Distribution fit matters here; component reconstruction is not an appropriate requirement.",
          "This intentionally blurred data is not representative of the separated-mode objective. Only global sliced distance is scored; HQ and component-mass/covariance claims are omitted.",
          tier="diagnostic", identifiable=False,
          covariances=[[[.6 ** 2, 0.], [0., .6 ** 2]] for _ in range(8)],
          thresholds=[["sw1_normalized", "<=", .18]]),
]

RESERVED_TASKS = [
    _spec("reserved_alternating_critic_updates", "update_cadence",
          "A previously unseen discriminator update cadence tests transfer when opponent feedback arrives less often, rather than merely at a different learning rate.",
          "Never evaluated in development. The target stays static; D updates every second outer step and G every step. Report both actual update counts and wall time because the work per outer step changes.",
          split="reserved", steps=2400, d_every=2, g_every=1),
]


def run_episode(spec, policy, *, ablation="none", fixed=False, allow_reserved=False):
    """Delegate episodes; the parent unlocks reserved dynamics after freezing."""
    if spec.get("split") != "development" and not allow_reserved:
        raise ValueError("reserved dynamics must remain unevaluated during development")
    from .vector_tasks import run_episode as run_vector_episode
    return run_vector_episode(spec, policy, ablation=ablation, fixed=fixed,
                              allow_reserved=allow_reserved)
