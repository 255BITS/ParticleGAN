"""Prospective policy-selected deployed-output measurement; no evaluator or learner.

The caller supplies the existing owner and a dedicated evaluation generator.
No prior, network, Recipe, checkpoint, or policy selector is replaced here.
"""
import math
import re


PROTOCOL_ID = "canonical-policy-selected-served-output-v1"
SAMPLING_LAW = "uniform_independent_public_served_sample"


def measurement_contract():
    """Declare one observation law, with no clean/EMA/score selection branch."""
    return {
        "schema": "pg_canonical_serving_measurement_contract_v1",
        "protocol_id": PROTOCOL_ID,
        "sampling_law": SAMPLING_LAW,
        "owner_api": "GANTrainer.served_model -> UpdatePolicy.served_model",
        "draw_api": "ServedModel.sample(n, generator=dedicated_eval_rng, output_noise=True)",
        "selection": "intrinsic public policy selector; fast or averaged as selected",
        "latent_perturbation": "unchanged public DV12 and selected backend",
        "output_noise": True,
        "output_sigma": "current policy output_sigma, including its learned floor law",
        "prior_and_network": "existing caller-owned modules; no override",
        "rng": "caller-owned dedicated evaluation stream, separate from every owner stream",
        "purity": "equal source-bound semantic state fingerprints before and after",
        "unsupported_sampling_laws": ["conditional_context", "routed_context", "enumerated_image_centers"],
        "task_metrics_thresholds_horizons": "unchanged by this helper; separately declared by the caller",
        "full_protocol_credit_from_one_observation": False,
    }


def _fingerprint(reader, owner):
    value = reader(owner)
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("semantic state reader must return an exact SHA256")
    return value


def measure_served_samples(owner, n, *, dedicated_eval_rng, state_fingerprint,
                           expected_step, sampling_law=SAMPLING_LAW, extra_owner_rngs=()):
    """Return actual samples and a compact receipt from one public selection/draw.

``state_fingerprint(owner)`` must be source-bound by the caller and cover live
models, optimizer state, policy/controllers/averages, global and owner RNGs,
callback resources and data cursors; it excludes only this dedicated eval RNG.
It must read semantic state, not tensor mutation-version counters. The public
snapshot may temporarily release/reapply a compatibility serving swap.

No model is constructed by the helper. At runtime the public serving API
copies the already-owned models into independent frozen inference modules.
This initial law cannot stand in for an enumerated image or contextual task.
"""
    if sampling_law != SAMPLING_LAW:
        raise ValueError("contextual or enumerated observation requires its own declared serving law")
    if type(n) is not int or n <= 0:
        raise ValueError("sample count must be a positive integer")
    if type(expected_step) is not int or expected_step < 0:
        raise ValueError("the declared observation clock must be a nonnegative integer")
    if not callable(state_fingerprint):
        raise TypeError("a source-bound semantic state reader is required")
    if dedicated_eval_rng is None:
        raise ValueError("a dedicated evaluation RNG is required")
    if type(extra_owner_rngs) not in (tuple, list):
        raise TypeError("extra owner/data RNGs must be an explicit sequence")
    policy = getattr(owner, "policy", owner)
    recipe = policy.recipe
    if (policy.row_semantics != "independent" or recipe.model != "gan"
            or recipe.conditioning != "scalar" or recipe.encoder_mode != "none"
            or getattr(policy, "encoder", None) is not None
            or getattr(policy, "router", None) is not None
            or getattr(policy, "generation", None) is not None):
        raise ValueError("this initial API supports unconditional direct-module GAN serving only")
    if policy._phase != "ready" or type(policy.completed_steps) is not int or policy.completed_steps != expected_step:
        raise ValueError("measurement requires the declared completed update boundary")
    for value in (owner, policy):
        for name in getattr(value, "_STREAMS", ()):
            if dedicated_eval_rng is getattr(value, name, None):
                raise ValueError("dedicated evaluation RNG must not alias any owner stream")
    if any(dedicated_eval_rng is stream for stream in extra_owner_rngs):
        raise ValueError("dedicated evaluation RNG must not alias caller data/training streams")
    before = _fingerprint(state_fingerprint, owner)
    try:
        served = owner.served_model()
        if (served.row_semantics != "independent" or served.routing is not None
                or served.encoder is not None or served.router is not None
                or served.generation is not None):
            raise ValueError("served snapshot must implement the same independent direct-module law")
        if served.source not in {"fast", "averaged"}:
            raise ValueError("unknown public served selection")
        if type(served.completed_steps) is not int or served.completed_steps != expected_step:
            raise ValueError("served snapshot clock differs from the declared measurement")
        if type(served.output_sigma) not in (int, float) or not math.isfinite(served.output_sigma) or served.output_sigma < 0:
            raise ValueError("served output-noise scale must be finite and nonnegative")
        if served.table is policy.table:
            raise ValueError("served table must be independent of the live owner")
        for role, owner_attribute in (("generator", "G"), ("critic", "D"), ("prior", "prior")):
            live = getattr(policy, owner_attribute, None)
            if live is not None and getattr(served, role, None) is live:
                raise ValueError("served modules must be independent of live training modules")
        samples = served.sample(n, generator=dedicated_eval_rng, output_noise=True)
        shape = tuple(int(dimension) for dimension in samples.shape)
        if not shape or shape[0] != n:
            raise ValueError("public serving output has the wrong batch shape")
    finally:
        after = _fingerprint(state_fingerprint, owner)
        if (after != before or policy._phase != "ready"
                or type(policy.completed_steps) is not int or policy.completed_steps != expected_step):
            raise RuntimeError("serving measurement changed semantic training/owner state")
    return samples, {
        "schema": "pg_canonical_serving_measurement_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "sampling_law": SAMPLING_LAW,
        "completed_steps": expected_step,
        "selected_source": served.source,
        "selection_origin": "intrinsic public policy; no metric-based choice",
        "output_noise": True,
        "output_sigma": float(served.output_sigma),
        "output_noise_mode": recipe.output_noise_mode,
        "latent_perturbation": "unchanged public DV12 and selected backend",
        "sample_count": n,
        "output_shape": list(shape),
        "dedicated_eval_rng_distinct_from_owner": True,
        "semantic_state_sha256_before": before,
        "semantic_state_sha256_after": after,
        "semantic_state_unchanged": True,
        "numeric_grade": None,
        "full_protocol_credit": False,
    }
