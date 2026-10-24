"""Public, explicit capabilities of the supplied prior and its update rule.

Capabilities describe the actual module, not a recipe label. In particular,
standardizing a mixture's reads couples all rows and is not a sparse A2 table.
"""
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior


def prior_capabilities(prior):
    """Return JSON-compatible sampling/update facts for built-in priors.

    Unknown modules remain usable by component APIs, but do not acquire an A2
    capability merely by having an attribute called ``z``.
    """
    base = getattr(prior, "base", prior)
    known = type(base) in (ParticlePrior, MoGParticlePrior)
    mog = type(base) is MoGParticlePrior
    learned = known and hasattr(base, "z") and base.z.requires_grad
    standardize = mog and getattr(base, "standardize", False)
    base_row_local = bool(known and not standardize)
    manipulator_row_local = True
    if hasattr(prior, "manipulators"):
        manipulator_row_local = all(getattr(m, "row_local_gradients", True) for m in prior.manipulators)
    row_local = bool(base_row_local and manipulator_row_local)
    a2_eligible = bool(learned and row_local)
    return {
        "kind": "mog" if mog else "particle_cloud" if known else
                "gaussian" if type(base) is GaussianPrior else "unknown",
        "learned_locations": bool(learned),
        "learned_width": False if known else None,
        "mixture_weights": "uniform" if known else None,
        "sigma": float(base.sigma) if mog else 0.0 if known else None,
        "standardize": bool(standardize) if known else None,
        "row_local_gradients": row_local,
        "a2_eligible": a2_eligible,
    }


def prior_mechanisms(prior, *, latent_damping_max_rate, prior_beta1):
    """Resolve A2 eligibility and report why a requested hook cannot run."""
    capabilities = prior_capabilities(prior)
    requested = latent_damping_max_rate > 0
    eligible = capabilities["a2_eligible"] and prior_beta1 == 0
    if not requested:
        reason = "disabled explicitly by latent_damping_max_rate=0"
    elif not capabilities["learned_locations"]:
        reason = "A2 requires a supported learned location table"
    elif not capabilities["row_local_gradients"]:
        if capabilities.get("standardize"):
            reason = "standardized MoG reads couple rows; sparse A2 is unsupported"
        else:
            reason = "prior or manipulator couples rows; sparse A2 is unsupported"
    elif prior_beta1 != 0:
        reason = "A2 requires prior beta1=0"
    else:
        reason = "row-local learned locations with prior beta1=0"
    return {"prior": capabilities, "a2": {"requested": requested,
            "supported": bool(eligible), "enabled": bool(requested and eligible),
            "reason": reason}}
