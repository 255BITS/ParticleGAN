"""Default projections for Recipe additions; archived packets keep their identity."""

SEARCH_RECIPE_DEFAULTS = {
    "prior_update": "learned", "prior_regularizer": "vicreg",
    "prior_coulomb_softening": .1, "prior_coulomb_confinement": .35, "prior_coulomb_neighbors": 8,
    "prior_reg_target_std": 1.0, "prior_reg_eps": 1e-4, "prior_l2": 0.0,
    "prior_proximal_strength": 0.0,
    "prior_sigreg_projections": 32, "prior_sigreg_knots": 17,
    "critic_step_mode": "none",
    "constraint_geometry_mode": "none",
    "kinetic_transport_weight": 0.0,
    "kinetic_transport_local_weight": 0.0,
    "kinetic_transport_projections": 32,
    "energy_distance_weight": 0.0,
    "output_mmd_weight": 0.0,
    "output_mmd_bandwidths": (.0625, .125, .25, .5, 1., 2., 4.),
    "optimizer_smoothing": 0.0,
    "optimizer_convolution": "none",
    "optimizer_svd_backend": "native",
    "d_betas": None, "d_eps": None, "prior_eps": None,
    "loss_labels": (0.0, 1.0, 1.0), "adam_variant": "pytorch",
    "lr_schedule": "cosine", "lr_decay_rate": 0.96, "lr_decay_steps": 50_000,
    "lr_decay_staircase": False,
}


def without_default_additions(values):
    result = dict(values)
    if "output_mmd_weight" in result:
        from .kernel_mmd import _finite_number
        weight = result["output_mmd_weight"]
        if not _finite_number(weight) or weight < 0:
            raise ValueError("invalid output_mmd_weight recipe packet")
        if weight and "output_mmd_bandwidths" not in result:
            raise ValueError("active output MMD recipe packet requires explicit bandwidths")
    if "output_mmd_bandwidths" in result:
        from .kernel_mmd import validate_bandwidths
        result["output_mmd_bandwidths"] = validate_bandwidths(result["output_mmd_bandwidths"])
    mmd_active = bool(result.get("output_mmd_weight", 0))
    for name, default in SEARCH_RECIPE_DEFAULTS.items():
        if name == "output_mmd_bandwidths" and mmd_active:
            continue
        value = result.get(name, default)
        if name == "loss_labels" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if value == default:
            result.pop(name, None)
    return result
