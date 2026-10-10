"""Default projections for Recipe additions; archived packets keep their identity."""

SEARCH_RECIPE_DEFAULTS = {
    "critic_step_mode": "none",
    "constraint_geometry_mode": "none",
    "hydraulic_travel_fraction": 0.0,
    "hydraulic_travel_radius": "real_spacing",
    "kinetic_transport_weight": 0.0,
    "kinetic_transport_local_weight": 0.0,
    "kinetic_transport_projections": 32,
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
    for name, default in SEARCH_RECIPE_DEFAULTS.items():
        value = result.get(name, default)
        if name == "loss_labels" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if value == default:
            result.pop(name, None)
    return result
