"""Keep architecture trials within one formulation and fixed training recipe.

This is a reporting layer. It never changes a numerical gate or trains a model.
"""
from copy import deepcopy

from .protocol import test_verdict


def axes(spec, runner):
    if runner not in ("image", "vector", "stress"):
        raise ValueError("legacy hosts require their explicit candidate card")
    image = runner == "image"
    arm = spec.get("gradient_penalty" if image else "reg_arm", "b_cap")
    formulation = dict(
        loss=spec.get("loss_type", "logistic"), mode=spec.get("gan_mode", "rp"),
        penalty=arm, coefficient=spec["penalty_coeff" if image else "reg_coeff"],
        kappa=spec.get("kappa" if image else "reg_kappa", 1.25) if arm in ("b_cap", "g_interp_cap") else None,
        particle_l2=spec.get("particle_l2", 0.),
        prior_regularization=spec["prior_weight" if image else "prior_reg"],
        prior_kind=spec.get("prior_kind", "particles") if spec.get("prior_learnable", True) else "frozen_particles",
    )
    lr = spec["lr_g" if image else "lr"]
    training = dict(
        optimizer="Adam", betas=list(spec["adam_betas" if image else "betas"]),
        lr_g=lr, lr_d=spec["lr_d"] if image else lr * spec["d_lr_mult"],
        lr_prior=lr * spec.get("prior_lr_multiplier" if image else "prior_lr_mult", 1.),
        schedule=spec.get("schedule", "cosine"),
        d_every=spec.get("d_every", 1), g_every=spec.get("g_every", 1),
    )
    if image:
        kind = spec["architecture"]
        architecture = dict(
            generator=dict(kind=kind if kind in ("residual_upsample", "uniform_generator") else "transpose",
                           width=spec["width"], z_dim=spec["z_dim"]),
            discriminator=dict(kind="mean_only" if kind == "mean_discriminator" else "convolutional",
                               width=max(12, spec["width"])),
        )
    else:
        architecture = dict(
            generator=dict(kind="mlp", width=spec["hidden"], layers=spec["layers"], z_dim=spec["z_dim"]),
            discriminator=dict(kind="mlp", width=spec.get("d_hidden", spec["hidden"]),
                               layers=spec.get("d_layers", spec["layers"]), fourier=spec["fourier"]),
        )
    resources = dict(steps=spec["steps"], particles=spec["particles"], batch=spec["batch_size" if image else "batch"])
    target_keys = ("name", "kind", "means", "covariances", "masses", "identifiable", "pattern", "modes",
                   "noise_std", "turns", "radius_min", "radius_max", "noise", "scale_start", "scale_end",
                   "scale_ramp_end", "thresholds")
    target = {key: deepcopy(spec[key]) for key in target_keys if key in spec}
    return dict(formulation=formulation, training=training, architecture=architecture, resources=resources, target=target)


def architecture_cell(variants, runner):
    """One problem earns at most one pass, across allowed architecture variants.

    Preserve every architecture result. A supported architecture need not make
    every tested architecture pass. Different formulations, optimizer settings,
    targets or budgets must never be silently combined into this cell.
    """
    if not variants:
        return dict(status="MISSING", supported=False, all_tested_architectures_pass=False, trials=[])
    reference = axes(variants[0]["spec"], runner)
    trials = []
    for variant in variants:
        actual = axes(variant["spec"], runner)
        for key in ("formulation", "training", "resources", "target"):
            if actual[key] != reference[key]:
                raise ValueError(f"architecture-only grouping changed {key}")
        verdict = test_verdict(variant["spec"], variant["result"])
        trials.append(dict(label=variant["label"], architecture=actual["architecture"],
                           verdict=verdict, artifact=variant.get("artifact")))
    supported = any(t["verdict"]["passed"] for t in trials)
    return dict(status="PASS" if supported else "FAIL", supported=supported,
                all_tested_architectures_pass=all(t["verdict"]["passed"] for t in trials),
                formulation=reference["formulation"], training=reference["training"],
                resources=reference["resources"], trials=trials)
