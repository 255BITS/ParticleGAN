"""One-sided sampled-real coverage correction for an independent particle prior.

This is an explicit coverage objective addition, not a pure GAN optimizer fix.
The clean generator must map each prior row independently. No optimizer state
is advanced here; the caller supplies its already-updated Adam diagonal.
"""
import torch


def coverage_loss(real, points):
    return torch.cdist(real, points).square().min(dim=1).values.mean()


def prior_adam_metric(optimizer, prior_z):
    """The diagonal P=lr/(sqrt(vhat)+eps) after one ordinary Adam update."""
    group = next((g for g in optimizer.param_groups if any(p is prior_z for p in g["params"])), None)
    if group is None:
        raise ValueError("prior parameter is absent from generator optimizer")
    state = optimizer.state[prior_z]
    beta2 = group["betas"][1]
    step = float(state["step"])
    denominator = (state["exp_avg_sq"] / (1 - beta2 ** step)).sqrt() + group["eps"]
    metric = group["lr"] / denominator
    if not torch.isfinite(metric).all() or not bool((metric > 0).all()):
        raise FloatingPointError("invalid prior Adam metric")
    return metric.detach()


def centroid_pullback(clean, prior_z, real, metric, *, max_halves=8, pinv_rtol=1e-6):
    """Project each Lloyd centroid move through its row-local G Jacobian.

    Real rows choose their nearest clean generated support, with first-index
    ties. Empty support cells stay fixed. For each nonempty cell, the latent
    correction is the minimum P^-1-norm solution to the linearized centroid
    displacement. Accept only a finite strict decrease in actual coverage
    loss after a whole-step or at most eight halvings; otherwise restore z.
    """
    if prior_z.ndim != 2 or real.ndim != 2 or metric.shape != prior_z.shape:
        raise ValueError("expected [particle,latent] and [batch,output] matrices")
    if not torch.isfinite(real).all() or not torch.isfinite(metric).all() or not bool((metric > 0).all()):
        raise FloatingPointError("nonfinite real batch or metric")
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        x = clean(base_z).detach()
        if x.ndim != 2 or x.shape[0] != prior_z.shape[0] or x.shape[1] != real.shape[1]:
            raise ValueError("clean generator must map prior rows to real-row shape")
        distance = torch.cdist(real, x)
        assignment = distance.argmin(dim=1)
        counts = torch.bincount(assignment, minlength=len(x))
        centroid = torch.stack([real[assignment == i].mean(dim=0) if counts[i] else x[i]
                                for i in range(len(x))])
        base_loss = distance.square().min(dim=1).values.mean()
    if not torch.isfinite(x).all() or not torch.isfinite(base_loss):
        raise FloatingPointError("invalid clean generated support")

    # The frozen mode-hold MLP maps rows independently. vmap/jacfwd avoids
    # differentiating through other particles or through the real assignment.
    one_row = lambda z: clean(z.unsqueeze(0)).squeeze(0)
    with torch.enable_grad():
        jacobian = torch.func.vmap(torch.func.jacfwd(one_row))(base_z).detach()
    if jacobian.shape != (len(x), x.shape[1], prior_z.shape[1]):
        raise RuntimeError("unexpected particle Jacobian shape")
    if not torch.isfinite(jacobian).all():
        raise FloatingPointError("nonfinite particle Jacobian")
    j = jacobian.double()
    p = metric.double()
    residual = (centroid - x).double()
    gram = torch.einsum("nmd,nd,nkd->nmk", j, p, j)
    singular = torch.linalg.svdvals(gram)
    numerical_rank = (singular > pinv_rtol * singular[:, :1]).sum(dim=1)
    inverse = torch.linalg.pinv(gram, rtol=pinv_rtol, atol=0.)
    correction = p * torch.einsum("nmd,nm->nd", j, torch.einsum("nmk,nk->nm", inverse, residual))
    correction = correction.to(prior_z.dtype)
    if not torch.isfinite(correction).all():
        raise FloatingPointError("nonfinite latent correction")
    correction[counts == 0] = 0
    linearized = torch.einsum("nmd,nd->nm", j, correction.double())
    residual_norm = (linearized - residual).norm(dim=1)
    tolerance = max(1e-12, 1e-8 * max(1., float(base_loss)))
    accepted = False
    after_loss = float(base_loss)
    alpha = 0.
    accepted_halvings = None
    trials = 0
    try:
        with torch.no_grad():
            for halves in range(max_halves + 1):
                trials += 1
                trial_alpha = 2. ** -halves
                prior_z.copy_(base_z + trial_alpha * correction)
                trial_x = clean(prior_z)
                if not torch.isfinite(trial_x).all():
                    continue
                trial_loss = coverage_loss(real, trial_x)
                if torch.isfinite(trial_loss) and float(trial_loss) < float(base_loss) - tolerance:
                    accepted = True
                    alpha = trial_alpha
                    accepted_halvings = halves
                    after_loss = float(trial_loss)
                    break
    finally:
        if not accepted:
            with torch.no_grad():
                prior_z.copy_(base_z)
    with torch.no_grad():
        output_after = clean(prior_z).detach()
    return {
        "accepted": accepted, "alpha": alpha, "coverage_before": float(base_loss),
        "coverage_after": after_loss, "coverage_tolerance": tolerance,
        "max_halves": max_halves, "accepted_halvings": accepted_halvings,
        "trial_evaluations": trials, "pinv_rtol": pinv_rtol,
        "numerical_rank": numerical_rank.tolist(),
        "linearized_output_residual_norm": residual_norm.tolist(),
        "assigned_counts": counts.tolist(),
        "empty_cells": int((counts == 0).sum()),
        "latent_correction_norm": float(correction.norm()),
        "max_particle_correction_norm": float(correction.norm(dim=1).max()),
        "actual_latent_displacement_norm": float((prior_z.detach() - base_z).norm()),
        "actual_output_displacement_norm": (output_after - x).norm(dim=1).tolist(),
        "output_before": x.tolist(),
        "output_after": output_after.tolist(),
    }
