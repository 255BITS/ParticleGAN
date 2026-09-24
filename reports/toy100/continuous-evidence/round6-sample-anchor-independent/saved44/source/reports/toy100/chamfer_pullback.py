"""Unit-mean bidirectional support correction for a row-independent generator.

This adds a sampled-data Chamfer objective; it is not an equilibrium-preserving
GAN optimizer. The caller supplies clean deterministic G, the current real
minibatch and the already-advanced prior Adam metric. No optimizer step, random
draw, target-center lookup, time schedule or coefficient is used here.
"""

import math

import torch


def chamfer_terms(real, points):
    """Return C=mean_real min_particle d² and Q=mean_particle min_real d²."""
    squared = torch.cdist(real, points).square()
    return squared.min(dim=1).values.mean(), squared.min(dim=0).values.mean()


def chamfer_targets(real, points):
    """Minimizer of C+Q with both nearest-neighbor assignments held fixed.

    For B real samples and N particles, t_j is
    (sum_{i assigned j} real_i/B + nearest_real_j/N)/(n_j/B + 1/N).
    Thus empty forward cells still receive a target from the current data.
    First-index ties follow torch.argmin. Accumulation is in float64.
    """
    if (real.ndim != 2 or points.ndim != 2 or not len(real) or not len(points)
            or real.shape[1] != points.shape[1]):
        raise ValueError("expected nonempty real and generated output matrices")
    if not torch.isfinite(real).all() or not torch.isfinite(points).all():
        raise FloatingPointError("nonfinite real data or generated support")
    distance = torch.cdist(real, points)
    assignment = distance.argmin(dim=1)
    nearest_real = distance.argmin(dim=0)
    counts = torch.bincount(assignment, minlength=len(points))
    sums = torch.zeros_like(points, dtype=torch.float64)
    sums.index_add_(0, assignment, real.double())
    weight = counts.double() / len(real) + 1. / len(points)
    target = (sums / len(real) + real[nearest_real].double() / len(points)) / weight[:, None]
    return target, counts, assignment, nearest_real


def chamfer_pullback(clean, prior_z, real, metric, *, max_halves=8, pinv_rtol=1e-6):
    """One minimum-Adam-metric latent correction with actual C+Q backtracking.

    P is positive diagonal. Each row solves P Jᵀ(J P Jᵀ)^†(target-G(z)).
    Try alpha=1,1/2,...,1/256 and accept the first finite strict decrease
    in the actual recomputed unit-mean C+Q. Rejection or zero movement rests;
    exceptions restore z. Only prior_z may be changed by this routine.
    """
    if (prior_z.ndim != 2 or metric.shape != prior_z.shape or real.ndim != 2
            or not len(prior_z) or not len(real)):
        raise ValueError("expected nonempty [particle,latent] and [batch,output] matrices")
    if type(max_halves) is not int or max_halves < 0:
        raise ValueError("max_halves must be a nonnegative integer")
    if not math.isfinite(pinv_rtol) or not 0 < pinv_rtol < 1:
        raise ValueError("pinv_rtol must be finite and in (0,1)")
    if (not torch.isfinite(prior_z).all() or not torch.isfinite(real).all()
            or not torch.isfinite(metric).all() or not bool((metric > 0).all())):
        raise FloatingPointError("invalid real data, prior or positive Adam metric")
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        x = clean(base_z).detach()
        if x.ndim != 2 or x.shape != (len(prior_z), real.shape[1]):
            raise ValueError("clean generator must map prior rows independently to real-row shape")
        target, counts, assignment, nearest_real = chamfer_targets(real, x)
        before_c, before_q = chamfer_terms(real, x)
        before_loss = before_c + before_q
    if not torch.isfinite(before_loss):
        raise FloatingPointError("nonfinite base Chamfer objective")
    one_row = lambda z: clean(z.unsqueeze(0)).squeeze(0)
    with torch.enable_grad():
        jacobian = torch.func.vmap(torch.func.jacfwd(one_row))(base_z).detach()
    if jacobian.shape != (len(x), x.shape[1], prior_z.shape[1]):
        raise RuntimeError("unexpected row-local generator Jacobian shape")
    if not torch.isfinite(jacobian).all():
        raise FloatingPointError("nonfinite generator Jacobian")
    j, p = jacobian.double(), metric.double()
    residual = target - x.double()
    gram = torch.einsum("nmd,nd,nkd->nmk", j, p, j)
    singular = torch.linalg.svdvals(gram)
    rank = (singular > pinv_rtol * singular[:, :1]).sum(dim=1)
    inverse = torch.linalg.pinv(gram, rtol=pinv_rtol, atol=0.)
    correction = p * torch.einsum("nmd,nm->nd", j, torch.einsum("nmk,nk->nm", inverse, residual))
    correction = correction.to(prior_z.dtype)
    if not torch.isfinite(correction).all():
        raise FloatingPointError("nonfinite latent correction")
    linearized = torch.einsum("nmd,nd->nm", j, correction.double())
    tolerance = max(1e-12, 1e-8 * max(1., float(before_loss)))
    accepted, alpha, halves_accepted, trials = False, 0., None, 0
    after_c, after_q = float(before_c), float(before_q)
    # Exact zero is a valid rest, including an unreachable zero-J target.
    try:
        if bool((correction != 0).any()):
            with torch.no_grad():
                for halves in range(max_halves + 1):
                    trials += 1
                    trial_alpha = 2. ** -halves
                    prior_z.copy_(base_z + trial_alpha * correction)
                    if not torch.isfinite(prior_z).all():
                        continue
                    trial_x = clean(prior_z)
                    if not torch.isfinite(trial_x).all():
                        continue
                    c, q = chamfer_terms(real, trial_x)
                    objective = c + q
                    if torch.isfinite(objective) and float(objective) < float(before_loss) - tolerance:
                        accepted, alpha, halves_accepted = True, trial_alpha, halves
                        after_c, after_q = float(c), float(q)
                        break
    finally:
        if not accepted:
            with torch.no_grad():
                prior_z.copy_(base_z)
    with torch.no_grad():
        after_x = clean(prior_z).detach()
    return {
        "accepted": accepted, "alpha": alpha, "accepted_halvings": halves_accepted,
        "trial_evaluations": trials, "max_halves": max_halves, "pinv_rtol": pinv_rtol,
        "objective": "mean_real min_particle squared_distance + mean_particle min_real squared_distance",
        "objective_before": float(before_loss), "objective_after": after_c + after_q,
        "objective_tolerance": tolerance, "coverage_before": float(before_c),
        "coverage_after": after_c, "backward_before": float(before_q), "backward_after": after_q,
        "assigned_counts": counts.tolist(), "empty_cells": int((counts == 0).sum()),
        "real_to_particle": assignment.tolist(), "particle_to_real": nearest_real.tolist(),
        "target_points": target.tolist(), "numerical_rank": rank.tolist(),
        "linearized_output_residual_norm": (linearized - residual).norm(dim=1).tolist(),
        "actual_target_residual_norm": (after_x.double() - target).norm(dim=1).tolist(),
        "latent_correction_norm": float(correction.norm()),
        "actual_latent_displacement_norm": float((prior_z.detach() - base_z).norm()),
        "actual_output_displacement_norm": (after_x - x).norm(dim=1).tolist(),
        "output_before": x.tolist(), "output_after": after_x.tolist(),
    }
