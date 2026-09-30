"""Prior pullback of one reverse-KL + chi-squared KDE drift.

Cao & Wei, arXiv:2603.10592, Table 1 and equation (8): the practical velocity
is the KDE mean-shift gap times (p/q + q/p), the sum of the reverse-KL and
chi-squared weights. The 1/h^2 factor is omitted, matching their h^2-scaled
drifting identity. This is an added objective, not a GAN optimizer and not a
forward-KL drifting field.

Bandwidth is the maximum real-to-nearest-particle distance, clamped to
[0.07, 3]. Output steps are capped at 0.1. The latent step is the
Levenberg-Marquardt solution in the current Adam metric, then scaled so no
particle moves more than 1 in latent norm. A step is kept only when the
frozen-bandwidth mixture energy strictly falls.
"""
import math

import torch


WIDTH_FLOOR = 0.07
WIDTH_CAP = 3.0
STEP_CAP = 0.1
LATENT_CAP = 1.0
RIDGE = 1e-2


def bandwidth(real, points):
    gap = torch.cdist(real, points).min(dim=1).values
    return torch.quantile(gap, 1.0).clamp(WIDTH_FLOOR, WIDTH_CAP)


def _kde(query, samples, h):
    log_k = -0.5 * torch.cdist(query, samples).square() / (h * h)
    log_mass = torch.logsumexp(log_k, dim=1) - math.log(len(samples))
    weight = (log_k - log_k.max(dim=1, keepdim=True).values).exp()
    mean = (weight @ samples) / weight.sum(dim=1, keepdim=True)
    return log_mass, mean


def mixture_energy(points, real, h):
    """KL(p||q) + chi^2(q||p) under one shared Gaussian KDE bandwidth."""
    log_p_real, _ = _kde(real, real, h)
    log_q_real, _ = _kde(real, points, h)
    log_p_part, _ = _kde(points, real, h)
    log_q_part, _ = _kde(points, points, h)
    reverse_kl = (log_p_real - log_q_real).mean()
    chi2 = (log_q_part - log_p_part).exp().mean() - 1.
    return reverse_kl + chi2


def drift(points, real, h):
    log_p, mean_p = _kde(points, real, h)
    log_q, mean_q = _kde(points, points, h)
    ratio = (log_p - log_q).exp().clamp(1e-6, 1e6)
    return (ratio + 1. / ratio)[:, None] * (mean_p - mean_q)


def mixture_pullback(clean, prior_z, real, metric, *, max_halves=8):
    """Move the prior toward one capped mixture step. Network weights stay fixed."""
    if prior_z.ndim != 2 or real.ndim != 2 or metric.shape != prior_z.shape:
        raise ValueError("expected [particle,latent] prior and [batch,output] reals")
    if not torch.isfinite(real).all() or not torch.isfinite(metric).all() or not bool((metric > 0).all()):
        raise FloatingPointError("nonfinite real batch or metric")
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        x = clean(base_z).detach()
        if x.ndim != 2 or x.shape[0] != prior_z.shape[0] or x.shape[1] != real.shape[1]:
            raise ValueError("clean generator must map prior rows to the real output shape")
        h = bandwidth(real, x)
        velocity = drift(x, real, h)
        speed = velocity.norm(dim=1, keepdim=True).clamp_min(1e-12)
        residual = velocity * (STEP_CAP / speed).clamp(max=1.)
        base_energy = mixture_energy(x, real, h)
    if not torch.isfinite(x).all() or not torch.isfinite(base_energy) or not torch.isfinite(residual).all():
        raise FloatingPointError("invalid mixture field")
    if float(residual.norm()) == 0.:
        return _row(True, 0., h, base_energy, base_energy, 0, None, 0, prior_z, base_z, x, x)

    one_row = lambda z: clean(z.unsqueeze(0)).squeeze(0)
    with torch.enable_grad():
        jacobian = torch.func.vmap(torch.func.jacfwd(one_row))(base_z).detach()
    if jacobian.shape != (len(x), x.shape[1], prior_z.shape[1]) or not torch.isfinite(jacobian).all():
        raise FloatingPointError("invalid particle Jacobian")
    j = jacobian.double()
    p = metric.double()
    gram = torch.einsum("nmd,nd,nkd->nmk", j, p, j)
    scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    ridge = RIDGE * scale
    solved = torch.linalg.solve(gram + ridge[:, None, None] * torch.eye(x.shape[1], dtype=torch.float64),
                                residual.double())
    correction = (p * torch.einsum("nmd,nm->nd", j, solved)).to(prior_z.dtype)
    row_norm = correction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    correction = correction * (LATENT_CAP / row_norm).clamp(max=1.)
    if not torch.isfinite(correction).all():
        raise FloatingPointError("nonfinite latent correction")

    accepted = False
    alpha = 0.
    accepted_halvings = None
    trials = 0
    after = float(base_energy)
    tolerance = max(1e-12, 1e-8 * max(1., abs(float(base_energy))))
    try:
        with torch.no_grad():
            for halves in range(max_halves + 1):
                trials += 1
                trial_alpha = 2. ** -halves
                prior_z.copy_(base_z + trial_alpha * correction)
                trial_x = clean(prior_z)
                if not torch.isfinite(trial_x).all():
                    continue
                trial_energy = mixture_energy(trial_x, real, h)
                if torch.isfinite(trial_energy) and float(trial_energy) < float(base_energy) - tolerance:
                    accepted = True
                    alpha = trial_alpha
                    accepted_halvings = halves
                    after = float(trial_energy)
                    break
    finally:
        if not accepted:
            with torch.no_grad():
                prior_z.copy_(base_z)
    with torch.no_grad():
        output_after = clean(prior_z).detach()
    return _row(accepted, alpha, h, float(base_energy), after, trials, accepted_halvings,
                float(correction.norm()), prior_z, base_z, x, output_after)


def _row(accepted, alpha, h, before, after, trials, halvings, correction_norm, prior_z, base_z, x, output_after):
    return dict(accepted=accepted, alpha=alpha, bandwidth=float(h),
                energy_before=before, energy_after=after, trial_evaluations=trials,
                accepted_halvings=halvings, latent_correction_norm=correction_norm,
                step_cap=STEP_CAP, latent_cap=LATENT_CAP,
                actual_latent_displacement_norm=float((prior_z.detach() - base_z).norm()),
                max_output_displacement=float((output_after - x).norm(dim=1).max()),
                output_before=x.tolist(), output_after=output_after.tolist())
