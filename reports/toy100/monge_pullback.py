"""Prior-only Monge correction: bijection to farthest real representatives.

Chamfer and one-sided Lloyd targets are Voronoi means. When one particle is
nearest to two modes, that mean sits in the score valley between them. Sinkhorn
barycentric drift has the same averaging geometry. A minimum-cost bijection
between the N particles and N farthest-point real representatives assigns a
surplus particle to uncovered mass and leaves its partner on the occupied mode.

The requested output step is capped, then pulled back through the row-local
generator Jacobian in the already-updated Adam metric. This is an added
sampled-data objective, not a GAN optimizer-only rule and not a mode-center
oracle. No RNG is consumed.
"""
import math

import torch


OUTPUT_CAP = 0.2


def hungarian(cost):
    """Minimum-cost bijection. ``cost`` is square. Returns column per row."""
    n = int(cost.shape[0])
    if cost.ndim != 2 or cost.shape[1] != n:
        raise ValueError("Hungarian cost must be square")
    a = cost.detach().double().cpu().tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return torch.tensor(assignment, dtype=torch.long, device=cost.device)


def farthest_representatives(real, count):
    """Deterministic farthest-point subset. Start at max norm, then lowest index."""
    if real.ndim != 2 or count < 1 or count > len(real):
        raise ValueError("need a nonempty real matrix and 1 <= count <= batch")
    # torch.argmax returns the lowest index on ties.
    start = int(torch.argmax(real.norm(dim=1)).item())
    chosen = [start]
    nearest = torch.cdist(real[start:start + 1], real).squeeze(0)
    for _ in range(count - 1):
        score = nearest.clone()
        score[chosen] = -1.
        chosen.append(int(torch.argmax(score).item()))
        nearest = torch.minimum(nearest, torch.cdist(real[chosen[-1]:chosen[-1] + 1], real).squeeze(0))
    return real[chosen]


def _cap_rows(residual, cap):
    length = residual.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return residual * torch.clamp(cap / length, max=1.)


def monge_pullback(clean, prior_z, real, metric, *, output_cap=OUTPUT_CAP,
                   max_halves=8, pinv_rtol=1e-6):
    """Move each prior row toward its matched real representative, at most ``output_cap``."""
    if prior_z.ndim != 2 or real.ndim != 2 or metric.shape != prior_z.shape:
        raise ValueError("expected [particle,latent] and [batch,output] matrices")
    if output_cap <= 0:
        raise ValueError("output cap must be positive")
    if not torch.isfinite(real).all() or not torch.isfinite(metric).all() or not bool((metric > 0).all()):
        raise FloatingPointError("nonfinite real batch or metric")
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        x = clean(base_z).detach()
        if x.ndim != 2 or x.shape[0] != prior_z.shape[0] or x.shape[1] != real.shape[1]:
            raise ValueError("clean generator must map prior rows to real-row shape")
        targets = farthest_representatives(real, len(x))
        assignment = hungarian(torch.cdist(x, targets).square())
        matched = targets[assignment]
        raw = matched - x
        residual = _cap_rows(raw, output_cap)
        base_loss = (matched - x).square().mean()
    if not torch.isfinite(x).all() or not torch.isfinite(base_loss):
        raise FloatingPointError("invalid clean generated support")

    one_row = lambda z: clean(z.unsqueeze(0)).squeeze(0)
    with torch.enable_grad():
        jacobian = torch.func.vmap(torch.func.jacfwd(one_row))(base_z).detach()
    if jacobian.shape != (len(x), x.shape[1], prior_z.shape[1]):
        raise RuntimeError("unexpected particle Jacobian shape")
    if not torch.isfinite(jacobian).all():
        raise FloatingPointError("nonfinite particle Jacobian")
    j = jacobian.double()
    p = metric.double()
    residual64 = residual.double()
    gram = torch.einsum("nmd,nd,nkd->nmk", j, p, j)
    singular = torch.linalg.svdvals(gram)
    numerical_rank = (singular > pinv_rtol * singular[:, :1]).sum(dim=1)
    inverse = torch.linalg.pinv(gram, rtol=pinv_rtol, atol=0.)
    correction = p * torch.einsum("nmd,nm->nd", j, torch.einsum("nmk,nk->nm", inverse, residual64))
    correction = correction.to(prior_z.dtype)
    if not torch.isfinite(correction).all():
        raise FloatingPointError("nonfinite latent correction")
    linearized = torch.einsum("nmd,nd->nm", j, correction.double())
    residual_norm = (linearized - residual64).norm(dim=1)
    tolerance = max(1e-12, 1e-8 * max(1., float(base_loss)))
    accepted = False
    after_loss = float(base_loss)
    alpha = 0.
    accepted_halvings = None
    trials = 0
    try:
        with torch.no_grad():
            if float(residual.norm()) <= tolerance:
                trials = 0
            else:
                for halves in range(max_halves + 1):
                    trials += 1
                    trial_alpha = 2. ** -halves
                    prior_z.copy_(base_z + trial_alpha * correction)
                    trial_x = clean(prior_z)
                    if not torch.isfinite(trial_x).all():
                        continue
                    trial_loss = (matched - trial_x).square().mean()
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
        "accepted": accepted, "alpha": alpha, "monge_before": float(base_loss),
        "monge_after": after_loss, "monge_tolerance": tolerance,
        "output_cap": float(output_cap), "max_halves": max_halves,
        "accepted_halvings": accepted_halvings, "trial_evaluations": trials,
        "pinv_rtol": pinv_rtol, "numerical_rank": numerical_rank.tolist(),
        "linearized_output_residual_norm": residual_norm.tolist(),
        "assignment": assignment.tolist(),
        "raw_output_gap_norm": raw.norm(dim=1).tolist(),
        "capped_output_step_norm": residual.norm(dim=1).tolist(),
        "latent_correction_norm": float(correction.norm()),
        "max_particle_correction_norm": float(correction.norm(dim=1).max()),
        "actual_latent_displacement_norm": float((prior_z.detach() - base_z).norm()),
        "actual_output_displacement_norm": (output_after - x).norm(dim=1).tolist(),
        "output_before": x.tolist(),
        "output_after": output_after.tolist(),
        "matched_targets": matched.tolist(),
    }
