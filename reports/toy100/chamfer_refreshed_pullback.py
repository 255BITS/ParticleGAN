"""Four assignment refreshes, then a prior-only Gauss-Newton landing.

The one-step Chamfer target is left unchanged in chamfer_pullback.py. This
helper freezes a target from the current real minibatch and clean support,
refreshes nearest-neighbor assignments four times in output space, and moves
only prior rows toward that frozen target. No target centers, coefficient
search, Adam update, or extra RNG.
"""

import torch

from reports.toy100.chamfer_pullback import chamfer_targets, chamfer_terms


ROUNDS = 4
GN_ITERS = 12


def refreshed_chamfer_target(real, points, rounds=ROUNDS):
    """Repeat the frozen-assignment minimizer, reassigning on the new cloud."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    cloud = points
    target = counts = assignment = nearest = None
    for _ in range(rounds):
        target, counts, assignment, nearest = chamfer_targets(real, cloud)
        cloud = target.to(dtype=points.dtype)
    return target, counts, assignment, nearest


def _prepare(clean):
    modules = clean.modules() if hasattr(clean, "modules") else ()
    for module in modules:
        if isinstance(module, torch.nn.LeakyReLU):
            module.inplace = False


def nonlinear_refreshed_pullback(clean, prior_z, real, *, rounds=ROUNDS, gn_iters=GN_ITERS):
    """Land prior rows on one refreshed target. Restore z unless actual C+Q falls."""
    if type(gn_iters) is not int or gn_iters < 1:
        raise ValueError("gn_iters must be a positive integer")
    if prior_z.ndim != 2 or real.ndim != 2 or not len(prior_z) or not len(real):
        raise ValueError("expected nonempty prior rows and a real minibatch")
    _prepare(clean)
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        base_x = clean(base_z).detach()
        before_c, before_q = chamfer_terms(real, base_x)
        before_loss = before_c + before_q
    if not torch.isfinite(before_loss):
        raise FloatingPointError("nonfinite base Chamfer objective")
    target, counts, assignment, nearest = refreshed_chamfer_target(real, base_x, rounds)
    target = target.to(dtype=base_x.dtype)
    one = lambda row: clean(row.unsqueeze(0)).squeeze(0)
    z = base_z.clone()
    history = []
    accepted_steps = 0
    accepted = False
    try:
        for _ in range(gn_iters):
            with torch.no_grad():
                current = clean(z).detach()
            error = float((current - target).norm())
            history.append(error)
            if error < 1e-4:
                break
            jacobian = torch.func.vmap(torch.func.jacfwd(one))(z).detach().double()
            if not torch.isfinite(jacobian).all():
                break
            residual = (target - current).double()
            gram = torch.einsum("nmd,nkd->nmk", jacobian, jacobian)
            eye = torch.eye(gram.shape[-1], dtype=gram.dtype)
            solved = torch.linalg.solve(gram + 1e-8 * eye, residual)
            delta = torch.einsum("nmd,nm->nd", jacobian, solved).to(dtype=z.dtype)
            if not torch.isfinite(delta).all() or not bool((delta != 0).any()):
                break
            moved = False
            with torch.no_grad():
                origin = z.clone()
                for halves in range(8):
                    trial = origin + (0.5 ** halves) * delta
                    if not torch.isfinite(trial).all():
                        continue
                    landed = clean(trial)
                    if not torch.isfinite(landed).all():
                        continue
                    if float((landed - target).norm()) < error * (1 - 1e-6):
                        z = trial
                        moved = True
                        accepted_steps += 1
                        break
            if not moved:
                break
        with torch.no_grad():
            final_x = clean(z).detach()
            after_c, after_q = chamfer_terms(real, final_x)
            after_loss = after_c + after_q
        tolerance = max(1e-12, 1e-8 * max(1.0, float(before_loss)))
        accepted = bool(torch.isfinite(after_loss) and float(after_loss) < float(before_loss) - tolerance
                        and not torch.equal(z, base_z))
        if not accepted:
            z = base_z
            after_c, after_q = before_c, before_q
            final_x = base_x
    finally:
        if not accepted:
            with torch.no_grad():
                prior_z.copy_(base_z)
        else:
            with torch.no_grad():
                prior_z.copy_(z)
    with torch.no_grad():
        after_x = clean(prior_z).detach()
    return {
        "accepted": accepted,
        "rounds": rounds,
        "gn_iterations": len(history),
        "accepted_gauss_newton_steps": accepted_steps,
        "target_error_trace": history,
        "final_target_error": float((after_x.double() - target.double()).norm()),
        "objective": "refreshed C+Q target, accept only if actual unit-mean C+Q decreases",
        "objective_before": float(before_loss),
        "objective_after": float(after_c + after_q),
        "coverage_before": float(before_c),
        "coverage_after": float(after_c),
        "backward_before": float(before_q),
        "backward_after": float(after_q),
        "assigned_counts": counts.tolist(),
        "empty_cells": int((counts == 0).sum()),
        "real_to_particle": assignment.tolist(),
        "particle_to_real": nearest.tolist(),
        "target_points": target.double().tolist(),
        "actual_target_residual_norm": (after_x.double() - target.double()).norm(dim=1).tolist(),
        "actual_latent_displacement_norm": float((prior_z.detach() - base_z).norm()),
        "output_before": base_x.tolist(),
        "output_after": after_x.tolist(),
    }
