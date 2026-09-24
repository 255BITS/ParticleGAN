"""Bounded nonlinear fitting of fixed output targets through G and its prior.

The target is external to this numerical helper and held fixed. Joint minimum
Euclidean-norm Gauss--Newton steps replace the earlier prior-only linear
pullback. Every accepted trial decreases actual fixed-target squared error;
there is no LR schedule, data draw, Adam update or zero-centered regularizer.
This helper does not supply a data objective or a GAN stability theorem.
"""

import torch
from torch.func import functional_call, jacrev


def fit_output_targets(generator, prior_z, target, *, max_iterations=20,
                       max_halves=12, rtol=1e-6, tolerance=1e-5):
    if (target.ndim != 2 or prior_z.ndim != 2 or len(target) != len(prior_z)
            or not torch.isfinite(target).all() or not torch.isfinite(prior_z).all()):
        raise ValueError('finite row-matched target and prior are required')
    if max_iterations < 1 or max_halves < 0 or not 0 < rtol < 1 or tolerance <= 0:
        raise ValueError('invalid solver settings')
    params = dict(generator.named_parameters())
    leaves = list(params.values()) + [prior_z]
    original = [value.detach().clone() for value in leaves]
    buffers = {name: value.detach().clone() for name, value in generator.named_buffers()}
    target = target.detach().to(prior_z)
    names = list(params)

    def evaluate(values, latent):
        return functional_call(generator, (dict(zip(names, values)), buffers), (latent,))

    def values():
        return tuple(value.detach() for value in params.values()), prior_z.detach()

    @torch.no_grad()
    def install(base, delta, alpha):
        offset = 0
        for leaf, saved in zip(leaves, base):
            size = leaf.numel()
            change = delta[offset:offset+size].reshape_as(leaf).to(leaf)
            leaf.copy_(saved + alpha * change)
            offset += size

    with torch.no_grad():
        initial = evaluate(*values()).detach()
    if initial.shape != target.shape or not torch.isfinite(initial).all():
        raise ValueError('generator output is not finite and row matched')
    scale = max(1., float(target.double().norm(dim=1).max()))
    threshold = tolerance * scale
    records = []
    status = 'BUDGET'
    try:
        for iteration in range(max_iterations):
            state = values()
            with torch.no_grad():
                points = evaluate(*state).detach()
            residual = (target - points).double().flatten()
            before = float(residual.square().sum())
            maximum = float(residual.reshape_as(target).norm(dim=1).max())
            if maximum <= threshold:
                status = 'CONVERGED'
                break
            with torch.enable_grad():
                derivatives = jacrev(evaluate, argnums=(0, 1))(*state)
            blocks = list(derivatives[0]) + [derivatives[1]]
            jacobian = torch.cat([block.reshape(target.numel(), -1) for block in blocks], 1).double()
            if not torch.isfinite(jacobian).all():
                raise FloatingPointError('nonfinite joint output Jacobian')
            # Thin SVD avoids squaring the condition number in J J^T.
            u, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
            active = singular > rtol * singular.max()
            inverse = torch.where(active, 1 / singular.clamp_min(torch.finfo(singular.dtype).tiny), 0.)
            delta = vh.T @ (inverse * (u.T @ residual))
            if not torch.isfinite(delta).all():
                raise FloatingPointError('nonfinite joint Gauss--Newton step')
            base = [value.detach().clone() for value in leaves]
            accepted = False
            trial_rows = []
            for halves in range(max_halves + 1):
                alpha = 2. ** -halves
                install(base, delta, alpha)
                with torch.no_grad():
                    trial = evaluate(*values())
                    error = (target - trial).double().flatten()
                    after = float(error.square().sum())
                predicted = residual - alpha * (jacobian @ delta)
                decrease = before - float(predicted.square().sum())
                finite = bool(torch.isfinite(trial).all())
                # Standard actual/predicted reduction safeguard. This tests
                # the nonlinear map, including all shared network movement.
                ratio = (before-after)/decrease if decrease > 0 and finite else None
                accepted = ratio is not None and ratio >= .1 and after < before
                trial_rows.append(dict(alpha=alpha, finite=finite,
                    squared_error=after if finite else None, ratio=ratio, accepted=accepted))
                if accepted:
                    break
            if not accepted:
                with torch.no_grad():
                    for leaf, saved in zip(leaves, base):
                        leaf.copy_(saved)
                status = 'NO_ACCEPTABLE_STEP'
            records.append(dict(iteration=iteration+1, rank=int(active.sum()),
                singular_min=float(singular[-1]), singular_max=float(singular[0]),
                parameter_step_norm=float(delta.norm()), before=before,
                after=after if accepted else before, accepted=accepted, trials=trial_rows))
            if not accepted:
                break
        with torch.no_grad():
            final = evaluate(*values()).detach()
        final_error = (target-final).double()
        if float(final_error.norm(dim=1).max()) <= threshold:
            status = 'CONVERGED'
    except BaseException:
        with torch.no_grad():
            for leaf, saved in zip(leaves, original):
                leaf.copy_(saved)
        raise
    if any(not torch.equal(value, buffers[name]) for name,value in generator.named_buffers()):
        raise RuntimeError('functional output fitting changed persistent buffers')
    return dict(status=status, objective='fixed-target sum squared output error',
        max_iterations=max_iterations, max_halves=max_halves, svd_rtol=rtol,
        tolerance=tolerance, absolute_threshold=threshold, records=records,
        initial_squared_error=float((target-initial).double().square().sum()),
        final_squared_error=float(final_error.square().sum()),
        final_max_row_error=float(final_error.norm(dim=1).max()),
        actual_output_displacement=(final-initial).double().norm(dim=1).tolist(),
        actual_parameter_displacement=float(sum((leaf.detach()-saved).double().square().sum()
            for leaf,saved in zip(leaves,original)).sqrt()),
        optimizer_scope='no Adam state, learning-rate or random-stream mutation')
