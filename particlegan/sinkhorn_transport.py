"""Finite differentiable, debiased entropic transport on existing batch rows.

This bounded numerical surrogate uses symmetric damped Sinkhorn mappings and
differentiates the complete finite dual objective. It makes no convergence or
population positivity claim. No RNG, labels, sampler, persistent potentials or
detached-plan/envelope approximation enters the objective.
"""
import math

import torch


def auxiliary_rows(values, maximum):
    """Evenly spaced original rows; no draw or change to the caller's batch."""
    if len(values) <= maximum:
        return values
    indices = torch.arange(maximum, device=values.device) * (len(values) - 1)
    return values[indices.div(maximum - 1, rounding_mode="floor")]


def _free_energy(cost, epsilon, iterations, residuals=None):
    """Evaluate the dual after exactly `iterations` symmetric damped mappings."""
    kernel = -cost / epsilon
    log_mass = -math.log(len(cost))
    f, g = kernel.new_zeros(len(cost)), kernel.new_zeros(len(cost))
    for _ in range(iterations):
        next_f = -torch.logsumexp(kernel + g[None, :] + log_mass, dim=1)
        next_g = -torch.logsumexp(kernel + f[:, None] + log_mass, dim=0)
        f, g = .5 * (f + next_f), .5 * (g + next_g)
    log_plan = kernel + f[:, None] + g[None, :] + 2 * log_mass
    if residuals is not None:
        plan = log_plan.detach().exp()
        residuals.append(torch.maximum((plan.sum(0) * len(cost) - 1).abs().max(),
                                       (plan.sum(1) * len(cost) - 1).abs().max()))
    # KL dual includes its mass correction away from the exact fixed point.
    return epsilon * (f.mean() + g.mean() - log_plan.exp().sum() + 1)


def sinkhorn_transport_loss(fake, real, *, epsilon=.1, iterations=24, max_samples=128, stats=None):
    """F(x,y) - F(x,x)/2 - F(y,y)/2, with fully active fake self gradients.

    Cost = mean-coordinate squared distance / (2 * detached real variance).
    All three costs use the same real normalizer and auxiliary row policy.
    The original real batch determines the scale before auxiliary subsampling.
    The finite loss is not clamped: finite-iteration error remains observable.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.device != real.device
            or fake.dtype != real.dtype):
        raise ValueError("Sinkhorn transport needs matching floating batches of at least two samples")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be finite and positive")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("Sinkhorn iterations must be a positive integer")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("Sinkhorn max_samples must be at least two")
    targets = real.detach().flatten(1)
    scale = (targets - targets.mean(0)).square().mean().clamp_min(torch.finfo(fake.dtype).eps)
    x = auxiliary_rows(fake.flatten(1), max_samples)
    y = auxiliary_rows(targets, max_samples)
    residuals = [] if stats is not None else None
    def energy(left, right):
        cost = (left[:, None, :] - right[None, :, :]).square().mean(2) / (2 * scale)
        return _free_energy(cost, epsilon, iterations, residuals)
    loss = energy(x, y) - .5 * energy(x, x) - .5 * energy(y, y)
    if stats is not None:
        stats.update(rows=len(x), input_rows=len(fake), iterations=3 * iterations,
                     maximum_relative_marginal_residual=float(torch.stack(residuals).max()),
                     loss=float(loss.detach()))
    return loss


class SinkhornAudit:
    """Checkpointed actual-call counters; all potentials remain call-local."""
    def __init__(self):
        self.values = dict(calls=0, input_rows=0, auxiliary_rows=0, iterations=0,
                          maximum_relative_marginal_residual=0.0, loss_sum=0.0,
                          negative_loss_calls=0)

    def observe(self, stats):
        self.values['calls'] += 1
        self.values['input_rows'] += stats['input_rows']
        self.values['auxiliary_rows'] += stats['rows']
        self.values['iterations'] += stats['iterations']
        self.values['maximum_relative_marginal_residual'] = max(
            self.values['maximum_relative_marginal_residual'], stats['maximum_relative_marginal_residual'])
        self.values['loss_sum'] += stats['loss']
        self.values['negative_loss_calls'] += int(stats['loss'] < 0)

    def state_dict(self):
        return dict(consumer='sinkhorn_finite_dual_v1', schema_version=1, **self.values)

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.state_dict())
                or state['consumer'] != 'sinkhorn_finite_dual_v1' or state['schema_version'] != 1
                or any(type(state[k]) is not int or state[k] < 0 for k in
                       ('calls', 'input_rows', 'auxiliary_rows', 'iterations', 'negative_loss_calls'))
                or state['negative_loss_calls'] > state['calls']
                or any(type(state[k]) not in (int, float) or not math.isfinite(state[k]) for k in
                       ('maximum_relative_marginal_residual', 'loss_sum'))
                or state['maximum_relative_marginal_residual'] < 0):
            raise ValueError('invalid Sinkhorn audit checkpoint')
        self.values = {k: state[k] for k in self.values}
