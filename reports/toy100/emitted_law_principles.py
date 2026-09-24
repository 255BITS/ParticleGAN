"""Small mathematical controls; no training adapter or target controller."""

import numpy as np
import torch


def fixed_affine_prior(outputs, weight, bias):
    """Reference-relative exact chart for a fixed invertible affine generator."""
    return torch.linalg.solve(weight, (outputs-bias).T).T


def gaussian_rp_generator_loss(reference_mean, trial_mean, real_sigma, fake_sigma, nodes=96):
    """1-D fixed population-ratio critic, evaluated by deterministic quadrature.

    The critic belongs to reference_mean.  Only trial_mean is a G decision;
    this is not a total derivative through a responding critic.
    """
    x, w = np.polynomial.hermite.hermgauss(nodes)
    x = torch.as_tensor(x, dtype=trial_mean.dtype, device=trial_mean.device)*(2.**.5)
    w = torch.as_tensor(w, dtype=trial_mean.dtype, device=trial_mean.device)/(np.pi**.5)
    real, fake = real_sigma*x, trial_mean+fake_sigma*x
    def critic(value):
        return (value-reference_mean).square()/(2*fake_sigma**2)-value.square()/(2*real_sigma**2)
    return (w[:, None]*w[None, :]*torch.nn.functional.softplus(
        critic(real)[:, None]-critic(fake)[None, :])).sum()


def kernel_mixture_nll(weights, kernel_values):
    """Convex finite-dictionary likelihood in probability weights."""
    return -(kernel_values@weights).log().mean()


def kernel_mixture_em_step(weights, kernel_values):
    """Exact EM/MM step for fixed columns; zero columns cannot be born here."""
    density = kernel_values@weights
    return weights*(kernel_values/density[:, None]).mean(0)


def new_atom_directional_derivative(current_density, candidate_density):
    """NLL derivative at (1-alpha)q+alpha*k; no geometric group identity."""
    return 1-(candidate_density/current_density).mean()


def count_quotas(counts, particles):
    """Largest remainders with integer arithmetic and stable identity ties."""
    if counts.dtype != torch.int64 or not (counts >= 0).all() or int(counts.sum()) <= 0:
        raise ValueError('nonnegative integer counts with positive total required')
    denominator = int(counts.sum())
    numerator = counts*particles
    quotas = numerator//denominator
    remaining = particles-int(quotas.sum())
    order = torch.argsort(numerator % denominator, descending=True, stable=True)
    quotas[order[:remaining]] += 1
    return quotas
