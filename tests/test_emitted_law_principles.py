"""Scoped response/no-Nash/convex-law controls, independent of host training."""
import torch

from reports.toy100.emitted_law_principles import (
    count_quotas, fixed_affine_prior, gaussian_rp_generator_loss,
    kernel_mixture_em_step, kernel_mixture_nll, new_atom_directional_derivative,
)


def test_fixed_affine_chart_is_exact_and_path_independent():
    weight = torch.tensor([[2., .4], [-.3, 1.]], dtype=torch.float64)
    bias = torch.tensor([.2, -.7], dtype=torch.float64)
    y = torch.tensor([[1., 2.], [3., -4.], [.4, .1]], dtype=torch.float64)
    original = fixed_affine_prior(y, weight, bias)
    assert torch.allclose(original@weight.T+bias, y, atol=1e-14)
    # A closed output loop cannot retain the parameter-space holonomy of a
    # nonlinear minimum-increment lift when each chart solve uses this reference.
    for current in (y+1, y-2, y):
        z = fixed_affine_prior(current, weight, bias)
    assert torch.equal(z, original)


def test_unrestricted_ratio_critic_has_tail_escape_despite_stationary_mean():
    mean = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    baseline = gaussian_rp_generator_loss(0., mean, .07, .029)
    first = torch.autograd.grad(baseline, mean, create_graph=True)[0]
    second = torch.autograd.grad(first, mean)[0]
    assert abs(float(first.detach())) < 1e-12
    assert float(second) < 0  # stationary partial field is not a G local minimum
    losses = [float(gaussian_rp_generator_loss(0., torch.tensor(m, dtype=torch.float64), .07, .029))
              for m in (0., .07, .21, .7)]
    assert all(a > b for a, b in zip(losses, losses[1:]))
    assert losses[-1] < 1e-12
    # In the same misspecified translated-kernel family, the proper population
    # negative log likelihood has a finite best mean and positive curvature.
    nll = (.07**2+mean.square())/(2*.029**2)
    assert torch.autograd.grad(torch.autograd.grad(nll, mean, create_graph=True)[0], mean)[0] > 0


def test_fixed_dictionary_objective_is_convex_and_em_decreases_it():
    k = torch.tensor([[.9, .1], [.8, .2], [.1, .9], [.2, .8]], dtype=torch.float64)
    w = torch.tensor([.8, .2], dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(lambda p: kernel_mixture_nll(p, k), w)
    assert torch.linalg.eigvalsh(hessian).min() > 0
    next_w = kernel_mixture_em_step(w.detach(), k)
    assert torch.all(next_w > 0) and torch.allclose(next_w.sum(), torch.tensor(1., dtype=w.dtype))
    assert kernel_mixture_nll(next_w, k) < kernel_mixture_nll(w, k)
    equilibrium = torch.tensor([.5, .5], dtype=w.dtype)
    assert torch.allclose(kernel_mixture_em_step(equilibrium, k), equilibrium, rtol=0, atol=1e-15)


def test_missing_atom_score_detects_signal_without_fixed_group_association():
    q = torch.tensor([.9, .9, .01, .01], dtype=torch.float64)
    new = torch.tensor([.01, .01, .9, .9], dtype=torch.float64)
    derivative = new_atom_directional_derivative(q, new)
    h = 1e-7
    fd = (-((1-h)*q+h*new).log().mean()+((1+h)*q-h*new).log().mean())/(2*h)
    assert derivative < 0 and torch.allclose(derivative, fd, rtol=1e-8)


def test_integer_allocation_error_bound_and_noise_covariance_floor():
    counts = torch.tensor([11, 19, 7, 23], dtype=torch.int64)
    n = 200
    quotas = count_quotas(counts, n)
    error = (quotas.double()/n-counts.double()/counts.sum()).abs()
    assert int(quotas.sum()) == n and torch.all(error < 1/n)
    assert float(error.sum()/2) < len(counts)/(2*n)
    # Emitted covariance is clean covariance plus the fixed output-noise floor.
    sigma = .029
    requested = torch.diag(torch.tensor([.02**2, .04**2], dtype=torch.float64))
    clean = requested-sigma**2*torch.eye(2, dtype=torch.float64)
    assert torch.linalg.eigvalsh(clean).min() < 0


def test_all_donor_exchange_certificate_bounds_convex_mixture_gap():
    # Three fixed feature atoms. Target lies outside their convex hull, so
    # the comparator is the actual convex optimum (.5,.5), not zero loss.
    features = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
    target = torch.tensor([.8, .8], dtype=torch.float64)
    optimum = float((target-torch.tensor([.5, .5])).square().sum())
    diameter2 = float(torch.cdist(features, features).square().max())
    n = 12
    for n0 in range(n+1):
        for n1 in range(n-n0+1):
            counts = torch.tensor([n0, n1, n-n0-n1])
            mean = counts.double()@features/n
            value = float((mean-target).square().sum())
            best_decrease = 0.
            for donor in torch.nonzero(counts).flatten():
                for candidate in features:
                    changed = mean+(candidate-features[donor])/n
                    best_decrease = max(best_decrease, value-float((changed-target).square().sum()))
            # No-improvement case is epsilon=0. The residual form remains
            # meaningful when a bounded search stops before stationarity.
            assert value-optimum <= diameter2/n+n*best_decrease+1e-12
