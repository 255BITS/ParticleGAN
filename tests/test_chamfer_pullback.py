"""Closed-form targets, nonlinear acceptance, resting and state-preservation."""

from copy import deepcopy

import pytest
import torch

from reports.toy100.chamfer_pullback import chamfer_pullback, chamfer_targets, chamfer_terms


def test_unit_means_target_matches_frozen_quadratic_minimizer_and_rescues_empty_cell():
    real = torch.tensor([[-2.], [0.], [8.]], dtype=torch.float64)
    points = torch.tensor([[-1.], [2.], [20.]], dtype=torch.float64)
    target, counts, assigned, nearest = chamfer_targets(real, points)
    assert counts.tolist() == [2, 1, 0]
    assert assigned.tolist() == [0, 0, 1] and nearest.tolist() == [0, 1, 2]
    assert torch.allclose(target, torch.tensor([[-4 / 3], [4.], [8.]], dtype=torch.float64))
    trial = target.detach().requires_grad_()
    frozen_quadratic = ((real - trial[assigned]).square().sum(dim=1).mean()
                        + (trial - real[nearest]).square().sum(dim=1).mean())
    assert torch.allclose(torch.autograd.grad(frozen_quadratic, trial)[0],
                          torch.zeros_like(trial), atol=1e-14, rtol=0)
    z = torch.nn.Parameter(points.clone())
    row = chamfer_pullback(torch.nn.Identity(), z, real, torch.ones_like(z))
    assert row['accepted'] and row['alpha'] == 1
    assert torch.allclose(z, target)
    assert row['objective_after'] < row['objective_before']


def test_unit_means_do_not_confuse_batch_size_with_particle_count():
    # B=4, N=2. Forward sums/counts and backward mass each have their own mean.
    real = torch.tensor([[0.], [1.], [2.], [10.]], dtype=torch.float64)
    points = torch.tensor([[1.], [20.]], dtype=torch.float64)
    target, counts, _, _ = chamfer_targets(real, points)
    assert counts.tolist() == [4, 0]
    assert torch.allclose(target, torch.tensor([[2.5], [10.]], dtype=torch.float64))


def test_backward_term_detects_forward_coverage_improvement_with_stray_particle():
    real = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    c0, q0 = chamfer_terms(real, torch.tensor([[-1.], [0.], [2.]], dtype=torch.float64))
    c1, q1 = chamfer_terms(real, torch.tensor([[-1.], [1.], [10.]], dtype=torch.float64))
    assert c1 < c0 and c1 + q1 > c0 + q0
    assert float(c0 + q0) == pytest.approx(7 / 6)
    assert float(c1 + q1) == pytest.approx(27.)


def test_metric_pullback_is_exact_weighted_minimum_norm_solution():
    model = torch.nn.Linear(4, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., 0., 1., 0.], [0., 3., 0., 1.]]))
    z = torch.nn.Parameter(torch.zeros((1, 4), dtype=torch.float64))
    metric = torch.tensor([[.5, 2., 1., 4.]], dtype=torch.float64)
    row = chamfer_pullback(model, z, torch.tensor([[.6, -.4]], dtype=torch.float64), metric)
    expected = torch.tensor([[.2, -2.4 / 22, .2, -1.6 / 22]], dtype=torch.float64)
    assert torch.allclose(z, expected, atol=1e-13, rtol=0)
    assert row['numerical_rank'] == [2] and row['objective_after'] < 1e-25


def test_matched_support_and_zero_jacobian_rest_without_rng_or_parameter_movement():
    for model, z0, real in ((torch.nn.Identity(), [[-1.], [2.]], [[-1.], [2.]]),
                            (lambda z: z * 0, [[.4]], [[1.]])):
        z = torch.nn.Parameter(torch.tensor(z0, dtype=torch.float64))
        base, rng = z.detach().clone(), torch.get_rng_state().clone()
        row = chamfer_pullback(model, z, torch.tensor(real, dtype=torch.float64), torch.ones_like(z))
        assert not row['accepted'] and row['alpha'] == 0 and row['trial_evaluations'] == 0
        assert torch.equal(z, base) and torch.equal(torch.get_rng_state(), rng)


def test_nonlinear_acceptance_checks_actual_chamfer_and_exhaustion_restores_prior():
    real = torch.tensor([[10.]], dtype=torch.float64)
    z = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    row = chamfer_pullback(torch.exp, z, real, torch.ones_like(z))
    assert row['accepted'] and row['alpha'] == .25 and row['trial_evaluations'] == 3
    assert z.item() == pytest.approx(2.25)
    assert row['objective_after'] == pytest.approx(2 * (10 - torch.exp(z).item()) ** 2)
    with torch.no_grad():
        z.zero_()
    row = chamfer_pullback(torch.exp, z, real, torch.ones_like(z), max_halves=0)
    assert not row['accepted'] and row['alpha'] == 0 and z.item() == 0
    assert row['objective_after'] == row['objective_before'] == 162.


def test_preserves_network_gradients_adam_state_and_rng_and_rejects_zero_metric():
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    z = torch.nn.Parameter(torch.tensor([[.2, -.4]], dtype=torch.float64))
    opt = torch.optim.Adam([*model.parameters(), z], lr=.02, betas=(0., .99))
    for p in [*model.parameters(), z]:
        p.grad = torch.ones_like(p)
    opt.step()
    weights = deepcopy(model.state_dict())
    state = deepcopy(opt.state_dict())
    grads = [p.grad.clone() for p in [*model.parameters(), z]]
    rng = torch.get_rng_state().clone()
    real = model(z).detach() + .1
    row = chamfer_pullback(model, z, real, torch.ones_like(z))
    assert row['accepted'] and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())
    assert all(torch.equal(p.grad, g) for p, g in zip([*model.parameters(), z], grads))
    now = opt.state_dict()
    assert now['param_groups'] == state['param_groups']
    for key, values in state['state'].items():
        assert all(torch.equal(v, now['state'][key][name]) for name, v in values.items())
    base = z.detach().clone()
    with pytest.raises(FloatingPointError, match='metric'):
        chamfer_pullback(model, z, real, torch.zeros_like(z))
    assert torch.equal(z, base)


def test_exception_during_nonlinear_trial_rolls_back():
    def fragile(z):
        if not torch.is_grad_enabled() and bool((z > .5).any()):
            raise RuntimeError('trial failure')
        return z
    z = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    with pytest.raises(RuntimeError, match='trial failure'):
        chamfer_pullback(fragile, z, torch.ones_like(z), torch.ones_like(z))
    assert z.item() == 0
