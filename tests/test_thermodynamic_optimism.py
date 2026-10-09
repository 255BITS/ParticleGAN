"""CPU algebra/checkpoint tests; task qualification uses the bounded CUDA study."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer, convolution_parameter_groups, polar_factor
from experiments.forge.techniques import validate_same_technique


def test_formula_first_step_zero_gradient_missing_gradient_and_current_rate():
    p = nn.Parameter(torch.tensor([[.2, -.3], [.4, .1]], dtype=torch.float64))
    opt = NormalizedOptimizer([p], lr=.03, smoothing=.001, thermodynamic_optimism=1.)
    expected, previous = p.detach().clone(), None
    for values, rate in [([[.3, .1], [-.2, .4]], .03), ([[.1, -.2], [.4, -.3]], .01),
                         ([[0., 0.], [0., 0.]], .02)]:
        p.grad = torch.tensor(values, dtype=p.dtype)
        current = polar_factor(p.grad, smoothing=.001)
        expected -= rate * (current if previous is None else 2 * current - previous)
        previous = current.clone()
        opt.param_groups[0]['lr'] = rate
        opt.step()
        torch.testing.assert_close(p, expected, atol=1e-15, rtol=0)
    before = deepcopy(opt.state_dict())
    p.grad = None
    opt.step()
    assert opt.state[p]['step'] == before['state'][0]['step']
    assert torch.equal(opt.state[p]['thermodynamic_previous_direction'], previous)


def test_recipe_prior_ownership_and_exact_resume():
    recipe = get_recipe('bcap', thermodynamic_optimism=1., num_particles=8, z_dim=2)
    g, d, prior = nn.Linear(2, 2), nn.Linear(2, 1), recipe.make_prior()
    og, od = recipe.make_optimizers(g, d, prior)
    for opt in (og, od):
        for group in opt.param_groups:
            for p in group['params']:
                p.grad = torch.full_like(p, .05)
    og.set_sampled_rows(prior.z, torch.tensor([1, 3]))
    before = prior.z.detach().clone()
    og.step(); od.step()
    assert torch.equal(prior.z[[0, 2, 4, 5, 6, 7]], before[[0, 2, 4, 5, 6, 7]])
    assert 'thermodynamic_previous_direction' not in og.state[prior.z]
    saved, values = deepcopy(og.state_dict()), [p.detach().clone() for group in og.param_groups for p in group['params']]
    for group in og.param_groups:
        for p in group['params']:
            p.grad = torch.full_like(p, -.02)
    og.set_sampled_rows(prior.z, torch.tensor([2, 4]))
    og.step()
    final = [p.detach().clone() for group in og.param_groups for p in group['params']]
    for p, v in zip([p for group in og.param_groups for p in group['params']], values):
        p.data.copy_(v)
    og.load_state_dict(saved)
    og.set_sampled_rows(prior.z, torch.tensor([2, 4]))
    og.step()
    assert all(torch.equal(p, v) for p, v in zip([p for group in og.param_groups for p in group['params']], final))
    corrupt = deepcopy(saved)
    corrupt['dualnorm']['thermodynamic_optimism'] = .5
    with pytest.raises(ValueError, match='optimism'):
        og.load_state_dict(corrupt)
    corrupt = deepcopy(saved)
    next(v for v in corrupt['state'].values() if 'thermodynamic_previous_direction' in v)['thermodynamic_previous_direction'].fill_(float('nan'))
    with pytest.raises(ValueError, match='previous_direction'):
        og.load_state_dict(corrupt)


@pytest.mark.parametrize('layer', [nn.Conv2d(2, 3, 3), nn.ConvTranspose2d(2, 3, 3)])
def test_convolution_normalization_is_applied_before_history(layer):
    p = layer.weight
    groups = convolution_parameter_groups([{'params': [p], 'role': 'generator'}], [layer], family='dualnorm')
    base = NormalizedOptimizer(groups, lr=.03, smoothing=.001, convolution='per_offset')
    trial_layer = deepcopy(layer)
    trial_p = trial_layer.weight
    trial_groups = convolution_parameter_groups([{'params': [trial_p], 'role': 'generator'}], [trial_layer], family='dualnorm')
    trial = NormalizedOptimizer(trial_groups, lr=.03, smoothing=.001, convolution='per_offset', thermodynamic_optimism=1.)
    p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape_as(p) / 100
    trial_p.grad = p.grad.clone()
    base.step(); trial.step()
    torch.testing.assert_close(p, trial_p, atol=1e-7, rtol=0)
    start = trial_p.detach().clone()
    trial_p.grad.zero_()
    previous = trial.state[trial_p]['thermodynamic_previous_direction'].clone()
    trial.step()
    torch.testing.assert_close(trial_p, start + .03 * previous, atol=1e-7, rtol=0)


def test_default_packet_and_structural_admission():
    p = nn.Parameter(torch.ones(2, 2))
    opt = NormalizedOptimizer([p])
    p.grad = torch.eye(2); opt.step()
    assert set(opt.state[p]) == {'step'}
    assert set(opt.state_dict()['dualnorm']) == {'schema', 'family', 'momentum', 'sampled_rows'}
    baseline = get_recipe('bcap')
    with pytest.raises(ValueError, match='technique mechanisms'):
        validate_same_technique(baseline, baseline.replace(thermodynamic_optimism=1.))
    for value in (True, -1., 1.1, float('nan')):
        with pytest.raises(ValueError, match='optimism'):
            baseline.replace(thermodynamic_optimism=value)
    with pytest.raises(ValueError, match='zero-momentum'):
        baseline.replace(thermodynamic_optimism=1., optimizer_momentum=.5)


def test_local_rotation_eigenvalues_and_asymptotic_energy():
    # Algebraic stability model only, not a trained toy or qualification cell.
    a = .1
    root = math.sqrt(1 - 4 * a * a)
    eigs = ((1 - 2j * a + root) / 2, (1 - 2j * a - root) / 2)
    assert all(abs(v) < 1 for v in eigs)
    assert abs(eigs[0]) ** 2 == pytest.approx((1 + root) / 2)
    x = complex(1, 0)
    previous = x
    x -= 1j * a * x  # Same ordinary-first-step contract.
    for _ in range(200):
        x, previous = x - 2j * a * x + 1j * a * previous, x
    assert abs(x) ** 2 < .15
    assert abs((1 - 1j * a) ** 201) ** 2 > 7
