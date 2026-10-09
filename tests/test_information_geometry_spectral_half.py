"""Algebra, public factory and exact-resume software contracts, no training."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor
from particlegan.optim.information_geometry import information_geometry_spectral_half as half


def test_singular_contrast_and_top_step_bound():
    gradient = torch.diag(torch.tensor([4., 1., .04, 0.], dtype=torch.float64))
    torch.testing.assert_close(half(gradient).diag(), torch.tensor([1., .5, .1, 0.], dtype=torch.float64))
    for smoothing in (0., .001, .3):
        candidate = half(gradient, smoothing=smoothing)
        control = polar_factor(gradient, smoothing=smoothing)
        torch.testing.assert_close(torch.linalg.matrix_norm(candidate, 2), torch.linalg.matrix_norm(control, 2))
        assert torch.linalg.matrix_norm(candidate) <= torch.linalg.matrix_norm(control)
        assert torch.count_nonzero(half(torch.zeros_like(gradient), smoothing=smoothing)) == 0


def test_orthogonal_equivariance_and_low_precision():
    gradient = torch.arange(15, dtype=torch.float64).reshape(3, 5).sin()
    left = torch.linalg.qr(torch.arange(9, dtype=torch.float64).reshape(3, 3).cos()).Q
    right = torch.linalg.qr(torch.arange(25, dtype=torch.float64).reshape(5, 5).sin()).Q
    torch.testing.assert_close(half(left @ gradient @ right.T, smoothing=.001), left @ half(gradient, smoothing=.001) @ right.T, rtol=1e-12, atol=1e-12)
    assert half(gradient.half(), smoothing=.001).dtype == torch.float16


@pytest.mark.parametrize('kind', [nn.Conv2d, nn.ConvTranspose2d])
def test_public_convolution_factory_and_exact_resume(kind):
    recipe = get_recipe('bcap', optimizer_family='information_geometry_spectral_half', optimizer_convolution='per_offset', optimizer_smoothing=.001)
    first = kind(4, 6, (2, 3), groups=2, dtype=torch.float64)
    second = deepcopy(first)
    opt1 = recipe.make_generator_optimizer(first)
    opt2 = recipe.make_generator_optimizer(second)
    for p in first.parameters():
        p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape).sin()
    opt1.step()
    second.load_state_dict(first.state_dict())
    opt2.load_state_dict(deepcopy(opt1.state_dict()))
    for p, q in zip(first.parameters(), second.parameters()):
        p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape).cos()
        q.grad = p.grad.clone()
    opt1.step(); opt2.step()
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, rtol=0., atol=0.)
    wrong = deepcopy(opt1.state_dict())
    wrong['dualnorm']['family'] = 'dualnorm'
    with pytest.raises(ValueError, match='family'):
        opt2.load_state_dict(wrong)


def test_bias_and_sampled_prior_are_matched():
    bias1 = nn.Parameter(torch.zeros(3)); prior1 = nn.Parameter(torch.zeros(5, 2))
    bias2 = nn.Parameter(bias1.detach().clone()); prior2 = nn.Parameter(prior1.detach().clone())
    for family, bias, prior in [('dualnorm', bias1, prior1), ('information_geometry_spectral_half', bias2, prior2)]:
        opt = NormalizedOptimizer([{'params':[bias], 'role':'generator'}, {'params':[prior], 'role':'prior'}], family=family, smoothing=.001)
        bias.grad = torch.tensor([.01, .1, .3]); prior.grad = torch.arange(10).float().reshape(5, 2) / 100
        opt.set_sampled_rows(prior, torch.tensor([1, 3])); opt.step()
    torch.testing.assert_close(bias1, bias2, rtol=0., atol=0.)
    torch.testing.assert_close(prior1, prior2, rtol=0., atol=0.)
    assert torch.count_nonzero(prior1[[0, 2, 4]]) == 0
