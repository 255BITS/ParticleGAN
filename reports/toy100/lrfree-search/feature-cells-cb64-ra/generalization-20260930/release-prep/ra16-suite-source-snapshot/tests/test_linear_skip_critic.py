import pytest
import torch

from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor
from benchmarks.transfer_suite.smooth_critic_research import SmoothFourierCritic
from particlegan.grad_regularizers import GradientPenalty


def test_zero_skip_preserves_base_function_but_receives_learning_signal():
    card = next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5')
    torch.manual_seed(0)
    baseline = SmoothFourierCritic(hidden_dim=96, architecture=dict(features='axis', activation='softplus', beta=5.))
    torch.manual_seed(0)
    critic = constructor(card)(hidden_dim=96)
    x = torch.randn(16, 2, generator=torch.Generator().manual_seed(1), requires_grad=True)
    assert torch.equal(baseline(x), critic(x))
    baseline_gradient = torch.autograd.grad(baseline(x).sum(), x)[0]
    assert torch.equal(baseline_gradient, torch.autograd.grad(critic(x).sum(), x)[0])
    critic(x).sum().backward()
    assert critic.skip.weight.grad.norm() > 0


@pytest.mark.parametrize('card', ARCHITECTURES, ids=lambda c: c['name'])
def test_active_cap_backpropagates_through_both_critic_paths(card):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    critic = constructor(card)(hidden_dim=card['hidden'])
    with torch.no_grad():
        critic.main.net[-1].weight.mul_(100.)
        critic.skip.weight.fill_(2.)
    rng = torch.Generator().manual_seed(0)
    real, fake = torch.randn(16, 2, generator=rng), torch.randn(16, 2, generator=rng)
    penalty = GradientPenalty(coeff=3., kappa=.01)(critic, real, fake, step=1)
    assert penalty > 0
    (penalty + critic(real).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert critic.main.net[0].weight.grad.norm() > 0
    assert critic.skip.weight.grad.norm() > 0
