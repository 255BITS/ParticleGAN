from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite.smooth_critic_research import ARCHITECTURES, constructor
from lib.toy_models import SimpleMLPDiscriminator
from benchmarks.legacy.grad_regularizers import GradientPenalty


@pytest.mark.parametrize('card', ARCHITECTURES, ids=lambda c: c['name'])
def test_smooth_critic_supports_active_gradient_penalty_backward(card):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    critic = constructor(card)(2, 64, 2, 2)
    with torch.no_grad():
        critic.net[-1].weight.mul_(100.)
    rng = torch.Generator().manual_seed(0)
    real = torch.randn(16, 2, generator=rng)
    fake = torch.randn(16, 2, generator=rng, requires_grad=True)
    penalty = GradientPenalty('b_cap', coeff=3., kappa=.01)(
        critic, real, fake.detach(), step=1, generator=rng)
    assert penalty > 0  # Exercise the second-derivative path, not an inactive cap.
    (penalty + critic(fake).mean()).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert critic.net[0].weight.grad.norm() > 0


def test_axis_variant_preserves_original_features_and_linear_initialization():
    card = next(c for c in ARCHITECTURES if c['name'] == 'axis_softplus5')
    torch.manual_seed(0)
    original = SimpleMLPDiscriminator(2, 64, 2, 2)
    torch.manual_seed(0)
    smooth = constructor(card)(2, 64, 2, 2)
    assert all(torch.equal(a, b) for a, b in zip(original.parameters(), smooth.parameters()))
    points = torch.randn(16, 2, generator=torch.Generator().manual_seed(0))
    phase = (points.unsqueeze(-1) * original.freqs).flatten(1)
    assert torch.equal(smooth.encode(points), torch.cat([points, phase.sin(), phase.cos()], 1))
    assert sum(p.numel() for p in smooth.parameters()) == 4929


def test_oriented_features_do_not_consume_global_initialization_rng():
    axis = next(c for c in ARCHITECTURES if c['name'] == 'axis_silu')
    oriented = next(c for c in ARCHITECTURES if c['name'] == 'oriented4_silu')
    torch.manual_seed(0)
    a = constructor(axis)(2, 64, 2, 2)
    state_a = torch.get_rng_state().clone()
    torch.manual_seed(0)
    b = constructor(oriented)(2, 64, 2, 2)
    assert torch.equal(state_a, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert torch.allclose(b.projection.norm(dim=1) / torch.pi, torch.tensor(oriented['radial_bands']))
    restored = constructor(oriented)(2, 64, 2, 2)
    restored.load_state_dict(deepcopy(b.state_dict()))
    points = torch.randn(12, 2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(restored(points), b(points))
