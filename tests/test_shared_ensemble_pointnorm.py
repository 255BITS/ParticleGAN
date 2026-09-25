"""Pointwise and second-order gradient checks for the new D architectures."""
import pytest
import torch
from torch import nn

from benchmarks.legacy.grad_regularizers import GradientPenalty
from benchmarks.transfer_suite.shared_ensemble_research import (
    ARCHITECTURES as ENSEMBLES, constructor as ensemble_constructor,
)
from benchmarks.transfer_suite.shared_pointnorm_research import (
    ARCHITECTURES as POINTNORMS, constructor as pointnorm_constructor,
)


CARDS = [(card, ensemble_constructor) for card in ENSEMBLES] + [
    (card, pointnorm_constructor) for card in POINTNORMS
]


@pytest.mark.parametrize('card,constructor', CARDS, ids=[card['name'] for card, _ in CARDS])
def test_pointwise_output_and_active_bcap_double_backward(card, constructor):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    width = card.get('width')
    layers = card.get('layers')
    fourier = card.get('fourier', len(card.get('frequency_multipliers', [])))
    critic = constructor(card)(2, width, layers, fourier)
    x = torch.randn(8, 2, requires_grad=True)
    together = critic(x)[0]
    alone = critic(x[:1])[0]
    torch.testing.assert_close(together, alone, atol=1e-6, rtol=1e-6)
    full_gradient = torch.autograd.grad(together, x, retain_graph=True)[0][0]
    single_gradient = torch.autograd.grad(alone, x, retain_graph=True)[0][0]
    torch.testing.assert_close(full_gradient, single_gradient, atol=1e-6, rtol=1e-6)

    class Scaled(nn.Module):
        def forward(self, points):
            return 10000 * critic(points)

    real = torch.randn(8, 2)
    fake = torch.randn(8, 2) + 1
    cap = GradientPenalty('b_cap', coeff=6., kappa=1.25)(
        Scaled(), real, fake, step=1, generator=torch.Generator().manual_seed(2))
    assert cap.item() > 0
    (critic(real).mean() + cap).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
