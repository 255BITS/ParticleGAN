"""Derivative checks for the separate virtual-critic b_cap helper."""

import torch

from particlegan.grad_regularizers import GradRegularizer
from reports.toy100.functional_b_cap import functional_b_cap


def test_native_value_d_gradient_and_nonzero_fake_to_d_mixed_derivative():
    torch.set_num_threads(1)
    regularizer = GradRegularizer(arm='b_cap', coeff=1., kappa=1.)
    real = torch.zeros((2, 1))
    fake = torch.ones((2, 1), requires_grad=True)
    slope = torch.tensor(1.5, requires_grad=True)
    critic = lambda x: slope * x.square()

    original = regularizer(critic, real, fake, step=7)
    functional = functional_b_cap(regularizer, critic, real, fake, step=7)
    original_d = torch.autograd.grad(original, slope, create_graph=True)[0]
    functional_d = torch.autograd.grad(functional, slope, create_graph=True)[0]
    assert torch.equal(original.detach(), functional.detach())
    assert torch.equal(original_d, functional_d.detach())

    mixed = torch.autograd.grad(functional_d, fake)[0]
    assert float(mixed.abs().max()) > 1.
    # The original cap detaches fake coordinates. A virtual D step using it
    # would miss the same nonzero mixed derivative.
    assert torch.autograd.grad(original_d, fake, allow_unused=True) == (None,)

    h = 1e-3
    def d_at(position):
        loss = functional_b_cap(regularizer, critic, real, position, step=7)
        return torch.autograd.grad(loss, slope)[0].detach()
    fd = (d_at(fake.detach() + h) - d_at(fake.detach() - h)) / (2 * h)
    assert torch.allclose(mixed.sum(), fd, rtol=2e-3, atol=2e-3)


def test_inactive_cap_has_zero_mixed_term_and_bilinear_virtual_step():
    regularizer = GradRegularizer(arm='b_cap', coeff=1., kappa=1.)
    slope = torch.tensor(.25, requires_grad=True)
    fake = torch.tensor([[.3]], requires_grad=True)
    real = torch.zeros_like(fake)
    critic = lambda x: slope * x
    cap = functional_b_cap(regularizer, critic, real, fake, step=1)
    cap_d = torch.autograd.grad(cap, slope, create_graph=True)[0]
    assert cap.item() == 0. and cap_d.item() == 0.
    mixed_cap = torch.autograd.grad(cap_d, fake, allow_unused=True)[0]
    assert mixed_cap is None or mixed_cap.item() == 0.

    # For L_D(a, x)=a*x, one fixed-metric virtual D step has da'/dx=-P.
    metric = torch.tensor(.125)
    d_gradient = torch.autograd.grad(slope * fake.sum() + cap, slope,
                                     create_graph=True)[0]
    virtual_slope = slope - metric * d_gradient
    mixed = torch.autograd.grad(virtual_slope, fake)[0]
    assert torch.allclose(mixed, torch.full_like(fake, -.125), atol=1e-7)
