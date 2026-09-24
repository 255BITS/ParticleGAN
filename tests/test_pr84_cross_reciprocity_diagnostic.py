"""Sign and null-control checks for the read-only cross-field probe."""

import math

import torch


def _scalar_field(d, g, *, zero_sum_g=False):
    d = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(g, dtype=torch.float64, requires_grad=True)
    advantage = d * (1 - g)
    critic_loss = torch.nn.functional.softplus(-advantage)
    generator_loss = (-critic_loss if zero_sum_g
                      else torch.nn.functional.softplus(advantage))
    return (torch.autograd.grad(critic_loss, d, retain_graph=True)[0].item(),
            torch.autograd.grad(generator_loss, g)[0].item())


def test_relativistic_logistic_cross_blocks_are_not_zero_sum():
    h = 1e-5
    d, g = 1., 0.  # Delta = d * (real - fake) = 1.
    d_cross = (_scalar_field(d, g + h)[0] - _scalar_field(d, g - h)[0]) / (2*h)
    g_cross = (_scalar_field(d + h, g)[1] - _scalar_field(d - h, g)[1]) / (2*h)
    s = 1 / (1 + math.exp(-1))
    analytic = 1 - 2*s - 2*s*(1-s)
    assert abs(d_cross + g_cross - analytic) < 1e-9
    assert abs(analytic + .855341023743) < 1e-12


def test_true_zero_sum_cross_control_cancels():
    h = 1e-5
    d, g = 1., 0.
    d_cross = (_scalar_field(d, g + h)[0] - _scalar_field(d, g - h)[0]) / (2*h)
    g_cross = (_scalar_field(d + h, g, zero_sum_g=True)[1]
               - _scalar_field(d - h, g, zero_sum_g=True)[1]) / (2*h)
    assert abs(d_cross + g_cross) < 1e-9
