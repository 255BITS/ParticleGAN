"""Independent numerical checks for the proposed confidence G update."""

import pytest
import torch

from particlegan.gan_loss import GANLoss
from reports.toy100.confidence_dynamics_scratch import (
    ConfidenceDynamics, warm_confidence,
)


def _optimizer():
    network = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    prior = torch.nn.Parameter(torch.tensor([.5, -.75]))
    optimizer = torch.optim.Adam([
        dict(params=[network], lr=.1, _comparison_prior=False),
        dict(params=[prior], lr=.2, _comparison_prior=True),
    ], betas=(0., .9))
    network.grad = torch.tensor([.3, -.4])
    prior.grad = torch.tensor([-.2, .6])
    return network, prior, optimizer


def _same_moments(left, right):
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(left[name], right[name])


def test_confidence_scales_only_network_proposal_with_full_adam_moments():
    network, prior, optimizer = _optimizer()
    reference_network, reference_prior, reference_optimizer = _optimizer()
    logits_real = torch.tensor([2., 0., -1.])
    logits_fake = torch.zeros_like(logits_real)
    expected_advantage = float(2 * torch.sigmoid(logits_real - logits_fake).mean() - 1)
    expected_scale = max(.02, min(1., expected_advantage / .5))
    assert .02 < expected_scale < 1.
    original_network = network.detach().clone()
    rng_before = torch.get_rng_state().clone()

    controller = ConfidenceDynamics(threshold=.5)
    controller.observe(logits_fake, logits_real)
    controller.step(optimizer, torch.optim.Adam.step)
    reference_optimizer.step()

    assert torch.equal(network, torch.lerp(original_network, reference_network,
                                          expected_scale))
    assert torch.equal(prior, reference_prior)
    _same_moments(optimizer.state[network], reference_optimizer.state[reference_network])
    _same_moments(optimizer.state[prior], reference_optimizer.state[reference_prior])
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert controller.receipt["updates"][0]["scale"] == pytest.approx(expected_scale)
    assert controller.pending is None
    with pytest.raises(RuntimeError, match="exactly one observed loss"):
        controller.step(optimizer, torch.optim.Adam.step)


def test_observe_only_is_bitwise_ordinary_adam_and_context_restores_loss():
    network, prior, optimizer = _optimizer()
    reference_network, reference_prior, reference_optimizer = _optimizer()
    original_loss = GANLoss.g_loss
    delegate = {}
    state = dict(opt_g=optimizer, base_adam_step=torch.optim.Adam.step,
                 set_step_delegate=lambda fn: delegate.update(step=fn))
    gan = GANLoss(loss_type="logistic", mode="rp")
    fake = torch.tensor([-1., .25])
    real = torch.tensor([.4, .8])
    expected_loss = gan.g_loss(fake, real)

    with warm_confidence(state, threshold=.25, observe_only=True) as receipt:
        assert GANLoss.g_loss is not original_loss
        actual_loss = gan.g_loss(fake, real)
        assert torch.equal(actual_loss, expected_loss)
        delegate["step"](optimizer)
        assert receipt["updates"][0]["scale"] == 1.

    reference_optimizer.step()
    assert GANLoss.g_loss is original_loss
    assert delegate["step"] is torch.optim.Adam.step
    assert torch.equal(network, reference_network)
    assert torch.equal(prior, reference_prior)
    _same_moments(optimizer.state[network], reference_optimizer.state[reference_network])
    _same_moments(optimizer.state[prior], reference_optimizer.state[reference_prior])


def test_negative_confidence_uses_positive_floor_and_requires_prior_group():
    network, prior, optimizer = _optimizer()
    reference_network, _, reference_optimizer = _optimizer()
    controller = ConfidenceDynamics(threshold=.25, floor=.02)
    controller.observe(torch.tensor([5.]), torch.tensor([-5.]))
    original = network.detach().clone()
    controller.step(optimizer, torch.optim.Adam.step)
    reference_optimizer.step()
    assert torch.equal(network, torch.lerp(original, reference_network, .02))
    assert controller.receipt["updates"][0]["scale"] == .02

    no_prior = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.]))], lr=.1)
    no_prior.param_groups[0]["params"][0].grad = torch.tensor([.2])
    controller.observe(torch.tensor([0.]), torch.tensor([1.]))
    with pytest.raises(RuntimeError, match="separate learned-prior group"):
        controller.step(no_prior, torch.optim.Adam.step)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_control_parameters_are_rejected(value):
    with pytest.raises(ValueError):
        ConfidenceDynamics(threshold=value)
    with pytest.raises(ValueError):
        ConfidenceDynamics(floor=value)
