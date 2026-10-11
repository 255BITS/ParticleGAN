"""Analytic BCAP damping contracts; these software checks add no training."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.grad_regularizers import GradientPenalty


class _LinearCritic(nn.Module):
    def __init__(self, weights, dtype=torch.float64, spatial=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weights, dtype=dtype))
        self.spatial = spatial
        self.calls = 0

    def forward(self, x, gain=1.0):
        self.calls += 1
        score = gain * (x.flatten(1) * self.weight).sum(1)
        # A constant logit map must retain the per-image slope units.
        return score[:, None, None].expand(-1, 2, 3) if self.spatial else score


def _panels(dtype=torch.float64):
    real = torch.tensor([[-2., 1.], [.4, .2], [1., 3.]], dtype=dtype,
                        requires_grad=True)
    fake = torch.tensor([[.1, -.3], [.7, .5]], dtype=dtype,
                        requires_grad=True)
    return real, fake


@pytest.mark.parametrize("spatial", [False, True])
def test_subcap_damping_has_exact_nonzero_parameter_gradient(spatial):
    critic = _LinearCritic([.3, .4], spatial=spatial)
    real, fake = _panels()
    value, stats = GradientPenalty(arm="b_cap", zero_coeff=.05).penalty(
        critic, real, fake)
    expected = .05 * critic.weight.square().sum()
    torch.testing.assert_close(value, expected, rtol=0, atol=1e-15)
    value.backward()
    torch.testing.assert_close(critic.weight.grad, .1 * critic.weight, rtol=0, atol=1e-15)
    assert stats["cap_pen"] == 0.0
    assert stats["zero_pen"] == pytest.approx(.0125)
    assert stats["pen"] == stats["zero_pen"]
    assert critic.calls == 2  # Both losses reuse one forward per panel.
    assert real.grad is None and fake.grad is None


def test_above_cap_adds_exact_damping_to_existing_loss_and_gradient():
    reference = _LinearCritic([1.2, 1.6])
    candidate = deepcopy(reference)
    real, fake = _panels()
    base = GradientPenalty(arm="b_cap", coeff=.7)(reference, real, fake)
    actual, stats = GradientPenalty(arm="b_cap", coeff=.7, zero_coeff=.05).penalty(
        candidate, real, fake)
    expected_zero = .05 * candidate.weight.square().sum()
    assert torch.equal(actual, base + expected_zero)
    base.backward()
    actual.backward()
    torch.testing.assert_close(candidate.weight.grad,
                               reference.weight.grad + .1 * candidate.weight,
                               rtol=0, atol=1e-15)
    assert stats["cap_pen"] == float(base.detach())
    assert stats["zero_pen"] == float(expected_zero.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("arm", ["k3p", "a_r1r2", "b_cap"])
def test_explicit_zero_preserves_values_gradients_stats_and_state(dtype, arm):
    reference = _LinearCritic([1.2, -.7], dtype=dtype)
    candidate = deepcopy(reference)
    real, fake = _panels(dtype)
    base = GradientPenalty(arm=arm, coeff=1.7, kappa=.8)
    disabled = GradientPenalty(arm=arm, coeff=1.7, kappa=.8, zero_coeff=0.)
    expected, expected_stats = base.penalty(reference, real, fake)
    actual, actual_stats = disabled.penalty(candidate, real, fake)
    assert torch.equal(actual, expected)
    assert actual_stats == expected_stats
    assert "cap_pen" not in actual_stats and "zero_pen" not in actual_stats
    assert disabled.state_dict() == base.state_dict()
    expected.backward()
    actual.backward()
    assert torch.equal(reference.weight.grad, candidate.weight.grad)


def test_lazy_interval_scales_both_terms_and_skips_without_forward():
    critic = _LinearCritic([1.2, 1.6])
    real, fake = _panels()
    lazy = GradientPenalty(arm="b_cap", coeff=.7, zero_coeff=.05, lazy_k=4)
    skipped, skipped_stats = lazy.penalty(critic, real, fake, step=3)
    assert skipped.item() == 0 and not skipped.requires_grad
    assert skipped_stats == {"applied": False, "pen": 0.0}
    assert critic.calls == 0 and lazy.record.calls == 0
    value, stats = lazy.penalty(critic, real, fake, step=4)
    base, base_stats = GradientPenalty(arm="b_cap", coeff=.7, zero_coeff=.05).penalty(
        critic, real, fake)
    assert torch.equal(value, 4 * base)
    assert stats["cap_pen"] == 4 * base_stats["cap_pen"]
    assert stats["zero_pen"] == 4 * base_stats["zero_pen"]
    assert lazy.record.calls == 1


def test_zero_slopes_have_no_epsilon_damping_loss():
    critic = _LinearCritic([0., 0.])
    real, fake = _panels()
    penalty = GradientPenalty(arm="b_cap", zero_coeff=.05)
    value, stats = penalty.penalty(critic, real, fake)
    assert value.item() == stats["zero_pen"] == 0.0
    value.backward()
    assert torch.equal(critic.weight.grad, torch.zeros_like(critic.weight))
    silent, no_stats = penalty.penalty(critic, real, fake, collect_stats=False)
    assert silent.item() == 0 and no_stats == {}


def test_public_penalty_factory_preserves_conditioning_and_panel_ownership():
    recipe = get_recipe("bcap")
    critic = _LinearCritic([.3, .4])
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True, zero_coeff=.05)
    real, fake = _panels()
    value = penalty(critic, real, fake, gain=.5)
    value.backward()
    torch.testing.assert_close(critic.weight.grad, .025 * critic.weight, rtol=0, atol=1e-15)
    assert penalty.last_stats["cap_pen"] == 0.0
    assert penalty.last_stats["zero_pen"] == pytest.approx(.003125)
    assert optimizer.record.calls == 1
    assert real.grad is None and fake.grad is None


@pytest.mark.parametrize("value", [-.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_damping_coefficients_fail_before_execution(value):
    with pytest.raises(ValueError, match="zero_coeff.*finite and nonnegative"):
        GradientPenalty(arm="b_cap", zero_coeff=value)


@pytest.mark.parametrize("arm", ["k3p", "a_r1r2"])
def test_damping_cannot_silently_change_another_penalty_arm(arm):
    with pytest.raises(ValueError, match="zero_coeff requires arm='b_cap'"):
        GradientPenalty(arm=arm, zero_coeff=.05)
