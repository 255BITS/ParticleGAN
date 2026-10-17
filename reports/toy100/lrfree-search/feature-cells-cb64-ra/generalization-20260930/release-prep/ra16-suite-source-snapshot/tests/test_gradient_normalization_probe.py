"""Derivative and route tests for the scratch GN formulation."""

import pytest
import torch
from torch import nn

from particlegan import BatchDistanceDiscriminator
from benchmarks.legacy.grad_regularizers import GradientPenalty
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100.gradient_normalization_probe import (
    GNReceipt, GradientNormalizedCritic, normalized_score,
)
from reports.toy100.gradient_normalization_screen import _check_gn_receipt, SCRATCH_SELECTOR


def test_linear_input_and_parameter_derivatives_include_denominator():
    base = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        base.weight.fill_(2.)
    receipt = GNReceipt()
    critic = GradientNormalizedCritic(base, data_index=0, receipt=receipt)
    x = torch.tensor([[1.]], requires_grad=True)
    out = critic(x)
    grad_x, grad_w = torch.autograd.grad(out.sum(), (x, base.weight))
    assert out.item() == pytest.approx(.5, abs=2e-8)
    # f=2x, ||grad_x f||=2, so d(2x/(2+2x))/dx at x=1 is 1/4.
    assert grad_x.item() == pytest.approx(.25, rel=1e-6)
    # f=w*x and ||grad_x f||=w: at x=1, fhat is 1/2 independent of w.
    assert grad_w.item() == pytest.approx(0., abs=1e-7)
    assert receipt.grad_enabled_calls == 1
    assert receipt.as_dict()["denominator_detached"] is False


def test_conditional_score_normalizes_only_generated_data_argument():
    class Conditional(nn.Module):
        def forward(self, slow, fast):
            return (2 * slow + 3 * fast).sum(dim=1)

    receipt = GNReceipt()
    critic = GradientNormalizedCritic(Conditional(), data_index=1, receipt=receipt)
    slow = torch.tensor([[1.]])
    fast = torch.tensor([[1.]], requires_grad=True)
    score = critic(slow, fast)
    grad = torch.autograd.grad(score.sum(), fast)[0]
    assert score.item() == pytest.approx(5 / 8, rel=1e-6)
    assert grad.item() == pytest.approx(9 / 64, rel=1e-6)
    assert receipt.as_dict()["data_indices"] == {"1": 1}


def test_generator_gradient_and_no_grad_evaluation_are_both_valid():
    generator = nn.Linear(1, 1, bias=False)
    critic_raw = nn.Linear(1, 1)
    with torch.no_grad():
        generator.weight.fill_(1.)
        critic_raw.weight.fill_(2.)
        critic_raw.bias.fill_(1.)
    receipt = GNReceipt()
    critic = GradientNormalizedCritic(critic_raw, data_index=0, receipt=receipt)
    fake = generator(torch.tensor([[1.]]))
    gradient = torch.autograd.grad(critic(fake).sum(), generator.weight)[0]
    # f=3, ||grad_x f||=2: d(f/(2+f))/dx = 2*2/25.
    assert gradient.item() == pytest.approx(.16, rel=1e-6)
    with torch.no_grad():
        evaluated = critic(torch.tensor([[1.]]))
    assert evaluated.item() == pytest.approx(3 / 5, rel=1e-6)
    assert not evaluated.requires_grad
    assert receipt.grad_enabled_calls == 1 and receipt.no_grad_calls == 1


def test_no_penalty_and_batch_coupled_critic_boundary():
    assert GradientPenalty(arm="f_none", coeff=0., kappa=0.).arm == "f_none"
    with pytest.raises(ValueError, match="exact per-row Jacobian"):
        GradientNormalizedCritic(BatchDistanceDiscriminator(), data_index=0,
                                 receipt=GNReceipt())


def test_dtype_epsilon_makes_zero_over_zero_finite():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    out = normalized_score(model, (torch.zeros(2, 1),), data_index=0,
                           receipt=GNReceipt(), route="zero_case")
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isfinite(out).all()


def test_saved_scratch_config_remains_unsupported_after_stamp_removal():
    # The evidence config carries a second marker in addition to the protocol
    # stamp. Removing only the protocol stamp cannot make ordinary regrade
    # resolve it as an ordinary shared recipe.
    with pytest.raises(ValueError):
        declared_recipe({"scratch_discriminator_policy": SCRATCH_SELECTOR})


def test_application_receipt_rejects_missing_data_route_or_calls():
    good = dict(calls=801, grad_enabled_calls=800, no_grad_calls=1,
                data_elements=10000, data_indices={"1": 801},
                routes={"wrapped_critic": 801}, denominator_min=.1,
                denominator_max=1.2, denominator_detached=False,
                formula="f/(||grad_data f||_2+abs(f)+finfo(dtype).eps)")
    _check_gn_receipt(good, task="trajectory", steps=400)
    with pytest.raises(ValueError, match="did not normalize"):
        _check_gn_receipt(good | {"grad_enabled_calls": 0}, task="trajectory", steps=400)
    with pytest.raises(ValueError, match="did not normalize"):
        _check_gn_receipt(good | {"data_indices": {"0": 801}}, task="trajectory", steps=400)
