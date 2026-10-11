"""Analytic software checks for a prior-only DualNorm response law."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim import dualnorm
from particlegan.optim.dualnorm import NormalizedOptimizer


def assert_packet_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_packet_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_packet_equal(a, b)
    else:
        assert left == right


def make_prior(value=None, *, smoothing=.001, rate=.03, dtype=torch.float64):
    prior = nn.Parameter(torch.zeros(4, 2, dtype=dtype))
    optimizer = NormalizedOptimizer([dict(params=[prior], role="prior")],
        smoothing=smoothing, prior_smoothing=value, lr=rate)
    return prior, optimizer


def test_prior_response_damps_small_forces_and_preserves_large_forces():
    prior, optimizer = make_prior(.01)
    prior.grad = prior.new_tensor([[.001, 0.], [.1, 0.], [0., 0.], [3., 4.]])
    optimizer.set_sampled_rows(prior, torch.tensor([0, 1, 2]))
    optimizer.step()
    expected = prior.new_tensor([[-.03 * .001 / math.hypot(.001, .01), 0.],
        [-.03 * .1 / math.hypot(.1, .01), 0.], [0., 0.], [0., 0.]])
    torch.testing.assert_close(prior, expected, rtol=0, atol=1e-17)
    assert abs(prior[0, 0]) < .0031 and abs(prior[1, 0]) > .0298
    stats = optimizer.prior_smoothing_stats
    assert stats["steps"] == 1 and stats["sampled_rows"] == 3
    assert stats["gradient_squared_sum"] == pytest.approx(.001**2 + .1**2)
    assert stats["proposed_displacement_squared_sum"] == pytest.approx(float(expected.square().sum()))
    assert stats["optimizer_displacement_squared_sum"] == pytest.approx(float(expected.square().sum()))


@pytest.mark.parametrize("smoothing", [0., .001])
def test_none_inherits_original_row_rule_and_checkpoint_packet(smoothing):
    prior, optimizer = make_prior(smoothing=smoothing, dtype=torch.float32)
    prior.grad = prior.new_tensor([[.001, .002], [.1, .2], [0., 0.], [3., 4.]])
    selected = torch.tensor([0, 2, 3])
    norm = prior.grad[selected].norm(dim=1, keepdim=True)
    denominator = (torch.hypot(norm, torch.full_like(norm, smoothing))
                   if smoothing else norm + 1e-8)
    expected = prior.detach().clone()
    expected.index_add_(0, selected, prior.grad[selected] / denominator, alpha=-.03)
    optimizer.set_sampled_rows(prior, selected)
    optimizer.step()
    assert torch.equal(prior, expected)
    assert not hasattr(optimizer, "prior_smoothing_stats")
    packet = deepcopy(optimizer.state_dict())
    assert "prior_smoothing" not in packet["dualnorm"]
    assert "prior_smoothing_stats" not in packet["dualnorm"]
    other_prior, other = make_prior(smoothing=smoothing, dtype=torch.float32)
    other.load_state_dict(packet)
    assert_packet_equal(other.state_dict(), packet)
    assert other.prior_smoothing is None


def test_smoothing_changes_only_sampled_prior_rows_not_network_or_critic():
    def build(value):
        prior = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
        weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        bias = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        critic = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        optimizer = NormalizedOptimizer([
            dict(params=[prior], role="prior"), dict(params=[weight, bias], role="generator"),
            dict(params=[critic], role="critic")], smoothing=.001, prior_smoothing=value, lr=.03)
        prior.grad = prior.new_full(prior.shape, .001)
        weight.grad = torch.diag(weight.new_tensor([.001, .1]))
        bias.grad = bias.new_tensor([.001, .1])
        critic.grad = critic.new_tensor([[.01, .02], [.03, .04]])
        optimizer.set_sampled_rows(prior, torch.tensor([1, 1]))
        optimizer.set_sampled_rows(prior, torch.tensor([3]))
        optimizer.step()
        return prior, weight, bias, critic, optimizer

    baseline, candidate = build(None), build(.01)
    assert not torch.equal(baseline[0], candidate[0])
    for a, b in zip(baseline[1:4], candidate[1:4]):
        assert torch.equal(a, b)
    assert torch.count_nonzero(candidate[0][[0, 2]]) == 0
    assert candidate[4].prior_smoothing_stats["sampled_rows"] == 2


def test_zero_override_uses_original_eps_normalization():
    prior, optimizer = make_prior(0., smoothing=3.)
    prior.grad = prior.new_tensor([[0., 4.], [0., 0.], [0., 0.], [0., 0.]])
    optimizer.set_sampled_rows(prior, torch.tensor([0, 1]))
    optimizer.step()
    assert float(prior[0, 1].detach()) == pytest.approx(-.03 * 4 / (4 + 1e-8))
    assert torch.count_nonzero(prior[1:]) == 0
    assert optimizer.state_dict()["dualnorm"]["prior_smoothing"] == 0.


def test_checkpoint_resumes_pending_sampled_rows_and_cumulative_statistics():
    prior, optimizer = make_prior(.01)
    prior.grad = prior.new_full(prior.shape, .002)
    optimizer.set_sampled_rows(prior, torch.tensor([0, 1]))
    optimizer.step()
    optimizer.set_sampled_rows(prior, torch.tensor([2, 3]))
    packet = deepcopy(optimizer.state_dict())
    restored_prior, restored = make_prior(.01)
    with torch.no_grad():
        restored_prior.copy_(prior)
    restored.load_state_dict(packet)
    assert_packet_equal(restored.state_dict(), packet)
    prior.grad = prior.new_full(prior.shape, .003)
    restored_prior.grad = prior.grad.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(prior, restored_prior)
    assert_packet_equal(optimizer.state_dict(), restored.state_dict())
    assert restored.prior_smoothing_stats["steps"] == 2
    assert restored.prior_smoothing_stats["sampled_rows"] == 4


def test_motion_statistics_distinguish_proposal_from_float32_rounding():
    prior, optimizer = make_prior(.01, dtype=torch.float32)
    with torch.no_grad():
        prior.fill_(1e8)
    prior.grad = prior.new_full(prior.shape, .01)
    optimizer.set_sampled_rows(prior, torch.tensor([0, 1]))
    optimizer.step()
    stats = optimizer.prior_smoothing_stats
    assert stats["proposed_displacement_squared_sum"] > 0
    assert stats["optimizer_displacement_squared_sum"] == 0


def test_checkpoint_prior_scale_rejects_boolean_at_zero():
    _, optimizer = make_prior(0.)
    packet = deepcopy(optimizer.state_dict())
    packet["dualnorm"]["prior_smoothing"] = False
    with pytest.raises(ValueError, match="prior smoothing"):
        optimizer.load_state_dict(packet)


@pytest.mark.parametrize("mutation", ["scale", "missing", "nan", "negative", "boolean", "extra"])
def test_bad_checkpoint_rejects_before_optimizer_state_or_ownership_changes(mutation):
    prior, optimizer = make_prior(.01)
    prior.grad = prior.new_full(prior.shape, .002)
    optimizer.set_sampled_rows(prior, torch.tensor([0]))
    optimizer.step()
    optimizer.set_sampled_rows(prior, torch.tensor([1]))
    before = deepcopy(optimizer.state_dict())
    invalid = deepcopy(before)
    meta = invalid["dualnorm"]
    if mutation == "scale":
        meta["prior_smoothing"] = .02
    elif mutation == "missing":
        meta.pop("prior_smoothing_stats")
    elif mutation == "nan":
        meta["prior_smoothing_stats"]["gradient_squared_sum"] = float("nan")
    elif mutation == "negative":
        meta["prior_smoothing_stats"]["optimizer_displacement_squared_sum"] = -1.
    elif mutation == "boolean":
        meta["prior_smoothing_stats"]["steps"] = True
    else:
        meta["prior_smoothing_stats"]["extra"] = 0
    with pytest.raises(ValueError):
        optimizer.load_state_dict(invalid)
    assert_packet_equal(optimizer.state_dict(), before)


def test_changed_recipe_and_old_packet_cannot_exchange_enabled_state():
    _, baseline = make_prior()
    _, enabled = make_prior(.01)
    _, changed = make_prior(.02)
    for target, source in ((baseline, enabled), (enabled, baseline), (changed, enabled)):
        before = deepcopy(target.state_dict())
        with pytest.raises(ValueError):
            target.load_state_dict(source.state_dict())
        assert_packet_equal(target.state_dict(), before)


@pytest.mark.parametrize("value", [-1., float("inf"), float("nan"), True, "0.01"])
def test_invalid_smoothing_values_fail_at_construction(value):
    with pytest.raises(ValueError, match="prior smoothing"):
        make_prior(value)


@pytest.mark.parametrize("family", [f for f in dualnorm.NORMALIZED_FAMILIES if f != "dualnorm"])
def test_prior_override_rejects_other_optimizer_families_even_at_zero(family):
    with pytest.raises(ValueError, match="prior smoothing"):
        NormalizedOptimizer([nn.Parameter(torch.ones(2, 2))], family=family, prior_smoothing=0.)


@pytest.mark.parametrize("value", [None, .01])
def test_factory_omits_disabled_constructor_argument_and_forwards_enabled_value(monkeypatch, value):
    captured = {}

    def capture(params, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(dualnorm, "NormalizedOptimizer", capture)
    recipe = get_recipe('bcap', constraint_geometry_mode='none',
                        optimizer_convolution='none', optimizer_prior_smoothing=value)
    dualnorm.make_normalized_optimizer(recipe, [nn.Parameter(torch.ones(2, 2))])
    if value is None:
        assert "prior_smoothing" not in captured
    else:
        assert captured["prior_smoothing"] == value
