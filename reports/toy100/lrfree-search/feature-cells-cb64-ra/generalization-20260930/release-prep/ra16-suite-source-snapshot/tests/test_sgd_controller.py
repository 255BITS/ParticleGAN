import pytest
import torch

from benchmarks.learned_lr.sgd_study import RawGradientLRAdapter, make_sgd


def test_raw_sgd_analytical_updates_have_no_momentum_or_coordinate_scaling():
    parameter = torch.nn.Parameter(torch.tensor([2., -3.], dtype=torch.float64))
    optimizer = make_sgd([parameter], .2)
    for gradient in (torch.tensor([.1, 100.], dtype=torch.float64), torch.tensor([-.7, 0.], dtype=torch.float64)):
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()
        optimizer.step()
        assert torch.allclose(parameter, before - .2 * gradient, atol=1e-14, rtol=0)
        assert optimizer.state == {}


def test_learned_policy_only_scales_the_raw_gradient_update():
    parameter = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    optimizer = make_sgd([parameter], .1)
    adapter = RawGradientLRAdapter([[.5, 0, .2, 0, 0]] * 2, 4, interval=1)
    for step in range(4):
        parameter.grad = torch.tensor([.03 * (step + 1), -4.], dtype=torch.float64)
        before, grad, rng = parameter.detach().clone(), parameter.grad.clone(), torch.get_rng_state().clone()
        multiplier = adapter.step(optimizer, step, role="g")
        assert torch.equal(parameter.grad, grad)
        assert torch.equal(torch.get_rng_state(), rng)
        optimizer.step()
        assert .05 <= multiplier <= 4.
        assert torch.allclose(parameter, before - .1 * multiplier * grad, atol=1e-14, rtol=0)
        assert optimizer.state == {}


def test_raw_gradient_controller_rejects_hidden_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    parameter.grad = torch.tensor([2.])
    optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
    adapter = RawGradientLRAdapter(torch.zeros(2, 5), 2)
    with pytest.raises(ValueError, match="momentum-free"):
        adapter.step(optimizer, 0, role="d")


def test_per_tensor_rule_uses_exact_raw_gradients_with_distinct_learning_rates():
    from benchmarks.learned_lr.relative_sgd_study import RelativeStepController, grouped_sgd
    x = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2., -2.], dtype=torch.float64))
    x.grad, y.grad = torch.tensor([.01, -.01], dtype=torch.float64), torch.tensor([1., -1.], dtype=torch.float64)
    before_x, before_y = x.detach().clone(), y.detach().clone()
    optimizer = grouped_sgd([x,y], .1)
    controller = RelativeStepController(None, 2, g_fraction=.003, d_fraction=.001)
    controller.step(optimizer, 0, role="g")
    rates = [g["lr"] for g in optimizer.param_groups]
    assert rates == pytest.approx([.3, .006])
    optimizer.step()
    assert torch.allclose(x, before_x - .3 * x.grad, atol=1e-14, rtol=0)
    assert torch.allclose(y, before_y - .006 * y.grad, atol=1e-14, rtol=0)
    assert optimizer.state == {}


def test_running_rms_uses_only_past_and_current_gradients_without_momentum():
    from benchmarks.learned_lr.relative_sgd_study import grouped_sgd
    from benchmarks.learned_lr.relative_sgd_followup import SmoothedRelativeController
    x = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    optimizer = grouped_sgd([x], .1)
    controller = SmoothedRelativeController(None, 2, g_fraction=.003, d_fraction=.001, rms_beta=.9)
    x.grad = torch.tensor([1., -1.], dtype=torch.float64)
    controller.step(optimizer, 0, role="g")
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.003)
    optimizer.step()
    before = x.detach().clone()
    x.grad = torch.tensor([.1, -.1], dtype=torch.float64)
    controller.step(optimizer, 1, role="g")
    expected_lr = .003 * .997 / (.9 * 1. + .1 * .01) ** .5
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    optimizer.step()
    assert torch.allclose(x, before - expected_lr * x.grad, atol=1e-14, rtol=0)
    assert optimizer.state == {}
