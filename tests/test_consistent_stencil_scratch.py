"""Independent stencil derivatives and original-host replay accounting."""
import pytest
import torch

from reports.toy100.alternating_curvature_scratch import alternating_curvature
from reports.toy100.consistent_stencil_scratch import (
    ConsistentStencilRecorder, consistent_stencil, stencil)
from tests.test_alternating_curvature_scratch import _host


def test_stencil_matches_quadratic_value_and_both_player_derivatives():
    # In 2D the five-point average adds 4*w^2/5 to ||x||^2.
    x = torch.tensor([[.7, -.3], [.2, 1.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    width = .15
    actual = stencil(lambda model, v: model * v.square().sum(-1), a, x, width)
    expected = a * (x.square().sum(-1) + 4 * width ** 2 / 5)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    ax, aa = torch.autograd.grad(actual.sum(), (x, a), create_graph=True)
    assert torch.allclose(ax, 2 * a * x, atol=1e-14, rtol=1e-14)
    assert torch.allclose(aa, (x.square().sum(-1) + 4 * width ** 2 / 5).sum())
    # D's input-gradient cap needs mixed parameter/input derivatives.
    mixed, = torch.autograd.grad(ax.square().sum(), a)
    assert mixed.item() == pytest.approx(float(8 * a.detach() * x.detach().square().sum()))


def test_zero_width_exactly_matches_bounded_alternating_host_and_rng():
    options = dict(start_step=0, curvature_bound=.25, d_curvature_bound=3.)
    ordinary = _host(alternating_curvature(bound_d=True, **options))
    control = _host(consistent_stencil(smooth_cap=0., **options))
    assert ordinary[0] == control[0]
    assert all(torch.equal(a, b) for a, b in zip(ordinary[1:3], control[1:3]))
    assert ordinary[3].records == control[3].records
    assert control[3].smoothing_calls == 0


def test_delayed_activation_matches_original_host():
    plain = _host()
    wrapped = _host(consistent_stencil(start_step=100, curvature_bound=.25, d_curvature_bound=3.))
    assert plain[0] == wrapped[0]
    assert all(torch.equal(a, b) for a, b in zip(plain[1:3], wrapped[1:3]))
    assert wrapped[3].outer_steps == wrapped[3].smoothing_calls == 0


def test_active_stencil_replays_one_batch_and_updates_moments_once():
    plain = _host()
    wrapped = _host(consistent_stencil(start_step=0, curvature_bound=.25, d_curvature_bound=3.))
    assert all(torch.equal(a, b) for a, b in zip(plain[1:3], wrapped[1:3]))
    recorder = wrapped[3]
    assert recorder.outer_steps == 3
    assert recorder.rng_replay_verified == 6
    assert recorder.smoothing_calls > 0
    assert all(0 < r['critic_width'] <= .15 for r in recorder.records)
    for opt in recorder.optimizers:
        assert recorder.rows[opt]['calls'] == 9
        assert all(float(s['step']) == 3 for s in opt.state.values())
    assert not recorder.smoothing_active


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        ConsistentStencilRecorder(smooth_cap=float('nan'))
