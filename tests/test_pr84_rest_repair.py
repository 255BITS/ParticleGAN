"""Independent stencil derivative, exact controls and host accounting."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100 import pr84_rest_repair_scratch as repair


SOURCE = Path(__file__).resolve().parents[1] / 'reports/toy100/continuous-evidence/pr84-rest-repair/source/pr84_head_original.py'
spec = importlib.util.spec_from_file_location('pr84_head_original_for_test', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
OPTIONS = dict(bound_d=True, curvature_bound=.25, d_curvature_bound=3., smooth_critic=True, smooth_cap=.15)


def measured_cubic(module):
    # A=5 puts the single slope .135 below the unchanged .2 gate, while
    # the twice-convolved slope .270 is above it. The tiny +5e-6 term is
    # the exact central-difference error for epsilon=1e-3, not tolerance.
    critic = SimpleMLPDiscriminator(hidden_dim=2, n_hidden=1).double()
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=.1, betas=(0., .9))

    def cubic(self, points):
        return 5 * points[:, 0] ** 3

    with patch.object(SimpleMLPDiscriminator, 'forward', cubic):
        with module.alternating_curvature(**OPTIONS) as (recorder, _):
            recorder._x_before = torch.zeros((2, 2), dtype=torch.float64)
            recorder._smooth_width = .15
            recorder._smooth_on = True
            recorder.optimizers = (None, optimizer)
            recorder.g_base = [torch.zeros_like(parameter)]
            recorder.row = {}
            recorder._scale_bounded_g_step({'critic': critic})
            return recorder.row, parameter.item()


def test_rest_slope_measures_one_convolution_instead_of_two():
    fixed, fixed_parameter = measured_cubic(repair)
    untouched, original_parameter = measured_cubic(original)
    expected_single = 5 * (6 / 5 * .15 ** 2 + .001 ** 2)
    expected_double = 5 * (12 / 5 * .15 ** 2 + .001 ** 2)
    assert fixed['mean_local_slope'] == pytest.approx(expected_single, abs=1e-12)
    assert untouched['mean_local_slope'] == pytest.approx(expected_double, abs=1e-12)
    assert fixed['slope_scale'] == fixed_parameter == 0
    assert untouched['slope_scale'] == original_parameter == 1


def test_toy_common_stencil_restores_opposite_cross_blocks():
    variance = 2 * .15 ** 2 / 5  # marginal variance of the five-point 2D stencil

    def unilateral_field(point):
        theta, x = point
        return torch.stack((-x ** 3, theta * (3 * x ** 2 + 3 * variance)))

    def common_field(point):
        theta, x = point
        return torch.stack((-(x ** 3 + 3 * variance * x), theta * (3 * x ** 2 + 3 * variance)))

    point = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    unilateral = torch.autograd.functional.jacobian(unilateral_field, point)
    common = torch.autograd.functional.jacobian(common_field, point)
    assert unilateral[0, 1] == 0
    assert unilateral[1, 0] == pytest.approx(3 * variance)
    assert torch.equal(unilateral @ unilateral, torch.zeros_like(unilateral))
    assert common[0, 1] == pytest.approx(-3 * variance)
    assert common[1, 0] == pytest.approx(3 * variance)
    assert torch.equal(common + common.T, torch.zeros_like(common))


class CapturePolicy(NoisePolicy):
    def register_generator_optimizer(self, opt_g, opt_d):
        self.optimizers = (opt_d, opt_g)
        return super().register_generator_optimizer(opt_g, opt_d)


def host(context=None):
    torch.set_num_threads(1)
    policy = CapturePolicy(.029, .5, .1, 1200, output_noise_rng='isolated')
    if context is None:
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=True)
        recorder = None
    else:
        with context as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=True)
    state = []
    for optimizer in policy.optimizers:
        for group in optimizer.param_groups:
            for p in group['params']:
                state.extend([p.detach().clone(), *[v.detach().clone() for v in optimizer.state[p].values()
                                                    if isinstance(v, torch.Tensor)]])
    rng = [torch.get_rng_state().clone(), policy.input_stream.get_state().clone(), policy.output_stream.get_state().clone()]
    return result, state, rng, policy.receipt(), recorder


def assert_exact(left, right):
    assert left[0] == right[0]
    assert left[3] == right[3]
    assert all(torch.equal(a, b) for a, b in zip(left[1] + left[2], right[1] + right[2]))


def test_disabled_smoothing_preserves_original_pr84_behavior_exactly():
    options = {**OPTIONS, 'smooth_critic': False}
    assert_exact(host(original.alternating_curvature(**options)), host(repair.alternating_curvature(**options)))


def test_without_rest_decision_smoothed_training_field_is_unchanged():
    with patch.object(original.BothBoundRecorder, '_scale_bounded_g_step', lambda self, local: None):
        untouched = host(original.alternating_curvature(**OPTIONS))
    with patch.object(repair.BothBoundRecorder, '_scale_bounded_g_step', lambda self, local: None):
        corrected = host(repair.alternating_curvature(**OPTIONS))
    assert_exact(untouched, corrected)


def test_delayed_activation_is_exact_ordinary_host():
    assert_exact(host(), host(repair.alternating_curvature(start_step=1000, **OPTIONS)))


def test_active_repair_keeps_rng_and_one_moment_update_per_outer():
    plain = host()
    active = host(repair.alternating_curvature(**OPTIONS))
    assert all(torch.equal(a, b) for a, b in zip(plain[2], active[2]))
    assert plain[3]['step_calls'] == active[3]['step_calls'] == 3
    recorder = active[-1]
    assert recorder.outer_steps == 3
    assert recorder.rng_replay_verified == 6
    for optimizer, row in recorder.rows.items():
        assert row['calls'] == 9
        assert all(int(optimizer.state[p]['step']) == 3 for group in optimizer.param_groups for p in group['params'])
