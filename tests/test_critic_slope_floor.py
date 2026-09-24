"""The occupied floor changes D's penalty and leaves G's stencil path in place."""

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.grad_regularizers import GradientPenalty
from reports.toy100.critic_slope_floor import (
    FLATNESS_WEIGHT, SLOPE_FLOOR, occupied_floor, stencil_values,
)
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


class _Linear(SimpleMLPDiscriminator):
    def forward(self, x):
        return x[..., :1].squeeze(-1)


class _Constant(SimpleMLPDiscriminator):
    def forward(self, x):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)


def test_floor_is_zero_on_unit_slope_and_full_on_a_flat_field():
    width = .15
    points = torch.tensor([[0., 0.], [.4, -.2], [-.3, .5]])
    linear = _Linear(2, hidden_dim=4, n_hidden=1, fourier=0)
    # Bypass the parent MLP forward by calling the subclass through stencil_values.
    forward = type(linear).forward
    slope_linear = stencil_values(linear, points, width, forward)
    assert torch.allclose(slope_linear, points[:, 0])
    penalty = occupied_floor(linear, points, points, width, forward)
    assert float(penalty) == 0.
    constant = _Constant(2, hidden_dim=4, n_hidden=1, fourier=0)
    penalty = occupied_floor(constant, points, points, width, type(constant).forward)
    expected = FLATNESS_WEIGHT * (SLOPE_FLOOR ** 2)
    assert torch.allclose(penalty, torch.tensor(expected))


def test_nonspatial_inputs_add_nothing():
    module = SimpleMLPDiscriminator(4, hidden_dim=4, n_hidden=1, fourier=0)
    real = torch.zeros(3, 4)
    assert float(occupied_floor(module, real, real, .15, SimpleMLPDiscriminator.forward)) == 0.


def test_floor_is_off_while_the_recorder_is_disabled_and_g_bounds_stay():
    torch.set_num_threads(1)
    penalty = GradientPenalty("b_cap", coeff=1.0, kappa=1.0)
    real = torch.zeros(2, 2)
    fake = torch.zeros(2, 2)
    with pr84_smoothed_candidate(task="mode_hold") as (recorder, _):
        recorder.enabled = False
        before = float(penalty(SimpleMLPDiscriminator(2, hidden_dim=4, n_hidden=1, fourier=0), real, fake))
    from reports.toy100.critic_slope_floor import critic_slope_floor
    with critic_slope_floor(task="mode_hold") as (recorder, _):
        recorder.enabled = False
        module = SimpleMLPDiscriminator(2, hidden_dim=4, n_hidden=1, fourier=0)
        assert float(penalty(module, real, fake)) == before
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 0
        recorder.row = {}
        active = penalty(module, real, fake)
        assert float(active) > before
        assert recorder.curvature_bound == .25
        assert recorder.d_curvature_bound == 3.
