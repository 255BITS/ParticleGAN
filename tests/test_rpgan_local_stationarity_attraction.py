"""Cancellation and attraction are distinct local learned-game properties.

These deterministic scalar derivatives supplement the frozen saved-critic
observation. They change no native optimizer, model or observation artifact.
"""
import math

import pytest
import torch
from torch import nn

from examples.e22_routed_game_stationarity import paired_games


class FixedQuadratic(nn.Module):
    def __init__(self, quadratic):
        super().__init__()
        self.quadratic = quadratic
    def forward(self, error, condition):
        return (.7*error + self.quadratic*error.square()).mean((1,2)).unsqueeze(-1)


class EvenScore(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic
    def forward(self, error, condition):
        return .5*(self.critic(error,condition)+self.critic(-error,condition))


def derivatives(point, *, quadratic, even):
    residual = torch.tensor([[[point]]], dtype=torch.float64, requires_grad=True)
    critic = FixedQuadratic(quadratic)
    critic = EvenScore(critic) if even else critic
    loss,_ = paired_games(critic,residual,torch.zeros(1,1),torch.full_like(residual,.3),antithetic=True)
    gradient = torch.autograd.grad(loss.sum(),residual,create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(),residual)[0]
    return float(loss.detach()),float(gradient.detach()),float(curvature)


def test_current_antithetic_has_finite_wrong_attractor_for_fixed_concave_critic():
    # D=.7x-.2x². This fixed root solves the analytical scalar loss derivative,
    # not a model optimization or an epsilon/quality search.
    wrong_root = 1.7103369216103252
    _,origin_force,_ = derivatives(0.,quadratic=-.2,even=False)
    _,wrong_force,wrong_curvature = derivatives(wrong_root,quadratic=-.2,even=False)
    assert origin_force == pytest.approx(-.35,abs=1e-14)
    assert abs(wrong_force) < 1e-14 and wrong_curvature > 0
    zero_loss,zero_force,zero_curvature = derivatives(0.,quadratic=-.2,even=True)
    assert zero_loss == pytest.approx(math.log(2),abs=1e-14)
    assert zero_force == 0 and zero_curvature == pytest.approx(.2036,abs=1e-14)


@pytest.mark.parametrize("point", [-1.,-.25,.25,1.])
def test_even_antithetic_concave_fixture_points_toward_its_exact_origin(point):
    loss,force,curvature = derivatives(point,quadratic=-.2,even=True)
    assert loss > math.log(2) and force*point > 0 and curvature > 0


def test_even_antithetic_cancellation_alone_can_leave_a_local_maximum():
    loss,force,curvature = derivatives(0.,quadratic=.2,even=True)
    assert loss == pytest.approx(math.log(2),abs=1e-14)
    assert force == 0
    assert curvature == pytest.approx(-.1964,abs=1e-14)
    # The saved-critic observation measured stationarity, not curvature or safe
    # acquisition. A correction cannot be promoted on cancellation alone.
