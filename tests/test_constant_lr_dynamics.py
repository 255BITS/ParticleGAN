"""The three constant-LR dynamics, each at the one setting named in its module."""
import os
import sys

import pytest
import torch
from torch import nn

import benchmarks.legacy.grad_regularizers as legacy_mod
import particlegan.grad_regularizers as k3p_mod
from benchmarks.legacy.grad_regularizers import GradientPenalty as LegacyPenalty
from particlegan.dynamics import pair_chord, unit_rms
from particlegan.dynamics.shared_batch import shared_batch_update
from particlegan.grad_regularizers import GradientPenalty as K3PPenalty

_LEGACY_PENALTY = legacy_mod.GradRegularizer.penalty
_K3P_PENALTY = k3p_mod.GradientPenalty.penalty


def _restore_penalties() -> None:
    legacy_mod.GradRegularizer.penalty = _LEGACY_PENALTY
    k3p_mod.GradientPenalty.penalty = _K3P_PENALTY
    finder = pair_chord._FINDER
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
    pair_chord._FINDER = None


def _linear():
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.1, 0.0]]))
    return layer


def test_unit_rms_is_scale_free_at_the_group_lr():
    torch.manual_seed(0)
    parameter = nn.Parameter(torch.zeros(4))
    opt = torch.optim.Adam([parameter], lr=0.00425)
    grad = torch.tensor([3.0, 0.0, 0.0, 0.0])
    rms = grad.square().mean().sqrt()
    expected = -0.00425 * grad / (rms + opt.param_groups[0]["eps"])
    try:
        unit_rms.install()
        parameter.grad = grad.clone()
        opt.step()
        moved = parameter.detach().clone()
        parameter.grad = grad * 10
        with torch.no_grad():
            parameter.zero_()
        opt.step()
    finally:
        unit_rms.uninstall()
    assert torch.allclose(moved, expected)
    assert torch.allclose(parameter.detach(), expected, atol=1e-10, rtol=0)


def test_pair_chord_matches_the_endpoint_term():
    real = torch.randn(8, 2)
    fake = torch.randn(8, 2)
    critic = _linear()
    legacy = LegacyPenalty("a_r1r2", coeff=1.0, kappa=1.0)
    base = legacy.penalty(critic, real, fake, collect_stats=False)[0].detach()
    capped = LegacyPenalty("b_cap", coeff=1.0, kappa=1.0)
    cap_before = capped.penalty(critic, real, fake, collect_stats=False)[0].detach()
    try:
        pair_chord.install()
        legacy_pen = legacy.penalty(critic, real, fake, collect_stats=False)[0]
        k3p = K3PPenalty(coeff=1.0, kappa=1.0)
        k3p_pen = k3p.penalty(critic, real, fake, collect_stats=False)[0]
        cap_after = capped.penalty(critic, real, fake, collect_stats=False)[0].detach()
        # Inplace activations, as on the ring critic. The chord has to backward.
        activated = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(8, 1))

        class _Critic(nn.Module):
            def forward(self, x):
                return activated(x).squeeze(-1)

        inplace = LegacyPenalty("a_r1r2", coeff=1.0, kappa=1.0)
        loss = inplace(_Critic(), real, fake)
        loss.backward()
    finally:
        _restore_penalties()
    # ||w||^2 = 0.01. Legacy endpoint term is (c/2) mean||g||^2 = 0.005.
    # K3P real term is (c/2) mean(||g||^2 / d) = 0.0025, and the fake cap is inactive.
    assert torch.allclose(legacy_pen.detach(), base + 0.005)
    assert torch.allclose(k3p_pen.detach(), torch.tensor(0.005))
    assert torch.allclose(cap_before, cap_after)
    assert loss.detach().isfinite()


def test_shared_batch_reuses_the_critic_draw(monkeypatch):
    from tests.test_k3p_trainer import _reals, _trainer

    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    trainer = _trainer()
    draws = {"n": 0}
    original = trainer.prior.sample

    def sample(*args, **kwargs):
        draws["n"] += 1
        return original(*args, **kwargs)

    trainer.prior.sample = sample
    paired = {"n": 0}

    def generator_real():
        paired["n"] += 1
        return _reals(1)[0]

    trainer.step(_reals(1)[0], generator_real=generator_real)
    assert draws["n"] == 2 and paired["n"] == 1

    monkeypatch.setenv("K3P_DYNAMICS", "shared_batch")
    trainer.step(_reals(1)[0], generator_real=generator_real)
    assert draws["n"] == 3 and paired["n"] == 1
    assert shared_batch_update() is True


def test_shared_batch_off_matches_two_draws(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    assert shared_batch_update() is False
    assert os.environ.get("K3P_DYNAMICS") is None


@pytest.fixture(autouse=True)
def _clear_dynamics(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
