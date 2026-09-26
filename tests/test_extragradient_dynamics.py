"""Extra-Adam at one published setting, and the unequal-mass penalty signature."""
import hashlib
import importlib.util
import inspect
import os
import sys

import pytest
import torch
from torch import nn

from particlegan.dynamics import extragradient
from particlegan.dynamics.extragradient import apply_joint_extragradient
from particlegan.grad_regularizers import GradientPenalty


def _adam_delta(grad, lr, eps=1e-8):
    """First Adam step at beta1 = 0: bias-corrected v is g^2, so the step is lr * g / (|g|+eps)."""
    return -lr * grad / (grad.abs() + eps)


def _hash_run(dynamics, steps=2):
    import benchmarks.locked_shared.mode_hold as mode_hold
    from particlegan.det_init import install as install_init
    from particlegan.det_init import uninstall as uninstall_init

    torch.set_num_threads(1)
    install_init("hid_q")
    if dynamics:
        extragradient.install()
    else:
        extragradient.uninstall()
    captured = {}
    original = mode_hold.checkpoint

    def grab(step, measure):
        if step == steps:
            frame = inspect.currentframe().f_back
            digest = hashlib.sha256()
            for name in ("generator", "critic", "prior"):
                module = frame.f_locals[name]
                for parameter in module.parameters():
                    digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            for name in ("opt_g", "opt_d"):
                opt = frame.f_locals[name]
                for state in opt.state.values():
                    step_t = state.get("step")
                    if torch.is_tensor(step_t):
                        digest.update(str(int(step_t.item())).encode())
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            digest.update(state[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            captured["sha256"] = digest.hexdigest()
        return original(step, measure)

    mode_hold.checkpoint = grab
    try:
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps), seed=0)
    finally:
        mode_hold.checkpoint = original
        uninstall_init()
    assert "sha256" in captured
    return captured["sha256"]


def test_extragradient_steps_from_the_origin_using_the_lookahead_gradient():
    torch.manual_seed(0)
    lr = 0.1

    class Player(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(value))

    disc, gen = Player([0.0]), Player([0.0])
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8, foreach=False)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8, foreach=False)
    extragradient.install()
    try:
        def evaluate():
            opt_d.zero_grad()
            opt_g.zero_grad()
            ((disc.w - gen.w.detach() - 0.05) ** 2).sum().backward()
            ((gen.w - 10.0 * disc.w.detach()) ** 2).sum().backward()

        apply_joint_extragradient((opt_d, opt_g), (disc, gen), evaluate, ())
    finally:
        extragradient.uninstall()
    # Lookahead moves d to about +lr (grad at 0 is negative). There the critic
    # gradient flips sign, so the committed step from 0 goes the other way.
    assert torch.allclose(disc.w.detach(), _adam_delta(torch.tensor(0.1), lr), atol=1e-6)
    assert torch.allclose(gen.w.detach(), _adam_delta(torch.tensor(-2.0), lr), atol=1e-6)
    assert int(opt_d.state[disc.w]["step"]) == 1
    assert int(opt_g.state[gen.w]["step"]) == 1


def test_flag_off_matches_the_host_and_two_runs_match(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    import benchmarks.locked_shared.mode_hold as mode_hold
    host = mode_hold.train_mode_hold
    path = os.path.join("reports", "toy100", "constant-lr-dynamics", "sitecustomize.py")
    spec = importlib.util.spec_from_file_location("k3p_clr_sitecustomize_off", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert mode_hold.train_mode_hold is host
    first = _hash_run(False)
    second = _hash_run(False)
    assert first == second
    monkeypatch.setenv("K3P_DYNAMICS", "extragradient")
    on = _hash_run(True)
    again = _hash_run(True)
    extragradient.uninstall()
    assert on == again
    assert on != first
    assert _hash_run(False) == first


def test_trainer_flag_off_matches_and_extragradient_is_deterministic(monkeypatch):
    from tests.test_k3p_trainer import _reals, _trainer

    monkeypatch.delenv("K3P_DYNAMICS", raising=False)

    def run(installed):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        if installed:
            extragradient.install()
        else:
            extragradient.uninstall()
        trainer = _trainer()
        for real in _reals(2):
            trainer.step(real)
        digest = hashlib.sha256()
        for module in (trainer.G, trainer.D, trainer.prior):
            for parameter in module.parameters():
                digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest(), int(next(iter(trainer.opt_d.state.values()))["step"])

    base_a, _ = run(False)
    base_b, _ = run(False)
    assert base_a == base_b
    eg_a, steps = run(True)
    eg_b, _ = run(True)
    extragradient.uninstall()
    assert eg_a == eg_b and eg_a != base_a and steps == 2
    assert run(False)[0] == base_a


def test_scaled_penalty_accepts_ema_critic_and_matches_k3p():
    root = os.path.join("reports", "toy100", "gap-fill-20260925", "sources", "k3p")
    sys.path.insert(0, root)
    import mechanism
    from benchmarks.legacy.grad_regularizers import GradientPenalty as LegacyPenalty

    torch.manual_seed(0)
    critic = nn.Sequential(nn.Linear(2, 4), nn.LeakyReLU(0.2), nn.Linear(4, 1))
    real, fake = torch.randn(4, 2), torch.randn(4, 2)
    left, right = GradientPenalty(coeff=1.0, kappa=1.0), GradientPenalty(coeff=1.0, kappa=1.0)
    ema = lambda x: x.sum() * 0  # not evaluated while s == 1
    original = mechanism._original_penalty(left, critic, real, fake, 1, False, ema_critic=ema)
    delegated = right.penalty(critic, real, fake, 1, False, ema_critic=ema)
    assert torch.equal(original[0], delegated[0])
    legacy = LegacyPenalty("a_r1r2", coeff=1.0, kappa=1.0)
    value = mechanism.scaled_penalty(
        legacy, critic, real, fake, step=1, generator=None, collect_stats=False, ema_critic=ema)
    assert torch.isfinite(value[0])


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    yield
    extragradient.uninstall()
