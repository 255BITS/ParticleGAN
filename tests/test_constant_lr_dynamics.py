"""Constant-LR dynamics, each at the one setting named in its module."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import benchmarks.legacy.grad_regularizers as legacy_mod
import particlegan.grad_regularizers as k3p_mod
from benchmarks.legacy.grad_regularizers import GradientPenalty as LegacyPenalty
from particlegan.dynamics import lookahead_minmax, pair_chord, unequal_penalty, unit_rms
from particlegan.dynamics.shared_batch import shared_batch_update
from particlegan.grad_regularizers import GradientPenalty as K3PPenalty

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "reports/toy100/constant-lr-dynamics"
K3P_SRC = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"

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


def _adam_players():
    torch.manual_seed(0)
    critic = nn.Parameter(torch.tensor([1.0, -0.5]))
    generator = nn.Parameter(torch.tensor([0.25, 0.5]))
    particles = nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
    opt_d = torch.optim.Adam([critic], lr=0.00425, betas=(0.0, 0.999), foreach=False)
    opt_g = torch.optim.Adam(
        [
            {"params": [generator], "lr": 0.00425, "betas": (0.0, 0.999)},
            {"params": [particles], "lr": 0.0085, "betas": (0.0, 0.999)},
        ],
        foreach=False,
    )
    return critic, generator, particles, opt_d, opt_g


def _assign_grads(critic, generator, particles):
    critic.grad = torch.tensor([0.3, -0.1])
    generator.grad = torch.tensor([-0.2, 0.4])
    particles.grad = torch.tensor([0.5, 0.0, -0.1])


def _paired_steps(critic, generator, particles, opt_d, opt_g, steps):
    for _ in range(steps):
        _assign_grads(critic, generator, particles)
        opt_d.step()
        opt_g.step()


def test_lookahead_minmax_joint_slow_step_uses_paper_defaults():
    assert lookahead_minmax.K == 5 and lookahead_minmax.ALPHA == 0.5
    plain = _adam_players()
    _paired_steps(*plain, 4)
    fast4 = tuple(tensor.detach().clone() for tensor in plain[:3])
    _paired_steps(*plain, 1)
    fast5 = tuple(tensor.detach().clone() for tensor in plain[:3])
    moments = {
        name: optimizer.state[parameter]["exp_avg"].detach().clone()
        for name, parameter, optimizer in (
            ("d", plain[0], plain[3]),
            ("g", plain[1], plain[4]),
            ("z", plain[2], plain[4]),
        )
    }
    initial = _adam_players()
    init = tuple(tensor.detach().clone() for tensor in initial[:3])
    try:
        lookahead_minmax.install()
        held = _adam_players()
        _paired_steps(*held, 4)
        assert all(torch.equal(a, b) for a, b in zip(held[:3], fast4))
        _assign_grads(*held[:3])
        held[3].step()
        # The critic's 5th fast step must not backtrack before the generator steps.
        assert torch.equal(held[0].detach(), fast5[0])
        held[4].step()
        expected = []
        for start, fast in zip(init, fast5):
            mixed = start.clone()
            mixed.add_(fast - mixed, alpha=0.5)
            expected.append(mixed)
        assert all(torch.equal(got.detach(), want) for got, want in zip(held[:3], expected))
        for name, parameter, optimizer in (
            ("d", held[0], held[3]),
            ("g", held[1], held[4]),
            ("z", held[2], held[4]),
        ):
            assert torch.equal(optimizer.state[parameter]["exp_avg"], moments[name])
        assert lookahead_minmax.receipt["syncs"] == 1
        assert lookahead_minmax.receipt["fast_steps"] == 10
    finally:
        lookahead_minmax.uninstall()


def test_lookahead_minmax_repeats_and_flag_off_matches_adam():
    script = r"""
import hashlib
import torch
from torch import nn
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
critic = nn.Parameter(torch.tensor([1.0, -0.5]))
generator = nn.Parameter(torch.tensor([0.25, 0.5]))
particles = nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
opt_d = torch.optim.Adam([critic], lr=0.00425, betas=(0.0, 0.999), foreach=False)
opt_g = torch.optim.Adam([
    {"params": [generator], "lr": 0.00425, "betas": (0.0, 0.999)},
    {"params": [particles], "lr": 0.0085, "betas": (0.0, 0.999)},
], foreach=False)
for _ in range(8):
    critic.grad = torch.tensor([0.3, -0.1])
    generator.grad = torch.tensor([-0.2, 0.4])
    particles.grad = torch.tensor([0.5, 0.0, -0.1])
    opt_d.step()
    opt_g.step()
raw = b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in (critic, generator, particles))
print("HASH " + hashlib.sha256(raw).hexdigest())
"""

    def run(env_update):
        env = os.environ.copy()
        env.pop("K3P_DYNAMICS", None)
        env.update(PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
        env.update(env_update)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        lines = [line.split()[1] for line in completed.stdout.splitlines() if line.startswith("HASH ")]
        assert len(lines) == 1
        return lines[0]

    baseline = run({})
    flag_off = run({"PYTHONPATH": str(SCREEN) + os.pathsep + str(ROOT)})
    assert flag_off == baseline
    first = run({"PYTHONPATH": str(SCREEN) + os.pathsep + str(ROOT), "K3P_DYNAMICS": "lookahead_minmax"})
    second = run({"PYTHONPATH": str(SCREEN) + os.pathsep + str(ROOT), "K3P_DYNAMICS": "lookahead_minmax"})
    assert first == second
    assert first != baseline


def test_unequal_penalty_accepts_ema_critic_and_matches_k3p():
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
from torch import nn
import mechanism
from particlegan.grad_regularizers import GradientPenalty
torch.manual_seed(0)
critic = nn.Linear(2, 1)
real, fake = torch.randn(4, 2), torch.randn(4, 2)
penalty = GradientPenalty(coeff=1.0, kappa=1.0)
try:
    got, _ = penalty.penalty(critic, real, fake, 1, False, ema_critic=lambda x: critic(x))
except TypeError as exc:
    print("TYPEERROR " + str(exc))
    raise SystemExit(0)
original, _ = mechanism._original_penalty(penalty, critic, real, fake, 1, False, ema_critic=lambda x: critic(x))
print("MATCH" if torch.equal(got, original) else "DIFFER")
"""
    bare = os.environ.copy()
    bare.pop("K3P_DYNAMICS", None)
    bare["PYTHONPATH"] = str(ROOT)
    bare["PYTHONHASHSEED"] = "0"
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(K3P_SRC)], cwd=ROOT, env=bare, capture_output=True, text=True, check=False,
    )
    assert crashed.returncode == 0, crashed.stderr
    assert any(line.startswith("TYPEERROR") for line in crashed.stdout.splitlines()), crashed.stdout
    hooked = os.environ.copy()
    hooked.pop("K3P_DYNAMICS", None)
    hooked["PYTHONPATH"] = str(SCREEN) + os.pathsep + str(ROOT)
    hooked["PYTHONHASHSEED"] = "0"
    fixed = subprocess.run(
        [sys.executable, "-c", script, str(K3P_SRC)], cwd=ROOT, env=hooked, capture_output=True, text=True, check=False,
    )
    assert fixed.returncode == 0, fixed.stderr
    assert "MATCH" in fixed.stdout.splitlines()


@pytest.fixture(autouse=True)
def _clear_dynamics(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    yield
    lookahead_minmax.uninstall()
    unequal_penalty.uninstall()
