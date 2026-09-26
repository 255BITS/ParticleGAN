"""The constant-LR dynamics, each at the one setting named in its module."""
import hashlib
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
from particlegan.dynamics import pair_chord, unit_rms
from particlegan.dynamics.d_replay import BUFFER_UPDATES, replay_fakes, reset as reset_replay
from particlegan.dynamics.shared_batch import shared_batch_update
from particlegan.grad_regularizers import GradientPenalty as K3PPenalty

ROOT = Path(__file__).resolve().parents[1]

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


def _batch(rows):
    return torch.arange(rows * 2, dtype=torch.float32).view(rows, 2)


def test_replay_is_half_current_and_half_fifo_stride(monkeypatch):
    monkeypatch.setenv("K3P_DYNAMICS", "d_replay")
    reset_replay()
    first = _batch(4)
    assert replay_fakes(first) is first
    second = _batch(4) + 8
    mixed = replay_fakes(second)
    # Stride across the one stored batch: indices 0 and 2.
    assert torch.equal(mixed[:2], second[:2])
    assert torch.equal(mixed[2:], first[[0, 2]])
    assert mixed.shape == first.shape
    assert BUFFER_UPDATES == 40


def test_replay_drops_the_oldest_update(monkeypatch):
    monkeypatch.setenv("K3P_DYNAMICS", "d_replay")
    reset_replay()
    batches = [_batch(4) + 8 * i for i in range(BUFFER_UPDATES + 2)]
    last = None
    for batch in batches:
        last = replay_fakes(batch)
    # The last draw happens before the 42nd push. After 41 pushes the FIFO has
    # dropped update 0, so the stride starts at update 1 and steps 80 rows.
    assert torch.equal(last[2], batches[1][0])
    assert torch.equal(last[3], batches[21][0])
    assert torch.equal(last[:2], batches[-1][:2])
    from particlegan.dynamics.d_replay import receipt
    assert receipt["passthrough"] == 1 and receipt["mixes"] == BUFFER_UPDATES + 1
    assert receipt["held"] == BUFFER_UPDATES * 4


def test_replay_flag_off_returns_the_same_tensor(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    reset_replay()
    fake = _batch(4)
    assert replay_fakes(fake) is fake
    second = fake + 1
    assert replay_fakes(second) is second


def _trainer_hash(flag, steps):
    from particlegan import GANTrainer, get_recipe

    reset_replay()
    if flag is None:
        os.environ.pop("K3P_DYNAMICS", None)
    else:
        os.environ["K3P_DYNAMICS"] = flag
    torch.manual_seed(0)
    torch.set_num_threads(1)
    recipe = get_recipe(num_particles=32, z_dim=2, batch_size=8, total_steps=steps,
                        network_lr_horizon_cap=16)
    generator = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    trainer = GANTrainer(recipe, generator, critic, seed=0)
    stream = torch.Generator().manual_seed(1)
    for _ in range(steps):
        trainer.step(torch.randn(8, 2, generator=stream))
    digest = hashlib.sha256()
    for parameter in list(trainer.G.parameters()) + list(trainer.D.parameters()) + [trainer.prior.z]:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_flag_off_matches_itself_and_the_first_replay_step_matches_baseline(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    off_one = _trainer_hash(None, 1)
    off_one_again = _trainer_hash(None, 1)
    on_one = _trainer_hash("d_replay", 1)
    off_two = _trainer_hash(None, 3)
    off_two_again = _trainer_hash(None, 3)
    on_two = _trainer_hash("d_replay", 3)
    on_two_again = _trainer_hash("d_replay", 3)
    assert off_one == off_one_again == on_one
    assert off_two == off_two_again
    assert on_two == on_two_again
    assert off_two != on_two


def test_same_command_twice_is_bit_identical():
    script = """
import hashlib, os
import torch
from torch import nn
from particlegan import GANTrainer, get_recipe
torch.manual_seed(0)
torch.set_num_threads(1)
recipe = get_recipe(num_particles=32, z_dim=2, batch_size=8, total_steps=3, network_lr_horizon_cap=16)
trainer = GANTrainer(recipe, nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)),
                     nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)), seed=0)
stream = torch.Generator().manual_seed(1)
for _ in range(3):
    trainer.step(torch.randn(8, 2, generator=stream))
digest = hashlib.sha256()
for parameter in list(trainer.G.parameters()) + list(trainer.D.parameters()) + [trainer.prior.z]:
    digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
print(digest.hexdigest())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["K3P_DYNAMICS"] = "d_replay"
    runs = [
        subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, env=env, text=True).strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    env.pop("K3P_DYNAMICS")
    baseline = subprocess.check_output(
        [sys.executable, "-c", script], cwd=ROOT, env=env, text=True).strip()
    assert baseline != runs[0]


def test_replay_does_not_add_a_generator_draw(monkeypatch):
    from tests.test_k3p_trainer import _reals, _trainer

    monkeypatch.setenv("K3P_DYNAMICS", "d_replay")
    reset_replay()
    trainer = _trainer()
    draws = {"n": 0}
    original = trainer.prior.sample

    def sample(*args, **kwargs):
        draws["n"] += 1
        return original(*args, **kwargs)

    trainer.prior.sample = sample
    trainer.step(_reals(1)[0])
    trainer.step(_reals(1)[0])
    assert draws["n"] == 4


def test_scaled_penalty_accepts_ema_critic_and_matches_the_current_penalty():
    import atexit
    import torch.optim.optimizer as optimizer_module

    k3p = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
    saved = k3p_mod.GradientPenalty.penalty
    pre = set(optimizer_module._global_optimizer_pre_hooks)
    post = set(optimizer_module._global_optimizer_post_hooks)
    sys.path.insert(0, str(k3p))
    import mechanism
    try:
        critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        real, fake = torch.randn(4, 2), torch.randn(4, 2)
        regularizer = k3p_mod.GradientPenalty(coeff=1.0, kappa=1.0, lr_floor=0.01)
        got = k3p_mod.GradientPenalty.penalty(
            regularizer, critic, real, fake, 1, False, ema_critic=critic)
        expect = saved(regularizer, critic, real, fake, 1, False, ema_critic=critic)
        assert torch.equal(got[0], expect[0])
        regularizer.record.lr_max = 1.0
        regularizer.record.lr_last = 0.01
        regularizer.record.anchor_started = True
        blended = k3p_mod.GradientPenalty.penalty(
            regularizer, critic, real, fake, 1, True, ema_critic=critic)
        reference = saved(regularizer, critic, real, fake, 1, True, ema_critic=critic)
        assert torch.equal(blended[0], reference[0])
    finally:
        k3p_mod.GradientPenalty.penalty = saved
        sys.path.remove(str(k3p))
        for key in list(optimizer_module._global_optimizer_pre_hooks):
            if key not in pre:
                del optimizer_module._global_optimizer_pre_hooks[key]
        for key in list(optimizer_module._global_optimizer_post_hooks):
            if key not in post:
                del optimizer_module._global_optimizer_post_hooks[key]
        atexit.unregister(mechanism._write_receipt)


@pytest.fixture(autouse=True)
def _clear_dynamics(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    reset_replay()
    yield
    reset_replay()
