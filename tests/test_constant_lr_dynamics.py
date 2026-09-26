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


def _adam_blob(optimizer):
    rows = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            rows.append(parameter.detach().cpu().clone())
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if key in state:
                    rows.append(state[key].detach().cpu().clone())
    return rows


def _hash_tensors(rows):
    import hashlib
    digest = hashlib.sha256()
    for row in rows:
        array = row.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_functional_adam_matches_single_tensor_adam():
    from particlegan.dynamics.unrolled import _single_adam

    torch.manual_seed(0)
    parameter = nn.Parameter(torch.randn(7))
    other = nn.Parameter(torch.randn(3))
    opt = torch.optim.Adam([parameter, other], lr=0.00425, betas=(0.0, 0.999), foreach=False, fused=False)
    parameter.grad = torch.randn(7)
    other.grad = torch.randn(3)
    group = opt.param_groups[0]
    clones = []
    for p in (parameter, other):
        clones.append(_single_adam(
            p.detach().clone(), p.grad.detach().clone(),
            torch.zeros_like(p), torch.zeros_like(p), 0, group,
        ))
    opt.step()
    for updated, p in zip(clones, (parameter, other)):
        param, exp_avg, exp_avg_sq, step = updated
        assert step == 1
        assert torch.equal(param, p.detach())
        assert torch.equal(exp_avg, opt.state[p]["exp_avg"])
        assert torch.equal(exp_avg_sq, opt.state[p]["exp_avg_sq"])
    # Second step, nonzero moments, beta1 > 0.
    opt2 = torch.optim.Adam([parameter], lr=0.0085, betas=(0.1, 0.999), foreach=False, fused=False)
    opt2.state[parameter]["step"] = torch.tensor(4.0)
    opt2.state[parameter]["exp_avg"] = torch.randn_like(parameter)
    opt2.state[parameter]["exp_avg_sq"] = torch.rand_like(parameter).add(0.1)
    parameter.grad = torch.randn(7)
    again = _single_adam(
        parameter.detach().clone(), parameter.grad.detach().clone(),
        opt2.state[parameter]["exp_avg"].clone(), opt2.state[parameter]["exp_avg_sq"].clone(),
        4, opt2.param_groups[0],
    )
    opt2.step()
    assert torch.equal(again[0], parameter.detach())
    assert torch.equal(again[1], opt2.state[parameter]["exp_avg"])
    assert torch.equal(again[2], opt2.state[parameter]["exp_avg_sq"])


def test_unroll_leaves_real_discriminator_and_changes_generator_grad(monkeypatch):
    from benchmarks.legacy.gan_loss import GANLoss
    from benchmarks.legacy.grad_regularizers import GradientPenalty
    from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
    from particlegan.dynamics.unrolled import critic_for_generator

    monkeypatch.setenv("K3P_DYNAMICS", "unrolled")
    torch.manual_seed(0)
    critic = SimpleMLPDiscriminator(2, 8, 2, fourier=1)
    for module in critic.modules():
        if isinstance(module, nn.LeakyReLU):
            module.inplace = True
    opt_d = torch.optim.Adam(critic.parameters(), lr=0.00425, betas=(0.0, 0.999), foreach=False, fused=False)
    loss = GANLoss("logistic", "rp")
    regularizer = GradientPenalty("a_r1r2", coeff=1.0, kappa=1.0)
    real = torch.randn(6, 2)
    fake = torch.nn.Parameter(torch.randn(6, 2))
    detached = fake.detach()
    d_loss = loss.d_loss(critic(real), critic(detached)) + regularizer(critic, real, detached, step=1)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()
    before = _hash_tensors(_adam_blob(opt_d))
    before_params = [p.detach().clone() for p in critic.parameters()]

    def d_loss_fn(module, real_b, fake_b):
        return loss.d_loss(module(real_b), module(fake_b)) + regularizer(module, real_b, fake_b, step=1)

    scorer = critic_for_generator(critic, fake, lambda: (opt_d, real, d_loss_fn, regularizer))
    g_loss = loss.g_loss(scorer(fake), scorer(real))
    opt_d.zero_grad()
    fake.grad = None
    g_loss.backward()
    unrolled_grad = fake.grad.detach().clone()
    assert _hash_tensors(_adam_blob(opt_d)) == before
    assert all(torch.equal(p.detach(), old) for p, old in zip(critic.parameters(), before_params))
    assert all(child.inplace for child in critic.modules() if isinstance(child, nn.LeakyReLU))
    assert torch.isfinite(unrolled_grad).all()
    # The reaction term is not the plain score on the current critic.
    fake.grad = None
    plain = loss.g_loss(critic(fake), critic(real))
    plain.backward()
    assert not torch.allclose(fake.grad, unrolled_grad)


def test_flag_off_skips_unroll_and_hashes_match(monkeypatch):
    from benchmarks.legacy.grad_regularizers import GradientPenalty
    from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, train_mode_hold
    from particlegan.dynamics import unrolled

    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    calls = {"n": 0}

    def _forbidden(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("unroll ran with the flag unset")

    monkeypatch.setattr(unrolled, "_unroll", _forbidden)

    def once():
        torch.manual_seed(0)
        row = train_mode_hold(
            ModeHoldRecipe(steps=2),
            cap_factory=lambda: GradientPenalty("a_r1r2", coeff=1.0, kappa=1.0),
            seed=0,
        )
        return row["modes"], row["hq"]

    first = once()
    second = once()
    assert first == second and calls["n"] == 0


def test_unrolled_mode_hold_is_deterministic(monkeypatch):
    from benchmarks.legacy.grad_regularizers import GradientPenalty
    from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, train_mode_hold

    monkeypatch.setenv("K3P_DYNAMICS", "unrolled")

    def once():
        torch.manual_seed(0)
        return train_mode_hold(
            ModeHoldRecipe(steps=2),
            cap_factory=lambda: GradientPenalty("a_r1r2", coeff=1.0, kappa=1.0),
            seed=0,
        )

    first, second = once(), once()
    assert (first["modes"], first["hq"]) == (second["modes"], second["hq"])


def test_trainer_flag_off_hash_matches_and_unrolled_is_deterministic(monkeypatch):
    from tests.test_k3p_trainer import _reals, _recipe, _trainer

    def run(flag):
        if flag:
            monkeypatch.setenv("K3P_DYNAMICS", "unrolled")
        else:
            monkeypatch.delenv("K3P_DYNAMICS", raising=False)
        torch.manual_seed(0)
        recipe = _recipe(lr_floor=1.0, network_lr_floor=1.0, lr_anneal_start=0.0, total_steps=3)
        trainer = _trainer(recipe)
        for real in _reals(2, seed=1):
            trainer.step(real)
        rows = []
        for module in (trainer.G, trainer.D, trainer.prior):
            rows.extend(p.detach().cpu() for p in module.parameters())
        rows.extend(_adam_blob(trainer.opt_d))
        rows.extend(_adam_blob(trainer.opt_g))
        return _hash_tensors(rows)

    assert run(False) == run(False)
    assert run(True) == run(True)
    assert run(True) != run(False)


def test_scaled_penalty_accepts_ema_critic_and_matches_k3p():
    import importlib.util
    from pathlib import Path

    import torch.optim.optimizer as opt_mod

    from particlegan.grad_regularizers import GradientPenalty

    original = GradientPenalty.penalty
    pre = dict(opt_mod._global_optimizer_pre_hooks)
    post = dict(opt_mod._global_optimizer_post_hooks)
    path = Path(__file__).resolve().parents[1] / "reports/toy100/gap-fill-20260925/sources/k3p/mechanism.py"
    spec = importlib.util.spec_from_file_location("k3p_mechanism_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        torch.manual_seed(0)
        critic = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            critic.weight.copy_(torch.tensor([[0.2, -0.1]]))
        penalty = GradientPenalty(coeff=1.0, kappa=1.0)
        real, fake = torch.randn(4, 2), torch.randn(4, 2)
        expected = original(penalty, critic, real, fake, 1, False, ema_critic=lambda x: critic(x))[0]
        got = penalty.penalty(critic, real, fake, 1, False, ema_critic=lambda x: critic(x))[0]
        assert torch.equal(got, expected)
        assert module._state["pending"] is False
    finally:
        GradientPenalty.penalty = original
        opt_mod._global_optimizer_pre_hooks.clear()
        opt_mod._global_optimizer_pre_hooks.update(pre)
        opt_mod._global_optimizer_post_hooks.clear()
        opt_mod._global_optimizer_post_hooks.update(post)


@pytest.fixture(autouse=True)
def _clear_dynamics(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
