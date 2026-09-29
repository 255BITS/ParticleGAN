"""The package default critic penalty (KA2) through GANTrainer, and the E22 configuration
(configs/100gaussians/e22-noout.json) end to end: phases, the R1 switch and exact resume."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import particlegan.ka2
from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.ka2 import KA2CriticAdam, KA2GradientPenalty

E22 = json.loads((Path(__file__).parents[1] / "configs/100gaussians/e22-noout.json").read_text())


@pytest.fixture
def short_warmup(monkeypatch):
    # KA2 blends after WARMUP_CALLS (800) critic calls; shorten it so short runs reach the blend.
    monkeypatch.setattr(particlegan.ka2, "WARMUP_CALLS", 4)


def _trainer(recipe, seed=0):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    return GANTrainer(recipe, G, D, seed=seed, optimizer_options={"foreach": False})


def _reals(n, batch=8, seed=1):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, 2, generator=rng) * 2 for _ in range(n)]


def _run(trainer, reals):
    return [trainer.step(real, collect_stats=True) for real in reals]


def _assert_same_run(left, right):
    for a, b in zip(left, right):
        for key in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(a[key], b[key]), key


def _assert_same_models(left, right):
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, value in left["models"][name].items():
            assert torch.equal(value, right["models"][name][key]), (name, key)


def test_default_critic_is_ka2():
    recipe = get_recipe()
    assert recipe == Recipe() and recipe.name == "ka2"
    trainer = _trainer(get_recipe(num_particles=64, z_dim=2, batch_size=8))
    assert isinstance(trainer.opt_d, KA2CriticAdam)
    assert isinstance(trainer.penalty.regularizer, KA2GradientPenalty)
    assert trainer.ema_D is not None and trainer.opt_d.guard is not None


def test_default_trainer_reaches_blend_and_resumes_exactly_inside_it(short_warmup):
    recipe = get_recipe(num_particles=64, z_dim=2, batch_size=8, total_steps=20, d_guard_min_steps=2)
    reals = _reals(16)
    full = _trainer(recipe)
    full_out = _run(full, reals)
    phases = [o["penalty_stats"]["phase"] for o in full_out]
    assert phases[:3] == ["a"] * 3 and phases[-1] == "blend"
    split = phases.index("blend") + 2
    first = _trainer(recipe)
    _run(first, reals[:split])
    checkpoint = first.state_dict()
    assert checkpoint["schema"] == 4
    resumed = _trainer(recipe, seed=5)  # different construction randomness; the checkpoint wins
    resumed.load_state_dict(checkpoint)
    _assert_same_run(_run(resumed, reals[split:]), full_out[split:])
    _assert_same_models(resumed.state_dict(), full.state_dict())


def test_r1_real_switch_removes_only_the_real_gradient_term():
    torch.manual_seed(0)
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    rng = torch.Generator().manual_seed(0)
    real = torch.randn(32, 2, generator=rng, dtype=torch.float64)
    fake = torch.randn(32, 2, generator=rng, dtype=torch.float64) + 3.0
    penalties = {}
    for r1 in (True, False):
        recipe = get_recipe(critic_r1_real=r1)
        opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D))
        penalties[r1] = recipe.make_critic_penalty(opt)(D, real, fake)
    x = real.clone().requires_grad_(True)
    grad = torch.autograd.grad(D(x).sum(), x)[0]
    r1_term = get_recipe().reg_coeff / 2 * (grad.square().sum(1) / 2).mean()
    torch.testing.assert_close(penalties[True] - penalties[False], r1_term, rtol=1e-10, atol=1e-14)


def _e22_recipe():
    options = dict(E22, num_particles=512, z_dim=2, batch_size=64)  # the harness sets these from the host
    return get_recipe(**options)


def test_e22_config_builds_and_resumes_bit_exactly():
    recipe = _e22_recipe()
    assert recipe.row_evidence_gate and recipe.birth_death_isolation and recipe.serve_average == 4
    assert recipe.birth_death_space == "critic" and recipe.reopen_signal == "none"
    reals = _reals(40, batch=64)
    full = _trainer(recipe)
    full_out = _run(full, reals)
    first = _trainer(recipe)
    _run(first, reals[:25])
    checkpoint = copy.deepcopy(first.state_dict())
    resumed = _trainer(recipe, seed=5)
    resumed.load_state_dict(checkpoint)
    _assert_same_run(_run(resumed, reals[25:]), full_out[25:])
    _assert_same_models(resumed.state_dict(), full.state_dict())
