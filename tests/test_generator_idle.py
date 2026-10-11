"""Opt-in generator idle rule (closed PR #366, rebuilt on the recipe factories)."""
import copy
import hashlib

import pytest
import torch
from torch import nn

from experiments.forge.mechanisms import MechanismAudit, mechanism_blockers
from experiments.forge.state import state_digest
from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.generator_idle import (
    IdleRecordingLoss, _GeneratorIdle, generator_is_idle)
from particlegan.training import _normalized_recipe

# GANTrainer state after 6 seed-0 bcap steps on develop 37e3668b5 (rule absent).
DEVELOP_DIGEST = "1d611edd4389a47747cf923951a44994b3d0ed438c6165ee6ec2e16770e3d12f"
DEVELOP_GRAD_SUM = "cb247c7cb7c48380e70bd40da96c0dc286884bbd1ce4738a9430c07bae44ade9"


def _models(recipe):
    G = nn.Sequential(nn.Linear(recipe.z_dim, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    prior = init.deterministic_orthogonal_(recipe.make_prior())
    return G, D, prior


def _real(step, n):
    generator = torch.Generator().manual_seed(1000 + step)
    angle = torch.randint(8, (n, 1), generator=generator) * torch.pi / 4
    return torch.cat([angle.cos(), angle.sin()], 1) + .05 * torch.randn(n, 2, generator=generator)


def _trainer(steps=6, run=None, **overrides):
    torch.manual_seed(0)
    recipe = get_recipe("bcap", total_steps=steps, batch_size=32, num_particles=64, **overrides)
    G, D, prior = _models(recipe)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    grads = []
    for step in range(steps if run is None else run):
        trainer.step(_real(step, recipe.batch_size))
        grads.append(sum(float(p.grad.double().sum()) for p in G.parameters() if p.grad is not None))
    return trainer, grads


def test_rule_edge_cases():
    ones = torch.ones(8)
    assert not generator_is_idle(ones, torch.zeros(8), 1.)          # constant nonzero gap, SE 0
    assert generator_is_idle(ones, ones, 1.)                          # constant zero gap
    assert generator_is_idle(ones, ones, 0.)
    assert not generator_is_idle(torch.ones(1), torch.ones(1), 1.)   # n < 2
    assert not generator_is_idle(torch.tensor([1., float("nan")]), torch.zeros(2), 1.)
    assert not generator_is_idle(torch.tensor([1., float("inf")]), torch.zeros(2), 1.)
    noisy = torch.tensor([1., -1., 1.2, -.9])
    assert generator_is_idle(noisy, torch.zeros(4), 1.)
    assert not generator_is_idle(noisy + 3., torch.zeros(4), 1.)


def test_recipe_field_validation_and_restore():
    assert Recipe().generator_idle_se is None
    assert get_recipe("bcap", generator_idle_se=1).generator_idle_se == 1.0
    for bad in (True, -1., float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError):
            get_recipe("bcap", generator_idle_se=bad)
    assert "generator_idle_se" not in _normalized_recipe({"generator_idle_se": None})
    assert _normalized_recipe({"generator_idle_se": 1.0})["generator_idle_se"] == 1.0
    assert type(get_recipe("bcap").make_loss(generator_optimizer=object())).__name__ == "GANLoss"


def test_off_path_matches_develop_and_keeps_checkpoint_keys():
    trainer, grads = _trainer()
    assert "generator_idle" not in trainer.opt_g.state_dict()
    assert not isinstance(trainer.opt_g, _GeneratorIdle)
    assert state_digest(trainer.state_dict()) == DEVELOP_DIGEST
    assert hashlib.sha256(repr(grads).encode()).hexdigest() == DEVELOP_GRAD_SUM


def test_on_path_trainer_logs_decisions_and_round_trips():
    trainer, _ = _trainer(generator_idle_se=1.)
    opt = trainer.opt_g
    assert opt.idle_decisions == 6 and opt.last_idle in (True, False)
    saved = trainer.state_dict()
    assert saved["optimizers"][0]["generator_idle"]["idle_steps"] == opt.idle_steps
    restored, _ = _trainer(steps=6, run=0, generator_idle_se=1.)
    restored.load_state_dict(copy.deepcopy(saved))
    assert (restored.opt_g.idle_steps, restored.opt_g.idle_decisions) == (opt.idle_steps, 6)
    assert state_digest(restored.state_dict()) == state_digest(saved)
    with pytest.raises(ValueError):
        restored.opt_g.load_state_dict({k: v for k, v in opt.state_dict().items() if k != "generator_idle"})


def _one_g_step(se, gap):
    recipe = get_recipe("bcap", total_steps=4, batch_size=16, num_particles=32, generator_idle_se=se)
    G, D, prior = _models(recipe)
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
    z, rows = prior.sample(16, generator=torch.Generator().manual_seed(3))
    loss = D(G(z)).mean()
    opt_g.zero_grad()
    opt_g.set_sampled_rows(prior.z, rows)
    constraint_geometry_backward(loss, opt_g, (loss,))
    if gap is not None:
        opt_g.record_critic_gap(gap, torch.zeros_like(gap))
    before = (copy.deepcopy(G.state_dict()), prior.z.detach().clone(), copy.deepcopy(opt_g.state_dict()))
    opt_g.step()
    return opt_g, G, prior, before


def test_idle_step_changes_nothing_and_active_step_matches_unwrapped():
    opt_g, G, prior, (g_before, z_before, state_before) = _one_g_step(1., torch.tensor([.1, -.1, .2, -.2]))
    assert opt_g.idle_steps == 1 and opt_g.last_idle is True
    assert all(torch.equal(G.state_dict()[k], v) for k, v in g_before.items())
    assert torch.equal(prior.z.detach(), z_before)
    state = opt_g.state_dict()
    assert state["state"] == state_before["state"] and opt_g.sampled_rows_for(prior.z) is None
    assert opt_g._protected is None
    active, G_on, prior_on, _ = _one_g_step(1., torch.tensor([3., 3.1, 2.9, 3.]))
    plain, G_off, prior_off, _ = _one_g_step(None, None)
    assert active.last_idle is False and active.idle_steps == 0
    assert all(torch.equal(G_on.state_dict()[k], G_off.state_dict()[k]) for k in G_off.state_dict())
    assert torch.equal(prior_on.z, prior_off.z)


def test_step_without_record_raises():
    with pytest.raises(RuntimeError, match="record_critic_gap"):
        _one_g_step(1., None)


def test_critic_still_steps_while_generator_idles():
    trainer, _ = _trainer(steps=6, generator_idle_se=1e6)  # every gap is unresolved
    assert trainer.opt_g.idle_steps == 6
    G, D, prior = _models(trainer.recipe)
    assert all(torch.equal(a, b) for a, b in zip(trainer.G.state_dict().values(), G.state_dict().values()))
    assert torch.equal(trainer.prior.z.detach(), prior.z.detach())
    assert not all(torch.equal(a, b) for a, b in zip(trainer.D.state_dict().values(), D.state_dict().values()))
    audit = MechanismAudit(trainer.recipe, trainer.opt_d, [trainer.opt_g]).receipt()
    row = audit["mechanisms"]["generator_idle"]
    assert (row["applied"], row["calls"], row["idle_fraction"]) == (6, 6, 1.)
    assert not [b for b in mechanism_blockers(audit) if b.startswith("generator_idle")]


def test_unexercised_rule_is_proved_by_probe():
    trainer, _ = _trainer(steps=2, generator_idle_se=0.)  # zero threshold: never idle on noisy gaps
    audit = MechanismAudit(trainer.recipe, trainer.opt_d, [trainer.opt_g]).receipt()
    row = audit["mechanisms"]["generator_idle"]
    assert row["applied"] == 0 and row["probe"]["measurements"]["idle_branch_applied"] is True
    assert not [b for b in mechanism_blockers(audit) if b.startswith("generator_idle")]


def test_user_loop_binding_under_constraint_geometry():
    recipe = get_recipe("bcap", total_steps=4, batch_size=16, num_particles=32, generator_idle_se=1.)
    G, D, prior = _models(recipe)
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
    penalty, gan = recipe.make_critic_penalty(opt_d), recipe.make_loss(generator_optimizer=opt_g)
    assert isinstance(gan, IdleRecordingLoss)
    for step in range(4):
        real = _real(step, 16)
        z, indices = prior.sample(16, generator=torch.Generator().manual_seed(step))
        fake = G(z)
        d_loss = gan.d_loss(D(real), D(fake.detach())) + penalty(D, real, fake.detach())
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        g_loss = gan.g_loss(D(fake), D(real))
        opt_g.zero_grad()
        opt_g.set_sampled_rows(prior.z, indices)
        constraint_geometry_backward(g_loss, opt_g, (g_loss,))
        opt_g.step()
    assert opt_g.idle_decisions == 4
    with pytest.raises(ValueError, match="real critic scores"):
        gan.g_loss(D(fake))
