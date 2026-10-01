"""Caller-owned tables, separate role groups, and penalty/controller wiring."""
from copy import deepcopy
import io

import pytest
import torch
from torch import nn

from particlegan import E22Policy, get_recipe


def _components():
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        particle_birth_death=False, birth_death_isolation=False,
                        birth_death_feature_scale="none", row_evidence_gate=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        generator, encoder, router = (nn.Linear(2, 2).double() for _ in range(3))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).double()
        table = nn.Parameter(torch.randn(16, 2, dtype=torch.float64))
    opt_g = recipe.make_generator_optimizer([
        {"params": list(module.parameters())} for module in (generator, encoder, router)],
        foreach=False)
    opt_table = recipe.make_generator_optimizer(
        [{"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}],
        latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=encoder,
                       router=router, generator_optimizer=opt_g,
                       critic_optimizer=opt_d, table_optimizer=opt_table,
                       roles=[["generator", "encoder", "router"], ["critic"], ["table"]], seed=11)
    return policy


def _update(policy):
    real = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 16
    noise = policy.begin_step(real)
    latent = policy.table[:8]
    policy.opt_d.zero_grad()
    with torch.no_grad():
        fake = policy.generate(latent, sigma=noise.output_sigma)
    policy.observe_critic_pair(real, fake)
    loss_d = policy.recipe.make_loss().d_loss(policy.D(real), policy.D(fake))
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.opt_g.zero_grad()
    policy.table_optimizer.zero_grad()
    flags = [p.requires_grad for p in policy.D.parameters()]
    try:
        policy.D.requires_grad_(False)
        routed = policy.router(policy.encoder(policy.table[:8]))
        fake = policy.generate(routed, sigma=noise.output_sigma)
        loss_g = policy.recipe.make_loss().g_loss(policy.D(fake), policy.D(real))
        loss_g.backward()
        policy.after_generator_backward(loss_gan=loss_g, loss_critic=loss_d)
        policy.opt_g.step()
        policy.table_optimizer.step()
        policy.after_generator_step()
    finally:
        for p, flag in zip(policy.D.parameters(), flags):
            p.requires_grad_(flag)
    policy.finish_step()
    return loss_d.detach(), loss_g.detach()


def test_standalone_table_and_encoder_router_roles_resume_exactly():
    policy = _components()
    before = {name: deepcopy(module.state_dict())
              for name, module in (("encoder", policy.encoder), ("router", policy.router))}
    table_before = policy.table.detach().clone()
    _update(policy)
    assert not torch.equal(policy.table, table_before)
    for name in before:
        assert not torch.equal(getattr(policy, name).weight, before[name]["weight"])
    assert policy.roles == [["generator", "encoder", "router", "noise"], ["critic"], ["table"]]
    saved = policy.state_dict()
    # The checkpoint contains tensor state and ordinary containers only.
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    expected_losses = _update(policy)
    expected = policy.state_dict()
    restored = _components()
    restored.load_state_dict(saved)
    actual_losses = _update(restored)
    for a, b in zip(expected_losses, actual_losses):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual = restored.state_dict()
    for family in ("models", "averages"):
        for name in expected[family]:
            for key in expected[family][name]:
                torch.testing.assert_close(expected[family][name][key], actual[family][name][key],
                                           rtol=0, atol=0)
    torch.testing.assert_close(expected["table"], actual["table"], rtol=0, atol=0)
    assert set(restored.served_snapshot()["models"]) == {"generator", "critic", "encoder", "router"}


def test_new_penalties_bind_the_same_policy_controller():
    policy = _components()
    additional = policy.recipe.make_critic_penalty(policy.opt_d)
    assert additional.regularizer.continuous_controller is policy.controller
    assert policy.penalty.regularizer.continuous_controller is policy.controller


def test_role_topology_mismatch_is_rejected_before_restoring_weights():
    policy = _components()
    _update(policy)
    state = policy.state_dict()
    state["roles"][0][0] = "encoder"
    before = policy.G.weight.detach().clone()
    with pytest.raises(ValueError, match="roles"):
        policy.load_state_dict(state)
    torch.testing.assert_close(policy.G.weight, before, rtol=0, atol=0)
