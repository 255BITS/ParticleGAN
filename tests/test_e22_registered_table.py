"""Explicit table ownership also works when a generator registers the table."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import E22Policy, GANTrainer, get_recipe


class RegisteredGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(16, 2, dtype=torch.float64))
        self.network = nn.Linear(2, 2).double()

    def forward(self, latent):
        return self.network(latent)


def make_policy():
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        particle_birth_death=False, row_evidence_gate=False,
                        birth_death_isolation=False, birth_death_feature_scale="none",
                        output_noise_mode="fixed", output_noise_std=.04, serve_average=4.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(761)
        generator = RegisteredGenerator()
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    opt_g = recipe.make_generator_optimizer(generator.network.parameters(), foreach=False)
    opt_table = recipe.make_generator_optimizer(
        [{"params": [generator.bank], "lr": recipe.lr * recipe.prior_lr_mult}],
        latent_table=generator.bank, foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    policy = E22Policy(recipe, generator, critic, table=generator.bank,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       table_optimizer=opt_table,
                       roles=[["generator"], ["critic"], ["table"]], seed=59)
    tester = policy._table_tester()
    tester.s, tester.b = .5, 4.
    return policy


def update(policy):
    real = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 16
    policy.begin_step(real)
    policy.opt_d.zero_grad()
    with torch.no_grad():
        fake = policy.generate(policy.table[:8])
    policy.observe_critic_pair(real, fake)
    loss = policy.recipe.make_loss()
    loss_d = loss.d_loss(policy.D(real), policy.D(fake))
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.opt_g.zero_grad()
    policy.table_optimizer.zero_grad()
    flags = [p.requires_grad for p in policy.D.parameters()]
    try:
        policy.D.requires_grad_(False)
        loss_g = loss.g_loss(policy.D(policy.generate(policy.table[:8])), policy.D(real))
        loss_g.backward()
        policy.after_generator_backward(loss_gan=loss_g, loss_critic=loss_d)
        policy.opt_g.step()
        policy.table_optimizer.step()
        policy.after_generator_step()
    finally:
        for parameter, flag in zip(policy.D.parameters(), flags):
            parameter.requires_grad_(flag)
    policy.finish_step()
    return loss_d.detach(), loss_g.detach()


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and left != left:
        assert right != right
    else:
        assert left == right


def test_registered_table_has_one_average_update_and_exact_resume():
    policy = make_policy()
    assert policy.averaged_table is policy.ema_G.bank
    assert len({id(p) for p in policy._served_parameters()}) == len(policy._served_parameters())
    old = policy.averaged_table.detach().clone()
    table = policy.table
    update(policy)
    rate = .5 / (4. * 4.)
    expected = old.mul_(1. - rate).add_(policy.table, alpha=rate)
    torch.testing.assert_close(policy.averaged_table, expected, rtol=0, atol=0)
    assert policy.table is table is policy.G.bank
    # The network signal excludes the table even when G registers it.
    assert policy.controller.previous_gradient.numel() == sum(p.numel() for p in policy.G.network.parameters())
    policy._table_tester().last_decisive = -1
    saved = policy.state_dict()
    reference_losses = update(policy)
    reference = policy.state_dict()
    restored = make_policy()
    identity = restored.table
    restored.load_state_dict(saved)
    assert restored.table is restored.G.bank is identity
    assert restored.averaged_table is restored.ema_G.bank
    assert_tree_equal(saved, restored.state_dict())
    assert_tree_equal(reference_losses, update(restored))
    assert_tree_equal(reference, restored.state_dict())


def test_registered_table_served_snapshot_owns_its_copied_table():
    policy = make_policy()
    update(policy)
    policy._table_tester().last_decisive = -1
    fast = deepcopy(policy.G.state_dict())
    snapshot = policy.served_snapshot()
    served = policy.served_model()
    assert served.source == snapshot["source"] == "averaged"
    assert served.table is served.generator.bank
    assert served.table is not policy.table and served.table is not policy.averaged_table
    assert_tree_equal(fast, policy.G.state_dict())
    assert_tree_equal(snapshot["table"], snapshot["models"]["generator"]["bank"])
    assert_tree_equal(served.table, policy.averaged_table)
    query = torch.zeros(3, 2, dtype=torch.float64)
    reference = served.generate(query, generator=torch.Generator().manual_seed(93))
    with torch.no_grad():
        policy.G.bank.add_(10.)
        policy.G.network.weight.add_(10.)
    actual = served.generate(query, generator=torch.Generator().manual_seed(93))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["table", "averaged_table"])
def test_conflicting_registered_table_checkpoint_alias_rejects_before_mutation(name):
    policy = make_policy()
    update(policy)
    policy._table_tester().last_decisive = -1
    before = policy.state_dict()
    snapshot = policy.served_snapshot()
    bad = deepcopy(before)
    # Replace this tensor so the malformed checkpoint conflicts with its
    # independently serialized module owner even if storage aliases survived.
    bad[name] = bad[name] + 1.
    bad["models"]["generator"]["network.weight"].zero_()
    with pytest.raises(ValueError, match="inconsistent.*alias"):
        policy.load_state_dict(bad)
    assert_tree_equal(before, policy.state_dict())
    assert_tree_equal(snapshot, policy.served_snapshot())


def test_embedded_and_explicit_prior_parameter_overlap_is_rejected():
    policy = make_policy()
    prior = nn.Module()
    prior.register_parameter("z", policy.table)
    with pytest.raises(ValueError, match="share parameters"):
        E22Policy(policy.recipe, policy.G, policy.D, prior=prior,
                  generator_optimizer=policy.opt_g, critic_optimizer=policy.opt_d,
                  table_optimizer=policy.table_optimizer)


class TiedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False).double()
        self.second = nn.Linear(2, 2, bias=False).double()
        self.second.weight = self.first.weight

    def forward(self, latent):
        return self.second(torch.tanh(self.first(latent)))


def test_tied_parameter_served_snapshot_matches_native_served_samples():
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        particle_birth_death=False, row_evidence_gate=False,
                        birth_death_isolation=False, birth_death_feature_scale="none",
                        output_noise_mode="fixed", output_noise_std=0.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(763)
        trainer = GANTrainer(recipe, TiedGenerator(),
                             nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).double())
    with torch.no_grad():
        trainer.G.first.weight.fill_(2.)
        trainer.ema_G.first.weight.fill_(3.)
    trainer._table_tester().last_decisive = -1
    trainer._serve_apply()
    snapshot = trainer.served_snapshot()
    assert torch.equal(snapshot["models"]["generator"]["first.weight"],
                       snapshot["models"]["generator"]["second.weight"])
    served = trainer.served_model()
    assert served.generator.first.weight is served.generator.second.weight
    torch.testing.assert_close(served.generator.first.weight,
                               torch.full((2, 2), 3., dtype=torch.float64), rtol=0, atol=0)
    expected = trainer.sample(11, generator=torch.Generator().manual_seed(94))
    actual = served.sample(11, generator=torch.Generator().manual_seed(94))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
