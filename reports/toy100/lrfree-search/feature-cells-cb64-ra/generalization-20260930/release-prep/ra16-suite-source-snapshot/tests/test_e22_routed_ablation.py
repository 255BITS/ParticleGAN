"""Evidence-only and frozen-bank routed loops keep distinct update lifecycles."""
from copy import deepcopy
import io
import math
from pathlib import Path
import runpy

import pytest
import torch
from torch import nn

from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(previous)


def test_evidence_only_refreshes_real_counterfactuals_without_structural_mutation_and_resumes():
    path = Path(__file__).resolve().parents[1] / "examples" / "e22_routed_paired.py"
    api = runpy.run_path(str(path))
    options = {"particle_birth_death": False}
    loop = api["make_loop"](recipe_overrides=options)
    policy = loop.policy
    assert policy.row_evidence is not None and policy.birth_death is None
    original_finish = policy.finish_step

    def observed_finish():
        # Compare immediately after the optimizer steps so ordinary learning
        # may change the bank, but the diagnostic refresh cannot restructure it.
        table = policy.table.detach().clone()
        router = deepcopy(policy.router.state_dict())
        event = original_finish()
        assert_tree_equal(table, policy.table)
        assert_tree_equal(router, policy.router.state_dict())
        assert event is None or event["moves"] == 0
        return event

    policy.finish_step = observed_finish
    initial_table = policy.table.detach().clone()
    initial_mass = policy.router.log_mass.detach().clone()
    rows, saved = [], None
    for index in range(20):
        rows.append(api["update"](loop))
        if index == 9:
            saved = cpu_roundtrip(api["checkpoint"](loop))
    control = policy.routed_control
    assert control.counters["probes"] > 0 and control.counters["evals"] > 0
    assert policy.row_evidence.valid
    assert control.counters["proposals"] == control.counters["moves"] == control.counters["splits"] == 0
    assert control.moved_rows is None and control.moved_parameters == {}
    assert not torch.equal(initial_table, policy.table)
    assert_tree_equal(initial_mass, policy.router.log_mass)
    expected = api["checkpoint"](loop)
    output = policy.served_model().routed_forward(loop.test_context)

    restored = api["make_loop"](recipe_overrides=options)
    api["restore"](restored, saved)
    for expected_row in rows[10:]:
        assert_tree_equal(expected_row, api["update"](restored))
    assert_tree_equal(expected, api["checkpoint"](restored))
    assert_tree_equal(output, restored.policy.served_model().routed_forward(restored.test_context))


class SitesGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 2)


class SitesRouter(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 2)
        self.register_buffer("log_mass", torch.zeros(rows))


class SitesCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 4), nn.Tanh())
        self.score = nn.Linear(4, 1)

    def forward(self, error):
        return self.score(self.features(error))


def sites_forward(models, context, candidate, routing):
    encoded = models["encoder"](context).tanh()
    query = models["router"].first(encoded)
    first = routing.mix("first", query @ candidate.table.T)
    hidden = models["generator"].first(encoded + first).tanh()
    query = models["router"].second(hidden)
    second = routing.mix("second", query @ candidate.table.T)
    return models["generator"].second(hidden + second)


def sites_features(models, context, samples, targets):
    return models["critic"].features(samples - targets).flatten(1)


def frozen_components(*, evidence=False, birth_death=False):
    recipe = get_recipe("e22_routed", num_particles=16, z_dim=2, batch_size=6,
                        row_evidence_gate=evidence, particle_birth_death=birth_death,
                        output_noise_std=.125)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(927)
        G, D, E, R = SitesGenerator(), SitesCritic(), nn.Linear(2, 2), SitesRouter(16)
        table = nn.Parameter(torch.randn(16, 2), requires_grad=False)
    opt_g = recipe.make_generator_optimizer([
        {"params": list(G.parameters())}, {"params": list(E.parameters())},
        {"params": list(R.parameters())},
        {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult},
    ], latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    rows = RoutedRows(model_forward=sites_forward, features=sites_features, sites=("first", "second"))
    return recipe, G, D, E, R, table, opt_g, opt_d, rows


def make_frozen_policy():
    recipe, G, D, E, R, table, opt_g, opt_d, rows = frozen_components()
    return E22Policy(recipe, G, D, encoder=E, router=R, table=table,
                     generator_optimizer=opt_g, critic_optimizer=opt_d,
                     roles=[["generator", "encoder", "router", "table"], ["critic"]],
                     routed_rows=rows, seed=31)


def frozen_update(policy, *, routed=None):
    context = torch.arange(24, dtype=policy.dtype).reshape(6, 2, 2) / 24
    target = .2 * context + .1 * context.cos()
    noise = policy.begin_step(target, routed=routed)
    loss = policy.recipe.make_loss()
    policy.D.train()
    with torch.no_grad():
        fake_prediction = policy.routed_generate(context, sigma=0)
        real = noise.output_sigma * torch.randn(target.shape, generator=policy.noise_generator)
        fake = real + fake_prediction - target
    policy.observe_critic_pair(real, fake)
    penalty = policy.penalty(policy.D, real, fake)
    loss_d = loss.d_loss(policy.D(real), policy.D(fake)) + penalty
    policy.opt_d.zero_grad()
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    flags = [p.requires_grad for p in policy.D.parameters()]
    try:
        policy.D.eval().requires_grad_(False)
        fake_prediction = policy.routed_generate(context, sigma=0)
        real = noise.output_sigma * torch.randn(target.shape, generator=policy.noise_generator)
        loss_g = loss.g_loss(policy.D(real + fake_prediction - target), policy.D(real.detach()))
        policy.opt_g.zero_grad()
        loss_g.backward()
        assert policy.table.grad is None
        policy.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=(loss_d - penalty).detach())
        policy.opt_g.step()
        policy.after_generator_step()
    finally:
        for parameter, flag in zip(policy.D.parameters(), flags):
            parameter.requires_grad_(flag)
    event = policy.finish_step()
    assert event is None
    return loss_d.detach(), loss_g.detach()


def test_frozen_two_site_bank_trains_without_row_observation_and_resumes_exactly():
    policy = make_frozen_policy()
    initial = policy.state_dict()
    context = torch.zeros(3, 2, 2)
    for _ in range(10):
        frozen_update(policy)
    saved = cpu_roundtrip(policy.state_dict())
    expected_losses = [frozen_update(policy) for _ in range(10)]
    expected = policy.state_dict()
    output = policy.served_model().routed_forward(context)
    assert_tree_equal(initial["table"], policy.table)
    assert_tree_equal(policy.table, policy.averaged_table)
    assert not torch.equal(initial["models"]["generator"]["first.weight"], policy.G.first.weight)
    assert policy.log_output_sigma is not None and policy.log_output_sigma.requires_grad
    assert not torch.equal(initial["output_noise"], policy.log_output_sigma.detach())
    assert policy.row_evidence is policy.birth_death is None
    assert all(count == 0 for count in policy.routed_control.counters.values())
    assert policy.routed_control.latest_gradient is None
    assert not policy.routed_control.evidence.valid
    assert all(pool is None for pool in policy.routed_control.state_dict()["pools"].values())
    restored = make_frozen_policy()
    restored.load_state_dict(saved)
    assert_tree_equal(expected_losses, [frozen_update(restored) for _ in range(10)])
    assert_tree_equal(expected, restored.state_dict())
    assert_tree_equal(output, restored.served_model().routed_forward(context))


@pytest.mark.parametrize("evidence,birth_death", [(True, False), (False, True), (True, True)])
def test_frozen_bank_with_row_control_rejects_before_learned_noise_mutates_optimizer(evidence, birth_death):
    recipe, G, D, E, R, table, opt_g, opt_d, rows = frozen_components(
        evidence=evidence, birth_death=birth_death)
    before = deepcopy(opt_g.state_dict())
    groups = len(opt_g.param_groups)
    with pytest.raises(ValueError, match="frozen table|trainable particle table"):
        E22Policy(recipe, G, D, table=table, encoder=E, router=R,
                  generator_optimizer=opt_g, critic_optimizer=opt_d, routed_rows=rows)
    assert len(opt_g.param_groups) == groups
    assert_tree_equal(before, opt_g.state_dict())


def test_disabled_row_controls_validate_optional_observation_without_populating_pools():
    policy = make_frozen_policy()
    before = policy.state_dict()
    with pytest.raises(TypeError, match="RoutedBatch"):
        policy.begin_step(torch.zeros(6, 2, 2), routed={})
    repeated = torch.zeros(6, 2, 2)
    overlap = RoutedBatch(repeated, repeated, repeated.clone(), repeated.clone())
    with pytest.raises(ValueError, match="guard|overlap"):
        policy.begin_step(repeated, routed=overlap)
    assert_tree_equal(before, policy.state_dict())
    batch = RoutedBatch(torch.zeros(6, 2, 2), torch.zeros(6, 2, 2),
                        torch.ones(6, 2, 2), torch.zeros(6, 2, 2))
    frozen_update(policy, routed=batch)
    assert all(pool is None for pool in policy.routed_control.state_dict()["pools"].values())
