"""Noisy game replay uses complete candidates and leaves training untouched."""
from copy import deepcopy
from dataclasses import replace
import importlib
import io
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GANLoss, RoutedRows, get_recipe
from particlegan.continuous import DataDriftController
from particlegan.ka2 import WARMUP_CALLS


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.fixture(scope="module")
def api():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
        game = importlib.import_module("e22_routed_game")
        sites = importlib.import_module("e22_routed_sites")
    return game, sites


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(previous)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            same(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


class Queries(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = nn.Linear(2, 2), nn.Linear(2, 2)
        with torch.no_grad():
            self.first.weight.copy_(.2 * torch.eye(2))
            self.second.weight.copy_(.3 * torch.eye(2))
            self.first.bias.zero_()
            self.second.bias.zero_()
        self.register_buffer("log_mass", torch.tensor([0., 0., -torch.inf, 0.]))


class ErrorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 3), nn.Tanh())
        self.score = nn.Linear(3, 1)
        self.register_buffer("scale", torch.ones(2))
        with torch.no_grad():
            self.features[0].weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]]))
            self.features[0].bias.copy_(torch.tensor([.05, -.04, .02]))
            self.score.weight.copy_(torch.tensor([[.8, 1., .3]]))
            self.score.bias.fill_(.1)

    def forward(self, error):
        return self.score(self.features(error).mean(1))


def complete_forward(models, context, candidate, routing):
    router, generator = models["router"], models["generator"]
    first = routing.mix("first", router.first(context) @ candidate.table.T / math.sqrt(2))
    hidden = generator(first) + .02 * context
    second = routing.mix("second", router.second(hidden) @ candidate.table.T / math.sqrt(2))
    return generator((first + second) / 2) + .02 * context


def features(models, context, samples, targets):
    return models["critic"].features((samples - targets) / models["critic"].scale).flatten(1)


def owners(*, device="cpu", phase="a"):
    # One completely specified fixture, rather than a search over seeds.
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        generator, router, critic = nn.Linear(2, 2), Queries(), ErrorCritic()
    with torch.no_grad():
        generator.weight.copy_(torch.eye(2))
        generator.bias.zero_()
    models = {"generator": generator, "router": router, "critic": critic}
    models = {name: module.to(device=device, dtype=torch.float64).eval() for name, module in models.items()}
    table = nn.Parameter(torch.tensor([[.2, .25], [.6, .7], [9., 11.], [1., 1.1]],
                                     dtype=torch.float64, device=device))
    recipe = get_recipe("e22_routed", num_particles=4, z_dim=2, batch_size=4,
                        row_evidence_gate=False, particle_birth_death=False)
    controller = DataDriftController("dv12")
    controller.observe_prior(controller.routed_prior(table, router.log_mass))
    optimizer = recipe.make_critic_optimizer(models["critic"], ema_critic=deepcopy(models["critic"]), foreach=False)
    optimizer.continuous_controller = controller
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True, coeff=1.7, kappa=.3)
    if phase == "blend":
        optimizer.record.calls = WARMUP_CALLS - 1
        optimizer.record.observed_steps = WARMUP_CALLS - 1
        optimizer.record.anchor_started = True
        optimizer.record.last_sur = .1
        with torch.no_grad():
            for value in optimizer.ema_critic.parameters():
                value.mul_(.7)
    rows = RoutedRows(model_forward=complete_forward, features=features, sites=("first", "second"))
    context = torch.linspace(-.4, .4, 24, device=device, dtype=torch.float64).reshape(4, 3, 2)
    target = torch.zeros_like(context)
    return models, table, controller, optimizer, penalty, rows, context, target


def capture(api, fixture, *, draws=1, sigma=.65, units="token"):
    game, _ = api
    models, table, controller, optimizer, penalty, rows, _, _ = fixture
    latent = torch.Generator(device=table.device).manual_seed(71)
    paired = torch.Generator(device=table.device).manual_seed(72)
    latent_states, paired_states = [], []
    for _ in range(draws):
        latent_states.append(latent.get_state())
        paired_states.append(paired.get_state())
        # Predeclare distinct diagnostic draws from the same private streams.
        torch.randn((17,), device=table.device, generator=latent)
        torch.randn((19,), device=table.device, generator=paired)
    return game.capture_game(models=models, table=table, controller=controller,
                             critic_optimizer=optimizer, penalty=penalty, rows=rows,
                             output_sigma=sigma, latent_rng_states=latent_states,
                             paired_rng_states=paired_states, penalty_units=units)


def split(candidate):
    state = {name: value.clone() for name, value in candidate.row_state.items()}
    table = candidate.table.clone()
    table[2].copy_(table[1])
    state["log_mass"][1].sub_(math.log(2))
    state["log_mass"][2].copy_(state["log_mass"][1])
    return replace(candidate, table=table, log_mass=state["log_mass"], row_state=state)


@pytest.mark.parametrize("phase", ["a", "blend"])
def test_complete_two_role_replay_matches_private_noise_loss_and_next_KA2_phase(api, phase):
    fixture = owners(phase=phase)
    models, _, controller, optimizer, _, rows, context, targets = fixture
    replay = capture(api, fixture)
    candidate = replay.candidate()
    receipt = replay.compare(context, targets, candidate, candidate, residual_scale=models["critic"].scale)
    draw = receipt["draws"][0]
    actual = draw["before"]
    assert receipt["probe"] == "fixed_critic_gradient_response"
    assert "output_mse_log" not in actual
    assert actual["penalty_phase"] == phase
    assert actual["penalty_calls_before"] == optimizer.record.calls
    assert actual["penalty_calls_after"] == optimizer.record.calls + 1
    assert (actual["penalty_stats"]["prox"] > 0) == (phase == "blend")
    assert [entry["role"] for entry in actual["dv12"]] == ["critic", "critic", "generator", "generator"]
    assert [entry["shape"] for entry in actual["dv12"]] == [(12, 2)] * 4
    assert all(entry["rms"] > 0 for entry in actual["dv12"])
    assert actual["role_gradient_norm"]["critic"] > 0
    assert actual["role_gradient_norm"]["generator"] > 0
    assert actual["role_gradient_norm"]["router"] > 0
    assert all(value == 0. for value in draw["delta"].values())

    latent = torch.Generator().set_state(replay.latent_states[0])
    paired = torch.Generator().set_state(replay.paired_states[0])
    copied_controller = deepcopy(controller)
    prior = copied_controller.routed_prior(candidate.table, candidate.log_mass)
    perturb = lambda codes: copied_controller.perturb_latent(codes, latent, prior, record=False)
    critic_noise = replay.output_sigma * torch.randn(targets.shape, dtype=targets.dtype, generator=paired)
    generator_noise = replay.output_sigma * torch.randn(targets.shape, dtype=targets.dtype, generator=paired)
    assert not torch.equal(critic_noise, generator_noise)
    with torch.no_grad():
        prediction_d = rows.forward(models, context, candidate, perturb_fn=perturb)
        prediction_g = rows.forward(models, context, candidate, perturb_fn=perturb)
        critic, loss = models["critic"], GANLoss()
        expected_d = loss.d_loss(critic(critic_noise), critic(critic_noise + prediction_d))
        expected_g = loss.g_loss(critic(generator_noise + prediction_g), critic(generator_noise))
        wrong_reused_g = loss.g_loss(critic(critic_noise + prediction_d), critic(critic_noise))
    assert actual["loss_d_game"] == pytest.approx(float(expected_d), abs=2e-15)
    assert actual["loss_g"] == pytest.approx(float(expected_g), abs=2e-15)
    assert abs(actual["loss_g"] - float(wrong_reused_g)) > 1e-4
    same(receipt, replay.compare(context, targets, candidate, candidate, residual_scale=1.))


@pytest.mark.parametrize("device", DEVICES)
def test_exact_mass_refinement_preserves_noisy_game_and_aggregate_gradient_geometry(api, device):
    fixture = owners(device=device)
    replay = capture(api, fixture, draws=3)
    before = replay.candidate()
    after = split(before)
    context, targets = fixture[-2:]
    receipt = replay.compare(context, targets, before, after, residual_scale=1., split_rows=(1, 2))
    for draw in receipt["draws"]:
        for field in ("loss_g", "loss_d_game", "loss_d_total", "penalty", "noisy_feature_error",
                      "critic_noisy_feature_error", "clean_feature_error", "mass_gradient_effective_rank",
                      "mass_gradient_concentration"):
            assert draw["after"][field] == pytest.approx(draw["before"][field], rel=3e-12, abs=3e-14)
        assert draw["split_aggregate_table_gradient_cosine"] == pytest.approx(1., abs=2e-14)
        for value in draw["global_gradient_cosine"].values():
            assert value == pytest.approx(1., abs=2e-14)
        grad_before = torch.tensor(draw["before"]["table_gradient"], dtype=torch.float64)
        grad_after = torch.tensor(draw["after"]["table_gradient"], dtype=torch.float64)
        torch.testing.assert_close(grad_after[1], grad_before[1] / 2, rtol=3e-12, atol=3e-14)
        torch.testing.assert_close(grad_after[2], grad_before[1] / 2, rtol=3e-12, atol=3e-14)
        assert draw["before"]["row_usage"][2] == 0.
        assert draw["before"]["row_context_ess"][2] == 0.
        assert draw["after"]["row_context_ess"][2] <= len(context) + 1e-12
    same(receipt, replay.compare(context, targets, before, after, residual_scale=1., split_rows=(1, 2)))


def test_clean_feature_gain_can_worsen_the_actual_noisy_fixed_critic_generator_payoff(api):
    fixture = owners()
    replay = capture(api, fixture, draws=4)
    before = replay.candidate()
    after = replace(before, table=before.table - .25)
    receipt = replay.compare(*fixture[-2:], before, after, residual_scale=1., log_output_error=True)
    assert receipt["mean_delta"]["clean_feature_error"] < -.1
    assert receipt["mean_delta"]["output_mse_log"] < -.1
    assert receipt["mean_delta"]["loss_g"] > .02
    for draw in receipt["draws"]:
        arm = draw["before"]
        assert abs(arm["clean_feature_error"] - arm["noisy_feature_error"]) > .01
        assert abs(arm["critic_noisy_feature_error"] - arm["noisy_feature_error"]) > 1e-3
    # The output-error option adds logging; it does not influence game replay.
    plain = replay.compare(*fixture[-2:], before, after, residual_scale=1.)
    for logged, unlogged in zip(receipt["draws"], plain["draws"]):
        for arm in ("before", "after"):
            actual = deepcopy(logged[arm])
            del actual["output_mse_log"]
            same(actual, unlogged[arm])


def test_current_and_prospective_bandwidth_are_separate_copied_measurements(api):
    fixture = owners()
    replay = capture(api, fixture)
    before = replay.candidate()
    after = replace(before, table=2 * before.table)
    current = replay.compare(*fixture[-2:], before, after, residual_scale=1.)
    future = replay.compare(*fixture[-2:], before, after, residual_scale=1., prospective_bandwidth=True)
    old = fixture[2].latent_bandwidth
    assert current["bandwidth_mode"] == "current" and future["bandwidth_mode"] == "prospective"
    same(current["draws"][0]["before"]["latent_bandwidth"], current["draws"][0]["after"]["latent_bandwidth"])
    expected = old.lerp(DataDriftController.routed_geometry(after.table, after.log_mass)[1], .01)
    torch.testing.assert_close(torch.tensor(future["draws"][0]["after"]["latent_bandwidth"], dtype=torch.float64),
                               expected, rtol=0, atol=0)
    assert future["draws"][0]["after"]["loss_g"] != current["draws"][0]["after"]["loss_g"]
    same(old, replay._bundle["controller"].latent_bandwidth)


@pytest.mark.parametrize("device", DEVICES)
def test_capture_compare_and_CPU_checkpoint_reconstruction_leave_every_live_owner_unchanged(api, device):
    game, sites = api
    loop = sites.make_loop(mode="no_rows", device=device, tokens=2, particles=8, batch_size=4)
    for _ in range(3):
        sites.update(loop)
    policy = loop.policy
    models = {"generator": policy.G, "encoder": policy.encoder, "router": policy.router, "critic": policy.D}
    # Mode/gradient preservation must include all nested modules and frozen state.
    policy.G.train()
    policy.D.train()
    modes = {id(module): module.training for root in models.values() for module in root.modules()}
    parameters = dict.fromkeys([parameter for root in models.values() for parameter in root.parameters()]
                               + [policy.table, policy.log_output_sigma])
    gradients = {parameter: None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters}
    flags = {parameter: parameter.requires_grad for parameter in parameters}
    checkpoint = cpu_roundtrip(sites.checkpoint(loop))
    names = deepcopy(policy.penalty._names)
    penalty_stats = deepcopy(policy.penalty.last_stats)
    default_rng = torch.random.get_rng_state()
    device_rng = torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None
    latent_state = torch.Generator(device=device).manual_seed(71).get_state()
    paired_state = torch.Generator(device=device).manual_seed(72).get_state()

    def captured(p):
        return game.capture_game(models={"generator": p.G, "encoder": p.encoder, "router": p.router, "critic": p.D},
                                 table=p.table, controller=p.controller, critic_optimizer=p.opt_d,
                                 penalty=p.penalty, rows=p.routed_control.spec, output_sigma=p.output_sigma(),
                                 latent_rng_states=latent_state, paired_rng_states=paired_state)

    replay = captured(policy)
    candidate = replay.candidate()
    modified = replace(candidate, table=candidate.table + .01)
    before_candidate = deepcopy(candidate)
    receipt = replay.compare(loop.guard_context[:4], loop.guard_targets[:4], candidate, modified,
                             residual_scale=policy.D.scale)
    same(candidate.table, before_candidate.table)
    same(candidate.row_state, before_candidate.row_state)
    same(checkpoint, cpu_roundtrip(sites.checkpoint(loop)))
    same(names, policy.penalty._names)
    same(penalty_stats, policy.penalty.last_stats)
    same(default_rng, torch.random.get_rng_state())
    if device_rng is not None:
        same(device_rng, torch.cuda.get_rng_state(device))
    assert modes == {id(module): module.training for root in models.values() for module in root.modules()}
    assert flags == {parameter: parameter.requires_grad for parameter in parameters}
    for parameter, gradient in gradients.items():
        same(gradient, parameter.grad)

    restored = sites.make_loop(mode="no_rows", device=device, tokens=2, particles=8, batch_size=4)
    sites.restore(restored, checkpoint)
    reconstructed = captured(restored.policy)
    loaded = reconstructed.candidate()
    same(receipt, reconstructed.compare(restored.guard_context[:4], restored.guard_targets[:4], loaded,
                                        replace(loaded, table=loaded.table + .01), residual_scale=restored.policy.D.scale))
    same(checkpoint, cpu_roundtrip(sites.checkpoint(restored)))


def test_default_rng_callbacks_are_rejected_without_consuming_live_rng(api):
    game, _ = api
    fixture = list(owners())

    def stochastic(models, context, candidate, routing):
        return complete_forward(models, context, candidate, routing) + torch.randn_like(context)

    fixture[5] = RoutedRows(model_forward=stochastic, features=features, sites=("first", "second"))
    replay = capture(api, fixture)
    rng_before = torch.random.get_rng_state()
    candidate = replay.candidate()
    with pytest.raises(ValueError, match="deterministic.*default RNG"):
        replay.compare(*fixture[-2:], candidate, candidate, residual_scale=1.)
    same(rng_before, torch.random.get_rng_state())


def test_global_gradient_vectors_keep_unused_parameter_coordinates(api):
    fixture = owners()
    generator = fixture[0]["generator"]
    generator.register_parameter("unused", nn.Parameter(torch.ones(3, dtype=torch.float64)))
    replay = capture(api, fixture)
    candidate = replay.candidate()
    result = replay.compare(*fixture[-2:], candidate, candidate, residual_scale=1.)
    vector = result["draws"][0]["before"]["role_gradient"]["generator"]
    assert len(vector) == sum(parameter.numel() for parameter in generator.parameters())
    assert vector[-3:] == [0., 0., 0.]


def test_invalid_draw_states_units_and_candidate_ownership_fail_explicitly(api):
    game, _ = api
    fixture = owners()
    models, table, controller, optimizer, penalty, rows, context, targets = fixture
    options = dict(models=models, table=table, controller=controller, critic_optimizer=optimizer,
                   penalty=penalty, rows=rows, output_sigma=.65,
                   latent_rng_states=torch.Generator().manual_seed(71).get_state(),
                   paired_rng_states=torch.Generator().manual_seed(72).get_state())
    with pytest.raises(ValueError, match="Generator states"):
        game.capture_game(**{**options, "latent_rng_states": torch.zeros(3)})
    with pytest.raises(ValueError, match="same number"):
        game.capture_game(**{**options, "paired_rng_states": [options["paired_rng_states"]] * 2})
    with pytest.raises(ValueError, match="token/context"):
        game.capture_game(**options, penalty_units="flat_score")
    replay = game.capture_game(**options)
    candidate = replay.candidate()
    with pytest.raises(ValueError, match="ownership shapes"):
        replay.compare(context, targets, candidate, replace(candidate, table=candidate.table[:3]), residual_scale=1.)
    with pytest.raises(ValueError, match="parent and child"):
        replay.compare(context, targets, candidate, candidate, residual_scale=1., split_rows=(1, 1))
