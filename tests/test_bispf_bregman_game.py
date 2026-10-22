"""Cheap first-principles controls; these are not trained toy qualifications."""
import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan.bregman_game import bregman_cost_matrix, critic_game_loss, generator_game_loss


def test_stable_sigmoid_force_restores_both_saturated_directions():
    for value, target in ((-40., 1.), (40., 0.)):
        a = torch.full((2, 1), value, dtype=torch.float64, requires_grad=True)
        y = torch.full_like(a, target)
        cost = bregman_cost_matrix(a.sigmoid(), y, kind="sigmoid", prelink=a)
        loss = generator_game_loss(cost, torch.zeros(2))
        gradient, = torch.autograd.grad(loss, a)
        expected = (a.sigmoid() - y) / 2
        torch.testing.assert_close(gradient, expected)
        assert torch.isfinite(cost).all() and gradient.abs().min() > .49


def test_same_moments_wrong_shape_has_nonzero_cost_at_declared_zero_prices():
    x = torch.tensor([[-math.sqrt(2)], [0.], [0.], [math.sqrt(2)]], dtype=torch.float64)
    y = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    torch.testing.assert_close(x.mean(), y.mean())
    torch.testing.assert_close(x.square().mean(), y.square().mean())
    value = generator_game_loss(bregman_cost_matrix(x, y), torch.zeros(4))
    assert value > .29


def test_missing_mode_can_have_zero_value_but_correct_nonzero_dual_price_force():
    x = torch.tensor([[-3.], [-3.], [-1.], [1.]], dtype=torch.float64)
    y = torch.tensor([[-3.], [-1.], [1.], [3.]], dtype=torch.float64)
    price = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    cost = bregman_cost_matrix(x, y)
    assert generator_game_loss(cost, price).item() == 0
    gradient, = torch.autograd.grad(critic_game_loss(cost, price), price)
    torch.testing.assert_close(gradient, torch.tensor([.25, 0., 0., -.25], dtype=torch.float64))


def test_matched_constant_law_has_zero_cost_force_and_no_epsilon_amplification():
    for kind in ("identity", "sigmoid"):
        a = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)
        x = a if kind == "identity" else a.sigmoid()
        y = x.detach().clone()
        c = bregman_cost_matrix(x, y, kind=kind, prelink=a if kind == "sigmoid" else None)
        force, = torch.autograd.grad(generator_game_loss(c, torch.zeros(4)), a)
        assert c.abs().max() < 1e-14 and force.abs().max() == 0
        shared_price = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        price_force, = torch.autograd.grad(critic_game_loss(c, shared_price.expand(4)), shared_price)
        assert price_force == 0


def test_gauge_invariance_target_detachment_and_rng_preservation():
    x = torch.tensor([[-1.3], [.7]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[-1.], [1.]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
    before = torch.random.get_rng_state().clone()
    c = bregman_cost_matrix(x, y)
    torch.testing.assert_close(critic_game_loss(c, p), critic_game_loss(c, p+17))
    first, = torch.autograd.grad(generator_game_loss(c, p), x, retain_graph=True)
    second, = torch.autograd.grad(generator_game_loss(c, p+17), x, retain_graph=True)
    torch.testing.assert_close(first, second)
    _, real_grad, price_grad = torch.autograd.grad(generator_game_loss(c, p), (x,y,p), allow_unused=True)
    assert real_grad is None and price_grad is None
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("kind", ["identity", "sigmoid"])
def test_optional_diagnostics_preserve_loss_gradient_rng_and_match_canonical_force(kind):
    a = torch.tensor([[-1.3, .2], [.7, 1.1]], dtype=torch.float64, requires_grad=True)
    x = a.sigmoid() if kind == "sigmoid" else a
    y = torch.tensor([[.1, .8], [.9, .2]], dtype=torch.float64)
    prices = torch.tensor([.02, -.01], dtype=torch.float64)
    arguments = {"kind": kind, "prelink": a if kind == "sigmoid" else None}
    cost = bregman_cost_matrix(x, y, **arguments)
    before = torch.get_rng_state().clone()
    plain = generator_game_loss(cost, prices, thermal_scale=.3)
    plain_gradient, = torch.autograd.grad(plain, a, retain_graph=True)
    diagnostics = {}
    observed = generator_game_loss(cost, prices, thermal_scale=.3,
        diagnostics=diagnostics, fake=x, real=y, **arguments)
    observed_gradient, = torch.autograd.grad(observed, a)
    assert torch.equal(plain, observed) and torch.equal(plain_gradient, observed_gradient)
    assert torch.equal(before, torch.get_rng_state())
    assert all(type(value) is float for value in diagnostics.values())
    assert diagnostics["price_min"] == pytest.approx(-.01)
    assert diagnostics["price_max"] == pytest.approx(.02)
    assert diagnostics["price_std"] == pytest.approx(.015)
    assert diagnostics["canonical_thermal_force_rms"] == pytest.approx(float(observed_gradient.square().mean().sqrt()))
    assert diagnostics["canonical_thermal_force_max"] == pytest.approx(float(observed_gradient.abs().max()))
    assert diagnostics["canonical_force_rms"] * .3 == pytest.approx(diagnostics["canonical_thermal_force_rms"])
    assert diagnostics["canonical_force_max"] * .3 == pytest.approx(diagnostics["canonical_thermal_force_max"])
    assert 0 < diagnostics["used_target_fraction"] <= 1
    assert 0 <= diagnostics["route_entropy_nats"] <= math.log(len(y))
    assert diagnostics["output_min"] == float(x.detach().min())
    if kind == "sigmoid":
        assert diagnostics["prelink_min"] == -1.3
    else:
        assert "prelink_min" not in diagnostics


def test_saturation_route_observer_uses_only_existing_cost_and_reports_known_force(monkeypatch):
    import particlegan.bregman_game as mechanism
    a = torch.full((2, 1), 40., requires_grad=True)
    x, y = a.sigmoid(), torch.zeros_like(a)
    cost = bregman_cost_matrix(x, y, kind="sigmoid", prelink=a)
    def forbidden(*args, **kwargs):
        raise AssertionError("the observer must not rebuild a cost matrix")
    monkeypatch.setattr(mechanism, "bregman_cost_matrix", forbidden)
    diagnostics = {}
    loss = generator_game_loss(cost, torch.zeros(2), thermal_scale=.2,
        diagnostics=diagnostics, fake=x, real=y, kind="sigmoid", prelink=a)
    assert diagnostics["output_exact_endpoint_fraction"] == 1.
    assert diagnostics["output_upper_near_saturation_fraction"] == 1.
    assert diagnostics["output_lower_near_saturation_fraction"] == 0.
    assert diagnostics["prelink_abs_ge16_fraction"] == 1.
    assert diagnostics["used_target_fraction"] == .5
    assert diagnostics["route_entropy_nats"] == 0.
    assert diagnostics["selected_cost_mean"] == 40.
    assert diagnostics["canonical_force_rms"] == .5
    assert diagnostics["canonical_thermal_force_rms"] == pytest.approx(.1)
    assert loss.requires_grad


def _public_trainer(active=True, image=False):
    from particlegan import GANTrainer, get_recipe, init
    from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator
    from benchmarks.transfer_suite.image_tasks import Generator, Discriminator, TASKS
    recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=2, batch_size=8,
                        total_steps=4, bispf_bregman_game=active,
                        prior_kind="mog", sigma_rel=.025, standardize=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if image:
            spec = {**next(s for s in TASKS if s["name"] == "img_intensity2"), "z_dim": 2}
            generator, critic = Generator(spec), Discriminator(spec)
        else:
            generator = SimpleMLPGenerator(2, 8, 1, 2)
            critic = SimpleMLPDiscriminator(2, 8, 1, fourier=0)
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            init.deterministic_orthogonal_(module, seed=index)
        streams = {name: torch.Generator().manual_seed(index + 10) for index, name in enumerate((
            "latent_generator", "penalty_generator", "noise_generator", "input_noise_generator",
            "prior_noise_generator", "eval_generator", "model_generator"))}
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0, **streams)


def _exact(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _exact(a, b)
    else:
        assert left == right


def _real(image):
    return (torch.linspace(.05, .95, 8*64).reshape(8, 1, 8, 8) if image
            else torch.linspace(-1., 1., 16).reshape(8, 2))


@pytest.mark.parametrize("image", [False, True])
def test_public_consumer_replaces_both_payoffs_once_using_existing_batches(monkeypatch, image):
    import particlegan.bregman_game as mechanism
    trainer, baseline = _public_trainer(image=image), _public_trainer(False, image=image)
    real, g_real = _real(image), _real(image).flip(0)
    generated, costs, d_losses, g_losses, target_calls = [], [], [], [], []
    handles = [trainer.G.register_forward_hook(lambda m, a, out: generated.append(out))]
    original_cost = mechanism.bregman_cost_matrix
    original_d, original_g = mechanism.critic_game_loss, mechanism.generator_game_loss
    def cost(fake, target, **kw):
        value = original_cost(fake, target, **kw)
        costs.append((fake, target, value))
        return value
    def d_loss(*args):
        value = original_d(*args)
        d_losses.append(value.detach().clone())
        return value
    def g_loss(*args, **kw):
        value = original_g(*args, **kw)
        g_losses.append(value.detach().clone())
        return value
    def forbidden(*args):
        raise AssertionError("active paired game must bypass both legacy score payoffs")
    monkeypatch.setattr(mechanism, "bregman_cost_matrix", cost)
    monkeypatch.setattr(mechanism, "critic_game_loss", d_loss)
    monkeypatch.setattr(mechanism, "generator_game_loss", g_loss)
    monkeypatch.setattr(trainer.loss, "d_loss", forbidden)
    monkeypatch.setattr(trainer.loss, "g_loss", forbidden)
    result = trainer.step(real, generator_real=lambda: target_calls.append(1) or g_real)
    for handle in handles:
        handle.remove()
    baseline.step(real, generator_real=g_real)
    assert target_calls == [1] and len(generated) == len(costs) == 2
    assert len(d_losses) == len(g_losses) == 1
    assert costs[0][0] is generated[0] and costs[1][0] is generated[1]
    assert torch.equal(costs[0][1], real) and torch.equal(costs[1][1], g_real)
    torch.testing.assert_close(result["loss_d"] - result["penalty"], d_losses[0])
    assert torch.equal(result["loss_gan"], g_losses[0])
    assert torch.equal(result["bregman_game"], g_losses[0])
    torch.testing.assert_close(result["loss_g"], result["loss_gan"] + trainer.recipe.prior_reg * result["prior_regularization"])
    _exact(trainer.state_dict()["streams"], baseline.state_dict()["streams"])
    assert not trainer.G._forward_hooks and not getattr(trainer.G, "output", nn.Identity())._forward_hooks
    assert any(not torch.equal(v, baseline.G.state_dict()[k]) for k, v in trainer.G.state_dict().items())


@pytest.mark.parametrize("image", [False, True])
def test_public_checkpoint_exact_continuation_and_capability_validation(image):
    trainer = _public_trainer(image=image)
    real = _real(image)
    trainer.step(real)
    checkpoint = deepcopy(trainer.state_dict())
    expected_result = trainer.step(real, generator_real=real.flip(0))
    expected_state = deepcopy(trainer.state_dict())
    resumed = _public_trainer(image=image)
    invalid = deepcopy(checkpoint)
    invalid["bregman_output_link"]["kind"] = "invalid"
    before = resumed.state_dict()
    with pytest.raises(ValueError, match="output capability"):
        resumed.load_state_dict(invalid)
    _exact(before, resumed.state_dict())
    resumed.load_state_dict(checkpoint)
    actual_result = resumed.step(real, generator_real=real.flip(0))
    _exact(expected_result, actual_result)
    _exact(expected_state, resumed.state_dict())


def test_actual_selected_image_capabilities_and_uniform_host_refusal():
    from benchmarks.transfer_suite.image_tasks import Generator, TASKS, RESERVED
    from particlegan.bregman_game import OutputLink
    with torch.random.fork_rng(devices=[]):
        for spec in [*TASKS, RESERVED]:
            host = Generator(spec)
            if spec["architecture"] == "uniform_generator":
                with pytest.raises(ValueError, match="declared"):
                    OutputLink(host)
                continue
            link = OutputLink(host)
            inputs = torch.zeros(2, spec["z_dim"])
            with link.capture() as observed:
                output = host(inputs)
            linked = link.attach(output, observed)
            assert torch.equal(linked, observed[0].sigmoid())
            assert not host.output._forward_hooks
            assert link.contract == {"kind": "sigmoid", "prelink_module": "output"}


def test_output_contract_rejects_unknown_and_false_sigmoid_without_lingering_hook():
    from particlegan.bregman_game import OutputLink
    with pytest.raises(ValueError, match="declared"):
        OutputLink(nn.ReLU())
    host = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
    link = OutputLink(host)
    with link.capture() as observed:
        output = host(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="actual generator output"):
        link.attach(output + .1, observed)
    assert not host[-1]._forward_hooks


def test_inactive_inherited_consumers_without_constructor_member_keep_original_path(monkeypatch):
    import particlegan.bregman_game as mechanism
    trainer = _public_trainer(False)
    del trainer._bregman_link  # Frozen component adapters historically bypass GANTrainer.__init__.
    def forbidden(*args, **kwargs):
        raise AssertionError("inactive inherited consumers must not enter the new game")
    monkeypatch.setattr(mechanism, "bregman_cost_matrix", forbidden)
    result = trainer.step(_real(False))
    checkpoint = trainer.state_dict()
    assert "bregman_game" not in result and "bregman_output_link" not in checkpoint
    trainer.load_state_dict(checkpoint)


@pytest.mark.parametrize("image", [False, True])
def test_public_collect_stats_emits_scalars_without_changing_trajectory(image):
    plain, observed = _public_trainer(image=image), _public_trainer(image=image)
    real = _real(image)
    plain_result = plain.step(real, collect_stats=False)
    observed_result = observed.step(real, collect_stats=True)
    assert "bregman_stats" not in plain_result
    assert all(type(value) is float for value in observed_result["bregman_stats"].values())
    assert ("prelink_min" in observed_result["bregman_stats"]) is image
    for key, value in plain_result.items():
        _exact(value, observed_result[key])
    _exact(plain.state_dict(), observed.state_dict())


def test_recipe_defaults_and_forge_ownership_admission():
    from particlegan import get_recipe
    from experiments.forge.api import CapabilityError, FormulationContext
    from experiments.forge.boundaries import RECIPE_FIELD_OWNERS
    recipe = get_recipe("bispf_optimized")
    assert "bispf_bregman_game" not in recipe.to_dict()
    assert RECIPE_FIELD_OWNERS["bispf_bregman_game"] == "technique"
    active = recipe.replace(bispf_bregman_game=True, prior_kind="mog", sigma_rel=0., standardize=False)
    assert active.to_dict()["bispf_bregman_game"] is True
    with pytest.raises(CapabilityError, match="does not consume"):
        FormulationContext(recipe_overrides=active.to_dict(), execution_path="public_components",
                           prior={"kind": "mog", "sigma": .025, "standardize": False})


@pytest.mark.parametrize("delta", [{"bispf_bregman_game": 1}, {"loss": "relativistic"},
    {"input_noise_std": .1}, {"output_noise_std": .1}, {"constraint_geometry_mode": "strict_progress"},
    {"output_energy_weight": 1.}, {"fidelity_transport_weight": 1.},
    {"bispf_thermal_application": "displacement"}])
def test_recipe_rejects_undeclared_combinations(delta):
    from particlegan import get_recipe
    with pytest.raises(ValueError, match="bispf_bregman_game"):
        get_recipe("bispf_optimized", **{"bispf_bregman_game": True, **delta})
