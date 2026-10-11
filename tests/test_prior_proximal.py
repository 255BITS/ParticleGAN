"""Public API and analytic software contracts; no Tier 2 qualification."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.prior_proximal import PriorProximalDirectionBlendOptimizer
from experiments.forge.api import FormulationContext, task_formulation_context
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.boundaries import recipe_field_owner, validate_registry
from experiments.forge.priors import prior_policy_receipt, validate_prior_policy
from experiments.forge.techniques import recipe_field_active, validate_same_technique


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def serial_cpu_contract():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(threads)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def recipe(strength=1., geometry="direction_blend", **options):
    return get_recipe("bcap", prior_proximal_strength=strength,
                      constraint_geometry_mode=geometry, **options)


def backward(optimizer, table, gradient):
    loss = (table * gradient).sum()
    constraint_geometry_backward(loss, optimizer, [loss])


def context(strength=1., policy="learned"):
    # Forge v1's shorthand bcap retains historical Adam; bind the public
    # leader explicitly so this software comparison uses the intended recipe.
    return FormulationContext(recipe_preset="bcap", recipe_overrides=dict(
        leader_candidate(strength)["recipe_overrides"], prior_update=policy,
        num_particles=16, z_dim=2, batch_size=8, total_steps=12),
        prior=dict(kind="mog", sigma=.025, standardize=False),
        prior_contract="recipe_owned_v1", seed=0)


def leader_candidate(strength=1.):
    candidate = json.loads((ROOT / "configs/forge/ideas/bcap-default-baseline-direction-v1.json").read_text())
    candidate["recipe_overrides"]["prior_proximal_strength"] = strength
    return candidate


def trainer(ctx):
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 1)), component="discriminator")
    return ctx.build_trainer(g, d)


@pytest.mark.parametrize("geometry", ["none", "direction_blend"])
def test_owned_rows_follow_proximal_law_after_complete_base_step(geometry):
    initial = torch.tensor([[-1., 0.], [0., 1.], [1., -.5], [2., 1.]], dtype=torch.float64)
    table, control = (nn.Parameter(initial.clone()) for _ in range(2))
    r = recipe(.7, geometry=geometry, lr=.03)
    opt = r.make_generator_optimizer([dict(params=[table], role="prior", lr=.09)])
    ordinary = r.replace(prior_proximal_strength=0).make_generator_optimizer(
        [dict(params=[control], role="prior", lr=.09)])
    # Existing displacement makes the magnitude-sensitive restoring direction
    # observable; this analytic fixture is distinct from trained evidence.
    with torch.no_grad():
        table.add_(.4)
        control.copy_(table)
    rows = torch.tensor([0, 0, 2])
    gradient = torch.tensor([[1., 2.], [20., 30.], [-3., 4.], [40., 50.]], dtype=table.dtype)
    before = table.detach().clone()
    for optimizer, parameter in ((opt, table), (ordinary, control)):
        optimizer.set_sampled_rows(parameter, rows)
        backward(optimizer, parameter, gradient)
        optimizer.step()
    expected = torch.lerp(control.detach()[[0, 2]], initial[[0, 2]], .09 * .7 / (1 + .09 * .7))
    torch.testing.assert_close(table[[0, 2]], expected, rtol=0, atol=1e-15)
    assert torch.equal(table[[1, 3]], before[[1, 3]])
    assert opt.prior_proximal_stats["rows"] == 2
    assert opt.prior_proximal_stats["steps"] == 1
    assert opt.prior_proximal_stats["max_restoration"] > 0


def test_spring_retains_magnitude_instead_of_unit_normalizing_it():
    table = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    opt = recipe(2., geometry="none", lr=.1).make_generator_optimizer(
        [dict(params=[table], role="prior")])
    with torch.no_grad():
        table[:, 0] = torch.tensor([1., 3., 5.])
    opt.set_sampled_rows(table, torch.tensor([0, 1]))
    table.grad = torch.zeros_like(table)
    opt.step()
    torch.testing.assert_close(table[:, 0], table.new_tensor([1 / 1.2, 3 / 1.2, 5.]))
    assert torch.equal(table[2], table.new_tensor([5., 0.]))


@pytest.mark.parametrize("skip", ["no_gradient", "no_rows", "zero_rate"])
def test_no_force_without_update_ownership_or_nonzero_rate(skip):
    table = nn.Parameter(torch.zeros(3, 2))
    opt = recipe(1., geometry="none").make_generator_optimizer([dict(params=[table], role="prior")])
    with torch.no_grad():
        table.add_(1.)
    before = table.detach().clone()
    if skip != "no_gradient":
        table.grad = torch.zeros_like(table)
        opt.set_sampled_rows(table, torch.tensor([], dtype=torch.long) if skip == "no_rows" else torch.tensor([1]))
    if skip == "zero_rate":
        opt.param_groups[0]["lr"] = 0.
    opt.step()
    assert torch.equal(table, before)
    assert opt.prior_proximal_stats["steps"] == 0


def test_missing_ownership_rejects_before_network_mutation():
    table = nn.Parameter(torch.zeros(3, 2))
    network = nn.Parameter(torch.ones(2, 2))
    opt = recipe(1., geometry="none").make_generator_optimizer(
        [dict(params=[network]), dict(params=[table], role="prior")])
    network.grad, table.grad = torch.ones_like(network), torch.zeros_like(table)
    before = network.detach().clone()
    with pytest.raises(ValueError, match="sampled rows"):
        opt.step()
    assert torch.equal(network, before)


def test_zero_preserves_leader_type_packets_and_exact_consumed_streams():
    assert "prior_proximal_strength" not in recipe(0).to_dict()
    a, b = context(0), context(0)
    left, right = trainer(a), trainer(b)
    assert type(left.opt_g) is DirectionBlendOptimizer
    assert "prior_proximal" not in left.opt_g.state_dict()
    assert not hasattr(left.opt_g, "_prior_initial")
    batch = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    for _ in range(3):
        left.step(batch)
        right.step(batch)
    equal(a.state_dict(), b.state_dict())
    historical = a.state_dict()
    historical["trainer"]["recipe"].pop("prior_proximal_strength", None)
    # Old implicit-zero recipes remain resumable without a new optimizer packet.
    left.load_state_dict(historical["trainer"])


def test_initial_reference_is_independent_and_resume_restores_original_prefix():
    c = context()
    original = trainer(c)
    initial = original.prior.z.detach().clone()
    assert isinstance(original.opt_g, PriorProximalDirectionBlendOptimizer)
    reference = original.opt_g._prior_initial[original.prior.z]
    assert reference.data_ptr() != original.prior.z.data_ptr()
    batch = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    for _ in range(3):
        original.step(batch)
    saved = c.state_dict()
    original.step(batch)
    expected = c.state_dict()
    restored = context()
    resumed = trainer(restored)
    # Construction's table can differ: a checkpoint must restore its own
    # original anchor, not silently rebase on the loaded trained locations.
    resumed.opt_g._prior_initial[resumed.prior.z].add_(7.)
    restored.load_state_dict(saved)
    assert torch.equal(resumed.opt_g._prior_initial[resumed.prior.z], initial)
    resumed.step(batch)
    equal(expected, restored.state_dict())


@pytest.mark.parametrize("corruption", ["reference_shape", "reference_nan", "strength", "order", "statistics"])
def test_checkpoint_rejection_is_atomic(corruption):
    c = context()
    trained = trainer(c)
    before = trained.state_dict()
    bad = deepcopy(before)
    meta = bad["optimizers"][0]["prior_proximal"]
    index = next(i for i, value in enumerate(meta["initial_locations"]) if value is not None)
    if corruption == "reference_shape":
        meta["initial_locations"][index] = torch.zeros(1, 2)
    elif corruption == "reference_nan":
        meta["initial_locations"][index][0, 0] = float("nan")
    elif corruption == "strength":
        meta["strength"] = 2.
    elif corruption == "order":
        meta["order"] = "before_direction_blend"
    else:
        meta["stats"]["rows"] = -1
    with pytest.raises(ValueError, match="checkpoint optimizer state"):
        trained.load_state_dict(bad)
    equal(before, trained.state_dict())


def test_pending_ownership_and_initial_reference_roundtrip_standalone():
    table = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    opt = recipe(1., geometry="none").make_generator_optimizer([dict(params=[table], role="prior")])
    opt.set_sampled_rows(table, torch.tensor([1, 3]))
    saved = deepcopy(opt.state_dict())
    other = nn.Parameter(table.detach().clone() + 2)
    resumed = recipe(1., geometry="none").make_generator_optimizer([dict(params=[other], role="prior")])
    resumed.load_state_dict(saved)
    assert torch.equal(resumed.sampled_rows_for(other), torch.tensor([1, 3]))
    assert torch.equal(resumed._prior_initial[other], torch.zeros_like(other))
    other.grad = torch.zeros_like(other)
    resumed.step()
    assert torch.equal(other[[0, 2]], torch.full_like(other[[0, 2]], 2))
    assert bool((other[[1, 3]] < 2).all())


def test_frozen_and_direct_coordinates_receive_no_spring():
    c = context(policy="frozen")
    trained = trainer(c)
    initial = trained.prior.z.clone()
    trained.step(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8)
    assert torch.equal(trained.prior.z, initial)
    assert "prior_proximal" not in trained.opt_g.state_dict()
    coordinates = nn.Parameter(torch.ones(3, 2))
    direct = recipe().make_generator_optimizer([dict(params=[coordinates], role="prior")], direct_particles=[coordinates])
    assert "prior_proximal" not in direct.state_dict()


@pytest.mark.parametrize("settings", [dict(prior_proximal_strength=-1), dict(prior_proximal_strength=True),
    dict(prior_proximal_strength=float("nan")), dict(optimizer_family="adam"),
    dict(optimizer_momentum=.5), dict(constraint_geometry_mode="nonascent"),
    dict(constraint_geometry_mode="strict_progress"), dict(row_evidence_gate=True),
    dict(particle_birth_death=True)])
def test_unsupported_settings_fail_at_recipe_boundary(settings):
    with pytest.raises(ValueError, match="prior_proximal_strength"):
        recipe().replace(**settings)


def test_forge_search_ownership_activation_and_receipt_match():
    validate_registry()
    assert recipe_field_owner("prior_proximal_strength") == "hyperparameter"
    validate_same_technique(recipe(.1), recipe(10))
    with pytest.raises(ValueError, match="prior_proximal"):
        validate_same_technique(recipe(.1), recipe(0))
    target = json.loads((ROOT / "configs/forge/tasks/vector_two_broad.json").read_text())
    direct = json.loads((ROOT / "configs/forge/tasks/two_pole.json").read_text())
    assert recipe_field_active("prior_proximal_strength", recipe(), task=target)
    assert not recipe_field_active("prior_proximal_strength", recipe(), task=direct)
    assert not recipe_field_active("prior_proximal_strength", recipe(prior_update="frozen"), task=target)
    ctx = task_formulation_context(leader_candidate(), target)
    prior = ctx.build_prior()
    proof = prior_policy_receipt(ctx.recipe, prior)
    assert proof["proximal_strength"] == 1
    assert validate_prior_policy(target, dict(prior_policy=proof), asdict(ctx.recipe)) is None
    proof["proximal_strength"] = 0
    assert validate_prior_policy(target, dict(prior_policy=proof), asdict(ctx.recipe))["status"] == "INVALID"


@pytest.mark.parametrize("host", ["ae_gan_hold", "trajectory", "cover_leftover"])
def test_behavioral_hosts_bind_the_same_global_spring_after_initializer(host):
    target = json.loads((ROOT / "configs/forge/tasks" / (host + ".json")).read_text())
    components = BehaviorComponents(dict(candidate=leader_candidate(), protocol=dict(seed=0)), target)
    components.recipe = components.recipe.replace(num_particles=12, z_dim=2 if host == "ae_gan_hold" else 4)
    options = {"sigma": components.context.prior_config["sigma"]} if components.recipe.prior_kind == "mog" else {}
    prior = components.recipe.make_prior(**options)
    # These standalone binding checks retain each task's actual prior dimensions.
    generator = nn.Linear(prior.z.shape[1], 2)
    critic = nn.Linear(2, 1)
    opt, _, _, _ = components.bind(generator=generator, critic=critic, priors=[prior], opt_g=None, opt_d=None)
    assert len(opt.optimizers) == 1
    actual = opt.optimizers[0]
    assert isinstance(actual, PriorProximalDirectionBlendOptimizer)
    assert torch.equal(actual._prior_initial[prior.z], prior.z)
    assert actual._prior_initial[prior.z].data_ptr() != prior.z.data_ptr()
