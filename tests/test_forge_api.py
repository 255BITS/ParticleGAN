"""CPU contract checks; these do not run qualification experiments."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import (GANTrainer, MoGParticlePrior, ParticlePrior, Recipe,
                        prior_capabilities, prior_mechanisms)
from experiments.forge.api import (CapabilityError, CapabilityRegistry, ExtensionSpec,
                                    FormulationContext)
from experiments.forge.rng import NamedStreams


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def make_context(**options):
    recipe = dict(num_particles=12, z_dim=2, batch_size=4, total_steps=6,
                  input_noise_std=.01, output_noise_std=.02, output_noise_warmup=0)
    recipe.update(options.pop("recipe_overrides", {}))
    return FormulationContext(recipe_overrides=recipe, **options)


def make_trainer(context=None, dropout=False):
    context = context or make_context()
    g = context.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2),
                          nn.Dropout(.2) if dropout else nn.Identity(), nn.Linear(6, 2)), component="generator")
    d = context.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 1)),
                          component="discriminator")
    return context, context.build_trainer(g, d)


def test_explicit_continuation_budget_preserves_original_schedule_and_prefix():
    def build(limit):
        context = make_context(recipe_overrides={"total_steps": 2})
        g = context.construct(lambda: nn.Linear(2, 2), component="generator")
        d = context.construct(lambda: nn.Linear(2, 1), component="discriminator")
        return context, context.build_trainer(g, d, max_steps=limit)
    short, a = build(2)
    long, b = build(4)
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    for _ in range(2):
        a.step(real)
        b.step(real)
    left, right = a.state_dict(), b.state_dict()
    assert right.pop("max_steps") == 4
    assert_equal(left, right)
    with pytest.raises(RuntimeError, match="budget"):
        a.step(real)
    b.step(real)
    resumed, c = build(4)
    resumed.load_state_dict(long.state_dict())
    b.step(real)
    c.step(real)
    assert_equal(b.state_dict(), c.state_dict())
    assert b.completed_steps == 4 and b.recipe.total_steps == 2


@pytest.mark.parametrize("output_noise", [False, True])
@pytest.mark.parametrize("ema", [False, True])
def test_clean_and_noisy_enumeration_preserve_mog_law_and_training_streams(output_noise, ema):
    from particlegan.training import output_noise_std
    context, trainer = make_trainer()
    model, prior = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
    sample_stream = context.streams.generator("eval", component="enumeration", purpose="combined-law")
    expected_stream = torch.Generator().set_state(sample_stream.get_state())
    before = trainer.state_dict()["streams"]
    modes = [(module, module.training) for root in (model, prior) for module in root.modules()]
    with torch.no_grad():
        latent, indices = prior.sample(7, generator=expected_stream, fixed_first_n=True, offset=3)
        expected = model(latent)
        assert torch.equal(indices, torch.arange(3, 10))
        assert not torch.equal(latent, prior.z[indices])
        if output_noise:
            expected = expected + output_noise_std(trainer.recipe, trainer.completed_steps) * torch.randn(
                expected.shape, generator=expected_stream)
    actual = trainer.sample(7, ema=ema, generator=sample_stream, fixed_first_n=True,
                            offset=3, output_noise=output_noise)
    with pytest.raises(ValueError, match="num_particles"):
        trainer.sample(17, fixed_first_n=True, offset=10, output_noise=output_noise)
    assert torch.equal(actual, expected)
    assert torch.equal(sample_stream.get_state(), expected_stream.get_state())
    assert_equal(before, trainer.state_dict()["streams"])
    assert all(module.training == flag for module, flag in modes)


def test_public_component_context_does_not_claim_scalar_trainer_for_autoencoder():
    context = make_context(execution_path="public_components", recipe_overrides={"encoder_mode": "ae"})
    assert context.receipt()["execution_path"] == "public_components"
    assert not context.capabilities()["public_trainer"]
    with pytest.raises(CapabilityError, match="does not own"):
        context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))


def test_initial_rng_manifest_declares_all_five_families_without_consuming():
    context = make_context()
    before = context.streams.audit()
    assert {binding["family"] for binding in context.streams.manifest()["bindings"].values()} == {
        "init", "data", "prior", "noise", "eval"}
    context.receipt()
    assert context.streams.audit() == before


def test_forge_rejects_an_aliased_named_training_stream(monkeypatch):
    context = make_context()
    original = context.streams.generator

    def aliased(family, *, component="default", purpose="default", **kwargs):
        if (family, component, purpose) == ("noise", "penalty", "training"):
            return original("prior", component="latent", purpose="indices", **kwargs)
        return original(family, component=component, purpose=purpose, **kwargs)

    monkeypatch.setattr(context.streams, "generator", aliased)
    with pytest.raises(CapabilityError, match="distinct named trainer streams"):
        make_trainer(context)
    assert context._trainer is None


def test_enumeration_preserves_mixture_noise_and_cloud_zero_noise_rng():
    cloud, trainer = make_trainer(make_context(
        recipe_overrides={"output_noise_std": 0},
        prior={"kind": "particle_cloud", "sigma": 0, "standardize": False,
               "exception_reason": "unit finite-cloud evaluator"}))
    stream = cloud.streams.generator("eval", component="enumeration", purpose="test")
    before = stream.get_state().clone()
    actual = trainer.sample(12, fixed_first_n=True, generator=stream)
    assert torch.equal(actual, trainer.G(trainer.prior.z))
    assert torch.equal(before, stream.get_state())
    mog, noisy = make_trainer(make_context(recipe_overrides={"output_noise_std": 0}))
    draw = noisy.sample(12, fixed_first_n=True)
    assert not torch.equal(draw, noisy.G(noisy.prior.z))


def test_public_penalty_maps_nested_condition_and_noise_wrappers_to_ema():
    class Wrapper(nn.Module):
        def __init__(self, child, scale):
            super().__init__()
            self.child, self.scale = child, scale
        def forward(self, value):
            return self.child(value) * self.scale
    recipe = Recipe(num_particles=12, z_dim=2)
    critic = nn.Sequential(nn.Linear(2, 4), nn.Softplus(), nn.Linear(4, 1))
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic))
    penalty = recipe.make_critic_penalty(optimizer)
    wrapped = Wrapper(Wrapper(critic, .5), 2.)
    view = penalty._ema_view(wrapped)(optimizer.ema_critic)
    assert view is not wrapped and view.child is not wrapped.child
    assert view.child.child is optimizer.ema_critic
    assert wrapped.child.child is critic
    assert (view.scale, view.child.scale) == (2., .5)
    real, fake = torch.ones(4, 2), torch.zeros(4, 2)
    value = penalty(wrapped, real, fake)
    value.backward()
    assert torch.isfinite(value)


def test_default_uses_public_mog_with_learned_locations_fixed_noise_and_a2():
    context, trainer = make_trainer()
    assert type(trainer) is GANTrainer
    assert type(trainer.prior) is MoGParticlePrior
    assert list(dict(trainer.prior.named_parameters())) == ["z"]
    assert prior_capabilities(trainer.prior)["mixture_weights"] == "uniform"
    assert trainer.latent_damping is not None
    sigma = trainer.prior.sigma.clone()
    before = trainer.prior.z.clone()
    result = trainer.step(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8)
    assert result["step"] == 1 and torch.isfinite(result["loss_g"])
    assert not torch.equal(before, trainer.prior.z)
    assert torch.equal(sigma, trainer.prior.sigma)
    assert context.receipt()["prior_mechanisms"]["a2"]["enabled"]


def test_standardized_mog_requires_explicitly_disabling_inapplicable_a2():
    with pytest.raises(CapabilityError, match="a2"):
        make_context(prior={"standardize": True})
    context, trainer = make_trainer(make_context(prior={"standardize": True},
                                                recipe_overrides={"latent_damping_max_rate": 0}))
    assert trainer.latent_damping is None
    assert not context.receipt()["prior_mechanisms"]["a2"]["requested"]
    recipe = Recipe(prior_kind="mog", standardize=True, num_particles=12, z_dim=2)
    prior = recipe.make_prior(sigma=.1)
    with pytest.raises(ValueError, match="required A2"):
        GANTrainer(recipe, nn.Linear(2, 2), nn.Linear(2, 1), prior=prior)
    # Older component users retain explicit introspection and optional strictness.
    opt, _ = recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), prior)
    assert not opt.prior_mechanisms["a2"]["enabled"]
    assert "standardized" in opt.prior_mechanisms["a2"]["reason"]
    with pytest.raises(ValueError, match="required A2"):
        recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), prior, require_latent_damping=True)


@pytest.mark.parametrize("prior", [
    {"kind": "particle_cloud"},
    {"kind": "particle_cloud", "sigma": 0, "standardize": False},
    {"kind": "particle_cloud", "sigma": .1, "standardize": False, "exception_reason": "host"},
    {"kind": "mog", "sigma": 0}, {"sigma": float("nan")}, {"learned_mass": True},
])
def test_prior_exceptions_and_extensions_cannot_be_implicit(prior):
    with pytest.raises(CapabilityError):
        make_context(prior=prior)


def test_cloud_exception_maps_public_prior_without_mog_noise_stream():
    context = make_context(prior={"kind": "particle_cloud", "sigma": 0,
                                  "standardize": False, "exception_reason": "frozen finite cloud host"})
    _, trainer = make_trainer(context)
    assert type(trainer.prior) is ParticlePrior
    assert "prior_noise_generator" not in trainer.state_dict()["streams"]
    assert context.receipt()["prior"]["exception_reason"]


def test_zero_sigma_equivalence_preserves_index_and_noise_streams():
    a = torch.Generator().manual_seed(4)
    b = torch.Generator().manual_seed(4)
    cloud = ParticlePrior(12, 2, generator=a)
    mog = MoGParticlePrior(12, 2, sigma=0, standardize=False, generator=b)
    noise = torch.Generator().manual_seed(123)
    noise_state = noise.get_state().clone()
    cloud.eval()
    mog.eval()
    x, i = cloud.sample(7, generator=a)
    y, j = mog.sample(7, generator=b, noise_generator=noise)
    assert torch.equal(x, y) and torch.equal(i, j)
    assert torch.equal(a.get_state(), b.get_state())
    assert torch.equal(noise_state, noise.get_state())
    mog.set_sigma(.1)
    y, j = mog.sample(7, generator=b, noise_generator=noise)
    assert not torch.equal(y, mog.means()[j])  # eval() keeps mixture noise
    assert not torch.equal(noise_state, noise.get_state())


def test_public_zero_sigma_mog_matches_cloud_updates_with_a2():
    torch.manual_seed(14)
    g, d = nn.Linear(2, 2), nn.Linear(2, 1)
    r = Recipe(num_particles=12, z_dim=2, total_steps=4, input_noise_std=0, output_noise_std=0)
    cloud = ParticlePrior(12, 2, generator=torch.Generator().manual_seed(10))
    mog = MoGParticlePrior(12, 2, sigma=0, standardize=False, generator=torch.Generator().manual_seed(10))
    a = GANTrainer(r, deepcopy(g), deepcopy(d), prior=cloud)
    b = GANTrainer(r.replace(prior_kind="mog", standardize=False), deepcopy(g), deepcopy(d), prior=mog)
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    for _ in range(3):
        assert_equal(a.step(real), b.step(real))
        assert torch.equal(a.prior.z, b.prior.z)
        assert_equal(a.opt_g.state_dict(), b.opt_g.state_dict())


def test_output_noisy_sampling_uses_public_mixture_law_and_preserves_training_streams():
    context, trainer = make_trainer()
    before = context.streams.audit()
    expected_stream = torch.Generator().manual_seed(71)
    latent, _ = trainer.prior.sample(11, generator=expected_stream)
    with torch.no_grad():
        expected = trainer.G(latent) + .02 * torch.randn((11, 2), generator=expected_stream)
    assert torch.equal(trainer.sample(11, generator=torch.Generator().manual_seed(71), output_noise=True), expected)
    trainer.sample(11)
    after = context.streams.audit()
    allowed = [key for key, binding in context.streams.manifest()["bindings"].items()
               if binding["family"] == "eval"]
    assert context.streams.compare(before, after, allowed=allowed)["unintended_rng_deviations"] == 0
    with pytest.raises(ValueError, match="separate"):
        trainer.sample(2, generator=trainer.prior_noise_generator)


def test_full_context_checkpoint_restores_exact_mog_dropout_continuation():
    context, trainer = make_trainer(dropout=True)
    data = context.streams.generator("data", component="task", purpose="batch")
    trainer.step(torch.randn((4, 2), generator=data))
    checkpoint = context.state_dict()
    preserved = deepcopy(checkpoint)
    expected_metrics = trainer.step(torch.randn((4, 2), generator=data))
    expected = context.state_dict()
    restored, other = make_trainer(dropout=True)
    restored.load_state_dict(checkpoint)
    data2 = restored.streams.generator("data", component="task", purpose="batch")
    assert_equal(expected_metrics, other.step(torch.randn((4, 2), generator=data2)))
    assert_equal(expected, restored.state_dict())
    assert_equal(checkpoint, preserved)


def test_checkpoint_invalid_extra_state_or_rng_conflict_is_atomic():
    context, trainer = make_trainer()
    before = context.state_dict()
    bad = deepcopy(before)
    bad["trainer"]["models"]["prior"]["_extra_state"]["standardize"] = True
    with pytest.raises(ValueError, match="incompatible state"):
        context.load_state_dict(bad)
    assert_equal(before, context.state_dict())
    bad = deepcopy(before)
    bad["trainer"]["streams"]["prior_noise_generator"] = torch.Generator().manual_seed(912).get_state()
    with pytest.raises(ValueError, match="RNG checkpoint mismatch"):
        context.load_state_dict(bad)
    assert_equal(before, context.state_dict())


def test_ema_prior_zero_noise_cache_follows_copied_sigma():
    _, trainer = make_trainer()
    trainer.prior.set_sigma(0)
    trainer.step(torch.ones(4, 2))
    assert not trainer.ema_prior._noise_enabled
    trainer.prior.set_sigma(.1)
    trainer.step(torch.ones(4, 2))
    assert trainer.ema_prior._noise_enabled
    assert torch.equal(trainer.prior.sigma, trainer.ema_prior.sigma)


def test_named_stream_extra_draws_do_not_shift_other_purposes_or_components():
    a, b = NamedStreams(19), NamedStreams(19)
    torch.randn((100,), generator=a.generator("noise", component="new_mechanism"))
    for family in ("init", "data", "prior", "noise", "eval"):
        x = torch.randn((4,), generator=a.generator(family, component="shared_component", purpose="normal"))
        y = torch.randn((4,), generator=b.generator(family, component="shared_component", purpose="normal"))
        assert torch.equal(x, y)
    assert a.seed_for("data", component="task") == b.seed_for("data", component="task")


def test_rng_restore_preserve_and_invalid_restore_do_not_mutate_live_state():
    streams = NamedStreams(42)
    g = streams.generator("prior")
    checkpoint = streams.state_dict()
    expected = torch.randn((8,), generator=g)
    streams.load_state_dict(checkpoint)
    assert torch.equal(expected, torch.randn((8,), generator=g))
    state = streams.state_dict()
    with pytest.raises(RuntimeError):
        with streams.preserve():
            torch.randn((3,), generator=g)
            streams.generator("noise", component="new")
            raise RuntimeError("evaluation failed")
    assert_equal(state, streams.state_dict())
    bad = deepcopy(state)
    bad["manifest"]["seed"] += 1
    with pytest.raises(ValueError):
        streams.load_state_dict(bad)
    assert_equal(state, streams.state_dict())


def test_global_rng_is_untouched_by_construction_initialization_and_model_dropout():
    original = torch.get_rng_state().clone()
    _, trainer = make_trainer(dropout=True)
    assert torch.equal(original, torch.get_rng_state())
    trainer.step(torch.ones(4, 2))
    assert torch.equal(original, torch.get_rng_state())


def test_parameter_named_initialization_survives_unrelated_architecture_changes():
    a, b = make_context(), make_context()
    first = a.construct(lambda: nn.ModuleDict({"shared": nn.Linear(2, 3)}), component="generator")
    second = b.construct(lambda: nn.ModuleDict({"extra": nn.Linear(9, 8), "shared": nn.Linear(2, 3)}), component="generator")
    a.initialize(first, component="generator")
    b.initialize(second, component="generator")
    assert_equal(first["shared"].state_dict(), second["shared"].state_dict())
    assert first["shared"].weight.shape != second["extra"].weight.shape


def test_registered_variable_reaches_public_recipe_and_checkpoint_once():
    registry = CapabilityRegistry().register_extension(ExtensionSpec(
        "critic_strength", "float", "recipe", "reg_coeff", "Public critic penalty coefficient", required=True))
    context, trainer = make_trainer(make_context(registry=registry, extensions={"critic_strength": .75},
                                                requires_capabilities=("critic_strength", "mog_prior")))
    assert trainer.recipe.reg_coeff == .75
    assert context.state_dict()["trainer"]["recipe"]["reg_coeff"] == .75
    assert context.receipt()["api_changes"][0]["argument"] == "reg_coeff"
    with pytest.raises(CapabilityError, match="missing required"):
        make_context(registry=registry)
    with pytest.raises(CapabilityError, match="requires float"):
        make_context(registry=registry, extensions={"critic_strength": "bad"})


def test_unsupported_fields_and_capabilities_block_before_model_construction():
    with pytest.raises(CapabilityError, match="unsupported extension"):
        make_context(extensions={"secret_knob": 1})
    with pytest.raises(CapabilityError, match="unavailable"):
        make_context(requires_capabilities=("learned_component_masses",))
    with pytest.raises(ValueError, match="supported"):
        CapabilityRegistry().register_extension(ExtensionSpec(
            "future", "float", "trainer", "missing_public_argument", "Unsupported public variable"))


def test_legacy_atom_checkpoint_shape_remains_schema_three():
    trainer = GANTrainer(Recipe(num_particles=8, z_dim=2, total_steps=3), nn.Linear(2, 2), nn.Linear(2, 1))
    state = trainer.state_dict()
    assert state["schema"] == 3
    assert set(state["streams"]) == {"latent_generator", "penalty_generator", "eval_generator", "noise_generator"}
    trainer.load_state_dict(state)
    assert_equal(state, trainer.state_dict())


def test_execution_extension_changes_only_allowance_and_rejects_reset():
    trainer = GANTrainer(Recipe(num_particles=8, z_dim=2, total_steps=3), nn.Linear(2, 2), nn.Linear(2, 1))
    before = trainer.state_dict()
    trainer.extend_execution(6)
    after = trainer.state_dict()
    assert after.pop("max_steps") == 6
    assert_equal(before, after)
    assert trainer.recipe.total_steps == 3
    for invalid in (6, 2, True, 6.5):
        with pytest.raises(ValueError, match="exceed"):
            trainer.extend_execution(invalid)
