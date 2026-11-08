"""Provenance regressions for ManipulatedPrior/PriorManipulator (review 2026-10-25 #1-3, #15-17)."""
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from particlegan import GANTrainer, MoGParticlePrior, ParticlePrior, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.priors import (
    ManipulatedPrior,
    PriorManipulator,
    SimplexRoutingManipulator,
    StratifiedSamplingManipulator,
    unwrap_prior,
)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _strip_base(state):
    return {key.removeprefix("base."): value for key, value in state.items()}


class ClampOutputs(PriorManipulator):
    def __init__(self):
        super().__init__()
        self.calls = []

    def postprocess_generated(self, generated):
        self.calls.append(generated.requires_grad)
        return generated.clamp(-0.25, 0.25)


class ZeroPenalty(PriorManipulator):
    def regularize(self, raw_rows, **kwargs):
        return raw_rows.sum() * 0.0


def _mog_trainer(manipulators=None, *, prior_noise_seed=77, **recipe_overrides):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", z_dim=2, num_particles=8, batch_size=4, total_steps=6,
                            prior_kind="mog", sigma_rel=.25, standardize=False,
                            **recipe_overrides)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for seed, component in enumerate((generator, critic, prior)):
            deterministic_orthogonal_(component, seed=seed)
        if manipulators is not None:
            prior = ManipulatedPrior(prior, manipulators)
        streams = {name: torch.Generator().manual_seed(seed) for name, seed in (
            ("latent_generator", 11), ("penalty_generator", 12), ("noise_generator", 13),
            ("eval_generator", 14))}
        noise = torch.Generator().manual_seed(prior_noise_seed)
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                             prior_noise_generator=noise, **streams)
    return trainer, noise


def _batches(count=3):
    return [torch.randn(4, 2, generator=torch.Generator().manual_seed(500 + i)) for i in range(count)]


# -----------------------------------------------------------------------------
# #1 unwrap_prior and the wrapped-MoG noise stream
# -----------------------------------------------------------------------------

def test_wrapped_mog_identity_manipulator_reproduces_plain_trajectory_and_checkpoints_noise_stream():
    plain, plain_noise = _mog_trainer()
    wrapped, wrapped_noise = _mog_trainer([PriorManipulator()])
    initial = wrapped_noise.get_state().clone()
    assert wrapped.prior_noise_generator is wrapped_noise
    for batch in _batches():
        _equal(plain.step(batch), wrapped.step(batch))
    assert not torch.equal(wrapped_noise.get_state(), initial), "caller's MoG noise stream was not consumed"
    assert torch.equal(wrapped_noise.get_state(), plain_noise.get_state())
    plain_state, wrapped_state = plain.state_dict(), wrapped.state_dict()
    assert "prior_noise_generator" in wrapped_state["streams"]
    _equal(plain_state["streams"], wrapped_state["streams"])
    for name in ("G", "D", "ema_G"):
        _equal(plain_state["models"][name], wrapped_state["models"][name])
    for name in ("prior", "ema_prior"):
        tensors = {k: v for k, v in _strip_base(wrapped_state["models"][name]).items()
                   if not k.startswith("manipulators.")}
        _equal(plain_state["models"][name], tensors)


def test_unwrap_prior_and_nested_wrappers_flatten():
    base = ParticlePrior(8, 2)
    a, b = PriorManipulator(), PriorManipulator()
    nested = ManipulatedPrior(ManipulatedPrior(base, [a]), [b])
    assert nested.base is base and list(nested.manipulators) == [a, b]
    assert unwrap_prior(nested) is base and unwrap_prior(base) is base and unwrap_prior(None) is None


def test_explicit_forwarding_keeps_hasattr_truthful():
    plain = ManipulatedPrior(ParticlePrior(8, 2), [PriorManipulator()])
    for name in ("sigma", "set_sigma", "means", "d0", "standardize", "sigma_rel", "_noise_enabled"):
        assert not hasattr(plain, name), name
    mog = MoGParticlePrior(8, 2, sigma=.1, standardize=False)
    wrapped = ManipulatedPrior(mog, [PriorManipulator()])
    assert wrapped.sigma is mog.sigma and wrapped.d0 is mog.d0 and wrapped.standardize is False
    wrapped.set_sigma(.2)
    assert float(mog.sigma) == pytest.approx(.2)
    assert not hasattr(wrapped, "_noise_enabled")


# -----------------------------------------------------------------------------
# #2 manipulator configuration provenance
# -----------------------------------------------------------------------------

def test_manipulator_spec_and_checkpoint_configuration_is_validated():
    base = ParticlePrior(8, 4)
    prior = ManipulatedPrior(base, [StratifiedSamplingManipulator(8, jitter_std=.1, k_nn=2),
                                    SimplexRoutingManipulator(3, temperature=.5)])
    spec = prior.manipulator_spec()
    assert spec[0] == {"class": "particlegan.priors.StratifiedSamplingManipulator",
                       "config": {"penalty_weight": 1.0, "num_particles": 8, "jitter_std": .1,
                                  "k_nn": 2, "initial_cursor": 0}}
    assert spec[1]["config"]["temperature"] == .5 and spec[1]["config"]["z_dim"] == 4
    state = prior.state_dict()
    assert state["manipulators.0._extra_state"] == spec[0]
    same = ManipulatedPrior(ParticlePrior(8, 4), [StratifiedSamplingManipulator(8, jitter_std=.1, k_nn=2),
                                                 SimplexRoutingManipulator(3, temperature=.5)])
    same.load_state_dict(state)
    other = ManipulatedPrior(ParticlePrior(8, 4), [StratifiedSamplingManipulator(8, jitter_std=.2, k_nn=2),
                                                  SimplexRoutingManipulator(3, temperature=.5)])
    with pytest.raises(ValueError, match="configuration mismatch"):
        other.load_state_dict(state)


def test_trainer_rejects_checkpoint_with_different_manipulator_settings():
    trainer, _ = _mog_trainer([StratifiedSamplingManipulator(8, jitter_std=.1)])
    trainer.step(_batches(1)[0])
    state = trainer.state_dict()
    other, _ = _mog_trainer([StratifiedSamplingManipulator(8, jitter_std=.3)])
    with pytest.raises(ValueError, match="checkpoint model prior has incompatible state"):
        other.load_state_dict(state)
    same, _ = _mog_trainer([StratifiedSamplingManipulator(8, jitter_std=.1)])
    same.load_state_dict(state)


# -----------------------------------------------------------------------------
# #3 SimplexRoutingManipulator construction
# -----------------------------------------------------------------------------

def test_simplex_construction_reads_no_global_rng_and_is_deterministic():
    base = ParticlePrior(8, 5)  # the table constructor's own draw is not under test
    before = torch.get_rng_state()
    first = SimplexRoutingManipulator(3, z_dim=5)
    lazy = ManipulatedPrior(base, [SimplexRoutingManipulator(3)]).manipulators[0]
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(first.projection, SimplexRoutingManipulator(3, z_dim=5).projection)
    assert torch.equal(first.projection, lazy.projection)
    gram = first.projection.T @ first.projection
    assert torch.allclose(gram, torch.eye(3), atol=1e-6)
    assert not torch.equal(first.projection, SimplexRoutingManipulator(3, z_dim=5, init_seed=1).projection)
    stream = torch.Generator().manual_seed(3)
    start = stream.get_state().clone()
    drawn = SimplexRoutingManipulator(3, z_dim=5, generator=stream)
    assert not torch.equal(stream.get_state(), start)
    assert drawn.config()["projection_init"] == "generator"
    assert torch.equal(torch.get_rng_state(), before)


def test_simplex_fixed_projection_without_z_dim_binds_as_buffer():
    m = SimplexRoutingManipulator(3, learnable_projection=False)
    assert m.projection is None
    prior = ManipulatedPrior(ParticlePrior(8, 4), [m])
    assert "manipulators.0.projection" in dict(prior.named_buffers())
    assert not list(m.parameters())
    codes, _ = prior.sample(4, generator=torch.Generator().manual_seed(0))
    assert codes.shape == (4, 3)


def test_simplex_forward_does_not_mutate_projection_or_attach_side_channel():
    m = SimplexRoutingManipulator(3, z_dim=4)
    stored = m.projection
    codes = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    out = m.transform(codes, base_prior=ParticlePrior(8, 4))
    assert out.dtype == torch.float64 and not hasattr(out, "weights")
    assert m.projection is stored and m.projection.dtype == torch.float32
    out.sum().backward()
    assert m.projection.grad is not None


def test_deterministic_orthogonal_initializes_wrapped_simplex_projection():
    prior = ManipulatedPrior(ParticlePrior(8, 4), [SimplexRoutingManipulator(3)])
    deterministic_orthogonal_(prior, seed=0)
    projection = prior.manipulators[0].projection
    assert torch.allclose(projection.T @ projection, torch.eye(3), atol=1e-5)


# -----------------------------------------------------------------------------
# #15 postprocess_generated on every critic/serving path
# -----------------------------------------------------------------------------

def test_postprocess_applies_to_critic_step_sampling_and_serving():
    clamp = ClampOutputs()
    trainer, _ = _mog_trainer([clamp])
    trainer.step(_batches(1)[0])
    assert clamp.calls == [False, True]  # critic-step fakes (no_grad), then generator-step fakes
    assert trainer.sample(16).abs().max() <= .25
    assert trainer.sample(16, ema=True).abs().max() <= .25
    assert trainer.served_model().sample(16, generator=torch.Generator().manual_seed(1)).abs().max() <= .25


# -----------------------------------------------------------------------------
# #16 strict-progress replay requires additive transforms
# -----------------------------------------------------------------------------

def test_strict_progress_rejects_non_additive_transform_at_construction():
    options = dict(constraint_geometry_mode="strict_progress", input_noise_std=0., output_noise_std=0.)
    with pytest.raises(ValueError, match="additive"):
        _mog_trainer([SimplexRoutingManipulator(2)], **options)
    trainer, _ = _mog_trainer([StratifiedSamplingManipulator(8)], **options)
    assert trainer.prior.additive_transforms


# -----------------------------------------------------------------------------
# #17 StratifiedSamplingManipulator
# -----------------------------------------------------------------------------

def test_stratified_indices_do_not_alias_the_permutation():
    m = StratifiedSamplingManipulator(8)
    base = ParticlePrior(8, 2)
    first = m.sample_indices(base, 3, generator=torch.Generator().manual_seed(0))
    kept = first.clone()
    m.sample_indices(base, 3)
    m.sample_indices(base, 3, generator=torch.Generator().manual_seed(1))  # wraps and reshuffles
    assert torch.equal(first, kept)


@pytest.mark.parametrize("cursor", [-1, 9, 1.0, True])
def test_stratified_rejects_invalid_initial_cursor(cursor):
    with pytest.raises(ValueError, match="initial_cursor"):
        StratifiedSamplingManipulator(8, initial_cursor=cursor)


def test_stratified_rejects_num_particles_mismatch_with_base():
    with pytest.raises(ValueError, match="num_particles"):
        ManipulatedPrior(ParticlePrior(16, 2), [StratifiedSamplingManipulator(8)])


# -----------------------------------------------------------------------------
# Penalty weight and per-step reporting
# -----------------------------------------------------------------------------

def test_penalty_weight_is_explicit_configuration():
    class Unit(PriorManipulator):
        def regularize(self, raw_rows, **kwargs):
            return raw_rows.new_ones(())

    prior = ManipulatedPrior(ParticlePrior(8, 2), [Unit(penalty_weight=.5), Unit(penalty_weight=0.)])
    assert prior.regularize(prior.z) == .5
    assert [s["config"]["penalty_weight"] for s in prior.manipulator_spec()] == [.5, 0.]
    with pytest.raises(ValueError, match="penalty_weight"):
        Unit(penalty_weight=-1.)
    assert not ManipulatedPrior(ParticlePrior(8, 2), [PriorManipulator()]).has_regularizer


def test_prior_manipulation_metric_is_present_every_step_and_absent_without_penalty():
    trainer, _ = _mog_trainer([ZeroPenalty()])
    for batch in _batches(2):
        result = trainer.step(batch)
        assert "prior_manipulation" in result and result["prior_manipulation"] == 0
    plain, _ = _mog_trainer([PriorManipulator()])
    assert "prior_manipulation" not in plain.step(_batches(1)[0])


# -----------------------------------------------------------------------------
# encode
# -----------------------------------------------------------------------------

def test_encode_rejects_code_transforming_manipulators():
    recipe = get_recipe(prior_kind="mog", sigma_rel=.025, encoder_mode="ae", z_dim=2, num_particles=8)
    base = recipe.make_prior(sigma=.1)
    query = torch.zeros(2, 2)
    plain = recipe.encode(query, ManipulatedPrior(base, [PriorManipulator()]), offset=query)
    _equal(plain.codes, recipe.encode(query, base, offset=query).codes)
    with pytest.raises(ValueError, match="transforms"):
        recipe.encode(query, ManipulatedPrior(base, [StratifiedSamplingManipulator(8)]), offset=query)


# -----------------------------------------------------------------------------
# Review follow-ups (PR #481): legacy checkpoints, coverage, receipts, RNG
# -----------------------------------------------------------------------------

def test_legacy_manipulator_checkpoint_without_extra_state_loads_strictly():
    source = ManipulatedPrior(ParticlePrior(8, 4), [SimplexRoutingManipulator(3, init_seed=5)])
    legacy = {k: v for k, v in source.state_dict().items() if not k.endswith("_extra_state")}
    target = ManipulatedPrior(ParticlePrior(8, 4), [SimplexRoutingManipulator(3)])
    result = target.load_state_dict(legacy)
    assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(target.manipulators[0].projection, source.manipulators[0].projection)


def test_birth_death_row_generator_applies_postprocess():
    clamp = ClampOutputs()
    recipe = get_recipe("bcap", z_dim=2, num_particles=8, batch_size=4, total_steps=6,
                        particle_birth_death=True)
    generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
    critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    prior = recipe.make_prior()
    for seed, component in enumerate((generator, critic, prior)):
        deterministic_orthogonal_(component, seed=seed)
    trainer = GANTrainer(recipe, generator, critic, prior=ManipulatedPrior(prior, [clamp]), seed=0)
    rows = trainer.policy.birth_death.rows
    out = rows.generate(torch.full((3, 2), 50.0))
    assert out.abs().max() <= .25


def test_simplex_slice_records_bound_width_and_rejects_mismatch():
    prior = ManipulatedPrior(ParticlePrior(8, 5), [SimplexRoutingManipulator(3, routing_mode="slice")])
    assert prior.manipulator_spec()[0]["config"]["z_dim"] == 5
    with pytest.raises(ValueError, match="does not match"):
        ManipulatedPrior(ParticlePrior(8, 5), [SimplexRoutingManipulator(3, routing_mode="slice", z_dim=4)])


def test_simplex_generator_init_records_stream_identity():
    stream = torch.Generator().manual_seed(3)
    m = SimplexRoutingManipulator(3, z_dim=5, generator=stream)
    record = m.config()["projection_init_generator"]
    assert record["initial_seed"] == 3 and len(record["state_sha256"]) == 64
    again = SimplexRoutingManipulator(3, z_dim=5, generator=torch.Generator().manual_seed(3))
    assert again.config() == m.config()
    assert SimplexRoutingManipulator(3, z_dim=5).config()["projection_init_generator"] is None


class NoisyShift(PriorManipulator):
    def transform(self, codes, idx=None, *, base_prior, generator=None, training=True):
        return codes + torch.randn(codes.shape, generator=generator)


def test_fixed_first_n_transform_uses_noise_generator():
    prior = ManipulatedPrior(ParticlePrior(8, 2), [NoisyShift()])
    draws = []
    for latent_seed in (0, 1):
        codes, _ = prior.sample(4, generator=torch.Generator().manual_seed(latent_seed), fixed_first_n=True,
                                noise_generator=torch.Generator().manual_seed(9))
        draws.append(codes)
    assert torch.equal(draws[0], draws[1])
