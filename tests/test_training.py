from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe, learning_rate_scale, learning_rate_scales


def make_trainer(*, particles=12, steps=8, buffers=False, dropout=False, **overrides):
    recipe = get_recipe(num_particles=particles, z_dim=3,
                        batch_size=6, total_steps=steps, **overrides)
    generator = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8) if buffers else nn.Identity(),
                              nn.LeakyReLU(.2), nn.Dropout(.25) if dropout else nn.Identity(),
                              nn.Linear(8, 2))
    discriminator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2),
                                  nn.Dropout(.2) if dropout else nn.Identity(), nn.Linear(8, 1))
    return GANTrainer(recipe, generator, discriminator)


def assert_models_equal(left, right):
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, tensor in left["models"][name].items():
            assert torch.equal(tensor, right["models"][name][key]), (name, key)


def assert_checkpoint_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_checkpoint_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_checkpoint_equal(a, b)
    else:
        assert left == right


def test_winning_recipe_is_the_common_default():
    assert get_recipe() == Recipe()
    winner = get_recipe()
    assert winner.name == 'ka2'
    assert (winner.reg_kappa, winner.reg_coeff, winner.prior_reg) == (1., 1., 0.)
    assert (winner.reg_anchor_weight, winner.direct_particle_gain) == (1., True)
    assert (winner.lr, winner.betas, winner.prior_lr_mult, winner.d_lr_mult) == (.00425, (0., .999), 2., 1.)
    assert isinstance(make_trainer(), GANTrainer)


def test_step_updates_prior_restores_frozen_parameters_and_detaches_results():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    original_flags = [p.requires_grad for p in trainer.D.parameters()]
    observations = []
    handle = trainer.D.register_forward_pre_hook(
        lambda model, inputs: observations.append((model.training, [p.requires_grad for p in model.parameters()])))
    before = trainer.prior.z.detach().clone()
    calls = []
    result = trainer.step(torch.randn(6, 2), generator_real=lambda: calls.append(1) or torch.zeros(6, 2), collect_stats=True)
    handle.remove()
    assert calls == [1]
    assert not torch.equal(before, trainer.prior.z)
    assert len(trainer.opt_g.param_groups) == 2
    assert [p.requires_grad for p in trainer.D.parameters()] == original_flags
    assert all(not any(flags) for training, flags in observations if not training)
    assert any(training for training, _ in observations)
    assert result["step"] == 1 and "penalty_stats" in result
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert result[key].ndim == 0 and not result[key].requires_grad
    assert torch.equal(result["loss_g"], result["loss_gan"] + trainer.recipe.prior_reg * result["prior_regularization"])


def test_freeze_restored_when_generator_callback_raises():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    flags = [p.requires_grad for p in trainer.D.parameters()]
    def fail():
        raise RuntimeError("caller failure")
    with pytest.raises(RuntimeError, match="caller failure"):
        trainer.step(torch.randn(6, 2), generator_real=fail)
    assert [p.requires_grad for p in trainer.D.parameters()] == flags


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_old_formulation_checkpoints_are_rejected_before_mutating_state(schema):
    trainer = make_trainer()
    trainer.step(torch.randn(6, 2))
    checkpoint = trainer.state_dict()
    old = {**checkpoint, "schema": schema}
    before = trainer.state_dict()
    with pytest.raises(ValueError, match="K3P|older|schema"):
        trainer.load_state_dict(old)
    assert_checkpoint_equal(before, trainer.state_dict())


def test_disguised_k3p_optimizer_is_rejected_without_mutating_live_state():
    trainer = make_trainer()
    trainer.step(torch.randn(6, 2))
    before = trainer.state_dict()
    bad = deepcopy(before)
    record = bad["optimizers"][1]["regularizer"]["record"]
    historical = ("lr_max", "lr_last", "anchor_started", "calls", "observed_steps")
    bad["optimizers"][1]["regularizer"]["record"] = {key: record[key] for key in historical}
    with pytest.raises(ValueError, match="optimizer state|K3P"):
        trainer.load_state_dict(bad)
    assert_checkpoint_equal(before, trainer.state_dict())


@pytest.mark.parametrize("particles", [12, 1025])
def test_prior_regularizes_full_small_table_or_unique_large_sample(particles):
    trainer = make_trainer(particles=particles)
    seen = []
    handle = trainer.prior_regularizer.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    trainer.step(torch.randn(6, 2))
    handle.remove()
    assert seen == [12] if particles == 12 else 1 <= seen[0] <= 6


def test_ema_updates_parameters_and_copies_float_and_integer_buffers():
    trainer = make_trainer(buffers=True)
    old = [p.clone() for p in trainer.ema_G.parameters()]
    trainer.step(torch.randn(6, 2))
    for previous, averaged, live in zip(old, trainer.ema_G.parameters(), trainer.G.parameters()):
        expected = previous.mul_(trainer.recipe.ema_decay).add_(live, alpha=1 - trainer.recipe.ema_decay)
        assert torch.equal(averaged, expected)
        assert not averaged.requires_grad
    for averaged, live in zip(trainer.ema_G.buffers(), trainer.G.buffers()):
        assert torch.equal(averaged, live)
    assert trainer.ema_G[1].num_batches_tracked.item() == 1
    assert not trainer.ema_G.training and not trainer.ema_prior.training


def test_sample_preserves_modes_and_training_randomness():
    trainer = make_trainer(buffers=True, dropout=True)
    trainer.G[1].eval()
    flags = [module.training for module in trainer.G.modules()]
    checkpoint = trainer.state_dict()
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    cpu_rng = torch.get_rng_state().clone()
    assert trainer.sample(17).shape == (17, 2)
    assert trainer.sample(9, ema=True).shape == (9, 2)
    assert flags == [module.training for module in trainer.G.modules()]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(checkpoint["streams"]["latent_generator"], trainer.latent_generator.get_state())
    first = trainer.step(real)
    trained = trainer.state_dict()
    trainer.load_state_dict(checkpoint)
    second = trainer.step(real)
    assert_models_equal(trained, trainer.state_dict())
    for key in ("loss_d", "loss_g", "penalty"):
        assert torch.equal(first[key], second[key])
    with pytest.raises(ValueError, match="separate"):
        trainer.sample(3, generator=trainer.latent_generator)


def test_checkpoint_exact_continuation_with_dropout_and_independent_storage():
    torch.manual_seed(0)
    trainer = make_trainer(dropout=True)
    real = torch.randn(6, 2)
    trainer.step(real)
    trainer.step(real)
    checkpoint = trainer.state_dict()
    preserved = deepcopy(checkpoint)
    expected_stats = trainer.step(real)
    expected = trainer.state_dict()
    assert_models_equal(checkpoint, preserved)
    restored = make_trainer(dropout=True)
    restored.load_state_dict(checkpoint)
    actual_stats = restored.step(real)
    assert_models_equal(expected, restored.state_dict())
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        assert torch.equal(expected_stats[key], actual_stats[key])
    assert restored.completed_steps == 3
    assert_models_equal(checkpoint, preserved)


def test_schedule_budget_and_rejected_checkpoint_do_not_change_models():
    trainer = make_trainer(steps=3)
    initial = deepcopy(trainer.initial_lrs)
    for step in range(3):
        trainer.step(torch.randn(6, 2))
        network, prior = learning_rate_scales(step, trainer.recipe)
        assert [g["lr"] for g in trainer.opt_g.param_groups] == [initial[0][0] * network, initial[0][1] * prior]
        assert trainer.opt_d.param_groups[0]["lr"] == initial[1][0] * network
    with pytest.raises(RuntimeError, match="budget"):
        trainer.step(torch.randn(6, 2))
    checkpoint = trainer.state_dict()
    bad = deepcopy(checkpoint)
    bad["recipe"]["reg_coeff"] = 7.
    with pytest.raises(ValueError, match="recipe"):
        trainer.load_state_dict(bad)
    assert_models_equal(checkpoint, trainer.state_dict())
    bad = deepcopy(checkpoint)
    bad["models"]["prior"]["z"] = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="incompatible"):
        trainer.load_state_dict(bad)
    assert_models_equal(checkpoint, trainer.state_dict())


def test_validation_rejects_unsupported_recipe_and_mismatched_models():
    trainer = make_trainer()
    with pytest.raises(ValueError, match="unconditional"):
        GANTrainer(get_recipe(model='ddgan', num_classes=4, conditioning='ucd'), trainer.G, trainer.D)
    with pytest.raises(ValueError, match="dimensions"):
        GANTrainer(trainer.recipe.replace(z_dim=4), trainer.G, trainer.D, prior=trainer.prior)
    with pytest.raises(ValueError, match="dtype"):
        GANTrainer(trainer.recipe, deepcopy(trainer.G).double(), trainer.D)
    with pytest.raises(ValueError, match="nonempty"):
        trainer.step(torch.zeros(0, 2))


@pytest.mark.parametrize("invalid", [
    torch.zeros(6, 2, dtype=torch.float64), torch.zeros(6, 2, device="meta"), torch.zeros(5, 2), torch.zeros(6, 3),
])
def test_generator_real_tensor_validation_precedes_any_update(invalid):
    trainer = make_trainer()
    real = torch.zeros(6, 2)
    trainer.step(real)
    before = trainer.state_dict()
    modes = [m.training for root in (trainer.G, trainer.D) for m in root.modules()]
    with pytest.raises(ValueError, match="generator_real"):
        trainer.step(real, generator_real=invalid)
    assert_checkpoint_equal(before, trainer.state_dict())
    assert modes == [m.training for root in (trainer.G, trainer.D) for m in root.modules()]


@pytest.mark.parametrize("broken", ["second_optimizer", "latent_rng", "cpu_rng"])
def test_malformed_checkpoint_does_not_mutate_any_live_state(broken):
    trainer = make_trainer(dropout=True)
    real = torch.zeros(6, 2)
    trainer.step(real)
    bad = trainer.state_dict()
    trainer.step(real)
    before = trainer.state_dict()
    if broken == "second_optimizer":
        bad["optimizers"][1]["param_groups"] = []
    elif broken == "latent_rng":
        bad["streams"]["latent_generator"] = torch.ones(1, dtype=torch.uint8)
    else:
        bad["cpu_rng"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="optimizer state|RNG state"):
        trainer.load_state_dict(bad)
    assert_checkpoint_equal(before, trainer.state_dict())


def test_checkpoint_requires_matching_parameter_freezing():
    trainer = make_trainer()
    trainer.D[-1].bias.requires_grad_(False)
    checkpoint = trainer.state_dict()
    assert checkpoint["requires_grad"]["D"]["3.bias"] is False
    restored = make_trainer()
    before = restored.state_dict()
    with pytest.raises(ValueError, match="requires_grad"):
        restored.load_state_dict(checkpoint)
    assert_checkpoint_equal(before, restored.state_dict())
    restored.D[-1].bias.requires_grad_(False)
    restored.load_state_dict(checkpoint)
    assert_checkpoint_equal(checkpoint, restored.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_accepts_current_device_streams_and_restores_checkpoint():
    recipe = get_recipe(num_particles=12, z_dim=3, total_steps=3)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(.2), nn.Linear(8, 2)).to(device)
    discriminator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).to(device)
    latent = torch.Generator(device="cuda").manual_seed(2)
    penalty = torch.Generator(device="cuda").manual_seed(3)
    trainer = GANTrainer(recipe, generator, discriminator,
                                  latent_generator=latent, penalty_generator=penalty)
    real = torch.randn(6, 2, device=device)
    trainer.step(real)
    checkpoint = trainer.state_dict()
    trainer.sample(7, generator=torch.Generator(device="cuda").manual_seed(4))
    trainer.step(real)
    expected = trainer.state_dict()
    trainer.load_state_dict(checkpoint)
    trainer.step(real)
    assert_models_equal(expected, trainer.state_dict())
    assert trainer.sample(5, ema=True).device == device
    with pytest.raises(ValueError, match="model device"):
        trainer.sample(1, generator=torch.Generator(device="cpu"))
    before = trainer.state_dict()
    invalid = deepcopy(checkpoint)
    invalid["cuda_rng"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="RNG state"):
        trainer.load_state_dict(invalid)
    assert_checkpoint_equal(before, trainer.state_dict())
