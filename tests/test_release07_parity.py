"""Compare current shared updates with the pinned v0.7.0 public package.

The immutable fixture is a test reference, never an experiment training path.
Tiny deterministic updates verify the full recipe's arithmetic, not quality.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe


RELEASE_COMMIT = "180d18f400335fb295611d624b48a4e072ae3bae"
SOURCE_HASHES = {
    "gan_loss": "9232adaafeaa8767cf102b05a0f303cb40dffed843ced5b0702041ddd8d057da",
    "grad_regularizers": "da03f7653b3d8718fbfa086b0c6eb20bf06943b1f6a8e360b6e14ebd66dabd3f",
    "particle_prior": "17e39404cefca5963c82d9981f8582ea6c650c9aae66b51ce11ecf05896bb9a9",
    "vicreg_loss": "ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36",
    "recipes": "56498c6b22895e30c68c28bb897ba7bc9d26707901a33991ec6b3d9cb5a4b060",
    "training": "d2f418ec1ab43ab24d57071382c2f1fd53e17eb5064fb6bbd391e84543eeb405",
}
NEUTRAL_K3P_FIELDS = {
    "network_lr_floor": None,
    "network_lr_horizon_cap": None,
    "reg_anchor_weight": 0.0,
    "d_guard_ratio": 0.0,
    "latent_damping_max_rate": 0.0,
    "direct_particle_gain": False,
    "input_noise_std": 0.0,
    "output_noise_std": 0.0,
}


@lru_cache(maxsize=1)
def _release_modules():
    fixture_path = Path(__file__).parent / "fixtures/release07-public-api.json"
    fixture = json.loads(fixture_path.read_text())
    assert fixture["schema_version"] == 1
    assert fixture["tag"] == "v0.7.0" and fixture["commit"] == RELEASE_COMMIT
    assert set(fixture["files"]) == {f"particlegan/{name}.py" for name in SOURCE_HASHES}
    package_name = "_pinned_release07_public_api"
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    modules = {}
    for name in ("gan_loss", "grad_regularizers", "particle_prior", "vicreg_loss", "recipes", "training"):
        path = f"particlegan/{name}.py"
        source = fixture["files"][path]["source"]
        actual = hashlib.sha256(source.encode()).hexdigest()
        assert actual == fixture["files"][path]["sha256"] == SOURCE_HASHES[name]
        module = ModuleType(f"{package_name}.{name}")
        module.__file__ = f"{RELEASE_COMMIT}:{path}"
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        modules[name] = module
    return modules


def _current_release_recipe(released):
    fields = released.to_dict()
    # These removed selectors have exactly the current fixed public semantics.
    assert fields.pop("loss_type") == "logistic"
    assert fields.pop("gan_mode") == "rp"
    assert fields.pop("reg_method") == "autograd"
    return Recipe(**fields, **NEUTRAL_K3P_FIELDS)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def test_release_default_recipe_loss_prior_and_regularizer_are_faithful():
    release = _release_modules()
    old = release["recipes"].Recipe()
    current = _current_release_recipe(old)
    assert (current.name, current.z_dim, current.num_particles, current.batch_size) == (
        "gan_v3", 4, 20_000, 256)
    assert (current.reg_arm, current.reg_coeff, current.reg_kappa, current.reg_every) == (
        "b_cap", 6.0, 1.25, 1)
    assert (current.lr, current.d_lr_mult, current.prior_lr_mult, current.betas) == (
        .00425, 1., 2., (0., .99))
    assert (current.prior_reg, current.ema_decay, current.total_steps) == (.05, .995, 7000)
    for key, value in old.to_dict().items():
        if key not in {"loss_type", "gan_mode", "reg_method"}:
            assert getattr(current, key) == value
    for key, value in NEUTRAL_K3P_FIELDS.items():
        assert getattr(current, key) == value

    real = torch.tensor([-2., .25, 3.], dtype=torch.float64)
    fake = torch.tensor([.75, -.4, 1.], dtype=torch.float64)
    _assert_equal(old.make_loss().d_loss(real, fake), current.make_loss().d_loss(real, fake))
    _assert_equal(old.make_loss().g_loss(fake, real), current.make_loss().g_loss(fake, real))
    old_rng = torch.Generator().manual_seed(183)
    current_rng = torch.Generator().manual_seed(183)
    old_prior = old.replace(num_particles=13).make_prior(generator=old_rng)
    current_prior = current.replace(num_particles=13).make_prior(generator=current_rng)
    _assert_equal(old_prior.z, current_prior.z)
    _assert_equal(old_rng.get_state(), current_rng.get_state())
    # standardize=True is inert for the release's plain finite particle cloud.
    indices = torch.tensor([0, 4, 7])
    _assert_equal(old_prior(indices), old_prior.z[indices])
    _assert_equal(current_prior(indices), current_prior.z[indices])
    _assert_equal(old.make_prior_regularizer()(old_prior.z),
                  current.make_prior_regularizer()(current_prior.z))


@pytest.mark.parametrize("particles", [9, 1031])
@pytest.mark.parametrize("completed_steps", [0, 4199, 6998])
def test_two_public_updates_match_release_with_active_penalty_and_spread(particles, completed_steps):
    """Cover both spread subsets and full-budget LR transition/final boundaries."""
    release = _release_modules()
    old_recipe = release["recipes"].Recipe(num_particles=particles, batch_size=8)
    current_recipe = _current_release_recipe(old_recipe)
    generator = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2)).double()
    discriminator = nn.Sequential(nn.Linear(2, 5), nn.LeakyReLU(.2), nn.Linear(5, 1)).double()
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.copy_(torch.linspace(-.35, .55, parameter.numel()).reshape_as(parameter))
        for parameter in discriminator.parameters():
            parameter.fill_(1.2)
    old_prior = old_recipe.make_prior().double()
    current_prior = current_recipe.make_prior().double()
    with torch.no_grad():
        old_prior.z.copy_(torch.linspace(-.4, .5, particles * 4).reshape(particles, 4))
        current_prior.z.copy_(old_prior.z)
    def streams():
        return {"latent_generator": torch.Generator().manual_seed(912),
                "penalty_generator": torch.Generator().manual_seed(913)}
    old = release["training"].GANTrainer(old_recipe, deepcopy(generator), deepcopy(discriminator),
                                          prior=old_prior, seed=0, **streams())
    current = GANTrainer(current_recipe, deepcopy(generator), deepcopy(discriminator),
                         prior=current_prior, seed=0, **streams())
    assert current.opt_d.guard is None and current.opt_g.latent_damping is None
    assert current.opt_g.direct_response is None
    old.completed_steps = current.completed_steps = completed_steps
    real = torch.linspace(-.2, .3, 16, dtype=torch.float64).reshape(8, 2)
    for _ in range(2):
        old_result = old.step(real, generator_real=lambda: real.flip(0), collect_stats=True)
        result = current.step(real, generator_real=lambda: real.flip(0), collect_stats=True)
        for key in old_result.keys() - {"penalty_stats"}:
            _assert_equal(old_result[key], result[key])
        assert result["penalty"] > 0 and result["prior_regularization"] > 0
        for name in ("G", "D", "prior", "ema_G", "ema_prior"):
            _assert_equal(getattr(old, name).state_dict(), getattr(current, name).state_dict())
        for old_opt, current_opt in zip((old.opt_g, old.opt_d), (current.opt_g, current.opt_d)):
            # Public K3P subclasses retain extra metadata; Adam arithmetic matches.
            current_state = current_opt.state_dict()
            current_state.pop("regularizer")
            _assert_equal(old_opt.state_dict(), current_state)
        for name in ("latent_generator", "penalty_generator", "eval_generator"):
            _assert_equal(getattr(old, name).get_state(), getattr(current, name).get_state())
        assert not current.opt_d.record.anchor_started
    for ema in (False, True):
        old_rng = torch.Generator().manual_seed(810)
        current_rng = torch.Generator().manual_seed(810)
        latent_before = current.latent_generator.get_state().clone()
        _assert_equal(old.sample(11, ema=ema, generator=old_rng),
                      current.sample(11, ema=ema, generator=current_rng))
        _assert_equal(old_rng.get_state(), current_rng.get_state())
        _assert_equal(latent_before, current.latent_generator.get_state())
