"""Explicit noise-free, bounded software cohort; no scientific qualification.

The manual BiGAN owner has no training-noise or UpdatePolicy implementation.
These tests declare a separate intact public recipe instead of changing presets
inside the host. Architectures, target law and prior width remain task-owned.
"""
from copy import deepcopy

from benchmarks.toy_audit.api_ring16 import (
    ALICESpiralBiGANFixture,
    Ring16BiGANFixture,
    _alice_spiral_case,
    _bigan_case,
)
from particlegan import get_recipe

COHORT = "bigan-noise-free-software-v1"


def build_fixture(case=None, *, max_steps=10, fixture_cls=None, recipe_name="k3p",
                  batch_size=16, particles=32, z_dim=2, **recipe_overrides):
    if fixture_cls is None:
        fixture_cls = (ALICESpiralBiGANFixture if case and case["id"] == "api-alice-spiral-bigan"
                       else Ring16BiGANFixture)
    case = deepcopy(case if case is not None else
                    (_alice_spiral_case() if fixture_cls is ALICESpiralBiGANFixture else _bigan_case()))
    case["software_cohort"] = COHORT
    case["scope"] += " Separate bounded noise-free software cohort; no learned-quality qualification."
    recipe = get_recipe(recipe_name, input_noise_std=0., output_noise_std=0.,
                        batch_size=batch_size, num_particles=particles, z_dim=z_dim,
                        prior_kind="mog", sigma_rel=0., standardize=False,
                        total_steps=max_steps, **recipe_overrides)
    return fixture_cls(case, recipe=recipe, max_steps=max_steps, batch_size=batch_size,
                       particles=particles, z_dim=z_dim, device="cpu", seed=0)
