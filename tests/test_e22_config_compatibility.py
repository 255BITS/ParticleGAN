"""Existing research config and pre-routed recovery retain independent E22."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from examples.e22_external_loop import make_loop, update
from particlegan import GANTrainer, Recipe, get_recipe


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    elif isinstance(left, float) and left != left:
        assert right != right
    else:
        assert left == right


def test_existing_research_config_resolves_identical_independent_controls():
    fields = json.loads((Path(__file__).parents[1] / "configs/100gaussians/e22-noout.json").read_text())
    resolved = Recipe(**fields)
    dimensions = {key: fields[key] for key in ("num_particles", "z_dim", "batch_size")}
    assert get_recipe("e22", **dimensions) == resolved.replace(name="e22")
    assert resolved.row_policy == "independent"
    assert resolved.particle_birth_death and resolved.row_evidence_gate
    assert resolved.birth_death_isolation and resolved.birth_death_feature_scale == "std"
    assert resolved.table_release_rule == "anchor" and resolved.reopen_signal == "none"


def _owner(surface):
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(709)
        G = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        D = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        prior = recipe.make_prior()
    if surface == "trainer":
        return GANTrainer(recipe, G, D, prior=prior, seed=101, serial_backward=True)
    return make_loop(recipe, G, D, prior, seed=101)


@pytest.mark.parametrize("surface", ["trainer", "policy"])
def test_pre_routed_checkpoints_without_new_config_fields_continue_exactly(surface):
    owner = _owner(surface)
    policy = owner.policy
    real = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8

    def step(loop, batch):
        with torch.autograd.set_multithreading_enabled(False):
            return loop.step(batch) if surface == "trainer" else update(loop, batch)

    step(owner, real)
    recovery = owner if surface == "trainer" else policy
    previous = deepcopy(recovery.state_dict())
    previous["recipe"].pop("row_policy")
    if surface == "policy":
        previous.pop("routing", None)
    expected_loss = step(owner, real.flip(0))
    expected = recovery.state_dict()
    expected_served = policy.served_snapshot()

    restored = _owner(surface)
    restored_recovery = restored if surface == "trainer" else restored.policy
    restored_recovery.load_state_dict(previous)
    _equal(expected_loss, step(restored, real.flip(0)))
    _equal(expected, restored_recovery.state_dict())
    _equal(expected_served, restored.policy.served_snapshot())
