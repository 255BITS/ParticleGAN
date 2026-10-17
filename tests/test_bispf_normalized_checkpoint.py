"""Software continuation on nonconstant image batches; no quality claim."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.bispf import _LAST_THERMAL_SCALE
from particlegan.init import deterministic_orthogonal_


@pytest.fixture(autouse=True)
def isolated_thermal_cache():
    previous = dict(_LAST_THERMAL_SCALE)
    _LAST_THERMAL_SCALE.clear()
    yield
    _LAST_THERMAL_SCALE.clear()
    _LAST_THERMAL_SCALE.update(previous)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def trainer(family=None):
    options = {} if family is None else {"optimizer_family": family}
    recipe = get_recipe("bispf_optimized" if family is None else "bispf",
                        num_particles=32, z_dim=4, batch_size=8, total_steps=6, **options)
    generator = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Unflatten(1, (1, 4, 4)))
    critic = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Tanh(), nn.Flatten(), nn.Linear(32, 1))
    prior = recipe.make_prior()
    for component in (generator, critic, prior):
        deterministic_orthogonal_(component, seed=0)
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def batch(step):
    values = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 1, 4, 4)
    return torch.sin(values * .13 + step) * (.3 + step * .1)


@pytest.mark.parametrize("family", [None, "adam", "formulation"])
def test_nonconstant_images_resume_exact_next_three_steps_and_observer(family):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        live = trainer(family)
        for step in range(3):
            stats = live.step(batch(step))
            assert all(torch.isfinite(value).all() for value in stats.values() if isinstance(value, torch.Tensor))
        saved = live.state_dict()
        record = saved["optimizers"][1]["regularizer"]["record"]
        observer = record.get("bispf", record)
        assert observer["initialized"] and observer["ema_data_scale"] != 1.
        for step in range(3, 6):
            live.step(batch(step))
        expected = live.state_dict()
        restored = trainer(family)
        restored.load_state_dict(saved)
        equal(restored.opt_d.state_dict(), saved["optimizers"][1])
        for step in range(3, 6):
            restored.step(batch(step))
        equal(expected, restored.state_dict())
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.parametrize("family", [None, "adam", "formulation"])
def test_exact_thermal_value_and_device_state_restore(family):
    live = trainer(family)
    live.opt_d.record.ema_residual = .0001
    live.opt_d.record.ema_data_scale = .37
    live.opt_d.record.initialized = True
    live.opt_d.record._thermal_scale = .00001
    saved = live.opt_d.state_dict()
    restored = trainer(family)
    restored.opt_d.load_state_dict(saved)
    assert restored.opt_d.record._thermal_scale == .00001
    assert _LAST_THERMAL_SCALE[torch.device("cpu")] == .00001
    equal(saved, restored.opt_d.state_dict())


def test_historical_formulation_packet_retains_earlier_thermal_reconstruction():
    live = trainer("formulation")
    saved = live.opt_d.state_dict()
    record = saved["regularizer"]["record"]
    record.pop("thermal_scale")
    record["ema_residual"] = .001
    live.opt_d.load_state_dict(saved)
    assert live.opt_d.record._thermal_scale == .05


def test_incomplete_historical_bispf_observer_fails_before_mutation():
    live = trainer()
    before = deepcopy(live.opt_d.state_dict())
    old = deepcopy(before)
    old["regularizer"]["record"].pop("bispf")
    with pytest.raises(ValueError, match="missing transport observer state"):
        live.opt_d.load_state_dict(old)
    equal(before, live.opt_d.state_dict())


@pytest.mark.parametrize("field,value", [("ema_residual", -1), ("ema_data_scale", float("nan")),
                                         ("thermal_scale", 1.1), ("initialized", 1), ("schema", True)])
def test_malformed_observer_rejected_atomically(field, value):
    live = trainer()
    before = deepcopy(live.opt_d.state_dict())
    bad = deepcopy(before)
    bad["regularizer"]["record"]["bispf"][field] = value
    with pytest.raises(ValueError, match="BISPF observer"):
        live.opt_d.load_state_dict(bad)
    equal(before, live.opt_d.state_dict())


def test_bcap_observer_packet_keeps_its_original_fields():
    optimizer = get_recipe("bcap").make_critic_optimizer(nn.Linear(2, 1))
    assert set(optimizer.state_dict()["regularizer"]["record"]) == {
        "lr_max", "lr_last", "anchor_started", "calls", "observed_steps"}
