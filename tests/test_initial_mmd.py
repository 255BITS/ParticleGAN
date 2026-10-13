"""Mathematical, public-host, disabled-parity and continuation checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import InitialMMDRegularizer, Recipe, get_recipe
from particlegan.initial_mmd import _panel
from experiments.forge.api import FormulationContext
from experiments.forge.behavior_adapters import run_behavior
from experiments.forge.techniques import recipe_field_active


ROOT = Path(__file__).resolve().parents[1]


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def context(**settings):
    return FormulationContext(recipe_preset="bcap", recipe_overrides={
        "num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4, **settings},
        prior=dict(kind="mog", sigma=.025, standardize=False),
        prior_contract="recipe_owned_v1", seed=0)


def trainer(ctx):
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component="discriminator")
    return ctx.build_trainer(g, d)


def test_matches_explicit_biased_imq_estimator_and_gradients():
    reference = torch.tensor([[-1., .2], [0., .5], [1., -.1]], dtype=torch.float64)
    rows = torch.tensor([[-.5, .1], [.2, .3], [1.4, -.7]], dtype=torch.float64, requires_grad=True)
    regularizer = InitialMMDRegularizer(reference, weight=.3)
    scale = (reference - reference.mean(0)).square().mean(0).sum()
    def mean_kernel(a, b):
        distance = (a[:, None] - b[None]).square().sum(-1)
        return sum((scale * w / (scale * w + distance)).mean() for w in (.25, 1., 4.)) / 3
    expected = .3 * (mean_kernel(rows, rows) + mean_kernel(reference, reference) - 2 * mean_kernel(rows, reference))
    actual = regularizer(rows)
    assert torch.allclose(actual, expected, atol=1e-15, rtol=1e-14)
    assert torch.allclose(torch.autograd.grad(actual, rows)[0], torch.autograd.grad(expected, rows)[0], atol=1e-15)
    assert torch.autograd.gradcheck(regularizer, (rows,))


def test_identity_and_particle_exchange_have_zero_distance():
    reference = torch.tensor([[-1., .2], [0., .5], [1., -.1]], dtype=torch.float64)
    regularizer = InitialMMDRegularizer(reference)
    assert regularizer(reference).item() == 0
    assert regularizer(reference[[2, 0, 1]]).item() <= 1e-15
    assert regularizer(reference + 1).item() > .1
    assert regularizer(reference * 2).item() > .01


def test_panels_are_bounded_deterministic_and_only_selected_rows_receive_gradients():
    rows = torch.arange(20_000, dtype=torch.float64)[:, None] / 10_000
    regularizer = InitialMMDRegularizer(rows)
    assert regularizer.reference.shape == (256, 1)
    assert torch.equal(regularizer.reference, _panel(rows))
    selected = (rows[[4, 8, 16, 32]] + 1).requires_grad_()
    assert torch.isfinite(torch.autograd.grad(regularizer(selected), selected)[0]).all()
    all_rows = (rows + .2).requires_grad_()
    gradient = torch.autograd.grad(regularizer(all_rows), all_rows)[0]
    assert (gradient.abs().sum(1) > 0).sum() <= 256


def test_collapsed_reference_uses_finite_fixed_scale_floor():
    regularizer = InitialMMDRegularizer(torch.zeros(4, 2, dtype=torch.float64))
    rows = torch.ones(4, 2, dtype=torch.float64, requires_grad=True)
    assert regularizer.scale.item() == 1e-4
    assert torch.isfinite(regularizer(rows))
    assert torch.isfinite(torch.autograd.grad(regularizer(rows), rows)[0]).all()


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        InitialMMDRegularizer(torch.zeros(2, 2), weight=weight)


def test_factory_requires_initialized_reference_and_weight_applies_once():
    recipe = Recipe(prior_regularizer="initial_mmd", prior_reg=.3, prior_l2=.2)
    with pytest.raises(ValueError, match="after public initialization"):
        recipe.make_prior_regularizer()
    prior = recipe.make_prior()
    before = prior.z.detach().clone()
    regularizer = recipe.make_prior_regularizer(prior=prior)
    prior.z.data.add_(.3)
    expected = .3 * InitialMMDRegularizer(before)(prior.z) + .2 * prior.z.square().mean()
    assert torch.equal(recipe.prior_regularization(prior.z, regularizer=regularizer), expected)
    rng = torch.get_rng_state().clone()
    rebind = recipe.make_prior_regularizer(prior=prior)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(rebind(prior.z), regularizer(prior.z))


@pytest.mark.parametrize("settings", [dict(prior_reg=0), dict(prior_update="frozen", prior_reg=.1)])
def test_disabled_or_frozen_factory_has_no_reference_work(settings, monkeypatch):
    monkeypatch.setattr(InitialMMDRegularizer, "from_priors", lambda *a, **kw: pytest.fail("reference captured"))
    recipe = Recipe(prior_regularizer="initial_mmd", **settings)
    regularizer = recipe.make_prior_regularizer()
    rows = torch.full((2, 2), float("nan"), requires_grad=True)
    assert regularizer(rows).item() == 0
    assert recipe.prior_regularization(rows, regularizer=regularizer).item() == 0


def test_zero_weight_matches_bcap_leader_models_optimizers_and_streams():
    base, disabled = context(), context(prior_regularizer="initial_mmd", prior_reg=0)
    a, b = trainer(base), trainer(disabled)
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    for _ in range(2):
        a.step(real); b.step(real)
    for name in ("G", "D", "prior", "opt_g", "opt_d"):
        equal(getattr(a, name).state_dict(), getattr(b, name).state_dict())
    equal(base.streams.state_dict(), disabled.streams.state_dict())
    assert not hasattr(b.prior, "_initial_mmd_reference")


def test_original_reference_survives_exact_checkpoint_continuation():
    c = context(prior_regularizer="initial_mmd", prior_reg=.1)
    a = trainer(c)
    original = a.prior._initial_mmd_reference.clone()
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    a.step(real)
    saved = c.state_dict()
    a.step(real)
    expected = c.state_dict()
    restored = context(prior_regularizer="initial_mmd", prior_reg=.1)
    b = trainer(restored)
    restored.load_state_dict(saved)
    b.step(real)
    equal(expected, restored.state_dict())
    assert torch.equal(original, b.prior._initial_mmd_reference)
    assert not torch.equal(original, b.prior.z)


def test_reference_follows_prior_dtype_conversion():
    recipe = get_recipe("bcap", prior_regularizer="initial_mmd", prior_reg=.1)
    prior = recipe.make_prior()
    regularizer = recipe.make_prior_regularizer(prior=prior)
    prior.double()
    assert regularizer(prior.z).dtype == torch.float64


def test_dualnorm_large_table_keeps_unsampled_rows_fixed():
    c = context(num_particles=2048, prior_regularizer="initial_mmd", prior_reg=.1)
    a = trainer(c)
    before = a.prior.z.detach().clone()
    a.step(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4)
    gradient_rows = a.prior.z.grad.abs().sum(1) > 0
    changed_rows = (a.prior.z.detach() != before).any(1)
    assert 0 < int(gradient_rows.sum()) <= 4
    assert not changed_rows[~gradient_rows].any()
    assert changed_rows.any()


@pytest.mark.parametrize("name", ["trajectory", "cover_leftover", "ae_gan_hold", "residual_student"])
def test_behavior_hosts_capture_after_public_initialization_and_call_once(name, tmp_path, monkeypatch):
    task = json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())
    task["execution"]["steps"] = 2
    calls = []
    original = InitialMMDRegularizer.forward
    def observed(module, rows):
        calls.append(module.weight)
        return original(module, rows)
    monkeypatch.setattr(InitialMMDRegularizer, "forward", observed)
    result = run_behavior(dict(candidate=dict(recipe_preset="bcap", recipe_overrides=dict(
        prior_regularizer="initial_mmd", prior_reg=.1)), protocol=dict(seed=0)), task, tmp_path)
    assert calls == [.1, .1]
    assert result["evidence"]["prior_policy"]["regularizer"] == "initial_mmd"
    saved = torch.load(tmp_path / "component-state.pt", weights_only=False)
    reference = saved["models"]["prior0"]["_initial_mmd_reference"]
    assert reference.shape[0] <= 256
    assert torch.isfinite(reference).all()


def test_forge_declares_weight_active_and_vicreg_only_knobs_inactive():
    recipe = get_recipe("bcap", prior_regularizer="initial_mmd", prior_reg=.1)
    assert recipe_field_active("prior_reg", recipe)
    assert not recipe_field_active("prior_reg_target_std", recipe)
    assert not recipe_field_active("prior_reg_eps", recipe)


def test_word_prefix_restore_keeps_original_reference_and_exact_next_update():
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.api import task_formulation_context
    task = json.loads((ROOT / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    candidate = dict(recipe_preset="bcap", recipe_overrides=dict(prior_regularizer="initial_mmd", prior_reg=.1))
    ctx = task_formulation_context(candidate, task, dict(seed=0))
    a = WordFixture(device="cpu", seed=0, recipe_name="bcap", max_steps=4, components=ctx)
    initial = a.prior._initial_mmd_reference.clone()
    a.step()
    saved, streams = deepcopy(a.state_dict()), deepcopy(ctx.streams.state_dict())
    a.step()
    expected = a.state_dict()
    other = task_formulation_context(candidate, task, dict(seed=0))
    b = WordFixture(device="cpu", seed=0, recipe_name="bcap", max_steps=4, components=other)
    other.streams.load_state_dict(streams)
    b.policy.load_state_dict(saved["api_state"])
    b.data_generator.set_state(saved["data_generator"])
    b.step()
    equal(expected, b.state_dict())
    assert torch.equal(initial, b.prior._initial_mmd_reference)


def test_direct_coordinate_host_remains_explicitly_not_sampled(tmp_path):
    task = json.loads((ROOT / "configs/forge/tasks/two_pole.json").read_text())
    task["execution"]["steps"] = 2
    result = run_behavior(dict(candidate=dict(recipe_preset="bcap", recipe_overrides=dict(
        prior_regularizer="initial_mmd", prior_reg=.1)), protocol=dict(seed=0)), task, tmp_path)
    assert result["evidence"]["prior_policy"]["applicability"] == "not_sampled"
