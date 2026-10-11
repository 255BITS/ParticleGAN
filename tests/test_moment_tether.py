"""Bounded software acceptance, distinct from Forge scientific qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.moment_tether import MomentTetherRegularizer
from experiments.forge.api import FormulationContext
from experiments.forge.behavior_adapters import run_behavior
from experiments.forge.techniques import recipe_field_active, validate_same_technique


ROOT = Path(__file__).resolve().parents[1]
LEADER = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json").read_text())


def bind(rows):
    prior = nn.Module()
    prior.z = nn.Parameter(rows.clone())
    get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.1).apply_prior_policy(prior)
    return prior


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("transform", ["shift", "expand", "contract", "shear"])
def test_analytic_objective_and_autograd(transform):
    initial = torch.tensor([[-1., 0.], [0., -1.], [1., 0.], [0., 1.]], dtype=torch.float64)
    prior = bind(initial)
    loss = MomentTetherRegularizer(prior, eps=.01, weight=.3)
    assert loss(initial).item() == 0
    if transform == "shift":
        current = initial + torch.tensor([.2, -.1])
    elif transform == "shear":
        current = initial @ torch.tensor([[1., .3], [0., 1.]], dtype=torch.float64)
    else:
        current = initial * (1.2 if transform == "expand" else .8)
    current.requires_grad_()
    delta = current.mean(0) - initial.mean(0)
    scale = initial.var(0) + .01
    cov = (current - current.mean(0)).T @ (current - current.mean(0)) / 3
    expected = .3 * ((delta.square() / scale).mean()
        + ((cov - initial.T @ initial / 3).square() / torch.outer(scale, scale)).mean())
    assert torch.equal(loss(current), expected)
    assert loss(current).item() > 0
    assert torch.autograd.gradcheck(loss, (current,))


def test_particle_permutation_is_free_and_multiple_priors_pool_exact_initial_moments():
    rows = torch.tensor([[-1., .1], [1., .2], [.2, -1.], [0., .8]], dtype=torch.float64)
    a, b = bind(rows[:1]), bind(rows[1:])
    loss = MomentTetherRegularizer((a, b))
    mean, covariance = loss.reference_moments()
    assert torch.allclose(mean, rows.mean(0), atol=1e-15, rtol=0)
    centered = rows - rows.mean(0)
    assert torch.allclose(covariance, centered.T @ centered / 3, atol=1e-15, rtol=0)
    assert loss(rows[[2, 0, 3, 1]]).item() < 1e-30


def test_zero_and_singleton_references_remain_finite_and_pull_mean():
    prior = bind(torch.zeros(1, 2, dtype=torch.float64))
    rows = torch.ones(1, 2, dtype=torch.float64, requires_grad=True)
    loss = MomentTetherRegularizer(prior, eps=.01)(rows)
    assert loss.item() == 100
    assert torch.equal(torch.autograd.grad(loss, rows)[0], torch.full_like(rows, 100.))


def build(*, weight=.01, regularizer="moment_tether", policy="learned", particles=12):
    ctx = FormulationContext(recipe_preset="bcap", recipe_overrides=dict(LEADER["recipe_overrides"],
        prior_regularizer=regularizer, prior_reg=weight, prior_update=policy,
        num_particles=particles, z_dim=2, batch_size=6, total_steps=4),
        prior=dict(kind="mog", sigma=.025, standardize=False),
        prior_contract="recipe_owned_v1", seed=0)
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)), component="discriminator")
    return ctx, ctx.build_trainer(g, d)


@pytest.mark.parametrize("policy", ["learned", "frozen"])
def test_disabled_tether_has_baseline_tensor_optimizer_rng_checkpoint_parity(policy):
    a_ctx, a = build(weight=0 if policy == "learned" else .1, policy=policy)
    b_ctx, b = build(weight=0 if policy == "learned" else .1, regularizer="vicreg", policy=policy)
    for _ in range(2):
        real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 5
        a.step(real)
        b.step(real)
    assert not any(name.startswith("moment_tether") for name in a.prior.state_dict())
    a_state, b_state = a_ctx.state_dict(), b_ctx.state_dict()
    # Declared regularizer kind differs; every trained/checkpointed byte agrees.
    for state in (a_state, b_state):
        state["recipe"]["prior_regularizer"] = "vicreg"
        state["trainer"]["recipe"]["prior_regularizer"] = "vicreg"
    equal(a_state, b_state)


def test_checkpoint_restore_retains_initial_reference_and_exact_next_update():
    ctx, trainer = build()
    reference = trainer.prior.moment_tether_mean.clone()
    real = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 5
    trainer.step(real)
    saved = ctx.state_dict()
    trainer.step(real)
    expected = ctx.state_dict()
    restored, next_trainer = build()
    # The checkpoint owns the reference; constructor bindings must read restored buffers.
    next_trainer.prior.moment_tether_mean.add_(10)
    restored.load_state_dict(saved)
    assert torch.equal(next_trainer.prior.moment_tether_mean, reference)
    next_trainer.step(real)
    equal(expected, restored.state_dict())


def test_reference_capture_and_objective_consume_no_rng_and_apply_weight_once():
    active_ctx, active = build(weight=.3)
    baseline_ctx, baseline = build(weight=0, regularizer="vicreg")
    equal(active_ctx.streams.state_dict(), baseline_ctx.streams.state_dict())
    assert torch.equal(active.prior.z, baseline.prior.z)
    before = torch.get_rng_state().clone()
    fixed_reference = active.prior.moment_tether_mean.clone()
    with torch.no_grad():
        active.prior.z.add_(.1)
    active.recipe.apply_prior_policy(active.prior)
    assert torch.equal(active.prior.moment_tether_mean, fixed_reference)
    actual = active.recipe.prior_regularization(active.prior.z, reference=active.prior)
    expected = .3 * active.prior_regularizer(active.prior.z)
    assert torch.equal(actual, expected)
    torch.autograd.grad(actual, active.prior.z)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("particles", [12, 1050])
def test_penalty_keeps_host_row_selection_and_unsampled_dualnorm_rows_fixed(particles):
    _, trainer = build(particles=particles)
    with torch.no_grad():
        trainer.prior.z.add_(.2)
    previous = trainer.prior.z.clone()
    seen = []
    hook = trainer.prior_regularizer.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    trainer.step(torch.arange(12, dtype=torch.float32).reshape(6, 2) / 5)
    hook.remove()
    moved = (trainer.prior.z != previous).any(1).sum().item()
    assert 0 < moved <= 6
    assert seen == [12] if particles == 12 else 1 <= seen[0] <= 6


def test_word_policy_resume_restores_original_reference():
    from benchmarks.toy_audit.api_images import WordFixture
    def fixture():
        ctx = FormulationContext(recipe_preset="bcap", recipe_overrides=dict(LEADER["recipe_overrides"],
            num_particles=5, z_dim=2, batch_size=256, total_steps=20000,
            prior_regularizer="moment_tether", prior_reg=.01),
            prior=dict(kind="particle_cloud", sigma=0., standardize=False,
                       exception_reason="declared five-row word software fixture"),
            prior_contract="recipe_owned_v1", seed=0)
        return WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=4, components=ctx)
    a = fixture()
    a.step()
    saved = deepcopy(a.policy.state_dict())
    data_state = a.data_generator.get_state().clone()
    a.step()
    expected = a.policy.state_dict()
    b = fixture()
    b.policy.load_state_dict(saved)
    b.data_generator.set_state(data_state)
    b.step()
    equal(expected, b.policy.state_dict())


def test_behavioral_multi_prior_reference_is_persistent_and_penalty_activates(tmp_path):
    task = json.loads((ROOT / "configs/forge/tasks/cover_leftover.json").read_text())
    task["execution"]["steps"] = 2
    candidate = deepcopy(LEADER)
    candidate["recipe_overrides"].update(prior_regularizer="moment_tether", prior_reg=.01)
    result = run_behavior(dict(candidate=candidate, protocol=dict(seed=0)), task, tmp_path)
    state = torch.load(tmp_path / "component-state.pt", weights_only=False)
    assert result["evidence"]["prior_policy"]["regularizer"] == "moment_tether"
    for name in ("prior0", "prior1"):
        assert state["models"][name]["moment_tether_count"].item() == 12


def test_forge_weight_axis_is_active_and_std_target_inactive():
    recipe = get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.01)
    assert recipe_field_active("prior_reg", recipe)
    assert recipe_field_active("prior_reg_eps", recipe)
    assert not recipe_field_active("prior_reg_target_std", recipe)
    validate_same_technique(recipe, recipe.replace(prior_reg=.1))
    with pytest.raises(ValueError, match="technique mechanisms"):
        validate_same_technique(recipe, recipe.replace(prior_regularizer="vicreg"))
