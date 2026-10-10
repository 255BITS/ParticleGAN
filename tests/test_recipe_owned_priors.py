"""Bounded software checks; no scientific qualification or recipe retuning."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GaussianPrior, Recipe, get_recipe
from particlegan.vicreg_loss import ParticleRegularizer
from experiments.forge.api import FormulationContext, task_formulation_context
from experiments.forge.behavior_adapters import BehaviorComponents, run_behavior
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.priors import (expected_prior_updates, prior_policy_receipt,
                                     task_prior, validate_prior_policy)
from experiments.forge.taskrecipes import bind_task_candidate
from experiments.forge.views import _guards


ROOT = Path(__file__).resolve().parents[1]


def task(name="vector_two_broad"):
    return json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())


def context(policy="learned", **settings):
    return FormulationContext(recipe_preset="bcap_adam",
        recipe_overrides=dict(prior_update=policy, num_particles=12, z_dim=2,
            batch_size=4, total_steps=4, **settings),
        prior=dict(kind="mog", sigma=.025, standardize=False),
        prior_contract="recipe_owned_v1", seed=0)


def trainer(ctx):
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component="discriminator")
    return ctx.build_trainer(g, d)


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


@pytest.mark.parametrize("policy", ["learned", "frozen"])
def test_prior_policy_is_recipe_owned_in_task_context_and_receipt(policy):
    t = task()
    c = task_formulation_context(dict(recipe_preset="bcap_adam", recipe_overrides={"prior_update": policy}), t)
    assert "learnable" not in task_prior(t)
    assert c.prior_config["learnable"] == (policy == "learned")
    assert c.build_prior().z.requires_grad == (policy == "learned")
    receipt = c.receipt()
    assert receipt["initial_prior"] == t["execution"]["prior"]
    assert receipt["field_ownership"]["task_contract"]["prior"]["update_policy"]["owner"] == "technique"
    assert recipe_field_owner("prior_reg", task("trajectory")) == "hyperparameter"


def test_new_task_rejects_hidden_learnable_and_legacy_still_requires_it():
    t = task()
    t["execution"]["prior"]["learnable"] = True
    with pytest.raises(ValueError, match="omit learnable"):
        task_prior(t)
    t["execution"].pop("prior_contract")
    assert task_prior(t)["learnable"] is True
    t["execution"]["prior"].pop("learnable")
    with pytest.raises(ValueError, match="explicit"):
        task_prior(t)


def test_frozen_and_learned_start_from_identical_locations_and_streams():
    learned, frozen = context(), context("frozen")
    a, b = trainer(learned), trainer(frozen)
    equal(a.prior.state_dict(), b.prior.state_dict())
    equal(learned.streams.state_dict(), frozen.streams.state_dict())
    for x, y in zip(a.G.parameters(), b.G.parameters()):
        assert torch.equal(x, y)
    assert not list(b.prior.parameters())
    assert all(b.prior.z is not p for group in b.opt_g.param_groups for p in group["params"])


def test_frozen_training_and_checkpoint_resume_preserve_locations_and_all_streams():
    c = context("frozen", prior_reg=.3, prior_l2=.2)
    a = trainer(c)
    locations = a.prior.z.clone()
    initial_g, initial_d = deepcopy(a.G.state_dict()), deepcopy(a.D.state_dict())
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    a.step(real)
    assert torch.equal(a.prior.z, locations)
    assert any(not torch.equal(value, initial_g[name]) for name, value in a.G.state_dict().items())
    assert any(not torch.equal(value, initial_d[name]) for name, value in a.D.state_dict().items())
    saved = c.state_dict()
    a.step(real)
    after = c.state_dict()
    restored = context("frozen", prior_reg=.3, prior_l2=.2)
    b = trainer(restored)
    restored.load_state_dict(saved)
    b.step(real)
    equal(after, restored.state_dict())
    assert torch.equal(b.prior.z, locations)


def test_regularizer_weight_and_l2_are_applied_once_with_declared_settings():
    recipe = Recipe(prior_reg=.3, prior_l2=.2, prior_reg_target_std=.7, prior_reg_eps=1e-3)
    rows = torch.tensor([[.1, -.1], [.2, .3], [.3, -.2]], requires_grad=True)
    actual = recipe.prior_regularization(rows)
    expected = .3 * ParticleRegularizer(target_std=.7, eps=1e-3)(rows) + .2 * rows.square().mean()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, rows)[0], torch.autograd.grad(expected, rows)[0])


@pytest.mark.parametrize("settings", [dict(prior_reg=0), dict(prior_regularizer="none"),
                                     dict(prior_update="frozen", prior_reg=.3, prior_l2=.2)])
def test_disabled_or_frozen_regularizer_does_no_auxiliary_work(settings, monkeypatch):
    monkeypatch.setattr(ParticleRegularizer, "forward", lambda *args: pytest.fail("disabled prior penalty evaluated"))
    recipe = Recipe(**settings)
    rows = torch.full((2, 2), float("nan"), requires_grad=True)
    assert recipe.prior_regularization(rows).item() == 0


def test_zero_factory_short_circuits_nonfinite_rows():
    assert ParticleRegularizer(weight=0)(torch.full((2, 2), float("nan"))).item() == 0


def test_gaussian_prior_has_no_location_policy_or_optimizer_group():
    recipe = Recipe(prior_update="frozen")
    assert len(recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), GaussianPrior(2))[0].param_groups) == 1


@pytest.mark.parametrize("settings", [dict(prior_update="invalid"), dict(prior_regularizer="unknown"),
    dict(prior_regularizer="none", prior_reg=.1), dict(prior_reg_eps=0), dict(prior_l2=-.1),
    dict(prior_reg_target_std=float("nan")), dict(prior_update="frozen", row_evidence_gate=True)])
def test_invalid_policy_or_regularizer_rejected_before_construction(settings):
    with pytest.raises(ValueError):
        Recipe(**settings)


def test_factory_and_raw_latent_optimizer_cannot_override_frozen_policy():
    recipe = get_recipe("bcap_adam", prior_update="frozen", num_particles=12, z_dim=2)
    with pytest.raises(ValueError, match="contradicts"):
        recipe.make_prior(learnable=True)
    rows = nn.Parameter(torch.zeros(12, 2))
    with pytest.raises(ValueError, match="contradicts"):
        recipe.make_generator_optimizer([rows], latent_table=rows)


def test_release_reference_prior_weight_is_not_delegated_to_new_host():
    candidate = json.loads((ROOT / "configs/forge/ideas/release07-gan-v3-task-adapted-v1.json").read_text())
    for name in ("trajectory", "ae_gan_hold", "two_pole"):
        assert bind_task_candidate(candidate, task(name))["recipe_overrides"]["prior_reg"] == .05


@pytest.mark.parametrize("name", ["trajectory", "residual_student", "cover_leftover", "ae_gan_hold"])
def test_behavioral_host_consumes_recipe_regularizer_once(name, tmp_path, monkeypatch):
    t = task(name)
    t["execution"]["steps"] = 2
    calls = []
    original = ParticleRegularizer.forward
    def observed(module, rows):
        calls.append((module.weight, module.target_std))
        return original(module, rows)
    monkeypatch.setattr(ParticleRegularizer, "forward", observed)
    candidate = dict(recipe_preset="bcap_adam", recipe_overrides=dict(prior_reg=.2, prior_reg_target_std=.8))
    result = run_behavior(dict(candidate=candidate, protocol=dict(seed=0)), t, tmp_path)
    assert calls == [(.2, .8)] * 2
    assert result["evidence"]["prior_policy"]["weight"] == .2


def test_frozen_behavior_has_no_prior_optimizer_and_regularization(tmp_path, monkeypatch):
    t = task("trajectory")
    t["execution"]["steps"] = 2
    monkeypatch.setattr(ParticleRegularizer, "forward", lambda *args: pytest.fail("frozen prior penalty evaluated"))
    candidate = dict(recipe_preset="bcap_adam", recipe_overrides=dict(prior_update="frozen", prior_reg=.2, prior_l2=.3))
    result = run_behavior(dict(candidate=candidate, protocol=dict(seed=0)), t, tmp_path)
    assert result["evidence"]["guards"]["optimizer_updates"]["prior"] == 0
    assert not result["evidence"]["prior_policy"]["trainable_locations"]
    state = torch.load(tmp_path / "component-state.pt", weights_only=False)
    assert not state["role_parameters"]["prior"]


def test_frozen_guard_requires_exact_zero_and_complete_network_budgets():
    t = task()
    t["execution"]["steps"] = 4
    t["evaluation"]["guards"] = dict(optimizer_roles=["generator", "discriminator", "prior"], exact_optimizer_updates=True)
    frozen = Recipe(prior_update="frozen").make_prior()
    evidence = dict(prior_policy=prior_policy_receipt(Recipe(prior_update="frozen"), frozen),
        guards=dict(optimizer_updates=dict(generator=4, discriminator=4, prior=0)))
    assert expected_prior_updates(t, evidence, 4) == 0
    assert _guards(t, evidence) is None
    for role, value in (("prior", 1), ("generator", 3), ("discriminator", 3)):
        wrong = deepcopy(evidence)
        wrong["guards"]["optimizer_updates"][role] = value
        assert _guards(t, wrong)["status"] == "INCOMPLETE"
    evidence.pop("prior_policy")
    assert _guards(t, evidence)["status"] == "FAIL"


def test_legacy_host_keeps_original_prior_weight_and_task_learning_policy():
    t = task("trajectory")
    t["execution"].pop("prior_contract")
    t["execution"]["prior"]["learnable"] = True
    components = BehaviorComponents(dict(candidate={}, protocol=dict(seed=0)), t)
    assert not components.uses_recipe_prior
    assert recipe_field_owner("prior_reg", t) == "task"


@pytest.mark.parametrize("policy", ["learned", "frozen"])
def test_policy_proof_cannot_lie_about_actual_recipe_in_either_direction(policy):
    recipe = Recipe(prior_update=policy, num_particles=12, z_dim=2)
    rows = recipe.make_prior()
    evidence = dict(prior_policy=prior_policy_receipt(recipe, rows))
    assert validate_prior_policy(task(), evidence, asdict(recipe)) is None
    other = recipe.replace(prior_update="frozen" if policy == "learned" else "learned")
    assert validate_prior_policy(task(), evidence, asdict(other))["status"] == "INVALID"
    assert validate_prior_policy(task(), {}, asdict(recipe))["status"] == "INCOMPLETE"


@pytest.mark.parametrize("name", ["gaussian1d_smoke", "five_word_joint_smoke"])
def test_specialized_frozen_reducers_preserve_network_budgets(name):
    if name == "gaussian1d_smoke":
        from test_forge_gaussian_smoke import evidence
        from experiments.forge.gaussian_tasks import grade
    else:
        from test_forge_word_smoke_hold import evidence
        from experiments.forge.word_tasks import grade
    recipe = Recipe(prior_update="frozen", num_particles=12, z_dim=2)
    raw = evidence()
    raw["prior_policy"] = prior_policy_receipt(recipe, recipe.make_prior())
    raw["guards"]["optimizer_updates"]["prior"] = 0
    assert grade(task(name), raw)["status"] == "PASS"
    raw["guards"]["optimizer_updates"]["generator"] -= 1
    assert grade(task(name), raw)["status"] == "INCOMPLETE"
