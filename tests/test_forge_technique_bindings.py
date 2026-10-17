"""Technique declarations bind mechanisms despite changes to public defaults."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from torch import nn

from experiments.forge.api import resolve_public_recipe, task_policy_blockers
from particlegan.ka2 import KA2CriticAdam
from particlegan.k3p import K3PCriticAdam


ROOT = Path(__file__).resolve().parents[1]


def idea(name):
    return json.loads((ROOT / "configs/forge/ideas" / f"{name}.json").read_text())


@pytest.mark.parametrize("name,field,value", [
    ("forge-no-critic-penalty", "reg_coeff", 0.0),
    ("forge-onboarding-anchor-ablation", "reg_anchor_weight", 0.0),
    ("k3p-a2-off-native-diagnostic", "latent_damping_max_rate", 0.0),
    ("k3p-no-output-noise-diagnostic", "output_noise_std", 0.0),
])
def test_k3p_ablation_changes_only_its_mechanism(name, field, value):
    baseline = resolve_public_recipe(idea("k3p"))
    card = idea(name)
    actual = resolve_public_recipe(card)
    assert actual == baseline.replace(**{field: value})
    assert actual.name == "k3p" and actual.effective_critic_formulation == "k3p"
    assert type(actual.make_critic_optimizer(nn.Linear(2, 1))) is K3PCriticAdam
    # Parent is provenance; dropping it cannot change the executed recipe.
    without_parent = deepcopy(card)
    without_parent.pop("parent")
    assert resolve_public_recipe(without_parent) == actual


@pytest.mark.parametrize("name,arm", [
    ("k3p-r1r2-matched-v1", "a_r1r2"),
    ("k3p-bcap-matched-v1", "b_cap"),
])
def test_fixed_penalty_baselines_retain_k3p_optimizer(name, arm):
    recipe = resolve_public_recipe(idea(name))
    assert recipe.reg_arm == arm and recipe.effective_critic_formulation == "k3p"
    assert recipe.reg_coeff == 1.0 and recipe.reg_kappa == 1.0
    optimizer = recipe.make_critic_optimizer(nn.Linear(2, 1))
    assert type(optimizer) is K3PCriticAdam
    assert recipe.make_critic_penalty(optimizer).regularizer.arm == arm


def test_ka2_reference_remains_the_current_public_formulation():
    recipe = resolve_public_recipe(idea("ka2"))
    assert recipe.name == "ka2" and recipe.effective_critic_formulation == "ka2"
    assert type(recipe.make_critic_optimizer(nn.Linear(2, 1))) is KA2CriticAdam


@pytest.mark.parametrize("name", ["e22", "atlas"])
def test_policy_techniques_cannot_inherit_clean_live_qualification(name):
    task = json.loads((ROOT / "configs/forge/tasks/mode_hold.json").read_text())
    card = idea(name)
    recipe = resolve_public_recipe(card)
    assert recipe.total_steps is None and recipe.serve_average == 4.0
    assert card["prior"]["kind"] == "particle_cloud"
    assert card["claim_contract"]["scoring_weights"] == "state_selected"
    assert any("policy-aware task" in blocker for blocker in task_policy_blockers(task, card))
