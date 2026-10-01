"""API integration probes retain E22 ownership; they are not quality gates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import Recipe, get_recipe
from particlegan.k3p import K3PCriticAdam
from particlegan.ka2 import KA2CriticAdam
from experiments.forge.api import (CapabilityError, FormulationContext,
                                   host_recipe_overrides, task_policy_blockers)
from experiments.forge.adapters import _context, adapter_preflight
from experiments.forge.clockfree import learning_state, source_audit
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[1]
CLOUD = {"kind": "particle_cloud", "sigma": 0., "standardize": False,
         "exception_reason": "API conformance fixture for independent equal-mass E22 rows"}


def test_formulations_are_explicit_and_default_checkpoint_recipe_stays_compatible():
    assert get_recipe("ka2") == get_recipe()
    assert "critic_formulation" not in get_recipe().to_dict()
    critic = nn.Linear(2, 1)
    assert type(get_recipe().make_critic_optimizer(critic)) is KA2CriticAdam
    k3p = get_recipe("k3p")
    assert type(k3p.make_critic_optimizer(critic)) is K3PCriticAdam
    assert k3p.to_dict()["critic_formulation"] == "k3p"
    assert Recipe(name="e22").continuous_policy is None


def test_installed_atlas_preset_matches_frozen_config_mechanisms():
    frozen = Recipe(**json.loads((ROOT / "configs/100gaussians/atlas.json").read_text()))
    assert get_recipe("atlas", num_particles=frozen.num_particles, z_dim=frozen.z_dim,
                      batch_size=frozen.batch_size).replace(name=frozen.name).to_dict() == frozen.to_dict()


@pytest.mark.parametrize("preset", ["e22", "atlas"])
def test_policy_context_resume_and_external_limit_preserve_full_state(preset):
    def build():
        context = FormulationContext(recipe_preset=preset, prior=CLOUD,
            recipe_overrides={"num_particles": 64, "z_dim": 2, "batch_size": 16}, seed=0)
        generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)),
                                      component="generator")
        critic = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)),
                                   component="discriminator")
        return context, context.build_trainer(generator, critic, max_steps=3)
    context, trainer = build()
    assert trainer.recipe.total_steps is None
    assert context.capabilities()["policy_controls"] and not context.capabilities()["live_sampling"]
    real = torch.arange(32, dtype=torch.float32).reshape(16, 2).sin()
    trainer.step(real)
    checkpoint = deepcopy(context.state_dict())
    restored, replica = build()
    restored.load_state_dict(checkpoint)
    trainer.step(real)
    replica.step(real)
    assert state_digest(context.state_dict()) == state_digest(restored.state_dict())
    assert context.receipt()["policy_lifecycle"]["private_rng"][0]["seed"] == 6
    # Sampling does not consume training streams or mutate policy state.
    before = state_digest(learning_state(context.state_dict()))
    trainer.sample(8, generator=context.streams.generator("eval", component="sampler", purpose="samples"))
    assert before == state_digest(learning_state(context.state_dict()))
    trainer.step(real)
    with pytest.raises(RuntimeError, match="budget"):
        trainer.step(real)
    trainer.extend_execution(4)
    trainer.step(real)
    assert trainer.completed_steps == 4 and trainer.recipe.total_steps is None


def test_policy_applicability_blocks_before_construction_or_task_reservation():
    with pytest.raises(CapabilityError, match="particles"):
        FormulationContext(recipe_preset="e22")
    with pytest.raises(CapabilityError, match="UpdatePolicy lifecycle"):
        FormulationContext(recipe_preset="e22", prior=CLOUD, execution_path="public_components")
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    candidate = {"recipe_preset": "atlas", "prior": CLOUD,
                 "resolved_recipe": Recipe().to_dict()}
    assert any("policy-aware task" in x for x in task_policy_blockers(task, candidate))
    assert any("policy-aware task" in x for x in adapter_preflight(task, candidate, root=ROOT))
    with pytest.raises(CapabilityError, match="policy-aware task"):
        _context({"candidate": candidate, "protocol": {"seed": 0}}, task, "cpu", {})


def test_host_resources_leave_schedule_free_horizon_and_external_budget_distinct():
    execution = {"steps": 7000, "original_schedule_horizon": 14000}
    resources = {"num_particles": 1000, "batch_size": 64}
    continuous = host_recipe_overrides({"recipe_preset": "e22"}, execution, resources)
    context = FormulationContext(recipe_preset="e22", recipe_overrides=continuous, prior=CLOUD)
    assert context.recipe.total_steps is None and context.recipe.num_particles == 1000
    scheduled = host_recipe_overrides({"recipe_preset": "k3p"}, execution, resources)
    assert scheduled["total_steps"] == 14000
    with pytest.raises(CapabilityError, match="frozen host resource"):
        host_recipe_overrides({"recipe_overrides": {"total_steps": 7}}, execution, resources)


def test_ka2_clock_audit_exposes_delayed_call_transition():
    recipe = Recipe(lr_floor=1, network_lr_floor=1, input_noise_std=0,
                    output_noise_warmup=0, d_guard_min_steps=0)
    dependencies = source_audit(recipe.to_dict(), {})["unexplained_clock_dependencies"]
    assert len(dependencies) == 1 and "call 800" in dependencies[0]
    assert not source_audit(recipe.replace(critic_formulation="k3p").to_dict(), {})["unexplained_clock_dependencies"]
