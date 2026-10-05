"""Optimizer study declarations and provenance; these launch no training."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from particlegan import get_recipe
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.configuration_search import (
    _declarations, _grid, _load_spec, configuration_id, recipe_identity_fields,
)
from experiments.forge.contracts import stable_hash
from experiments.forge.techniques import recipe_field_active, validate_same_technique


ROOT = Path(__file__).resolve().parents[1]


def test_implicit_optimizer_defaults_preserve_all_archived_configuration_ids():
    for path in (ROOT / "configs/forge/configurations").glob("*.json"):
        card = json.loads(path.read_text())
        frozen = card["resolved_configuration_recipe"]
        expanded = {"optimizer_momentum": 0., "optimizer_adam_lr": None, **frozen}
        assert recipe_identity_fields(frozen) == recipe_identity_fields(expanded)
        assert configuration_id(card, resolved_recipe=expanded) == card["configuration_id"]


def test_optimizer_controls_have_explicit_ownership_and_momentum_boundary():
    assert recipe_field_owner("optimizer_family") == "technique"
    for name in ("optimizer_momentum", "optimizer_adam_lr"):
        assert recipe_field_owner(name) == "hyperparameter"
    base = get_recipe("bcap", optimizer_family="dualnorm", optimizer_momentum=.5)
    validate_same_technique(base, base.replace(optimizer_momentum=.9, lr=.03))
    with pytest.raises(ValueError, match="dualnorm_momentum"):
        validate_same_technique(base, base.replace(optimizer_momentum=0))
    assert recipe_field_active("optimizer_momentum", base)
    assert not recipe_field_active("betas", base)
    assert not recipe_field_active("prior_betas", base)
    assert not recipe_field_active("optimizer_adam_lr", base)


def test_focused_screen_is_exactly_41_whole_recipes_under_one_campaign():
    specs = sorted((ROOT / "configs/forge/searches").glob("bcap-optim-*-tier1-v1.json"))
    assert len(specs) == 9
    count, campaigns, arms = 0, [], set()
    for path in specs:
        spec = _load_spec(ROOT, path)
        campaigns.append(spec["campaign"])
        choices = _grid(spec["grid"])
        declarations = _declarations(ROOT, spec)
        assert len(choices) == len(declarations)
        count += len(declarations)
        assert spec["tuning_through_tier"] == 1
        for card, _ in declarations:
            recipe = card["resolved_configuration_recipe"]
            arms.add(recipe["optimizer_family"])
            baseline = asdict(get_recipe("bcap"))
            changed = {name for name, value in recipe.items()
                       if stable_hash(value) != stable_hash(baseline[name])}
            assert changed <= {"optimizer_family", "optimizer_momentum", "optimizer_adam_lr",
                               "lr", "d_lr_mult", "prior_lr_mult", "prior_kind", "sigma_rel",
                               "standardize"}
            assert recipe["reg_arm"] == "b_cap"
            assert recipe["reg_coeff"] == 1
            assert recipe["loss"] == "relativistic"
    assert count == 41
    assert all(campaign == campaigns[0] for campaign in campaigns)
    assert campaigns[0]["budget_seconds"] == count * campaigns[0]["candidate_budget_seconds"] == 103320
    assert arms == {"adam", "sgda", "nsgda_global", "nsgda_layer", "ada_nsgda", "dualnorm",
                    "dualnorm_D_only", "particle_rownorm_only"}


def test_hybrid_isolation_rates_preserve_native_adam_incumbent():
    for slug in ("dualnorm-d-only", "particle-rownorm-only"):
        spec = _load_spec(ROOT, f"bcap-optim-{slug}-tier1-v1")
        for card, _ in _declarations(ROOT, spec):
            recipe = card["resolved_configuration_recipe"]
            assert recipe["optimizer_adam_lr"] == .00425
            if slug == "dualnorm-d-only":
                assert recipe["prior_lr_mult"] == 2
            else:
                assert recipe["d_lr_mult"] == 1
                assert recipe["prior_lr_mult"] == 1


@pytest.mark.parametrize("family", ["dualnorm", "particle_rownorm_only"])
def test_behavioral_prior_uses_latest_g_draw_and_preserves_unsampled_rows(family):
    from experiments.forge.behavior_adapters import BehaviorComponents

    task = json.loads((ROOT / "configs/forge/tasks/ae_gan_hold.json").read_text())
    overrides = {"optimizer_family": family, "lr": .01}
    if family == "particle_rownorm_only":
        overrides["optimizer_adam_lr"] = .00425
    components = BehaviorComponents({"candidate": {"recipe_preset": "bcap", "recipe_overrides": overrides},
                                     "protocol": {"seed": 0}}, task)
    prior = components.make_prior(components.recipe)
    opt_g, _, _, _ = components.bind(generator=torch.nn.Linear(2, 2), critic=torch.nn.Linear(2, 1),
                                     priors=[prior], opt_g=None, opt_d=None)
    table_opt = next(optimizer for optimizer in opt_g.optimizers
                     if any(parameter is prior.z for group in optimizer.param_groups for parameter in group["params"]))
    prior.sample(2)  # The critic's earlier draw must not own this G update.
    _, g_rows = prior.sample(2)
    assert torch.equal(table_opt.sampled_rows_for(prior.z), torch.unique(g_rows))
    with components.noise.evaluation(0):
        prior.sample(100)
    assert torch.equal(table_opt.sampled_rows_for(prior.z), torch.unique(g_rows))
    opt_g.zero_grad(set_to_none=True)  # AE clears gradients after its G draw.
    before = prior.z.detach().clone()
    prior.z.grad = torch.ones_like(prior.z)  # An auxiliary all-table gradient.
    opt_g.step()
    unsampled = torch.ones(len(prior.z), dtype=torch.bool)
    unsampled[g_rows] = False
    assert bool(unsampled.any())
    assert torch.equal(prior.z.detach()[unsampled], before[unsampled])
    assert not torch.equal(prior.z.detach()[torch.unique(g_rows)], before[torch.unique(g_rows)])
