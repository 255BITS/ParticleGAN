"""Public trainer and Forge integration for the simple BCAP formulation.

These tiny CPU updates test software behavior, not scientific acquisition or
qualification. They declare no new task, seed study, or training campaign.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from experiments.forge.api import task_formulation_context
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.configuration_search import _grid
from experiments.forge.contracts import read_json
from experiments.forge.techniques import validate_same_technique
from experiments.forge.views import load_tasks, load_view


ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")


def trainer_for(loss="relativistic"):
    torch.manual_seed(21)
    recipe = get_recipe("bcap", loss=loss, z_dim=2, num_particles=8,
                        batch_size=4, total_steps=7000, standardize=False,
                        prior_kind="mog", sigma_rel=.025)
    generator = nn.Sequential(nn.Linear(2, 4), nn.SiLU(), nn.Linear(4, 2))
    critic = nn.Sequential(nn.Linear(2, 4), nn.SiLU(), nn.Linear(4, 1))
    return GANTrainer(recipe, generator, critic, seed=12)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_state_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def batch():
    return torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])


@pytest.mark.parametrize("loss", LOSSES)
def test_public_updates_apply_selected_loss_with_native_adam_at_late_constant_rates(loss):
    trainer = trainer_for(loss)
    assert trainer.loss.loss == loss
    assert type(trainer.opt_g) is type(trainer.opt_d) is torch.optim.Adam
    assert trainer.opt_g.latent_damping is trainer.opt_g.direct_response is None
    assert trainer.opt_d.guard is trainer.opt_d.anchor is trainer.opt_d.ema_critic is None
    assert trainer.penalty.regularizer.arm == "b_cap"
    initial = {name: deepcopy(model.state_dict()) for name, model in
               (("generator", trainer.G), ("critic", trainer.D), ("prior", trainer.prior))}
    base_rates = deepcopy(trainer.initial_lrs)
    # Advance only the external update label, then execute actual API updates
    # past the old network-horizon cap and within the old cosine decay interval.
    trainer.completed_steps = 6000
    for index in range(3):
        result = trainer.step(batch(), collect_stats=True)
        assert trainer.completed_steps == 6001 + index
        assert trainer.opt_d.record.observed_steps == index + 1
        assert all(torch.isfinite(result[name]) for name in ("loss_g", "loss_d"))
        assert trainer.penalty.regularizer.coeff == trainer.recipe.reg_coeff == 1.
        assert trainer._noisy_D.std == trainer.last_output_sigma == 0.
        assert [[group["lr"] for group in optimizer.param_groups]
                for optimizer in (trainer.opt_g, trainer.opt_d)] == base_rates
        assert all(group["betas"] == (0., .999)
                   for optimizer in (trainer.opt_g, trainer.opt_d)
                   for group in optimizer.param_groups)
    for name, model in (("generator", trainer.G), ("critic", trainer.D), ("prior", trainer.prior)):
        assert any(not torch.equal(tensor, initial[name][key])
                   for key, tensor in model.state_dict().items()), name


@pytest.mark.parametrize("loss", LOSSES)
def test_loss_checkpoint_roundtrip_continues_exact_public_updates(loss, tmp_path):
    trainer = trainer_for(loss)
    trainer.step(batch())
    checkpoint_path = tmp_path / "trainer.pt"
    torch.save(trainer.state_dict(), checkpoint_path)
    trainer.step(batch())
    trainer.step(batch())
    expected = trainer.state_dict()

    restored = trainer_for(loss)
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    restored.step(batch())
    restored.step(batch())
    assert restored.loss.loss == loss
    assert_state_equal(expected, restored.state_dict())


@pytest.mark.parametrize("saved_loss,live_loss", [("hinge", "relativistic"),
                                               ("relativistic", "hinge"),
                                               ("wasserstein", "least_squares")])
def test_loss_checkpoint_mismatch_rejects_before_parameters_optimizers_or_rng_mutate(saved_loss, live_loss):
    source = trainer_for(saved_loss)
    source.step(batch())
    checkpoint = source.state_dict()
    live = trainer_for(live_loss)
    live.step(batch())
    before = live.state_dict()
    with pytest.raises(ValueError, match="recipe does not match"):
        live.load_state_dict(checkpoint)
    assert_state_equal(before, live.state_dict())


def test_forge_records_loss_as_a_technique_and_rejects_numeric_search_relabeling():
    assert recipe_field_owner("loss") == "technique"
    with pytest.raises(ValueError, match="forbidden Recipe grid field.*loss"):
        _grid({"loss": ["relativistic", "hinge"]})
    base = get_recipe("bcap")
    with pytest.raises(ValueError, match="Recipe.loss"):
        validate_same_technique(asdict(base), asdict(base.replace(loss="hinge")))
    # Positive rate endpoints stay within the same technique.
    validate_same_technique(asdict(base), asdict(base.replace(lr=.0010625)))


@pytest.mark.parametrize("loss", LOSSES)
def test_every_initial_tier1_host_binds_the_loss_and_its_own_prior_without_extra_mechanisms(loss):
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "discriminator_stability")
    protocol = read_json(ROOT / "configs/forge/protocols/screening.json")
    candidate = {"recipe_preset": "bcap", "recipe_overrides": {"loss": loss},
                 "requires_capabilities": ["learned_locations", "named_rng"]}
    tier1 = [row for row in view["assignments"] if row["qualification_tier"] == 1]
    assert {tasks[row["task"]]["execution"]["prior"]["kind"] for row in tier1} == {"mog", "particle_cloud"}
    for assignment in tier1:
        task = tasks[assignment["task"]]
        context = task_formulation_context(candidate, task, protocol, root=ROOT)
        recipe = context.recipe
        assert recipe.loss == context.recipe.make_loss().loss == loss
        assert recipe.optimizer_family == "adam"
        assert recipe.reg_arm == "b_cap"
        assert recipe.lr_floor == recipe.resolved_network_lr_floor == 1.
        assert recipe.beta2_end is recipe.reg_coeff_end is recipe.continuous_policy is None
        assert recipe.d_guard_ratio == recipe.reg_anchor_weight == recipe.latent_damping_max_rate == 0.
        assert recipe.input_noise_std == recipe.output_noise_std == recipe.prior_reg == recipe.ema_decay == 0.
        assert recipe.direct_particle_gain is False
        assert context.prior_config["kind"] == task["execution"]["prior"]["kind"]
        assert context.prior_config["sigma"] == task["execution"]["prior"]["sigma"]
        loss_binding = context.receipt()["field_ownership"]["recipe_fields"]["loss"]
        assert loss_binding["owner"] == "technique"
        assert loss_binding["value"] == loss
