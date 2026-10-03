"""Construction-only checks: reporting must not change the public update law."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.techniques import recipe_field_active
from experiments.forge.views import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def components(host, optimizer_family="formulation"):
    overrides = dict(lr=.003, d_lr_mult=1.5, prior_lr_mult=.125,
                     prior_betas=(0., .7), input_noise_std=0., output_noise_std=0.,
                     network_lr_horizon_cap=None)
    if optimizer_family == "adam":
        overrides.update(optimizer_family="adam", reg_arm="a_r1r2", reg_anchor_weight=0.,
                         latent_damping_max_rate=0., direct_particle_gain=False,
                         d_guard_ratio=0., beta2_end=.99)
    return BehaviorComponents({"candidate": {"recipe_overrides": overrides},
                               "protocol": {"seed": 0}}, load_tasks(ROOT)[host])


@pytest.mark.parametrize("optimizer_family", ["formulation", "adam"])
def test_direct_coordinate_metadata_preserves_generator_side_optimizer_law(optimizer_family):
    bound = components("two_pole", optimizer_family)
    particles = torch.nn.Parameter(torch.zeros(8, 1))
    generator, critic, _, _ = bound.bind(generator=None, critic=torch.nn.Linear(1, 1),
                                        opt_g=None, opt_d=None, direct_particles=[particles])
    group = generator.param_groups[0]
    # Latent-prior controls are deliberately different, but do not bind this
    # existing public direct-output group. No optimizer step is performed.
    assert group["lr"] == bound.recipe.lr != bound.recipe.lr * bound.recipe.prior_lr_mult
    assert group["betas"] == bound.recipe.betas != bound.recipe.prior_betas
    before = (deepcopy(generator.state_dict()), deepcopy(critic.state_dict()),
              bound.context.streams.audit())
    receipt = bound.receipt()
    row = receipt["optimizer_group_bindings"][0]
    assert row["representation"] == "direct_sample_coordinates"
    assert row["counter_role"] == row["lr_schedule"] == "prior"
    assert row["base_lr"] == row["current_lr"] == bound.recipe.lr
    assert tuple(row["base_betas"]) == tuple(row["current_group_betas"]) == bound.recipe.betas
    assert not row["prior_lr_mult_consumed"] and not row["prior_betas_consumed"]
    response = row["direct_particle_response"]
    if optimizer_family == "formulation":
        assert response == {"installed": True, "step_betas": list(bound.recipe.direct_particle_betas),
                            "gain_enabled": True, "lr_gain_range": [1., 2.]}
        assert row["group_betas_overridden_during_step"]
    else:
        assert not response["installed"] and response["step_betas"] is None
        assert row["beta2_schedule_declared"] and not row["group_betas_overridden_during_step"]
    fields = receipt["field_ownership"]["recipe_fields"]
    for name in ("prior_lr_mult", "prior_betas"):
        assert fields[name]["value"] is None
        assert fields[name]["status"] == "not_applicable"
        value = getattr(bound.recipe, name)
        assert fields[name]["reference_recipe_value"] == (list(value) if isinstance(value, tuple) else value)
    prior = receipt["field_ownership"]["task_contract"]["prior"]
    assert prior["code_path"] is None and prior["declared_code_path"] == "ParticlePrior"
    assert prior["control_binding"]["representation"] == "direct_sample_coordinates"
    assert bound.context.streams.audit() == before[2]
    assert group["lr"] == before[0][0]["param_groups"][0]["lr"]
    assert group["betas"] == before[0][0]["param_groups"][0]["betas"]
    assert critic.record.observed_steps == 0 and all(not opt.state for opt in generator.optimizers)


def test_latent_table_controls_remain_effective_in_construction_receipts():
    bound = components("ae_gan_hold")
    prior = bound.make_prior(bound.recipe)
    generator, _, _, _ = bound.bind(generator=torch.nn.Linear(2, 2), critic=torch.nn.Linear(2, 1),
                                    opt_g=None, opt_d=None, priors=[prior])
    receipt = bound.receipt()
    row = next(row for row in receipt["optimizer_group_bindings"]
               if row["representation"] == "latent_prior_locations")
    assert row["base_lr"] == bound.recipe.lr * bound.recipe.prior_lr_mult
    assert tuple(row["base_betas"]) == bound.recipe.prior_betas
    assert row["prior_lr_mult_consumed"] and row["prior_betas_consumed"]
    assert not row["direct_particle_response"]["installed"]
    fields = receipt["field_ownership"]["recipe_fields"]
    assert fields["prior_lr_mult"]["value"] == bound.recipe.prior_lr_mult
    assert fields["prior_betas"]["value"] == list(bound.recipe.prior_betas)
    assert fields["prior_lr_mult"]["status"] == fields["prior_betas"]["status"] == "effective"
    assert all(not opt.state for opt in generator.optimizers)


def test_nonsampled_parameter_host_does_not_claim_latent_prior_controls():
    bound = components("unused_token_hold")
    # Generic boundary reporting does not need a constructed model or optimizer.
    receipt = bound.context.receipt()["field_ownership"]
    assert receipt["task_contract"]["prior"]["control_binding"]["representation"] == "not_sampled"
    assert receipt["task_contract"]["prior"]["code_path"] is None
    assert all(receipt["recipe_fields"][name]["status"] == "not_applicable"
               for name in ("prior_lr_mult", "prior_betas"))


@pytest.mark.parametrize("field", ["prior_lr_mult", "prior_betas"])
def test_search_axes_share_direct_coordinate_applicability(field):
    tasks = load_tasks(ROOT)
    recipe = components("two_pole").recipe
    assert not recipe_field_active(field, recipe, task=tasks["two_pole"])
    assert not recipe_field_active(field, recipe, task=tasks["unused_token_hold"])
    assert recipe_field_active(field, recipe, task=tasks["ae_gan_hold"])
    assert recipe_field_active(field, recipe, task=tasks["five_word_joint_acquisition"])


@pytest.mark.parametrize("field,trial", [("prior_lr_mult", .5), ("prior_betas", [0., .8])])
def test_two_pole_only_search_rejects_unused_prior_axes(monkeypatch, field, trial):
    from experiments.forge import views
    from experiments.forge.configuration_search import _validate_tuning_axes

    recipe = components("two_pole").recipe
    monkeypatch.setattr(views, "load_view", lambda *_: {"assignments": [
        {"task": "two_pole", "qualification_tier": 1}]})
    trial_recipe = recipe.replace(**{field: trial}).to_dict()
    with pytest.raises(ValueError, match=field + " is inactive or task-owned"):
        _validate_tuning_axes(ROOT, {"view": "discriminator_stability", "tuning_through_tier": 1},
                              recipe.to_dict(), [({"resolved_configuration_recipe": trial_recipe}, {field: trial})])


@pytest.mark.parametrize("optimizer_family", ["formulation", "adam"])
def test_direct_coordinate_steps_retain_public_adam_response(optimizer_family):
    """Compare synthetic API updates to Adam; this is not a scientific run."""
    bound = components("two_pole", optimizer_family)
    particles = torch.nn.Parameter(torch.zeros(8, 1))
    expected = torch.nn.Parameter(torch.zeros_like(particles))
    generator, _, _, _ = bound.bind(generator=None, critic=torch.nn.Linear(1, 1),
                                   opt_g=None, opt_d=None, direct_particles=[particles])
    public = generator.optimizers[0]
    betas = bound.recipe.direct_particle_betas if optimizer_family == "formulation" else bound.recipe.betas
    reference = torch.optim.Adam([expected], lr=bound.recipe.lr, betas=betas, eps=bound.recipe.eps)
    gradient = torch.arange(1., 9.).reshape(8, 1)
    for step in range(2):
        bound.schedule_optimizer(generator, step)
        particles.grad = gradient.clone()
        expected.grad = gradient.clone()
        if optimizer_family == "formulation":
            # Identical centered gradients produce the public gain 1, then 2.
            reference.param_groups[0]["lr"] = bound.recipe.lr * (1 if step == 0 else 2)
        else:
            reference.param_groups[0]["betas"] = public.param_groups[0]["betas"]
        public.step()
        reference.step()
        assert torch.allclose(particles, expected, atol=1e-9, rtol=1e-6)
        assert torch.equal(public.state[particles]["exp_avg_sq"], reference.state[expected]["exp_avg_sq"])
        assert public.param_groups[0]["lr"] == bound.recipe.lr
    if optimizer_family == "formulation":
        assert public.direct_response.last_gain == pytest.approx(2.)
        assert public.param_groups[0]["betas"] == bound.recipe.betas
