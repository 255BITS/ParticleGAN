from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite import image_tasks as tasks
from particlegan import ParticlePrior


def small(spec):
    spec = deepcopy(spec)
    spec.update(steps=24, batch_size=4, particles=8)
    return spec


def test_declarations_and_quality_regions_are_separated():
    assert len(tasks.TASKS) == 8
    assert len({spec["name"] for spec in tasks.TASKS}) == 8
    assert sum(spec["tier"] == "ranking" for spec in tasks.TASKS) == 4
    assert sum(spec["tier"] == "diagnostic" for spec in tasks.TASKS) == 4
    assert tasks.RESERVED["split"] == "reserved"
    assert tasks.RESERVED["family"] not in {spec["family"] for spec in tasks.TASKS}
    for spec in tasks.TASKS:
        assert spec["split"] == "development"
        assert spec["importance_reason"] and spec["limitations"]
        assert len(tasks.evaluation_steps(spec)) == 24
        centers = tasks.templates(spec).flatten(1)
        distances = torch.pdist(centers) / 8
        assert distances.min() > 2 * spec["thresholds"]["quality_rmse"]


def test_quality_gates_coverage_and_rejects_collapse_and_blends():
    spec = tasks.TASKS[1]
    centers = tasks.templates(spec)
    metrics = tasks.image_metrics(centers.repeat_interleave(8, 0), centers, spec["thresholds"])
    assert metrics["modes"] == 4 and metrics["hq"] == 1
    assert metrics["distribution_tv"] == 0
    collapsed = tasks.image_metrics(centers[:1].expand(32, -1, -1, -1), centers, spec["thresholds"])
    assert collapsed["modes"] == 1 and collapsed["hq"] == 1
    blended = tasks.image_metrics(centers.mean(0, keepdim=True).expand(32, -1, -1, -1), centers, spec["thresholds"])
    assert blended["modes"] == 0 and blended["hq"] == 0


@torch.no_grad()
def test_diagnostic_information_and_representation_limits():
    torch.manual_seed(0)
    mean_spec = next(spec for spec in tasks.TASKS if spec["architecture"] == "mean_discriminator")
    discriminator = tasks.Discriminator(mean_spec)
    scores = discriminator(tasks.templates(mean_spec))
    torch.testing.assert_close(scores, scores[0].expand_as(scores), rtol=0, atol=0)
    uniform_spec = next(spec for spec in tasks.TASKS if spec["architecture"] == "uniform_generator")
    generator = tasks.Generator(uniform_spec)
    images = generator(torch.randn(16, uniform_spec["z_dim"]))
    assert torch.equal(images, images[:, :, :1, :1].expand_as(images))
    assert tasks.image_metrics(images, tasks.templates(uniform_spec), uniform_spec["thresholds"])["hq"] == 0


def test_evaluation_is_rng_isolated_and_live_ema_are_independent():
    torch.manual_seed(0)
    spec = tasks.TASKS[0]
    generator = tasks.Generator(spec)
    prior = ParticlePrior(spec["particles"], spec["z_dim"])
    ema_generator = deepcopy(generator)
    before = torch.get_rng_state().clone()
    result = tasks.measure(generator, prior, tasks.templates(spec), spec["thresholds"])
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        generator.output.bias.add_(10)
    assert tasks.measure(ema_generator, prior, tasks.templates(spec), spec["thresholds"]) == result


def test_episode_trains_on_the_shared_runner_under_the_declared_recipe():
    spec = small(tasks.TASKS[0])
    recipe = tasks.spec_recipe(spec)
    assert (recipe.lr, recipe.d_lr_mult, recipe.betas) == (spec["lr_g"], 1., tuple(spec["adam_betas"]))
    assert (recipe.reg_arm, recipe.reg_coeff, recipe.reg_kappa) == ("b_cap", 3., 1.25)
    assert (recipe.prior_reg, recipe.ema_decay, recipe.total_steps) == (.05, .99, 24)
    groups, shapes = tasks.receipts(spec, recipe)
    assert [(g["optimizer"], g["role"]) for g in groups] == [
        ("K3PGeneratorAdam", "network"), ("K3PGeneratorAdam", "prior"), ("K3PCriticAdam", "critic")]
    assert shapes["prior"] == [8, 8] and shapes["generator_output"] == [2, 1, 8, 8]
    first, second = tasks.run_episode(spec), tasks.run_episode(spec)
    assert "error" not in first and first["route"] == "benchmarks.toy_runner"
    assert len(first["observations"]) == 24
    assert all(isinstance(point["ema"], dict) for point in first["observations"])
    assert first["convergence"]["complete"]
    strip = lambda rows: [{k: v for k, v in row.items() if k != "seconds"} for row in rows]
    assert strip(first["observations"]) == strip(second["observations"])  # deterministic


def test_no_lr_controller_can_drive_the_image_host():
    with pytest.raises(ValueError, match="no LR controller"):
        tasks.run_episode(small(tasks.TASKS[0]), fixed=False)


def test_supervised_witness_trains_without_a_critic():
    spec = small(tasks.TASKS[0])
    witness = tasks.SupervisedWitness(spec)
    result = tasks.train(spec, tasks.spec_recipe(spec).replace(prior_reg=0.), problem=witness)
    assert result["update_counts"] == {"g": 24, "d": 0}
    assert result["losses"][-1]["supervised_mse"] < result["losses"][0]["supervised_mse"]


def test_incomplete_and_transient_curves_cannot_sustain():
    spec = tasks.TASKS[0]
    expected = tasks.evaluation_steps(spec)
    requirements = [("modes", ">=", 2), ("hq", ">=", .9)]
    curve = [dict(step=step, modes=2, hq=1.) for step in expected]
    assert tasks.sustained(curve, requirements, expected_steps=expected)["confirmed_step"] == expected[4]
    assert tasks.sustained(curve[:-1], requirements, expected_steps=expected)["confirmed_step"] is None
    for point in curve[:-4]:
        point["hq"] = 0.
    assert tasks.sustained(curve, requirements, expected_steps=expected)["confirmed_step"] is None
