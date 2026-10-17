from copy import deepcopy

import torch

from benchmarks.transfer_suite import image_tasks as tasks


def zero_policy():
    return dict(weights=torch.zeros(2, 2, 5).tolist(), schedule="cosine")


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
    prior = tasks.ParticlePrior(spec["particles"], spec["z_dim"])
    ema_generator = deepcopy(generator)
    before = torch.get_rng_state().clone()
    result = tasks.measure(generator, prior, tasks.templates(spec), spec["thresholds"])
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        generator.output.bias.add_(10)
    assert tasks.measure(ema_generator, prior, tasks.templates(spec), spec["thresholds"]) == result


def test_zero_feedback_matches_fixed_numerics_and_follows_backward(monkeypatch):
    spec = deepcopy(tasks.TASKS[0])
    spec.update(steps=24, batch_size=4, particles=8)
    observed = []
    original = tasks.GradientFeedback.step
    def after_backward(self, optimizer, index, *, role):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        for parameter in parameters:
            assert optimizer.state.get(parameter, {}).get("step", 0) == index
        observed.append((index, role))
        return original(self, optimizer, index, role=role)
    monkeypatch.setattr(tasks.GradientFeedback, "step", after_backward)
    fixed = tasks.run_episode(spec, zero_policy(), fixed=True)
    adaptive = tasks.run_episode(spec, zero_policy())
    assert "error" not in fixed and "error" not in adaptive
    assert fixed["live"] == adaptive["live"]
    assert fixed["ema"] == adaptive["ema"]
    assert fixed["losses"] == adaptive["losses"]
    assert len(fixed["observations"]) == len(adaptive["observations"]) == 24
    assert [(i, role) for i in range(24) for role in ("d", "g")] == observed
    for left, right in zip(fixed["observations"], adaptive["observations"]):
        assert {k: v for k, v in left.items() if k != "seconds"} == {k: v for k, v in right.items() if k != "seconds"}


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
