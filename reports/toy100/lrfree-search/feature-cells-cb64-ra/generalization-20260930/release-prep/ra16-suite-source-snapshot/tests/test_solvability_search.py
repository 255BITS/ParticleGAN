"""Research plans must not silently change the question being measured."""
from copy import deepcopy

import pytest

from benchmarks.transfer_suite.solvability_search import jobs
from benchmarks.transfer_suite.suite import manifest


@pytest.fixture
def declared():
    return {s["name"]: s for s in manifest()["tasks"]}


def plan(task, **options):
    return dict(tasks=[task], candidates=[dict(name="candidate", **options)])


@pytest.mark.parametrize("overrides", [dict(loss_type="hinge"), dict(noise_std=0),
                                       dict(thresholds={}), dict(pattern="blobs4")])
def test_reject_ignored_image_options_and_changed_targets(declared, overrides):
    with pytest.raises(ValueError, match="unsupported training option"):
        jobs(plan("img_stripes2", overrides=overrides), declared)


def test_reject_per_task_threshold_override(declared):
    with pytest.raises(ValueError, match="unsupported training option"):
        jobs(plan("vector_two_broad", task_overrides={"vector_two_broad": {"thresholds": []}}), declared)


def test_reject_silent_image_architecture_fallback(declared):
    with pytest.raises(ValueError, match="unsupported image architecture"):
        jobs(plan("img_stripes2", overrides=dict(architecture="residuall")), declared)


def test_preserves_task_perturbations_and_original_specs(declared):
    original = deepcopy(declared)
    card = dict(tasks=["stress_fast_critic", "stress_small_batch"],
                candidates=[dict(name="prior30", overrides=dict(prior_lr_mult=30), steps_multiplier=2)])
    for _, initial, effective, policy in jobs(card, declared):
        assert effective == initial | dict(prior_lr_mult=30, steps=2 * initial["steps"])
        assert policy["schedule"] == "cosine"
    assert declared == original


def test_reject_legacy_translation_and_unknown_schedule(declared):
    with pytest.raises(ValueError, match="explicit config translation"):
        jobs(plan("mode_hold", overrides=dict(reg_coeff=10)), declared)
    with pytest.raises(ValueError, match="unsupported schedule"):
        jobs(plan("vector_two_broad", schedule="typo"), declared)


def test_prevents_artifact_overwrite(declared):
    card = plan("vector_two_broad")
    card["candidates"] *= 2
    with pytest.raises(ValueError, match="unique"):
        jobs(card, declared)
    card = plan("vector_two_broad")
    card["tasks"] *= 2
    with pytest.raises(ValueError, match="unique"):
        jobs(card, declared)


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), True, .001])
def test_invalid_budget_rejected_before_training(declared, multiplier):
    with pytest.raises(ValueError):
        jobs(plan("vector_two_broad", steps_multiplier=multiplier), declared)
