"""Discriminating controls for the newly declared audit evaluators."""
import numpy as np
import pytest

from benchmarks.toy_audit.definition_quality import (
    five_word_metrics, gaussian_metrics, landing_metrics, paired_edit_metrics,
    ring_metrics, terminal_window, two_pole_metrics, useful_code_metrics,
    word_probabilities, word_templates,
)


def test_travel_and_pole_coverage_do_not_imply_the_target_width():
    from benchmarks.locked_shared.two_pole import real_batch
    from benchmarks.legacy.locked_shared import LOCKED_SHARED
    actual_target = real_batch(LOCKED_SHARED.n_particles).numpy()
    assert two_pole_metrics(actual_target)["passed"]
    assert two_pole_metrics(actual_target[::-1])["passed"]
    assert not two_pole_metrics(np.full(12, .6))["passed"]
    centers = two_pole_metrics(np.repeat([-1., 1.], 6))
    assert centers["support_fraction"] == 1 and centers["mass_tv"] == 0
    assert not centers["passed"]
    assert not two_pole_metrics(real_batch(256).numpy())["passed"]


def test_gaussian_rejects_an_isotropic_law_with_the_correct_moments():
    theta = np.arange(4096) * 2 * np.pi / 4096
    circle = 1 + .2 * np.sqrt(2) * np.stack([np.cos(theta), np.sin(theta)], 1)
    result = gaussian_metrics(circle)
    np.testing.assert_allclose(result["covariance_eigenvalues"], [1., 1.], atol=1e-12)
    assert result["radial_ks"] > .5 and not result["passed"]
    assert gaussian_metrics(1 + .2 * np.random.default_rng(713).standard_normal((4096, 2)))["passed"]


def test_word_gate_requires_confidence_mass_and_paired_reconstruction():
    templates = word_templates()
    words = np.repeat(templates, 40, axis=0)
    assert five_word_metrics(words, templates)["passed"]
    assert not five_word_metrics(.04 * words + .96 / 28, templates)["passed"]
    assert not five_word_metrics(words, np.roll(templates, 1, axis=0))["passed"]
    assert not five_word_metrics(np.repeat(templates[:1], 200, axis=0), templates)["passed"]
    logits = np.where(words == 1, 8., -8.)
    assert five_word_metrics(word_probabilities(logits), templates)["passed"]


def test_word_padding_cannot_be_hidden_by_a_lossy_string_decoder():
    templates = word_templates()
    wrong_padding = np.repeat(templates, 40, axis=0)
    # The original display strips both '_' and trailing spaces. The complete
    # categorical record must nevertheless preserve the declared '_' token.
    wrong_padding[:, :, -1] = 0
    wrong_padding[:, 27, -1] = 1
    result = five_word_metrics(wrong_padding, templates)
    assert result["quality_fraction"] == 0 and not result["passed"]


def test_paired_gate_rejects_an_identical_output_marginal_with_wrong_pairs():
    target = np.array([[-1., -2.], [1., 2.]])
    np.testing.assert_array_equal(np.sort(-target, axis=0), np.sort(target, axis=0))
    assert paired_edit_metrics(target, target, np.zeros_like(target))["passed"]
    wrong = paired_edit_metrics(-target, target, np.zeros_like(target))
    assert wrong["relative_mse"] == 4 and not wrong["passed"]
    with pytest.raises(ValueError, match="nonzero edit"):
        paired_edit_metrics(target, target, target)


def test_reflecting_generated_state_and_action_can_preserve_the_entire_joint_law():
    state = np.array([[1., 0.], [-1., 0.]])
    action = -state
    joint = np.concatenate([state, action], 1)
    np.testing.assert_array_equal(joint, (-joint)[::-1])
    # Matching generated (state,action) does not fix the action relative to
    # the actual caller state when the generated state can also be reflected.
    assert not paired_edit_metrics(-action, action, np.zeros_like(action))["passed"]


def test_every_judge_must_find_the_code_helpful():
    assert useful_code_metrics([.7, .8], [.8, .9])["passed"]
    assert not useful_code_metrics([.8, .9], [.7, .8])["passed"]
    assert not useful_code_metrics([.7, .8], [.8, .7])["passed"]
    assert not useful_code_metrics([.7, .8], [.7, .8])["passed"]


def test_fast_success_on_a_selected_subset_does_not_hide_timeouts():
    good = np.ones(100, dtype=bool)
    assert landing_metrics(good, ~good, np.full(100, 20.))["passed"]
    selected = np.zeros(100, dtype=bool)
    selected[:10] = True
    result = landing_metrics(selected, np.zeros(100, dtype=bool), np.where(selected, 10., 48.))
    assert result["timeout_rate"] == .9 and result["restricted_mean_steps"] > 40
    assert not result["passed"]
    with pytest.raises(ValueError, match="disjoint"):
        landing_metrics(good, good, np.full(100, 20.))


def test_ring_needs_width_and_the_current_target_after_a_shift():
    angles = np.arange(8) * np.pi / 4
    centers = 3 * np.stack([np.cos(angles), np.sin(angles)], 1)
    ring = np.repeat(centers, 512, axis=0) + .07 * np.random.default_rng(71).standard_normal((4096, 2))
    assert ring_metrics(ring)["passed"]
    assert not ring_metrics(np.repeat(centers, 512, axis=0))["passed"]
    assert not ring_metrics(ring, shift=(1., 0.))["passed"]
    assert ring_metrics(ring + [1, 0], shift=(1., 0.))["passed"]


def test_transient_or_best_checkpoint_pass_is_not_sustained_convergence():
    assert terminal_window([False] * 4 + [True] * 5)["passed"]
    assert not terminal_window([True] * 20 + [False])["passed"]
    assert not terminal_window([False] * 20 + [True] * 4)["passed"]
