"""The diagnosis must preserve scientific failures and expose real gate gaps."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.toy_audit.failure_diagnosis import (
    check_receipt, image_attribution, noisy_gaussian_fit_witness, trajectory,
    validate_coverage, vector_components,
)


def test_terminal_pass_does_not_certify_a_short_or_broken_hold():
    gates = [("hq", ">=", .9)]
    rows = [{"step": 25 * (i + 1), "hq": hq}
            for i, hq in enumerate([1., .8, 1., 1., 1., 1.])]
    result = trajectory(rows, gates)
    assert result["terminal"]["hq"] == 1.
    assert result["passing_observations"] == 5
    assert result["passing_suffix"] == 4
    assert not result["terminal_stability_met"]
    rows.append({"step": 175, "hq": 1.})
    assert trajectory(rows, gates)["terminal_stability_met"]


def test_far_outlier_covariance_is_separated_from_a_perfect_core():
    core = np.tile([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], (25, 1))
    points = np.concatenate([core, [[25., 0.]]])
    spec = {"means": [[0., 0.]], "covariances": [[[.5, 0.], [0., .5]]],
            "masses": [1.], "particles": 101}
    component, = vector_components(points, spec, 1)
    assert component["core_shape"]["covariance_error"] == 0.
    assert component["full_shape"]["covariance_error"] > 8.
    assert component["far_spill_count_beyond_4sigma"] == 1
    assert component["spill_and_between_share_of_variance"] > .85
    assert component["covariance_decomposition_max_error"] < 1e-12


def test_spatial_information_and_constant_image_floor_are_real_constraints():
    templates = np.zeros((2, 1, 8, 8))
    templates[0, 0, 3:5, :] = 1.
    templates[1, 0, :, 3:5] = 1.
    thresholds = {"quality_rmse": .1, "hq_min": .9, "min_mode_fraction": .25}
    # Every constant c has MSE Var(template)+(c-Mean(template))^2.
    optimal_constant = np.full((32, 1, 8, 8), .25)
    result = image_attribution(optimal_constant, templates, thresholds)
    assert result["target_template_means"] == [.25, .25]
    assert np.allclose(result["uniform_generator_best_possible_rmse"], np.sqrt(.1875))
    assert result["quality_atoms"] == 0 and result["modes"] == 0
    assert result["nearest_assignment_is_not_a_quality_hit"]
    assert np.mean((templates[0] - templates[1]) ** 2) > .1 ** 2


def test_modified_artifact_cannot_inherit_an_original_receipt(tmp_path):
    from benchmarks.toy_audit.failure_diagnosis import sha
    result = tmp_path / "result.json"
    result.write_text('{"live": {"hq": 0.0}}')
    receipt = {"result_sha256": sha(result)}
    result.write_text('{"live": {"hq": 1.0}}')
    with pytest.raises(ValueError, match="differs from frozen receipt"):
        check_receipt(receipt, tmp_path)


def test_a_perfect_noisy_gaussian_fit_can_fail_the_clean_covariance_gate():
    witness = noisy_gaussian_fit_witness(.03, .029)
    # Independent Gaussian convolution adds variances, exactly recovering the
    # target distribution, while the model's clean component is much thinner.
    clean_sigma = witness["exact_fit_clean_sigma"]
    assert np.isclose(clean_sigma ** 2 + .029 ** 2, .03 ** 2, rtol=0., atol=1e-15)
    assert clean_sigma / .03 < .65
    assert witness["exact_gaussian_convolution_fit_possible"]
    assert not witness["population_clean_covariance_gate_would_pass"]


def test_exhaustive_report_keeps_all_failing_catalog_arms_and_boundaries():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "reports/toy_audit/catalog.json").read_text())
    report = json.loads((root / "reports/toy_audit/failure-diagnosis.json").read_text())
    validate_coverage(catalog, report)
    assert report["counts"]["frozen_reference_failures"] == 17
    assert report["counts"]["proposal_failing_arms"] == 43
    assert report["counts"]["historical_api_errors"] == 13
    assert all(row["confidence"] and row["next_action"] for row in report["diagnoses"])
    missing = deepcopy(report)
    missing["diagnoses"].pop()
    with pytest.raises(ValueError, match="coverage differs"):
        validate_coverage(catalog, missing)
    duplicate = deepcopy(report)
    duplicate["diagnoses"].append(duplicate["diagnoses"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_coverage(catalog, duplicate)
    misgan = [row for row in report["diagnoses"] if row["arm"] == "generation_and_conditional_imputation"]
    assert len(misgan) == 4
    assert all(row["no_original_binary_gate"] and not row["failed_final_gates"] for row in misgan)
