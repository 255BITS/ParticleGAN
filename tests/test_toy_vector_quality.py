"""Keep initializer confounds visible and calibrate component shape gates."""
from copy import deepcopy
import math

import pytest
import torch

from benchmarks.toy_audit.vector_quality import comparison_factors, finite_cloud_witness, projection_ks, score_control
from benchmarks.transfer_suite import vector_tasks


def test_spec_equality_cannot_hide_patched_prior_initialization():
    spec = dict(kind="gaussian_mixture", means=[[-1, 0], [1, 0]], covariances=[], masses=[.5, .5])
    left, right = deepcopy(spec), deepcopy(spec)
    left["research_discriminator"], right["fourier"] = {"kernel_scales": [.1, .25]}, 2
    result = comparison_factors([{"spec": left}, {"spec": right}], [{"init_std": .5}, {"init_std": 4.}])
    assert result["law_matched"]
    assert result["prior_initialization_changed"]
    assert not result["critic_only_identified"]
    assert not result["individual_critic_mechanism_identified"]


def test_unknown_initializer_cannot_certify_an_isolated_intervention():
    result = comparison_factors([{"spec": {"fourier": 0}}, {"spec": {"fourier": 2}}], [{}, {}])
    assert result["initializer_unknown"] and not result["critic_only_identified"]
    with pytest.raises(ValueError, match="two arms"):
        comparison_factors([], [])


def test_high_quality_balanced_centers_do_not_imply_gaussian_shape():
    spec = vector_tasks.resolve(next(deepcopy(row) for row in vector_tasks.TASKS if row["name"] == "vector_two_broad"))
    result = score_control(torch.tensor(spec["means"]).repeat_interleave(2048, 0), spec)
    assert result["metrics"]["hq"] == 1 and result["metrics"]["mass_tv"] == 0
    assert not result["passed"]
    assert any(row["metric"] == "component_min_eigen_ratio" for row in result["failed_bounds"])


def test_projection_gate_rejects_zero_width_when_original_w1_accepts_it():
    # The original overlapping-ring stress gates only SW1 <= .18. Its zero
    # width witness is well inside that bar, despite not being a smooth law.
    means = [[3 * math.cos(theta * math.pi / 4), 3 * math.sin(theta * math.pi / 4)] for theta in range(8)]
    spec = dict(kind="gaussian_mixture", identifiable=False, means=means,
                covariances=[[[.36, 0], [0, .36]]] * 8, masses=[.125] * 8,
                steps=1200, thresholds=[["sw1_normalized", "<=", .18]])
    points = torch.tensor(means).repeat_interleave(512, 0)
    result = score_control(points, spec)
    assert result["passed"] and not result["revised_passed"]
    assert result["projection_ks"] > .06
    target = vector_tasks.sample_target(spec, 4096, torch.Generator().manual_seed(1931), 1200)
    assert score_control(target, spec)["revised_passed"]


def test_finite_population_can_pass_without_a_continuous_sampling_oracle():
    spec = vector_tasks.resolve(next(deepcopy(row) for row in vector_tasks.TASKS if row["name"] == "vector_two_broad"))
    assert score_control(finite_cloud_witness(spec), spec)["revised_passed"]


def test_analytic_projection_metric_is_invariant_to_shared_units():
    spec = vector_tasks.resolve(next(deepcopy(row) for row in vector_tasks.TASKS if row["name"] == "vector_two_broad"))
    points = vector_tasks.sample_target(spec, 1024, torch.Generator().manual_seed(1931), spec["steps"])
    scaled = deepcopy(spec)
    scaled["means"] = [[24 * value for value in row] for row in spec["means"]]
    scaled["covariances"] = [[[24 ** 2 * value for value in row] for row in matrix] for matrix in spec["covariances"]]
    assert projection_ks(points, spec, spec["steps"]) == pytest.approx(
        projection_ks(points.double() * 24, scaled, scaled["steps"]), abs=1e-12)
