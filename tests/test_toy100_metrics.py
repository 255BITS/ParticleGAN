"""Real samples pass; coverage and shape shortcuts fail the frozen gate."""

import copy

import pytest
import torch

from benchmarks.toy100.metrics import (
    EVAL_N,
    MIN_HQ_MODE_MASS,
    REQUIRED_KEYS,
    evaluate_samples,
    passes,
)
from benchmarks.toy100.problems import (
    DATA_STD,
    PROBLEM_NAMES,
    evaluation_geometry,
    sample_real,
)


def _oracle(problem_name="grid100"):
    return sample_real(problem_name, EVAL_N, generator=torch.Generator().manual_seed(318))


@pytest.mark.parametrize("problem_name", PROBLEM_NAMES)
def test_targets_have_100_separated_modes_and_pass_frozen_gate(problem_name):
    centers, sigma = evaluation_geometry(problem_name)
    assert centers.shape == (100, 2)
    assert sigma == DATA_STD
    assert float(torch.pdist(centers).min()) > 0.9

    samples = _oracle(problem_name)
    result = evaluate_samples(samples, problem_name)
    assert result["passed"]
    assert passes(problem_name, {"metrics": result})
    assert result["modes"] == 100
    assert result["min_hq_count"] >= MIN_HQ_MODE_MASS * EVAL_N
    assert set(REQUIRED_KEYS) <= result.keys()


def test_sampler_is_reproducible_and_rejects_unknown_problem():
    a = sample_real("grid100", 31, generator=torch.Generator().manual_seed(91))
    b = sample_real("grid100", 31, generator=torch.Generator().manual_seed(91))
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="unknown"):
        sample_real("missing", 31)
    with pytest.raises(ValueError, match="positive"):
        sample_real("grid100", 0)


def test_finite_particle_table_can_meet_shape_gate():
    # The benchmark's learned prior has 20k atoms. An ideal table, sampled
    # with replacement like the runner's prior, must remain eligible.
    table = sample_real("grid100", EVAL_N, generator=torch.Generator().manual_seed(119))
    indices = torch.randint(EVAL_N, (EVAL_N,), generator=torch.Generator().manual_seed(120))
    result = evaluate_samples(table[indices], "grid100")
    assert result["passed"]


@pytest.mark.parametrize("problem_name", PROBLEM_NAMES)
def test_balanced_memorized_centers_fail_spread(problem_name):
    centers, _ = evaluation_geometry(problem_name)
    fake = centers.repeat_interleave(EVAL_N // 100, dim=0)
    metrics = evaluate_samples(fake, problem_name)
    assert metrics["modes"] == 100
    assert metrics["hq"] == 1.0
    assert metrics["mass_tv"] == pytest.approx(0.0)
    assert metrics["min_cov_eig_ratio"] == 0.0
    assert metrics["min_radial_median_ratio"] == 0.0
    assert not metrics["passed"]


def test_one_hit_per_missing_mode_is_not_full_coverage():
    centers, _ = evaluation_geometry("grid100")
    fake = torch.cat((centers, centers[:1].expand(EVAL_N - 100, -1)))
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["min_hq_count"] == 1
    assert metrics["modes"] == 1
    assert not metrics["passed"]


def test_nearest_bin_balance_cannot_replace_in_radius_precision():
    centers, _ = evaluation_geometry("grid100")
    fake = centers.repeat_interleave(EVAL_N // 100, dim=0) + torch.tensor([0.0, 0.25])
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["mass_tv"] == pytest.approx(0.0)
    assert metrics["hq"] == 0.0
    assert metrics["modes"] == 0
    assert not metrics["passed"]


def test_one_collapsed_mode_fails_worst_case_covariance():
    centers, sigma = evaluation_geometry("grid100")
    noise = torch.randn(EVAL_N, 2, generator=torch.Generator().manual_seed(81)) * sigma
    fake = centers.repeat_interleave(EVAL_N // 100, dim=0) + noise
    fake[: EVAL_N // 100] = centers[0]
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["modes"] == 100
    assert metrics["precision"] > 0.97
    assert metrics["min_cov_eig_ratio"] == 0.0
    assert not metrics["passed"]


def test_one_dimensional_blobs_fail_even_with_good_radial_width():
    centers, sigma = evaluation_geometry("grid100")
    offset = torch.linspace(-2.0, 2.0, EVAL_N // 100) * sigma
    fake = centers.repeat_interleave(EVAL_N // 100, dim=0)
    fake[:, 0] += offset.repeat(100)
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["modes"] == 100
    assert metrics["hq"] == 1.0
    assert metrics["min_radial_median_ratio"] > 0.65
    assert metrics["min_cov_eig_ratio"] == 0.0
    assert not metrics["passed"]


def test_single_oversized_mode_fails_max_mass_cap():
    centers, sigma = evaluation_geometry("grid100")
    counts = torch.full((100,), 200, dtype=torch.long)
    counts[0] += 210
    counts[1:71] -= 3
    fake = torch.repeat_interleave(centers, counts, dim=0)
    fake += sigma * torch.randn(EVAL_N, 2, generator=torch.Generator().manual_seed(44))
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["modes"] == 100
    assert metrics["mass_tv"] < 0.1
    assert metrics["max_mode_mass"] > 0.02
    assert not metrics["passed"]


def test_nonfinite_output_and_corrupt_report_cannot_pass():
    fake = _oracle().clone()
    fake[0, 0] = float("nan")
    metrics = evaluate_samples(fake, "grid100")
    assert metrics["precision"] > 0.97
    assert metrics["valid_n"] == EVAL_N - 1
    assert not metrics["passed"]

    clean = evaluate_samples(_oracle(), "grid100")
    clean["passed"] = False
    assert passes("grid100", clean)  # stored stamp is never authoritative
    for key, bad in (("valid_n", EVAL_N - 1), ("mass_tv", float("nan")),
                     ("precision", 1.1), ("n", 0)):
        altered = copy.deepcopy(clean)
        altered[key] = bad
        altered["passed"] = True
        assert not passes("grid100", altered)
    missing = copy.deepcopy(clean)
    missing.pop("min_cov_eig_ratio")
    missing["passed"] = True
    assert not passes("grid100", missing)
