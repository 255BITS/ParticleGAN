"""Focused numerical and host-route checks for the scratch KDE rule."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from reports.toy100.bandwidth_kde_probe import capture_first_real, estimate_bandwidth
from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


def test_two_distinct_points_have_analytic_loo_optimum():
    # With n=2 and d=1, LOO log likelihood is -log(h)-distance²/(2h²).
    actual = estimate_bandwidth(np.array([[0.], [2.]]))
    assert actual["width"] == pytest.approx(2., rel=1e-6)
    assert actual["duplicate_rows"] == 0


def test_duplicate_limit_requires_every_row_to_have_a_twin():
    atoms = estimate_bandwidth(np.array([[0.], [0.], [1.], [1.]]))
    assert atoms["width"] == 0.
    assert atoms["boundary"] == "all_observations_duplicated"
    mixed = estimate_bandwidth(np.array([[0.], [0.], [1.]]))
    assert mixed["width"] > 0.
    assert mixed["duplicate_rows"] == 2


def test_real_host_capture_is_deterministic_and_takes_no_optimizer_step():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/toy100/shared_candidate.json").read_text())
    base, noise, _ = declared_recipe(config)
    jobs, profile = load_declaration()
    trajectory = next(job for job in jobs if job["spec"]["name"] == "trajectory")
    spec, card, _ = declared_spec(trajectory, profile, base)
    with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("training update")):
        first = capture_first_real(spec, card, base, noise)
        second = capture_first_real(spec, card, base, noise)
    assert first.shape == (12, 16)
    np.testing.assert_array_equal(first, second)
