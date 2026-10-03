from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite import stress_tasks
from benchmarks.smart_descent.study import policy


def test_reserved_dynamics_cannot_enter_development_runner():
    reserved = deepcopy(stress_tasks.RESERVED_TASKS[0])
    with pytest.raises(ValueError, match="reserved"):
        stress_tasks.run_episode(reserved, None)
    assert reserved["family"] == "update_cadence"
    assert reserved["d_every"] == 2 and reserved["g_every"] == 1


def test_diagnostic_labels_do_not_become_component_quality_requirements():
    overlap = next(t for t in stress_tasks.TASKS if t["name"] == "stress_overlapping_data")
    assert not overlap["identifiable"]
    assert {metric for metric, _, _ in overlap["thresholds"]} == {"sw1_normalized"}
    assert overlap["tier"] == "diagnostic"
    weak = next(t for t in stress_tasks.TASKS if t["name"] == "stress_weak_critic")
    assert weak["tier"] == "diagnostic"


def test_tiny_episode_reports_live_curve_and_separate_ema_without_changing_spec():
    torch.set_num_threads(1)
    spec = deepcopy(stress_tasks.TASKS[0])
    spec.update(steps=24, hidden=8, layers=1, d_hidden=8, d_layers=1,
                particles=32, batch=8)
    before = deepcopy(spec)
    card = policy(torch.zeros(2, 2, 5, dtype=torch.float64))
    result = stress_tasks.run_episode(spec, card, fixed=True)
    assert "error" not in result, result.get("error")
    assert spec == before
    assert [point["step"] for point in result["observations"]] == list(range(1, 25))
    assert result["convergence"]["complete"]
    assert result["convergence"]["minimum_stable_checks"] == 5
    assert result["live"] is not result["ema"]
    assert all(torch.isfinite(torch.tensor(point["sw1_normalized"])) for point in result["observations"])
    required = {key for key, _, _ in spec["thresholds"]}
    assert required <= result["live"].keys() and required <= result["ema"].keys()
    assert all(required <= point.keys() for point in result["observations"])


def test_memorized_mode_centers_fail_the_stress_distribution_gate():
    from benchmarks.transfer_suite import vector_tasks as vector
    spec = stress_tasks.TASKS[0]
    fake = torch.tensor(spec["means"], dtype=torch.float32).repeat(128, 1)
    metrics = vector.score_samples(fake, spec, spec["steps"])
    assert metrics["hq"] == 1. and metrics["mass_tv"] == 0.
    assert metrics["component_covariance_error"] == pytest.approx(1.)
    assert not vector.passes(metrics, spec["thresholds"])


def test_reference_distribution_satisfies_the_gate_without_claiming_training_solvability():
    from benchmarks.transfer_suite import vector_tasks as vector
    spec = stress_tasks.TASKS[0]
    samples = vector.sample_target(spec, 4096, torch.Generator().manual_seed(0), spec["steps"])
    metrics = vector.score_samples(samples, spec, spec["steps"])
    assert vector.passes(metrics, spec["thresholds"])


def test_partial_component_collapse_cannot_hide_behind_average_covariance():
    from benchmarks.transfer_suite import vector_tasks as vector
    spec = stress_tasks.TASKS[0]
    corners = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]) * (.12 * 2 ** .5)
    components = []
    for index, mean in enumerate(spec["means"]):
        center = torch.tensor(mean, dtype=torch.float32)
        # Six collapsed components and two with exactly the target covariance.
        offsets = torch.zeros(128, 2) if index < 6 else corners.repeat(32, 1)
        components.append(center + offsets)
    metrics = vector.score_samples(torch.cat(components), spec, spec["steps"])
    old_bounds = [bound for bound in spec["thresholds"] if bound[0] != "component_min_eigen_ratio"]
    assert metrics["hq"] == 1. and metrics["mass_tv"] == 0.
    assert metrics["component_covariance_error"] == pytest.approx(.75, abs=1e-5)
    assert vector.passes(metrics, old_bounds)
    assert metrics["component_min_eigen_ratio"] == pytest.approx(0.)
    assert not vector.passes(metrics, spec["thresholds"])
