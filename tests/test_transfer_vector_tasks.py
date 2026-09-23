from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite import vector_tasks as vectors


def test_predeclared_inventory_and_reserved_guard():
    assert len(vectors.TASKS) == 8 and len(vectors.RESERVED) == 1
    assert len({x["name"] for x in vectors.TASKS}) == 8
    assert {x["family"] for x in vectors.TASKS}.isdisjoint({x["family"] for x in vectors.RESERVED})
    assert sum(x["tier"] == "ranking" for x in vectors.TASKS) == 6
    for spec in vectors.TASKS:
        assert spec["importance_reason"] and spec["limitations"]
        assert vectors.resolve(spec)["thresholds"]
    with pytest.raises(ValueError, match="reserved"):
        vectors.resolve(vectors.RESERVED[0])
    # Structural authorization only: never sample or evaluate the reserved task.
    assert vectors.resolve(vectors.RESERVED[0], allow_reserved=True)["split"] == "reserved"
    with pytest.raises(ValueError, match="unsupported"):
        vectors.resolve(dict(vectors.TASKS[0], kind="unimplemented_dynamics"))


@pytest.mark.parametrize("spec", vectors.TASKS, ids=lambda x: x["name"])
def test_independent_target_draw_passes_predeclared_metric_bounds(spec):
    torch.set_num_threads(1)
    state = torch.get_rng_state().clone()
    samples = vectors.sample_target(spec, vectors.EVAL_SAMPLES, torch.Generator().manual_seed(993), spec["steps"])
    result = vectors.score_samples(samples, spec, spec["steps"])
    assert vectors.passes(result, spec["thresholds"]), result
    assert torch.equal(state, torch.get_rng_state())


def test_unequal_mass_is_scored_against_target_not_uniformity():
    spec = next(x for x in vectors.TASKS if x["family"] == "unequal_mass")
    uniform = dict(spec, masses=[.25]*4)
    samples = vectors.sample_target(uniform, 4096, torch.Generator().manual_seed(993), spec["steps"])
    assert vectors.score_samples(samples, spec, spec["steps"])["mass_tv"] > .25


def test_high_quality_at_centers_does_not_pass_distribution_spread():
    spec = vectors.TASKS[0]
    centers = torch.tensor(spec["means"]).repeat(2048, 1)
    result = vectors.score_samples(centers, spec, spec["steps"])
    assert result["hq"] == 1.
    assert result["component_covariance_error"] == 1.
    assert not vectors.passes(result, spec["thresholds"])


def test_partial_center_collapse_cannot_hide_behind_average_covariance():
    spec = next(x for x in vectors.TASKS if x["family"] == "unequal_width")
    samples = torch.tensor(spec["means"]).repeat_interleave(1024, dim=0)
    noise = torch.randn(1024, 2, generator=torch.Generator().manual_seed(993))
    samples[-1024:] += noise@torch.linalg.cholesky(torch.tensor(spec["covariances"][-1])).T
    result = vectors.score_samples(samples, spec, spec["steps"])
    assert result["component_covariance_error"] <= .85
    assert result["component_min_eigen_ratio"] == 0.
    previous_bounds = [x for x in spec["thresholds"] if x[0] != "component_min_eigen_ratio"]
    assert vectors.passes(result, previous_bounds)
    assert not vectors.passes(result, spec["thresholds"])


def test_overlap_rejects_unidentifiable_component_requirements():
    spec = deepcopy(next(x for x in vectors.TASKS if x["family"] == "overlapping"))
    spec["thresholds"].append(["mass_tv", "<=", .15])
    with pytest.raises(ValueError, match="overlapping"):
        vectors.resolve(spec)


def test_scale_drift_updates_target_then_stops_before_final_window():
    spec = next(x for x in vectors.TASKS if x["family"] == "changing_scale")
    assert vectors.target_scale(spec, 0) == .6
    assert vectors.target_scale(spec, spec["steps"]) == 1.6
    assert vectors.target_scale(spec, int(.8*spec["steps"])) == 1.6


def test_complete_episode_zero_feedback_matches_fixed_and_preserves_missing_failure(monkeypatch):
    monkeypatch.setattr(vectors, "EVAL_SAMPLES", 128)
    spec = dict(vectors.TASKS[0], steps=24, hidden=8, layers=1, particles=12, batch=16)
    policy = vectors.fixed_policy()
    fixed = vectors.run_episode(spec, policy, fixed=True)
    feedback = vectors.run_episode(spec, policy)
    assert "error" not in fixed and "error" not in feedback
    assert fixed["live"] == feedback["live"]
    assert fixed["ema"] == feedback["ema"]
    assert len(fixed["observations"]) == 24
    assert fixed["convergence"]["complete"]
    assert fixed["update_counts"] == {"g": 24, "d": 24}
    assert fixed["actions"] and feedback["actions"]
    assert not vectors.passes({"hq": float("nan")}, [["hq", ">=", .85]])
    assert not vectors.passes({}, [["hq", ">=", .85]])
