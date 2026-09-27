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
    # Protocol v1 (no eigenvalue bound) accepts the collapse; v2 rejects it.
    v1_bounds = [x for x in vectors.SEPARATED_BOUNDS if x[0] != "component_min_eigen_ratio"]
    assert vectors.passes(result, v1_bounds)
    assert not vectors.passes(result, vectors.SEPARATED_BOUNDS)
    # The v4 core/spill rule rejects it too: collapsed cores have zero spread.
    assert result["component_core_min_eigen_ratio"] == 0.
    assert not vectors.passes(result, spec["thresholds"])


def test_protocol_v5_core_spill_declarations():
    by_family = {x["family"]: x["thresholds"] for x in vectors.TASKS}
    core = vectors.CORE_SPILL_BOUNDS
    assert by_family["anisotropic"] == core and by_family["unequal_width"] == core
    assert by_family["unequal_mass"] == core + [["min_mass_ratio", ">=", .25]]
    for family in ("separated_broad", "narrow_resolution", "changing_scale"):
        assert by_family[family] == vectors.SEPARATED_BOUNDS
    for family in ("overlapping", "curved_continuous"):
        assert by_family[family] == vectors.DISTRIBUTION_BOUNDS
    # v5 keeps every v4 bound and value; only the three shape/spill statistics become resolved_* aggregates.
    renamed = dict(zip(("component_core_covariance_error", "component_core_min_eigen_ratio", "max_component_spill"),
                       vectors.RESOLVED_SHAPE_METRICS))
    assert core == [[renamed.get(k, k), op, b] for k, op, b in vectors.CORE_SPILL_BOUNDS_V4]
    # Oracle-derived floor (reports/transfer_suite/silu_rare_collapse/oracle.py); only the 2% component is exempt.
    assert vectors.PARTICLE_FLOOR == 32
    resolved = {x["family"]: vectors.component_resolved(x) for x in vectors.TASKS if x["kind"] == "gaussian_mixture"}
    assert resolved.pop("unequal_mass") == [True, True, True, False]
    assert all(all(flags) for flags in resolved.values())


def _unequal_mass_with_rare(rare_points):
    """A true unequal_mass draw whose rare (2%) component is replaced by rare_points."""
    spec = next(x for x in vectors.TASKS if x["family"] == "unequal_mass")
    samples = vectors.sample_target(spec, 4096, torch.Generator().manual_seed(993), spec["steps"])
    means = torch.tensor(spec["means"])
    rare = torch.cdist(samples, means).argmin(1) == 3
    samples[rare] = rare_points(int(rare.sum()), means[3])
    return spec, vectors.score_samples(samples, spec, spec["steps"])


def test_particle_floor_exempts_under_resolved_component_shape_but_not_its_mass():
    # The 2% component (~5 particles) collapsed to a single point: v4 fails it on core eig, v5 exempts its shape.
    spec, result = _unequal_mass_with_rare(lambda n, mean: mean.repeat(n, 1))
    assert result["component_resolved"] == [True, True, True, False]
    assert result["component_core_eigen_ratios"][3] == 0. and result["component_core_min_eigen_ratio"] == 0.
    assert result["resolved_core_min_eigen_ratio"] == min(result["component_core_eigen_ratios"][:3]) > .15
    v4 = spec["thresholds"][:3] + vectors.CORE_SPILL_BOUNDS_V4[3:] + [["min_mass_ratio", ">=", .25]]
    assert not vectors.passes(result, v4)
    assert vectors.passes(result, spec["thresholds"])
    # Its mass is still gated: dropping it (points moved onto the big component) fails min_mass_ratio.
    spec, dropped = _unequal_mass_with_rare(lambda n, mean: torch.tensor(spec["means"][0]).repeat(n, 1))
    assert dropped["component_mass"][3] == 0. and dropped["resolved_max_component_spill"] <= .05
    assert not vectors.passes(dropped, spec["thresholds"])
    assert [k for k, op, b in spec["thresholds"] if not vectors.passes(dropped, [[k, op, b]])] == ["min_mass_ratio"]


def test_particle_floor_still_gates_resolved_components():
    # The 13% component (~33 particles) is resolved: collapsing it fails v5.
    spec = next(x for x in vectors.TASKS if x["family"] == "unequal_mass")
    samples = vectors.sample_target(spec, 4096, torch.Generator().manual_seed(993), spec["steps"])
    means = torch.tensor(spec["means"])
    member = torch.cdist(samples, means).argmin(1) == 2
    samples[member] = means[2]
    result = vectors.score_samples(samples, spec, spec["steps"])
    assert result["component_resolved"][2]
    assert result["resolved_core_min_eigen_ratio"] == 0.
    assert not vectors.passes(result, spec["thresholds"])


@pytest.mark.parametrize("spec", [x for x in vectors.TASKS if x["kind"] != "gaussian_mixture" or not x["identifiable"]
                                  or not ({k for k, _, _ in x["thresholds"]} & set(vectors.RESOLVED_SHAPE_METRICS))
                                  or all(vectors.component_resolved(x))], ids=lambda x: x["name"])
def test_particle_floor_leaves_other_tasks_unchanged(spec):
    """Non-identifiable and whole-component tasks gain no gate; fully resolved core/spill tasks gate the same values."""
    samples = vectors.sample_target(spec, 4096, torch.Generator().manual_seed(993), spec["steps"])
    samples[:200] += 1.  # Some spill so the comparison is not only at exact-match values.
    result = vectors.score_samples(samples, spec, spec["steps"])
    if spec["kind"] != "gaussian_mixture" or not spec["identifiable"]:
        assert not set(vectors.RESOLVED_SHAPE_METRICS) & result.keys()
        return
    assert result["resolved_core_covariance_error"] == result["component_core_covariance_error"]
    assert result["resolved_core_min_eigen_ratio"] == result["component_core_min_eigen_ratio"]
    assert result["resolved_max_component_spill"] == result["max_component_spill"]


def test_particle_floor_resolve_guards():
    anisotropic = next(x for x in vectors.TASKS if x["family"] == "anisotropic")
    with pytest.raises(ValueError, match="particle floor"):
        vectors.resolve(dict(anisotropic, particles=64))  # 21 particles per component: nothing resolved.
    with pytest.raises(ValueError, match="min_mass_ratio"):
        vectors.resolve(dict(anisotropic, masses=[.8, .1, .1]))  # 26-particle components without a mass floor.
    unequal_mass = next(x for x in vectors.TASKS if x["family"] == "unequal_mass")
    assert vectors.component_resolved(vectors.resolve(dict(unequal_mass, particles=1024)))[3] is False  # 20.5
    assert vectors.component_resolved(vectors.resolve(dict(unequal_mass, particles=1600))) == [True]*4


@pytest.mark.parametrize("family", ["anisotropic", "unequal_width", "unequal_mass"])
def test_far_outliers_are_spill_not_shape(family):
    spec = next(x for x in vectors.TASKS if x["family"] == family)
    samples = vectors.sample_target(spec, 4096, torch.Generator().manual_seed(993), spec["steps"])
    means = torch.tensor(spec["means"])
    first = (torch.cdist(samples, means).argmin(1) == 0).nonzero().squeeze(1)
    # 8% of component 0 moved 1.5-2.5 units off its mean: still nearest to it, far outside 4 sigma.
    stray = first[:int(.08*len(first))]
    offsets = torch.linspace(1.5, 2.5, len(stray))
    samples[stray] = means[0] + torch.stack([torch.zeros_like(offsets), -offsets], 1)
    result = vectors.score_samples(samples, spec, spec["steps"])
    old_bounds = [["component_covariance_error", "<=", .85], ["component_min_eigen_ratio", ">=", .15]]
    core_bounds = [x for x in spec["thresholds"] if x[0] != "resolved_max_component_spill"]
    assert result["component_covariance_errors"][0] > 2.6
    assert not vectors.passes(result, old_bounds)
    assert result["component_core_covariance_error"] <= .2
    assert vectors.passes(result, core_bounds)
    assert result["max_component_spill"] == pytest.approx(result["component_spill"][0]) and result["max_component_spill"] > .05
    assert result["resolved_max_component_spill"] == result["max_component_spill"]
    assert not vectors.passes(result, spec["thresholds"])


@pytest.mark.parametrize("key", ["component_core_covariance_error", "component_core_min_eigen_ratio",
                                 "max_component_spill", "mass_tv", *vectors.RESOLVED_SHAPE_METRICS])
def test_overlap_rejects_unidentifiable_component_requirements(key):
    spec = deepcopy(next(x for x in vectors.TASKS if x["family"] == "overlapping"))
    spec["thresholds"].append([key, "<=", .15])
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
