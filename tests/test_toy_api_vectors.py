"""Law/gate discrimination and actual public-API caller software controls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.toy_audit import api_vectors as api
from particlegan import GANTrainer, Recipe


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def definition(identifier):
    return next(case for case in api.list_cases() if case["id"] == identifier)


def oracle(case, *, n=None, step=None, seed=713):
    return api._target(case, n or case["eval_samples"], torch.Generator().manual_seed(seed),
                       case["default_steps"] if step is None else step)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        # Optimizer records use uninitialized NaN sentinels. Compare their
        # exact representation too; NaN != NaN is not trajectory divergence.
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_every_owned_historical_question_and_all_proposal_arms_are_mapped():
    cases = api.list_cases()
    assert len(cases) == len({case["id"] for case in cases}) == 54
    catalog = json.loads((Path(__file__).parents[1] / "reports/toy_audit/catalog.json").read_text())["cases"]
    expected = {case["id"] for case in catalog if case["id"].startswith("atlas-")
                or case.get("spec", {}).get("kind") in {"gaussian_mixture", "spiral", "annulus"}}
    expected |= {"develop-two_pole", "develop-mode_hold", "source-family-10", "source-family-16"}
    assert {old for case in cases for old in case["legacy_ids"]} == expected
    for case in cases:
        assert case["goal"] and case["thresholds"] and case["sampling"] and case["scope"]
        assert case["default_steps"] > 0 and case["eval_samples"] > 0
    for pr in api.PROPOSAL_DEFINITIONS:
        assert {case["id"] for case in cases if case.get("comparison_group") == f"api-pr{pr}"} == {f"api-pr{pr}-published", f"api-pr{pr}-control"}
    # Metadata copies cannot silently change another observation's contract.
    cases[0]["legacy_ids"].clear()
    assert api.list_cases()[0]["legacy_ids"]


@pytest.mark.parametrize("case", api.list_cases(), ids=lambda case: case["id"])
def test_fixed_gate_accepts_independent_full_target_law(case):
    result = api.score_case(case, oracle(case), case["default_steps"])
    assert result["passed"], result["failed_bounds"]
    assert result["failed_bounds"] == []


def test_missing_rare_mass_is_rejected_even_below_resolved_covariance_floor():
    case = definition("api-vector-unequal-mass")
    changed = deepcopy(case)
    changed["spec"]["masses"] = [.55 / .98, .30 / .98, .13 / .98, 0.]
    result = api.score_case(case, oracle(changed), case["default_steps"])
    assert not result["passed"]
    assert "min_mass_ratio >= 0.25" in result["failed_bounds"]
    assert case["spec"]["masses"][3] * case["particles"] < vector_floor()


def vector_floor():
    return api.vector_tasks.PARTICLE_FLOOR


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-vector-narrow", "api-stress-fast-critic"])
def test_correct_centers_and_masses_cannot_substitute_for_width(identifier):
    case = definition(identifier)
    centers = torch.tensor(case["spec"]["means"])
    fake = centers[torch.arange(case["eval_samples"]) % len(centers)]
    result = api.score_case(case, fake, case["default_steps"])
    assert not result["passed"]
    assert any("eigen" in failure for failure in result["failed_bounds"])
    assert result["metrics"]["projection_ks"] > .06


def test_overlapping_ring_centers_false_positive_fails_analytic_cdf():
    case = definition("api-stress-overlapping-data")
    centers = torch.tensor(case["spec"]["means"])
    fake = centers[torch.arange(4096) % 8]
    old = api.vector_tasks.score_samples(fake, case["spec"], case["default_steps"])
    assert api.vector_tasks.passes(old, case["spec"]["thresholds"])
    new = api.score_case(case, fake, case["default_steps"])
    assert not new["passed"]
    assert new["failed_bounds"] == ["projection_ks <= 0.06"]
    assert not any("component" in bound[0] for bound in case["thresholds"])


def test_anisotropic_trace_matched_wrong_axes_are_rejected():
    case = definition("api-vector-anisotropic")
    changed = deepcopy(case)
    covariance = np.asarray(changed["spec"]["covariances"])
    changed["spec"]["covariances"] = [(np.eye(2) * np.trace(c) / 2).tolist() for c in covariance]
    result = api.score_case(case, oracle(changed), case["default_steps"])
    assert not result["passed"]
    assert result["metrics"]["projection_ks"] > .06


def test_continuous_annulus_wrong_radial_mass_is_rejected():
    case = definition("api-reserved-annulus")
    angle = torch.arange(4096) * (2 * torch.pi / 4096)
    fake = case["spec"]["radius_max"] * torch.stack([angle.cos(), angle.sin()], 1)
    result = api.score_case(case, fake, case["default_steps"])
    assert not result["passed"]
    assert result["metrics"]["projection_ks"] > .06


def test_gaussian_gate_rejects_same_mean_covariance_non_gaussian_circle():
    case = definition("api-gaussian2d")
    angle = torch.arange(4096) * (2 * torch.pi / 4096)
    fake = 1 + .2 * np.sqrt(2) * torch.stack([angle.cos(), angle.sin()], 1)
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert result["metrics"]["radial_ks"] > .075


def test_twelve_clean_ring_rows_have_an_explicit_mass_and_width_obstruction():
    case = definition("api-ring8-resolution12")
    angle = torch.arange(8) * torch.pi / 4
    centers = 3 * torch.stack([angle.cos(), angle.sin()], 1)
    twelve = centers[torch.arange(12) % 8]
    fake = twelve[torch.arange(12288) % 12]
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert result["metrics"]["mass_tv"] == pytest.approx(1 / 6)
    assert result["metrics"]["min_cov_eigen"] == 0
    assert case["resource_change"]["current_rows"] == 12
    assert definition("api-ring8-acquire")["resource_change"]["current_rows"] == 256


def test_256_atom_ring_has_a_constructive_gate_witness_not_a_trained_pass():
    case = definition("api-ring8-acquire")
    modes = torch.arange(8) * torch.pi / 4
    centers = 3 * torch.stack([modes.cos(), modes.sin()], 1)
    u = (torch.arange(16) + .5) / 16
    radii = torch.sqrt(-2 * torch.log(1 - u))
    angle = torch.arange(16) * torch.pi * (3 - np.sqrt(5))
    half = .07 * radii[:, None] * torch.stack([angle.cos(), angle.sin()], 1)
    offsets = torch.cat([half, -half])
    population = (centers[:, None] + offsets[None]).reshape(256, 2)
    # One predeclared iid draw from a fixed constructive population; never
    # used to initialize a GAN, tune a gate, select a seed or claim learning.
    samples = population[torch.randint(256, (4096,), generator=torch.Generator().manual_seed(1931))]
    result = api.score_case(case, samples)
    assert result["passed"], result["failed_bounds"]


def test_two_pole_exact_batch_law_rejects_centers_and_one_pole_travel():
    case = definition("api-two-pole-grid12")
    exact = oracle(case)
    assert api.score_case(case, exact)["passed"]
    for fake in (torch.tensor([-1.] * 6 + [1.] * 6)[:, None], torch.full((12, 1), .6)):
        assert not api.score_case(case, fake)["passed"]
    assert not api.score_case(case, exact.repeat(2, 1))["passed"]


def test_native_all_modes_with_zero_width_and_missing_one_mode_fail():
    case = definition("api-grid100")
    centers = api.problems.evaluation_geometry("grid100")[0]
    collapsed = centers[torch.arange(20000) % 100]
    assert not api.score_case(case, collapsed)["passed"]
    fake = oracle(case)
    ids = torch.cdist(fake, centers).argmin(1)
    fake[ids == 0] += centers[1] - centers[0]
    result = api.score_case(case, fake)
    assert not result["passed"]
    assert "modes >= 100" in result["failed_bounds"]


def test_moving_targets_use_exact_original_update_boundaries():
    case = definition("api-rotated100-moving")
    assert api._native_angle(case, 500) == 0
    assert api._native_angle(case, 501) == pytest.approx(np.pi / 6)
    assert api._native_angle(case, 1000) == pytest.approx(np.pi / 6)
    assert api._native_angle(case, 1001) == pytest.approx(np.pi / 3)
    assert not api.score_case(case, oracle(case, step=500), 1500)["passed"]
    drift = definition("api-vector-scale-drift")
    assert api.vector_tasks.target_scale(drift["spec"], 32) != api.vector_tasks.target_scale(drift["spec"], 1600)
    assert drift["law"]["schedule_budget"] == 1600


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-pr49-published", "api-pr49-control", "api-stress-r1-r2", "api-ring8-acquire", "api-two-pole-grid12", "api-gaussian2d", "api-grid100"])
def test_real_public_trainer_update_short_schedule_and_pure_observation(identifier):
    fixture = api.build_case(identifier, recipe_name="auto", max_steps=2)
    assert isinstance(fixture.recipe, Recipe) and isinstance(fixture.trainer, GANTrainer)
    full = fixture.metadata["default_steps"]
    assert fixture.recipe.total_steps == (full if fixture.recipe.continuous_policy is None else None)
    assert fixture.step()["step"] == 1
    assert fixture.trainer.completed_steps == fixture.completed_steps == 1
    before = fixture.state_dict()
    global_rng = torch.get_rng_state().clone()
    observation = fixture.observe(n=64)
    assert isinstance(observation["passed"], bool)
    assert observation["failed_bounds"] and not observation["passed"]
    assert all(view["target"].numel() and view["samples"].numel() for view in observation["views"])
    assert torch.equal(global_rng, torch.get_rng_state())
    assert_nested_equal(before, fixture.state_dict())


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-reserved-alternating-critic-updates", "api-ring8-acquire", "api-gaussian2d"])
def test_checkpoint_resume_retains_data_rng_optimizers_and_next_update(identifier):
    straight = api.build_case(identifier, recipe_name="auto", max_steps=3)
    straight.step()
    checkpoint = straight.state_dict()
    resumed = api.build_case(identifier, recipe_name="auto", max_steps=3)
    resumed.load_state_dict(checkpoint)
    assert_nested_equal(straight.state_dict(), resumed.state_dict())
    straight.observe(n=64)  # independent read in only one continuation
    straight.step(); resumed.step()
    assert_nested_equal(straight.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("phase,completed", [("hold", 1200), ("shift", 2400), ("shift", 2800)])
def test_failed_scientific_phase_prerequisite_stops_before_training(phase, completed):
    # A synthetic phase clock exercises only the protocol guard, with no
    # trained model or simulated convergence. Any attempt to access a trainer
    # after the failed held-out gate would raise independently in this fixture.
    fixture = object.__new__(api.VectorFixture)
    fixture.metadata = definition("api-ring8-" + phase)
    fixture.execution_steps = fixture.metadata["default_steps"]
    fixture.completed_steps = completed
    fixture.phase_receipts = {}
    fixture.observe = lambda **kwargs: {"metrics": {"mass_tv": .20}, "passed": False,
                                       "failed_bounds": ["mass_tv <= 0.075"]}
    with pytest.raises(RuntimeError, match="scientific prerequisite failed"):
        fixture.step()
    assert fixture.completed_steps == completed
    assert not fixture.phase_receipts[str(completed)]["passed"]


def test_half_frequency_d_loop_preserves_phase_and_uses_public_factories():
    fixture = api.build_case("api-reserved-alternating-critic-updates", recipe_name="auto", max_steps=4)
    assert fixture.recipe.name == "ka2" and fixture.recipe.continuous_policy is None
    assert fixture.recipe.total_steps == 2400
    assert fixture.recipe.input_noise_std == fixture.recipe.output_noise_std == 0
    for step in range(1, 5):
        counts = fixture.step()
        assert counts["g"] == step and counts["d"] == (step + 1) // 2
    state = fixture.state_dict()
    fixture.observe(n=64)
    assert_nested_equal(state, fixture.state_dict())
    with pytest.raises(ValueError, match="explicitly KA2"):
        api.build_case("api-reserved-alternating-critic-updates", recipe_name="atlas", max_steps=1)
    with pytest.raises(ValueError, match="lacks the Atlas/DV12"):
        api.build_case("api-stress-r1-r2", recipe_name="atlas", max_steps=1)


@pytest.mark.parametrize("identifier", ["api-vector-two-broad", "api-reserved-alternating-critic-updates"])
def test_corrupted_phase_clock_is_rejected_before_resume_mutates_any_owner(identifier):
    fixture = api.build_case(identifier, recipe_name="auto", max_steps=3)
    fixture.step()
    before = fixture.state_dict()
    corrupt = deepcopy(before)
    corrupt["completed_steps"] += 1
    with pytest.raises(ValueError, match="phase clock"):
        fixture.load_state_dict(corrupt)
    assert_nested_equal(before, fixture.state_dict())


def test_short_draws_and_nonfinite_arrays_cannot_pass():
    case = definition("api-vector-two-broad")
    assert not api.score_case(case, oracle(case, n=64))["passed"]
    with pytest.raises(ValueError, match="finite"):
        api.score_case(case, torch.full((4096, 2), float("nan")))
    with pytest.raises(ValueError, match="prefix"):
        api.build_case(case["id"], max_steps=case["default_steps"] + 1)
