"""Public image fixture conformance and discriminating scientific controls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from particlegan import GANTrainer, Recipe, UpdatePolicy
from benchmarks.toy_audit.api_images import (
    ImageCritic, ImageGenerator, build_case, conditional_problem, list_cases,
    oracle_controls, ordered_bank_sha256, score_case, template_bank, WORD_CASE_ID,
    score_words, word_bank, word_oracle_controls,
)


ALL_CASES = list_cases()
CASES = [case for case in ALL_CASES if case["kind"] == "image"]
BY_ID = {case["id"]: case for case in CASES}
CONDITIONAL = [case for case in CASES if case.get("query")]
UNCONDITIONAL = [case for case in CASES if not case.get("query")]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def equal_state(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert first.dtype == second.dtype and first.shape == second.shape
        # Native policy diagnostics use NaNs for unobserved rows. Exact byte
        # equality proves preservation without treating a sentinel as a
        # changed value (or relaxing finite model/output checks).
        assert torch.equal(first.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           second.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(first, np.ndarray):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            equal_state(a, b)
    else:
        assert first == second


def test_every_retained_image_and_architecture_is_declared_without_history_edits():
    catalog = json.loads((Path(__file__).parents[1] / "reports/toy_audit/catalog.json").read_text())
    legacy = {case["id"] for case in catalog["cases"] if "pattern" in case.get("spec", {})}
    assert len(legacy) == 39
    assert {name for case in CASES for name in case["legacy_ids"]} == legacy
    assert len(BY_ID) == len(CASES) == 79
    assert len({case["pattern"] for case in CASES}) == 34
    assert len(UNCONDITIONAL) == 73 and len(CONDITIONAL) == 6
    assert all(case["scientific_status"] == "NEW_VARIANT_UNMEASURED" for case in CASES)
    for legacy_id in [f"pr{case['pr']}" for case in catalog["cases"] if case.get("pr") and "pattern" in case.get("spec", {})]:
        variants = [case for case in UNCONDITIONAL if legacy_id in case["legacy_ids"]]
        assert {(case["architecture"], case["width"]) for case in variants} == {("transpose", 12), ("residual_upsample", 16)}
    source = [case for case in CASES if case["arm_identity"] == "checked-in-source-host"]
    assert len(source) == 4 and all(case["architecture"] == "transpose" and case["width"] == 12 for case in source)
    assert all(case["default_steps"] == (480 if case["architecture"] in ("mean_discriminator", "uniform_generator") or case["width"] == 2 else 600) for case in CASES)


@pytest.mark.parametrize("healthy_legacy,other_legacy", [
    ("develop-img_bars4", "develop-img_tiny_generator"),
    ("develop-img_blobs4", "develop-img_mean_discriminator"),
    ("develop-img_stripes2", "develop-img_uniform_generator"),
    ("develop-img_intensity2", "pr58"),
])
def test_shared_bank_does_not_overwrite_healthy_host_question(healthy_legacy, other_legacy):
    healthy = [case for case in CASES if case["legacy_ids"] == [healthy_legacy]]
    other = [case for case in CASES if case["legacy_ids"] == [other_legacy]]
    assert healthy and other
    assert {case["ordered_template_sha256"] for case in healthy + other} == {
        healthy[0]["ordered_template_sha256"]}
    assert not ({case["goal"] for case in healthy} & {case["goal"] for case in other})
    for case in healthy:
        assert "balanced" in case["goal"]
        assert "control_reason" not in case
        assert not any(claim in case["goal"] for claim in (
            "width2", "one-dimensional latent", "mean-only", "spatially uniform", "reuses"))


@pytest.mark.parametrize("legacy_id,mechanism", [
    ("develop-img_tiny_generator", "width2"),
    ("develop-img_mean_discriminator", "mean-only"),
    ("develop-img_uniform_generator", "spatially uniform"),
])
def test_restricted_host_question_names_its_actual_control(legacy_id, mechanism):
    case = next(case for case in CASES if case["legacy_ids"] == [legacy_id])
    assert mechanism in case["goal"]
    assert "diagnostic" in case["goal"] or "negative control" in case["goal"]
    assert case["control_reason"]
    assert case["default_steps"] == 480
    assert case["scientific_status"] == "NEW_VARIANT_UNMEASURED"


@pytest.mark.parametrize("pattern", sorted({case["pattern"] for case in CASES}))
def test_lossless_ordered_banks_are_bound_and_independent(pattern):
    cases = [case for case in CASES if case["pattern"] == pattern]
    bank = template_bank(pattern)
    assert bank.dtype == torch.float32 and bank.shape[1:] == (1, 8, 8)
    assert ordered_bank_sha256(bank) == cases[0]["ordered_template_sha256"]
    assert torch.isfinite(bank).all() and bank.min() >= 0 and bank.max() <= 1
    original = bank.clone()
    bank.zero_()
    assert torch.equal(template_bank(pattern), original)
    assert ordered_bank_sha256(original.flip(0)) != ordered_bank_sha256(original)


@pytest.mark.parametrize("case", UNCONDITIONAL, ids=lambda case: case["id"])
def test_every_unconditional_gate_rejects_mass_error_even_with_quality_and_coverage(case):
    controls = oracle_controls(case["id"])
    assert all(row["passed"] == row["expected_pass"] for row in controls.values())
    unequal = controls["mass_imbalance"]
    assert unequal["metrics"]["hq"] == 1
    assert unequal["metrics"]["modes"] == case["thresholds"]["modes"]
    assert set(unequal["failed_bounds"]) == {"distribution_tv", "finite_template_tv"}


@pytest.mark.parametrize("case", CONDITIONAL, ids=lambda case: case["id"])
def test_conditional_oracles_detect_wrong_correspondence_masks_and_channels(case):
    controls = oracle_controls(case["id"])
    assert all(row["passed"] == row["expected_pass"] for row in controls.values())
    swapped = controls["swapped_contexts"]
    assert swapped["metrics"]["marginal_hq"] == 1
    assert swapped["metrics"]["marginal_distribution_tv"] == 0
    assert any("context_" in bound for bound in swapped["failed_bounds"])
    if case["query"] == "masked_completion":
        oracle = controls["oracle"]["metrics"]
        assert oracle["context_0_modes"] == 2 and oracle["context_0_distribution_tv"] == 0
        assert oracle["context_1_modes"] == oracle["context_2_modes"] == 1
    if case["query"] == "rgb_assignment":
        assert "context_0_chromatic_order_accuracy" in controls["grayscale_without_color"]["failed_bounds"]
    else:
        assert any("observed_rmse" in bound for bound in controls["observed_pixel_corruption"]["failed_bounds"])


def test_actual_conditional_contexts_differentiate_queries_without_mode_labels():
    sparse = conditional_problem("sparse_completion")
    assert sparse["contexts"].shape == (2, 2, 8, 8)
    assert not torch.equal(sparse["contexts"][0], sparse["contexts"][1])
    inpaint = conditional_problem("masked_completion")
    assert [len(bank) for bank in inpaint["valid_targets"]] == [2, 1, 1]
    assert inpaint["contexts"].shape == (3, 2, 8, 8)
    color = conditional_problem("rgb_assignment")
    assert color["contexts"].shape == (2, 1, 8, 8)
    assert color["marginal_targets"].shape == (2, 3, 8, 8)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_every_factory_completes_one_real_public_api_update_and_goal_views(case):
    fixture = build_case(case["id"], max_steps=1)
    assert isinstance(fixture.recipe, Recipe)
    assert fixture.recipe.batch_size == case["batch_size"]
    assert fixture.max_steps == 1
    if case.get("query"):
        assert fixture.trainer is None and isinstance(fixture.policy, UpdatePolicy)
        assert fixture.policy.row_semantics == "conditional"
        for key, value in case["conditional_api_overrides"].items():
            assert getattr(fixture.recipe, key) == value
    else:
        assert isinstance(fixture.trainer, GANTrainer)
    row = fixture.step()
    assert row["step"] == fixture.completed_steps == 1
    assert all(torch.isfinite(value) for value in row.values() if isinstance(value, torch.Tensor))
    observed = fixture.observe(n=32)
    assert isinstance(observed["passed"], bool)
    assert isinstance(observed["failed_bounds"], list)
    assert observed["views"]
    assert all(isinstance(value, (float, int)) and np.isfinite(value) for value in observed["metrics"].values())
    for view in observed["views"]:
        assert view["kind"] == "image" and view["target"].ndim == view["samples"].ndim == 4
        assert view["target"].shape[2:] == view["samples"].shape[2:] == (8, 8)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        fixture.step()


@pytest.mark.parametrize("case_id", [
    "image-develop-img_bars4-residual_upsample16",
    "image-pr61-conditional-transpose12",
    "image-pr63-conditional-residual_upsample16",
    "image-pr65-conditional-transpose12",
])
def test_observation_is_pure_and_next_update_exactly_matches_unobserved_prefix(case_id):
    baseline = build_case(case_id, max_steps=2)
    observed = build_case(case_id, max_steps=2)
    baseline.step()
    observed.step()
    equal_state(baseline.state_dict(), observed.state_dict())
    before = deepcopy(observed.state_dict())
    rng = torch.get_rng_state().clone()
    modes = [(module, module.training) for root in (observed.G, observed.D, observed.prior) for module in root.modules()]
    first = observed.observe(n=64, seed=777)
    second = observed.observe(n=64, seed=777)
    equal_state(before, observed.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(module.training == mode for module, mode in modes)
    equal_state(first, second)
    baseline.step()
    observed.step()
    equal_state(baseline.state_dict(), observed.state_dict())


def test_original_information_and_representation_controls_have_their_mathematical_reason():
    mean_case = next(case for case in CASES if case["architecture"] == "mean_discriminator")
    critic = ImageCritic(mean_case)
    bank = template_bank(mean_case["pattern"])
    assert torch.equal(critic(bank), critic(bank[:1].expand(len(bank), -1, -1, -1)))
    uniform_case = next(case for case in CASES if case["architecture"] == "uniform_generator")
    generator = ImageGenerator(uniform_case)
    samples = generator(torch.zeros(32, uniform_case["z_dim"]))
    assert torch.equal(samples, samples[:, :, :1, :1].expand_as(samples))
    stripe_std = template_bank(uniform_case["pattern"]).flatten(1).std(1, unbiased=False)
    assert float(stripe_std.min()) > uniform_case["thresholds"]["quality_rmse"]


@pytest.mark.parametrize("architecture", ["transpose", "residual_upsample", "uniform_generator", "mean_discriminator"])
def test_unconditional_cores_match_the_original_architecture_math(architecture):
    from benchmarks.transfer_suite.image_tasks import Generator, Discriminator
    case = next(case for case in UNCONDITIONAL if case["architecture"] == architecture)
    spec = dict(architecture=architecture, width=case["width"], z_dim=case["z_dim"])
    original_g, original_d = Generator(spec), Discriminator(spec)
    api_g, api_d = ImageGenerator(case), ImageCritic(case)
    original_g.load_state_dict(api_g.state_dict())
    original_d.load_state_dict(api_d.state_dict())
    latent = torch.linspace(-1, 1, 32 * case["z_dim"]).reshape(32, -1)
    targets = template_bank(case["pattern"])
    assert torch.equal(original_g(latent), api_g(latent))
    assert torch.equal(original_d(targets), api_d(targets))


@pytest.mark.parametrize("alias", [case for case in CASES if case.get("alias_of")], ids=lambda case: case["id"])
def test_aliases_have_identical_resolved_api_trajectories_and_serving(alias):
    first = build_case(alias["id"], max_steps=1)
    second = build_case(alias["alias_of"], max_steps=1)
    assert first.recipe.to_dict() == second.recipe.to_dict()
    equal_state(first.state_dict()["api_state"], second.state_dict()["api_state"])
    first.step()
    second.step()
    equal_state(first.state_dict()["api_state"], second.state_dict()["api_state"])
    equal_state(first.data_generator.get_state(), second.data_generator.get_state())
    a, b = first.observe(64), second.observe(64)
    equal_state(a["metrics"], b["metrics"])
    equal_state(a["views"][0]["samples"], b["views"][0]["samples"])


@pytest.mark.parametrize("case_id", ["image-pr61-conditional-transpose12", "image-develop-img_bars4-residual_upsample16"])
def test_scheduled_public_recipe_keeps_its_full_horizon_under_short_execution(case_id):
    fixture = build_case(case_id, recipe_name="ka2", max_steps=1)
    assert fixture.recipe.total_steps == 600
    assert fixture.recipe.continuous_policy is None
    fixture.step()
    assert fixture.observe(32)["metrics"]["policy_latent_perturbation"] == 0


def test_unknown_cases_malformed_images_and_missing_conditional_contexts_fail_closed():
    with pytest.raises(ValueError, match="unknown image case"):
        build_case("not-a-toy")
    with pytest.raises(ValueError, match="unknown image pattern"):
        template_bank("not-a-pattern")
    case = UNCONDITIONAL[0]
    for invalid in [torch.empty(0, 1, 8, 8), torch.full((32, 1, 8, 8), float("nan")), torch.zeros(32, 8, 8)]:
        with pytest.raises(ValueError, match="finite nonempty"):
            score_case(case["id"], invalid)
    conditional = CONDITIONAL[0]
    for invalid_ids in [None, torch.zeros(32, dtype=torch.long), torch.full((32,), -1, dtype=torch.long), torch.zeros(32)]:
        with pytest.raises(ValueError, match="every declared conditional context"):
            score_case(conditional["id"], torch.zeros(32, 1, 8, 8), context_ids=invalid_ids)
    with pytest.raises(ValueError, match="max_steps"):
        build_case(case["id"], max_steps=True)


def test_word_gate_checks_confidence_mass_correct_pairing_and_padding():
    controls = word_oracle_controls()
    assert all(row["passed"] == row["expected_pass"] for row in controls.values())
    assert controls["oracle"]["metrics"]["reconstruction_nll"] == 0
    assert controls["diffuse_correct_argmax"]["metrics"]["quality_fraction"] == 0
    assert controls["mass_imbalance"]["metrics"]["quality_fraction"] == 1
    assert controls["mass_imbalance"]["metrics"]["modes"] == 5
    assert controls["mass_imbalance"]["failed_bounds"] == ["mass_tv"]
    assert "reconstruction_exact" in controls["swapped_reconstruction"]["failed_bounds"]
    assert controls["wrong_padding"]["metrics"]["quality_fraction"] == 0
    target = word_bank().numpy()
    with pytest.raises(ValueError, match="normalized probabilities"):
        score_words(np.repeat(target * 2, 40, axis=0), target)
    word = next(case for case in ALL_CASES if case["id"] == WORD_CASE_ID)
    assert word["legacy_ids"] == ["source-family-15"]
    assert word["default_steps"] == 20001 and word["recipe_schedule_horizon"] == 20000


@pytest.mark.parametrize("recipe_name", ["ka2", "atlas"])
def test_joint_word_public_api_reconstruction_observation_and_prefix_purity(recipe_name):
    baseline = build_case(WORD_CASE_ID, recipe_name=recipe_name, max_steps=2)
    fixture = build_case(WORD_CASE_ID, recipe_name=recipe_name, max_steps=2)
    if recipe_name == "ka2":
        assert fixture.recipe.total_steps == 20000
    baseline.step()
    fixture.step()
    equal_state(baseline.state_dict(), fixture.state_dict())
    before = deepcopy(fixture.state_dict())
    global_rng = torch.get_rng_state().clone()
    observed = fixture.observe(n=128, seed=901)
    equal_state(before, fixture.state_dict())
    assert torch.equal(global_rng, torch.get_rng_state())
    assert all(np.isfinite(value) for value in observed["metrics"].values())
    texts = [view for view in observed["views"] if view["kind"] == "text"]
    assert texts[0]["target_labels"] == ["apple_", "grape_", "lemon_", "melon_", "berry_"]
    for view in texts:
        assert len(view["sample_labels"]) == len(view["samples"])
        assert all(len(label) == 6 for label in view["sample_labels"])
    images = [view for view in observed["views"] if view["kind"] == "image"]
    assert all(view["samples"].shape[1:] == (1, 28, 6) for view in images)
    baseline.step()
    fixture.step()
    equal_state(baseline.state_dict(), fixture.state_dict())
