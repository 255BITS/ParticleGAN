"""CPU structural controls only; no prospective GPU numerical credit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from experiments.forge.policy_contracts import (
    COHORT, SUFFIX, EXECUTION_DEVICE, PRECISION, IMAGE_C6, NATIVE_C6, PARENT_TASK_IDS, SELECTION_SOURCE,
    VECTOR_C6, is_policy_task, load_policy_variants, policy_contract_blockers,
    policy_recipe_overrides, resolve_policy_view, validate_policy_observation,
)


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@pytest.fixture
def parents():
    return {name: read(ROOT / "configs/forge/tasks" / f"{name}.json") for name in PARENT_TASK_IDS}


@pytest.fixture
def variants(parents):
    return load_policy_variants(ROOT, parents)


@pytest.fixture
def view():
    return read(ROOT / "configs/forge/views/discriminator_stability.json")


@pytest.fixture
def sandbox(tmp_path, parents):
    """Current small source/declaration bytes, no historical Git/raw dependencies."""
    relative_files = {SELECTION_SOURCE}
    for name in PARENT_TASK_IDS:
        relative_files.add(f"configs/forge/tasks/{name}.json")
        variant_path = f"configs/forge/task-variants/{COHORT}/{name}{SUFFIX}.json"
        relative_files.add(variant_path)
        relative_files.update(parents[name]["evaluation"]["sources"])
        relative_files.update(read(ROOT / variant_path)["execution"]["policy_contract"]["sources"])
    for relative in sorted(relative_files):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def variant_path(root, name):
    return root / "configs/forge/task-variants" / COHORT / f"{name}{SUFFIX}.json"


def recipe_for(name):
    # Frozen reference metadata, not a mirror of validator REQUIRED_RECIPE.
    card = read(ROOT / SELECTION_SOURCE)
    group = "atlas-vector" if name in VECTOR_C6 else "atlas-image" if name in IMAGE_C6 else "atlas-native100"
    recipe = deepcopy(card["resolved_recipe_groups"][group]["recipe"])
    recipe.update(prior_kind="particles", sigma_rel=0., standardize=False)
    return recipe


def test_all26_exact_gate_host_initializer_horizon_and_observation_contracts_are_preserved(parents, variants):
    assert set(variants) == {name + SUFFIX for name in PARENT_TASK_IDS}
    for name, parent in parents.items():
        variant = variants[name + SUFFIX]
        assert is_policy_task(variant)
        assert variant["policy_parent"]["task_sha256"] == hashlib.sha256(
            (ROOT / "configs/forge/tasks" / f"{name}.json").read_bytes()).hexdigest()
        before, after = deepcopy(parent["evaluation"]), deepcopy(variant["evaluation"])
        assert before.pop("scoring_weights") == "live"
        assert after.pop("scoring_weights") == "state_selected"
        after.pop("policy_observation")
        assert before == after  # Includes every numeric kernel, source, count and guard.
        for field in parent["execution"]:
            if field in {"prior", "continuation_of", "execution_group", "device"}:
                continue
            assert variant["execution"][field] == parent["execution"][field]
        prior = variant["execution"]["prior"]
        assert {key: prior[key] for key in ("kind", "sigma", "standardize", "learnable")} == {
            "kind": "particle_cloud", "sigma": 0., "standardize": False, "learnable": True}
        expected_resources = {**parent["resources"], "device": "cuda", "gpus": 1,
                              "gpu_memory_mb": max(2048, parent["resources"]["gpu_memory_mb"])}
        if "allow_cpu" in expected_resources:
            expected_resources["allow_cpu"] = False
        assert variant["resources"] == expected_resources
        assert "live_sampling" not in variant["requires_capabilities"]
        assert "policy_controls" in variant["requires_capabilities"]
        assert "served_sampling" in variant["requires_capabilities"]


def test_explicit_cohort_resolves_same_main_goal_with_complete5_19_2_and_preserves_inputs(parents, variants, view):
    tasks = {**parents, **variants}
    originals = deepcopy((tasks, view))
    resolved, selected = resolve_policy_view(view, tasks, {"task_cohort": COHORT})
    assert (tasks, view) == originals
    assert resolved["id"] == view["id"] and resolved["goal"] == view["goal"]
    assert view["revision"] == 3 and resolved["revision"] == 4
    assert resolved["task_cohort"] == COHORT
    assert len(resolved["parent_view_fingerprint"]) == len(resolved["cohort_fingerprint"]) == 64
    assert len(selected) == 26
    assert [sum(a["qualification_tier"] == tier for a in resolved["assignments"]) for tier in (1, 2, 3)] == [5, 19, 2]
    for old, new in zip(view["assignments"], resolved["assignments"]):
        assert new == {**old, "task": old["task"] + SUFFIX}
    selected[PARENT_TASK_IDS[0] + SUFFIX]["execution"]["steps"] = 1
    assert tasks[PARENT_TASK_IDS[0] + SUFFIX]["execution"]["steps"] == 80


@pytest.mark.parametrize("candidate", [
    {}, {"id": "atlas", "recipe_preset": "atlas"},
    {"recipe_preset": "atlas", "claim_contract": {"scoring_weights": "state_selected"}},
])
def test_family_or_serving_claim_never_implies_task_cohort(candidate, parents, view):
    copied_view, copied_tasks = resolve_policy_view(view, parents, candidate)
    assert copied_view == view and copied_tasks == parents
    assert copied_view is not view and copied_tasks is not parents
    assert not any(is_policy_task(t) for t in copied_tasks.values())


@pytest.mark.parametrize("cohort", ["atlas", "", False, {"id": COHORT}])
def test_unknown_or_malformed_explicit_cohort_fails_closed(cohort, parents, variants, view):
    with pytest.raises(ValueError, match="unknown explicit"):
        resolve_policy_view(view, {**parents, **variants}, {"task_cohort": cohort})


def test_dependency_checkpoint_and_uninterrupted_group_are_in_new_namespace(variants):
    hold = variants["ring_hold" + SUFFIX]
    extension = variants["ring_extension" + SUFFIX]
    assert hold["dependencies"] == [{"task": "mode_hold" + SUFFIX, "kind": "gate"}]
    assert extension["dependencies"] == [{"task": "ring_hold" + SUFFIX, "kind": "checkpoint"}]
    assert extension["execution"]["continuation_of"] == "ring_hold" + SUFFIX
    assert hold["execution"]["execution_group"] == extension["execution"]["execution_group"] == "ring_endurance" + SUFFIX
    assert hold["execution"]["uninterrupted"] and extension["execution"]["uninterrupted"]
    assert extension["execution"]["extension_steps"] == 300


def test_two_pole_retains_original_table_critic_and_no_sampling_path(variants, parents):
    task = variants["two_pole" + SUFFIX]
    assert task["execution"]["resources"] == {"num_particles": 12, "z_dim": 1, "batch_size": 12}
    assert task["execution"]["fixed_initialization"] == parents["two_pole"]["execution"]["fixed_initialization"] == {
        "critic": "stored_host_weights", "particles": "zeros"}
    observation = task["evaluation"]["policy_observation"]
    assert observation["sampler"] == "served_snapshot"
    assert observation["parameter_measurement"] == "selected_table_and_critic_gradient"
    assert observation["latent_policy"] == "not_applied_to_parameter_measurement"
    assert observation["output_noise"] is False
    assert set(task["execution"]["policy_resource_sources"]) == {
        "benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py"}


def test_all26_gpu_variants_bind_original_device_and_do_not_claim_numerical_parity(variants, parents):
    for name in PARENT_TASK_IDS:
        task = variants[name + SUFFIX]
        parent = parents[name]
        execution, resources = task["execution"], task["resources"]
        contract = execution["policy_contract"]
        provenance = execution["policy_device_provenance"]
        assert execution["device"] == resources["device"] == EXECUTION_DEVICE == "cuda"
        assert resources["gpus"] == 1 and resources["gpu_memory_mb"] >= 2048
        assert resources.get("allow_cpu", False) is False
        assert contract["execution_device"] == "cuda"
        assert contract["precision"] == PRECISION == "preserve_task_tensor_dtypes_no_autocast"
        assert contract["cpu_controls_scope"] == provenance["cpu_controls_scope"] == "structural_only"
        assert contract["numerical_equivalence_to_parent"] is False
        assert provenance["numerical_equivalence_to_parent"] is False
        assert provenance["parent_execution_device"] == parent["execution"].get("device")
        assert provenance["parent_resources"] == parent["resources"]
        assert provenance["parent_execution_fingerprint"] == task["policy_parent"]["execution_fingerprint"]
        assert task["resources"]["timeout_seconds"] == parent["resources"]["timeout_seconds"]
        assert task["execution"]["steps"] == parent["execution"]["steps"]
    assert parents["two_pole"]["execution"]["device"] == "cpu"
    assert parents["two_pole"]["resources"]["gpus"] == 0
    assert parents["two_pole"]["resources"]["allow_cpu"] is True


@pytest.mark.parametrize("section,field,value", [
    ("execution", "device", "cpu"), ("resources", "device", "cpu"),
    ("resources", "gpus", 0), ("resources", "allow_cpu", True),
    ("resources", "gpu_memory_mb", 0),
])
def test_cpu_fallback_or_removed_cuda_resources_cannot_qualify_a_prospective_declaration(section, field, value, variants):
    task = deepcopy(variants["two_pole" + SUFFIX])
    task[section][field] = value
    assert policy_contract_blockers(task, recipe_for("two_pole"))


@pytest.mark.parametrize("field,value", [
    ("execution_device", "cpu"), ("precision", "float16_autocast"),
    ("cpu_controls_scope", "numerical_qualification"), ("numerical_equivalence_to_parent", True),
])
def test_precision_or_cpu_numerical_credit_cannot_be_substituted(field, value, sandbox, parents):
    path = variant_path(sandbox, "two_pole")
    variant = read(path)
    variant["execution"]["policy_contract"][field] = value
    write(path, variant)
    with pytest.raises(ValueError, match="changes a frozen parent field"):
        load_policy_variants(sandbox, parents)


def test_parent_cpu_device_provenance_cannot_be_relabelled_as_gpu(sandbox, parents):
    path = variant_path(sandbox, "two_pole")
    variant = read(path)
    variant["execution"]["policy_device_provenance"]["parent_execution_device"] = "cuda"
    variant["execution"]["policy_device_provenance"]["parent_resources"]["gpus"] = 1
    write(path, variant)
    with pytest.raises(ValueError, match="changes a frozen parent field"):
        load_policy_variants(sandbox, parents)


def test_scheduled_component_noise_and_native_diagnostics_remain_separate(variants):
    for name in ("trajectory", "residual_student", "ae_gan_hold"):
        evaluation = variants[name + SUFFIX]["evaluation"]
        assert evaluation["eval_output_noise"] == "public_recipe_schedule"
        assert evaluation["policy_observation"]["output_noise"] == "resolved_recipe"
        assert evaluation["policy_observation"]["sampler"] == "ServedModel.generate"
    for name in NATIVE_C6:
        evaluation = variants[name + SUFFIX]["evaluation"]
        assert evaluation["eval_output_noise"] == "clean"
        assert evaluation["policy_observation"]["output_noise"] is False
        assert evaluation["policy_observation"]["diagnostic_weights"] == ["forced_ema"]
        assert evaluation["policy_observation"]["diagnostic_credit"] is False
        assert evaluation["eval_samples"] == 20000 and evaluation["holdout_samples"] == 100000
        assert evaluation["minimum_stable_checks"] == 5
        assert evaluation["early_eval_steps"] == [0, 1, 10, 25, 50, 100]
        assert evaluation["eval_interval"] == 250


def test_c6_exceptions_apply_only_to_known13_hosts(variants):
    for name in PARENT_TASK_IDS:
        overrides = policy_recipe_overrides(variants[name + SUFFIX])
        if name in VECTOR_C6:
            assert overrides == {"d_lr_mult": 1.5, "betas": [0., .99], "prior_reg": .05}
        elif name in IMAGE_C6 | NATIVE_C6:
            assert overrides == {"d_lr_mult": 1., "betas": [0., .999], "prior_reg": 0.}
        else:
            assert overrides == {}
            assert variants[name + SUFFIX]["execution"]["policy_recipe_overrides_provenance"] is None
        if overrides:
            provenance = variants[name + SUFFIX]["execution"]["policy_recipe_overrides_provenance"]
            assert provenance["owner"] == "task" and provenance["evidence_reuse"] is False


@pytest.mark.parametrize("mutation", [
    ("evaluation", "thresholds", [["hq", ">=", .5]]),
    ("evaluation", "observations", 3),
    ("evaluation", "minimum_stable_checks", 1),
    ("evaluation", "measurement", {"quality_rmse": 10.}),
    ("execution", "steps", 20),
    ("execution", "initializer", "supplied"),
    ("execution", "host_definition", {"architecture": "residual16"}),
    ("execution", "host", "img_blobs4"),
    ("resources", "timeout_seconds", 20),
    ("resources", "gpus", 0),
])
def test_coherent_variant_scientific_changes_cannot_hide_under_parent_pin(mutation, sandbox, parents):
    path = variant_path(sandbox, "img_intensity2")
    variant = read(path)
    section, key, value = mutation
    variant[section][key] = value
    write(path, variant)
    with pytest.raises(ValueError, match="changes a frozen parent field"):
        load_policy_variants(sandbox, parents)


@pytest.mark.parametrize("field,value", [("kind", "mog"), ("sigma", .025), ("standardize", True), ("learnable", False)])
def test_prior_width_standardization_and_ownership_cannot_change(field, value, sandbox, parents):
    path = variant_path(sandbox, "img_bars4")
    variant = read(path)
    variant["execution"]["prior"][field] = value
    write(path, variant)
    with pytest.raises(ValueError, match="changes a frozen parent field"):
        load_policy_variants(sandbox, parents)


def test_parent_byte_drift_is_rejected_even_when_json_semantics_match(sandbox, parents):
    path = sandbox / "configs/forge/tasks/two_pole.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="parent task byte identity drift"):
        load_policy_variants(sandbox, parents)


def test_supplied_parent_and_stale_parent_evaluation_fingerprints_are_rejected(sandbox, parents):
    changed = deepcopy(parents)
    changed["img_blobs4"]["evaluation"]["thresholds"][0][2] = 1
    with pytest.raises(ValueError, match="supplied parent task differs"):
        load_policy_variants(sandbox, changed)
    path = variant_path(sandbox, "img_blobs4")
    variant = read(path)
    variant["policy_parent"]["evaluation_fingerprint"] = "0" * 64
    write(path, variant)
    with pytest.raises(ValueError, match="fingerprint drift"):
        load_policy_variants(sandbox, parents)


@pytest.mark.parametrize("relative", ["particlegan/policy.py", "benchmarks/toy100/accuracy.py", SELECTION_SOURCE])
def test_public_policy_evaluator_and_reference_card_source_drift_rejects(relative, sandbox, parents):
    path = sandbox / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source binding drift"):
        load_policy_variants(sandbox, parents)


def test_policy_manifest_cannot_mask_parent_evaluator_hash_with_new_source(sandbox, parents):
    path = variant_path(sandbox, "ae_gan_hold")
    variant = read(path)
    relative = "benchmarks/transfer_suite/protocol.py"
    source = sandbox / relative
    source.write_bytes(source.read_bytes() + b"\n")
    variant["execution"]["policy_contract"]["sources"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(path, variant)
    with pytest.raises(ValueError, match="cannot replace a parent evaluator"):
        load_policy_variants(sandbox, parents)


@pytest.mark.parametrize("relative", ["../outside.py", "/tmp/source.py", "particlegan/../policy.py", "particlegan\\policy.py"])
def test_source_paths_are_relative_and_do_not_escape_snapshot(relative, sandbox, parents):
    path = variant_path(sandbox, "two_pole")
    variant = read(path)
    variant["execution"]["policy_contract"]["sources"][relative] = "0" * 64
    write(path, variant)
    with pytest.raises(ValueError, match="safe relative paths"):
        load_policy_variants(sandbox, parents)


def test_missing_declaration_and_extra_duplicate_filename_are_rejected(sandbox, parents):
    path = variant_path(sandbox, "ring_extension")
    extra = path.parent / "copy.json"
    extra.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="filename/ID mismatch"):
        load_policy_variants(sandbox, parents)
    extra.unlink()
    path.unlink()
    with pytest.raises(ValueError, match="exactly the original 26"):
        load_policy_variants(sandbox, parents)


@pytest.mark.parametrize("change", ["dependency", "continuation", "group"])
def test_original_cross_cohort_dependency_checkpoint_or_group_is_rejected(change, sandbox, parents):
    path = variant_path(sandbox, "ring_extension")
    variant = read(path)
    if change == "dependency":
        variant["dependencies"][0]["task"] = "ring_hold"
    elif change == "continuation":
        variant["execution"]["continuation_of"] = "ring_hold"
    else:
        variant["execution"]["execution_group"] = "ring_endurance"
    write(path, variant)
    with pytest.raises(ValueError, match="changes a frozen parent field"):
        load_policy_variants(sandbox, parents)


def test_optin_without_all_variants_and_retiering_cannot_shrink_denominator(parents, variants, view):
    tasks = {**parents, **variants}
    tasks.pop("ring_extension" + SUFFIX)
    with pytest.raises(ValueError, match="lacks parent/variant"):
        resolve_policy_view(view, tasks, {"task_cohort": COHORT})
    changed = deepcopy(view)
    changed["assignments"][0]["qualification_tier"] = 2
    changed["assignments"][5]["qualification_tier"] = 1
    with pytest.raises(ValueError, match="unchanged revision-3"):
        resolve_policy_view(changed, {**parents, **variants}, {"task_cohort": COHORT})


@pytest.mark.parametrize("name", ["img_bars4", "vector_two_broad", "grid100", "two_pole"])
def test_actual_full_atlas_recipe_metadata_passes_static_check_without_learning_credit(name, variants):
    recipe = recipe_for(name)
    assert policy_contract_blockers(variants[name + SUFFIX], recipe) == []
    assert policy_contract_blockers(variants[name + SUFFIX], SimpleNamespace(**recipe)) == []
    assert "status" not in variants[name + SUFFIX]


@pytest.mark.parametrize("field,value", [
    ("continuous_policy", None), ("total_steps", 600), ("lr_control", "mobility"),
    ("prior_kind", "mog"), ("standardize", True), ("row_policy", "routed_paired"),
    ("row_evidence_gate", False), ("particle_birth_death", False),
    ("row_evidence_hot", False), ("row_evidence_exclude", False), ("row_evidence_hold", False),
    ("birth_death_backend", "knn"), ("birth_death_isolation", False),
    ("serve_average", 0.), ("output_noise_mode", "fixed"), ("reopen_guard", None),
    ("amsgrad", False), ("d_lr_mult", 2.), ("betas", [0., .9]), ("prior_reg", 0.),
    ("conditioning", "ucd"), ("encoder_mode", "ae"), ("lr", float("nan")),
    ("prior_lr_mult", 0.),
])
def test_wrong_actual_recipe_controls_prior_rates_or_task_exception_block(field, value, variants):
    recipe = recipe_for("vector_two_broad")
    recipe[field] = value
    assert policy_contract_blockers(variants["vector_two_broad" + SUFFIX], recipe)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("lifecycle", "custom_loop"), ("owner", "adapter_label"),
    ("row_semantics", "conditional"), ("external_limit", "compressed_budget"),
    ("schedule", "fixed_horizon"), ("execution_path", "public_components"),
])
def test_incompatible_policy_declaration_blocks(field, value, variants):
    task = deepcopy(variants["img_bars4" + SUFFIX])
    task["execution"]["policy_contract"][field] = value
    assert policy_contract_blockers(task, recipe_for("img_bars4"))


@pytest.mark.parametrize("field,value", [
    ("weight_selector", "live"), ("sampler", "direct_G"), ("output_noise", True),
    ("latent_policy", "perturbation_disabled"), ("row_selection", "best_of_n"),
    ("eval_streams", "training"), ("diagnostic_credit", True),
])
def test_observation_cannot_override_selected_sampler_or_award_diagnostic_credit(field, value, variants):
    task = deepcopy(variants["img_bars4" + SUFFIX])
    task["evaluation"]["policy_observation"][field] = value
    assert policy_contract_blockers(task, recipe_for("img_bars4"))


@pytest.mark.parametrize("overrides", [
    {"row_evidence_gate": False}, {"d_lr_mult": float("inf")}, {"prior_reg": -1.},
    {"betas": [False, .99]}, {"betas": [0., 1.]}, {"betas": [0.]},
    {"d_lr_mult": 1.5, "betas": [0., .99], "prior_reg": 0.},
])
def test_unknown_nonfinite_or_redefined_recipe_exception_is_rejected(overrides, variants):
    task = deepcopy(variants["vector_two_broad" + SUFFIX])
    task["execution"]["policy_recipe_overrides"] = overrides
    with pytest.raises(ValueError):
        policy_recipe_overrides(task)


def test_no_c6_exception_transplant_into_other_host_and_no_reference_grade_credit(variants):
    task = deepcopy(variants["mode_hold" + SUFFIX])
    known = variants["vector_two_broad" + SUFFIX]
    task["execution"]["policy_recipe_overrides"] = deepcopy(known["execution"]["policy_recipe_overrides"])
    task["execution"]["policy_recipe_overrides_provenance"] = deepcopy(known["execution"]["policy_recipe_overrides_provenance"])
    with pytest.raises(ValueError, match="frozen host-specific"):
        policy_recipe_overrides(task)
    changed = deepcopy(known)
    changed["execution"]["policy_recipe_overrides_provenance"]["evidence_reuse"] = True
    with pytest.raises(ValueError, match="provenance"):
        policy_recipe_overrides(changed)


def test_original_live_task_cannot_be_labelled_policy_eligible(parents):
    assert not is_policy_task(parents["img_bars4"])
    assert policy_contract_blockers(parents["img_bars4"], recipe_for("img_bars4"))


def test_policy_observation_returns_isolated_validated_declaration(variants, parents):
    task = variants["grid100" + SUFFIX]
    result = validate_policy_observation(task)
    result["diagnostic_credit"] = True
    assert task["evaluation"]["policy_observation"]["diagnostic_credit"] is False
    with pytest.raises(ValueError, match="explicit policy"):
        validate_policy_observation(parents["grid100"])
    changed = deepcopy(task)
    changed["evaluation"].pop("policy_observation")
    with pytest.raises(ValueError, match="selected law"):
        validate_policy_observation(changed)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("output_noise", 0), ("diagnostic_credit", 0)])
def test_policy_boolean_and_version_fields_cannot_be_coerced(field, value, variants):
    task = deepcopy(variants["img_bars4" + SUFFIX])
    task["evaluation"]["policy_observation"][field] = value
    with pytest.raises(ValueError, match="selected law"):
        validate_policy_observation(task)
