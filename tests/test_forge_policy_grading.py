"""Synthetic receipt controls only: no training or qualification evidence.

Every positive curve below is invented software input, explicitly labelled as
such. The tests exercise numeric receipt grading and prospective policy proof
boundaries; they never evaluate a sample cloud or deserialize a checkpoint.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.forge.policy_contracts import (
    COHORT, PARENT_TASK_IDS, SUFFIX, load_policy_variants, resolve_policy_view,
)
from experiments.forge.policy_adapters import HOOKS
from experiments.forge.sampling import FIELDS, grade_sampling
from experiments.forge.views import grade_result, qualify


ROOT = Path(__file__).resolve().parents[1]
POLICY_OWNERS = (
    "continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
    "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard",
)


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def declarations():
    parents = {name: json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())
               for name in PARENT_TASK_IDS}
    return parents, load_policy_variants(ROOT, parents)


@pytest.fixture
def task(declarations):
    return deepcopy(declarations[1]["img_intensity2" + SUFFIX])


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def receipt(task, *, metrics=None):
    """A complete, internally consistent *synthetic* selected-policy receipt."""
    horizon = task["execution"]["steps"]
    if metrics is None:
        metrics = {name: bound for name, _, bound in task["evaluation"]["thresholds"]}
    steps = [math.ceil(i * horizon / 24) for i in range(1, 25)]
    observations = [{"step": step, **deepcopy(metrics)} for step in steps]
    policies, purities = [], []
    for step in steps:
        policies.append({**deepcopy(task["evaluation"]["policy_observation"]),
            "observed": True, "policy_owner": "particlegan.UpdatePolicy",
            "completed_steps": step, "selected_source": "fast", "controller": "dv12",
            "backend_selection": {"backend": "knn", "reason": "synthetic software input"},
            "snapshot_sha256": sha(f"synthetic-selected-snapshot-{step}")})
        purities.append({"completed_steps": step, "digest_kind": "typed_policy_state_v1",
            "before_sha256": sha(f"synthetic-training-state-{step}"),
            "after_sha256": sha(f"synthetic-training-state-{step}"), "pure": True,
            "allowed_changes": "independent named evaluation RNG streams only"})
    evidence = {
        **{key: task["evaluation"][key] for key in FIELDS},
        "scoring_weights": "state_selected", "observations": observations,
        "live": deepcopy(metrics), "policy_observation": deepcopy(policies[-1]),
        "policy_observations": policies, "policy_purity": purities,
        "checkpoint_digest_kind": "typed_policy_state_v1",
        "policy_controls": {
            "schema_version": 1, "cohort": COHORT, "row_semantics": "independent",
            "completed_steps": horizon, "requested": dict.fromkeys(POLICY_OWNERS, True),
            "enabled": dict.fromkeys(POLICY_OWNERS, True), "requested_owners_bound": True,
            "implementation_observed": True, "served_source": "fast", "output_sigma": .029,
            "execution": {"model_devices": ["cuda:0"], "floating_dtypes": ["torch.float32"],
                          "autocast_enabled": False},
            "row_evidence_observations": horizon, "roles": ["generator", "prior", "discriminator"],
            "effective_group_lrs": [[.0053125, .00796875], [.0053125]],
            "quality_qualification": False,
            "lifecycle": {
                "owner": "particlegan.UpdatePolicy", "kind": "instance_local_successful_public_hooks",
                "start_completed_steps": 0, "end_completed_steps": horizon,
                "observed_updates": horizon, "calls": dict.fromkeys(HOOKS, horizon),
                "last_order": list(HOOKS), "pending": [], "order_errors": 0, "complete": True,
            },
        },
        "guards": {"all_finite": True, "optimizer_updates": {
            "generator": horizon, "discriminator": horizon, "prior": horizon, "encoder": horizon},
            "hooks_exercised": True, "unintended_rng_deviations": 0,
            "mechanism_audit": {"schema_version": 1, "mechanisms": {
                "critic_penalty": {"requested": True, "enabled": True, "calls": horizon, "eligible": horizon, "applied": horizon},
                **{name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                   for name in ("critic_anchor", "critic_guard", "a2", "direct_particle_gain")},
            }}},
    }
    return {"task_id": task["id"], "gate_status": "PASS", "evidence": evidence,
            "software_fixture_only": True, "cost": {"seconds": 0}}


def original_receipt(task, *, metrics=None):
    metrics = metrics or {name: bound for name, _, bound in task["evaluation"]["thresholds"]}
    horizon = task["execution"]["steps"]
    return {"task_id": task["id"], "gate_status": "PASS", "software_fixture_only": True,
        "evidence": {**{key: task["evaluation"][key] for key in FIELDS}, "scoring_weights": "live",
            "live": deepcopy(metrics), "observations": [
                {"step": math.ceil(i * horizon / 24), **deepcopy(metrics)} for i in range(1, 25)]}}


def grade(task, raw):
    before = deepcopy(raw)
    result = grade_result(task, raw)
    assert raw == before  # No rewriting of the original receipt by the grader.
    return result["status"]


def test_selected_policy_numeric_bounds_are_exact_and_not_the_pass_stamp(task):
    raw = receipt(task)
    assert grade(task, raw) == "PASS"
    raw["gate_status"] = "FAIL"
    assert grade(task, raw) == "PASS"
    raw["gate_status"] = "PASS"
    raw["evidence"]["live"]["hq"] = math.nextafter(.9, 0.)
    assert grade(task, raw) == "FAIL"
    raw = receipt(task)
    raw["evidence"]["live"]["modes"] = 1
    assert grade(task, raw) == "FAIL"


def test_selected_image_tv_is_diagnostic_not_a_new_numeric_exemption_or_gate(task):
    raw = receipt(task, metrics={"modes": 2, "hq": .95, "mass_tv": .99})
    assert task["evaluation"]["thresholds"] == [["modes", ">=", 2], ["hq", ">=", .9]]
    assert grade(task, raw) == "PASS"


@pytest.mark.parametrize("bad_index,expected", [(18, "PASS"), (19, "FAIL"), (23, "FAIL")])
def test_unchanged_five_observation_terminal_suffix_is_required(task, bad_index, expected):
    raw = receipt(task)
    raw["evidence"]["observations"][bad_index]["hq"] = .89
    assert grade(task, raw) == expected


@pytest.mark.parametrize("name", ["vector_two_broad", "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap", "vector_spiral"])
def test_each_scalar_bound_is_regraded_without_sample_clouds(name, declarations):
    task = declarations[1][name + SUFFIX]
    assert grade(task, receipt(task)) == "PASS"
    for name, op, bound in task["evaluation"]["thresholds"]:
        raw = receipt(task)
        raw["evidence"]["live"][name] = math.nextafter(float(bound), math.inf if op == "<=" else -math.inf)
        assert grade(task, raw) == "FAIL"


@pytest.mark.parametrize("field", ["policy_controls", "policy_purity", "policy_observations", "policy_observation"])
def test_missing_lifecycle_or_measurement_receipts_are_incomplete(task, field):
    raw = receipt(task)
    del raw["evidence"][field]
    assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("field", ["lifecycle", "requested", "enabled", "completed_steps"])
def test_missing_owner_or_update_evidence_is_incomplete(task, field):
    raw = receipt(task)
    del raw["evidence"]["policy_controls"][field]
    assert grade(task, raw) == "INCOMPLETE"


def test_missing_lifecycle_owner_is_incomplete(task):
    raw = receipt(task)
    del raw["evidence"]["policy_controls"]["lifecycle"]["owner"]
    assert grade(task, raw) == "INCOMPLETE"


def test_missing_measurement_owner_or_snapshot_is_incomplete(task):
    for field in ("policy_owner", "snapshot_sha256", "observed"):
        raw = receipt(task)
        del raw["evidence"]["policy_observation"][field]
        assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("owner", POLICY_OWNERS)
def test_requested_control_without_enabled_owner_is_blocked(task, owner):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["enabled"][owner] = False
    raw["evidence"]["policy_controls"]["requested_owners_bound"] = False
    assert grade(task, raw) == "BLOCKED"


def test_a_label_cannot_replace_all_required_owner_keys(task):
    raw = receipt(task)
    controls = raw["evidence"]["policy_controls"]
    controls["requested"] = {"unrelated_label": False}
    controls["enabled"] = {"unrelated_label": False}
    assert grade(task, raw) == "INVALID"


def test_required_mechanisms_cannot_be_declared_unrequested_to_avoid_binding(task):
    raw = receipt(task)
    controls = raw["evidence"]["policy_controls"]
    controls["requested"]["birth_death"] = False
    controls["enabled"]["birth_death"] = False
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("owner", ["adapter_label", "other.UpdatePolicy"])
def test_wrong_owner_is_invalid(task, owner):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["lifecycle"]["owner"] = owner
    assert grade(task, raw) == "INVALID"
    raw = receipt(task)
    raw["evidence"]["policy_observation"]["policy_owner"] = owner
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [
    ("last_order", list(reversed(HOOKS))), ("pending", ["begin_step"]), ("order_errors", 1),
    ("observed_updates", 599), ("end_completed_steps", 599), ("complete", False),
])
def test_wrong_hook_order_or_update_count_is_invalid(task, field, value):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["lifecycle"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("hook", HOOKS)
def test_each_public_hook_must_cover_every_real_update(task, hook):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["lifecycle"]["calls"][hook] = 599
    assert grade(task, raw) == "INVALID"


def test_coherently_shorter_lifecycle_cannot_certify_a_full_terminal_curve(task):
    raw = receipt(task)
    controls = raw["evidence"]["policy_controls"]
    controls["completed_steps"] = 300
    controls["lifecycle"].update(end_completed_steps=300, observed_updates=300, calls=dict.fromkeys(HOOKS, 300))
    assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("where", ["final", "curve", "purity"])
def test_measurement_clock_must_match_actual_observation_schedule(task, where):
    raw = receipt(task)
    if where == "final":
        raw["evidence"]["policy_observation"]["completed_steps"] = 599
    elif where == "curve":
        raw["evidence"]["policy_observations"][7]["completed_steps"] = 599
    else:
        raw["evidence"]["policy_purity"][7]["completed_steps"] = 599
    assert grade(task, raw) == "INVALID"


def test_single_measurement_pair_cannot_cover24_scored_checkpoints(task):
    raw = receipt(task)
    raw["evidence"]["policy_observations"] = raw["evidence"]["policy_observations"][-1:]
    raw["evidence"]["policy_purity"] = raw["evidence"]["policy_purity"][-1:]
    assert grade(task, raw) == "INCOMPLETE"


def test_duplicate_measurement_pair_cannot_hide_missing_checkpoint(task):
    raw = receipt(task)
    raw["evidence"]["policy_observations"][7] = deepcopy(raw["evidence"]["policy_observations"][6])
    raw["evidence"]["policy_purity"][7] = deepcopy(raw["evidence"]["policy_purity"][6])
    assert grade(task, raw) == "INVALID"


def test_reordered_policy_proofs_cannot_borrow_an_ordered_numeric_curve(task):
    raw = receipt(task)
    for field in ("policy_observations", "policy_purity"):
        values = raw["evidence"][field]
        values[6], values[7] = values[7], values[6]
    assert grade(task, raw) == "INVALID"


def test_a_midrun_selected_snapshot_cannot_certify_the_final_policy(task):
    raw = receipt(task)
    for field in ("policy_observations", "policy_purity"):
        raw["evidence"][field][-1] = deepcopy(raw["evidence"][field][6])
    raw["evidence"]["policy_observation"] = deepcopy(raw["evidence"]["policy_observations"][-1])
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("pure", False), ("after_sha256", sha("changed-optimizer-or-training-rng"))])
def test_changed_training_state_or_rng_is_invalid(task, field, value):
    raw = receipt(task)
    raw["evidence"]["policy_purity"][3][field] = value
    assert grade(task, raw) == "INVALID"


def test_missing_exact_typed_purity_identity_is_incomplete(task):
    raw = receipt(task)
    del raw["evidence"]["policy_purity"][3]["before_sha256"]
    assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("deviations", [1, 2])
def test_global_or_named_training_rng_deviation_cannot_hide_behind_equal_context_hashes(task, deviations):
    raw = receipt(task)
    raw["evidence"]["guards"]["unintended_rng_deviations"] = deviations
    assert grade(task, raw) == "INVALID"


def test_producer_nonfinite_state_guard_cannot_be_ignored_on_an_image_parent(task):
    raw = receipt(task)
    raw["evidence"]["guards"]["all_finite"] = False
    assert grade(task, raw) == "FAIL"


@pytest.mark.parametrize("field", ["all_finite", "unintended_rng_deviations"])
def test_missing_health_or_rng_observation_is_incomplete(task, field):
    raw = receipt(task)
    del raw["evidence"]["guards"][field]
    assert grade(task, raw) == "INCOMPLETE"


def test_sampling_boundary_does_not_infer_executed_overlay_from_declared_task(task):
    raw = receipt(task)
    assert grade_sampling(task, raw["evidence"]) is None
    del raw["evidence"]["policy_observation"]
    assert grade_sampling(task, raw["evidence"])["status"] == "INCOMPLETE"


def test_wrong_valid_snapshot_digest_cannot_borrow_final_measurement_identity(task):
    raw = receipt(task)
    raw["evidence"]["policy_observation"]["snapshot_sha256"] = sha("another-source-snapshot")
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [
    ("weight_selector", "live"), ("weight_selector", "forced_ema"), ("sampler", "direct_G"),
    ("output_noise", True), ("latent_policy", "disabled"), ("eval_streams", "training"),
    ("row_selection", "best_checkpoint"), ("diagnostic_credit", True),
    ("selected_source", "forced_ema"), ("controller", "dv11"),
])
def test_wrong_measurement_law_or_diagnostic_qualification_is_invalid(task, field, value):
    raw = receipt(task)
    raw["evidence"]["policy_observation"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("output_noise", 0), ("diagnostic_credit", 0), ("schema_version", True)])
def test_observed_contract_does_not_coerce_false_and_zero_or_schema_bool(task, field, value):
    raw = receipt(task)
    raw["evidence"]["policy_observation"][field] = value
    assert grade(task, raw) == "INVALID"


def test_final_selected_source_must_match_lifecycle_and_last_observation(task):
    raw = receipt(task)
    raw["evidence"]["policy_observation"]["selected_source"] = "averaged"
    assert grade(task, raw) == "INVALID"
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["served_source"] = "averaged"
    assert grade(task, raw) == "INVALID"


def test_all_averaged_primary_measurements_can_pass_without_forcing_ema(task):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["served_source"] = "averaged"
    for measurement in raw["evidence"]["policy_observations"]:
        measurement["selected_source"] = "averaged"
    raw["evidence"]["policy_observation"] = deepcopy(raw["evidence"]["policy_observations"][-1])
    assert grade(task, raw) == "PASS"


def test_cross_cohort_controls_and_conditional_row_claim_are_invalid(task):
    for field, value in (("cohort", "original_live_mog"), ("row_semantics", "conditional")):
        raw = receipt(task)
        raw["evidence"]["policy_controls"][field] = value
        assert grade(task, raw) == "INVALID"


def test_cpu_execution_cannot_supply_gpu_numerical_credit(task):
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["execution"]["model_devices"] = ["cpu"]
    assert grade(task, raw) == "INVALID"


def test_unreported_or_changed_precision_is_not_a_gpu_pass(task):
    raw = receipt(task)
    del raw["evidence"]["policy_controls"]["execution"]
    assert grade(task, raw) == "INCOMPLETE"
    raw = receipt(task)
    raw["evidence"]["policy_controls"]["execution"]["autocast_enabled"] = True
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field", FIELDS)
def test_original_sampling_fields_are_still_required(task, field):
    raw = receipt(task)
    del raw["evidence"][field]
    assert grade(task, raw) == "INCOMPLETE"


def test_forced_ema_success_cannot_rescue_failed_primary_selected_metrics(task):
    raw = receipt(task)
    raw["evidence"]["live"]["hq"] = .89
    raw["evidence"]["ema"] = {"modes": 2, "hq": 1.}
    raw["evidence"]["forced_ema_verdict"] = "PASS"
    assert grade(task, raw) == "FAIL"
    raw["evidence"]["scoring_weights"] = "ema"
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("name", ["img_intensity2", "vector_two_broad"])
def test_original_live_and_mog_grades_need_no_new_policy_receipts(name, declarations):
    original = declarations[0][name]
    raw = original_receipt(original)
    assert grade(original, raw) == "PASS"
    key, op, bound = original["evaluation"]["thresholds"][0]
    raw["evidence"]["live"][key] = math.nextafter(float(bound), math.inf if op == "<=" else -math.inf)
    assert grade(original, raw) == "FAIL"
    raw["evidence"]["scoring_weights"] = "state_selected"
    assert grade(original, raw) == "INVALID"


def test_legacy_receipt_cannot_fill_any_slot_of_prospective_main_view(declarations):
    parents, variants = declarations
    view = json.loads((ROOT / "configs/forge/views/discriminator_stability.json").read_text())
    resolved, tasks = resolve_policy_view(view, {**parents, **variants}, {"task_cohort": COHORT})
    old = original_receipt(parents["img_intensity2"])
    board = qualify(resolved, tasks, [old])
    assert board["required_total"] == 26 and board["required_passed"] == 0
    assert not board["eligible"]
    assert board["task_statuses"]["img_intensity2" + SUFFIX] == "NOT_RUN"


@pytest.mark.parametrize("mutation,status", [
    ("missing", "INCOMPLETE"), ("duplicate", "INVALID"), ("nonfinite", "INVALID"),
])
def test_numeric_curve_requires_unchanged24_unique_finite_points(task, mutation, status):
    raw = receipt(task)
    if mutation == "missing":
        raw["evidence"]["observations"].pop()
    elif mutation == "duplicate":
        raw["evidence"]["observations"].append(deepcopy(raw["evidence"]["observations"][-1]))
    else:
        raw["evidence"]["observations"][3]["hq"] = math.inf
    assert grade(task, raw) == status


def test_native_full_numeric_contract_is_preserved_before_any_file_based_grader(declarations):
    for name in ("grid100", "rotated100", "staggered100"):
        original, variant = declarations[0][name], declarations[1][name + SUFFIX]
        for field in ("coverage_thresholds", "accuracy_limits", "eval_interval", "early_eval_steps",
                      "eval_samples", "holdout_samples", "minimum_stable_checks"):
            assert variant["evaluation"][field] == original["evaluation"][field]
        assert variant["evaluation"]["eval_samples"] == 20000
        assert variant["evaluation"]["holdout_samples"] == 100000
        assert variant["evaluation"]["coverage_thresholds"]["min_modes"] == 100
        assert variant["evaluation"]["accuracy_limits"]["radial_ks"] == .04
        assert variant["evaluation"]["policy_observation"]["diagnostic_weights"] == ["forced_ema"]
