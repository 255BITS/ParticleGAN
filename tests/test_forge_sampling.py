"""Prospective sampler declarations cannot certify the wrong observed path."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.sampling import (
    ADAPTER_POLICIES, BEHAVIOR_POLICIES, CONDITIONAL_CENTERS, ENUMERATED_PRIOR_CLEAN,
    FIELDS, GENERATED_AND_RECONSTRUCTED, PARAMETER_MEASUREMENT, PARTICLES_AND_GRADIENT,
    PUBLIC_PRIOR_CLEAN, candidate_blockers, executed_receipt, expected_policy,
    grade_sampling, task_blockers, validate_declaration,
)
from experiments.forge.views import grade_result, load_tasks, task_evaluation_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def task():
    return {"id": "coverage", "adapter": "transfer_vector", "execution": {"steps": 24},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["error", "<=", 1.]],
                           **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")}}


def result(error=.5):
    return {"gate_status": "PASS", "evidence": {
        "observations": [{"step": i, "error": error} for i in range(1, 25)], "live": {"error": error},
        **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")}}


@pytest.mark.parametrize("error,status", [(.5, "PASS"), (2., "FAIL")])
def test_observed_contract_allows_independent_scientific_grade(error, status):
    assert grade_result(task(), result(error))["status"] == status


@pytest.mark.parametrize("field", FIELDS)
def test_missing_observation_cannot_be_inferred_from_task_or_claim(field):
    raw = result()
    raw["claim_contract"] = {"sampling_law": "task_declared"}
    del raw["evidence"][field]
    assert grade_result(task(), raw)["status"] == "INCOMPLETE"


@pytest.mark.parametrize("field,value", [("sampling_contract_version", True),
    ("sampling_contract_version", 2), ("sampling_law", ENUMERATED_PRIOR_CLEAN),
    ("sampling_law", "public_noisy"), ("eval_output_noise", "public_recipe_schedule")])
def test_contradictory_observed_policy_is_invalid_not_a_scientific_failure(field, value):
    raw = result(2.)
    raw["evidence"][field] = value
    assert grade_result(task(), raw)["status"] == "INVALID"


@pytest.mark.parametrize("historical_law", [None, PUBLIC_PRIOR_CLEAN, "public_noisy"])
def test_archived_unversioned_receipts_keep_original_grade_and_bytes(historical_law):
    frozen_task, raw = task(), result(2.)
    del frozen_task["evaluation"]["sampling_contract_version"]
    if historical_law is None:
        frozen_task["evaluation"].pop("sampling_law")
    else:
        frozen_task["evaluation"]["sampling_law"] = historical_law
    raw["claim_contract"] = {"sampling_law": "public_noisy"}
    for field in FIELDS:
        raw["evidence"].pop(field)
    before = deepcopy((frozen_task, raw))
    assert grade_result(frozen_task, raw)["status"] == "FAIL"
    assert (frozen_task, raw) == before
    assert task_blockers(frozen_task)


@pytest.mark.parametrize("version", [None, True, 0, 2, "1"])
def test_invalid_explicit_version_never_falls_back_to_archive_semantics(version):
    spec = task()
    spec["evaluation"]["sampling_contract_version"] = version
    assert grade_result(spec, result())["status"] == "INVALID"
    assert task_blockers(spec)


def test_supported_string_for_wrong_host_still_blocks():
    spec = task()
    spec["evaluation"].update(executed_receipt(ENUMERATED_PRIOR_CLEAN, eval_output_noise="clean"))
    assert task_blockers(spec)
    raw = result()
    raw["evidence"].update(executed_receipt(ENUMERATED_PRIOR_CLEAN, eval_output_noise="clean"))
    assert grade_result(spec, raw)["status"] == "INVALID"


def test_unimplemented_host_is_never_enabled_by_task_delegation():
    spec = task()
    spec["adapter"] = "fixture"
    assert "fixture" not in ADAPTER_POLICIES
    assert not candidate_blockers({"claim_contract": {"sampling_law": "task_declared"}})
    assert task_blockers(spec)


@pytest.mark.parametrize("claims", [{}, {"sampling_law": "public_noisy"},
    {"sampling_law": PUBLIC_PRIOR_CLEAN}, {"sampling_law": "unknown"}, None])
def test_new_candidate_claims_must_delegate_explicitly(claims):
    assert candidate_blockers({"claim_contract": claims})


def test_execution_helper_requires_explicit_known_policy_and_noise_pair():
    with pytest.raises(ValueError, match="unsupported"):
        executed_receipt("public_noisy", eval_output_noise="clean")
    with pytest.raises(ValueError, match="requires eval_output_noise"):
        executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="public_recipe_schedule")


def test_all_current_tasks_opt_in_without_erasing_host_exceptions():
    tasks = load_tasks(ROOT)
    for spec in tasks.values():
        assert validate_declaration(spec, required=True) == expected_policy(spec)
        assert not task_blockers(spec)
    assert expected_policy(tasks["img_intensity2"])["sampling_law"] == ENUMERATED_PRIOR_CLEAN
    assert expected_policy(tasks["mode_hold"])["sampling_law"] == PUBLIC_PRIOR_CLEAN
    for name in ("trajectory", "residual_student"):
        assert expected_policy(tasks[name])["sampling_law"] == CONDITIONAL_CENTERS
    assert expected_policy(tasks["ae_gan_hold"])["sampling_law"] == GENERATED_AND_RECONSTRUCTED
    assert expected_policy(tasks["two_pole"])["sampling_law"] == PARTICLES_AND_GRADIENT
    for name in ("unipolar", "unused_token_hold", "cover_leftover", "mid_scale_identity"):
        assert expected_policy(tasks[name])["sampling_law"] == PARAMETER_MEASUREMENT
        assert expected_policy(tasks[name])["eval_output_noise"] == "not_applied_to_measurement"
    # This contract controls output measurement; it never changes the prior law.
    assert tasks["ae_gan_hold"]["execution"]["prior"]["sigma"] == .025
    assert tasks["mode_hold"]["execution"]["prior"]["sigma"] == .025
    assert tasks["img_intensity2"]["execution"]["prior"]["sigma"] == 0


def test_versioned_contract_enters_evaluator_identity():
    original = task()
    legacy = deepcopy(original)
    legacy["evaluation"].pop("sampling_contract_version")
    assert task_evaluation_fingerprint(original) != task_evaluation_fingerprint(legacy)
