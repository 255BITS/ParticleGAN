"""Definitions-only Atlas844 software-proof and diagnostic identity contract."""
from copy import deepcopy
import hashlib
import json

CANDIDATE_ID = "atlas-existing-mog-radius-observer844-v1"
VIEW_ID = "atlas_existing_mog_radius_observer844_v1"
TRACK_ID = "atlas844_existing_mog_radius_observer"
STUDY_ID = "atlas-existing-mog-radius-observer844-study-v1"
TASK_ID = "gaussian1d_acquisition"
TASK_DIGEST = "2ee559cdf9589c0051d463b20d03b00c28801442b9c903bcebd63bf38200ec0c"
PROOF_SCHEMA = "forge_atlas844_radius_observer_software_proof_v1"
SOFTWARE_CHECKS = ("duplicate_positive_radius", "distinct_support", "all_zero_support",
    "radius_without_effect", "clipped_and_rounded_effects", "fake_pool_coverage",
    "primary_and_isolation_coverage", "same_pair_commit", "on_off_original_calls_and_rng",
    "populated_fast_purity", "mutation_is_detected", "forbidden_getters_and_models",
    "witness_overflow", "missing_coverage_is_unknown", "scheduled_snapshot_bounds")
EXTRA_OPERATIONS = ("extra_prior_samples", "extra_rng_draws", "extra_model_forwards",
    "extra_backward_calls", "extra_optimizer_steps", "extra_decision_evaluations",
    "state_getter_calls", "state_mutations", "foreign_device_initializations")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_software_proof(request, measured_controls, source_pins):
    """Serialize ROOT's actual passing controls, never manufacture a PASS.

    This pure function reads no files and runs no controls. ROOT supplies the
    measured control report and full Queue-added admitted request inside its
    paid phase. The runtime separately verifies persisted bytes and literal pins.
    """
    from .contracts import stable_hash
    expected_checks = {name: "PASS" for name in SOFTWARE_CHECKS}
    expected_operations = {name: 0 for name in EXTRA_OPERATIONS}
    candidate = request["candidate"]
    if (measured_controls.get("schema") != "forge_atlas844_software_control_result_v1"
            or measured_controls.get("status") != "PASS"
            or measured_controls.get("checks") != expected_checks
            or measured_controls.get("added_operations") != expected_operations
            or any(type(value) is not int for value in measured_controls["added_operations"].values())
            or candidate.get("id") != CANDIDATE_ID
            or candidate.get("claim_contract", {}).get("experimental_track") != TRACK_ID
            or request.get("view", {}).get("id") != VIEW_ID
            or request.get("study", {}).get("id") != STUDY_ID
            or not isinstance(request.get("request_id"), str) or not request["request_id"]
            or set(request["tasks"]) != {TASK_ID}):
        raise ValueError("actual reviewed844 controls and full one-case admitted binding required")
    task = deepcopy(request["tasks"][TASK_ID])
    task.pop("preflight_blockers", None)
    task.pop("field_ownership", None)
    if stable_hash(task) != TASK_DIGEST:
        raise ValueError("the original immutable Gaussian task differs")
    files = request["source"]["files"]
    if stable_hash(files) != request["source"]["digest"] or not source_pins:
        raise ValueError("full copied Source index required")
    for path, pin in source_pins.items():
        if (set(pin) != {"bytes", "sha256"} or type(pin["bytes"]) is not int
                or pin["bytes"] <= 0 or files.get(path) != pin["sha256"]):
            raise ValueError("software proof Source pin differs: " + path)
    return dict(schema=PROOF_SCHEMA, status="PASS", candidate_id=CANDIDATE_ID,
        view_id=VIEW_ID, study_id=STUDY_ID, track_marker=TRACK_ID,
        request_id=request["request_id"], request_sha256=stable_hash(request),
        source_digest=request["source"]["digest"], candidate_revision=request["candidate_revision"],
        task_id=TASK_ID, task_digest=TASK_DIGEST, source_pins=deepcopy(source_pins),
        checks=expected_checks, added_operations=expected_operations)
