"""Exact state certificates for a separate failure-eligible extension.

This never invokes or weakens ordinary passing-parent continuation. The caller
must restore the state with the existing public host before certifying it here.
"""
from pathlib import Path
import torch

from .contracts import file_hash, stable_hash
from .state import state_digest, require_optimizer_steps, require_same_formulation


def _clock(state, schema):
    if schema == "public_context_v1":
        trainer = state["trainer"]
        step = trainer["completed_steps"]
        horizon = state["recipe"]["total_steps"]
    elif schema == "joint_word_v1":
        trainer = state["fixture"]["api_state"]
        step = trainer["completed_steps"]
        horizon = state["fixture"]["recipe"]["total_steps"]
    else:
        raise ValueError("unsupported resumable checkpoint schema")
    require_optimizer_steps({"trainer": trainer}, step)
    return step, horizon


def _read_certified(checkpoint, path):
    if file_hash(Path(path)) != checkpoint["sha256"]:
        raise ValueError("certified checkpoint bytes differ")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_digest(state) != checkpoint["state_sha256"]:
        raise ValueError("checkpoint state digest differs")
    return state


def verify_extension_prefix(policy, original_result, checkpoint_path, restored_path):
    if (policy["kind"] != "failure_eligible_training_extension_v1"
            or stable_hash(original_result) != policy["original_result_sha256"]
            or original_result["checkpoint"] != policy["checkpoint"]):
        raise ValueError("extension policy is not bound to immutable original result")
    if original_result["gate_status"] not in {"PASS", "FAIL", "INCOMPLETE"}:
        raise ValueError("unsupported original grade")
    checkpoint = policy["checkpoint"]
    original = _read_certified(checkpoint, checkpoint_path)
    restored = torch.load(restored_path, map_location="cpu", weights_only=True)
    if state_digest(original) != state_digest(restored):
        raise ValueError("extension did not restore every model/optimizer/RNG/policy state exactly")
    step, horizon = _clock(original, checkpoint["schema"])
    if step != checkpoint["step"] or horizon != policy["original_schedule_horizon"]:
        raise ValueError("earlier checkpoint step or original schedule horizon differs")
    return dict(restored_exactly=True, history_reset=False, checkpoint_step=step,
                policy_sha256=stable_hash(policy), checkpoint_sha256=checkpoint["sha256"],
                checkpoint_state_sha256=state_digest(original),
                restored_state_sha256=state_digest(restored),
                ordinary_passing_parent_dependency_weakened=False,
                original_grade=original_result["gate_status"])


def verify_extension_final(policy, parent_path, final_checkpoint, final_path):
    original = _read_certified(policy["checkpoint"], parent_path)
    final = _read_certified(final_checkpoint, final_path)
    schema = policy["checkpoint"]["schema"]
    step, horizon = _clock(final, schema)
    if step != final_checkpoint["step"] or horizon != policy["original_schedule_horizon"]:
        raise ValueError("final accepted update count or schedule horizon differs")
    if schema == "public_context_v1":
        require_same_formulation(original, final)
    else:
        left, right = original["fixture"], final["fixture"]
        for key in ("version", "case", "seed", "recipe", "api_components"):
            if left[key] != right[key]:
                raise ValueError("word extension changed static formulation")
        if original["streams"]["manifest"] != final["streams"]["manifest"]:
            raise ValueError("word extension changed named RNG bindings")
    return dict(final_state_verified=True, final_checkpoint=final_checkpoint)
