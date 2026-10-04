"""Versioned evaluation sampling contracts, separate from training noise/prior width.

Version 1 is required for new planning. An archived task without the version
marker keeps its frozen grading semantics; a historical label is never upgraded
or interpreted as an alias. Policies describe the executed measurement path,
including parameter-only hosts and the clock probe's evaluation-cadence branch.
"""
from __future__ import annotations

VERSION = 1
FIELDS = ("sampling_contract_version", "sampling_law", "eval_output_noise")
PUBLIC_PRIOR_CLEAN = "public_prior_without_output_noise"
ENUMERATED_PRIOR_CLEAN = "enumerated_prior_without_output_noise"
CONDITIONAL_CENTERS = "conditional_prior_centers_with_scheduled_output_noise"
GENERATED_AND_RECONSTRUCTED = "generated_and_reconstructed_prior_with_scheduled_output_noise"
JOINT_WORDS_CLEAN = "generated_and_paired_reconstructed_prior_without_output_noise"
PARTICLES_AND_GRADIENT = "learned_particles_and_critic_gradient"
PARAMETER_MEASUREMENT = "learned_parameter_measurement"
AE_ROUTED_RECONSTRUCTION = "ae_routed_selected_mog_reconstruction"

# A scheduled policy remains the same policy when its configured amplitude is
# zero. Prior kernels remain governed by execution.prior, never by this field.
POLICIES = {
    PUBLIC_PRIOR_CLEAN: "clean",
    ENUMERATED_PRIOR_CLEAN: "clean",
    CONDITIONAL_CENTERS: "public_recipe_schedule",
    GENERATED_AND_RECONSTRUCTED: "public_recipe_schedule",
    JOINT_WORDS_CLEAN: "clean",
    PARTICLES_AND_GRADIENT: "not_applied_to_measurement",
    PARAMETER_MEASUREMENT: "not_applied_to_measurement",
    # This named routed AE observes the complete selected hard-AE/MoG
    # function without an evaluation DV12 or output-noise draw.  Its parent
    # retains GENERATED_AND_RECONSTRUCTED and its scheduled-noise contract.
    AE_ROUTED_RECONSTRUCTION: "clean_complete_routed_function",
}
ADAPTER_POLICIES = {
    "word_joint": JOINT_WORDS_CLEAN,
    "transfer_vector": PUBLIC_PRIOR_CLEAN,
    "transfer_image": ENUMERATED_PRIOR_CLEAN,
    "native100": PUBLIC_PRIOR_CLEAN,
    "native100_continuation": PUBLIC_PRIOR_CLEAN,
    "ring_endurance": PUBLIC_PRIOR_CLEAN,
    "paired_adaptation": PUBLIC_PRIOR_CLEAN,
    # The metric compares state. This law describes the actual samples drawn in
    # its evaluation-cadence perturbation, not a distribution-quality score.
    "clockfree_audit": PUBLIC_PRIOR_CLEAN,
}
BEHAVIOR_POLICIES = {
    "mode_hold": PUBLIC_PRIOR_CLEAN,
    "two_pole": PARTICLES_AND_GRADIENT,
    "trajectory": CONDITIONAL_CENTERS,
    "residual_student": CONDITIONAL_CENTERS,
    "ae_gan_hold": GENERATED_AND_RECONSTRUCTED,
    "unipolar": PARAMETER_MEASUREMENT,
    "cover_leftover": PARAMETER_MEASUREMENT,
    "unused_token_hold": PARAMETER_MEASUREMENT,
    "mid_scale_identity": PARAMETER_MEASUREMENT,
}


def _policy_observation_contract(task):
    """Fail closed on partial declarations and dispatch only known cohorts."""
    from .policy_cohorts import is_policy_task, validate_policy_observation
    evaluation = task.get("evaluation", {})
    declared = (is_policy_task(task) or task.get("task_cohort") is not None
                or task.get("policy_family") is not None or task.get("policy_parent") is not None
                or "policy_observation" in evaluation)
    if not declared:
        return None
    return validate_policy_observation(task)


def executed_receipt(sampling_law: str, *, eval_output_noise: str) -> dict:
    """Attest an explicitly chosen executed path; never copy task declarations.

    Call at the adapter branch that actually selected the sampler/measurement.
    This records its policy; RNG/state isolation remains independently audited.
    """
    if not isinstance(sampling_law, str) or sampling_law not in POLICIES:
        raise ValueError(f"unsupported evaluation sampling law: {sampling_law!r}")
    if eval_output_noise != POLICIES[sampling_law]:
        raise ValueError(f"sampling law {sampling_law!r} requires eval_output_noise={POLICIES[sampling_law]!r}")
    return {"sampling_contract_version": VERSION, "sampling_law": sampling_law,
            "eval_output_noise": eval_output_noise}


def expected_policy(task: dict) -> dict:
    """One central binding of implemented adapter/host paths to their policies."""
    observation = _policy_observation_contract(task)
    if observation is not None and task["task_cohort"] == "ae_routed_policy_v1":
        return executed_receipt(AE_ROUTED_RECONSTRUCTION,
                                eval_output_noise=POLICIES[AE_ROUTED_RECONSTRUCTION])
    adapter = task.get("adapter")
    if adapter == "transfer_behavior":
        host = task.get("execution", {}).get("host", task.get("id"))
        law = BEHAVIOR_POLICIES.get(host)
    else:
        law = ADAPTER_POLICIES.get(adapter)
    if law is None:
        raise ValueError(f"{task.get('id', '<task>')}: no supported evaluation sampling policy for adapter/host")
    return executed_receipt(law, eval_output_noise=POLICIES[law])


def validate_declaration(task: dict, *, required: bool = False) -> dict | None:
    """Validate an opt-in declaration without changing archived task semantics."""
    evaluation = task.get("evaluation", {})
    policy = _policy_observation_contract(task)
    if "sampling_contract_version" not in evaluation:
        if required or policy is not None:
            raise ValueError("new planning requires evaluation.sampling_contract_version=1")
        return None
    version = evaluation["sampling_contract_version"]
    if type(version) is not int or version != VERSION:
        raise ValueError(f"unsupported sampling_contract_version: {version!r}")
    missing = set(FIELDS) - evaluation.keys()
    if missing:
        raise ValueError(f"sampling contract lacks {sorted(missing)}")
    return executed_receipt(evaluation["sampling_law"], eval_output_noise=evaluation["eval_output_noise"])


def candidate_blockers(candidate: dict) -> list[str]:
    claims = candidate.get("claim_contract", {})
    if not isinstance(claims, dict) or claims.get("sampling_law") != "task_declared":
        return ["new planning requires claim_contract.sampling_law='task_declared'; evaluation policies belong to each task"]
    return []


def task_blockers(task: dict) -> list[str]:
    from .paired_sampling import paired_sampling_blockers
    paired_blockers = paired_sampling_blockers(task)
    if paired_blockers:
        return paired_blockers
    try:
        declared = validate_declaration(task, required=True)
        if declared != expected_policy(task):
            raise ValueError("evaluation sampling declaration differs from the implemented adapter/host policy")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return [f"{task.get('id', '<task>')}: {exc}"]
    return []


def grade_sampling(task: dict, evidence: dict) -> dict | None:
    """Return only contract errors; the scientific grader still judges metrics."""
    try:
        contract = _policy_observation_contract(task)
        declared = validate_declaration(task)
        if declared is None:
            return None
        if declared != expected_policy(task):
            raise ValueError("evaluation sampling declaration differs from the implemented adapter/host policy")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {"status": "INVALID", "reason": str(exc)}
    missing = set(FIELDS) - evidence.keys()
    if missing:
        return {"status": "INCOMPLETE", "reason": f"missing observed sampling contract fields: {sorted(missing)}"}
    if type(evidence["sampling_contract_version"]) is not int or any(
            evidence[key] != declared[key] for key in FIELDS):
        return {"status": "INVALID", "reason": "observed sampling policy differs from the frozen task contract"}
    if contract is not None:
        observed = evidence.get("policy_observation")
        if not isinstance(observed, dict):
            return {"status": "INCOMPLETE", "reason": "missing observed selected-policy measurement contract"}
        if any(key not in observed or type(observed[key]) is not type(value)
               or observed[key] != value for key, value in contract.items()):
            return {"status": "INVALID", "reason": "observed policy weights, sampler, noise or diagnostics differ from the frozen contract"}
        if observed.get("observed") is not True:
            return {"status": "INCOMPLETE", "reason": "selected-policy measurement was not observed"}
        if "policy_owner" not in observed or "selected_source" not in observed:
            return {"status": "INCOMPLETE", "reason": "measurement lacks its actual public selected-policy owner/source"}
        if (observed.get("policy_owner") != "particlegan.UpdatePolicy"
                or observed.get("selected_source") not in {"fast", "averaged"}):
            return {"status": "INVALID", "reason": "measurement lacks its actual public selected-policy owner/source"}
        if "controller" not in observed:
            return {"status": "INCOMPLETE", "reason": "measurement lacks its actual Atlas controller identity"}
        if observed["controller"] != "dv12":
            return {"status": "INVALID", "reason": "measurement controller differs from this declared Atlas policy"}
        execution_contract = task["execution"]["policy_contract"]
        if task["task_cohort"] in {"word_joint_policy_min11_v1", "word_joint_policy_min11_rates_v1"}:
            if "family" not in observed:
                return {"status": "INCOMPLETE", "reason": "word measurement lacks its actual min11 joint-cloud family"}
            if observed["family"] != execution_contract["family"]:
                return {"status": "INVALID", "reason": "word measurement borrowed another family or five-row law"}
        if execution_contract["row_semantics"] == "conditional":
            for field, expected in (("family", execution_contract["family"]),
                                    ("row_policy", execution_contract["row_policy"])):
                if field not in observed:
                    return {"status": "INCOMPLETE", "reason": "measurement lacks its actual named routed family/row owner"}
                if observed[field] != expected:
                    return {"status": "INVALID", "reason": "measurement borrowed another family or row law"}
            if task["task_cohort"] == "conditional_policy_selected_cloud_v1":
                import math
                sigma = observed.get("output_sigma_used")
                if (type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0
                        or "latent_perturbation_applied" not in observed):
                    return {"status": "INCOMPLETE", "reason": "conditional measurement lacks its actual noise/latent read"}
                if (observed["latent_perturbation_applied"] is not False
                        or contract["output_noise"] is False and sigma != 0):
                    return {"status": "INVALID", "reason": "conditional measurement contradicts its original clean/enumerated law"}
        digest = observed.get("snapshot_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(letter not in "0123456789abcdef" for letter in digest)):
            return {"status": "INCOMPLETE", "reason": "measurement lacks a selected-policy snapshot identity"}
    return None


def validate_request_sampling(request: dict, *, task_ids=None) -> None:
    """Recompute prospective policies at queue/dispatch boundaries.

    The verified execution source, not a removable request flag, sets the
    version floor. Historical snapshots without this module keep their frozen
    execution contract. A new snapshot cannot downgrade by deleting its source
    manifest entry: verification also rejects undeclared files on disk.

    Queue callers check all authorized grouped members. Runtime callers pass
    every member of the job immediately before executing its adapter.
    """
    from pathlib import Path
    from .sources import verify_snapshot

    module = "experiments/forge/sampling.py"
    source = request.get("source", {})
    declared = module in source.get("files", {})
    snapshot = source.get("snapshot_path")
    present = isinstance(snapshot, str) and bool(snapshot) and (Path(snapshot) / module).is_file()
    if not declared and not present:
        return
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("sampling contract validation requires a frozen source snapshot")
    verify_snapshot(Path(snapshot), source)
    blockers = candidate_blockers(request.get("candidate", {}))
    if task_ids is None:
        authorized = {row["task"] for row in request["view"]["assignments"]
                      if row["qualification_tier"] <= request["through_tier"]}
        members = set()
        for job in request["jobs"]:
            group = set(job.get("task_ids", [job["task_id"]]))
            if group & authorized:
                members.update(group)
        # Missing jobs must not erase an authorized task from this validation.
        members.update(authorized)
    else:
        members = set(task_ids)
    tasks = request.get("tasks", {})
    for member in sorted(members):
        if member not in tasks:
            blockers.append(f"{member}: authorized task lacks its frozen definition")
        else:
            blockers.extend(task_blockers(tasks[member]))
    if blockers:
        raise ValueError("sampling contract blocked: " + "; ".join(blockers))
