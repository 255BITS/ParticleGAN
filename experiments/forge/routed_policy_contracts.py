"""One explicit routed-family adaptation of the original unused-token goal.

The target, shared/slot scaffold, auxiliary hold loss and numerical gates stay
with the original host. ``atlas_routed`` is a distinct conditional paired-row
law; declarations confer no independent-Atlas or learned qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from .contracts import stable_hash
from .policy_declaration_sources import DECLARATION_SOURCES


COHORT = "routed_policy_selected_cloud_v1"
SUFFIX = "_" + COHORT
FAMILY = "atlas_routed"
PARENT_ID = "unused_token_hold"
TASK_ID = PARENT_ID + SUFFIX
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
HOST_RESOURCES = {"num_particles": 2, "z_dim": 2, "batch_size": 8}
UNMATCHED_LOGIT = -1048576.
TABLE_OPTIMIZER_SEMANTICS = {"latent_damping_owner": True,
    "independent_direct_particle_response_owner": False,
    "reason": "public_routed_transport_rejects_coupled_direct_row_history",
    "recipe_direct_particle_gain_preserved": True}
HOST_SOURCE = "benchmarks/locked_shared/hosts/unused_token_hold.py"
REQUIRED_SOURCES = DECLARATION_SOURCES | frozenset({
    HOST_SOURCE, "benchmarks/locked_shared/baseline.py", "benchmarks/locked_shared/observation.py",
    "benchmarks/transfer_suite/protocol.py", "particlegan/policy.py", "particlegan/routing.py",
    "particlegan/recipes.py", "particlegan/training.py", "particlegan/init.py", "experiments/forge/rng.py",
    "experiments/forge/policy_adapters.py", "experiments/forge/mechanisms.py",
    "experiments/forge/routed_policy_contracts.py", "experiments/forge/routed_policy_adapters.py"})


def is_routed_policy_task(task):
    return isinstance(task, dict) and task.get("task_cohort") == COHORT


def _source_manifest(sources):
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise ValueError("routed contract needs the original host and complete public implementation sources")
    for path, digest in sources.items():
        if (not isinstance(path, str) or not path or "\\" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("routed sources need safe relative paths and SHA256 hashes")


def _execution_hash(task):
    return stable_hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def _contract(sources):
    return {"schema_version": 1, "cohort": COHORT, "family": FAMILY,
            "owner": "particlegan.UpdatePolicy", "lifecycle": "ordered_public_update",
            "execution_path": "public_components", "row_semantics": "conditional",
            "row_policy": "routed_paired", "schedule": "preserve_resolved_recipe_none",
            "external_limit": "task.execution.steps", "execution_device": "cuda",
            "precision": "preserve_original_fp32_no_autocast", "cpu_controls_scope": "structural_only",
            "numerical_equivalence_to_parent": False,
            "table_owner": "generator.slot", "shared_owner": "generator.shared",
            "table_optimizer_semantics": deepcopy(TABLE_OPTIMIZER_SEMANTICS),
            "initialization": {"generator": "original_constructor_zeros_new_subclass_public_KEEP",
                               "critic": "parent_named_deterministic_orthogonal",
                               "original_class_registration_changed": False},
            "routing": {"callback": "complete_model_forward", "sites": ["slot_lookup"],
                        "row_buffers": ["slot_ids"], "log_mass": "router.log_mass",
                        "matched_logit": 0., "unmatched_logit": UNMATCHED_LOGIT,
                        "counterfactual": "full_context_mass_aware_retire_split_rerun"},
            "paired_controls": {"fit_context": "CONCEPT_scale1", "guard_context": "UNUSED_scale1",
                                "guard_use": "structural_acceptance_only",
                                "known_unused_anchor_is_also_original_training_loss": True,
                                "heldout_generalization_claim": False,
                                "feature_max_context_harm": 0., "output_max_context_harm": 0.},
            "control_scope": "conditional_paired_diagnostic_and_guarded_mass_transport",
            "independent_atom_bh_or_feature_cell_claim": False,
            "inherited_auto_backend": "delegated_to_public_routed_control_not_independent_feature_cells",
            "training_latent_policy": "actual_DV12_mixed_codes",
            "controls": "resolved_recipe_requested_enabled_eligible_applied",
            "checkpoint": "complete_public_policy_models_optimizers_streams_and_cursor",
            "sources": deepcopy(sources)}


def _observation():
    return {"schema_version": 1, "weight_selector": "state_selected", "sampler": "served_snapshot",
            "parameter_measurement": "selected_complete_routed_shared_slot_function",
            "output_noise": False, "latent_policy": "not_applied_to_parameter_measurement",
            "row_selection": "original_UNUSED_CONCEPT_slot_roles", "eval_streams": "forge-rng-v1_isolated",
            "diagnostic_weights": [], "diagnostic_credit": False}


def _variant(parent, pin, sources):
    task = deepcopy(parent)
    task.update(id=TASK_ID, task_cohort=COHORT, policy_family=FAMILY, policy_parent=deepcopy(pin))
    execution = task["execution"]
    execution.update(resources=deepcopy(HOST_RESOURCES), policy_contract=_contract(sources),
                     policy_recipe_overrides={"row_policy": "routed_paired"},
                     policy_recipe_overrides_provenance={
                         "schema_version": 1, "family": FAMILY, "change": "named_conditional_paired_row_adaptation",
                         "original_table": "SharedSlotStudent.slot", "original_shared": "SharedSlotStudent.shared",
                         "row_role": "slot_generator_parameter_becomes_explicit_table_optimizer_group",
                         "objective": "original_concept_RpGAN_plus_matched_unused_hold_MSE",
                         "initialization": "parent_named_orthogonal_with_new_subclass_KEEP_for_original_zeros",
                         "direct_fixture_initializer_requirement": "SharedSlotStudent_needs_explicit_KEEP_when_not_previously_registered",
                         "original_complete_behavior_adapter_already_registers_KEEP": True,
                         "initializer_adaptation": "public_init_register_new_subclass_only_shared_KEEP_slot_KEEP",
                         "independent_atlas_equivalence": False, "evidence_reuse": False},
                     schedule_policy="public_policy_stationarity_no_declared_horizon", device="cuda")
    # The table is the original two slot residuals, not the unrelated 12-atom
    # demo card. No sampled latent prior or replacement model is introduced.
    execution["prior"]["exception_reason"] = (
        "Original shared/slot parameter host; generator.slot is an explicit routed table, not a sampled independent prior.")
    task["resources"].update(device="cuda", gpus=1, gpu_memory_mb=max(2048, parent["resources"]["gpu_memory_mb"]),
                             allow_cpu=False)
    task["evaluation"].update(scoring_weights="state_selected", policy_observation=_observation())
    task["requires_capabilities"] = list(dict.fromkeys([
        "served_sampling" if value == "live_sampling" else value for value in parent["requires_capabilities"]
    ] + ["policy_controls", "policy_serving", "routed_rows"]))
    return task


def make_unused_variant(root, *, sources=None):
    """Read declarations only; root owns any later freezing/publication."""
    root = Path(root).resolve()
    raw = (root / "configs/forge/tasks/unused_token_hold.json").read_bytes()
    parent = json.loads(raw)
    sources = ({relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in REQUIRED_SOURCES} if sources is None else deepcopy(sources))
    _source_manifest(sources)
    pin = {"id": PARENT_ID, "task_sha256": hashlib.sha256(raw).hexdigest(),
           "execution_fingerprint": _execution_hash(parent),
           "evaluation_fingerprint": stable_hash(parent["evaluation"])}
    return _variant(parent, pin, sources)


def validate_routed_task(task, *, parent=None, root=None):
    if (not is_routed_policy_task(task) or task.get("id") != TASK_ID
            or task.get("policy_family") != FAMILY):
        raise ValueError("an explicit atlas_routed unused-token task is required")
    pin = task.get("policy_parent")
    if (not isinstance(pin, dict) or set(pin) != {
            "id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or pin["id"] != PARENT_ID or any(not isinstance(pin[key], str) or len(pin[key]) != 64
                or any(c not in "0123456789abcdef" for c in pin[key])
                for key in ("task_sha256", "execution_fingerprint", "evaluation_fingerprint"))):
        raise ValueError("routed task needs its complete original parent identity")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    _source_manifest(sources)
    if root is not None:
        root = Path(root).resolve()
        raw = (root / "configs/forge/tasks/unused_token_hold.json").read_bytes()
        parent = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != pin["task_sha256"]:
            raise ValueError("routed original parent byte identity drift")
        for relative, digest in sources.items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"routed source binding drift: {relative}")
    if parent is not None:
        if (_execution_hash(parent) != pin["execution_fingerprint"]
                or stable_hash(parent["evaluation"]) != pin["evaluation_fingerprint"]
                or stable_hash(task) != stable_hash(_variant(parent, pin, sources))):
            raise ValueError("routed task changed its original objective, host, initialization or gates")
    else:
        # Runtime still checks the entire declared lifecycle and measurement
        # overlay; full parent/source verification belongs before admission.
        expected = _contract(sources)
        if (stable_hash(task["execution"]["policy_contract"]) != stable_hash(expected)
                or task["execution"].get("resources") != HOST_RESOURCES
                or task["execution"].get("steps") != 200
                or task["execution"].get("policy_recipe_overrides") != {"row_policy": "routed_paired"}
                or task["evaluation"].get("thresholds") != [["unused_hold", ">=", .85], ["concept_move", ">=", .85]]
                or task["evaluation"].get("observations") != 24
                or task["evaluation"].get("minimum_stable_checks") != 5
                or task["evaluation"].get("scoring_weights") != "state_selected"
                or task["evaluation"].get("policy_observation") != _observation()):
            raise ValueError("routed runtime contract or original numerical bounds changed")
    return deepcopy(task["execution"]["policy_contract"])


def routed_recipe_overrides(task):
    validate_routed_task(task)
    return {"row_policy": "routed_paired"}


def validate_routed_observation(task):
    validate_routed_task(task)
    return deepcopy(task["evaluation"]["policy_observation"])


def resolved_recipe(candidate, task):
    """Metadata-only public preset resolution, with explicit new-family scope."""
    from particlegan import get_recipe
    validate_routed_task(task)
    if candidate.get("task_cohort") != COHORT or candidate.get("recipe_preset") != "atlas":
        raise ValueError("atlas_routed requires its explicit cohort and public Atlas base preset")
    if candidate.get("recipe_overrides") != SHARED_OVERRIDES:
        raise ValueError("atlas_routed uses exactly the declared C6 shared LR/prior-rate pair")
    return get_recipe("atlas", **SHARED_OVERRIDES, **HOST_RESOURCES, row_policy="routed_paired",
                      prior_kind="particles", sigma_rel=0., standardize=False)


def routed_policy_contract_blockers(task, recipe):
    """Static only: actual callbacks, state and lifecycle remain runtime proof."""
    try:
        expected = resolved_recipe({"task_cohort": COHORT, "recipe_preset": "atlas",
                                    "recipe_overrides": SHARED_OVERRIDES}, task).to_dict()
        actual = recipe if isinstance(recipe, dict) else recipe.to_dict()
        if stable_hash(actual) != stable_hash(expected):
            raise ValueError("effective routed Recipe differs from the declared base controls/resources")
        return []
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"{task.get('id', TASK_ID)}: {exc}"]
