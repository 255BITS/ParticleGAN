"""Named conditional two-bank policy; source/gradient parity is not a win."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .contracts import stable_hash
from .policy_declaration_sources import DECLARATION_SOURCES
from .routed_policy_contracts import _execution_hash

COHORT = "multibank_policy_v1"
FAMILY = "atlas_multibank"
PARENT_ID = "cover_leftover"
TASK_ID = PARENT_ID + "_" + COHORT
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
HOST_RESOURCES = {"num_particles": 24, "z_dim": 4, "batch_size": 32}
HOST_SOURCE = "benchmarks/locked_shared/hosts/cover_leftover.py"
SOURCES = tuple(sorted(DECLARATION_SOURCES | {
    HOST_SOURCE, "benchmarks/locked_shared/baseline.py", "benchmarks/locked_shared/observation.py",
    "benchmarks/transfer_suite/protocol.py", "particlegan/policy.py", "particlegan/routing.py",
    "particlegan/recipes.py", "particlegan/training.py", "particlegan/init.py", "particlegan/particle_prior.py",
    "particlegan/vicreg_loss.py", "particlegan/birth_death.py", "experiments/forge/rng.py",
    "experiments/forge/policy_adapters.py", "experiments/forge/mechanisms.py",
    "experiments/forge/routed_policy_contracts.py", "experiments/forge/routed_policy_adapters.py",
    "experiments/forge/multibank_policy_contracts.py", "experiments/forge/multibank_policy_adapters.py"}))


def observation():
    return dict(schema_version=1, weight_selector="state_selected", sampler="served_snapshot",
        parameter_measurement="selected_original_odd_even_residual_geometry", output_noise=False,
        latent_policy="not_applied_to_parameter_measurement", row_selection="original_two_conditioned_poles",
        eval_streams="forge-rng-v1_isolated", diagnostic_weights=[], diagnostic_credit=False)


def contract(sources):
    return dict(schema_version=1, cohort=COHORT, family=FAMILY, owner="particlegan.UpdatePolicy",
        lifecycle="ordered_public_update", execution_path="public_components", row_policy="routed_paired",
        row_semantics="conditional", schedule="preserve_resolved_recipe_none", external_limit="task.execution.steps",
        execution_device="cuda", precision="preserve_original_fp32_no_autocast", cpu_controls_scope="structural_only",
        numerical_equivalence_to_parent=False, controls="resolved_recipe_requested_enabled_eligible_applied",
        table_owner="prior.z", banks={"plus": [0, 12], "minus": [12, 24]},
        bank_labels="immutable_router_bank_ids_original_storage_strata",
        transport="public_master_table_and_mass_moments_with_fixed_bank_identity_and_both_pole_guards",
        cross_bank_values="explicit_guarded_candidate_copy_not_sampling_the_opposite_stratum",
        original_objective="RpGAN_plus_concat_VICReg_weight.05_target.05_L2.02_two_cover_terms_weight1.5",
        loss_normalization="one_original_concat24_regularizer_and_two_separate_cover_means_no_factor2",
        initializer="original_zero_residual_subclass_KEEP_original_two_separate_R2_prior_initializations_named_critic",
        routing={"callback": "complete_model_forward", "sites": ["polar_bank_lookup"],
                 "fixed_bank_membership": True, "row_buffers": [], "log_mass": "router.log_mass",
                 "matched_logit": 0., "same_bank_fallback_logit": -1048576., "other_bank_logit": -2097152.},
        paired_controls={"fit": "original_plus_minus_cloud_and_sampled_indices_jitter.01",
            "guard": "both_original_teacher_poles_all24_rows_zero_host_jitter",
            "guard_use": "structural_acceptance_only", "heldout_generalization_claim": False,
            "feature_max_context_harm": 0., "output_max_context_harm": 0.},
        independent_atlas_qualification=False, independent_direct_response_owner=False,
        direct_response_reason="public_routed_transport_rejects_coupled_direct_row_history",
        inherited_auto_backend="public_routed_owner_not_independent_feature_cells",
        checkpoint="complete_public_policy_models_optimizers_streams_and_cursor", sources=deepcopy(sources))


def _variant(parent, pin, sources):
    task = deepcopy(parent)
    task.update(id=TASK_ID, task_cohort=COHORT, policy_family=FAMILY, policy_parent=deepcopy(pin))
    task["execution"].update(resources=deepcopy(HOST_RESOURCES), device="cuda", policy_contract=contract(sources),
        policy_recipe_overrides={"row_policy": "routed_paired"},
        policy_recipe_overrides_provenance={"family": FAMILY, "evidence_reuse": False,
            "original_two_banks": [12, 12], "new_storage": "concatenate_to_one24x4_parameter_without_new_degrees",
            "bank_identity": "fixed_original_storage_strata", "transport_guards": "both_conditioned_poles",
            "table_history": "one_public_master_latent_history_not_two_independent_histories"},
        schedule_policy="public_policy_stationarity_no_declared_horizon")
    task["resources"].update(device="cuda", gpus=1, gpu_memory_mb=2048, allow_cpu=False)
    task["evaluation"].update(scoring_weights="state_selected", policy_observation=observation())
    task["requires_capabilities"] = ["named_rng", "served_sampling", "policy_controls", "policy_serving", "routed_rows", "learned_locations", "particle_cloud"]
    return task


def make_variant(root):
    root = Path(root).resolve()
    raw = (root / f"configs/forge/tasks/{PARENT_ID}.json").read_bytes(); parent = json.loads(raw)
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in SOURCES}
    pin = dict(id=PARENT_ID, task_sha256=hashlib.sha256(raw).hexdigest(),
               execution_fingerprint=_execution_hash(parent), evaluation_fingerprint=stable_hash(parent["evaluation"]))
    return _variant(parent, pin, hashes)


def validate_task(task, *, root=None):
    if (task.get("id") != TASK_ID or task.get("task_cohort") != COHORT or task.get("policy_family") != FAMILY):
        raise ValueError("explicit atlas_multibank task required")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("complete multibank source binding required")
    # The exact source-key roster prevents arbitrary paths from entering the binding.
    for path, digest in sources.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid multibank source hash")
    if root is not None:
        if stable_hash(task) != stable_hash(make_variant(root)):
            raise ValueError("multibank original source/parent/objective/gates binding drift")
    e, v = task["execution"], task["evaluation"]
    if (e.get("resources") != HOST_RESOURCES or e.get("steps") != 800
            or stable_hash(e["policy_contract"]) != stable_hash(contract(sources))
            or e.get("policy_recipe_overrides") != {"row_policy": "routed_paired"}
            or v.get("policy_observation") != observation() or v.get("scoring_weights") != "state_selected"
            or v.get("observations") != 24 or v.get("minimum_stable_checks") != 5
            or v.get("thresholds") != [["u_kept", ">=", .85], ["content_kept", ">=", .75], ["leak_ratio", "<=", .2],
                ["pole_rel_err_plus", "<=", .2], ["pole_rel_err_minus", "<=", .2], ["same_dir", "<=", .25]]):
        raise ValueError("multibank original scientific contract changed")
    return deepcopy(e["policy_contract"])


def resolved_recipe(candidate, task):
    from particlegan import get_recipe
    validate_task(task)
    if (candidate.get("task_cohort") != COHORT or candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != SHARED_OVERRIDES):
        raise ValueError("multibank needs its explicit family and fixed shared C6 pair")
    return get_recipe("atlas", **SHARED_OVERRIDES, **HOST_RESOURCES, row_policy="routed_paired",
                      prior_kind="particles", sigma_rel=0., standardize=False)


def blockers(task, recipe):
    try:
        expected = resolved_recipe(dict(task_cohort=COHORT, recipe_preset="atlas", recipe_overrides=SHARED_OVERRIDES), task)
        if stable_hash(recipe.to_dict() if hasattr(recipe, "to_dict") else recipe) != stable_hash(expected.to_dict()):
            raise ValueError("multibank resolved Recipe drift")
        return []
    except (ValueError, KeyError, TypeError) as error:
        return [str(error)]
