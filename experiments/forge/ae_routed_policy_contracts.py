"""Explicit routed MoG AE variant; no independent-Atlas evidence reuse.

The original sampled width is fixed sigma=.025. Recipe.sigma_rel=0 explicitly
disables spacing calibration; the caller supplies that unchanged positive width.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from .contracts import stable_hash
from .policy_declaration_sources import DECLARATION_SOURCES

COHORT = "ae_routed_policy_v1"
FAMILY = "atlas_ae_routed"
PARENT_ID = "ae_gan_hold"
TASK_ID = PARENT_ID + "_" + COHORT
SAMPLING_LAW = "ae_routed_selected_mog_reconstruction"
EVAL_OUTPUT_NOISE = "clean_complete_routed_function"
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
HOST_RESOURCES = {"num_particles": 12, "z_dim": 2, "batch_size": 64}
ADAPTATION = {"encoder_mode": "ae", "prior_kind": "mog", "sigma_rel": 0.,
              "standardize": False, "row_policy": "routed_paired"}
ADAPTATION_PROVENANCE = {"schema_version": 1, "family": FAMILY,
    "change": "named_AE_conditional_paired_MoG_adaptation",
    "original_prior": "fixed_sigma_.025_nonstandardized_MoG", "original_encoder": "hard_AE_query_offset",
    "recipe_width_field": "sigma_rel_0_disables_spacing_calibration_actual_make_prior_sigma_.025_unchanged",
    "original_gradient": "detached_mean_query_proxy_and_selected_table_gradients",
    "row_role": "original_prior.z_keeps_fixed_width_and_gets_complete_context_mass_transport",
    "optimizer_group_change": "separate_generator_encoder_table_groups_for_public_owner_validation",
    "initialization": "parent_named_deterministic_orthogonal_each_original_complete_model",
    "independent_atlas_equivalence": False, "evidence_reuse": False}
HOST_SOURCE = "benchmarks/locked_shared/hosts/ae_gan_hold.py"
REQUIRED_SOURCES = DECLARATION_SOURCES | frozenset({HOST_SOURCE, "benchmarks/locked_shared/baseline.py",
    "benchmarks/locked_shared/observation.py", "benchmarks/transfer_suite/protocol.py",
    "particlegan/autoencoder.py", "particlegan/particle_prior.py", "particlegan/policy.py",
    "particlegan/routing.py", "particlegan/recipes.py", "particlegan/init.py", "particlegan/training.py",
    "experiments/forge/rng.py", "experiments/forge/policy_adapters.py",
    "experiments/forge/mechanisms.py", "experiments/forge/ae_routed_policy_contracts.py",
    "experiments/forge/ae_routed_policy_adapters.py"})


def is_ae_routed_task(task):
    return isinstance(task, dict) and task.get("task_cohort") == COHORT


def _source_manifest(sources):
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise ValueError("AE routed contract requires complete original/public/callback sources")
    for name, sha in sources.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name or not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("AE source identity requires safe relative paths and SHA256")


def _execution_hash(task):
    return stable_hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def _contract(sources):
    return {"schema_version": 1, "cohort": COHORT, "family": FAMILY,
        "owner": "particlegan.UpdatePolicy", "lifecycle": "ordered_public_update",
        "execution_path": "public_components", "row_semantics": "conditional", "row_policy": "routed_paired",
        "prior_requirement": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True,
                              "recipe_sigma_rel": 0., "actual_sigma_rel": 0., "actual_d0": 0.,
                              "width": "explicit_fixed_buffer_no_spacing_calibration"},
        "schedule": "preserve_resolved_recipe_none", "external_limit": "task.execution.steps",
        "execution_device": "cuda", "precision": "preserve_original_fp32_no_autocast",
        "cpu_controls_scope": "structural_only", "numerical_equivalence_to_parent": False,
        "routing": {"callback": "complete_model_forward", "sites": ["ae_mog_bank"],
                    "log_mass": "router.log_mass", "row_buffers": [],
                    "encoder": "original_2_to_4_MLP_query_offset",
                    "clean_AE": "public_Recipe.encode_hard_center_bounded_offset_detached_mean_query_proxy",
                    "mass_changes": "hard_center_and_query_proxy_use_candidate_log_mass",
                    "training_DV12": "actual_routing_mix_displacement_on_mixed_codes",
                    "unconditional": "caller_uniform_and_normal_context_mass_CDF_fixed_MoG_width",
                    "counterfactual": "rerun_supplied_generator_encoder_prior_router_and_candidate_table"},
        "initialization": "parent_named_deterministic_orthogonal_each_original_complete_model_and_fixed_width_prior",
        "control_scope": "conditional_paired_diagnostic_and_guarded_mass_transport",
        "independent_atom_bh_or_feature_cell_claim": False,
        "inherited_auto_backend": "delegated_to_public_routed_control_not_independent_feature_cells",
        "objective": {"reconstruction_weight": 1., "adversarial_weight": 1.,
                      "cover_weight": 1.5, "particle_l2": .02, "fm_weight": 0.},
        "guard": {"pool": "separate_known_anchor_contexts_not_training_batch",
                  "targets": "original_two_anchors", "use": "structural_acceptance_only",
                  "reconstruction_probes": "two_original_anchor_inputs",
                  "generation_probes": "fixed_uniform_.25_.75_zero_Gaussian_with_ordered_anchor_targets",
                  "generation_pairing_scope": "explicit_structural_proxy_not_an_original_component_label",
                  "heldout_generalization_claim": False, "max_feature_context_harm": 0.,
                  "max_output_context_harm": 0.},
        "table_optimizer_semantics": {"latent_damping_owner": True,
            "independent_direct_particle_response_owner": False,
            "reason": "public_routed_transport_rejects_coupled_direct_row_history"},
        "optimizer_roles": {"generator": "own_group", "encoder": "own_group",
                            "table": "own_prior_rate_group", "critic": "own_optimizer",
                            "noise": "public_policy_added_group"},
        "controls": "resolved_recipe_requested_enabled_eligible_applied",
        "table_owner": "prior.z", "encoder_owner": "encoder",
        "checkpoint": "complete_public_models_optimizers_controller_routing_streams_and_cursor",
        "independent_atlas_qualification": False, "evidence_reuse": False, "sources": deepcopy(sources)}


def _observation():
    return {"schema_version": 1, "weight_selector": "state_selected", "sampler": "served_snapshot",
        "output_noise": False, "latent_policy": "clean_complete_routed_function_without_DV12_evaluation_draw",
        "row_selection": "mass_aware_hard_AE_and_unconditional_MoG", "eval_streams": "forge-rng-v1_isolated",
        "diagnostic_weights": [], "diagnostic_credit": False, "family": FAMILY}


def _variant(parent, pin, sources):
    task = deepcopy(parent)
    task.update(id=TASK_ID, task_cohort=COHORT, policy_family=FAMILY, policy_parent=deepcopy(pin))
    task["execution"].update(resources=deepcopy(HOST_RESOURCES), policy_contract=_contract(sources),
        policy_recipe_overrides=deepcopy(ADAPTATION), policy_recipe_overrides_provenance=deepcopy(ADAPTATION_PROVENANCE),
        schedule_policy="public_policy_stationarity_no_declared_horizon",
        device="cuda")
    task["resources"].update(device="cuda", gpus=1, gpu_memory_mb=2048, allow_cpu=False)
    task["evaluation"].update(scoring_weights="state_selected", policy_observation=_observation(),
        sampling_law=SAMPLING_LAW, eval_output_noise=EVAL_OUTPUT_NOISE)
    task["requires_capabilities"] = ["named_rng", "served_sampling", "learned_locations", "mog_prior",
                                     "policy_controls", "policy_serving", "routed_rows", "ae_encoder"]
    return task


def make_ae_variant(root, *, sources=None):
    root = Path(root).resolve()
    raw = (root / "configs/forge/tasks/ae_gan_hold.json").read_bytes()
    parent = json.loads(raw)
    sources = ({p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in REQUIRED_SOURCES}
               if sources is None else deepcopy(sources))
    _source_manifest(sources)
    pin = {"id": PARENT_ID, "task_sha256": hashlib.sha256(raw).hexdigest(),
           "execution_fingerprint": _execution_hash(parent), "evaluation_fingerprint": stable_hash(parent["evaluation"])}
    return _variant(parent, pin, sources)


def validate_ae_task(task, *, root=None):
    if (not is_ae_routed_task(task) or task.get("id") != TASK_ID or task.get("policy_family") != FAMILY):
        raise ValueError("explicit atlas_ae_routed family/cohort required")
    pin = task.get("policy_parent", {})
    if (set(pin) != {"id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or pin["id"] != PARENT_ID or any(not isinstance(pin[k], str) or len(pin[k]) != 64
                or any(c not in "0123456789abcdef" for c in pin[k])
                for k in ("task_sha256", "execution_fingerprint", "evaluation_fingerprint"))):
        raise ValueError("AE parent binding missing")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    _source_manifest(sources)
    if root is not None:
        root = Path(root).resolve()
        parent = json.loads((root / "configs/forge/tasks/ae_gan_hold.json").read_text())
        if stable_hash(make_ae_variant(root, sources=sources)) != stable_hash(task):
            raise ValueError("AE original task/objective/gates or overlay drift")
        if (_execution_hash(parent) != pin["execution_fingerprint"]
                or stable_hash(parent["evaluation"]) != pin["evaluation_fingerprint"]):
            raise ValueError("AE original parent contract drift")
        for relative, sha in sources.items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("AE callback/public source binding drift: " + relative)
    if (stable_hash(task["execution"].get("policy_contract")) != stable_hash(_contract(sources))
            or stable_hash(task["execution"].get("policy_recipe_overrides")) != stable_hash(ADAPTATION)
            or stable_hash(task["execution"].get("policy_recipe_overrides_provenance")) != stable_hash(ADAPTATION_PROVENANCE)
            or stable_hash(task["execution"].get("resources")) != stable_hash(HOST_RESOURCES) or type(task["execution"].get("steps")) is not int
            or task["execution"].get("steps") != 250
            or stable_hash(task["execution"].get("prior")) != stable_hash({"kind": "mog", "sigma": .025, "standardize": False, "learnable": True})
            or stable_hash(task["evaluation"].get("thresholds")) != stable_hash([["recon_mse", "<=", .05], ["hold", "<=", .35]])
            or task["evaluation"].get("observations") != 24 or task["evaluation"].get("minimum_stable_checks") != 5
            or task["evaluation"].get("scoring_weights") != "state_selected"
            or task["evaluation"].get("sampling_law") != SAMPLING_LAW
            or task["evaluation"].get("eval_output_noise") != EVAL_OUTPUT_NOISE
            or task["evaluation"].get("policy_observation") != _observation()):
        raise ValueError("AE original scope/owners/metric bounds changed")
    return deepcopy(task["execution"]["policy_contract"])


def ae_recipe_overrides(task):
    validate_ae_task(task)
    return deepcopy(ADAPTATION)


def validate_ae_observation(task):
    validate_ae_task(task)
    return deepcopy(task["evaluation"]["policy_observation"])


def resolved_recipe(candidate, task):
    from particlegan import get_recipe
    validate_ae_task(task)
    if (candidate.get("task_cohort") != COHORT or candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != SHARED_OVERRIDES):
        raise ValueError("atlas_ae_routed needs its explicit family identity and frozen shared pair")
    # Do not disable birth/death to evade the public independent-MoG refusal.
    # Current releases without the narrow routed-MoG API extension fail here.
    return get_recipe("atlas", **SHARED_OVERRIDES, **HOST_RESOURCES, **ADAPTATION)


def ae_routed_preflight(task, candidate):
    try:
        resolved_recipe(candidate, task)
        return []
    except (TypeError, ValueError, KeyError) as error:
        return [f"{task.get('id', TASK_ID)}: {error}"]


def ae_policy_contract_blockers(task, recipe):
    """Validate metadata only; a declaration does not prove executed controls."""
    try:
        validate_ae_task(task)
        candidate = {"task_cohort": COHORT, "recipe_preset": "atlas", "recipe_overrides": SHARED_OVERRIDES}
        expected = resolved_recipe(candidate, task)
        actual = recipe if isinstance(recipe, dict) else recipe.to_dict()
        if stable_hash(actual) != stable_hash(expected.to_dict()):
            raise ValueError("resolved AE Recipe differs from the fixed family/resources/fixed-width law")
        return []
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        return [f"{task.get('id', TASK_ID)}: {error}"]
