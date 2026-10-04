"""Explicit conditional reforms of four original Forge behavioral questions.

The original targets, complete objectives, degrees of freedom and gates remain
owned by their parents.  Routed paired evidence, selected serving and the
storage/optimizer adaptations below define a prospective ``atlas_conditional``
law.  They confer no independent-atom Atlas or historical qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from .contracts import stable_hash
from .policy_declaration_sources import DECLARATION_SOURCES


COHORT = "conditional_policy_selected_cloud_v1"
FAMILY = "atlas_conditional"
SUFFIX = "_" + COHORT
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
UNMATCHED_LOGIT = -1048576.
HOSTS = {
    "trajectory": {"source": "benchmarks/locked_shared/trajectory.py", "steps": 400,
                   "resources": {"num_particles": 12, "z_dim": 4, "batch_size": 12},
                   "sites": ["identity_latent"], "table_owner": "prior.z",
                   "thresholds": [["identity_mse", "<=", .02]],
                   "objective": "original_paired_RpGAN_cover1.5_particle_l2.02_vicreg.05",
                   "fit": "original_even_identity_rows", "guard": "original_odd_identity_rows"},
    "residual_student": {"source": "benchmarks/locked_shared/hosts/residual_student.py", "steps": 400,
                   "resources": {"num_particles": 12, "z_dim": 4, "batch_size": 12},
                   "sites": ["identity_latent"], "table_owner": "prior.z",
                   "thresholds": [["identity_mse", "<=", .02], ["success_rate", ">=", 1.],
                                  ["wrong_pad_rate", "<=", 0.]],
                   "objective": "original_paired_RpGAN_cover1.5_particle_l2.02_vicreg.05_both_land_residual1",
                   "fit": "original_even_identity_rows", "guard": "original_odd_identity_rows"},
    "unipolar": {"source": "benchmarks/locked_shared/hosts/unipolar.py", "steps": 400,
                   "resources": {"num_particles": 3, "z_dim": 4, "batch_size": 8},
                   "sites": ["odd", "even", "origin"], "table_owner": "generator.bank",
                   "thresholds": [["cover", ">=", .85], ["off_caption", "<=", .05],
                                  ["neu_hold", ">=", .85]],
                   "objective": "original_equal_RpGAN_scales_0_and_1_no_auxiliary_loss",
                   "fit": "original_scale1", "guard": "original_scale0"},
    "mid_scale_identity": {"source": "benchmarks/locked_shared/hosts/mid_scale_identity.py", "steps": 800,
                   "resources": {"num_particles": 4, "z_dim": 4, "batch_size": 8},
                   "sites": ["odd", "even", "origin", "mid"], "table_owner": "generator.bank",
                   "thresholds": [["concept_cos_plus", ">=", .85], ["concept_cos_minus", ">=", .85],
                                  ["concept_mag_plus", ">=", .75], ["concept_mag_plus", "<=", 1.25],
                                  ["concept_mag_minus", ">=", .75], ["concept_mag_minus", "<=", 1.25],
                                  ["identity_at_0", ">=", .85], ["identity_at_mid", ">=", .85]],
                   "objective": "original_equal_RpGAN_minus1_0_half_1_plus_cover1.5_mean_coordinate_MSE",
                   "fit": "original_minus1_and_plus1", "guard": "original_0_and_half"},
}
COMMON_SOURCES = DECLARATION_SOURCES | frozenset({
    "benchmarks/locked_shared/baseline.py", "benchmarks/locked_shared/observation.py",
    "benchmarks/locked_shared/trajectory.py", "benchmarks/locked_shared/hosts/cover_leftover.py",
    "benchmarks/transfer_suite/protocol.py", "benchmarks/legacy/locked_shared.py",
    "particlegan/policy.py", "particlegan/routing.py", "particlegan/recipes.py",
    "particlegan/init.py", "particlegan/training.py", "experiments/forge/rng.py",
    "experiments/forge/policy_adapters.py", "experiments/forge/mechanisms.py",
    "experiments/forge/conditional_policy_contracts.py", "experiments/forge/conditional_policy_adapters.py"})


def is_conditional_policy_task(task):
    return isinstance(task, dict) and task.get("task_cohort") == COHORT


def _host(task):
    host = task.get("execution", {}).get("host")
    if host not in HOSTS:
        raise ValueError("conditional policy requires one of its four explicit original hosts")
    return host


def _source_manifest(host, sources):
    required = COMMON_SOURCES | {HOSTS[host]["source"]}
    if not isinstance(sources, dict) or not required <= sources.keys():
        raise ValueError("conditional contract needs complete original host/public implementation sources")
    for relative, digest in sources.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or str(PurePosixPath(relative)) != relative or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("conditional source paths/hashes must be safe relative SHA256 bindings")


def _execution_hash(task):
    return stable_hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def _contract(host, sources):
    info = HOSTS[host]
    return {"schema_version": 1, "cohort": COHORT, "family": FAMILY,
            "owner": "particlegan.UpdatePolicy", "lifecycle": "ordered_public_update",
            "execution_path": "public_components", "row_semantics": "conditional", "row_policy": "routed_paired",
            "schedule": "preserve_resolved_recipe_none", "external_limit": "task.execution.steps",
            "execution_device": "cuda", "precision": "preserve_original_fp32_no_autocast",
            "cpu_controls_scope": "structural_only", "numerical_equivalence_to_parent": False,
            "table_owner": info["table_owner"], "original_objective": info["objective"],
            "routing": {"callback": "complete_model_forward", "sites": info["sites"],
                        "row_buffers": ["row_roles"], "log_mass": "router.log_mass",
                        "matched_logit": 0., "unmatched_logit": UNMATCHED_LOGIT,
                        "counterfactual": "complete_context_mass_aware_retire_split_rerun"},
            "paired_controls": {"fit_context": info["fit"], "guard_context": info["guard"],
                                "all_original_training_contexts_preserved": True,
                                "guard_use": "structural_acceptance_only",
                                "guards_are_known_original_training_contexts": True,
                                "guard_tensors_never_used_for_training_or_candidate_choice": True,
                                "heldout_generalization_claim": False,
                                "feature_max_context_harm": 0., "output_max_context_harm": 0.},
            "table_optimizer_semantics": {"latent_damping_owner": True,
                "independent_direct_particle_response_owner": False,
                "recipe_direct_particle_gain_preserved": True,
                "reason": "public_routed_transport_rejects_coupled_direct_row_history"},
            "storage_adaptation": ("none_original_MLP_and_prior_z" if info["table_owner"] == "prior.z" else
                "original_zero_basis_vectors_pack_into_new_subclass_matrix_parameter_same_formula_and_degrees"),
            "initialization": ("parent_named_orthogonal_networks_and_R2_prior" if info["table_owner"] == "prior.z" else
                "new_subclass_bank_KEEP_original_zeros_and_parent_named_orthogonal_critic"),
            "training_latent_policy": "actual_public_DV12_mixed_codes_at_each_declared_site",
            "control_scope": "conditional_paired_evidence_and_guarded_mass_transport",
            "independent_atom_bh_or_feature_cell_claim": False,
            "controls": "resolved_recipe_requested_enabled_eligible_applied",
            "checkpoint": "complete_public_policy_models_optimizers_streams_and_cursor",
            "sources": deepcopy(sources)}


def _observation(host):
    arcs = HOSTS[host]["table_owner"] == "prior.z"
    return {"schema_version": 1, "weight_selector": "state_selected",
            "sampler": "served_routed_forward_original_identity_grid" if arcs else "served_snapshot",
            "parameter_measurement": "original_paired_prediction" if arcs else "selected_complete_routed_basis_function",
            "output_noise": "resolved_recipe" if arcs else False,
            "latent_policy": "not_applied_to_original_enumerated_measurement",
            "row_selection": "original_identity_order" if arcs else "original_scale_grid",
            "eval_streams": "forge-rng-v1_isolated", "diagnostic_weights": [], "diagnostic_credit": False}


def _variant(parent, pin, sources):
    host = _host(parent); info = HOSTS[host]
    task = deepcopy(parent)
    task.update(id=host + SUFFIX, task_cohort=COHORT, policy_family=FAMILY, policy_parent=deepcopy(pin))
    execution = task["execution"]
    execution.update(resources=deepcopy(info["resources"]), policy_contract=_contract(host, sources),
        policy_recipe_overrides={"row_policy": "routed_paired"},
        policy_recipe_overrides_provenance={"schema_version": 1, "family": FAMILY,
            "source": info["source"], "change": "complete_conditional_routing_and_selected_public_policy",
            "original_objective": info["objective"], "all_original_training_contexts_preserved": True,
            "table_owner": info["table_owner"], "storage_adaptation": _contract(host, sources)["storage_adaptation"],
            "initializer_adaptation": _contract(host, sources)["initialization"],
            "training_law_change": "public_routed_DV12_jitter_and_mass_aware_counterfactuals",
            "table_rate_owner": "explicit_prior_lr_mult_group_also_for_basis_bank",
            "independent_atlas_equivalence": False, "evidence_reuse": False},
        schedule_policy="public_policy_stationarity_no_declared_horizon", device="cuda")
    task["resources"].update(device="cuda", gpus=1, gpu_memory_mb=max(2048, parent["resources"]["gpu_memory_mb"]),
                             allow_cpu=False)
    task["evaluation"].update(scoring_weights="state_selected", policy_observation=_observation(host))
    task["requires_capabilities"] = list(dict.fromkeys([
        "served_sampling" if field == "live_sampling" else field for field in parent["requires_capabilities"]
    ] + ["policy_controls", "policy_serving", "routed_rows"]))
    return task


def make_conditional_variant(root, host, *, sources=None):
    if host not in HOSTS:
        raise ValueError("unknown conditional parent")
    root = Path(root).resolve()
    raw = (root / f"configs/forge/tasks/{host}.json").read_bytes(); parent = json.loads(raw)
    sources = ({relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in COMMON_SOURCES | {HOSTS[host]["source"]}} if sources is None else deepcopy(sources))
    _source_manifest(host, sources)
    pin = {"id": host, "task_sha256": hashlib.sha256(raw).hexdigest(),
           "execution_fingerprint": _execution_hash(parent), "evaluation_fingerprint": stable_hash(parent["evaluation"])}
    return _variant(parent, pin, sources)


def validate_conditional_task(task, *, parent=None, root=None):
    host = _host(task); info = HOSTS[host]
    if (not is_conditional_policy_task(task) or task.get("id") != host + SUFFIX
            or task.get("policy_family") != FAMILY):
        raise ValueError("an explicitly named atlas_conditional task/cohort is required")
    pin = task.get("policy_parent")
    if (not isinstance(pin, dict) or set(pin) != {"id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or pin["id"] != host or any(not isinstance(pin[key], str) or len(pin[key]) != 64
                or any(c not in "0123456789abcdef" for c in pin[key]) for key in pin if key != "id")):
        raise ValueError("conditional task needs its complete original parent identity")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    _source_manifest(host, sources)
    if root is not None:
        root = Path(root).resolve(); raw = (root / f"configs/forge/tasks/{host}.json").read_bytes()
        parent = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != pin["task_sha256"]:
            raise ValueError("conditional original parent byte identity drift")
        for relative, digest in sources.items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"conditional source binding drift: {relative}")
    if parent is not None:
        if (_execution_hash(parent) != pin["execution_fingerprint"]
                or stable_hash(parent["evaluation"]) != pin["evaluation_fingerprint"]
                or stable_hash(task) != stable_hash(_variant(parent, pin, sources))):
            raise ValueError("conditional task changed its original objective, host, initializer, gates or resources")
    elif (task["execution"].get("steps") != info["steps"]
            or task["execution"].get("initializer") != "deterministic_orthogonal"
            or task["execution"].get("resources") != info["resources"]
            or task["execution"].get("policy_recipe_overrides") != {"row_policy": "routed_paired"}
            or stable_hash(task["execution"].get("policy_contract")) != stable_hash(_contract(host, sources))
            or task["evaluation"].get("thresholds") != info["thresholds"]
            or task["evaluation"].get("observations") != 24
            or task["evaluation"].get("minimum_stable_checks") != 5
            or task["evaluation"].get("scoring_weights") != "state_selected"
            or task["evaluation"].get("policy_observation") != _observation(host)):
        raise ValueError("conditional runtime contract or original numerical bounds changed")
    return deepcopy(task["execution"]["policy_contract"])


def conditional_recipe_overrides(task):
    validate_conditional_task(task)
    return {"row_policy": "routed_paired"}


def validate_conditional_observation(task):
    validate_conditional_task(task)
    return deepcopy(_observation(_host(task)))


def load_conditional_variants(root, parent_tasks=None):
    """Explicit new-cohort declarations; never replace original task IDs."""
    root = Path(root).resolve()
    folder = root / "configs/forge/task-variants" / COHORT
    expected = {host + SUFFIX + ".json" for host in HOSTS}
    paths = {path.name: path for path in folder.glob("*.json")}
    if set(paths) != expected:
        raise ValueError("conditional cohort needs exactly its four frozen variant declarations")
    result = {}
    for filename in sorted(paths):
        task = json.loads(paths[filename].read_bytes())
        host = _host(task)
        parent = None if parent_tasks is None else parent_tasks[host]
        validate_conditional_task(task, parent=parent, root=root)
        if filename != task["id"] + ".json" or task["id"] in result:
            raise ValueError("conditional variant filename/identity conflicts")
        result[task["id"]] = task
    return result


def resolved_recipe(candidate, task):
    from particlegan import get_recipe
    validate_conditional_task(task)
    if (candidate.get("task_cohort") != COHORT or candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != SHARED_OVERRIDES):
        raise ValueError("atlas_conditional needs its explicit cohort and exact shared Atlas LR/prior-rate pair")
    return get_recipe("atlas", **SHARED_OVERRIDES, **HOSTS[_host(task)]["resources"], row_policy="routed_paired",
                      prior_kind="particles", sigma_rel=0., standardize=False)


def conditional_policy_contract_blockers(task, recipe):
    try:
        expected = resolved_recipe({"task_cohort": COHORT, "recipe_preset": "atlas",
                                    "recipe_overrides": SHARED_OVERRIDES}, task).to_dict()
        actual = recipe if isinstance(recipe, dict) else recipe.to_dict()
        if stable_hash(actual) != stable_hash(expected):
            raise ValueError("effective conditional Recipe differs from its declared controls/resources")
        return []
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"{task.get('id', COHORT)}: {exc}"]
