"""Prospective policy-selected cloud tasks; no model construction or scoring.

Original declarations and their live/MoG evidence remain immutable. Explicit
candidate opt-in resolves a separate revision of the same main view. A valid
declaration is a prerequisite, never proof that a host implements the lifecycle
or that any learning gate passed. Runtime ownership and counters are attested by
the public execution adapter.

All prospective numerical attempts require CUDA. This explicit device change
does not assert numerical equivalence to the parent CPU/default-device cohort.
CPU metadata tests and tiny software controls remain structural checks only;
they cannot qualify any prospective task or supply its training evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


COHORT = "policy_selected_cloud_v1"
SUFFIX = "_" + COHORT
PARENT_VIEW_ID = "discriminator_stability"
PARENT_REVISION = 3
REVISION = 4
PARENT_TASK_IDS = (
    "two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition",
    "five_word_joint_acquisition", "trajectory", "residual_student", "unipolar",
    "cover_leftover", "mid_scale_identity", "mode_hold", "vector_two_broad",
    "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic",
    "vector_overlap", "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4",
    "img_intensity2", "grid100", "rotated100", "staggered100", "ring_hold",
    "ring_extension",
)
VECTOR_C6 = frozenset(PARENT_TASK_IDS[11:17])
IMAGE_C6 = frozenset(PARENT_TASK_IDS[17:21])
NATIVE_C6 = frozenset(PARENT_TASK_IDS[21:24])
OVERRIDE_FIELDS = frozenset({"d_lr_mult", "betas", "prior_reg"})
SELECTION_SOURCE = "reports/forge/c6-baseline-debug-20261003/BASELINE_SELECTION.json"
SELECTION_SHA256 = "bb5dafad7f04305a579f498093ad65d27f9ef5cc096caa709d4ce6e1fc7c8114"
C6_SOURCE_COMMIT = "8021a1c50c4aff90ddea5010d368cffdc857b2f6"
C6_RECIPES = {
    "atlas-vector": "5c6714ed92d1b90c669bc6140069b0f55fa618628c698da03bf8140df169b23b",
    "atlas-image": "8569265702f682b316c09f15a2396b75e97e4381943cf2ba46e558ef8b069b0c",
    "atlas-native100": "802ef265b8abaa7146ef7051074982abc8f4ba284925793d9989dad0f5b8155b",
}
REQUIRED_POLICY_SOURCES = frozenset({
    "particlegan/recipes.py", "particlegan/policy.py", "particlegan/training.py",
    "experiments/forge/policy_contracts.py",
})
PARAMETER_LAWS = frozenset({"learned_particles_and_critic_gradient", "learned_parameter_measurement"})
EXECUTION_DEVICE = "cuda"
PRECISION = "preserve_task_tensor_dtypes_no_autocast"
MIN_GPU_MEMORY_MB = 2048

# The declared Atlas mechanism, not a family-name bypass or a training verdict.
# Host dimensions are checked by normal task ownership after Recipe resolution.
REQUIRED_RECIPE = {
    "continuous_policy": "dv12", "total_steps": None, "lr_control": "stationarity",
    "prior_kind": "particles", "sigma_rel": 0., "standardize": False,
    "row_policy": "independent", "conditioning": "scalar", "encoder_mode": "none",
    "row_evidence_gate": True, "particle_birth_death": True,
    "table_release_rule": "anchor", "row_evidence_hot": True,
    "row_evidence_exclude": True, "row_evidence_hold": True,
    "row_evidence_null": "scaled", "birth_death_space": "critic",
    "birth_death_feature_scale": "std", "birth_death_isolation": True,
    "birth_death_backend": "auto", "birth_death_cells": 128,
    "birth_death_parent_policy": "real_anchor", "serve_average": 4.,
    "reopen_signal": "optimizer", "reopen_anchor": "release", "reopen_guard": "settled",
    "output_noise_mode": "learnable", "output_noise_warmup": 0.,
    "input_noise_std": 0., "amsgrad": True,
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _execution_fingerprint(task):
    return _hash({key: task.get(key) for key in (
        "schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def _parent_record(parent, task_sha256):
    return {"id": parent["id"], "task_sha256": task_sha256,
            "execution_fingerprint": _execution_fingerprint(parent),
            "evaluation_fingerprint": _hash(parent["evaluation"])}


def _parent_id(task):
    record = task.get("policy_parent", {})
    if (not isinstance(record, dict) or set(record) != {
            "id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or record.get("id") not in PARENT_TASK_IDS
            or any(not _digest(record.get(key)) for key in (
                "task_sha256", "execution_fingerprint", "evaluation_fingerprint"))):
        raise ValueError("policy task needs a complete known parent identity")
    return record["id"]


def is_policy_task(task):
    """Identify an explicit declaration; validation is independently required."""
    return (isinstance(task, dict) and task.get("task_cohort") == COHORT
            and isinstance(task.get("execution"), dict)
            and isinstance(task["execution"].get("policy_contract"), dict))


def _c6_group(parent_id):
    if parent_id in VECTOR_C6:
        return "atlas-vector"
    if parent_id in IMAGE_C6:
        return "atlas-image"
    if parent_id in NATIVE_C6:
        return "atlas-native100"
    return None


def _override_declaration(parent_id):
    group = _c6_group(parent_id)
    if group is None:
        return {}, None
    values = {"d_lr_mult": 1.5, "betas": [0., .99], "prior_reg": .05} if group == "atlas-vector" else {
        "d_lr_mult": 1., "betas": [0., .999], "prior_reg": 0.}
    return values, {"schema_version": 1, "owner": "task",
                    "source": SELECTION_SOURCE, "source_sha256": SELECTION_SHA256,
                    "reference_source_commit": C6_SOURCE_COMMIT,
                    "recipe_group": group, "reference_recipe_sha256": C6_RECIPES[group],
                    "fields": sorted(OVERRIDE_FIELDS), "evidence_reuse": False}


def policy_recipe_overrides(task):
    """Return only validated, task-owned prospective numerical exceptions.

    These are not host_adaptation permissions. Unknown behavioral/ring/word
    hosts receive no C6 constants. The complete effective Recipe is a separate
    execution receipt, including unchanged candidate settings.
    """
    if not is_policy_task(task):
        raise ValueError("policy Recipe exceptions require an explicit policy task")
    parent_id = _parent_id(task)
    execution = task["execution"]
    values = execution.get("policy_recipe_overrides")
    expected, provenance = _override_declaration(parent_id)
    if not isinstance(values, dict) or set(values) - OVERRIDE_FIELDS:
        raise ValueError("policy_recipe_overrides permits only d_lr_mult, betas, prior_reg")
    for name in ("d_lr_mult", "prior_reg"):
        if name in values and (not _finite(values[name]) or values[name] < 0
                               or (name == "d_lr_mult" and values[name] == 0)):
            raise ValueError(f"policy_recipe_overrides.{name} must be finite and valid")
    if "betas" in values and (not isinstance(values["betas"], list) or len(values["betas"]) != 2
                              or any(not _finite(x) or not 0 <= x < 1 for x in values["betas"])):
        raise ValueError("policy_recipe_overrides.betas requires two finite values in [0,1)")
    if (_hash(values) != _hash(expected)
            or _hash(execution.get("policy_recipe_overrides_provenance")) != _hash(provenance)):
        raise ValueError("policy Recipe exceptions/provenance differ from the frozen host-specific declaration")
    return deepcopy(values)


def _observation(parent):
    evaluation = parent["evaluation"]
    parameter = evaluation["sampling_law"] in PARAMETER_LAWS
    components = parent["execution"].get("execution_path", "public_trainer") == "public_components"
    result = {"schema_version": 1, "weight_selector": "state_selected",
              "sampler": "served_snapshot" if parameter else (
                  "ServedModel.generate" if components else "GANTrainer.sample"),
              "output_noise": "resolved_recipe" if evaluation["eval_output_noise"] == "public_recipe_schedule" else False,
              "latent_policy": "not_applied_to_parameter_measurement" if parameter else "actual_selected_public_policy",
              "row_selection": "original_task_strategy", "eval_streams": "forge-rng-v1_isolated",
              "diagnostic_weights": ["forced_ema"] if parent["adapter"] == "native100" else [],
              "diagnostic_credit": False}
    if parameter:
        result["parameter_measurement"] = ("selected_table_and_critic_gradient" if parent["id"].removesuffix(SUFFIX) == "two_pole"
                                            else "selected_host_parameters")
    return result


def validate_policy_observation(task):
    """Validate the opt-in overlay; never reinterpret historical live evidence."""
    if not is_policy_task(task):
        raise ValueError("policy observation requires an explicit policy task")
    parent_id = _parent_id(task)
    if task.get("id") != parent_id + SUFFIX:
        raise ValueError("policy observation variant ID/parent mismatch")
    evaluation = task.get("evaluation", {})
    observed = evaluation.get("policy_observation")
    if (evaluation.get("scoring_weights") != "state_selected"
            or not isinstance(observed, dict) or _hash(observed) != _hash(_observation(task))):
        raise ValueError("policy observation must use the declared selected law, isolated streams and separate diagnostics")
    return deepcopy(observed)


def _contract(execution_path, sources):
    return {"schema_version": 1, "cohort": COHORT, "owner": "particlegan.UpdatePolicy",
            "lifecycle": "ordered_public_update", "execution_path": execution_path,
            "execution_device": EXECUTION_DEVICE, "precision": PRECISION,
            "cpu_controls_scope": "structural_only", "numerical_equivalence_to_parent": False,
            "row_semantics": "independent",
            "prior_requirement": {"kind": "particle_cloud", "sigma": 0., "standardize": False, "learnable": True},
            "schedule": "preserve_resolved_recipe_none", "external_limit": "task.execution.steps",
            "controls": "resolved_recipe_requested_enabled_eligible_applied",
            "checkpoint": "complete_public_policy_models_optimizers_streams_and_cursor",
            "sources": deepcopy(sources)}


def _source_manifest(sources):
    if not isinstance(sources, dict) or not REQUIRED_POLICY_SOURCES <= sources.keys():
        raise ValueError("policy contract lacks required public policy/source bindings")
    for relative, digest in sources.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or str(PurePosixPath(relative)) != relative or not _digest(digest)):
            raise ValueError("policy sources require safe relative paths and SHA256 identities")


def _prospective_variant(parent, parent_pin, sources):
    """One allowed transformation, also used to reject coherent scientific drift."""
    variant = deepcopy(parent)
    parent_id = parent["id"]
    variant["id"] = parent_id + SUFFIX
    variant["task_cohort"] = COHORT
    variant["policy_parent"] = deepcopy(parent_pin)
    execution = variant["execution"]
    # Preserve task-owned initialization metadata, including init_std if present.
    prior = deepcopy(execution["prior"])
    prior.update(kind="particle_cloud", sigma=0., standardize=False, learnable=True,
                 exception_reason="Explicit prospective Atlas policy-selected cloud cohort; original task law and evidence retain their own identity.")
    execution["prior"] = prior
    execution["policy_contract"] = _contract(execution.get("execution_path", "public_trainer"), sources)
    values, provenance = _override_declaration(parent_id)
    execution["policy_recipe_overrides"] = values
    execution["policy_recipe_overrides_provenance"] = provenance
    # Device is a prospective execution contrast, never a rewrite of the
    # parent's recorded device or permission to pool CPU/GPU numerical grades.
    execution["policy_device_provenance"] = {
        "schema_version": 1, "change": "prospective_cuda_numerical_execution",
        "parent_execution_device": parent["execution"].get("device"),
        "parent_resources": deepcopy(parent["resources"]),
        "parent_execution_fingerprint": parent_pin["execution_fingerprint"],
        "numerical_equivalence_to_parent": False, "cpu_controls_scope": "structural_only"}
    execution["device"] = EXECUTION_DEVICE
    variant["resources"].update(device=EXECUTION_DEVICE, gpus=1,
                                gpu_memory_mb=max(MIN_GPU_MEMORY_MB, parent["resources"]["gpu_memory_mb"]))
    if "allow_cpu" in variant["resources"]:
        variant["resources"]["allow_cpu"] = False
    if parent_id == "two_pole":
        # Explicitly materialize the original host's direct table / real batch;
        # these are not public Recipe resource defaults or an architecture menu.
        execution["resources"] = {"num_particles": 12, "z_dim": 1, "batch_size": 12}
        execution["policy_resource_sources"] = {
            path: sources[path] for path in (
                "benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py")}
    for field in ("continuation_of", "execution_group"):
        if field in execution:
            execution[field] += SUFFIX
    for dependency in variant["dependencies"]:
        dependency["task"] += SUFFIX
    variant["evaluation"]["scoring_weights"] = "state_selected"
    variant["evaluation"]["policy_observation"] = _observation(parent)
    caps = ["served_sampling" if name == "live_sampling" else (
        "particle_cloud" if name == "mog_prior" else name) for name in variant["requires_capabilities"]]
    variant["requires_capabilities"] = list(dict.fromkeys(caps + ["policy_controls", "policy_serving"]))
    return variant


def _validate_variant(variant, parent):
    if not is_policy_task(variant):
        raise ValueError("missing explicit policy task contract/cohort")
    parent_id = _parent_id(variant)
    if parent.get("id") != parent_id or variant.get("id") != parent_id + SUFFIX:
        raise ValueError("policy variant ID/parent mismatch")
    pin = variant["policy_parent"]
    if (_execution_fingerprint(parent) != pin["execution_fingerprint"]
            or _hash(parent["evaluation"]) != pin["evaluation_fingerprint"]):
        raise ValueError("policy parent execution/evaluation fingerprint drift")
    contract = variant["execution"]["policy_contract"]
    _source_manifest(contract.get("sources"))
    expected = _prospective_variant(parent, pin, contract["sources"])
    if _hash(expected) != _hash(variant):
        raise ValueError("policy variant changes a frozen parent field or its explicit policy transformation")
    policy_recipe_overrides(variant)


def load_policy_variants(root, parent_tasks):
    """Load all 26 new declarations, verify exact parents and on-disk sources.

    Returns only variants; callers retain/merge the original task mapping. It
    performs no imports of evaluators, recipes, models or training artifacts.
    """
    root = Path(root)
    directory = root / "configs/forge/task-variants" / COHORT
    variants = {}
    for path in sorted(directory.glob("*.json")):
        variant = json.loads(path.read_text())
        parent_id = _parent_id(variant)
        parent_path = root / "configs/forge/tasks" / f"{parent_id}.json"
        parent_bytes = parent_path.read_bytes()
        parent = json.loads(parent_bytes)
        if hashlib.sha256(parent_bytes).hexdigest() != variant["policy_parent"]["task_sha256"]:
            raise ValueError(f"{parent_id}: parent task byte identity drift")
        if parent_id not in parent_tasks or _hash(parent_tasks[parent_id]) != _hash(parent):
            raise ValueError(f"{parent_id}: supplied parent task differs from the bound declaration")
        _validate_variant(variant, parent)
        if path.stem != variant["id"] or variant["id"] in variants:
            raise ValueError("policy variant filename/ID mismatch or duplicate")
        sources = {**parent["evaluation"].get("sources", {}),
                   **variant["execution"]["policy_contract"]["sources"]}
        for relative, digest in parent["evaluation"].get("sources", {}).items():
            if sources[relative] != digest:
                raise ValueError(f"{parent_id}: policy source cannot replace a parent evaluator binding")
        provenance = variant["execution"]["policy_recipe_overrides_provenance"]
        if provenance is not None:
            sources[provenance["source"]] = provenance["source_sha256"]
            selection = json.loads((root / SELECTION_SOURCE).read_text())
            group = selection["resolved_recipe_groups"][provenance["recipe_group"]]
            if (group["sha256"] != provenance["reference_recipe_sha256"]
                    or {name: group["recipe"][name] for name in OVERRIDE_FIELDS} != policy_recipe_overrides(variant)):
                raise ValueError("C6 override values differ from their bound reference Recipe")
        for relative, digest in sources.items():
            _source_manifest({**dict.fromkeys(REQUIRED_POLICY_SOURCES, "0" * 64), relative: digest})
            source_path = root / relative
            if (not source_path.is_file() or not source_path.resolve().is_relative_to(root.resolve())
                    or hashlib.sha256(source_path.read_bytes()).hexdigest() != digest):
                raise ValueError(f"{variant['id']}: source binding drift for {relative}")
        variants[variant["id"]] = variant
    if {name.removesuffix(SUFFIX) for name in variants} != set(PARENT_TASK_IDS):
        raise ValueError("policy cohort requires exactly the original 26 main-view questions")
    for variant in variants.values():
        if any(entry["task"] not in variants for entry in variant["dependencies"]):
            raise ValueError("policy dependency refers outside the prospective cohort")
        continuation = variant["execution"].get("continuation_of")
        if continuation is not None and continuation not in variants:
            raise ValueError("policy continuation refers outside the prospective cohort")
    return variants


def resolve_policy_view(view, tasks, candidate):
    """Resolve explicit opt-in without changing the live declaration or evidence."""
    cohort = candidate.get("task_cohort")
    if cohort is None:
        return deepcopy(view), deepcopy(tasks)
    if cohort != COHORT:
        raise ValueError(f"unknown explicit policy task_cohort {cohort!r}")
    assignments = view.get("assignments", [])
    if (view.get("id") != PARENT_VIEW_ID or view.get("goal") != PARENT_VIEW_ID
            or view.get("revision") != PARENT_REVISION
            or [entry["task"] for entry in assignments] != list(PARENT_TASK_IDS)
            or [sum(entry["qualification_tier"] == tier for entry in assignments) for tier in (1, 2, 3)] != [5, 19, 2]
            or any(entry.get("importance") != "required"
                   or entry.get("qualification_tier") != (1 if index < 5 else 2 if index < 24 else 3)
                   or entry.get("order") != index - (0 if index < 5 else 5 if index < 24 else 24)
                   for index, entry in enumerate(assignments))):
        raise ValueError("policy cohort resolves only the unchanged revision-3 main 5/19/2 view")
    selected = {}
    for parent_id in PARENT_TASK_IDS:
        variant_id = parent_id + SUFFIX
        if parent_id not in tasks or variant_id not in tasks:
            raise ValueError(f"policy cohort lacks parent/variant {parent_id}")
        _validate_variant(tasks[variant_id], tasks[parent_id])
        selected[variant_id] = deepcopy(tasks[variant_id])
    resolved = deepcopy(view)
    resolved.update(revision=REVISION, task_cohort=COHORT, parent_view_fingerprint=_hash(view),
                    cohort_fingerprint=_hash(selected),
                    policy_scope="Prospective CUDA Atlas independent policy-selected cloud tasks; auto density backend and settled reopen guard are required. CPU checks are structural only. E22 and historical CPU/live/MoG evidence confer no credit.")
    for entry in resolved["assignments"]:
        entry["task"] += SUFFIX
    return resolved, selected


def _recipe_values(recipe):
    if isinstance(recipe, dict):
        return recipe
    if is_dataclass(recipe) and not isinstance(recipe, type):
        return asdict(recipe)
    # Public Recipe-like objects in pure software controls; never instantiate.
    return {name: getattr(recipe, name) for name in set(REQUIRED_RECIPE) | OVERRIDE_FIELDS | {"lr", "prior_lr_mult"}
            if hasattr(recipe, name)}


def policy_contract_blockers(task, recipe):
    """Pure static preflight; runtime adapters must prove ownership/lifecycle.

    It does not count controls, build a host, sample, relax a gate or confer
    learned qualification. Incompatible routed/AE hosts stay blocked by their
    separately implemented execution-path ownership checks.
    """
    try:
        if not is_policy_task(task):
            raise ValueError("an explicit policy-selected cloud declaration is required")
        parent_id = _parent_id(task)
        if task["id"] != parent_id + SUFFIX:
            raise ValueError("policy variant ID/parent mismatch")
        execution = task["execution"]
        contract = execution["policy_contract"]
        _source_manifest(contract.get("sources"))
        if _hash(contract) != _hash(_contract(execution.get("execution_path", "public_trainer"), contract["sources"])):
            raise ValueError("policy contract differs from ordered public policy lifecycle")
        if contract["execution_path"] not in {"public_trainer", "public_components"}:
            raise ValueError("policy execution_path is unsupported")
        resources = task.get("resources", {})
        if (execution.get("device") != EXECUTION_DEVICE or resources.get("device") != EXECUTION_DEVICE
                or type(resources.get("gpus")) is not int or resources["gpus"] != 1
                or type(resources.get("gpu_memory_mb")) is not int
                or resources["gpu_memory_mb"] < MIN_GPU_MEMORY_MB
                or resources.get("allow_cpu", False) is not False):
            raise ValueError("prospective numerical policy attempts require explicit CUDA resources; CPU controls are structural only")
        prior = execution.get("prior", {})
        if (type(execution.get("steps")) is not int or execution["steps"] <= 0
                or prior.get("kind") != "particle_cloud"
                or not _finite(prior.get("sigma")) or prior["sigma"] != 0
                or prior.get("standardize") is not False or prior.get("learnable") is not True):
            raise ValueError("policy requires its external full task horizon and explicit unstandardized learned cloud")
        validate_policy_observation(task)
        overrides = policy_recipe_overrides(task)
        values = _recipe_values(recipe)
        blockers = []
        for name, expected in {**REQUIRED_RECIPE, **overrides}.items():
            actual = values.get(name, "<undeclared>")
            if _hash(actual) != _hash(expected) or (type(expected) is bool and type(actual) is not bool):
                blockers.append(f"{task['id']}: effective Recipe.{name} must equal {expected!r}")
        if not _finite(values.get("lr")) or values["lr"] <= 0:
            blockers.append(f"{task['id']}: effective Recipe.lr must be finite and positive")
        if not _finite(values.get("prior_lr_mult")) or values["prior_lr_mult"] <= 0:
            blockers.append(f"{task['id']}: effective Recipe.prior_lr_mult must be finite and positive")
        return blockers
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"{task.get('id', '<task>')}: {exc}"]
