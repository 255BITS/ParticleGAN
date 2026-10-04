"""Explicit dispatch of prospective policy laws; no historical reinterpretation.

Contract modules are imported lazily from a fixed whitelist. No producer,
model, evaluator or trained artifact is imported by this registry.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path


_DISPATCH = {
    "policy_selected_cloud_v1": ("policy_contracts", None, None,
        "validate_policy_observation", "policy_recipe_overrides", "policy_contract_blockers"),
    "conditional_policy_selected_cloud_v1": ("conditional_policy_contracts", "atlas_conditional",
        "validate_conditional_task", "validate_conditional_observation", "conditional_recipe_overrides",
        "conditional_policy_contract_blockers"),
    "routed_policy_selected_cloud_v1": ("routed_policy_contracts", "atlas_routed",
        "validate_routed_task", "validate_routed_observation", "routed_recipe_overrides", "routed_policy_contract_blockers"),
    "multibank_policy_v1": ("multibank_policy_contracts", "atlas_multibank", "validate_task",
        "observation", None, "blockers"),
    "ae_routed_policy_v1": ("ae_routed_policy_contracts", "atlas_ae_routed", "validate_ae_task",
        "validate_ae_observation", "ae_recipe_overrides", "ae_policy_contract_blockers"),
    "word_joint_policy_min11_v1": ("word_joint_policy_contracts", "atlas_word_joint_min11", "validate_task",
        "validate_word_observation", "word_recipe_overrides", "blockers"),
}
KNOWN_COHORTS = frozenset(_DISPATCH)
COMPILER_ANNOTATIONS = frozenset({"preflight_blockers", "field_ownership"})


def policy_task_declaration(task, *, allow_compiler_annotations=True):
    """Remove only typed, inert annotations appended by ``resolve_idea``.

    The complete remaining object still has to equal its canonical parent
    transformation. Raw JSON loaders can explicitly prohibit annotations;
    compiled requests retain their original scientific source fingerprints.
    Neither annotation is a runtime override or policy/qualification receipt.
    """
    if not isinstance(task, dict):
        raise ValueError("policy task must be a declared mapping")
    annotations = COMPILER_ANNOTATIONS & task.keys()
    if not annotations:
        return task
    if allow_compiler_annotations is not True:
        raise ValueError("raw policy declaration cannot contain compiler annotations")
    if "preflight_blockers" in annotations:
        blockers = task["preflight_blockers"]
        if not isinstance(blockers, list) or any(not isinstance(reason, str) for reason in blockers):
            raise ValueError("compiled preflight_blockers must be a list of strings")
    if "field_ownership" in annotations:
        ownership = task["field_ownership"]
        required = {"schema_version", "version", "task_id", "recipe_fields", "task_contract",
                    "delegated_reference_values", "inactive_legacy_host_fields", "reference_declarations"}
        if (not isinstance(ownership, dict) or not required <= ownership.keys()
                or ownership.keys() - required - {"protocol"}
                or type(ownership["schema_version"]) is not int or ownership["schema_version"] != 1
                or ownership["version"] != "forge-field-boundaries-v2"
                or ownership["task_id"] != task.get("id")
                or any(not isinstance(ownership[key], dict) for key in required - {
                    "schema_version", "version", "task_id"})
                or ("protocol" in ownership and not isinstance(ownership["protocol"], dict))):
            raise ValueError("compiled field_ownership must be the typed task-specific ownership receipt")
        try:
            json.dumps(ownership, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("compiled field_ownership must contain finite JSON metadata") from error
    return {key: deepcopy(value) for key, value in task.items() if key not in annotations}


def is_policy_task(task):
    """Any declared policy_contract, including a malformed one, needs validation."""
    return (isinstance(task, dict) and isinstance(task.get("execution"), dict)
            and "policy_contract" in task["execution"])


def module_for_task(task):
    if not is_policy_task(task):
        raise ValueError("explicit named policy task contract required")
    contract = task["execution"]["policy_contract"]
    if not isinstance(contract, dict):
        raise ValueError("policy_contract must be a declared mapping")
    cohort = task.get("task_cohort")
    if (not isinstance(cohort, str) or cohort not in KNOWN_COHORTS
            or contract.get("cohort") != cohort):
        raise ValueError("unknown or contradictory named policy cohort")
    name, family, *_ = _DISPATCH[cohort]
    if family is not None and (task.get("policy_family") != family or contract.get("family") != family):
        raise ValueError("named policy family differs from its whitelisted cohort")
    return importlib.import_module("." + name, __package__)


def _parent(task):
    record = task.get("policy_parent")
    if (not isinstance(record, dict) or set(record) != {
            "id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or not isinstance(record.get("id"), str) or not record["id"]
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in record["id"])
            or any(type(record[k]) is not str or len(record[k]) != 64
                   or any(c not in "0123456789abcdef" for c in record[k])
                   for k in ("task_sha256", "execution_fingerprint", "evaluation_fingerprint"))):
        raise ValueError("named policy requires a complete original parent identity")
    return record


def _validate_named_parent(task, module, root):
    """Validate the complete transform even for frozen, source-declared grading.

    With no explicit root, source hashes remain declarations for the grader's
    separately verified snapshot. Parent task bytes and every transformed field
    are still checked; current implementation hashes never replace frozen ones.
    """
    pin = _parent(task)
    directory = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    raw = (directory / "configs/forge/tasks" / (pin["id"] + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["task_sha256"]:
        raise ValueError("named policy original parent byte identity drift")
    parent = json.loads(raw)
    if (module._execution_hash(parent) != pin["execution_fingerprint"]
            or module.stable_hash(parent["evaluation"]) != pin["evaluation_fingerprint"]
            or module.stable_hash(task) != module.stable_hash(
                module._variant(parent, pin, task["execution"]["policy_contract"]["sources"]))):
        raise ValueError("named policy changed a frozen parent field or its explicit transformation")


def _validate_cloud(task, module, root):
    # The original module's public loader validates the complete 26-file set.
    # Context resolution validates one exact variant against the same parent
    # transformation without requiring the other 25 jobs to be executable.
    parent_id = module._parent_id(task)
    directory = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    raw = (directory / "configs/forge/tasks" / (parent_id + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != task["policy_parent"]["task_sha256"]:
        raise ValueError("policy original parent byte identity drift")
    parent = json.loads(raw)
    module._validate_variant(task, parent)
    if root is not None:
        sources = {**parent["evaluation"].get("sources", {}),
                   **task["execution"]["policy_contract"]["sources"]}
        for relative, digest in parent["evaluation"].get("sources", {}).items():
            if sources[relative] != digest:
                raise ValueError("policy source cannot replace its parent evaluator binding")
        provenance = task["execution"]["policy_recipe_overrides_provenance"]
        if provenance is not None:
            sources[provenance["source"]] = provenance["source_sha256"]
        for relative, digest in sources.items():
            path = directory / relative
            if (not path.resolve().is_relative_to(directory) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError("policy source binding drift: " + relative)
    return deepcopy(task["execution"]["policy_contract"])


def validate_policy_task(task, root=None, *, allow_compiler_annotations=True):
    task = policy_task_declaration(task, allow_compiler_annotations=allow_compiler_annotations)
    module = module_for_task(task)
    _parent(task)
    validator = _DISPATCH[task["task_cohort"]][2]
    if validator is None:
        return _validate_cloud(task, module, root)
    contract = getattr(module, validator)(task, root=root)
    _validate_named_parent(task, module, root)
    return contract


def validate_policy_observation(task):
    task = policy_task_declaration(task)
    validate_policy_task(task)
    module = module_for_task(task)
    function = getattr(module, _DISPATCH[task["task_cohort"]][3])
    return deepcopy(function() if task["task_cohort"] == "multibank_policy_v1" else function(task))


def policy_recipe_overrides(task):
    task = policy_task_declaration(task)
    validate_policy_task(task)
    module = module_for_task(task)
    name = _DISPATCH[task["task_cohort"]][4]
    return deepcopy(task["execution"]["policy_recipe_overrides"] if name is None else getattr(module, name)(task))


def policy_contract_blockers(task, recipe):
    try:
        task = policy_task_declaration(task)
        validate_policy_task(task)
        module = module_for_task(task)
        return getattr(module, _DISPATCH[task["task_cohort"]][5])(task, recipe)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        return [f"{task.get('id', '<task>') if isinstance(task, dict) else '<task>'}: {error}"]
