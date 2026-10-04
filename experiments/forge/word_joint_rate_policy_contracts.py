"""One prospective half-base rate contrast for the original N11/free-E joint law.

This module resolves metadata through public Recipe only. It never constructs
models, optimizers, policies, samples or gates, and confers no learned credit.
The existing C6 declaration remains a separate, immutable scientific identity.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from .contracts import stable_hash
from .priors import resolve_prior
from . import word_joint_policy_contracts as original

COHORT = "word_joint_policy_min11_rates_v1"
FAMILY = "atlas_word_joint_min11_rates"
KIND = "forge_word_joint_policy_min11_rates_v1"
PARENT_ID = original.PARENT_ID
TASK_ID = PARENT_ID + "_" + COHORT
HOST_RESOURCES = deepcopy(original.HOST_RESOURCES)
RESOURCE_ADAPTATION = deepcopy(original.RESOURCE_ADAPTATION)
THRESHOLDS = deepcopy(original.THRESHOLDS)
RATE_PROFILES = MappingProxyType({
    "half_base": (.00265625, 1.5, 1.0),
})
PROFILE_IDS = MappingProxyType({p: "word-min11-" + p + "-rates-v1" for p in RATE_PROFILES})
RATE_FIELDS = ("lr", "prior_lr_mult", "d_lr_mult")
SOURCES = tuple(sorted(set(original.SOURCES) | {
    "experiments/forge/word_joint_rate_policy_contracts.py",
    "experiments/forge/named_policy_planning.py", "experiments/forge/adapters.py",
    "experiments/forge/policy_snapshot_publication.py",
    "experiments/forge/priors.py",
}))
ROOT = Path(__file__).resolve().parents[2]


def rates(profile):
    if type(profile) is not str or profile not in RATE_PROFILES:
        raise ValueError("unknown prospective word rate profile")
    return dict(zip(RATE_FIELDS, RATE_PROFILES[profile]))


def candidate(profile, *, identifier=None):
    """Pure minimal caller declaration; the root protocol still owns budgets."""
    values = rates(profile)
    if identifier is not None and identifier != PROFILE_IDS[profile]:
        raise ValueError("prospective word tuple ID is fixed by its declared profile")
    return dict(id=PROFILE_IDS[profile],
        recipe_preset="atlas", task_cohort=COHORT, trainer_family=FAMILY,
        execution_path="public_trainer", word_rate_profile=profile,
        prior=dict(kind="particle_cloud", sigma=0., standardize=False, learnable=True,
            exception_reason="Explicit N11/free-E same-code word law; no original N5 or C6 qualification reuse."),
        recipe_overrides=values)


def contract(sources):
    result = original.contract(sources)
    result.update(cohort=COHORT, family=FAMILY, prospective_rate_binding=dict(
        schema_version=1, owner="particlegan.Recipe", profiles={p: rates(p) for p in RATE_PROFILES},
        tuple_ids=dict(PROFILE_IDS),
        reference_execution_path="public_trainer", actual_execution_path="public_components",
        profile_selector="candidate.word_rate_profile", optimizer_owner="original_public_Recipe_factories",
        runtime_override=False, mechanism_changes=False, original_C6_qualification_reuse=False,
        capacity_reuse=False, qualification_credit="requires_actual_gates"))
    return result


def observation():
    return original.observation()


def _execution_hash(task):
    return original._execution_hash(task)


def _variant(parent, pin, sources):
    result = original._variant(parent, pin, sources)
    result.update(id=TASK_ID, task_cohort=COHORT, policy_family=FAMILY)
    result["execution"]["policy_contract"] = contract(sources)
    result["execution"]["policy_recipe_overrides_provenance"]["family"] = FAMILY
    return result


def make_variant(root):
    root = Path(root).resolve()
    raw = (root / f"configs/forge/tasks/{PARENT_ID}.json").read_bytes()
    parent = json.loads(raw)
    pin = dict(id=PARENT_ID, task_sha256=hashlib.sha256(raw).hexdigest(),
        execution_fingerprint=_execution_hash(parent), evaluation_fingerprint=stable_hash(parent["evaluation"]))
    sources = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES}
    return _variant(parent, pin, sources)


def validate_task(task, *, root=None):
    from .policy_cohorts import policy_task_declaration
    task = policy_task_declaration(task)
    if (not isinstance(task, dict) or task.get("id") != TASK_ID
            or task.get("task_cohort") != COHORT or task.get("policy_family") != FAMILY):
        raise ValueError("explicit prospective min11 word-rate task required")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    if (not isinstance(sources, dict) or set(sources) != set(SOURCES)
            or any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                   for h in sources.values())):
        raise ValueError("complete prospective word-rate source binding required")
    directory = ROOT if root is None else Path(root).resolve()
    raw = (directory / f"configs/forge/tasks/{PARENT_ID}.json").read_bytes()
    parent = json.loads(raw)
    pin = dict(id=PARENT_ID, task_sha256=hashlib.sha256(raw).hexdigest(),
        execution_fingerprint=_execution_hash(parent), evaluation_fingerprint=stable_hash(parent["evaluation"]))
    if stable_hash(task) != stable_hash(_variant(parent, pin, sources)):
        raise ValueError("prospective word-rate parent/source/law/gate/case identity drift")
    if root is not None and any(hashlib.sha256((directory/p).read_bytes()).hexdigest() != h
                                for p,h in sources.items()):
        raise ValueError("prospective word-rate implementation source drift")
    return deepcopy(task["execution"]["policy_contract"])


def validate_word_observation(task):
    validate_task(task)
    return observation()


def word_recipe_overrides(task):
    validate_task(task)
    return deepcopy(task["execution"]["policy_recipe_overrides"])


def profile_for(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("prospective word-rate candidate must be a mapping")
    profile = candidate.get("word_rate_profile")
    expected = rates(profile)
    actual = candidate.get("recipe_overrides")
    prior = resolve_prior(candidate.get("prior"), explicit=True)
    if (candidate.get("id") != PROFILE_IDS[profile]
            or candidate.get("task_cohort") != COHORT or candidate.get("trainer_family") != FAMILY
            or candidate.get("recipe_preset") != "atlas"
            or candidate.get("execution_path") != "public_trainer"
            or candidate.get("extensions", {}) != {} or candidate.get("host_adaptation", {}) != {}
            or candidate.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"
            or set(prior) != {"kind", "sigma", "standardize", "learnable", "exception_reason"}
            or prior["kind"] != "particle_cloud" or prior["sigma"] != 0.
            or prior["standardize"] is not False or prior["learnable"] is not True
            or not isinstance(actual, dict)
            or set(actual) != set(RATE_FIELDS)
            or any(type(actual[k]) not in (int,float) or not math.isfinite(actual[k])
                   or actual[k] != expected[k] for k in RATE_FIELDS)):
        raise ValueError("prospective word rate profile/tuple/family/preset binding differs")
    return profile


def resolved_recipe(candidate, task):
    validate_task(task)
    profile = profile_for(candidate)
    return _profile_recipe(profile)


def _profile_recipe(profile):
    from particlegan import get_recipe
    return get_recipe("atlas", **rates(profile), **HOST_RESOURCES, encoder_mode="none",
                      row_policy="independent", prior_kind="particles", sigma_rel=0., standardize=False)


def validate_request(request, task, *, root=None):
    if (not isinstance(request, dict) or not isinstance(request.get("protocol"), dict)
            or type(request["protocol"].get("seed")) is not int or request["protocol"]["seed"] != 0):
        raise ValueError("prospective word rates preserve the original seed-zero named protocol")
    validate_task(task, root=root)
    return resolved_recipe(request.get("candidate"), task)


def binding_receipt(recipe, profile):
    value = recipe.to_dict() if hasattr(recipe,"to_dict") else recipe
    expected = _profile_recipe(profile).to_dict()
    if stable_hash(value) != stable_hash(expected):
        raise ValueError("complete prospective word-rate Recipe differs")
    return dict(schema_version=1, profile=profile, tuple_id=PROFILE_IDS[profile],
        overrides=rates(profile), owner="particlegan.Recipe",
        resolved_recipe=deepcopy(value), resolved_recipe_sha256=stable_hash(value))


def validate_binding_receipt(receipt, task):
    validate_task(task)
    if not isinstance(receipt, dict):
        raise ValueError("missing actual prospective word-rate Recipe receipt")
    profile = receipt.get("profile")
    expected_recipe = resolved_recipe(candidate(profile), task)
    expected = dict(schema_version=1, profile=profile, tuple_id=PROFILE_IDS[profile],
        overrides=rates(profile), owner="particlegan.Recipe",
        resolved_recipe=expected_recipe.to_dict(), resolved_recipe_sha256=stable_hash(expected_recipe.to_dict()))
    if stable_hash(receipt) != stable_hash(expected):
        raise ValueError("actual prospective word-rate profile/complete Recipe receipt differs")
    return deepcopy(expected)


def validate_result_binding(request, task, result):
    """Bind a numerically graded receipt to its actual requested profile.

    This reads JSON metadata only. Typed checkpoint/array/source identity and
    the unchanged numerical gates remain the ordinary grader's responsibility.
    """
    recipe = validate_request(request, task)
    profile = profile_for(request["candidate"])
    expected = binding_receipt(recipe, profile)
    if not isinstance(result, dict):
        raise ValueError("prospective word result must be a mapping")
    applied = result.get("applied")
    evidence = result.get("evidence")
    if (result.get("task_id") != TASK_ID or not isinstance(applied, dict) or not isinstance(evidence, dict)
            or applied.get("family") != FAMILY or applied.get("task_cohort") != COHORT
            or applied.get("execution_path") != "public_components"
            or stable_hash(applied.get("actual_resources")) != stable_hash(HOST_RESOURCES)
            or stable_hash(applied.get("recipe")) != stable_hash(recipe.to_dict())):
        raise ValueError("actual applied word Recipe/family differs from the requested rate profile")
    lifecycle = applied.get("policy_lifecycle")
    controls = evidence.get("policy_controls")
    if (not isinstance(lifecycle, dict) or lifecycle.get("owner") != "particlegan.UpdatePolicy"
            or not isinstance(lifecycle.get("controls"), dict) or not isinstance(controls, dict)
            or stable_hash(controls.get("word_rate_binding")) != stable_hash(expected)
            or stable_hash(lifecycle["controls"].get("word_rate_binding")) != stable_hash(expected)):
        raise ValueError("actual word rate receipts differ from the requested profile/complete Recipe")
    return deepcopy(expected)


def blockers(task, recipe):
    try:
        validate_task(task)
        value = recipe.to_dict() if hasattr(recipe,"to_dict") else recipe
        if not any(stable_hash(value) == stable_hash(resolved_recipe(candidate(p), task).to_dict())
                   for p in RATE_PROFILES):
            raise ValueError("prospective word Recipe is not the complete declared half-base profile")
        return []
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        return [str(error)]
