"""Frozen architecture/initialization transfer for opt-in native tasks.

The original MLP tasks are unchanged. This profile borrows a declared affine
architecture and resources, not historical seeds, recipe settings or results.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .contracts import canonical, identifier, stable_hash

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "native_affine_square_named_v1"
PROFILE_SOURCE = "configs/toy100/constraints_simple_regularization.json"
PROFILE_SHA256 = "4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7"
COMPONENTS = {"generator", "discriminator", "prior"}


def profile_declaration():
    return {"id": PROFILE_ID, "revision": 1, "source": {"path": PROFILE_SOURCE, "sha256": PROFILE_SHA256}}


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def resolve_host_initialization(value, *, initializer="deterministic_orthogonal", prior=None, requirements=None):
    """Validate task-owned component policies before construction or RNG draws.

    Requirements are a future explicit API constraint, not the legacy fallback
    initializer string. No new idea-file field is implicitly accepted here.
    """
    if requirements is not None and not isinstance(requirements, dict):
        raise ValueError("invalid explicit component initializer requirements")
    if value is None:
        if requirements:
            raise ValueError("explicit component initializer requirements need a declared host policy")
        return None
    if not isinstance(value, dict) or set(value) != {"schema_version", "owner", "profile_sha256", "components"}:
        raise ValueError("host initialization has missing or unsupported fields")
    digest = value["profile_sha256"]
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["owner"] != "task"
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("invalid host initialization identity")
    policies = value["components"]
    if not isinstance(policies, dict) or not policies or not set(policies) <= COMPONENTS:
        raise ValueError("host initialization needs known component policies")
    if initializer == "supplied":
        raise ValueError("supplied initialization conflicts with task-pinned component policies")
    if requirements is not None and (not isinstance(requirements, dict) or not set(requirements) <= COMPONENTS):
        raise ValueError("invalid explicit component initializer requirements")
    for component, policy in policies.items():
        if not isinstance(policy, dict):
            raise ValueError("component initializer must be a declaration, not a callback")
        method = policy.get("method")
        if method == "identity_linear_v1":
            if set(policy) != {"method"}:
                raise ValueError("identity initializer has unsupported fields")
        elif method == "xavier_uniform_zero_bias_v1":
            if set(policy) != {"method", "gain"} or not _positive(policy["gain"]):
                raise ValueError("Xavier initializer requires a positive finite gain")
        elif method == "sample_distributions_v1":
            parameters = policy.get("parameters")
            if set(policy) != {"method", "parameters"} or not isinstance(parameters, dict) or not parameters:
                raise ValueError("sample initializer requires explicit parameter distributions")
            for name, distribution in parameters.items():
                if not isinstance(name, str) or not name or not isinstance(distribution, dict):
                    raise ValueError("invalid parameter distribution declaration")
                kind = distribution.get("kind")
                expected = {"uniform": {"kind", "low", "high"}, "normal": {"kind", "mean", "std"}, "keep": {"kind"}}
                if kind not in expected or set(distribution) != expected[kind]:
                    raise ValueError("unsupported literal parameter distribution")
                numeric = [v for k, v in distribution.items() if k != "kind"]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
                    raise ValueError("distribution bounds must be finite numbers")
                if kind == "uniform" and distribution["low"] >= distribution["high"]:
                    raise ValueError("uniform initialization requires low < high")
                if kind == "normal" and distribution["std"] <= 0:
                    raise ValueError("normal initialization requires positive std")
        else:
            raise ValueError("unsupported component initialization method")
        if component == "prior" and (method != "sample_distributions_v1" or set(policy["parameters"]) != {"z"}):
            raise ValueError("public prior initialization requires an explicit z distribution")
        if component in (requirements or {}) and canonical(requirements[component]) != canonical(policy):
            raise ValueError(f"candidate initializer requirement conflicts with pinned {component} policy")
    if set(requirements or {}) - set(policies):
        raise ValueError("required component initializer is not pinned by this host")
    if "prior" in policies and "init_std" in (prior or {}):
        raise ValueError("explicit candidate prior.init_std conflicts with task-pinned location initialization")
    return deepcopy(value)


def _profile_spec(root):
    data = (Path(root) / PROFILE_SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest() != PROFILE_SHA256:
        raise ValueError("native profile source changed; declare a new profile revision")
    source = json.loads(data)
    if source["toy100_model"] != "affine_square_v1" or source["z_dim"] != 2:
        raise ValueError("native source no longer declares the selected affine host")
    return {"generator": {"kind": "linear", "in_features": source["z_dim"], "out_features": 2, "bias": True},
        "discriminator": {"kind": "simple_mlp", "in_dim": 2, "hidden_dim": source["d_hidden"],
                          "n_hidden": source["n_hidden"], "fourier": source["fourier"]},
        "resources": {key: source[key] for key in ("z_dim", "num_particles", "batch_size")},
        "initialization": {"generator": {"method": "identity_linear_v1"},
            "discriminator": {"method": "xavier_uniform_zero_bias_v1", "gain": 1.0},
            "prior": {"method": "sample_distributions_v1", "parameters": {"z": {"kind": "uniform", "low": -5., "high": 5.}}}}}


def _measurement(task):
    from benchmarks.toy100.accuracy import LIMITS
    from benchmarks.toy100.metrics import REQUIREMENTS
    from .sampling import PUBLIC_PRIOR_CLEAN, executed_receipt
    expected = {"kind": "native_accuracy", "scoring_weights": "live", "eval_interval": 250,
        "early_eval_steps": [0, 1, 10, 25, 50, 100], "eval_samples": 20000, "holdout_samples": 100000,
        "minimum_stable_checks": 5, "coverage_thresholds": REQUIREMENTS, "accuracy_limits": LIMITS,
        "evaluator": "benchmarks.toy100.accuracy_gate:evaluate_suite",
        "coverage_evaluator": "benchmarks.toy100.gate:evaluate_suite"}
    expected.update(executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"))
    if any(canonical(task["evaluation"].get(key)) != canonical(value) for key, value in expected.items()):
        raise ValueError("native profile must retain the full frozen clean live coverage and accuracy protocol")


def resolve_native_spec(task, *, root=None):
    """Return a validated explicit profile, or None for an unchanged old task."""
    execution = task["execution"]
    if "native_profile" not in execution:
        if task.get("adapter") in {"native100", "native100_continuation"} and ({"host_definition", "host_initialization", "resources", "initialization"} & set(execution)):
            raise ValueError("native model overrides require an explicit profile")
        return None
    if task.get("adapter") not in {"native100", "native100_continuation"}:
        raise ValueError("native profiles apply only to native tasks")
    if canonical(execution["native_profile"]) != canonical(profile_declaration()):
        raise ValueError("unsupported or mismatched native profile declaration")
    if "host_initialization" in execution:
        raise ValueError("native component policies belong in the frozen host_definition, not a second override")
    if {"resources", "initialization"} & set(execution):
        raise ValueError("native resource and initializer fields belong in the frozen host_definition")
    expected = _profile_spec(ROOT if root is None else root)
    if canonical(execution.get("host_definition")) != canonical(expected):
        raise ValueError("native host card differs from its frozen profile")
    if execution.get("problem") not in {"grid100", "rotated100", "staggered100"}:
        raise ValueError("unsupported native profile problem")
    continuation = task["adapter"] == "native100_continuation"
    if execution.get("steps") != (14000 if continuation else 7000):
        raise ValueError("native profile must retain its complete qualification horizon")
    if continuation and any(execution.get(key) != value for key, value in {
            "incremental_steps": 7000, "preserve_prefix_steps": 7000, "original_schedule_horizon": 7000}.items()):
        raise ValueError("native continuation must preserve its original 7k schedule and prefix")
    if execution.get("original_schedule_horizon", 7000) != 7000:
        raise ValueError("native profile schedule horizon must remain 7000")
    _measurement(task)
    return deepcopy(expected)


def native_profile_source_files(task):
    if "native_profile" not in task.get("execution", {}):
        return []
    if canonical(task["execution"]["native_profile"]) != canonical(profile_declaration()):
        raise ValueError("unsupported native profile source declaration")
    return [PROFILE_SOURCE]


def native_host_initialization(task, *, root=None):
    spec = resolve_native_spec(task, root=root)
    if spec is None:
        return None
    # No task ID/problem/observation/budget fields: 7k and its 14k own-state
    # continuation must construct precisely the same initial static context.
    return {"schema_version": 1, "owner": "task", "profile_sha256": stable_hash({
        "native_profile": task["execution"]["native_profile"], "host_definition": spec}),
        "components": deepcopy(spec["initialization"])}


def native_profile_blockers(task, candidate, *, root=None):
    try:
        spec = resolve_native_spec(task, root=root)
        if spec is None:
            return []
        prior = candidate.get("prior", task["execution"].get("prior", {}))
        if (not isinstance(prior, dict) or prior.get("kind") != "mog" or not _positive(prior.get("sigma")) or prior.get("standardize") is not False
                or prior.get("learnable") is not True or set(prior) - {"kind", "sigma", "standardize", "learnable", "init_std"}):
            raise ValueError("native affine profile requires learned MoG locations, positive explicit width, no standardization and uniform masses")
        resolve_host_initialization(native_host_initialization(task, root=root), initializer=candidate.get("initializer", "deterministic_orthogonal"), prior=prior)
        fixed = {**spec["resources"], "total_steps": 7000}
        for key, value in fixed.items():
            if key in candidate.get("recipe_overrides", {}) and candidate["recipe_overrides"][key] != value:
                raise ValueError(f"candidate overrides frozen native host resource {key}")
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"{task.get('id', '<task>')}: {error}"]
    return []


def validate_native_continuation(parent, child, *, root=None):
    a, b = resolve_native_spec(parent, root=root), resolve_native_spec(child, root=root)
    if a is None and b is None:
        return
    if (a is None or b is None or canonical(a) != canonical(b)
            or parent["adapter"] != "native100" or child["adapter"] != "native100_continuation"
            or parent["execution"]["problem"] != child["execution"]["problem"]
            or child["execution"].get("continuation_of") != parent["id"]
            or {"task": parent["id"], "kind": "checkpoint"} not in child.get("dependencies", [])):
        raise ValueError("native continuation must bind its matching profile and own 7k parent")
    if native_host_initialization(parent, root=root) != native_host_initialization(child, root=root):
        raise ValueError("native continuation changes host initialization identity")


def task_from_profile(base_task, task_id, *, parent_task_id=None, root=None):
    identifier(task_id, "native profile task id")
    if task_id == base_task["id"]:
        raise ValueError("native profile needs a distinct task id")
    task = deepcopy(base_task)
    task["id"] = task_id
    task["execution"].update(native_profile=profile_declaration(),
                             host_definition=_profile_spec(ROOT if root is None else root))
    if task["adapter"] == "native100_continuation":
        if not parent_task_id:
            raise ValueError("profile continuation requires its explicit matching parent")
        old = task["execution"]["continuation_of"]
        task["execution"]["continuation_of"] = parent_task_id
        for dependency in task["dependencies"]:
            if dependency["task"] == old:
                dependency["task"] = parent_task_id
    resolve_native_spec(task, root=root)
    return task


def build_native_models(context, spec):
    """Use the pure shared model classes; context owns init, prior and trainer."""
    from benchmarks.toy100.native_models import native_component
    return tuple(context.construct(lambda role=role: native_component(spec[role], role=role), component=role).to(context.device)
                 for role in ("generator", "discriminator"))
