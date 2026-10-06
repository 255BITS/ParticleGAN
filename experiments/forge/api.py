"""Forge's thin construction contract around the public ParticleGAN API.

There is no second optimizer or training loop here. New experiment variables
bind explicitly to a documented public Recipe or GANTrainer argument.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import math

import torch

from particlegan import GANTrainer, Recipe, get_recipe, init, prior_capabilities, prior_mechanisms
from .rng import NamedStreams, RNG_VERSION
from .priors import DEFAULT_PRIOR, resolve_prior


API_VERSION = "forge-api-v1"
TRAINER_STREAM_BINDINGS = {
    "latent_generator": ("prior", "latent", "indices"),
    "penalty_generator": ("noise", "penalty", "training"),
    "noise_generator": ("noise", "generator", "output"),
    "input_noise_generator": ("noise", "critic", "input"),
    "eval_generator": ("eval", "sampler", "samples"),
    "model_generator": ("noise", "models", "stochastic_layers"),
    "prior_noise_generator": ("noise", "prior", "gaussian"),
}
BUILTIN_CAPABILITIES = (
    "public_trainer", "public_components", "scalar_gan", "mog_prior", "particle_cloud", "learned_locations",
    "uniform_masses", "fixed_prior_width", "a2", "checkpoint", "named_rng", "live_sampling",
    "policy_controls", "policy_serving", "served_sampling",
)
_MISSING = object()


class CapabilityError(ValueError):
    """Pretraining incompatibility; callers record BLOCKED, not scientific FAIL."""
    status = "BLOCKED"

    def __init__(self, blockers):
        self.blockers = list(blockers)
        super().__init__("; ".join(self.blockers))


def resolve_public_recipe(candidate, **overrides):
    """Resolve a preset before task resources; a Recipe.name is only a label."""
    options = {**candidate.get("recipe_overrides", {}), **overrides}
    preset = candidate.get("recipe_preset")
    try:
        return Recipe(**options) if preset is None else get_recipe(preset, **options)
    except (TypeError, ValueError) as error:
        raise CapabilityError([f"unsupported recipe: {error}"]) from error


def policy_controls(recipe):
    """Mechanisms requiring the ordered public UpdatePolicy lifecycle."""
    return bool(recipe.continuous_policy is not None or recipe.lr_control == "stationarity"
                or recipe.row_evidence_gate or recipe.particle_birth_death
                or recipe.serve_average or recipe.output_noise_mode != "fixed")


def host_recipe_overrides(candidate, execution, resources):
    """Preserve schedule-free None; task steps bound GANTrainer.max_steps."""
    recipe = resolve_public_recipe(candidate)
    fixed = dict(resources)
    if recipe.total_steps is not None:
        fixed["total_steps"] = execution.get("original_schedule_horizon", execution["steps"])
    conflicts = [key for key, value in fixed.items()
                 if key in candidate.get("recipe_overrides", {})
                 and candidate["recipe_overrides"][key] != value]
    if conflicts:
        raise CapabilityError([f"candidate overrides frozen host resource {key}; declare a different task"
                               for key in sorted(conflicts)])
    return {**candidate.get("recipe_overrides", {}), **fixed}


def task_recipe_resources(task):
    """Resolve only the resources consumed by the task's existing adapter."""
    from .noisy_prior_tier1 import is_noisy_task
    if is_noisy_task(task):
        from .noisy_prior_adapters import task_resources
        return task_resources(task)
    execution, adapter = task["execution"], task["adapter"]
    if adapter in {"transfer_vector", "paired_adaptation", "transfer_image"}:
        spec = execution["host_definition"]
        return {"num_particles": spec["particles"], "z_dim": spec["z_dim"],
                "batch_size": spec["batch_size"] if adapter == "transfer_image" else spec["batch"]}
    if adapter == "ring_endurance" or (adapter == "transfer_behavior" and execution.get("host") == "mode_hold"):
        from benchmarks.legacy.locked_shared import LOCKED_SHARED
        return {"num_particles": LOCKED_SHARED.n_particles, "z_dim": 4, "batch_size": 128}
    if adapter in {"native100", "native100_continuation"} and "native_profile" in execution:
        return deepcopy(execution["host_definition"]["resources"])
    if adapter == "word_joint":
        return deepcopy(execution["host_definition"]["resources"])
    return deepcopy(execution.get("resources", {}))


def task_recipe_overrides(candidate, task):
    """Materialize task binding once, identically for planning and execution."""
    from .taskrecipes import bind_task_candidate
    bound = bind_task_candidate(candidate, task)
    execution = task["execution"]
    from .noisy_prior_tier1 import is_noisy_task, validate
    if is_noisy_task(task):
        from .noisy_prior_adapters import blockers, task_resources
        validate(task)
        reasons = blockers(task, bound)
        if reasons:
            raise CapabilityError(reasons)
        return host_recipe_overrides(bound, execution, task_resources(task))
    from .atlas_existing_mog import is_candidate as existing_mog_candidate, blockers as existing_mog_blockers, task_resources as existing_mog_resources
    if existing_mog_candidate(bound):
        reasons = existing_mog_blockers(task, bound)
        if reasons:
            raise CapabilityError(reasons)
        if task["id"] != "two_pole":
            return host_recipe_overrides(bound, execution, existing_mog_resources(task))
    from .atlas_two_pole import supports, TASK_BINDINGS
    if supports(task, bound):
        # The public Atlas schedule remains intrinsic None; 80 bounds the host.
        return {**bound.get("recipe_overrides", {}), **TASK_BINDINGS}
    if (task["adapter"] == "transfer_behavior" and execution.get("host") != "mode_hold"
            and task.get("task_cohort") != "tier1_policy_selected_cloud_v1"):
        from .behavior_adapters import behavior_preflight
        blockers = behavior_preflight(task, bound)
        if blockers:
            raise CapabilityError(blockers)
        overrides = {**bound.get("recipe_overrides", {}),
                     "total_steps": execution.get("original_schedule_horizon", execution["steps"])}
        if execution.get("host") == "ae_gan_hold":
            from benchmarks.locked_shared.hosts.ae_gan_hold import HoldConfig
            cfg = HoldConfig(name="forge")
            overrides.update(encoder_mode="ae", z_dim=2, num_particles=cfg.n_particles, batch_size=cfg.batch)
        return overrides
    return host_recipe_overrides(bound, execution, task_recipe_resources(task))


def task_formulation_context(candidate, task, protocol=None, *, device="cpu", root=None):
    """Resolve a task's actual public configuration without constructing models."""
    from .initialization import task_initializer
    from .nativeprofiles import native_host_initialization
    from .priors import task_prior
    from .taskrecipes import adaptation_receipt, bind_task_candidate
    from .noisy_prior_tier1 import is_noisy_task as recognized_noisy
    if recognized_noisy(task):
        from .noisy_prior_adapters import blockers as original_noisy_blockers
        reasons = original_noisy_blockers(task, candidate, root=root)
        if reasons:
            raise CapabilityError(reasons)
    bound = bind_task_candidate(candidate, task)
    from .atlas_existing_mog import is_candidate as existing_mog_candidate, blockers as existing_mog_blockers
    existing_mog = existing_mog_candidate(bound)
    if existing_mog:
        reasons = existing_mog_blockers(task, bound, root=root)
        if reasons:
            raise CapabilityError(reasons)
    from .atlas_two_pole import supports, resolve_binding, validate_effective_recipe
    from .noisy_prior_tier1 import is_noisy_task, validate as validate_noisy_task
    noisy_task = is_noisy_task(task)
    if noisy_task:
        validate_noisy_task(task, root=root)
    ordinary_two_pole = supports(task, bound) and not noisy_task
    noisy_two_pole = noisy_task and task["execution"].get("host") == "two_pole"
    noisy_binding = resolve_binding(root, bound, task, protocol) if noisy_two_pole else None
    noisy_ae = noisy_task and task["execution"].get("host") == "ae_gan_hold"
    noisy_ae_binding = None
    if noisy_ae:
        import json
        from pathlib import Path
        from . import atlas_noisy_ae
        ae_root = Path(__file__).resolve().parents[2] if root is None else root
        ae_protocol = (json.loads(atlas_noisy_ae._source(ae_root, atlas_noisy_ae.PROTOCOL_PATH))
                       if protocol is None else protocol)
        noisy_ae_binding = atlas_noisy_ae.resolve_binding(ae_root, bound, task, ae_protocol)
    existing_mog_ae_binding = None
    if existing_mog and task["id"] == "ae_gan_hold":
        import json
        from pathlib import Path
        from . import atlas_existing_mog_ae
        ae_root = Path(__file__).resolve().parents[2] if root is None else root
        ae_protocol = (json.loads(atlas_existing_mog_ae._source(ae_root, atlas_existing_mog_ae.PROTOCOL_PATH))
                       if protocol is None else protocol)
        existing_mog_ae_binding = atlas_existing_mog_ae.resolve_binding(ae_root, bound, task, ae_protocol)
    ordinary_binding = (resolve_binding(root, bound, task, protocol)
                        if ordinary_two_pole else None)
    blockers = task_policy_blockers(task, bound)
    if blockers:
        raise CapabilityError(blockers)
    context = FormulationContext(recipe_preset=bound.get("recipe_preset"),
        recipe_overrides=task_recipe_overrides(candidate, task), prior=task_prior(task),
        seed=(protocol or {}).get("seed", 0), device=device,
        requires_capabilities=tuple(bound.get("requires_capabilities", ())) + tuple(task["requires_capabilities"]),
        extensions=bound.get("extensions", {}), initializer=task_initializer(task, candidate),
        host_initialization=native_host_initialization(task, root=root),
        execution_path=task["execution"].get("execution_path", bound.get("execution_path", "public_trainer")),
        candidate_id=bound.get("id"),
        policy_task=task if (ordinary_two_pole or noisy_task or existing_mog or
            task.get("task_cohort") == "tier1_policy_selected_cloud_v1") else None)
    # Typed extensions receive the same ownership and policy checks as ordinary
    # overrides before a worker can reserve this task.
    if ordinary_two_pole:
        validate_effective_recipe(asdict(context.recipe))
        context.ordinary_two_pole_binding = ordinary_binding
        blockers = []
    elif existing_mog:
        if existing_mog_ae_binding is not None:
            if atlas_existing_mog_ae.canonical(asdict(context.recipe)) != atlas_existing_mog_ae.canonical(existing_mog_ae_binding["recipe"]):
                raise CapabilityError(["actual current79 differs from the original existing MoG AE binding"])
            context.existing_mog_ae_binding = existing_mog_ae_binding
        blockers = existing_mog_blockers(task, bound, root=root)
    elif noisy_task:
        if noisy_two_pole:
            validate_effective_recipe(asdict(context.recipe), noisy=True)
            context.noisy_two_pole_binding = noisy_binding
        if noisy_ae:
            if atlas_noisy_ae.canonical(asdict(context.recipe)) != atlas_noisy_ae.canonical(noisy_ae_binding["recipe"]):
                raise CapabilityError(["actual current79 metadata differs from the original Noisy AE owner binding"])
            context.noisy_ae_binding = noisy_ae_binding
        blockers = task_policy_blockers(task, bound)
    else:
        blockers = task_policy_blockers(task, {"recipe_overrides": asdict(context.recipe)})
    if (not ordinary_two_pole and not noisy_task and not existing_mog and task["adapter"] == "transfer_behavior"
            and task["execution"].get("host") != "mode_hold"
            and task.get("task_cohort") != "tier1_policy_selected_cloud_v1"):
        from .behavior_adapters import behavior_preflight
        blockers.extend(behavior_preflight(task, {"recipe_overrides": context.bindings["recipe"]}))
    if blockers:
        raise CapabilityError(blockers)
    from .boundaries import ownership_receipt
    ownership_receipt(candidate, task, asdict(context.recipe), protocol,
                      initializer=context.initializer,
                      extension_recipe_bindings=context.bindings["recipe"])
    context.host_adaptation = adaptation_receipt(candidate, task)
    context.ownership_contract = {"candidate": deepcopy(candidate), "task": deepcopy(task),
                                  "protocol": deepcopy(protocol or {}),
                                  "extension_recipe_bindings": deepcopy(context.bindings["recipe"])}
    return context


def task_policy_blockers(task, candidate):
    """Current Forge tasks certify clean/live component laws, not E22 controls."""
    from .atlas_existing_mog import is_candidate as existing_mog_candidate, blockers as existing_mog_blockers
    if existing_mog_candidate(candidate):
        return existing_mog_blockers(task, candidate)
    from .noisy_prior_tier1 import is_noisy_task
    if is_noisy_task(task):
        from .noisy_prior_adapters import blockers
        return blockers(task, candidate)
    try:
        recipe = resolve_public_recipe(candidate)
    except CapabilityError as error:
        return error.blockers
    from .atlas_two_pole import supports
    if supports(task, candidate):
        return []
    if task.get("task_cohort") == "tier1_policy_selected_cloud_v1":
        from .tier1_policy import blockers
        return blockers(task, recipe)
    if not policy_controls(recipe):
        return []
    host = "public components" if task.get("adapter") == "transfer_behavior" else "clean/live scoring"
    return [f"{task.get('id', '<task>')}: {host} has no declared policy-control evidence and "
            "served-sampling contract; freeze a policy-aware task before reservation"]


@dataclass(frozen=True)
class ExtensionSpec:
    """One typed binding to a public API field, declared centrally.

    ``target`` is ``recipe`` or ``trainer``; no ignored catch-all arguments.
    Stateful/trainable additions must declare their public implementation's
    ownership, RNG and checkpoint behavior in this card.
    """
    name: str
    value_type: str
    target: str
    argument: str
    description: str
    required: bool = False
    default: object = _MISSING
    ownership: str = "public API"
    rng_policy: str = "no additional draws"
    checkpoint: str = "resolved recipe or public trainer state"
    shape: str = "scalar or public argument schema"
    producer: str = "candidate declaration"
    gradient_ownership: str = "public API field implementation"
    optimizer_binding: str = "public Recipe optimizer factories"
    initialization: str = "public API field implementation"
    supported_paths: tuple = ("public_trainer", "public_components")

    def declaration(self):
        result = {name: getattr(self, name) for name in (
            "name", "value_type", "target", "argument", "description", "required",
            "ownership", "rng_policy", "checkpoint", "shape", "producer",
            "gradient_ownership", "optimizer_binding", "initialization", "supported_paths")}
        result["public_definition"] = f"particlegan.{'Recipe' if self.target == 'recipe' else 'GANTrainer'}.{self.argument}"
        result["has_default"] = self.default is not _MISSING
        if self.default is not _MISSING:
            result["default"] = deepcopy(self.default)
        return result


class CapabilityRegistry:
    """Explicit public argument bindings shared by all task families."""
    def __init__(self):
        self.extensions = {}

    def register_extension(self, spec):
        if not isinstance(spec, ExtensionSpec):
            raise TypeError("register_extension requires an ExtensionSpec")
        if (not spec.name or spec.name in BUILTIN_CAPABILITIES or spec.name in self.extensions
                or spec.target not in ("recipe", "trainer")
                or spec.value_type not in ("bool", "int", "float", "str", "list", "dict")
                or any(not isinstance(value, str) or not value.strip()
                       for value in (spec.argument, spec.description, spec.ownership,
                                     spec.rng_policy, spec.checkpoint, spec.shape, spec.producer,
                                     spec.gradient_ownership, spec.optimizer_binding, spec.initialization))
                or not spec.supported_paths
                or set(spec.supported_paths) - {"public_trainer", "public_components"}):
            raise ValueError("invalid or duplicate extension declaration")
        arguments = {field.name for field in fields(Recipe)} if spec.target == "recipe" else \
                    set(inspect.signature(GANTrainer).parameters)
        reserved = {"recipe", "generator", "discriminator", "prior", "seed", "optimizer_options",
                    "penalty_options", "require_latent_damping"}
        if spec.argument not in arguments or (spec.target == "trainer" and (
                spec.argument in reserved or spec.argument.endswith("_generator"))):
            raise ValueError("extension must bind a supported nonreserved public argument")
        if any(old.target == spec.target and old.argument == spec.argument for old in self.extensions.values()):
            raise ValueError("a public argument already has an extension binding")
        self.extensions[spec.name] = spec
        return self

    @staticmethod
    def _validate_value(spec, value):
        expected = {"bool": bool, "int": int, "str": str, "list": list, "dict": dict}.get(spec.value_type)
        valid = (type(value) in (int, float) and math.isfinite(value)) if spec.value_type == "float" else type(value) is expected
        if not valid:
            raise CapabilityError([f"extension {spec.name} requires {spec.value_type}"])

    def resolve(self, supplied):
        if not isinstance(supplied, dict):
            raise CapabilityError(["extensions must be a declared mapping"])
        unknown = set(supplied) - set(self.extensions)
        if unknown:
            raise CapabilityError([f"unsupported extension {name}" for name in sorted(unknown)])
        values, bindings = {}, {"recipe": {}, "trainer": {}}
        for name, spec in self.extensions.items():
            value = supplied.get(name, spec.default)
            if value is _MISSING:
                if spec.required:
                    raise CapabilityError([f"missing required extension {name}"])
                continue
            self._validate_value(spec, value)
            values[name] = deepcopy(value)
            bindings[spec.target][spec.argument] = deepcopy(value)
        return values, bindings


def default_registry():
    """The single catalog for public extension bindings shared by every host.

    Add a supported ExtensionSpec here after implementing its public API field;
    ordinary Recipe fields require no extension declaration.
    """
    return CapabilityRegistry()


def _resolved_prior(value):
    try:
        return resolve_prior(value)
    except ValueError as error:
        raise CapabilityError([str(error)]) from error


class FormulationContext:
    """Resolve one formulation once, then build public trainers across tasks.

    Recipe-only variables stay in ``recipe_overrides``. Prior width is explicit
    and does not run historical nearest-neighbor calibration. ``extensions``
    are validated bindings registered once through ``CapabilityRegistry``.
    """
    def __init__(self, *, recipe_preset=None, recipe_overrides=None, prior=None, seed=0, device="cpu",
                 requires_capabilities=(), registry=None, extensions=None,
                 initializer="deterministic_orthogonal", rng_version=RNG_VERSION,
                 execution_path="public_trainer", host_initialization=None,
                 initializer_requirements=None, policy_task=None, candidate_id=None):
        if execution_path not in ("public_trainer", "public_components"):
            raise CapabilityError(["unsupported public execution path"])
        self.execution_path = execution_path
        self.policy_task = deepcopy(policy_task)
        from .atlas_existing_mog import CANDIDATE_ID, validate as validate_existing_mog, task_resources as existing_mog_resources
        self._existing_mog = candidate_id in {CANDIDATE_ID, 'atlas-existing-mog-radius-observer844-v1'}
        if candidate_id == 'atlas-existing-mog-radius-observer844-v1':
            self.candidate_id = candidate_id
        if candidate_id == 'atlas-existing-mog-radius-observer844-v1' and self.policy_task is not None:
            from .atlas844_radius_owner import validate as validate844
            validate844(self.policy_task)
        if self._existing_mog:
            if recipe_preset != "atlas" or extensions or initializer != "deterministic_orthogonal":
                raise CapabilityError(["Track B permits only the unchanged Atlas preset"])
            if self.policy_task is not None:
                validate_existing_mog(self.policy_task)
                if self.policy_task["id"] not in {"gaussian1d_acquisition", "two_pole", "ae_gan_hold", "ring16_acquisition"}:
                    raise CapabilityError(["Track B owner unsupported before construction"])
                expected = existing_mog_resources(self.policy_task)
                if self.policy_task["id"] == "two_pole":
                    from .atlas_two_pole import TASK_BINDINGS
                    expected = TASK_BINDINGS
                if (recipe_overrides or {}) != expected:
                    raise CapabilityError(["Track B context permits only fixed task resource bindings"])
            elif recipe_overrides:
                raise CapabilityError(["Track B metadata must resolve the unchanged preset before resources"])
        from .atlas_two_pole import is_task
        self._ordinary_two_pole = is_task(self.policy_task)
        from .noisy_prior_tier1 import is_noisy_task, validate as validate_noisy_task
        self._noisy_task = is_noisy_task(self.policy_task)
        if self._noisy_task:
            validate_noisy_task(self.policy_task)
            from .noisy_prior_adapters import blockers
            reasons = blockers(self.policy_task, dict(recipe_preset=recipe_preset,
                recipe_overrides={}, extensions=extensions or {}, initializer=initializer))
            if reasons:
                raise CapabilityError(reasons)
            # Task resource bindings are resolved separately below; constructor
            # validates the unchanged candidate controls, never arbitrary overrides.
            expected = {**task_recipe_resources(self.policy_task)}
            if (recipe_overrides or {}) != expected:
                raise CapabilityError(["Noisy task context permits only its fixed task resources"])
            if (recipe_preset != "atlas" or extensions or initializer != "deterministic_orthogonal"
                    or self.policy_task["prior_substitution_parent"]["task"] not in
                       {"gaussian1d_acquisition", "two_pole", "ae_gan_hold", "ring16_acquisition"}):
                raise CapabilityError(reasons or ["Noisy prior owner unsupported before construction"])
        if self.policy_task is not None and not self._ordinary_two_pole and not self._noisy_task and not self._existing_mog:
            from .tier1_policy import validate
            validate(self.policy_task)
        self.registry = registry if registry is not None else default_registry()
        self.extension_values, self.bindings = self.registry.resolve(extensions or {})
        incompatible = [name for name in self.extension_values
                        if execution_path not in self.registry.extensions[name].supported_paths
                        or (execution_path == "public_components" and self.registry.extensions[name].target == "trainer")]
        if incompatible:
            raise CapabilityError([f"extension {name} is unsupported by {execution_path}" for name in incompatible])
        self.prior_config = _resolved_prior(prior)
        if self._noisy_task or self._existing_mog and self.policy_task is not None:
            from .priors import task_prior
            if self.prior_config != task_prior(self.policy_task):
                raise CapabilityError(["Noisy context prior must equal the fixed parent-width declaration"])
        overrides = dict(recipe_overrides or {})
        duplicates = set(overrides) & set(self.bindings["recipe"])
        if duplicates:
            raise CapabilityError([f"recipe/extension conflict: {sorted(duplicates)}"])
        overrides.update(self.bindings["recipe"])
        kind = {"mog": "mog", "particle_cloud": "particles", "noisy_particle_cloud": "noisy_particles"}[self.prior_config["kind"]]
        prior_fields = {"prior_kind": kind, "sigma_rel": 0., "standardize": self.prior_config["standardize"]}
        for key, value in prior_fields.items():
            if key in overrides and overrides[key] != value:
                raise CapabilityError([f"recipe {key} conflicts with explicit prior declaration"])
        self.recipe_preset = recipe_preset
        self.recipe = resolve_public_recipe({"recipe_preset": recipe_preset,
                                             "recipe_overrides": overrides}, **prior_fields)
        if self._ordinary_two_pole or self._noisy_task and self.policy_task["execution"].get("host") == "two_pole":
            from .atlas_two_pole import validate_effective_recipe
            validate_effective_recipe(asdict(self.recipe), noisy=self._noisy_task)
        if self.recipe.row_policy != "independent":
            raise CapabilityError(["Forge has no RoutedRows host binding; declare a separate routed task"])
        if (self.prior_config["kind"] not in {"particle_cloud", "noisy_particle_cloud"}
                and not (self._existing_mog and self.prior_config["kind"] == "mog"
                         and self.prior_config["standardize"] is False)
                and (self.recipe.row_evidence_gate or self.recipe.particle_birth_death)):
            raise CapabilityError(["independent policy row controls require an explicit particle_cloud cohort; MoG is unsupported"])
        if execution_path == "public_components" and policy_controls(self.recipe) and self.policy_task is None:
            raise CapabilityError(["public_components hosts do not bind the ordered UpdatePolicy lifecycle"])
        if initializer not in ("deterministic_orthogonal", "supplied"):
            raise CapabilityError(["unsupported initializer"])
        self.initializer = initializer
        from .nativeprofiles import resolve_host_initialization
        try:
            self.host_initialization = resolve_host_initialization(host_initialization,
                initializer=initializer, prior=self.prior_config, requirements=initializer_requirements)
        except ValueError as error:
            raise CapabilityError([str(error)]) from error
        self._host_initialized_models = {}
        self._host_prior = None
        self.device = torch.device(device)
        self.streams = NamedStreams(seed, version=rng_version, device=self.device)
        # Declare the fixed purpose names before any task executes. Unused
        # streams are deliberately left untouched (including sigma=0 noise).
        for family, component, purpose in (
                ("init", "prior", "locations"), ("data", "target", "training"),
                ("prior", "latent", "indices"), ("noise", "penalty", "training"),
                ("noise", "generator", "output"), ("noise", "critic", "input"),
                ("noise", "models", "stochastic_layers"), ("noise", "prior", "gaussian"),
                ("eval", "sampler", "samples"), ("eval", "target", "reference")):
            self.streams.generator(family, component=component, purpose=purpose)
        for component in ("generator", "discriminator"):
            self.streams.generator("init", component=component, purpose="construction", device="cpu")
        self.requires_capabilities = tuple(requires_capabilities)
        self.initialization = {}
        self._trainer = None
        self._policy = None
        self._policy_max_steps = None
        self._check_capabilities()

    def capabilities(self):
        scalar = self.recipe.model == "gan" and self.recipe.conditioning == "scalar" and self.recipe.encoder_mode == "none"
        p = self.prior_config
        a2 = (p["learnable"] and not p["standardize"] and self.recipe.latent_damping_max_rate > 0
              and (self.recipe.prior_betas or self.recipe.betas)[0] == 0)
        return {"public_trainer": scalar, "public_components": True, "scalar_gan": scalar,
                "mog_prior": p["kind"] == "mog" or self._noisy_task and p["kind"] == "noisy_particle_cloud",
                "particle_cloud": p["kind"] in {"particle_cloud", "noisy_particle_cloud"},
                "learned_locations": p["learnable"], "uniform_masses": True, "fixed_prior_width": True,
                "a2": a2, "checkpoint": True, "named_rng": True,
                "live_sampling": self._ordinary_two_pole or self._noisy_task or self._existing_mog or not (self.recipe.serve_average or self.recipe.continuous_policy),
                "policy_controls": policy_controls(self.recipe),
                "policy_serving": self.recipe.serve_average > 0,
                "served_sampling": bool(self.policy_task) and not self._ordinary_two_pole and not self._noisy_task and not self._existing_mog and policy_controls(self.recipe),
                **{name: True for name in self.extension_values}}

    def _check_capabilities(self):
        available = self.capabilities()
        required = set(self.requires_capabilities) | {self.execution_path}
        if self.recipe.latent_damping_max_rate > 0:
            required.add("a2")
        blockers = [f"required capability unavailable: {name}" for name in sorted(required) if not available.get(name, False)]
        if blockers:
            raise CapabilityError(blockers)

    def construct(self, factory, *, component):
        """Construct a network without perturbing data/prior/global streams."""
        with self.streams.fork("init", component=component, purpose="construction", device="cpu"):
            return factory()

    def initialize(self, model, *, component):
        policies = (self.host_initialization or {}).get("components", {})
        if component in policies:
            if component in self._host_initialized_models:
                if self._host_initialized_models[component] is not model:
                    raise CapabilityError([f"pinned {component} initialization already belongs to another model"])
                return model
            return self._initialize_host_component(model, component)
        if self.initializer == "supplied":
            return model
        seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                 for name, p in model.named_parameters() if p.requires_grad and p.numel()}
        init.deterministic_orthogonal_(model, parameter_seeds=seeds)
        self.initialization[component] = {"initializer": "deterministic_orthogonal_named_parameters_v1",
                                         "parameter_seeds": seeds}
        return model

    def _host_init_options(self, model, component, *, probe=False):
        """Bind policy descriptors and named streams to the additive public API."""
        policy = self.host_initialization["components"][component]
        options = {"method": policy["method"]}
        parameters = dict(model.named_parameters())
        if policy["method"] == "xavier_uniform_zero_bias_v1":
            options["gain"] = policy["gain"]
            names = [name for name, value in parameters.items() if value.requires_grad and name.split(".")[-1] == "weight"]
        elif policy["method"] == "sample_distributions_v1":
            descriptors = {}
            for name, value in policy["parameters"].items():
                kind = value["kind"]
                descriptors[name] = (init.Uniform(value["low"], value["high"]) if kind == "uniform" else
                    init.Normal(value["mean"], value["std"]) if kind == "normal" else init.KEEP)
            options["distributions"] = descriptors
            resolved = {**init.declarations(model), **descriptors}
            names = [name for name, value in parameters.items() if value.requires_grad and
                     isinstance(resolved.get(name), (init.Uniform, init.Normal))]
        else:
            names = []
        if probe:
            # Validate every actual model before mutating any component. These
            # scratch generators never register/advance a live named stream.
            generators = {name: torch.Generator(device="cpu").manual_seed(
                self.streams.seed_for("init", component=component, purpose=name)) for name in names}
        else:
            generators = {name: self.streams.generator("init", component=component, purpose=name, device="cpu") for name in names}
        options["parameter_generators"] = generators
        return options

    def _initialize_host_component(self, model, component):
        from .state import state_digest
        # Public validation/staging protects the component against partial
        # writes; cloned streams protect the context against registration on a
        # rejected component. G/D are also validated together in build_trainer.
        init.initialize_(deepcopy(model), **self._host_init_options(model, component, probe=True))
        options = self._host_init_options(model, component)
        generators = options["parameter_generators"]
        digest = lambda stream: hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest()
        before = {name: digest(stream) for name, stream in generators.items()}
        init.initialize_(model, **options)
        policy = self.host_initialization["components"][component]
        manifest = {}
        for name, parameter in model.named_parameters():
            row = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                   "requires_grad": parameter.requires_grad, "tensor_sha256": state_digest(parameter),
                   "policy": deepcopy(policy if parameter.requires_grad else {"method": "retain_frozen"})}
            if name in generators:
                row["random_stream"] = {"family": "init", "component": component, "purpose": name,
                    "seed": self.streams.seed_for("init", component=component, purpose=name), "device": "cpu",
                    "initial_state_sha256": before[name], "final_state_sha256": digest(generators[name])}
            manifest[name] = row
        self.initialization[component] = {"initializer": policy["method"], "owner": "task",
            "fallback_initializer": self.initializer, "effective_policy": deepcopy(policy),
            "profile_sha256": self.host_initialization["profile_sha256"],
            "model_class": type(model).__module__ + "." + type(model).__qualname__, "parameters": manifest,
            "buffers": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "tensor_sha256": state_digest(value)}
                        for name, value in model.named_buffers()}}
        if component == "prior":
            self.initialization[component]["constructor_locations"] = "implicit factory init_std creates temporary locations; public policy replaces them once"
        self._host_initialized_models[component] = model
        return model

    def _construct_prior(self, *, dtype, probe=False):
        """Build through the same public factory, optionally using a scratch RNG."""
        p = self.prior_config
        stream = self.streams.generator("init", component="prior", purpose="locations")
        if probe:
            stream = torch.Generator(device=stream.device).set_state(stream.get_state())
        options = {"device": self.device, "dtype": dtype, "learnable": p["learnable"],
                   "generator": stream, "init_std": p.get("init_std", 1.)}
        if p["kind"] in {"mog", "noisy_particle_cloud"}:
            options["sigma"] = p["sigma"]
        return self.recipe.make_prior(**options)

    def build_prior(self, *, dtype=torch.float32):
        if self.host_initialization and self._host_prior is not None:
            if self._host_prior.z.dtype != dtype:
                raise CapabilityError(["pinned prior was already constructed with a different dtype"])
            return self._host_prior
        prior = self._construct_prior(dtype=dtype)
        self.initialize(prior, component="prior")
        if self.host_initialization:
            self._host_prior = prior
        return prior

    def bind_update_policy(self, policy, *, external_max_steps):
        """Bind a caller-owned public lifecycle for the scoped direct-table host."""
        from particlegan import UpdatePolicy
        if self.policy_task is None or not isinstance(policy, UpdatePolicy):
            raise CapabilityError(["a declared task and actual public UpdatePolicy are required"])
        if self._trainer is not None or self._policy is not None:
            raise ValueError("context already owns a public lifecycle")
        if self._ordinary_two_pole or self._noisy_task and self.policy_task["execution"].get("host") == "two_pole":
            from .atlas_two_pole import canonical, validate_effective_recipe
            validate_effective_recipe(asdict(policy.recipe), noisy=self._noisy_task)
            if self._noisy_task:
                from particlegan.noisy_particle_prior import NoisyParticlePrior
                if (type(policy.prior) is not NoisyParticlePrior or policy.prior.z is not policy.table
                        or float(policy.prior.sigma) != 0.):
                    raise CapabilityError(["direct Noisy two-pole requires the same zero-width unsampled table"])
            elif policy.prior is not None:
                raise CapabilityError(["ordinary two-pole has no sampled prior"] )
            if (canonical(asdict(policy.recipe)) != canonical(asdict(self.recipe))
                    or external_max_steps != 80 or policy.table.shape != (12, 1)
                    or policy.row_policy != "independent"):
                raise CapabilityError(["ordinary two-pole direct-table owner differs from its task"] )
            self._policy, self._policy_max_steps = policy, external_max_steps
            return
        if (policy.recipe.to_dict() != self.recipe.to_dict() or
                external_max_steps != self.policy_task["execution"]["steps"] or
                policy.table.shape != (self.recipe.num_particles, self.recipe.z_dim) or
                policy.row_policy != "independent" or policy.prior.z is not policy.table):
            raise CapabilityError(["policy recipe/table/external budget differs from its task"])
        self._policy, self._policy_max_steps = policy, external_max_steps

    def build_trainer(self, generator, discriminator, *, initialize=True, max_steps=None):
        """Build the shared public update path; no task-private training loop."""
        self._check_capabilities()
        if self._existing_mog and self.policy_task is None:
            raise CapabilityError(["Track B model construction requires its exact original task binding"])
        if self.execution_path != "public_trainer":
            raise CapabilityError(["public_components context does not own a scalar GANTrainer"])
        if self._trainer is not None:
            raise ValueError("a context owns one trainer; create a new context for another task/attempt")
        if self.host_initialization:
            if not initialize:
                raise CapabilityError(["initialize=False cannot bypass task-pinned component initialization"])
            if any(next(model.parameters()).device != self.device for model in (generator, discriminator)):
                raise CapabilityError(["context and network devices differ"])
            for component, model in (("generator", generator), ("discriminator", discriminator)):
                if component in self.host_initialization["components"]:
                    init.initialize_(deepcopy(model), **self._host_init_options(model, component, probe=True))
                else:
                    seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                             for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.numel()}
                    init.deterministic_orthogonal_(deepcopy(model), parameter_seeds=seeds)
            dtype = next(generator.parameters()).dtype
            if self._host_prior is not None:
                if self._host_prior.z.dtype != dtype:
                    raise CapabilityError(["pinned prior was already constructed with a different dtype"])
            else:
                # Prior shape/dtype checks can reject a syntactically valid
                # policy. Perform them before either caller network changes.
                # The temporary factory and initializer use scratch generators;
                # successful construction still consumes the original one draw.
                probe_prior = self._construct_prior(dtype=dtype, probe=True)
                if "prior" in self.host_initialization["components"]:
                    init.initialize_(probe_prior, **self._host_init_options(probe_prior, "prior", probe=True))
                else:
                    seeds = {name: self.streams.seed_for("init", component="prior", purpose=name)
                             for name, parameter in probe_prior.named_parameters() if parameter.requires_grad and parameter.numel()}
                    init.deterministic_orthogonal_(probe_prior, parameter_seeds=seeds)
        if initialize:
            self.initialize(generator, component="generator")
            self.initialize(discriminator, component="discriminator")
        first = next(generator.parameters())
        if first.device != self.device:
            raise CapabilityError(["context and network devices differ"])
        prior = self.build_prior(dtype=first.dtype)
        options = {
            **{name: self.streams.generator(family, component=component, purpose=purpose)
               for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()
               if name != "prior_noise_generator" or type(prior).__name__ in {"MoGParticlePrior", "NoisyParticlePrior"}},
            "require_latent_damping": self.recipe.latent_damping_max_rate > 0,
            **self.bindings["trainer"],
        }
        if self._noisy_task:
            limit = self.policy_task["execution"]["steps"]
            if max_steps is not None and max_steps != limit:
                raise CapabilityError(["Noisy task external horizon differs"])
            max_steps = limit
        if max_steps is not None:
            if "max_steps" in options:
                raise CapabilityError(["execution budget conflicts with trainer extension"])
            options["max_steps"] = max_steps
        streams = [options[name] for name in TRAINER_STREAM_BINDINGS if name in options]
        if len({id(stream) for stream in streams}) != len(streams):
            raise CapabilityError(["Forge requires distinct named trainer streams"])
        self._trainer = GANTrainer(self.recipe, generator, discriminator, prior=prior,
                                   seed=self.streams.seed, **options)
        return self._trainer

    def receipt(self):
        ownership = {}
        if getattr(self, "ownership_contract", None) is not None:
            from .boundaries import ownership_receipt
            ownership["field_ownership"] = ownership_receipt(**self.ownership_contract,
                resolved_recipe=asdict(self.recipe), initializer=self.initializer)
        policy = None
        if (self._trainer is not None or self._policy is not None) and policy_controls(self.recipe):
            owner = self._trainer if self._trainer is not None else self._policy
            birth = owner.birth_death
            policy = {"owner": "particlegan.UpdatePolicy", "completed_steps": owner.completed_steps,
                      "external_max_steps": self._trainer.max_steps if self._trainer is not None else self._policy_max_steps,
                      "continuous_policy": self.recipe.continuous_policy,
                      "lr_control": self.recipe.lr_control, "row_policy": self.recipe.row_policy,
                      "serving": "state_selected" if self.recipe.serve_average else "fast",
                      "quality_qualification": False,
                      "private_rng": [] if birth is None else [{
                          "owner": "UpdatePolicy.birth_death", "seed": self.streams.seed + 6,
                          "derivation": "public policy seed + 6; independent of Forge named streams",
                          "state_sha256": hashlib.sha256(birth.stream.get_state().cpu().numpy().tobytes()).hexdigest(),
                          "checkpoint_path": "trainer.birth_death.stream"}]}
        return {"api_version": API_VERSION, "execution_path": self.execution_path,
                **ownership,
                 **({"noisy_prior_contract": {"cohort": self.policy_task["task_cohort"],
                      "parent": deepcopy(self.policy_task["prior_substitution_parent"]),
                      "measurement_owner": "original_live", "causal_recovery_established": False}}
                    if self._noisy_task else {}),
                **({"ordinary_component_contract": deepcopy(self.ordinary_two_pole_binding["source_contract"])}
                    if getattr(self, "ordinary_two_pole_binding", None) is not None else {}),
                **({"host_adaptation": self.host_adaptation} if getattr(self, "host_adaptation", None) else {}),
                "recipe_preset": self.recipe_preset,
                "recipe": self.recipe.to_dict(), "prior": deepcopy(self.prior_config),
                "capabilities": self.capabilities(), "requires_capabilities": list(self.requires_capabilities),
                "extensions": deepcopy(self.extension_values),
                "api_changes": [self.registry.extensions[name].declaration() for name in sorted(self.extension_values)],
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "rng": self.streams.manifest(), "policy_lifecycle": policy,
                "prior_mechanisms": None if self._trainer is None else deepcopy(self._trainer.prior_mechanisms)}

    def state_dict(self):
        if self._trainer is None:
            raise ValueError("build a trainer before checkpointing")
        return {"schema": 1, "api_version": API_VERSION, "recipe": self.recipe.to_dict(),
                "prior": deepcopy(self.prior_config), "extensions": deepcopy(self.extension_values),
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "streams": self.streams.state_dict(), "trainer": self._trainer.state_dict()}

    def load_state_dict(self, state):
        if self._trainer is None:
            raise ValueError("build a trainer before restoring")
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid context checkpoint schema")
        for key in ("schema", "api_version", "recipe", "prior", "extensions", "initializer", "initialization"):
            if state[key] != expected[key]:
                raise ValueError(f"context checkpoint {key} mismatch")
        self.streams.validate_state_dict(state["streams"])
        # A trainer and named registry refer to the same stream objects. Reject
        # conflicting duplicated states before either loader mutates anything.
        for name, value in state["trainer"].get("streams", {}).items():
            stream = getattr(self._trainer, name, None)
            key = next((key for key, registered in self.streams._streams.items() if registered is stream), None)
            if key is None or key not in state["streams"]["states"] or not torch.equal(value, state["streams"]["states"][key]):
                raise ValueError("trainer/named RNG checkpoint mismatch")
        self._trainer.load_state_dict(state["trainer"])
        self.streams.load_state_dict(state["streams"])
