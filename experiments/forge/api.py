"""Forge's thin construction contract around the public ParticleGAN API.

There is no second optimizer or training loop here. New experiment variables
bind explicitly to a documented public Recipe or GANTrainer argument.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import math
from pathlib import Path

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
    "routed_rows", "mog_encoder", "ae_encoder",
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
    execution, adapter = task["execution"], task["adapter"]
    from .policy_cohorts import is_policy_task, validate_policy_task
    if is_policy_task(task) and "resources" in execution:
        validate_policy_task(task)
        return deepcopy(execution["resources"])
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
    from .policy_cohorts import is_policy_task
    if is_policy_task(task):
        # A prospective policy host explicitly owns these frozen exceptions.
        # This does not enlarge the original host_adaptation allowlist.
        from .policy_cohorts import policy_recipe_overrides
        if candidate.get("task_cohort") != task.get("task_cohort"):
            raise CapabilityError(["candidate must explicitly opt in to the task's named policy cohort"])
        bound["recipe_overrides"] = {
            **bound.get("recipe_overrides", {}), **policy_recipe_overrides(task)}
        return host_recipe_overrides(bound, execution, task_recipe_resources(task))
    if task["adapter"] == "transfer_behavior" and execution.get("host") != "mode_hold":
        from .behavior_adapters import behavior_preflight
        blockers = behavior_preflight(task, bound)
        if blockers:
            raise CapabilityError(blockers)
        overrides = {**bound.get("recipe_overrides", {}), "total_steps": execution["steps"]}
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
    from .policy_cohorts import is_policy_task, policy_task_declaration, validate_policy_task
    if is_policy_task(task):
        try:
            task = policy_task_declaration(task)
            validate_policy_task(task, root=root)
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            raise CapabilityError([str(error)]) from error
    bound = bind_task_candidate(candidate, task)
    blockers = task_policy_blockers(task, bound)
    if blockers:
        raise CapabilityError(blockers)
    context = FormulationContext(recipe_preset=bound.get("recipe_preset"),
        recipe_overrides=task_recipe_overrides(candidate, task), prior=task_prior(task),
        seed=(protocol or {}).get("seed", 0), device=device,
        requires_capabilities=tuple(bound.get("requires_capabilities", ())) + tuple(task["requires_capabilities"]),
        extensions=bound.get("extensions", {}), initializer=task_initializer(task, candidate),
        host_initialization=native_host_initialization(task, root=root),
        policy_contract=task["execution"].get("policy_contract"),
        policy_task=task if is_policy_task(task) else None,
        policy_root=root,
        execution_path=task["execution"].get("execution_path", bound.get("execution_path", "public_trainer")))
    # Typed extensions receive the same ownership and policy checks as ordinary
    # overrides before a worker can reserve this task.
    blockers = task_policy_blockers(task, {**bound, "recipe_preset": None,
                                         "recipe_overrides": asdict(context.recipe)})
    if (task["adapter"] == "transfer_behavior" and task["execution"].get("host") != "mode_hold"
            and task["execution"].get("policy_contract") is None):
        from .behavior_adapters import behavior_preflight
        blockers.extend(behavior_preflight(task, {"recipe_overrides": context.bindings["recipe"]}))
    if blockers:
        raise CapabilityError(blockers)
    from .boundaries import ownership_receipt
    ownership_receipt(candidate, task, asdict(context.recipe), protocol,
                      initializer=context.initializer,
                      extension_recipe_bindings=context.bindings["recipe"])
    context.host_adaptation = adaptation_receipt(candidate, task)
    context._policy_expected_max_steps = task["execution"]["steps"]
    context.ownership_contract = {"candidate": deepcopy(candidate), "task": deepcopy(task),
                                  "protocol": deepcopy(protocol or {}),
                                  "extension_recipe_bindings": deepcopy(context.bindings["recipe"])}
    return context


def task_policy_blockers(task, candidate):
    """Current Forge tasks certify clean/live component laws, not E22 controls."""
    try:
        from .policy_cohorts import is_policy_task, validate_policy_task, policy_contract_blockers
        execution = task.get("execution", {})
        if is_policy_task(task):
            validate_policy_task(task)
            prior = _resolved_prior(execution.get("prior"))
            recipe = resolve_public_recipe({
                **candidate, "recipe_overrides": task_recipe_overrides(candidate, task)},
                prior_kind="mog" if prior["kind"] == "mog" else "particles",
                sigma_rel=0., standardize=prior["standardize"])
        else:
            recipe = resolve_public_recipe(candidate)
    except (CapabilityError, AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        return error.blockers if isinstance(error, CapabilityError) else [str(error)]
    if is_policy_task(task):
        return policy_contract_blockers(task, recipe)
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
                 initializer_requirements=None, policy_contract=None, policy_task=None, policy_root=None):
        if execution_path not in ("public_trainer", "public_components"):
            raise CapabilityError(["unsupported public execution path"])
        self.execution_path = execution_path
        self.policy_contract = deepcopy(policy_contract)
        self.policy_task = deepcopy(policy_task)
        self._policy_root = (Path(__file__).resolve().parents[2] if policy_root is None
                             else Path(policy_root).resolve())
        if policy_contract is not None or policy_task is not None:
            from .policy_cohorts import policy_task_declaration, validate_policy_task
            try:
                self.policy_task = policy_task_declaration(self.policy_task)
                declared = validate_policy_task(self.policy_task)
            except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
                raise CapabilityError([str(error)]) from error
            if declared != policy_contract or declared["execution_path"] != execution_path:
                raise CapabilityError(["context policy contract differs from its validated named task"])
        self.registry = registry if registry is not None else default_registry()
        self.extension_values, self.bindings = self.registry.resolve(extensions or {})
        incompatible = [name for name in self.extension_values
                        if execution_path not in self.registry.extensions[name].supported_paths
                        or (execution_path == "public_components" and self.registry.extensions[name].target == "trainer")]
        if incompatible:
            raise CapabilityError([f"extension {name} is unsupported by {execution_path}" for name in incompatible])
        self.prior_config = _resolved_prior(prior)
        overrides = dict(recipe_overrides or {})
        duplicates = set(overrides) & set(self.bindings["recipe"])
        if duplicates:
            raise CapabilityError([f"recipe/extension conflict: {sorted(duplicates)}"])
        overrides.update(self.bindings["recipe"])
        kind = "mog" if self.prior_config["kind"] == "mog" else "particles"
        prior_fields = {"prior_kind": kind, "sigma_rel": 0., "standardize": self.prior_config["standardize"]}
        for key, value in prior_fields.items():
            if key in overrides and overrides[key] != value:
                raise CapabilityError([f"recipe {key} conflicts with explicit prior declaration"])
        self.recipe_preset = recipe_preset
        self.recipe = resolve_public_recipe({"recipe_preset": recipe_preset,
                                             "recipe_overrides": overrides}, **prior_fields)
        if self.recipe.row_policy != "independent" and self.policy_task is None:
            raise CapabilityError(["Forge has no RoutedRows host binding; declare a separate routed task"])
        if self.prior_config["kind"] != "particle_cloud" and (
                self.recipe.row_evidence_gate or self.recipe.particle_birth_death) and self.recipe.row_policy == "independent":
            raise CapabilityError(["independent policy row controls require an explicit particle_cloud cohort; MoG is unsupported"])
        if self.policy_task is not None:
            from .policy_cohorts import policy_contract_blockers
            blockers = policy_contract_blockers(self.policy_task, self.recipe)
            if blockers:
                raise CapabilityError(blockers)
            if self.prior_config != _resolved_prior(self.policy_task["execution"]["prior"]):
                raise CapabilityError(["context prior differs from the named task's actual law"])
        if execution_path == "public_components" and policy_controls(self.recipe) and (
                not isinstance(self.policy_contract, dict)
                or self.policy_contract.get("owner") != "particlegan.UpdatePolicy"
                or self.policy_contract.get("lifecycle") != "ordered_public_update"
                or self.policy_contract.get("execution_path") != "public_components"):
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
        self._policy_expected_max_steps = (None if self.policy_task is None
                                           else self.policy_task["execution"]["steps"])
        self._check_capabilities()

    def capabilities(self):
        routed = self.recipe.row_policy == "routed_paired" and self.policy_task is not None
        joint = (self.policy_task is not None
                 and self.policy_task["task_cohort"] == "word_joint_policy_min11_v1")
        named_ae = bool(routed and self.policy_task["task_cohort"] == "ae_routed_policy_v1"
                        and self.recipe.encoder_mode == "ae" and self.prior_config["kind"] == "mog")
        scalar = (self.recipe.model == "gan" and self.recipe.conditioning == "scalar"
                  and (self.recipe.encoder_mode == "none" or named_ae))
        p = self.prior_config
        a2 = (p["learnable"] and not p["standardize"] and self.recipe.latent_damping_max_rate > 0
              and (self.recipe.prior_betas or self.recipe.betas)[0] == 0)
        return {"public_trainer": scalar and self.recipe.encoder_mode == "none" and not routed and not joint,
                "public_components": True, "scalar_gan": scalar,
                "mog_prior": p["kind"] == "mog", "particle_cloud": p["kind"] == "particle_cloud",
                "learned_locations": p["learnable"], "uniform_masses": not routed, "fixed_prior_width": True,
                "a2": a2, "checkpoint": True, "named_rng": True,
                "live_sampling": not (self.recipe.serve_average or self.recipe.continuous_policy),
                "policy_controls": policy_controls(self.recipe),
                "policy_serving": self.recipe.serve_average > 0,
                "served_sampling": bool(policy_controls(self.recipe) and self.recipe.serve_average > 0),
                "routed_rows": routed,
                "mog_encoder": named_ae, "ae_encoder": named_ae,
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
        if p["kind"] == "mog":
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

    def build_trainer(self, generator, discriminator, *, initialize=True, max_steps=None):
        """Build the shared public update path; no task-private training loop."""
        self._check_capabilities()
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
               if name != "prior_noise_generator" or type(prior).__name__ == "MoGParticlePrior"},
            "require_latent_damping": self.recipe.latent_damping_max_rate > 0,
            **self.bindings["trainer"],
        }
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

    @property
    def policy(self):
        """The actual public lifecycle owned by this one task attempt."""
        return self._trainer.policy if self._trainer is not None else self._policy

    def bind_update_policy(self, policy, *, external_max_steps):
        """Bind caller-owned components to a real, declared public lifecycle."""
        from particlegan import UpdatePolicy
        if self.execution_path != "public_components" or self.policy_contract is None:
            raise CapabilityError(["caller-owned policy requires a declared public-components task contract"])
        if not isinstance(policy, UpdatePolicy):
            raise TypeError("policy must be a particlegan.UpdatePolicy")
        if self._trainer is not None or self._policy is not None:
            raise ValueError("a context owns one public update lifecycle")
        if policy.recipe.to_dict() != self.recipe.to_dict():
            raise CapabilityError(["bound policy Recipe differs from the resolved task Recipe"])
        if type(external_max_steps) is not int or external_max_steps < 1:
            raise ValueError("external_max_steps must be a positive integer")
        if (self._policy_expected_max_steps is not None
                and external_max_steps != self._policy_expected_max_steps):
            raise CapabilityError(["bound policy execution limit differs from its frozen task budget"])
        if policy.device != self.device or policy.row_semantics != self.policy_contract["row_semantics"]:
            raise CapabilityError(["bound policy device or row semantics differ from its task contract"])
        precision = self.policy_contract["precision"]
        if ((precision.endswith("_no_autocast") and torch.is_autocast_enabled(policy.device.type))
                or (precision == "preserve_original_fp32_no_autocast" and policy.table.dtype != torch.float32)):
            raise CapabilityError(["bound policy must preserve the named task's FP32/no-autocast execution"])
        if self.policy_task is None:
            raise CapabilityError(["bound policy requires its validated named task identity"])
        from .policy_cohorts import policy_contract_blockers
        blockers = policy_contract_blockers(self.policy_task, policy.recipe)
        if blockers:
            raise CapabilityError(blockers)
        self._validate_policy_owners(policy)
        self._policy = policy
        self._policy_max_steps = external_max_steps
        return policy

    bind_policy = bind_update_policy

    def _validate_policy_owners(self, policy):
        """Bind the real public row/serving owner, rather than a declared label."""
        from particlegan.particle_prior import MoGParticlePrior, ParticlePrior
        table_owner = self.policy_contract.get("table_owner", "prior.z")
        owner, _, key = table_owner.partition(".")
        if policy._table_location != (owner, key):
            raise CapabilityError(["bound policy table owner differs from the named task"])
        if policy.table.requires_grad is not self.prior_config["learnable"]:
            raise CapabilityError(["bound policy table learning law differs from the named task"])
        if policy.table.shape != (self.recipe.num_particles, self.recipe.z_dim):
            raise CapabilityError(["bound policy table shape differs from the frozen task resources"])
        if owner == "prior":
            if self.prior_config["kind"] == "mog":
                if not isinstance(policy.prior, MoGParticlePrior):
                    raise CapabilityError(["named AE requires its real fixed-width MoG owner"])
                policy._validate_routed_mog()
                if float(policy.prior.sigma) != float(policy.prior.sigma.new_tensor(self.prior_config["sigma"])):
                    raise CapabilityError(["bound MoG width differs from the named task"])
            elif not isinstance(policy.prior, ParticlePrior) or isinstance(policy.prior, MoGParticlePrior):
                raise CapabilityError(["bound policy requires its actual particle-cloud prior owner"])
            if policy.prior.z is not policy.table:
                raise CapabilityError(["bound policy prior and table are not the same owner"])
        elif policy.prior is not None:
            raise CapabilityError(["named parameter-bank task cannot acquire a substituted sampled prior"])
        if policy.row_policy != self.recipe.row_policy:
            raise CapabilityError(["bound policy row law differs from the named task"])
        if policy.row_policy != "routed_paired":
            if policy.routed_control is not None:
                raise CapabilityError(["independent task cannot bind a routed row owner"])
            if self.policy_task["task_cohort"] == "word_joint_policy_min11_v1":
                self._validate_word_joint_owners(policy)
            return
        control = policy.routed_control
        rows = getattr(control, "spec", None)
        routing = self.policy_contract["routing"]
        if (rows is None or not callable(rows.model_forward) or tuple(routing["sites"]) != rows.sites
                or tuple(routing["row_buffers"]) != rows.row_buffers
                or "router." + rows.log_mass_key != routing["log_mass"]):
            raise CapabilityError(["bound policy needs the exact complete named routing sites and mass owner"])
        fast_models = policy._training_modules()
        selected_models = {**fast_models, **policy._average_modules()}
        if (control.table is not policy.table or control.averaged_table is not policy.averaged_table
                or control.spec is not policy._routed_rows
                or set(control.models) != set(fast_models)
                or any(control.models[name] is not model for name, model in fast_models.items())
                or set(control.averaged_models) != set(selected_models)
                or any(control.averaged_models[name] is not model for name, model in selected_models.items())
                or len(control.optimizers) != len(policy.optimizers)
                or any(a is not b for a, b in zip(control.optimizers, policy.optimizers))
                or control.table_optimizer is not policy.table_optimizer
                or control.controller is not policy.controller):
            raise CapabilityError(["routed control must bind this policy's actual models, table and optimizers"])
        callbacks = {
            "conditional_policy_selected_cloud_v1": ("conditional_policy_adapters",
                "complete_arc_forward" if self.policy_task["execution"]["host"] in {"trajectory", "residual_student"}
                else "complete_basis_forward"),
            "routed_policy_selected_cloud_v1": ("routed_policy_adapters", "complete_slot_forward"),
            "multibank_policy_v1": ("multibank_policy_adapters", "complete_polar_forward"),
            "ae_routed_policy_v1": ("ae_routed_policy_adapters", "complete_ae_forward"),
        }
        module_name, function_name = callbacks[self.policy_task["task_cohort"]]
        self._validate_complete_callback(rows.model_forward, module_name, function_name)
        if self.recipe.particle_birth_death and policy.birth_death is not policy.routed_control:
            raise CapabilityError(["requested routed birth/death owner is missing"])
        if self.recipe.row_evidence_gate and policy.row_evidence is not policy.routed_control.evidence:
            raise CapabilityError(["requested routed evidence owner is missing"])
        policy.routed_control.validate_optimizer_transport()

    def _validate_complete_callback(self, fn, module_name, function_name):
        if not inspect.isfunction(fn):
            raise CapabilityError(["bound policy requires its frozen complete generation callback"])
        relative = "experiments/forge/" + module_name + ".py"
        source = inspect.getsourcefile(fn)
        if (getattr(fn, "__module__", None) != "experiments.forge." + module_name
                or getattr(fn, "__name__", None) != function_name or source is None
                or Path(source).resolve() != (self._policy_root / relative).resolve()
                or hashlib.sha256(Path(source).read_bytes()).hexdigest() != self.policy_contract["sources"].get(relative)):
            raise CapabilityError(["bound complete routing callback differs from its frozen named source"])

    def _validate_word_joint_owners(self, policy):
        """Independent whole-joint atoms retain their auxiliary free encoder."""
        from particlegan.birth_death import ParticleBirthDeath, ScalarHeadFeatures
        from particlegan.row_evidence import RowEvidence
        self._validate_complete_callback(policy.generation, "word_joint_policy_adapters", "joint_generation")
        relative = "benchmarks/toy_audit/api_images.py"
        for model, name in ((policy.G, "WordGenerator"), (policy.D, "WordJointCritic"),
                            (policy.encoder, "WordEncoder"), (policy.ema_G, "WordGenerator"),
                            (policy.ema_encoder, "WordEncoder")):
            model_type = type(model)
            source = inspect.getsourcefile(model_type) if model is not None else None
            if (model_type.__name__ != name or model_type.__module__ != "benchmarks.toy_audit.api_images"
                    or source is None or Path(source).resolve() != (self._policy_root / relative).resolve()
                    or hashlib.sha256(Path(source).read_bytes()).hexdigest() != self.policy_contract["sources"].get(relative)):
                raise CapabilityError(["word joint policy requires its original G, D and free encoder owners"])
        if (any(not p.requires_grad for model in (policy.G, policy.D, policy.encoder) for p in model.parameters())
                or any(p.requires_grad for model in (policy.ema_G, policy.ema_encoder) for p in model.parameters())):
            raise CapabilityError(["word joint keeps trainable G/D/free E and frozen selected averages"])
        parameters = {id(p): name for name, module in policy._training_modules().items()
                      for p in module.parameters()}
        parameters[id(policy.table)] = "table"
        if policy.log_output_sigma is not None:
            parameters[id(policy.log_output_sigma)] = "noise"
        seen = set()
        if len(policy.roles) != len(policy.optimizers):
            raise CapabilityError(["word joint optimizer roles differ from its public owners"])
        for optimizer, roles in zip(policy.optimizers, policy.roles):
            if len(roles) != len(optimizer.param_groups):
                raise CapabilityError(["word joint optimizer roles differ from its public owners"])
            for group, role in zip(optimizer.param_groups, roles):
                for parameter in group["params"]:
                    if parameters.get(id(parameter)) != role or id(parameter) in seen:
                        raise CapabilityError(["word joint G/E/table/noise parameters need exact public optimizer ownership"])
                    seen.add(id(parameter))
        if seen != set(parameters):
            raise CapabilityError(["word joint G/E/table/noise parameters need exact public optimizer ownership"])
        if self.recipe.particle_birth_death:
            birth = policy.birth_death
            if (not isinstance(birth, ParticleBirthDeath) or birth.rows.table is not policy.table
                    or birth.rows.averaged_table is not policy.averaged_table
                    or birth.rows.optimizer is not policy.table_optimizer
                    or birth.rows.controller is not policy.controller
                    or policy.encoder not in birth.rows.evaluation_modules
                    or not isinstance(birth.rows.critic_features, ScalarHeadFeatures)
                    or birth.rows.critic_features.critic is not policy.D):
                raise CapabilityError(["word joint birth/death must own the actual complete independent atoms"])
        if self.recipe.row_evidence_gate and (not isinstance(policy.row_evidence, RowEvidence)
                or (policy.row_evidence.n, policy.row_evidence.d) != tuple(policy.table.shape)):
            raise CapabilityError(["word joint needs its actual independent row evidence owner"])

    def receipt(self):
        ownership = {}
        if getattr(self, "ownership_contract", None) is not None:
            from .boundaries import ownership_receipt
            ownership["field_ownership"] = ownership_receipt(**self.ownership_contract,
                resolved_recipe=asdict(self.recipe), initializer=self.initializer)
        policy = None
        if self.policy is not None and policy_controls(self.recipe):
            actual_policy = self.policy
            birth = actual_policy.birth_death
            policy = {"owner": "particlegan.UpdatePolicy", "completed_steps": actual_policy.completed_steps,
                      "external_max_steps": self._trainer.max_steps if self._trainer is not None else self._policy_max_steps,
                      "continuous_policy": self.recipe.continuous_policy,
                      "lr_control": self.recipe.lr_control, "row_policy": self.recipe.row_policy,
                      "serving": "state_selected" if self.recipe.serve_average else "fast",
                      "quality_qualification": False,
                      "private_rng": [] if birth is None else [{
                          "owner": "UpdatePolicy.birth_death", "seed": self.streams.seed + 6,
                          "derivation": "public policy seed + 6; independent of Forge named streams",
                          "state_sha256": hashlib.sha256(birth.stream.get_state().cpu().numpy().tobytes()).hexdigest(),
                          "checkpoint_path": "trainer.birth_death.stream" if self._trainer is not None else "policy.birth_death.stream"}]}
            if self.policy_contract is not None:
                from .policy_adapters import controls_receipt
                policy["contract"] = deepcopy(self.policy_contract)
                controls = controls_receipt(actual_policy, actual_policy.completed_steps)
                controls.update(cohort=self.policy_contract["cohort"], row_policy=actual_policy.row_policy)
                if "family" in self.policy_contract:
                    controls.update(family=self.policy_contract["family"], independent_atlas_qualification=False)
                if "table_optimizer_semantics" in self.policy_contract:
                    controls["table_optimizer_semantics"] = deepcopy(self.policy_contract["table_optimizer_semantics"])
                policy["controls"] = controls
        return {"api_version": API_VERSION, "execution_path": self.execution_path,
                **ownership,
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
        owner = self._trainer if self._trainer is not None else self._policy
        if owner is None:
            raise ValueError("build a trainer or bind a public policy before checkpointing")
        kind = "trainer" if self._trainer is not None else "policy"
        return {"schema": 1, "api_version": API_VERSION, "recipe": self.recipe.to_dict(),
                "prior": deepcopy(self.prior_config), "extensions": deepcopy(self.extension_values),
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "streams": self.streams.state_dict(), kind: owner.state_dict(),
                **({} if kind == "trainer" else {"policy_contract": deepcopy(self.policy_contract),
                                                "external_max_steps": self._policy_max_steps})}

    def load_state_dict(self, state):
        owner = self._trainer if self._trainer is not None else self._policy
        if owner is None:
            raise ValueError("build a trainer or bind a public policy before restoring")
        kind = "trainer" if self._trainer is not None else "policy"
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError("invalid context checkpoint schema")
        for key in ("schema", "api_version", "recipe", "prior", "extensions", "initializer", "initialization"):
            if state[key] != expected[key]:
                raise ValueError(f"context checkpoint {key} mismatch")
        if not isinstance(state[kind], dict):
            raise ValueError(f"context checkpoint {kind} must be a mapping")
        if kind == "policy":
            if (state["policy_contract"] != expected["policy_contract"]
                    or state["external_max_steps"] != expected["external_max_steps"]
                    or type(state["policy"].get("completed_steps")) is not int
                    or not 0 <= state["policy"]["completed_steps"] <= self._policy_max_steps):
                raise ValueError("context checkpoint policy contract/budget mismatch")
        self.streams.validate_state_dict(state["streams"])
        # A trainer and named registry refer to the same stream objects. Reject
        # conflicting duplicated states before either loader mutates anything.
        for name, value in state[kind].get("streams", {}).items():
            stream = getattr(owner, name, None)
            key = next((key for key, registered in self.streams._streams.items() if registered is stream), None)
            if key is None or key not in state["streams"]["states"] or not torch.equal(value, state["streams"]["states"][key]):
                raise ValueError(f"{kind}/named RNG checkpoint mismatch")
        owner.load_state_dict(state[kind])
        self.streams.load_state_dict(state["streams"])
