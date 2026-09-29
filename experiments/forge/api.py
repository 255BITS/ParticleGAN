"""Forge's thin construction contract around the public ParticleGAN API.

There is no second optimizer or training loop here. New experiment variables
bind explicitly to a documented public Recipe or GANTrainer argument.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import inspect
import math

import torch

from particlegan import GANTrainer, Recipe, init, prior_capabilities, prior_mechanisms
from .rng import NamedStreams, RNG_VERSION


API_VERSION = "forge-api-v1"
DEFAULT_PRIOR = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
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
)
_MISSING = object()


class CapabilityError(ValueError):
    """Pretraining incompatibility; callers record BLOCKED, not scientific FAIL."""
    status = "BLOCKED"

    def __init__(self, blockers):
        self.blockers = list(blockers)
        super().__init__("; ".join(self.blockers))


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
    given = {} if value is None else dict(value)
    unknown = set(given) - {"kind", "sigma", "standardize", "learnable", "exception_reason", "init_std"}
    if unknown:
        raise CapabilityError([f"unsupported prior fields: {sorted(unknown)}"])
    prior = {**DEFAULT_PRIOR, **given}
    if prior["kind"] not in ("mog", "particle_cloud"):
        raise CapabilityError(["prior kind must be mog or particle_cloud"])
    if (type(prior["sigma"]) not in (int, float) or not math.isfinite(prior["sigma"])
            or prior["sigma"] < 0 or type(prior["standardize"]) is not bool
            or type(prior["learnable"]) is not bool):
        raise CapabilityError(["prior sigma/standardize/learnable fields are invalid"])
    if prior["kind"] == "particle_cloud":
        if (not {"sigma", "standardize", "exception_reason"}.issubset(given)
                or prior["sigma"] != 0 or prior["standardize"]
                or not isinstance(prior["exception_reason"], str) or not prior["exception_reason"].strip()):
            raise CapabilityError(["particle_cloud requires explicit sigma=0, standardize=False and exception_reason"])
    elif prior["sigma"] <= 0:
        raise CapabilityError(["MoG requires nonzero sigma; declare an explicit particle_cloud exception for zero noise"])
    if "init_std" in prior and (type(prior["init_std"]) not in (int, float)
                                or not math.isfinite(prior["init_std"]) or prior["init_std"] < 0):
        raise CapabilityError(["prior init_std must be finite and nonnegative"])
    return prior


class FormulationContext:
    """Resolve one formulation once, then build public trainers across tasks.

    Recipe-only variables stay in ``recipe_overrides``. Prior width is explicit
    and does not run historical nearest-neighbor calibration. ``extensions``
    are validated bindings registered once through ``CapabilityRegistry``.
    """
    def __init__(self, *, recipe_overrides=None, prior=None, seed=0, device="cpu",
                 requires_capabilities=(), registry=None, extensions=None,
                 initializer="deterministic_orthogonal", rng_version=RNG_VERSION,
                 execution_path="public_trainer"):
        if execution_path not in ("public_trainer", "public_components"):
            raise CapabilityError(["unsupported public execution path"])
        self.execution_path = execution_path
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
        try:
            self.recipe = Recipe(**{**overrides, **prior_fields})
        except (TypeError, ValueError) as error:
            raise CapabilityError([f"unsupported recipe: {error}"]) from error
        if initializer not in ("deterministic_orthogonal", "supplied"):
            raise CapabilityError(["unsupported initializer"])
        self.initializer = initializer
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
        self._check_capabilities()

    def capabilities(self):
        scalar = self.recipe.model == "gan" and self.recipe.conditioning == "scalar" and self.recipe.encoder_mode == "none"
        p = self.prior_config
        a2 = (p["learnable"] and not p["standardize"] and self.recipe.latent_damping_max_rate > 0
              and (self.recipe.prior_betas or self.recipe.betas)[0] == 0)
        return {"public_trainer": scalar, "public_components": True, "scalar_gan": scalar,
                "mog_prior": p["kind"] == "mog", "particle_cloud": p["kind"] == "particle_cloud",
                "learned_locations": p["learnable"], "uniform_masses": True, "fixed_prior_width": True,
                "a2": a2, "checkpoint": True, "named_rng": True, "live_sampling": True,
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
        if self.initializer == "supplied":
            return model
        seeds = {name: self.streams.seed_for("init", component=component, purpose=name)
                 for name, p in model.named_parameters() if p.requires_grad and p.numel()}
        init.deterministic_orthogonal_(model, parameter_seeds=seeds)
        self.initialization[component] = {"initializer": "deterministic_orthogonal_named_parameters_v1",
                                         "parameter_seeds": seeds}
        return model

    def build_prior(self, *, dtype=torch.float32):
        p = self.prior_config
        options = {"device": self.device, "dtype": dtype, "learnable": p["learnable"],
                   "generator": self.streams.generator("init", component="prior", purpose="locations"),
                   "init_std": p.get("init_std", 1.)}
        if p["kind"] == "mog":
            options["sigma"] = p["sigma"]
        prior = self.recipe.make_prior(**options)
        self.initialize(prior, component="prior")
        return prior

    def build_trainer(self, generator, discriminator, *, initialize=True, max_steps=None):
        """Build the shared public update path; no task-private training loop."""
        self._check_capabilities()
        if self.execution_path != "public_trainer":
            raise CapabilityError(["public_components context does not own a scalar GANTrainer"])
        if self._trainer is not None:
            raise ValueError("a context owns one trainer; create a new context for another task/attempt")
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
        self._trainer = GANTrainer(self.recipe, generator, discriminator, prior=prior, **options)
        return self._trainer

    def receipt(self):
        return {"api_version": API_VERSION, "execution_path": self.execution_path,
                "recipe": self.recipe.to_dict(), "prior": deepcopy(self.prior_config),
                "capabilities": self.capabilities(), "requires_capabilities": list(self.requires_capabilities),
                "extensions": deepcopy(self.extension_values),
                "api_changes": [self.registry.extensions[name].declaration() for name in sorted(self.extension_values)],
                "initializer": self.initializer, "initialization": deepcopy(self.initialization),
                "rng": self.streams.manifest(),
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
