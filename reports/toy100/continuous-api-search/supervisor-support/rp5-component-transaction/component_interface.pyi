"""Proposed public surface, NOT an implementation or a qualified RP5 extension.

Tensor/Module/Recipe aliases describe existing package objects. The library
constructs all optimizers from the recipe; callbacks supply objectives only.
"""
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TypedDict

Tensor = Any
Module = Any
Recipe = Any


class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, value: Mapping[str, Any]) -> None: ...


class ParameterGroup(TypedDict):
    parameters: Sequence[str]  # Qualified registry names, stable explicit order.
    rate_role: Literal["generator", "prior", "critic"]


class GeneratorOptimizer(TypedDict):
    name: str
    groups: Sequence[ParameterGroup]
    latent_table: str | None  # Exactly one plain table, alone in its group.
    direct_particles: Sequence[str]  # Empty or exactly one complete group.


class CriticOptimizer(TypedDict):
    name: str
    module: str  # One critic root suffices for all eight current custom hosts.
    groups: Sequence[ParameterGroup]


class EMA(TypedDict):
    name: str
    sources: Sequence[str]
    buffer_rule: Literal["copy", "none"]
    decay: float


class FieldModes(TypedDict):
    discriminator: Mapping[str, bool]
    generator: Mapping[str, bool]


class Registry(TypedDict):
    modules: Mapping[str, Module]
    parameters: Mapping[str, Tensor]  # Standalone parameters only; no duplicate ownership.
    generator_order: Sequence[str]  # Preserve G then prior order on ordinary binding.
    critic_order: Sequence[str]
    generator_optimizers: Sequence[GeneratorOptimizer]
    critic_optimizer: CriticOptimizer
    ema: Sequence[EMA]
    stateful: Mapping[str, Stateful]  # Caller RNG/cursors/context state requiring rollback.
    modes: FieldModes
    binding_id: str  # Caller pins objective/role/noise-site source identity.


class Loss(TypedDict):
    total: Tensor  # Complete caller-built expression, with original sum order.
    terms: Mapping[str, Tensor]  # Detached diagnostics only; never controller inputs.


class PrecisionProbe(TypedDict):
    real: Tensor  # Data coordinates only, no concatenated condition labels.
    view: Callable[[Module, Tensor], Tensor]  # Must evaluate supplied live/reference root.
    context_id: str  # Trace label; current frozen context is closed over by view.


class FieldContext(Protocol):
    modules: Mapping[str, Module]
    parameters: Mapping[str, Tensor]
    discriminator: Module
    # Helpers use package-owned current noise amplitudes and replayable streams.
    def input_noise(self, data: Tensor) -> Tensor: ...
    def output_noise(self, generated: Tensor) -> Tensor: ...
    def sample_prior(self, name: str, n: int) -> Any: ...
    def critic_penalty(self, view: Module, real: Tensor, fake: Tensor) -> Tensor: ...
    def adversarial_loss(self) -> Any: ...
    def prior_regularizer(self) -> Any: ...


class Objectives(Protocol):
    # No optimizer steps, controller calls, LR changes, or evaluator metrics.
    # Parameter-dependent G/encoder/prior values must be recomputed each field.
    def discriminator_loss(self, context: FieldContext, batch: Any) -> Loss: ...
    def generator_loss(self, context: FieldContext, batch: Any) -> Loss: ...
    def precision_probes(self, batch: Any) -> Sequence[PrecisionProbe]: ...


class ComponentTrainer:
    def __init__(self, recipe: Recipe, *, registry: Registry,
                 seed: int = 0, serial_backward: bool = False,
                 optimizer_options: Mapping[str, Any] | None = None) -> None: ...
    def step(self, batch: Any, *, objectives: Objectives,
             collect_stats: bool = False) -> Mapping[str, Any]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, value: Mapping[str, Any]) -> None: ...
