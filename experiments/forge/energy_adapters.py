"""Opt-in energy bindings around byte-pinned native component loops.

The original task evaluator hashes include their adapter/fixture files. These
subclasses inherit construction, training and scoring, and only extend existing
output-loss hooks. Scoped factories use a context-local selection: unrelated
threads still construct the exact original classes. Binding sessions serialize
global factory installation and restore every original on exceptions.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import RLock

import torch

from particlegan.energy_distance import OutputEnergyU
from . import behavior_adapters, word_adapter, word_tasks
from .api import resolve_public_recipe
from .contracts import atomic_json, file_hash
from benchmarks.toy_audit import api_images


_Behavior = behavior_adapters.BehaviorComponents
_Word = api_images.WordFixture
_Restore = word_tasks.restore_parent
_LOCK = RLock()
_ACTIVE = ContextVar("forge_energy_binding", default=None)


def _binding(native):
    return dict(version=1, consumer="output_energy_u_v1", native_loop=native,
                source="experiments/forge/energy_adapters.py",
                source_sha256=file_hash(Path(__file__)),
                objective_source="particlegan/energy_distance.py",
                objective_source_sha256=file_hash(Path(__file__).resolve().parents[2] / "particlegan/energy_distance.py"))


class EnergyBehaviorComponents(_Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.recipe.energy_distance_weight:
            raise ValueError("energy component binding requires positive weight")
        self.energy_consumer = OutputEnergyU(self.recipe)

    def add_transport_loss(self, total, fake, real, *, conditioning=None):
        total = super().add_transport_loss(total, fake, real, conditioning=conditioning)
        return self.energy_consumer.add(total, fake.flatten(1), real.flatten(1), conditioning=conditioning)

    def guards(self):
        result = super().guards()
        calls = self.energy_consumer.active_calls
        expected = result["optimizer_updates"].get("generator", result["optimizer_updates"].get("prior", 0))
        result.update(component_energy_requested=True, component_energy_active_calls=calls,
                      hooks_exercised=result["hooks_exercised"] and calls == expected)
        return result

    def receipt(self):
        return {**super().receipt(), "component_energy": self.energy_consumer.state_dict(),
                "energy_binding": _binding("experiments.forge.behavior_adapters.run_behavior")}

    def provenance_state(self):
        return {**super().provenance_state(), "component_energy": self.energy_consumer.state_dict()}


class _OutputBridge:
    def __init__(self, transport, energy):
        self.transport, self.energy = transport, energy

    def add(self, total, fake, real, *, conditioning=None):
        if self.transport is not None:
            total = self.transport.add(total, fake, real, conditioning=conditioning)
        return self.energy.add(total, fake, real, conditioning=conditioning)


class EnergyWordFixture(_Word):
    """The exact inherited word step with one additional output-marginal hook."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.recipe.energy_distance_weight:
            raise ValueError("energy word binding requires positive weight")
        self.energy_consumer = OutputEnergyU(self.recipe)

    def step(self):
        original = self.transport
        self.transport = _OutputBridge(original, self.energy_consumer)
        try:
            return super().step()
        finally:
            self.transport = original

    def state_dict(self):
        return {**super().state_dict(), "component_energy": self.energy_consumer.state_dict()}

    def restore_component_energy(self, state, *, validate_only=False):
        if state is None:
            raise ValueError("active word fixture is missing energy checkpoint")
        self.energy_consumer.validate_state_dict(state)
        if not validate_only:
            self.energy_consumer.load_state_dict(state)

    def restore_component_transport(self, state):
        # The pinned parent restore calls this existing hook after restoring
        # policy state. Energy was independently validated before any mutation.
        if not hasattr(self, "_pending_energy_state"):
            raise ValueError("active word restoration requires its separate energy checkpoint")
        super().restore_component_transport(state)
        self.restore_component_energy(self._pending_energy_state)


def _restore_parent(request, task, prerequisites, fixture, context):
    if _ACTIVE.get() is None or not isinstance(fixture, EnergyWordFixture):
        return _Restore(request, task, prerequisites, fixture, context)
    parent = (prerequisites or {}).get(task["execution"]["continuation_of"])
    if not parent:
        return _Restore(request, task, prerequisites, fixture, context)
    evidence = parent["result"]["evidence"]
    checkpoint = evidence["checkpoint"]
    saved = torch.load(Path(evidence["artifact_root"]) / checkpoint["path"],
                       weights_only=True, map_location=context.device)
    energy = saved["fixture"].get("component_energy")
    fixture.restore_component_energy(energy, validate_only=True)
    if energy["active_calls"] != checkpoint["completed_steps"]:
        raise ValueError("word energy calls differ from acquisition prefix")
    fixture._pending_energy_state = deepcopy(energy)
    try:
        return _Restore(request, task, prerequisites, fixture, context)
    finally:
        del fixture._pending_energy_state


@contextmanager
def _factories():
    with _LOCK:
        instances = []
        token = _ACTIVE.set(instances)
        originals = (behavior_adapters.BehaviorComponents, api_images.WordFixture, word_tasks.restore_parent)
        def behavior_factory(*args, **kwargs):
            if _ACTIVE.get() is None:
                return _Behavior(*args, **kwargs)
            request = args[0] if args else kwargs["request"]
            cls = EnergyBehaviorComponents if resolve_public_recipe(request["candidate"]).energy_distance_weight else _Behavior
            instance = cls(*args, **kwargs)
            _ACTIVE.get().append(instance)
            return instance
        def word_factory(*args, **kwargs):
            components = kwargs.get("components")
            if _ACTIVE.get() is None or components is None or not components.recipe.energy_distance_weight:
                return _Word(*args, **kwargs)
            instance = EnergyWordFixture(*args, **kwargs)
            _ACTIVE.get().append(instance)
            return instance
        behavior_adapters.BehaviorComponents = behavior_factory
        api_images.WordFixture = word_factory
        word_tasks.restore_parent = _restore_parent
        try:
            yield instances
        finally:
            behavior_adapters.BehaviorComponents, api_images.WordFixture, word_tasks.restore_parent = originals
            _ACTIVE.reset(token)


def run_behavior(request, task, output_dir, device="cpu"):
    if not resolve_public_recipe(request["candidate"]).energy_distance_weight:
        return behavior_adapters.run_behavior(request, task, output_dir, device)
    with _factories() as instances:
        result = behavior_adapters.run_behavior(request, task, output_dir, device)
        consumer = next(x.energy_consumer for x in instances if isinstance(x, EnergyBehaviorComponents))
        result["evidence"]["component_energy"] = consumer.state_dict()
        atomic_json(Path(output_dir) / "result.json", result)
        return result


def run_word(request, task, output, device, **kwargs):
    if not resolve_public_recipe(request["candidate"]).energy_distance_weight:
        return word_adapter.run_word(request, task, output, device, **kwargs)
    with _factories() as instances:
        returned = word_adapter.run_word(request, task, output, device, **kwargs)
        fixture = next(x for x in instances if isinstance(x, EnergyWordFixture))
        result = returned[0] if isinstance(returned, tuple) else returned
        evidence = result["evidence"]
        calls = fixture.energy_consumer.active_calls
        evidence["component_energy"] = fixture.energy_consumer.state_dict()
        evidence["energy_binding"] = _binding("benchmarks.toy_audit.api_images.WordFixture.step")
        guards = evidence["guards"]
        guards.update(component_energy_requested=True, component_energy_active_calls=calls,
                      hooks_exercised=guards["hooks_exercised"] and calls == fixture.completed_steps)
        atomic_json(Path(output) / "adapter-receipt.json", result)
        return returned
