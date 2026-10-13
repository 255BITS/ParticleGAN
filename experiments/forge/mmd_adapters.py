"""Opt-in MMD bindings around byte-pinned native component loops.

The original task evaluator hashes include their adapter/fixture files. These
subclasses inherit construction, training and scoring, and only extend existing
output-loss hooks. Scoped factories use a context-local selection: unrelated
threads still construct the exact original classes. Binding sessions serialize
global factory installation and restore every original on exceptions.
Energy U and MMD composition is intentionally unsupported on these component
bindings; the native public trainer can consume both objectives. MMD uses only
the original generated output and real minibatch, with no conditioning
coordinates, paired supervision, forward passes or random draws added.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

import torch

from particlegan.kernel_mmd import OutputMMDU
from . import behavior_adapters, word_adapter, word_tasks, energy_adapters
from .api import CapabilityError, resolve_public_recipe, task_policy_blockers
from .contracts import atomic_json, file_hash
from benchmarks.toy_audit import api_images


_Behavior = energy_adapters._Behavior
_Word = energy_adapters._Word
_Restore = energy_adapters._Restore
_LOCK = energy_adapters._LOCK
_ACTIVE = ContextVar("forge_mmd_binding", default=None)


def _binding(native):
    return dict(version=1, consumer="output_mmd_u_v1", native_loop=native,
                source="experiments/forge/mmd_adapters.py",
                source_sha256=file_hash(Path(__file__)),
                objective_source="particlegan/kernel_mmd.py",
                objective_source_sha256=file_hash(Path(__file__).resolve().parents[2] / "particlegan/kernel_mmd.py"),
                conditioning_used_in_distance=False, new_paired_supervision=False,
                new_forward_passes=0, new_random_draws=0)


class MMDBehaviorComponents(_Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recipe.energy_distance_weight:
            raise CapabilityError(["public_components cannot compose energy U and output MMD"])
        if not self.recipe.output_mmd_weight:
            raise ValueError("MMD component binding requires positive weight")
        self.mmd_consumer = OutputMMDU(self.recipe)

    def add_transport_loss(self, total, fake, real, *, conditioning=None):
        total = super().add_transport_loss(total, fake, real, conditioning=conditioning)
        return self.mmd_consumer.add(total, fake.flatten(1), real.flatten(1), conditioning=conditioning)

    def guards(self):
        result = super().guards()
        calls = self.mmd_consumer.active_calls
        expected = result["optimizer_updates"].get("generator", result["optimizer_updates"].get("prior", 0))
        result.update(component_mmd_requested=True, component_mmd_active_calls=calls,
                      hooks_exercised=result["hooks_exercised"] and calls == expected)
        return result

    def receipt(self):
        return {**super().receipt(), "component_mmd": self.mmd_consumer.state_dict(),
                "mmd_binding": _binding("experiments.forge.behavior_adapters.run_behavior")}

    def provenance_state(self):
        return {**super().provenance_state(), "component_mmd": self.mmd_consumer.state_dict()}


class _OutputBridge:
    def __init__(self, transport, mmd):
        self.transport, self.mmd = transport, mmd

    def add(self, total, fake, real, *, conditioning=None):
        if self.transport is not None:
            total = self.transport.add(total, fake, real, conditioning=conditioning)
        return self.mmd.add(total, fake, real, conditioning=conditioning)


class MMDWordFixture(_Word):
    """The exact inherited word step with one additional output-marginal hook."""
    def __init__(self, *args, **kwargs):
        components = kwargs.get("components")
        if components is not None and components.recipe.energy_distance_weight:
            raise CapabilityError(["public_components cannot compose energy U and output MMD"])
        super().__init__(*args, **kwargs)
        if self.recipe.energy_distance_weight:
            raise CapabilityError(["public_components cannot compose energy U and output MMD"])
        if not self.recipe.output_mmd_weight:
            raise ValueError("MMD word binding requires positive weight")
        self.mmd_consumer = OutputMMDU(self.recipe)

    def step(self):
        original = self.transport
        self.transport = _OutputBridge(original, self.mmd_consumer)
        try:
            return super().step()
        finally:
            self.transport = original

    def state_dict(self):
        return {**super().state_dict(), "component_mmd": self.mmd_consumer.state_dict()}

    def restore_component_mmd(self, state, *, validate_only=False):
        if state is None:
            raise ValueError("active word fixture is missing MMD checkpoint")
        self.mmd_consumer.validate_state_dict(state)
        if not validate_only:
            self.mmd_consumer.load_state_dict(state)

    def restore_component_transport(self, state):
        # The pinned parent restore calls this existing hook after restoring
        # policy state. MMD was independently validated before any mutation.
        if not hasattr(self, "_pending_mmd_state"):
            raise ValueError("active word restoration requires its separate MMD checkpoint")
        super().restore_component_transport(state)
        self.restore_component_mmd(self._pending_mmd_state)


def _restore_parent(request, task, prerequisites, fixture, context):
    if not isinstance(fixture, MMDWordFixture):
        return _Restore(request, task, prerequisites, fixture, context)
    if _ACTIVE.get() is None:
        raise ValueError("active MMD word restoration requires a scoped binding")
    parent = (prerequisites or {}).get(task["execution"]["continuation_of"])
    if not parent:
        return _Restore(request, task, prerequisites, fixture, context)
    evidence = parent["result"]["evidence"]
    checkpoint = evidence["checkpoint"]
    saved = torch.load(Path(evidence["artifact_root"]) / checkpoint["path"],
                       weights_only=True, map_location=context.device)
    mmd = saved["fixture"].get("component_mmd")
    fixture.restore_component_mmd(mmd, validate_only=True)
    # Public consumer JSON receipts accept lists. A native Torch checkpoint
    # must retain its exact tuple/float representation, because the inherited
    # parent verifies the full fixture digest after loading; reject here before
    # it can mutate policy, optimizer or stream state.
    if (type(mmd["bandwidths"]) is not tuple
            or any(type(width) is not float for width in mmd["bandwidths"])
            or type(mmd["weight"]) is not type(fixture.recipe.output_mmd_weight)):
        raise ValueError("word MMD checkpoint must retain native weight and tuple/float bandwidth representation")
    if mmd["active_calls"] != checkpoint["completed_steps"]:
        raise ValueError("word MMD calls differ from acquisition prefix")
    fixture._pending_mmd_state = deepcopy(mmd)
    try:
        return _Restore(request, task, prerequisites, fixture, context)
    finally:
        del fixture._pending_mmd_state


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
            cls = MMDBehaviorComponents if resolve_public_recipe(request["candidate"]).output_mmd_weight else _Behavior
            instance = cls(*args, **kwargs)
            _ACTIVE.get().append(instance)
            return instance
        def word_factory(*args, **kwargs):
            components = kwargs.get("components")
            if _ACTIVE.get() is None or components is None or not components.recipe.output_mmd_weight:
                return _Word(*args, **kwargs)
            instance = MMDWordFixture(*args, **kwargs)
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
    if not resolve_public_recipe(request["candidate"]).output_mmd_weight:
        return energy_adapters.run_behavior(request, task, output_dir, device)
    blockers = task_policy_blockers(task, request["candidate"])
    if blockers:
        raise CapabilityError(blockers)
    with _factories() as instances:
        result = behavior_adapters.run_behavior(request, task, output_dir, device)
        consumer = next(x.mmd_consumer for x in instances if isinstance(x, MMDBehaviorComponents))
        result["evidence"]["component_mmd"] = consumer.state_dict()
        atomic_json(Path(output_dir) / "result.json", result)
        return result


def run_word(request, task, output, device, **kwargs):
    if not resolve_public_recipe(request["candidate"]).output_mmd_weight:
        return energy_adapters.run_word(request, task, output, device, **kwargs)
    blockers = task_policy_blockers(task, request["candidate"])
    if blockers:
        raise CapabilityError(blockers)
    with _factories() as instances:
        returned = word_adapter.run_word(request, task, output, device, **kwargs)
        fixture = next(x for x in instances if isinstance(x, MMDWordFixture))
        result = returned[0] if isinstance(returned, tuple) else returned
        evidence = result["evidence"]
        calls = fixture.mmd_consumer.active_calls
        evidence["component_mmd"] = fixture.mmd_consumer.state_dict()
        evidence["mmd_binding"] = _binding("benchmarks.toy_audit.api_images.WordFixture.step")
        guards = evidence["guards"]
        guards.update(component_mmd_requested=True, component_mmd_active_calls=calls,
                      hooks_exercised=guards["hooks_exercised"] and calls == fixture.completed_steps)
        atomic_json(Path(output) / "adapter-receipt.json", result)
        return returned
