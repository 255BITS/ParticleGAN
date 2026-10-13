"""Passive tensor and stream identities for matched trainer comparisons.

These receipts describe actual inputs, independently of candidate losses and
optimizer settings. No observations, random draws or tensor operations used by
training are changed; bulk learning/RNG states remain in provenance checkpoints.
"""
from contextlib import contextmanager
import hashlib

import torch

from .state import state_digest


def _rng_digest(value):
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def _tensor_metadata(value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}}
    if not isinstance(value, (dict, list, tuple)):
        return {}
    items = value.items() if isinstance(value, dict) else enumerate(value)
    return {name: metadata for key, child in items
            for name, metadata in _tensor_metadata(child, f"{prefix}.{key}" if prefix else str(key)).items()}


class ComparisonProtocol:
    """Hash initial model values and each role's ordered actual target batches."""
    def __init__(self, streams, components):
        values = {name: value.state_dict() if hasattr(value, "state_dict") else value
                  for name, value in sorted(components.items())}
        self.initial_tensors = {name: state_digest(value) for name, value in values.items()}
        self.initial_metadata = {name: _tensor_metadata(value) for name, value in values.items()}
        initial = streams.state_dict()
        self.initial_rng = {key: _rng_digest(value) for key, value in initial["states"].items()}
        self.targets = {}

    def _record_digest(self, role, tensor, digest):
        record = self.targets.setdefault(role, {
            "batches": 0, "samples": 0,
            "sha256": hashlib.sha256(b"forge-actual-target-batches-v1\0").hexdigest()})
        record["sha256"] = hashlib.sha256(
            bytes.fromhex(record["sha256"]) + bytes.fromhex(digest)).hexdigest()
        record["batches"] += 1
        record["samples"] += len(tensor)

    def record(self, role, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError("actual training targets must be batched tensors")
        self._record_digest(role, tensor, state_digest(tensor))

    def record_pair(self, critic, generator):
        if critic is not generator:
            self.record("critic", critic)
            self.record("generator", generator)
            return
        if not isinstance(critic, torch.Tensor) or critic.ndim == 0:
            raise ValueError("actual training targets must be batched tensors")
        digest = state_digest(critic)
        self._record_digest("critic", critic, digest)
        self._record_digest("generator", generator, digest)

    @contextmanager
    def word_step(self, fixture):
        """Observe existing E inputs: categorical data, before learned encoding.

        The first E call is the critic's target batch and the second is the
        generator/inverse target batch. Further protected-loss probes do not
        count as additional consumed training batches. Evaluation has no hook.
        """
        roles = iter(("critic", "generator"))
        observed = []
        def capture(module, args):
            role = next(roles, None)
            if role is not None:
                self.record(role, args[0])
                observed.append(role)
        handle = fixture.E.register_forward_pre_hook(capture)
        try:
            yield
            if len(observed) != 2:
                raise ValueError("word update did not expose both actual target batches")
        finally:
            handle.remove()

    def receipt(self, streams):
        """Certify coverage of the named stream state saved with provenance."""
        state = streams.state_dict()
        bindings = state["manifest"]["bindings"]
        if set(bindings) != set(state["states"]):
            raise ValueError("named RNG checkpoint lacks a declared stream")
        if not self.initial_rng.keys() <= bindings.keys():
            raise ValueError("a named training stream vanished during execution")
        rng = {}
        for key, binding in sorted(bindings.items()):
            # Streams bound after construction have consumed no earlier draws.
            before = self.initial_rng.get(key, binding["initial_state_sha256"])
            after = _rng_digest(state["states"][key])
            rng[key] = {"family": binding["family"], "initial_state_sha256": before,
                        "final_state_sha256": after, "changed": before != after}
        return {"schema_version": 1,
                "initial_tensors_sha256": dict(self.initial_tensors),
                "initial_tensor_metadata": self.initial_metadata,
                "actual_target_batches": {key: dict(value) for key, value in sorted(self.targets.items())},
                "rng_checkpoints": {"bindings": rng, "all_named_streams_checkpointed": True},
                "target_scope": "ordered original data tensors consumed by each training role; learned encodings excluded"}
