"""Versioned named PyTorch streams; independent of queue order and candidate ID."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

import torch


RNG_VERSION = "forge-rng-v1"
STREAM_NAMES = ("init", "data", "prior", "noise", "eval")


def _digest(state):
    return hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()


class NamedStreams:
    """Lazily derive generators from (seed, family, component, purpose).

    A draw on one stream cannot consume another stream. Audits compare exact
    states, not guessed draw counts (PyTorch kernels consume RNG differently).
    Names deliberately exclude tensor shapes, tasks, candidates, and workers.
    """
    def __init__(self, seed=0, *, version=RNG_VERSION, device="cpu"):
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if version != RNG_VERSION:
            raise ValueError("unsupported RNG derivation version")
        self.seed, self.version, self.device = seed, version, torch.device(device)
        self._streams, self._bindings = {}, {}

    @staticmethod
    def _key(name, component, purpose, device):
        if name not in STREAM_NAMES:
            raise ValueError("unknown RNG stream family")
        if any(not isinstance(value, str) or not value for value in (component, purpose)):
            raise ValueError("stream component and purpose must be nonempty strings")
        return json.dumps([name, component, purpose, str(device)], separators=(",", ":"))

    def seed_for(self, name, *, component="shared", purpose="default"):
        self._key(name, component, purpose, "cpu")  # validate; device does not select the seed
        payload = json.dumps([self.version, self.seed, name, component, purpose], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 63)

    def generator(self, name, *, component="shared", purpose="default", device=None):
        device = self.device if device is None else torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = self._key(name, component, purpose, device)
        if key not in self._streams:
            seed = self.seed_for(name, component=component, purpose=purpose)
            self._streams[key] = torch.Generator(device=device).manual_seed(seed)
            self._bindings[key] = {"family": name, "component": component, "purpose": purpose,
                                   "device": str(device), "seed": seed,
                                   "initial_state_sha256": _digest(self._streams[key].get_state())}
        return self._streams[key]

    @contextmanager
    def fork(self, name, *, component="shared", purpose="default", device=None):
        """Bind global torch draws to a named stream and restore caller RNG."""
        stream = self.generator(name, component=component, purpose=purpose, device=device)
        device = stream.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if device.type == "cuda":
                torch.cuda.set_rng_state(stream.get_state(), device)
            else:
                torch.set_rng_state(stream.get_state())
            try:
                yield stream
            finally:
                state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
                stream.set_state(state)

    def manifest(self):
        return {"version": self.version, "seed": self.seed,
                "runtime": {"torch": str(torch.__version__), "rng": "torch.Generator"},
                "bindings": {key: dict(self._bindings[key]) for key in sorted(self._bindings)}}

    def audit(self):
        return {key: _digest(stream.get_state()) for key, stream in sorted(self._streams.items())}

    @staticmethod
    def compare(before, after, *, allowed=()):
        """List unexpected changed/new/deleted bindings; never certify by count alone."""
        changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
        unintended = [key for key in changed if key not in set(allowed)]
        return {"changed_streams": changed, "unintended_streams": unintended,
                "unintended_rng_deviations": len(unintended)}

    @contextmanager
    def preserve(self):
        """Restore all named streams even when an evaluator raises."""
        state = self.state_dict()
        try:
            yield self
        finally:
            self.load_state_dict(state)

    def state_dict(self):
        return {"schema": 1, "manifest": self.manifest(),
                "states": {key: stream.get_state().clone() for key, stream in sorted(self._streams.items())}}

    def validate_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"schema", "manifest", "states"} or state["schema"] != 1:
            raise ValueError("invalid named-stream checkpoint schema")
        manifest = state["manifest"]
        if (not isinstance(manifest, dict) or set(manifest) != {"version", "seed", "runtime", "bindings"}
                or manifest["version"] != self.version or manifest["seed"] != self.seed
                or manifest["runtime"] != self.manifest()["runtime"]
                or not isinstance(manifest["bindings"], dict) or not isinstance(state["states"], dict)
                or manifest["bindings"].keys() != state["states"].keys()):
            raise ValueError("named-stream checkpoint identity mismatch")
        probe = NamedStreams(self.seed, version=self.version, device=self.device)
        try:
            for key, binding in manifest["bindings"].items():
                stream = probe.generator(binding["family"], component=binding["component"],
                                         purpose=binding["purpose"], device=binding["device"])
                if probe._bindings.get(key) != binding:
                    raise ValueError("RNG binding disagrees with derivation")
                stream.set_state(state["states"][key].cpu())
        except (KeyError, TypeError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid named-stream state") from error
        return probe

    def load_state_dict(self, state):
        probe = self.validate_state_dict(state)  # all validation precedes mutation
        for key, stream in probe._streams.items():
            if key in self._streams:
                self._streams[key].set_state(stream.get_state())
            else:
                self._streams[key] = stream
        for key in set(self._streams) - set(probe._streams):
            del self._streams[key]
        self._bindings = deepcopy(probe._bindings)
