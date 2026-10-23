"""Deterministic execution scope and isolated named RNG streams for particlegan_forge."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

import torch

RNG_VERSION = "forge-rng-v1"
STREAM_NAMES = ("init", "data", "prior", "noise", "eval")


def _digest(state: torch.Tensor) -> str:
    return hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()


class NamedStreams:
    """Lazily derive isolated PyTorch generators from (seed, family, component, purpose).

    Separates constructor/init, data/stream, prior, training_noise, and evaluation draws
    so changing a loss or running an evaluation check cannot shift training streams.
    """

    def __init__(self, seed: int = 0, *, version: str = RNG_VERSION, device: str | torch.device = "cpu"):
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if version != RNG_VERSION:
            raise ValueError("unsupported RNG derivation version")
        self.seed = seed
        self.version = version
        self.device = torch.device(device)
        self._streams: dict[str, torch.Generator] = {}
        self._bindings: dict[str, dict] = {}

    @staticmethod
    def _key(name: str, component: str, purpose: str, device: str | torch.device) -> str:
        if name not in STREAM_NAMES:
            raise ValueError(f"unknown RNG stream family: {name!r}")
        if any(not isinstance(value, str) or not value for value in (component, purpose)):
            raise ValueError("stream component and purpose must be nonempty strings")
        return json.dumps([name, component, purpose, str(device)], separators=(",", ":"))

    def seed_for(self, name: str, *, component: str = "shared", purpose: str = "default") -> int:
        self._key(name, component, purpose, "cpu")
        payload = json.dumps([self.version, self.seed, name, component, purpose], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2 ** 63)

    def generator(
        self,
        name: str,
        *,
        component: str = "shared",
        purpose: str = "default",
        device: str | torch.device | None = None,
    ) -> torch.Generator:
        device = self.device if device is None else torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = self._key(name, component, purpose, device)
        if key not in self._streams:
            seed = self.seed_for(name, component=component, purpose=purpose)
            self._streams[key] = torch.Generator(device=device).manual_seed(seed)
            self._bindings[key] = {
                "family": name,
                "component": component,
                "purpose": purpose,
                "device": str(device),
                "seed": seed,
                "initial_state_sha256": _digest(self._streams[key].get_state()),
            }
        return self._streams[key]

    @contextmanager
    def fork(
        self,
        name: str,
        *,
        component: str = "shared",
        purpose: str = "default",
        device: str | torch.device | None = None,
    ):
        """Bind global torch draws to a named stream and restore caller RNG afterward."""
        stream = self.generator(name, component=component, purpose=purpose, device=device)
        dev = stream.device
        devices = [dev.index] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if dev.type == "cuda":
                torch.cuda.set_rng_state(stream.get_state(), dev)
            else:
                torch.set_rng_state(stream.get_state())
            try:
                yield stream
            finally:
                state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()
                stream.set_state(state)

    def manifest(self) -> dict:
        return {
            "version": self.version,
            "seed": self.seed,
            "runtime": {"torch": str(torch.__version__), "rng": "torch.Generator"},
            "bindings": {key: dict(self._bindings[key]) for key in sorted(self._bindings)},
        }

    def audit(self) -> dict[str, str]:
        return {key: _digest(stream.get_state()) for key, stream in sorted(self._streams.items())}

    def eval_stream_keys(self) -> set[str]:
        return {key for key, meta in self._bindings.items() if meta["family"] == "eval"}

    @staticmethod
    def compare(before: dict[str, str], after: dict[str, str], *, allowed=()) -> dict:
        changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
        unintended = [key for key in changed if key not in set(allowed)]
        return {
            "changed_streams": changed,
            "unintended_streams": unintended,
            "unintended_rng_deviations": len(unintended),
        }

    @contextmanager
    def isolated_evaluation(self):
        """Run an evaluation callback while auditing that only 'eval' streams advance."""
        before = self.audit()
        grad_enabled = torch.is_grad_enabled()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
            try:
                with torch.no_grad():
                    yield self
            finally:
                torch.set_grad_enabled(grad_enabled)
        after = self.audit()
        allowed = self.eval_stream_keys()
        self.last_eval_audit = self.compare(before, after, allowed=allowed)

    def state_dict(self) -> dict:
        return {
            "schema": 1,
            "manifest": self.manifest(),
            "states": {key: stream.get_state().clone() for key, stream in sorted(self._streams.items())},
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("invalid named-stream checkpoint schema")
        manifest = state["manifest"]
        probe = NamedStreams(self.seed, version=self.version, device=self.device)
        for key, binding in manifest["bindings"].items():
            stream = probe.generator(
                binding["family"],
                component=binding["component"],
                purpose=binding["purpose"],
                device=binding["device"],
            )
            stream.set_state(state["states"][key].cpu())
        self._streams = probe._streams
        self._bindings = deepcopy(probe._bindings)


@contextmanager
def deterministic_scope(seed: int = 0):
    """Run inside a deterministic, single-thread PyTorch scope and restore caller state."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_bench = torch.backends.cudnn.benchmark if torch.backends.cudnn.is_available() else False
    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.set_num_threads(prev_threads)
            torch.use_deterministic_algorithms(prev_det)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = prev_cudnn_bench
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
