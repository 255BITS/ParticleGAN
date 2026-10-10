"""Content identity for public checkpoints, independent of torch.save envelopes."""
import hashlib
import json
import math

import torch


def require_same_formulation(original, current):
    """Continuation preserves the entire public formulation and RNG binding law."""
    for key in ("schema", "api_version", "recipe", "prior", "extensions", "initializer", "initialization"):
        if original[key] != current[key]:
            raise ValueError(f"continuation changed static context field {key}")
    for key in ("schema", "recipe", "optimizer_options", "penalty_options", "device", "dtype", "requires_grad", "initial_lrs"):
        if original["trainer"][key] != current["trainer"][key]:
            raise ValueError(f"continuation changed static trainer field {key}")
    if original["streams"]["manifest"] != current["streams"]["manifest"]:
        raise ValueError("continuation changed named RNG stream bindings")
    for state in (original, current):
        if state["recipe"] != state["trainer"]["recipe"]:
            raise ValueError("context and trainer recipe disagree")
        require_consistent_rng(state)


def require_consistent_rng(state):
    from .api import TRAINER_STREAM_BINDINGS
    required = set(TRAINER_STREAM_BINDINGS)
    if state["prior"]["kind"] != "mog":
        required.remove("prior_noise_generator")
    if set(state["trainer"]["streams"]) != required:
        raise ValueError("public trainer RNG checkpoint is incomplete")
    named = state["streams"]
    bindings = named["manifest"]["bindings"]
    if set(bindings) != set(named["states"]):
        raise ValueError("named RNG states and bindings disagree")
    for name, value in state["trainer"]["streams"].items():
        expected = TRAINER_STREAM_BINDINGS.get(name)
        matching = [key for key, binding in bindings.items()
                    if (binding["family"], binding["component"], binding["purpose"]) == expected]
        if len(matching) != 1 or not torch.equal(value, named["states"][matching[0]]):
            raise ValueError("trainer and named RNG checkpoint states disagree")


def require_optimizer_steps(state, expected):
    """Cross-check external update labels against both actual Adam state tables."""
    optimizers = state["trainer"]["optimizers"]
    if len(optimizers) != 2:
        raise ValueError("public GAN state requires both optimizer histories")
    for optimizer in optimizers:
        counters = [row["step"] for row in optimizer["state"].values() if "step" in row]
        if not counters:
            raise ValueError("measured optimizer counters are missing")
        for count in counters:
            if isinstance(count, torch.Tensor):
                if count.numel() != 1:
                    raise ValueError("optimizer counter must be scalar")
                count = count.item()
            if type(count) not in (int, float) or not math.isfinite(count) or count != expected:
                raise ValueError("measured optimizer counters differ from the declared updates")


def state_digest(value):
    digest = hashlib.sha256(b"forge-state-v1\0")

    def write(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"T")
            tensor = item.detach().cpu().contiguous()
            write(("tensor", str(tensor.dtype), tuple(tensor.shape)))
            # Byte view covers bfloat16 as well as ordinary numpy dtypes.
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        elif isinstance(item, dict):
            digest.update(b"D" + len(item).to_bytes(8, "big"))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                write(key)
                write(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update((b"L" if isinstance(item, list) else b"U") + len(item).to_bytes(8, "big"))
            for child in item:
                write(child)
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(b"S")
            payload = json.dumps([type(item).__name__, item], allow_nan=False).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        else:
            raise TypeError(f"unsupported checkpoint value: {type(item).__name__}")

    write(value)
    return digest.hexdigest()
