"""Per-update particle and loss hashes for the cross-build repro audit.

Reads tensors that the host already holds. Does not sample, does not step,
and does not change weights. Installed only when the gate runner is passed
``--trace``.
"""

import hashlib
import json
import sys

_INSTALLED = False


def _sha_array(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha_tensor(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    return _sha_array(array), float(array.astype("float64", copy=False).sum()), int(array.size)


def _sha_module(module):
    digest = hashlib.sha256()
    total = 0.0
    count = 0
    for param in module.parameters():
        array = param.detach().contiguous().cpu().numpy()
        digest.update(array.tobytes())
        total += float(array.astype("float64", copy=False).sum())
        count += int(array.size)
    return digest.hexdigest(), total, count


def _scalar(local, name):
    value = local.get(name)
    if value is None or not hasattr(value, "detach"):
        return None
    if getattr(value, "numel", lambda: 0)() != 1:
        return None
    return float(value.detach())


def _train_frame():
    frame = sys._getframe(2)
    while frame is not None:
        local = frame.f_locals
        if "prior" in local and "generator" in local and "critic" in local:
            return frame.f_code.co_name, local
        frame = frame.f_back
    return None, None


def _state(local, *, losses):
    import torch
    prior = local.get("prior")
    row = {}
    if prior is not None and hasattr(prior, "z") and isinstance(prior.z, torch.Tensor):
        digest, total, count = _sha_tensor(prior.z)
        row.update(z=digest, z_sum=total, z_count=count)
    generator = local.get("generator")
    critic = local.get("critic")
    if generator is not None and hasattr(generator, "parameters"):
        digest, total, count = _sha_module(generator)
        row.update(g=digest, g_sum=total, g_count=count)
    if critic is not None and hasattr(critic, "parameters"):
        digest, total, count = _sha_module(critic)
        row.update(d=digest, d_sum=total, d_count=count)
    if losses:
        row["d_loss"] = _scalar(local, "d_loss")
        row["g_loss"] = _scalar(local, "g_loss")
    return row


def install_state_trace(path):
    """Patch host checkpoints. Call before the phase, after torch is imported."""
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("state trace is already installed")
    from pathlib import Path

    from benchmarks.locked_shared import mode_hold, trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", buffering=1)
    real_checkpoint = mode_hold.checkpoint
    real_set_step = NoisePolicy.set_step
    runs = {"n": 0}

    def write(row):
        handle.write(json.dumps(row, default=float) + "\n")
        handle.flush()

    def set_step(self, completed_steps):
        if completed_steps == 0:
            runs["n"] += 1
            host, local = _train_frame()
            if local is not None:
                write(dict(event="INIT", run=runs["n"], host=host,
                           **_state(local, losses=False)))
        return real_set_step(self, completed_steps)

    def checkpoint(step, measure):
        host, local = _train_frame()
        if local is not None:
            write(dict(event="STATE", run=runs["n"], step=int(step), host=host,
                       **_state(local, losses=True)))
        return real_checkpoint(step, measure)

    mode_hold.checkpoint = checkpoint
    trajectory.checkpoint = checkpoint
    NoisePolicy.set_step = set_step
    _INSTALLED = True
    return handle
