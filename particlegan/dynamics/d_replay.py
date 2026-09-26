"""Critic history buffer (Shrivastava et al., CVPR 2017).

D's fake batch is half the current generator or particle samples and half
samples drawn from a FIFO of past fakes. The generator loss and the particle
update still use the current sample only. There is no per-mode logic.

Buffer length: 40 critic updates. Declared before the run and not swept.
#177: the gradients that empty a mode onto a neighbor the critic scores
higher jump about 40 updates before the crossing, and the emptied mode stays
empty behind a real-vs-fake logit gap of about 4. A FIFO of 40 updates is
long enough that, through that climb, D is still trained on fakes from the
mode that is leaving, and short enough that the memory is this generator's
recent history rather than a second dataset.

The half split is the history buffer's usual rate (a stored fake stands in
for a current one half the time), taken as a fixed cut of the batch so the
draw adds no RNG. The historical half is an equal stride across the FIFO,
oldest sample first, so one critic step sees the whole window. The oldest
update is dropped when the buffer exceeds 40.
"""
from __future__ import annotations

import atexit
import json
import os

import torch

BUFFER_UPDATES = 40

receipt = {
    "mechanism": "d_replay",
    "buffer_updates": BUFFER_UPDATES,
    "mixes": 0,
    "passthrough": 0,
    "held": 0,
}
_state = {"samples": None, "batch": None, "cap": None}
_INSTALLED = False
_LOGGED = False


def reset() -> None:
    """Drop the FIFO. Tests use this; a training process starts empty."""
    _state["samples"] = None
    _state["batch"] = None
    _state["cap"] = None
    receipt["mixes"] = 0
    receipt["passthrough"] = 0
    receipt["held"] = 0


def install() -> None:
    """Log the declared buffer. The mix itself is the fake batch D is given."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "d_replay",
        "buffer_updates": BUFFER_UPDATES,
        "setting": "D fake batch is half current and half an even stride of a 40-update FIFO",
    }), flush=True)


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def _note(kind: str) -> None:
    global _LOGGED
    receipt[kind] += 1
    receipt["held"] = 0 if _state["samples"] is None else int(_state["samples"].shape[0])
    if not _LOGGED:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "d_replay",
            "buffer_updates": BUFFER_UPDATES,
            "batch": _state["batch"],
            kind: receipt[kind],
        }), flush=True)


def replay_fakes(current: torch.Tensor) -> torch.Tensor:
    """Fake batch for the critic. Flag off: ``current`` itself, unmodified."""
    if os.environ.get("K3P_DYNAMICS") != "d_replay":
        return current
    if current.ndim < 2 or current.shape[0] < 2 or current.shape[0] % 2:
        raise ValueError("d_replay needs a fake batch with a positive even count")
    batch = int(current.shape[0])
    if _state["batch"] is None:
        _state["batch"] = batch
        _state["cap"] = BUFFER_UPDATES * batch
    elif _state["batch"] != batch:
        raise ValueError("d_replay keeps one batch size per process")
    half = batch // 2
    stored = _state["samples"]
    if stored is None or stored.shape[0] < half:
        shown = current
        _push(current)
        _note("passthrough")
        return shown
    if stored.shape[1:] != current.shape[1:] or stored.dtype != current.dtype or stored.device != current.device:
        raise ValueError("d_replay buffer does not match this fake batch")
    index = (torch.arange(half, device=stored.device) * stored.shape[0]) // half
    history = stored.index_select(0, index).detach().clone()
    shown = torch.cat((current[:half].detach(), history), dim=0)
    _push(current)
    _note("mixes")
    return shown


def _push(current: torch.Tensor) -> None:
    fresh = current.detach().clone()
    stored = _state["samples"]
    _state["samples"] = fresh if stored is None else torch.cat((stored, fresh), dim=0)
    cap = _state["cap"]
    if _state["samples"].shape[0] > cap:
        _state["samples"] = _state["samples"][-cap:].contiguous()
