"""One real batch and one latent batch for both players, critic then generator.

The ring host draws a second latent batch and a second real batch after the
critic step, and ``GANTrainer`` does the same. The two gradients are then not
the gradient of one game on one sample. The splice diagnosis of this probe
found the pass/fail cut in short windows of the critic's particle index and of
the data batch, not in one shared update.

This keeps the alternating order (critic step, then generator step on the
updated critic). It does not extrapolate and it does not average players.
The generator re-forwards the latents the critic just scored, and it is paired
with the same real batch. No coefficient.

Output noise on the generator forward is still drawn, because that draw is the
observation of this forward, not a second minibatch of data or particles.
"""
from __future__ import annotations

import atexit
import json
import os

receipt = {"mechanism": "shared_batch", "steps": 0}
_LOGGED = False
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "shared_batch",
        "setting": "alternating D then G on one real batch and one latent batch",
    }), flush=True)


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def shared_batch_update() -> bool:
    """True when this update should reuse the critic's real batch and latents."""
    global _LOGGED
    if os.environ.get("K3P_DYNAMICS") != "shared_batch":
        return False
    receipt["steps"] += 1
    if not _LOGGED:
        _LOGGED = True
        print(json.dumps({"event": "dynamics_step", "name": "shared_batch", "steps": 1}), flush=True)
    return True
