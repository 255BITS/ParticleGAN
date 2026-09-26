"""Baseline dynamics until every target mode is occupied, then unrolled k=5.

Acquisition condition, fixed before the run. It is not a step index, and it
is not a term in the losses.

The detector is the ring gate's live mode/hq detector,
``benchmarks.locked_shared.mode_hold.diversity``: a target mode is occupied
when at least one generated sample lies inside its 3σ ball, and ``modes`` is
how many target modes are occupied. ``n_modes`` is that detector's target
count (8 on the ring). The latch fires when ``modes == n_modes``. The hq
fraction is not a second threshold. A mode with no sample inside its ball
does not count, which is already how ``modes`` is defined.

The check runs only on the probe's normal checkpoint cadence: completed
update ``step % 50 == 0``. On a 1200-step ring that is the host recorder's
whole 24-point grid (50, 100, ..., 1200). Hold's ``diagnostic_every`` is 50.
Shift's every-10 grid includes those same steps. The hold probe's every-step
window after update 1200 is the dense convergence diagnostic, not this
cadence, so a step in that window counts only when it is a multiple of 50.
The first such checkpoint that shows every target mode occupied latches
unrolled k=5 for every later generator step. The update that produced the
checkpoint stays on baseline dynamics. Learning rates are not changed.

With the flag unset this module is not installed. ``critic_for_generator``
then never consults the latch.
"""
from __future__ import annotations

import atexit
import json
import os

# Frozen probe spacing. Not a chosen switch time: the latch step is whichever
# checkpoint on this grid first reports every target mode occupied.
CADENCE = 50

receipt = {
    "mechanism": "unrolled_after_acquire",
    "k": 5,
    "cadence": CADENCE,
    "condition": "modes == n_modes at the live mode/hq detector",
    "switch_step": None,
    "switch_hq": None,
    "generator_steps": 0,
}
_HOOKED = False
_RECORD_WRAPPED = False


def latched() -> bool:
    return receipt["switch_step"] is not None


def install() -> None:
    """Log the condition and read the probe checkpoints the host already takes."""
    global _HOOKED, _RECORD_WRAPPED
    if _HOOKED:
        return
    _HOOKED = True
    atexit.register(_emit)
    if not _RECORD_WRAPPED:
        from benchmarks.locked_shared.observation import Recorder

        original = Recorder.record

        def record(self, step, measure, original=original):
            original(self, step, measure)
            if os.environ.get("K3P_DYNAMICS") != "unrolled_after_acquire":
                return
            if self.curve and self.curve[-1].get("step") == step:
                note_checkpoint(step, self.curve[-1])

        Recorder.record = record
        _RECORD_WRAPPED = True
    print(json.dumps({
        "event": "dynamics",
        "name": "unrolled_after_acquire",
        "k": 5,
        "cadence": CADENCE,
        "condition": receipt["condition"],
        "setting": "baseline until modes == n_modes on the 50-step probe cadence, then unrolled k=5",
    }), flush=True)


def note_checkpoint(step, measured) -> None:
    """Latch once if this probe checkpoint shows every target mode occupied."""
    if os.environ.get("K3P_DYNAMICS") != "unrolled_after_acquire":
        return
    if receipt["switch_step"] is not None:
        return
    if type(step) is not int or step <= 0 or step % CADENCE != 0:
        return
    if not isinstance(measured, dict):
        return
    modes = measured.get("modes")
    n_modes = measured.get("n_modes")
    if isinstance(modes, bool) or isinstance(n_modes, bool):
        return
    if not isinstance(modes, int) or not isinstance(n_modes, int) or n_modes < 1:
        return
    if modes != n_modes:
        return
    hq = measured.get("hq")
    receipt["switch_step"] = step
    receipt["switch_hq"] = hq
    print(json.dumps({
        "event": "dynamics_switch",
        "name": "unrolled_after_acquire",
        "step": step,
        "modes": modes,
        "n_modes": n_modes,
        "hq": hq,
        "k": 5,
    }), flush=True)


def _emit() -> None:
    from particlegan.dynamics import unrolled

    receipt["generator_steps"] = unrolled.receipt.get("generator_steps", 0)
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)
