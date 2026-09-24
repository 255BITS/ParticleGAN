"""Stall reach (#107) plus one post-arm common-mode null on G's output gradient.

Stall reach, both curvature bounds, and the losses stay as they are until an
already-scheduled ring check reports 8 modes and HQ >= 0.9. That log only arms
the mechanism. It is not a loss, and mode count is not a training target.

After the arm, every G backward replaces ``dL/dy`` with ``dL/dy - mean(dL/dy)``
over the batch before that gradient enters G's weights. ``y = G(z)`` itself is
not centered, the mean is not restored after Adam, Adam's betas are untouched,
and no step is skipped. D still trains on the raw fakes.
"""

from contextlib import contextmanager
import json
import math
import os

import torch

from benchmarks.locked_shared.observation import set_ring_listener
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


METHOD = "stall_reach_post_arm_g_common_mode_grad_null"
# Log threshold only. The null itself is always a full subtract, never a scale.
NULL_EPS = 1e-6
ARM_MODES = 8
ARM_HQ = 0.9
ACQUIRE_BUDGET = 1200
DROPOUT_LO = 1720
DROPOUT_HI = 2150


def subtract_common_mode(grad):
    """Return ``grad - mean_batch(grad)`` and the subtracted batch-mean vector.

    ``grad`` matches ``y`` and is ``(batch, features)``. The mean is over the
    batch and broadcasts. A pure translation (constant rows) becomes zero.
    """
    if not torch.is_tensor(grad) or grad.ndim != 2 or grad.shape[0] < 1 or grad.shape[1] < 1:
        raise RuntimeError("common-mode null expects a batched generator output gradient")
    if not torch.isfinite(grad).all():
        raise RuntimeError("generator output gradient is not finite")
    mean = grad.mean(dim=0)
    return grad - mean, mean


class CommonModeGradNullRecorder(ReachRecorder):
    """#107 stall reach, then drop only the common mode of G's output gradient."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update = None
        self.arm_phase = None
        self._hook = None
        self._hooked = None
        self._phase_l2 = []
        self.null_rows = []

    def note_ring(self, step, modes, hq):
        """Arm once on an already-logged ring check. Not a loss and not a predicate."""
        if self.armed:
            return
        try:
            modes, hq, step = int(modes), float(hq), int(step)
        except (TypeError, ValueError):
            return
        if modes != ARM_MODES or not math.isfinite(hq) or hq < ARM_HQ:
            return
        self.armed = True
        self.arm_update = step
        if self.start_step > 0:
            self.arm_phase = "warm"
        elif step <= ACQUIRE_BUDGET:
            self.arm_phase = "cold_acquire"
        else:
            self.arm_phase = "stay"
        print(json.dumps(dict(
            event="CM_NULL_ARM", step=step, phase=self.arm_phase,
            cpu=torch.backends.cpu.get_cpu_capability(),
            aten=os.environ.get("ATEN_CPU_CAPABILITY"),
        )), flush=True)

    def _null_active(self):
        return self.armed and self.enabled and not self.passthrough and self.phase is not None

    def _sync_hook(self, generator, step):
        """Install the forward hook only on updates that will actually null."""
        want = (self.armed and self.enabled and generator is not None
                and not (isinstance(step, int) and step < self.start_step))
        if want and self._hook is None:
            self._hook = generator.register_forward_hook(self._on_forward)
            self._hooked = generator
        elif not want and self._hook is not None:
            self._hook.remove()
            self._hook = None
            self._hooked = None

    def _on_forward(self, module, inputs, output):
        # Return None so y is the tensor D and the loss already see.
        if not self._null_active() or not torch.is_tensor(output) or not output.requires_grad:
            return None
        phase = self.phase

        def _hook(grad, phase=phase):
            nulled, mean = subtract_common_mode(grad)
            l2 = float(mean.detach().norm())
            if not math.isfinite(l2):
                raise RuntimeError("common-mode gradient norm is not finite")
            if phase == self.phase:
                self._phase_l2.append(l2)
            return nulled

        output.register_hook(_hook)
        return None

    def _commit_null(self, phase):
        if len(self._phase_l2) != 1:
            raise RuntimeError(
                f"armed G backward produced {len(self._phase_l2)} output gradients, expected 1")
        l2 = self._phase_l2[0]
        if phase != 1:
            return
        step = self.row.get("outer_step")
        self.row["cm_null_l2"] = l2
        self.null_rows.append(dict(step=step, l2=l2))
        print(json.dumps(dict(
            event="CM_NULL", step=step, l2=round(l2, 6), nontrivial=l2 > NULL_EPS,
        )), flush=True)

    def phases(self, step, opt_d, opt_g, local):
        generator = local.get("generator") if isinstance(local, dict) else None
        self._sync_hook(generator, step)
        for phase in super().phases(step, opt_d, opt_g, local):
            self._phase_l2 = []
            yield phase

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.game_bound:
            raise RuntimeError("common-mode grad null stays on stall reach, without the game bound")
        result = super().step(optimizer, ordinary_step, closure)
        if (self._null_active() and self.optimizers is not None
                and optimizer is self.optimizers[1]):
            self._commit_null(self.phase)
        return result

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
            self._hooked = None

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     cm_null_armed=self.armed, cm_null_arm_update=self.arm_update,
                     cm_null_arm_phase=self.arm_phase, cm_null_eps=NULL_EPS,
                     cpu_capability=torch.backends.cpu.get_cpu_capability(),
                     aten_cpu_capability=os.environ.get("ATEN_CPU_CAPABILITY"),
                     **_window_stats(self.null_rows))
        return value


def _median(values):
    if not values:
        return 0.
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _window_stats(rows):
    def pack(subset, prefix):
        norms = [row["l2"] for row in subset]
        return {
            f"{prefix}_n": len(norms),
            f"{prefix}_nontrivial": sum(norm > NULL_EPS for norm in norms),
            f"{prefix}_median_l2": _median(norms),
            f"{prefix}_max_l2": max(norms, default=0.),
        }

    acquire = [row for row in rows if row["step"] is not None and row["step"] <= ACQUIRE_BUDGET]
    stay = [row for row in rows if row["step"] is not None and row["step"] > ACQUIRE_BUDGET]
    dropout = [row for row in rows
               if row["step"] is not None and DROPOUT_LO <= row["step"] <= DROPOUT_HI]
    stats = pack(rows, "cm_null")
    stats.update(pack(acquire, "cm_null_acquire"))
    stats.update(pack(stay, "cm_null_stay"))
    stats.update(pack(dropout, "cm_null_dropout"))
    return stats


@contextmanager
def pr84_common_mode_grad_null(*, task="mode_hold", start_step=0):
    """Stall reach (ramp ``stall``, reach .5) with the post-arm output-gradient null."""
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        CommonModeGradNullRecorder.__init__(self, start_step=start_step)

    base.SmoothedBothBoundRecorder = type(
        "CommonModeGradNullRecorder", (CommonModeGradNullRecorder,),
        dict(reach=REACH, ramp="stall", game_bound=False, game_steps=1, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            previous = set_ring_listener(value[0].note_ring)
            try:
                yield value
            finally:
                set_ring_listener(previous)
                value[0].close()
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "CommonModeGradNullRecorder", "pr84_common_mode_grad_null", "subtract_common_mode"]
