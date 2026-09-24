"""Stall reach (#107) plus one post-step G output-mean restore.

The G Adam step and the stall-reach curvature bound run on the usual
uncentered forward, the one D and the loss already saw. After that accepted
step, the final linear bias is shifted so the same training batch's clean
output mean matches its pre-step mean. The train forward is not centered and
the loss inputs are not rewritten.

This does not retune reach, combine with a falling-trust G scale, or add an
extra D step.
"""

from contextlib import contextmanager
import json
import math

import torch
from torch import nn

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


METHOD = "stall_reach_post_step_g_output_mean_restore"
# Dust below this is counted as not having moved the batch mean.
FIRE_EPS = 1e-6
MATCH_ATOL = 1e-4


def output_bias(module):
    """Last linear bias of the clean generator. The ring and trajectory hosts both end there."""
    found = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            found = layer.bias
    if found is None:
        raise RuntimeError("generator has no output bias to restore")
    return found


def batch_mean(output, bias):
    if output.ndim < 1 or output.shape[-1] != bias.numel():
        raise RuntimeError("generator output is not a batch with the output-bias width")
    return output.reshape(-1, bias.numel()).mean(0)


@torch.no_grad()
def restore_output_mean(module, bias, inputs, pre_mean):
    """Shift ``bias`` so ``module(*inputs)`` has mean ``pre_mean``. Returns the pre-correction delta."""
    post = batch_mean(module(*inputs), bias)
    delta = post - pre_mean
    bias.add_(-delta)
    fixed = batch_mean(module(*inputs), bias)
    if not torch.allclose(fixed, pre_mean, atol=MATCH_ATOL, rtol=0.):
        raise RuntimeError("output-bias restore did not put the batch mean back")
    return delta, fixed


class MeanRestoreRecorder(ReachRecorder):
    """#107 stall reach, then cancel the accepted step's clean output translation."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._hook = None
        self._clean = None
        self._bias = None
        self._last = None
        self._pre = None

    def _ensure_hook(self, generator):
        if generator is None or self._hook is not None:
            return
        clean = getattr(generator, "model", generator)
        self._clean = clean
        self._bias = output_bias(clean)
        self._hook = clean.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        # Observe only. Returning None leaves the tensor D and the loss see.
        if self.passthrough or self.phase != 1 or not torch.is_tensor(output):
            return None
        cloned = tuple(value.detach().clone() if torch.is_tensor(value) else value for value in inputs)
        self._last = (cloned, batch_mean(output.detach(), self._bias).clone(), int(output.reshape(-1, self._bias.numel()).shape[0]))
        return None

    def _capture_pre_step(self):
        if self._last is None:
            raise RuntimeError("G step had no generator forward to measure")
        inputs, mean, batch = self._last
        self._pre = (inputs, mean.clone(), batch)

    def _restore_output_mean(self):
        if self._pre is None or self._clean is None or self._bias is None:
            raise RuntimeError("post-step mean restore ran without a pre-step batch")
        inputs, pre_mean, batch = self._pre
        delta, fixed = restore_output_mean(self._clean, self._bias, inputs, pre_mean)
        norm = float(delta.norm())
        if not math.isfinite(norm):
            raise RuntimeError("output-mean restore delta is not finite")
        self.row["mean_restore_delta"] = norm
        self.row["mean_restore_pre"] = [round(float(v), 6) for v in pre_mean]
        self.row["mean_restore_post"] = [round(float(v), 6) for v in (pre_mean + delta)]
        self.row["mean_restore_fixed"] = [round(float(v), 6) for v in fixed]
        self.row["mean_restore_batch"] = batch
        self.row["mean_restore_residual"] = float((fixed - pre_mean).norm())
        print(json.dumps(dict(
            event="MEAN_RESTORE", step=self.row.get("outer_step"),
            delta=round(norm, 6), batch=batch,
            pre=self.row["mean_restore_pre"], post=self.row["mean_restore_post"],
        )), flush=True)
        self._pre = None

    def phases(self, step, opt_d, opt_g, local):
        self._ensure_hook(local.get("generator"))
        self._last = None
        yield from super().phases(step, opt_d, opt_g, local)

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.game_bound:
            raise RuntimeError("output-mean restore stays on stall reach, without the game bound")
        g_update = (not self.passthrough and self.phase in (1, 2) and self.optimizers is not None
                    and optimizer is self.optimizers[1])
        if g_update and self.phase == 1:
            # The host's G forward has already run, still at the pre-step weights.
            self._capture_pre_step()
        result = super().step(optimizer, ordinary_step, closure)
        if g_update and self.phase == 2:
            # Curvature bound has been applied. Cancel only the resulting translation.
            self._restore_output_mean()
        return result

    def receipt(self):
        value = super().receipt()
        rows = [row for row in self.records if "mean_restore_delta" in row]
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, **_window_stats(rows))
        return value


def _median(values):
    if not values:
        return 0.
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _window_stats(rows):
    def pack(subset, prefix):
        deltas = [row["mean_restore_delta"] for row in subset]
        return {
            f"{prefix}_n": len(deltas),
            f"{prefix}_fires": sum(delta > FIRE_EPS for delta in deltas),
            f"{prefix}_median": _median(deltas),
            f"{prefix}_max": max(deltas, default=0.),
        }

    acquire = [row for row in rows if row["outer_step"] <= 1200]
    dropout = [row for row in rows if 1690 <= row["outer_step"] <= 2300]
    stats = pack(rows, "mean_restore")
    stats.update(pack(acquire, "mean_restore_acquire"))
    stats.update(pack(dropout, "mean_restore_dropout"))
    stats["mean_restore_batch"] = rows[-1]["mean_restore_batch"] if rows else 0
    return stats


@contextmanager
def pr84_mean_restore_candidate(*, task="mode_hold", start_step=0):
    """Stall reach (ramp ``stall``, reach .5) with post-step output-mean restore."""
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        MeanRestoreRecorder.__init__(self, start_step=start_step)

    base.SmoothedBothBoundRecorder = type(
        "MeanRestoreRecorder", (MeanRestoreRecorder,),
        dict(reach=REACH, ramp="stall", game_bound=False, game_steps=1, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "MeanRestoreRecorder", "pr84_mean_restore_candidate", "restore_output_mean"]
