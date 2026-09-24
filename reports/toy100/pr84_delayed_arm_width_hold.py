"""Delayed-arm G Adam lr ×0.5, plus one mechanism: hold critic width at .15.

#140's sticky arm is unchanged. The first diagnostic with update_index ≥ 1200,
modes ≥ 8 and HQ ≥ .9 sticks, and every later G Adam step uses 0.5 × the
pre-arm rates. Pre-arm stall reach is unchanged, including the widen to .5
when D's slope is ≥ .6 and G's own-trust average is ≤ .1.

After that arm, G's five-point critic width is .15 on every update. That
replaces both the stall override of .5 and the peak reach width that opened
to ~.45 at the step-2190 dip (slope ~.89, trust ~.21, so the stall predicate
was off). The hold sticks for the rest of training. D's rate, the curvature
bound and the losses are not changed.

Every post-arm width decision logs ``armed``, ``update_index``,
``width_before``, ``width_after`` and the fire count.
"""

from contextlib import contextmanager
import json

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_arm_g_lr import (
    METHOD as PARENT_METHOD, DelayedArmRecorder, diagnostic_log,
)
from reports.toy100.pr84_reach_candidate import METHOD as REACH_METHOD


METHOD = "stall_reach_delayed_arm_width_hold_015"
HELD_WIDTH = base.SMOOTH_WIDTH_CAP


class WidthHoldRecorder(DelayedArmRecorder):
    held_width = HELD_WIDTH

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.width_fires = 0
        self._last_width = None

    def _arm_smoothed_critic(self):
        super()._arm_smoothed_critic()
        if not (self.armed and self.enabled and self._smooth_on):
            return
        before = float(self._smooth_width)
        after = float(self.held_width)
        self._smooth_width = after
        self.row["critic_width"] = after
        self.width_fires += 1
        row = dict(event="width_hold_apply", armed=True,
                   update_index=self.row.get("outer_step"),
                   width_before=before, width_after=after,
                   fires=self.width_fires, phase=self.phase)
        self._last_width = row
        _log(**row)

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, parent_method=PARENT_METHOD,
                     reach_method=REACH_METHOD,
                     mechanism="delayed_arm_hold_critic_width_015",
                     held_width=self.held_width,
                     width_hold_fires=self.width_fires,
                     cpu_capability=torch.backends.cpu.get_cpu_capability())
        return value


def _log(**row):
    print(json.dumps(row), flush=True)


@contextmanager
def pr84_delayed_arm_width_hold(*, task="mode_hold", start_step=0):
    """#140 delayed-arm G lr, plus a sticky post-arm critic width of .15."""
    from benchmarks.locked_shared.observation import Recorder
    import benchmarks.toy100.continuous_probe as probe

    holder = {}
    original_record = Recorder.record
    original_run = probe._run_extended

    def record(self, step, measure):
        original_record(self, step, measure)
        target = holder.get("recorder")
        point = self.curve[-1] if self.curve else None
        if target is not None and point is not None and point.get("step") == step:
            target.consider_diagnostic(step, point.get("modes"), point.get("hq"))

    def run_extended(*args, **kwargs):
        target = holder.get("recorder")
        kwargs["log"] = diagnostic_log(target, kwargs.get("log")) if target is not None else kwargs.get("log")
        return original_run(*args, **kwargs)

    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        WidthHoldRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = base.G_CURVATURE_BOUND

    base.SmoothedBothBoundRecorder = type(
        "WidthHoldRecorder", (WidthHoldRecorder,),
        dict(ramp="stall", __init__=init))
    Recorder.record = record
    probe._run_extended = run_extended
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            holder["recorder"] = value[0]
            yield value
    finally:
        probe._run_extended = original_run
        Recorder.record = original_record
        base.SmoothedBothBoundRecorder = original


__all__ = ["HELD_WIDTH", "METHOD", "WidthHoldRecorder", "pr84_delayed_arm_width_hold"]
