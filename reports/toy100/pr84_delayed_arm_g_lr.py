"""Stall reach with one added mechanism: delayed-arm G Adam lr ×0.5.

Stall reach (width, D, losses, G own-curvature bound .25) is unchanged.
On the first diagnostic whose update index is at least 1200 and which
already holds the full ring (modes ≥ 8 and HQ ≥ .9), the arm sticks.
Every later G Adam step uses 0.5 × the pre-arm group rates. D is not
scaled, the curvature bound is not changed, and the post-bound placement
is not scaled again.

The half rate is installed on G's param groups only for the Adam call
(and the own-curvature metric that reads that same lr). It is restored
when the call returns so the host schedule can write its pre-arm rate
on the next update without compounding. Applies are logged as JSON.
"""

from contextlib import contextmanager
import json
import math

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import METHOD as REACH_METHOD
from reports.toy100.pr84_reach_candidate import ReachRecorder


METHOD = "stall_reach_delayed_arm_g_adam_lr_half"
ARM_UPDATE_INDEX = 1200
ARM_MODES = 8
ARM_HQ = .9
G_LR_SCALE = .5


class DelayedArmRecorder(ReachRecorder):
    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update_index = None
        self.arm_g_lr = None
        self._pre_arm = None
        self.fires = 0
        self._host_step = None
        self._last_apply = None

    def phases(self, step, opt_d, opt_g, local):
        self._host_step = step
        yield from super().phases(step, opt_d, opt_g, local)

    def consider_diagnostic(self, update_index, modes, hq):
        """Arm once, only at update_index ≥ 1200 with modes ≥ 8 and HQ ≥ .9."""
        if self.armed or not self.enabled:
            return
        if type(update_index) is not int or update_index < ARM_UPDATE_INDEX:
            return
        if type(modes) is not int or modes < ARM_MODES:
            return
        if (isinstance(hq, bool) or not isinstance(hq, (int, float))
                or not math.isfinite(hq) or hq < ARM_HQ):
            return
        if self.optimizers is None:
            return
        groups = self.optimizers[1].param_groups
        rates = [float(group["lr"]) for group in groups]
        if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            return
        self._pre_arm = rates
        self.armed = True
        self.arm_update_index = update_index
        self.arm_g_lr = rates[_generator_index(groups)]
        _log(event="g_lr_arm", armed=True, update_index=update_index,
             g_lr_before=self.arm_g_lr, g_lr_after=self.arm_g_lr * G_LR_SCALE,
             fires=self.fires)

    def step(self, optimizer, ordinary_step, closure=None):
        saved = None

        def apply_then(opt, closure=None):
            nonlocal saved
            saved = self._install_half(opt)
            return ordinary_step(opt, closure)

        try:
            return super().step(optimizer, apply_then, closure)
        finally:
            if saved is not None:
                for group, rate in zip(optimizer.param_groups, saved):
                    group["lr"] = rate

    def _install_half(self, optimizer):
        if (not self.armed or not self.enabled or self.optimizers is None
                or optimizer is not self.optimizers[1]):
            return None
        groups = optimizer.param_groups
        if self._pre_arm is None or len(groups) != len(self._pre_arm):
            raise RuntimeError("G optimizer groups changed after the delayed arm")
        saved = [group["lr"] for group in groups]
        index = _generator_index(groups)
        before = float(saved[index])
        for group, base in zip(groups, self._pre_arm):
            group["lr"] = base * G_LR_SCALE
        after = float(groups[index]["lr"])
        self.fires += 1
        row = dict(event="g_lr_apply", armed=True,
                   update_index=None if self._host_step is None else self._host_step + 1,
                   g_lr_before=before, g_lr_after=after, fires=self.fires)
        prior = next((i for i, group in enumerate(groups) if group.get("_comparison_prior")), None)
        if prior is not None:
            row["prior_lr_before"] = float(saved[prior])
            row["prior_lr_after"] = float(groups[prior]["lr"])
        self._last_apply = row
        _log(**row)
        return saved

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, parent_method=REACH_METHOD,
                     mechanism="delayed_arm_g_adam_lr_half",
                     arm_update_index_min=ARM_UPDATE_INDEX,
                     arm_modes=ARM_MODES, arm_hq=ARM_HQ, g_lr_scale=G_LR_SCALE,
                     armed=self.armed, arm_update_index=self.arm_update_index,
                     arm_g_lr=self.arm_g_lr, g_lr_fires=self.fires,
                     cpu_capability=torch.backends.cpu.get_cpu_capability())
        return value


def _generator_index(groups):
    return next((i for i, group in enumerate(groups) if not group.get("_comparison_prior")), 0)


def _log(**row):
    print(json.dumps(row), flush=True)


def diagnostic_log(recorder, user_log=None):
    """Probe checkpoint logger that can arm this recorder and no other state."""
    def log(row):
        if user_log is not None:
            user_log(row)
        if isinstance(row, dict) and row.get("event") == "checkpoint":
            recorder.consider_diagnostic(row.get("step"), row.get("modes"), row.get("hq"))
    return log


@contextmanager
def pr84_delayed_arm_g_lr(*, task="mode_hold", start_step=0):
    """Stall reach, plus sticky G Adam lr ×0.5 after the delayed arm."""
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
        DelayedArmRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = base.G_CURVATURE_BOUND

    base.SmoothedBothBoundRecorder = type(
        "DelayedArmRecorder", (DelayedArmRecorder,),
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


__all__ = ["ARM_HQ", "ARM_MODES", "ARM_UPDATE_INDEX", "G_LR_SCALE", "METHOD",
           "DelayedArmRecorder", "diagnostic_log", "pr84_delayed_arm_g_lr"]
