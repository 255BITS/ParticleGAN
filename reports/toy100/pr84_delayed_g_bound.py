"""Stall reach (#107) with one delayed G own-curvature clamp.

The pre-budget cap stays at PR #82 / #107's .25. On the mode-hold host the
update index is the checkpoint number of the update in progress (loop step + 1).
The clamp latches when that index is >= 1200 and then holds G's own-curvature
trust bound at .125 for every later G step. There is no mode-count or HQ arm,
no Adam rescaling, and no change to D, the width rule, or the losses.

Trajectory's loop step is already 1-based and ends at 400, so the same predicate
never fires there and the cold trajectory is stall reach unchanged.
"""

from contextlib import contextmanager
import json

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


METHOD = "stall_reach_delayed_budget_g_own_curvature_0.125"
BUDGET_UPDATE = 1200
POST_BUDGET_G_BOUND = .125


class DelayedBudgetReachRecorder(ReachRecorder):
    budget_update = BUDGET_UPDATE
    post_budget_g_bound = POST_BUDGET_G_BOUND
    # mode_hold counts finished updates in the loop variable.
    index_offset = 1

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.curvature_bound = base.G_CURVATURE_BOUND
        self.armed = False
        self.fires = 0
        self.applies = 0
        self.update_index = None

    def note_update(self, host_step):
        """Latch the budget. Never clears ``armed``."""
        self.update_index = int(host_step) + self.index_offset
        if self.update_index >= self.budget_update:
            self.armed = True

    def phases(self, step, opt_d, opt_g, local):
        self.note_update(step)
        yield from super().phases(step, opt_d, opt_g, local)

    def _g_bound_is_applied(self, optimizer):
        if self.passthrough or self.game_bound or self.phase != 2:
            return False
        return self.optimizers is not None and optimizer is self.optimizers[1]

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        applying = self._g_bound_is_applied(optimizer)
        before = self.curvature_bound
        after = self.post_budget_g_bound if applying and self.armed else before
        if applying and self.armed:
            self.curvature_bound = after
        try:
            result = super().step(optimizer, ordinary_step, closure)
        finally:
            self.curvature_bound = before
        if applying:
            self._log_apply(before, after)
        return result

    def _log_apply(self, before, after):
        self.applies += 1
        if self.armed:
            self.fires += 1
        row = dict(armed=bool(self.armed), g_bound_before=before, g_bound_after=after,
                   update_index=self.update_index, fires=self.fires)
        if isinstance(getattr(self, "row", None), dict):
            self.row["delayed_g_bound"] = row
        payload = dict(event="G_BOUND", cpu=torch.backends.cpu.get_cpu_capability(), **row)
        print(json.dumps(payload), flush=True)

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     delayed_budget_update=self.budget_update,
                     pre_budget_g_bound=base.G_CURVATURE_BOUND,
                     post_budget_g_bound=self.post_budget_g_bound,
                     arm_predicate=f"update_index>={self.budget_update}",
                     armed=self.armed, fires=self.fires, g_bound_applies=self.applies,
                     cpu=torch.backends.cpu.get_cpu_capability())
        return value


@contextmanager
def delayed_budget_g_bound(*, task="mode_hold", start_step=0):
    """Stall-reach recorder with the delayed G bound. One mechanism."""
    original = base.SmoothedBothBoundRecorder
    index_offset = 1 if task == "mode_hold" else 0

    def init(self, *, start_step=0):
        DelayedBudgetReachRecorder.__init__(self, start_step=start_step)
        self.index_offset = index_offset

    base.SmoothedBothBoundRecorder = type(
        "DelayedBudgetReachRecorder", (DelayedBudgetReachRecorder,),
        dict(__init__=init, reach=REACH, ramp="stall", game_bound=False, game_steps=1))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "BUDGET_UPDATE", "POST_BUDGET_G_BOUND",
           "DelayedBudgetReachRecorder", "delayed_budget_g_bound"]
