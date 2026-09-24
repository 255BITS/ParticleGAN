"""#141 delayed G clamp, with #107's stall test scored on the .25-cap factor.

After the budget latches (mode-hold update index >= 1200, sticky) the G step
still applies own-curvature bound .125. The stall predicate does not read that
shortened factor. It reads the factor the same rho would have produced under
the pre-budget cap .25:

    applied = min(1, .125 / rho)
    scored  = min(1, .25 / rho)

Pre-budget updates do not write a stall score, so the predicate keeps using
the actual factor and the bound stays .25. Width .5, D, Adam, and the losses
are unchanged. Trajectory never reaches the budget.
"""

from contextlib import contextmanager

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_g_bound import DelayedBudgetReachRecorder
from reports.toy100.pr84_reach_candidate import (
    B_CAP_SLOPE, REACH, SATURATED_UTILISATION, STALL_TRUST, STALL_WINDOW,
)


METHOD = "stall_reach_delayed_g125_scored_on_counterfactual_g25"


def trust_factor(rho, cap):
    """Same map BothBoundRecorder uses: ``min(1, cap / rho)``."""
    return min(1., cap / rho) if rho > 0 else 1.


class CounterfactualStallRecorder(DelayedBudgetReachRecorder):
    """Apply the latched .125 ball. Score stall on the .25-cap factor."""

    def _scored_trust(self, row):
        scored = row.get("stall_score")
        if isinstance(scored, dict) and "factor" in scored:
            return scored["factor"]
        return row["g"]["factor"]

    def _stalled(self, sharpness):
        recent = self.records[-STALL_WINDOW:]
        if sharpness / B_CAP_SLOPE < SATURATED_UTILISATION or not recent:
            return False
        return sum(self._scored_trust(row) for row in recent) / len(recent) <= STALL_TRUST

    def _attach_stall_score(self, before, after):
        """Counterfactual only. Does not rewrite the applied ``g`` factor."""
        g = self.row.get("g") if isinstance(getattr(self, "row", None), dict) else None
        if not isinstance(g, dict) or "rho" not in g:
            return
        self.row["stall_score"] = dict(
            factor=trust_factor(g["rho"], before),
            cap=float(before),
            applied_factor=g["factor"],
            applied_cap=float(after),
        )

    def _log_apply(self, before, after):
        if self.armed:
            self._attach_stall_score(before, after)
        super()._log_apply(before, after)

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     stall_score="counterfactual_pre_budget_cap_after_latch",
                     stall_counterfactual_cap=base.G_CURVATURE_BOUND,
                     stall_trust=STALL_TRUST)
        return value


@contextmanager
def stall_counterfactual_g25(*, task="mode_hold", start_step=0):
    """#141 clamp plus counterfactual stall scoring. One mechanism."""
    original = base.SmoothedBothBoundRecorder
    index_offset = 1 if task == "mode_hold" else 0

    def init(self, *, start_step=0):
        CounterfactualStallRecorder.__init__(self, start_step=start_step)
        self.index_offset = index_offset

    base.SmoothedBothBoundRecorder = type(
        "CounterfactualStallRecorder", (CounterfactualStallRecorder,),
        dict(__init__=init, reach=REACH, ramp="stall", game_bound=False, game_steps=1))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "CounterfactualStallRecorder", "stall_counterfactual_g25", "trust_factor"]
