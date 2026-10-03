"""CPU review prototype: expire a population verdict when its rows are replaced.

This is a proposed scheduling rule, not a new gradient-drift test. It uses the
existing Q as the maximum fraction of rows outside an active population verdict.
The original sequential tests, their error levels, and the averaging/noise rules
remain in the frozen module imported by the caller.
"""
from copy import deepcopy
import math

import torch


def population_test_class(base_class):
    class PopulationSettleTest(base_class):
        def __init__(self, table, q=.05, **kwargs):
            super().__init__(**kwargs)
            self.rows = len(table)
            self.population_q = float(q)
            self.population_law = "finite_two_pair_participation_Q_survival_one_descent_undo_v1"
            self.stationary_rows = torch.zeros(self.rows, device=table.device, dtype=torch.bool)
            self.stationary_undo_s = None
            self.population_active = False
            self.population_expiries = 0
            self.population_coverage_rejections = 0
            self.last_population = {}

        def _participants(self, sb, s2):
            # Use the same negative scale as the original accepted decision.
            pairs = self.r_b if sb["verdict"] == -1 else self.r_2b
            if not pairs:
                return torch.zeros_like(self.stationary_rows)
            mask = torch.isfinite(torch.stack(pairs)).sum(0) >= 2
            if self.exclude is not None:
                mask &= ~self.exclude.to(mask.device)
            return mask

        def _population_ok(self, mask):
            return int(mask.sum()) >= self.rows - math.floor(self.population_q * self.rows)

        def _conclude(self, sb, s2, tested_b, step, early):
            vb, v2 = sb["verdict"], s2["verdict"]
            stationary = vb is not None and vb * (0 if v2 is None else v2) != -1 \
                and (vb == -1 or v2 == -1)
            mask = self._participants(sb, s2) if stationary else None
            before = self.s
            rejected = stationary and not self._population_ok(mask)
            if rejected:
                # Insufficient population coverage is inconclusive; keep the
                # original longer-scale search instead of halving the rate.
                self.population_coverage_rejections += 1
                sb, s2 = dict(sb, verdict=0), dict(s2, verdict=0)
            decision = super()._conclude(sb, s2, tested_b, step, early)
            if decision == "stationary":
                self.stationary_rows = mask.clone()
                self.stationary_undo_s = before
                self.population_active = True
            elif decision == "drift":
                self.stationary_rows.zero_()
                self.stationary_undo_s = None
                self.population_active = False
            self.last_population = dict(step=step,
                participating_rows=None if mask is None else int(mask.sum()),
                required_rows=self.rows - math.floor(self.population_q * self.rows),
                coverage_rejected=bool(rejected), scheduling_decision=decision)
            return decision

        @torch.no_grad()
        def rebase(self, params, rows):
            if not len(rows):
                return
            super().rebase(params, rows)
            if not self.population_active:
                return
            self.stationary_rows[rows] = False
            if self._population_ok(self.stationary_rows):
                return
            # The population supporting that descent changed. Undo only that
            # descent once; preserve the live window, b, tau and untouched-row
            # evidence. This does not declare positive gradient drift.
            self.s = max(self.s, float(self.stationary_undo_s))
            self.last_decisive = 0
            self.last_decisive_scale = None
            self.b_anchor = None
            self.population_active = False
            self.stationary_undo_s = None
            self.population_expiries += 1

        def restart(self, params, reopen=False):
            super().restart(params, reopen=reopen)
            if reopen:
                self.stationary_rows.zero_()
                self.stationary_undo_s = None
                self.population_active = False

        def load_state_dict(self, state, numel):
            # A new state law must reject old checkpoints, never infer historical
            # participation. This is validation before any object mutation.
            if not isinstance(state, dict) or set(state) != set(self.__dict__):
                raise ValueError("incompatible population-settle checkpoint law")
            mask = state["stationary_rows"]
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool \
                    or mask.shape != (self.rows,) or state["population_q"] != self.population_q \
                    or state["population_law"] != self.population_law:
                raise ValueError("incompatible population-settle coverage state")
            before = deepcopy(self.__dict__)
            try:
                super().load_state_dict(state, numel)
            except Exception:
                self.__dict__.clear()
                self.__dict__.update(before)
                raise
            self.stationary_rows = self.stationary_rows.clone().to(self.anchor.device
                if self.anchor is not None else mask.device)

    return PopulationSettleTest
