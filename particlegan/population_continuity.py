"""RA11 population continuity, scoped to the actual feature backend."""
import math
import torch
from .continuous import SequentialSettleTest


class PopulationSequentialSettleTest(SequentialSettleTest):
    """SequentialSettleTest whose STATIONARY descent also needs population continuity.

    The statistic, scale search, early looks, window-end tests, row-lineage
    handling and actions are inherited from ``SequentialSettleTest``. This class
    adds the population law ``population_policy``:

    * a STATIONARY decision is accepted only when at least ``rows - floor(Q rows)``
      rows (``Q = population_q``) have two finite observations at the accepted
      scale and are not excluded; otherwise it becomes INCONCLUSIVE;
    * the participating rows of an accepted descent are recorded. Once birth-death
      replacements or a whole-group restart leave fewer than that many, the
      descent is undone (``_population_revoke``) without a DRIFT verdict.
    """

    def __init__(self, early=True, final_table=None, early_stationary_only=False):
        super().__init__(early=early, final_table=final_table, early_stationary_only=early_stationary_only)
        # Population continuity is separate from the gradient direction test.
        # A negative mean over a small surviving subset cannot close the whole
        # table. Q is the existing .05 population budget, not another test level.
        self.population_schema = 1
        self.population_policy = "two_pair_participation_Q_survival_one_descent_undo_v1"
        self.population_q = .05
        self.stationary_rows = None
        self.stationary_undo_s = None
        self.population_active = False
        self.last_population = {}
        self.counts["population_coverage_rejections"] = 0
        self.counts["population_expiries"] = 0

    @torch.no_grad()
    def begin(self, params):
        super().begin(params)
        if self.rows is not None and self.stationary_rows is None:
            self.stationary_rows = torch.zeros(self.rows, device=self.anchor.device, dtype=torch.bool)

    def _population_participants(self, sb, s2):
        """Rows with two finite observations at the accepted negative scale.

        This records participation in the population test, not separate
        per-row stationarity verdicts. Rebase has already removed old lineages.
        """
        if self.rows is None:
            return None
        pairs = self.r_b if sb["verdict"] == -1 else self.r_2b
        if not pairs:
            return torch.zeros_like(self.stationary_rows)
        participants = torch.isfinite(torch.stack(pairs)).sum(0) >= 2
        if self.exclude is not None:
            participants &= ~self.exclude.to(participants.device)
        return participants

    def _population_covered(self, participants):
        return participants is None or int(participants.sum()) >= self.rows - math.floor(self.population_q * self.rows)

    def _population_revoke(self, reason):
        """Undo only the accepted descent whose population continuity expired.

        A replaced population is not a positive-gradient DRIFT verdict. The
        current b/tau/window and untouched-row evidence remain intact.
        """
        if not self.population_active:
            return
        remaining = int(self.stationary_rows.sum())
        before = self.s
        self.s = max(self.s, float(self.stationary_undo_s))
        self.last_decisive = 0
        self.last_decisive_scale = None
        self.b_anchor = None
        self.population_active = False
        self.stationary_undo_s = None
        self.counts["population_expiries"] += 1
        self.last_population = dict(event="expired", reason=reason, participating_rows=remaining,
                                   required_rows=self.rows - math.floor(self.population_q * self.rows),
                                   s_before=before, s_after=self.s)

    @torch.no_grad()
    def rebase(self, params, rows):
        """Invalidate all copied or newly born rows in the population verdict."""
        if not len(rows):
            return
        super().rebase(params, rows)
        if self.population_active:
            self.stationary_rows[rows] = False
            if not self._population_covered(self.stationary_rows):
                self._population_revoke("row_replacement")

    def restart(self, params, reopen=False):
        super().restart(params, reopen=reopen)
        # A whole-group jump replaces every lineage. Existing completed pairs
        # can remain in the window as dropped observations, never as new rows'
        # evidence. Reopen already cleared them in the base method.
        if self.rows is not None:
            for pair in self.r_b + self.r_2b:
                pair.fill_(float("nan"))
        if self.stationary_rows is not None:
            self.stationary_rows.zero_()
        self._population_revoke("whole_group_restart")

    def load_state_dict(self, state, numel):
        """Validate the distinct population law before mutating the tester."""
        if not isinstance(state, dict) or set(state) != set(self.__dict__):
            raise ValueError("incompatible population-settle checkpoint schema")
        if (type(state["population_schema"]) is not int
                or state["population_schema"] != self.population_schema
                or state["population_policy"] != self.population_policy
                or state["population_q"] != self.population_q
                or type(state["population_active"]) is not bool):
            raise ValueError("incompatible population-settle checkpoint law")
        rows, mask = state["rows"], state["stationary_rows"]
        if rows is not None and (type(rows) is not int or rows <= 0 or numel % rows):
            raise ValueError("invalid population-settle row count")
        if self.rows is not None and rows != self.rows:
            raise ValueError("population-settle row count does not match the table")
        if mask is not None and (rows is None or not isinstance(mask, torch.Tensor)
                or mask.shape != (rows,) or mask.dtype != torch.bool):
            raise ValueError("invalid population-settle participation mask")
        active, undo = state["population_active"], state["stationary_undo_s"]
        if (active and (mask is None or rows is None or state["last_decisive"] != -1
                or int(mask.sum()) < rows - math.floor(self.population_q * rows))):
            raise ValueError("invalid active population-settle certificate")
        if rows is not None and (state["last_decisive"] == -1) != active:
            raise ValueError("population-settle stationary stamp is not active")
        if active:
            if (type(undo) not in (float, int) or not math.isfinite(undo)
                    or not 0 < undo <= 1 or undo != state["s"] / self.GAMMA):
                raise ValueError("invalid population-settle descent undo")
        elif undo is not None:
            raise ValueError("inactive population-settle checkpoint retains an undo")
        if (not isinstance(state["last_population"], dict) or not isinstance(state["counts"], dict) or any(
                type(state["counts"].get(key)) is not int or state["counts"][key] < 0
                for key in ("population_coverage_rejections", "population_expiries"))):
            raise ValueError("invalid population-settle diagnostics")
        super().load_state_dict(state, numel)
        if self.stationary_rows is not None:
            device = self.anchor.device if self.anchor is not None else self.stationary_rows.device
            self.stationary_rows = self.stationary_rows.clone().to(device)

    def _conclude(self, sb, s2, tested_b, step, early):
        """SequentialSettleTest._conclude plus the population coverage rule and descent record."""
        vb, v2, decision = self._table_decision(sb, s2, tested_b)
        participants = self._population_participants(sb, s2) if decision == "stationary" else None
        coverage_rejected = decision == "stationary" and not self._population_covered(participants)
        if coverage_rejected:
            decision = "inconclusive"
            self.counts["population_coverage_rejections"] += 1
        s_before = self.s
        self.counts[decision] += 1
        if early:
            self.counts["early"] += 1
        if decision in ("stationary", "drift"):
            if self._apply_decisive(decision, vb, v2, tested_b) < 0:
                if participants is not None:
                    self.stationary_rows = participants.clone()
                    self.stationary_undo_s = s_before
                    self.population_active = True
            else:
                if self.stationary_rows is not None:
                    self.stationary_rows.zero_()
                self.stationary_undo_s = None
                self.population_active = False
        elif decision in ("mixed", "inconclusive"):
            self.b *= 2
        self._record_decision(sb, s2, vb, v2, tested_b, step, early, decision)
        self.last_population = dict(event="decision", step=step,
            participating_rows=None if participants is None else int(participants.sum()),
            required_rows=None if self.rows is None else self.rows - math.floor(self.population_q * self.rows),
            coverage_rejected=bool(coverage_rejected), scheduling_decision=decision)
        self._close_window()
        return decision

    def diagnostics(self):
        out = super().diagnostics()
        out["population_schema"] = self.population_schema
        out["population_policy"] = self.population_policy
        out["population_q"] = self.population_q
        out["population_active"] = self.population_active
        out["stationary_participating_rows"] = (None if self.stationary_rows is None
                                               else int(self.stationary_rows.sum()))
        out["stationary_undo_s"] = self.stationary_undo_s
        out["last_population"] = dict(self.last_population)
        return out
