"""RA11 population continuity, scoped to the actual feature backend."""
from copy import deepcopy
import math
import torch
from .continuous import SettleTest, _T99375, _LOG_EARLY, _log_t_bayes_factor


class PopulationSequentialSettleTest(SettleTest):
    """SettleTest that may decide after any completed pair instead of only at the window end.

    Same statistic (row-averaged cosine of consecutive block displacements at scales b and 2b),
    same scale search, same actions and the same nominal error per decision (.05) as SettleTest.
    What changes is *when* a decision may be taken:

    * after every completed pair the evidence collected so far in the window is tested with an
      anytime-valid e-process per scale (mixture t-test Bayes factor >= 1/.0125); a crossing
      decides immediately, so strong evidence no longer waits for the 2K-block window to close;
    * if nothing has crossed when the window fills (2K blocks) the fixed-n one-sided t tests are
      applied at alpha .00625 per direction per scale (the remaining half of the .05 budget), and
      the usual STATIONARY / DRIFT / MIXED / INCONCLUSIVE actions follow.

    ``early=False`` with ``final_table=_T9875`` reproduces SettleTest's decisions exactly.
    Row-lineage handling (birth-death rebase) is inherited: the moved rows are removed from every
    pair of the running window, and the scale means are recomputed from the row vectors at each look.
    """

    def __init__(self, early=True, final_table=None, early_stationary_only=False):
        super().__init__()
        self.early = bool(early)
        self.early_stationary_only = bool(early_stationary_only)
        self.final_table = tuple(_T99375 if final_table is None else final_table)
        self.counts["early"] = 0
        self.counts["held"] = 0
        self.looks = 0
        self.last_look = {}
        # Off-support gate (set by GANTrainer each step from the birth-death support test):
        # ``exclude``: rows that do not vote (off-support rows); ``hold_descent``: a STATIONARY verdict
        # is deferred (early: evidence kept; window end: window discarded) while too much of the table
        # is off-support.
        self.exclude = None
        self.hold_descent = False
        # Table release rule (a DRIFT verdict below the ceiling raises s): "any" = the SettleTest table rule (either scale
        # positive and not opposed); "both" = both scales significantly positive; "never" = no release from this tester.
        self.release_rule = "any"
        # ANCHOR: evidence scale (in intrinsic time) of the last accepted STATIONARY decision = the scale at which the bulk
        # mean-reverts. Under a stationary bulk the block cosine is positive below it and negative near it, so a DRIFT verdict
        # whose evidence scale is below twice this anchor is what a settled table produces and is not evidence of drift.
        self.b_anchor = None
        self.counts["drift_blocked"] = 0
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
        if reopen:
            self.b_anchor = None

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

    @torch.no_grad()
    def _scale_values(self):
        """Per-pair scalars (mean over the rows that have a direction) for the b and 2b scales;
        one host sync."""
        nb = len(self.r_b)
        if not (self.r_b or self.r_2b):
            return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
        values = torch.stack(self.r_b + self.r_2b)
        if self.rows is not None:
            if self.exclude is not None:
                values = values.masked_fill(self.exclude.to(values.device)[None, :], float("nan"))
            values = torch.nanmean(values, dim=1)
        values = values.cpu().double()
        return values[:nb], values[nb:]

    @staticmethod
    def _early_stat(values):
        """Anytime-valid look: dict(verdict, n, mean, t, log_bf); verdict 0 until the e-process crosses."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        out = dict(verdict=0, n=n, mean=float(r.mean()) if n else float("nan"), t=float("nan"),
                   log_bf=float("-inf"))
        if n < 2:
            return out
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return out          # degenerate spread: left to the window-end rule
        t = out["mean"] / (sd / math.sqrt(n))
        out["t"] = t
        out["log_bf"] = _log_t_bayes_factor(t, n)
        if out["log_bf"] >= _LOG_EARLY:
            out["verdict"] = 1 if t > 0 else -1
        return out

    def _final_verdict(self, values):
        """Window-end fixed-n one-sided t tests (table quantiles of ``final_table``)."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        if n == 0:
            return dict(verdict=None, n=0, mean=float("nan"), t=float("nan"), log_bf=float("nan"))
        mean = float(r.mean())
        if n == 1:
            return dict(verdict=0, n=1, mean=mean, t=float("nan"), log_bf=float("nan"))
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return dict(verdict=(0 if mean == 0. else (1 if mean > 0 else -1)), n=n, mean=mean,
                        t=math.copysign(float("inf"), mean), log_bf=float("nan"))
        t = mean / (sd / math.sqrt(n))
        crit = self.final_table[min(n - 1, len(self.final_table)) - 1]
        return dict(verdict=(1 if t > crit else -1 if t < -crit else 0), n=n, mean=mean, t=t,
                    log_bf=float("nan"))

    def _conclude(self, sb, s2, tested_b, step, early):
        """Map the two scale verdicts to a decision (table rule of SettleTest) and act on it."""
        vb, v2 = sb["verdict"], s2["verdict"]
        v2 = 0 if v2 is None else v2
        if vb is None:
            decision = "frozen"
        elif vb * v2 == -1:
            decision = "mixed"
        elif vb == -1 or v2 == -1:
            decision = "stationary"
        elif vb == 1 or v2 == 1:
            decision = "drift"
            if self.s < 1.0:
                drift_scale = 2 * tested_b if vb != 1 and v2 == 1 else tested_b
                if self.release_rule == "never" or (self.release_rule == "both" and not (vb == 1 and v2 == 1)):
                    decision = "inconclusive"   # a release needs the stated evidence; otherwise search a longer scale
                elif (self.release_rule == "anchor" and self.b_anchor is not None
                      and drift_scale < 2 * self.b_anchor):
                    decision = "inconclusive"
                    self.counts["drift_blocked"] += 1
        else:
            decision = "inconclusive"
        if decision == "stationary" and self.hold_descent:
            decision = "held"           # off-support mass above the FDR budget: descent is deferred
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
            sign = -1 if decision == "stationary" else 1
            evidence_scale = 2 * tested_b if vb != sign and v2 == sign else tested_b
            at_ceiling = self.s == 1.0
            self.s = self.s * self.GAMMA if sign < 0 else min(1.0, self.s / self.GAMMA)
            if decision == "drift" and at_ceiling and self.rows is not None:
                self.b *= 4 if v2 == 1 else 2
            elif self.last_decisive == -sign and self.last_decisive_scale == evidence_scale:
                self.counts["reversal"] += 1
                self.b = 2 * evidence_scale
            elif vb == 0 and v2 != 0:
                self.b = max(self.B0, self.b)
            else:
                self.b = max(self.B0, self.b / 2)
            self.last_decisive = sign
            self.last_decisive_scale = evidence_scale
            if sign < 0:
                self.b_anchor = evidence_scale
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
        self.last = dict(decision=decision, step=step, tested_b=tested_b, verdict_b=vb, verdict_2b=v2,
                         evidence_scale=(2 * tested_b if vb == 0 and v2 != 0 else tested_b),
                         t_b=sb["t"], t_2b=s2["t"], mean_r_b=sb["mean"], mean_r_2b=s2["mean"],
                         n_b=sb["n"], n_2b=s2["n"], early=bool(early),
                         log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        self.log = (self.log + [[step, decision, self.s, self.b]])[-8:]
        self.last_population = dict(event="decision", step=step,
            participating_rows=None if participants is None else int(participants.sum()),
            required_rows=None if self.rows is None else self.rows - math.floor(self.population_q * self.rows),
            coverage_rejected=bool(coverage_rejected), scheduling_decision=decision)
        self.r_b, self.r_2b, self.blocks = [], [], []
        self.blocks_in_window = 0
        return decision

    def _look(self, step):
        """Early-stopping look after a completed pair; None while no e-process has crossed."""
        if len(self.r_b) < 2 and len(self.r_2b) < 2:
            return None
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._early_stat(vb_values), self._early_stat(v2_values)
        self.looks += 1
        self.last_look = dict(step=step, n_b=sb["n"], n_2b=s2["n"], t_b=sb["t"], t_2b=s2["t"],
                              log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        if sb["verdict"] == 0 and s2["verdict"] == 0:
            return None
        if self.hold_descent and (sb["verdict"] == -1 or s2["verdict"] == -1):
            return None                 # a descent is due but held: keep the evidence, decide when the hold lifts
        if self.early_stationary_only:
            # Only a descent (STATIONARY: cheap to undo) may be decided early, and only when the other
            # scale can speak (>= 2 values) and does not oppose it. DRIFT / release and MIXED are decided
            # at the window end with the full two-scale table, so an oscillation (r_b > 0, r_2b < 0)
            # keeps its MIXED outcome instead of becoming an early DRIFT.
            calm_b = sb["verdict"] == -1 and s2["verdict"] != 1 and s2["n"] >= 2 and s2["mean"] <= 0.
            calm_2b = s2["verdict"] == -1 and sb["verdict"] != 1 and sb["n"] >= 2 and sb["mean"] <= 0.
            if not (calm_b or calm_2b):
                return None
        self.windows += 1
        return self._conclude(sb, s2, self.b, step, early=True)

    def _decide(self, step):
        """Window end (2K blocks, nothing crossed early): fixed-n tests, then the usual actions."""
        tested_b = self.b
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._final_verdict(vb_values), self._final_verdict(v2_values)
        self.counts["dropped_pairs"] += (len(vb_values) - sb["n"]) + (len(v2_values) - s2["n"])
        self.windows += 1
        return self._conclude(sb, s2, tested_b, step, early=False)

    @torch.no_grad()
    def observe(self, params, ratio, step=None):
        """Call after the group's optimizer step with applied LR / base LR."""
        if self.anchor is None:
            raise RuntimeError("SettleTest.observe before the anchor was taken")
        self.tau += float(ratio)
        if self.tau < self.b:
            return None
        self.tau -= self.b
        theta = self.flat(params)
        d = theta - self.anchor
        if self.invalid_block_rows is not None and self.rows is not None:
            d.view(self.rows, -1)[self.invalid_block_rows] = float("nan")
            self.invalid_block_rows.zero_()
        self.anchor = theta.clone()
        self.last_block = d
        self.blocks.append(d)
        self.blocks_in_window += 1
        new_pair = False
        if len(self.blocks) % 2 == 0:
            self.r_b.append(self._cos(self.blocks[-2], self.blocks[-1]))
            new_pair = True
        if len(self.blocks) == 4:
            self.r_2b.append(self._cos(self.blocks[0] + self.blocks[1], self.blocks[2] + self.blocks[3]))
            self.blocks = []
        if self.blocks_in_window >= 2 * self.K:
            return self._decide(step)
        if new_pair and self.early:
            return self._look(step)
        return None

    def diagnostics(self):
        out = super().diagnostics()
        out["sequential"] = True
        out["hold_descent"] = self.hold_descent
        out["excluded_rows"] = None if self.exclude is None else int(self.exclude.sum())
        out["b_anchor"] = self.b_anchor
        out["early_decisions"] = self.counts.get("early", 0)
        out["looks"] = self.looks
        out["last_look"] = self.last_look
        out["population_schema"] = self.population_schema
        out["population_policy"] = self.population_policy
        out["population_q"] = self.population_q
        out["population_active"] = self.population_active
        out["stationary_participating_rows"] = (None if self.stationary_rows is None
                                               else int(self.stationary_rows.sum()))
        out["stationary_undo_s"] = self.stationary_undo_s
        out["last_population"] = dict(self.last_population)
        return out
