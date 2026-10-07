"""Per-row force-persistence evidence from the table's own gradients (information class I0: no real data, no model output).

A table row that sits in a region of the critic's field that pushes it persistently in one direction has a gradient history with a
non-zero mean; a row at equilibrium has a zero-mean history. Every touch (a row appears in the generator batch) gives the row one gradient
g in R^d. The row keeps exponentially weighted sufficient statistics over its own touches (weights (1-lam)^age, so the effective sample size
is n_eff = (sum w)^2 / sum w^2, at most (2-lam)/lam) and tests H0: E[g] = 0 with a Hotelling T^2 statistic. For d = 2 the null law is
exactly p = (1 + T^2/(n-1))^(-(n-2)/2) (the F(2, n-2) survival function); for other d the chi^2_d survival function is used, which is
anti-conservative below n ~ 30, so a p-value needs n_eff >= 3 d (a test needs more observations than parameters; otherwise p = 1).
Rows are then flagged by Benjamini-Hochberg at level Q over all rows (Q is used only as this FDR level).

null="scaled" (row_evidence_null): the law above assumes a row's touch gradients are independent with zero mean. Neighbouring rows share a slowly
varying critic force, gradients are correlated across touches, and a row's mean force is not zero away from the centre of its mode, so the statistics of
the bulk are inflated by a common factor that depends on the noise level and the critic's speed. The scaled null therefore applies the same law to
t2 / c, where c >= 1 is the smallest scale that puts the median p-value of the tested rows at .5 (the typical row is the null; at least half of the
rows are assumed to be null). Only rows extreme relative to the bulk are flagged; the accumulators are then float64 (variance by subtraction).
"""
import math
import torch


class RowEvidence:
    """Evidence for rows with independent particle lineages.

    Observe the full table gradient after backward and before per-row LR
    scaling or the optimizer step. Dense soft routing and conditional-bank
    restructuring are unsupported; the policy requires explicit independent
    row semantics before using this gate. Row evidence never moves rows.
    """

    def __init__(self, table, window=50, level=0.05, null="theory", *, semantics="independent"):
        if semantics != "independent":
            raise ValueError("row evidence requires independent particle rows; conditional and soft-routed banks are unsupported")
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or not all(table.shape)
                or table.layout != torch.strided or not table.is_floating_point()):
            raise ValueError("row evidence needs a floating particle table [rows, latent_dim]")
        if not math.isfinite(window) or window < 1 or not math.isfinite(level) or not 0 < level < 1:
            raise ValueError("row-evidence window must be at least one and level must be between zero and one")
        if null not in ("theory", "scaled"):
            raise ValueError("row-evidence null must be theory or scaled")
        n, d = table.shape
        dev, dt = table.device, (torch.float64 if null == "scaled" else table.dtype)
        self.n, self.d, self.lam, self.Q, self.null = n, d, 1.0 / float(window), float(level), null
        self.scale_c = 1.0
        self.M = torch.zeros(n, d, device=dev, dtype=dt)      # sum_j w_j g_j
        self.Qs = torch.zeros(n, device=dev, dtype=dt)        # sum_j w_j |g_j|^2
        self.W = torch.zeros(n, device=dev, dtype=dt)         # sum_j w_j
        self.S = torch.zeros(n, device=dev, dtype=dt)         # sum_j w_j^2
        self.flag = torch.zeros(n, device=dev, dtype=torch.bool)
        self.fraction = 0.0
        self.valid = False
        self.counters = {"updates": 0, "flagged_rows_sum": 0, "resets": 0}

    @torch.no_grad()
    def update(self, grad):
        """Observe the dense, aggregated table gradient and refresh row flags.

        Nonzero rows count as one touch per backward pass. Duplicate particle
        draws are already summed by autograd and count once here. Supply the
        full ``table.grad`` before any LR scaling, not per-sample gradients;
        absent or sparse gradients do not provide the required observation.
        """
        if (not isinstance(grad, torch.Tensor) or grad.shape != self.M.shape
                or grad.layout != torch.strided or grad.device != self.M.device or not grad.is_floating_point()):
            raise ValueError("row evidence requires the full floating table gradient on the table device")
        idx = (grad != 0).any(1).nonzero().flatten()
        if len(idx):
            g = grad[idx].to(self.M.dtype)
            keep = 1.0 - self.lam
            self.M[idx] = keep * self.M[idx] + g
            self.Qs[idx] = keep * self.Qs[idx] + g.square().sum(1)
            self.W[idx] = keep * self.W[idx] + 1.0
            self.S[idx] = keep * keep * self.S[idx] + 1.0
        self._test()
        self.counters["updates"] += 1
        self.counters["flagged_rows_sum"] += int(self.flag.sum())

    @torch.no_grad()
    def _test(self):
        d = self.d
        W = self.W.clamp_min(1e-30)
        n_eff = W * W / self.S.clamp_min(1e-30)
        mean = self.M / W.unsqueeze(1)
        # unbiased per-coordinate variance of the weighted sample
        ss = (self.Qs / W - mean.square().sum(1)).clamp_min(0.0)
        var = ss * n_eff / (n_eff - 1.0).clamp_min(1e-6) / d
        ok = (n_eff >= 3.0 * d) & (var > 0)
        t2 = n_eff * mean.square().sum(1) / var.clamp_min(1e-38)
        if self.null == "scaled":
            t2 = t2 / self._scale(t2, n_eff, ok)
        p = self._pvalue(t2, n_eff)
        p = torch.where(ok, p, torch.ones_like(p))
        ps, order = p.sort()
        ranks = torch.arange(1, self.n + 1, device=p.device, dtype=p.dtype)
        passed = (ps <= ranks * self.Q / self.n).nonzero()
        flag = torch.zeros(self.n, device=p.device, dtype=torch.bool)
        if len(passed):
            flag[order[:int(passed[-1]) + 1]] = True
        self.flag = flag
        self.fraction = float(flag.double().mean())
        self.valid = True

    def _pvalue(self, t2, n_eff):
        if self.d == 2:
            return (-(n_eff - 2.0) / 2.0 * torch.log1p(t2 / (n_eff - 1.0).clamp_min(1e-6))).exp()
        return torch.special.gammaincc(torch.tensor(self.d / 2.0, device=t2.device, dtype=t2.dtype), t2 / 2.0)

    def _scale(self, t2, n_eff, ok):
        """Smallest c >= 1 with median_ok p(t2 / c) = .5 (bisection on log c in [0, ln 1e6], no host sync)."""
        lo = torch.zeros((), device=t2.device, dtype=t2.dtype)
        hi = torch.full_like(lo, math.log(1e6))
        nan = torch.full_like(t2, float("nan"))
        if self.d == 2:
            # These factors stay fixed for all 24 trials. Keep the two
            # divisions in their original order to preserve rounding.
            exponent = -(n_eff - 2.0) / 2.0
            denominator = (n_eff - 1.0).clamp_min(1e-6)

            def pvalue(value):
                return (exponent * torch.log1p(value / denominator)).exp()
        else:
            def pvalue(value):
                return self._pvalue(value, n_eff)
        for _ in range(24):
            mid = (lo + hi) / 2
            med = torch.where(ok, pvalue(t2 / mid.exp()), nan).nanmedian()
            below = med < 0.5
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        c = ((lo + hi) / 2).exp()
        self.scale_c = float(c)
        self.counters["scale_sum"] = self.counters.get("scale_sum", 0.0) + self.scale_c
        return c

    @torch.no_grad()
    def reset(self, rows):
        """A teleported row begins a new lineage: its gradient history belongs to another place."""
        self.M[rows] = 0
        self.Qs[rows] = 0
        self.W[rows] = 0
        self.S[rows] = 0
        self.flag[rows] = False
        self.counters["resets"] += int(len(rows))

    _TENSORS = ("M", "Qs", "W", "S", "flag")

    def state_dict(self):
        return dict(**{k: getattr(self, k).clone() for k in self._TENSORS}, fraction=self.fraction, valid=self.valid,
                    counters=dict(self.counters), scale_c=self.scale_c, config=self._config())

    def _config(self):
        return {"rows": self.n, "latent_dim": self.d, "decay": self.lam, "level": self.Q, "null": self.null}

    def check_state(self, state):
        keys = set(self._TENSORS) | {"fraction", "valid", "counters"}
        if not isinstance(state, dict) or set(state) not in (keys, keys | {"scale_c", "config"}):
            raise ValueError("invalid row-evidence state")
        for k in self._TENSORS:
            if (not isinstance(state[k], torch.Tensor) or state[k].shape != getattr(self, k).shape
                    or state[k].dtype != getattr(self, k).dtype):
                raise ValueError(f"row-evidence {k} does not match the table")
        if "config" in state:
            if state["config"] != self._config():
                raise ValueError("row-evidence state configuration does not match the particle rows")
            if not isinstance(state["scale_c"], (int, float)) or not math.isfinite(state["scale_c"]) or state["scale_c"] < 1:
                raise ValueError("invalid row-evidence scale")
        if (not isinstance(state["fraction"], (int, float)) or not 0 <= state["fraction"] <= 1
                or type(state["valid"]) is not bool or not isinstance(state["counters"], dict)):
            raise ValueError("invalid row-evidence diagnostics")

    def load_state_dict(self, state):
        self.check_state(state)
        for k in self._TENSORS:
            setattr(self, k, state[k].clone().to(self.M.device))
        self.fraction, self.valid, self.counters = float(state["fraction"]), bool(state["valid"]), dict(state["counters"])
        self.scale_c = float(state.get("scale_c", 1.0))
