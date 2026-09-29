"""Per-row force-persistence evidence from the table's own gradients (information class I0: no real data, no model output).

A table row that sits in a region of the critic's field that pushes it persistently in one direction has a gradient history with a
non-zero mean; a row at equilibrium has a zero-mean history. Every touch (a row appears in the generator batch) gives the row one gradient
g in R^d. The row keeps exponentially weighted sufficient statistics over its own touches (weights (1-lam)^age, so the effective sample size
is n_eff = (sum w)^2 / sum w^2, at most (2-lam)/lam) and tests H0: E[g] = 0 with a Hotelling T^2 statistic. For d = 2 the null law is
exactly p = (1 + T^2/(n-1))^(-(n-2)/2) (the F(2, n-2) survival function); for other d the chi^2_d survival function is used, which is
anti-conservative below n ~ 30, so a p-value needs n_eff >= 3 d (a test needs more observations than parameters; otherwise p = 1).
Rows are then flagged by Benjamini-Hochberg at level Q over all rows (Q is used only as this FDR level).
"""
import torch


class RowEvidence:
    def __init__(self, table, window=50, level=0.05):
        n, d = table.shape
        dev, dt = table.device, table.dtype
        self.n, self.d, self.lam, self.Q = n, d, 1.0 / float(window), float(level)
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
        """Fold this step's table gradient in (rows with a non-zero gradient were touched) and refresh the flags."""
        idx = (grad != 0).any(1).nonzero().flatten()
        if len(idx):
            g = grad[idx]
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
        if d == 2:
            logp = -(n_eff - 2.0) / 2.0 * torch.log1p(t2 / (n_eff - 1.0).clamp_min(1e-6))
            p = logp.exp()
        else:
            p = torch.special.gammaincc(torch.tensor(d / 2.0, device=t2.device, dtype=t2.dtype), t2 / 2.0)
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
                    counters=dict(self.counters))

    def check_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self._TENSORS) | {"fraction", "valid", "counters"}:
            raise ValueError("invalid row-evidence state")
        for k in self._TENSORS:
            if not isinstance(state[k], torch.Tensor) or state[k].shape != getattr(self, k).shape:
                raise ValueError(f"row-evidence {k} does not match the table")

    def load_state_dict(self, state):
        self.check_state(state)
        for k in self._TENSORS:
            setattr(self, k, state[k].clone().to(self.M.device))
        self.fraction, self.valid, self.counters = float(state["fraction"]), bool(state["valid"]), dict(state["counters"])
