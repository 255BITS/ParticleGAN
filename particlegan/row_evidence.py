"""Per-row force-persistence evidence from the table's own gradients (information class I0: no real data, no model output).

A table row that sits in a region of the critic's field that pushes it persistently in one direction has a gradient history with a
non-zero mean; a row at equilibrium has a zero-mean history. Every touch (a row appears in the generator batch) gives the row one gradient
g in R^d. The row keeps exponentially weighted sufficient statistics over its own touches (weights (1-lam)^age, so the effective sample size
is n_eff = (sum w)^2 / sum w^2, strictly below (2-lam)/lam) and tests H0: E[g] = 0.

statistic="sphericity" (legacy default, used by the qualified e22 recipe; historically mislabelled "Hotelling T^2"):
t2 = n_eff |mean|^2 / (tr(S) / d), i.e. the squared mean scaled by the AVERAGE per-coordinate variance, not by the inverse
covariance. It is a Hotelling statistic only under a spherical covariance S = sigma^2 I. For d = 2 the law used is
p = (1 + T^2/(n-1))^(-(n-2)/2) (the F(2, n-2) survival function); for other d the chi^2_d survival function, which is
anti-conservative below n ~ 30, so a p-value needs n_eff >= 3 d (otherwise p = 1). Under anisotropic null noise it is
miscalibrated: at d = 8 with a spread spectrum P(p <= .05) is about .17 and BH flags rows that are all false. With window
50, n_eff < 99, so z_dim >= 33 can never test: the constructor warns once and sets ``untestable`` (it does not raise, to keep
qualified configurations loadable).

statistic="hotelling" (declared option): the true Hotelling T^2 = n_eff mean^T S^-1 mean with the full weighted covariance
S (float64 per-row d x d accumulators: n * d * d * 8 bytes, e.g. about 2 GB at n = 60k rows and d = 64), and the F law (n_eff - d) / (d (n_eff - 1)) T^2 ~ F(d, n_eff - d), evaluated with
an in-torch regularized incomplete beta. A p-value needs n_eff >= d + 2 and a positive-definite S (otherwise p = 1); a
configuration that can never reach that raises. The EWMA weights make n_eff an effective (Kish) count, so the law is
approximate; tests/test_row_evidence_calibration.py checks its null calibration under isotropic and anisotropic noise.

Rows are then flagged by Benjamini-Hochberg at level Q over all rows (Q is used only as this FDR level).

null="scaled" (row_evidence_null): the law above assumes a row's touch gradients are independent with zero mean. Neighbouring rows share a slowly
varying critic force, gradients are correlated across touches, and a row's mean force is not zero away from the centre of its mode, so the statistics of
the bulk are inflated by a common factor that depends on the noise level and the critic's speed. The scaled null therefore applies the same law to
t2 / c, where c >= 1 is the smallest scale that puts the median p-value of the tested rows at .5 (the typical row is the null; at least half of the
rows are assumed to be null). Only rows extreme relative to the bulk are flagged; the accumulators are then float64 (variance by subtraction).
"""
import math
import warnings

import torch

from ._stats import bh_rejections


STATISTICS = ("sphericity", "hotelling")
_WARNED = set()


class RowEvidenceUntestableWarning(UserWarning):
    """The configured row-evidence gate can never produce a p-value below one."""


def _betacf(a, b, x, iterations):
    """Continued fraction of the incomplete beta (modified Lentz, Numerical Recipes 6.4)."""
    tiny = 1e-300

    def fix(value):
        return torch.where(value.abs() < tiny, torch.full_like(value, tiny), value)

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = torch.ones_like(x)
    d = 1.0 / fix(1.0 - qab * x / qap)
    h = d
    for m in range(1, iterations + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / fix(1.0 + aa * d)
        c = fix(1.0 + aa / c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / fix(1.0 + aa * d)
        c = fix(1.0 + aa / c)
        h = h * d * c
    return h


def betainc(a, b, x, iterations=200):
    """Regularized incomplete beta I_x(a, b) for tensors (float64 recommended)."""
    a, b, x = torch.broadcast_tensors(a, b, x)
    x = x.clamp(0.0, 1.0)
    swap = x > (a + 1.0) / (a + b + 2.0)
    a2, b2 = torch.where(swap, b, a), torch.where(swap, a, b)
    x2 = torch.where(swap, 1.0 - x, x).clamp(1e-300, 1.0 - 1e-16)
    log_front = (a2 * torch.log(x2) + b2 * torch.log1p(-x2)
                 - (torch.lgamma(a2) + torch.lgamma(b2) - torch.lgamma(a2 + b2)) - torch.log(a2))
    value = (log_front.exp() * _betacf(a2, b2, x2, iterations)).clamp(0.0, 1.0)
    value = torch.where(swap, 1.0 - value, value)
    return torch.where(x <= 0, torch.zeros_like(value), torch.where(x >= 1, torch.ones_like(value), value))


def f_survival(f, d1, d2, iterations=200):
    """P(F(d1, d2) > f) = I_{d2 / (d2 + d1 f)}(d2 / 2, d1 / 2)."""
    f = f.clamp_min(0.0)
    return betainc(d2 / 2.0, torch.full_like(f, d1 / 2.0), d2 / (d2 + d1 * f), iterations)


class RowEvidence:
    """Evidence for rows with independent particle lineages.

    Observe the full table gradient after backward and before per-row LR
    scaling or the optimizer step. Dense soft routing and conditional-bank
    restructuring are unsupported; the policy requires explicit independent
    row semantics before using this gate. Row evidence never moves rows.
    """

    def __init__(self, table, window=50, level=0.05, null="theory", *, semantics="independent",
                 statistic="sphericity"):
        if semantics != "independent":
            raise ValueError("row evidence requires independent particle rows; conditional and soft-routed banks are unsupported")
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or not all(table.shape)
                or table.layout != torch.strided or not table.is_floating_point()):
            raise ValueError("row evidence needs a floating particle table [rows, latent_dim]")
        if not math.isfinite(window) or window < 1 or not math.isfinite(level) or not 0 < level < 1:
            raise ValueError("row-evidence window must be at least one and level must be between zero and one")
        if null not in ("theory", "scaled"):
            raise ValueError("row-evidence null must be theory or scaled")
        if statistic not in STATISTICS:
            raise ValueError("row-evidence statistic must be sphericity or hotelling")
        n, d = table.shape
        dev = table.device
        dt = torch.float64 if null == "scaled" or statistic == "hotelling" else table.dtype
        self.n, self.d, self.lam, self.Q, self.null = n, d, 1.0 / float(window), float(level), null
        self.statistic = statistic
        # n_eff = (2-lam)/lam * (1-q^k)/(1+q^k) after k touches: strictly below the supremum.
        self.max_n_eff = (2.0 - self.lam) / self.lam
        self.required_n_eff = 3.0 * d if statistic == "sphericity" else d + 2.0
        self.untestable = self.required_n_eff >= self.max_n_eff
        if self.untestable:
            message = (f"row evidence ({statistic}) can never test: latent_dim {d} needs n_eff >= "
                       f"{self.required_n_eff:g} but window {window:g} keeps n_eff below {self.max_n_eff:g}; "
                       "every p-value is 1 and the gate is a no-op")
            if statistic != "sphericity":
                raise ValueError(message)
            key = (statistic, d, float(window))
            if key not in _WARNED:
                _WARNED.add(key)
                warnings.warn(message, RowEvidenceUntestableWarning, stacklevel=2)
        self._TENSORS = ("M", "Qs", "W", "S", "flag") + (("C",) if statistic == "hotelling" else ())
        if statistic == "hotelling":
            self.C = torch.zeros(n, d, d, device=dev, dtype=dt)  # sum_j w_j g_j g_j^T
            a_max = max(self.max_n_eff - d, 1.0) / 2.0
            self._iterations = min(400, 20 + int(10 * math.sqrt(max(a_max, d / 2.0))))
        self.p_values = None  # last p-values (diagnostic, not checkpointed)
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
            if self.statistic == "hotelling":
                self.C[idx] = keep * self.C[idx] + g.unsqueeze(2) * g.unsqueeze(1)
        self._test()
        self.counters["updates"] += 1
        self.counters["flagged_rows_sum"] += int(self.flag.sum())

    @torch.no_grad()
    def _test(self):
        d = self.d
        W = self.W.clamp_min(1e-30)
        n_eff = W * W / self.S.clamp_min(1e-30)
        mean = self.M / W.unsqueeze(1)
        if self.statistic == "hotelling":
            t2, ok = self._hotelling(W, n_eff, mean)
        else:
            # Sphericity statistic: unbiased AVERAGE per-coordinate variance of the weighted sample.
            ss = (self.Qs / W - mean.square().sum(1)).clamp_min(0.0)
            var = ss * n_eff / (n_eff - 1.0).clamp_min(1e-6) / d
            ok = (n_eff >= 3.0 * d) & (var > 0)
            t2 = n_eff * mean.square().sum(1) / var.clamp_min(1e-38)
        if self.null == "scaled":
            t2 = t2 / self._scale(t2, n_eff, ok)
        p = self._pvalue(t2, n_eff)
        p = torch.where(ok, p, torch.ones_like(p))
        self.p_values = p
        ps, order = p.sort()
        rejected = bh_rejections(ps, self.Q)
        flag = torch.zeros(self.n, device=p.device, dtype=torch.bool)
        if rejected:
            flag[order[:rejected]] = True
        self.flag = flag
        self.fraction = float(flag.double().mean())
        self.valid = True

    def _hotelling(self, W, n_eff, mean):
        """Full-covariance T^2 = n_eff m^T S^-1 m; rows without n_eff >= d + 2 or a PD S are untested."""
        d = self.d
        scale = (n_eff / (n_eff - 1.0).clamp_min(1e-6)).view(-1, 1, 1)
        cov = (self.C / W.view(-1, 1, 1) - mean.unsqueeze(2) * mean.unsqueeze(1)) * scale
        ok = n_eff >= d + 2.0
        eye = torch.eye(d, device=cov.device, dtype=cov.dtype).expand_as(cov)
        cov = torch.where(ok.view(-1, 1, 1), cov, eye)
        chol, info = torch.linalg.cholesky_ex(cov)
        ok = ok & (info == 0)
        chol = torch.where(ok.view(-1, 1, 1), chol, eye)
        solved = torch.cholesky_solve(mean.unsqueeze(2), chol).squeeze(2)
        t2 = (n_eff * (mean * solved).sum(1)).clamp_min(0.0)
        return torch.where(ok, t2, torch.zeros_like(t2)), ok

    def _pvalue(self, t2, n_eff):
        if self.statistic == "hotelling":
            d2 = (n_eff - self.d).clamp_min(1.0)
            f = d2 / (self.d * (n_eff - 1.0).clamp_min(1e-6)) * t2
            return f_survival(f, float(self.d), d2, self._iterations)
        if self.d == 2:
            return (-(n_eff - 2.0) / 2.0 * torch.log1p(t2 / (n_eff - 1.0).clamp_min(1e-6))).exp()
        return torch.special.gammaincc(torch.tensor(self.d / 2.0, device=t2.device, dtype=t2.dtype), t2 / 2.0)

    def _scale(self, t2, n_eff, ok):
        """Smallest c >= 1 with median_ok p(t2 / c) = .5 (bisection on log c in [0, ln 1e6], no host sync)."""
        lo = torch.zeros((), device=t2.device, dtype=t2.dtype)
        hi = torch.full_like(lo, math.log(1e6))
        nan = torch.full_like(t2, float("nan"))
        for _ in range(24):
            mid = (lo + hi) / 2
            med = torch.where(ok, self._pvalue(t2 / mid.exp(), n_eff), nan).nanmedian()
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
        if self.statistic == "hotelling":
            self.C[rows] = 0
        self.flag[rows] = False
        self.counters["resets"] += int(len(rows))

    def state_dict(self):
        return dict(**{k: getattr(self, k).clone() for k in self._TENSORS}, fraction=self.fraction, valid=self.valid,
                    counters=dict(self.counters), scale_c=self.scale_c, config=self._config())

    def _config(self):
        config = {"rows": self.n, "latent_dim": self.d, "decay": self.lam, "level": self.Q, "null": self.null}
        if self.statistic != "sphericity":  # legacy checkpoints keep their exact configuration
            config["statistic"] = self.statistic
        return config

    def diagnostics(self):
        """Persistent gate diagnostics (derived from the configuration; never checkpointed)."""
        return {"statistic": self.statistic, "untestable": self.untestable,
                "required_n_eff": self.required_n_eff, "max_n_eff": self.max_n_eff,
                "fraction": self.fraction, "counters": dict(self.counters)}

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
