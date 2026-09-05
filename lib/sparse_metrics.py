"""
Metrics for the sparse conditional mixed-output toy (report §4).

Every function takes tensors that already live on one device and returns
plain python floats. The organising idea is that each axis of the toy has a
metric that can fail *independently* of the others, so a run's failure mode is
readable from the summary row:

  coverage      modes / hq / mode_recall / hist_kl  -- did G find the modes
  sparsity      sparse_prec@eps / sparse_rec / exact_zero_frac / smear
                -- are the inactive dims exactly zero and the active ones live
  conditioning  cond_acc / cond_sep_ratio / cond_var_ratio
                -- does the requested class actually steer x
  symbols       sym_acc_mode / sym_acc_cond / sym_cond_kl / joint_acc
                -- is the emitted symbol glued to the real part and to c
  shape         core_ratio (median-based, active dims only) / sliced W1

`convergence_bar` turns a metrics dict into the single pass/fail the study
reports, plus the per-axis verdicts.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch

from lib.sparse_toy import SparseMixedToy
from lib.toy_metrics import sliced_w1

# Cache of median ||N(0, I_k)|| per k, so core widths can be read as a sigma ratio.
_CHI_MEDIAN: Dict[int, float] = {}


def chi_median(k: int) -> float:
    if k not in _CHI_MEDIAN:
        g = torch.Generator().manual_seed(12345)
        _CHI_MEDIAN[k] = float(torch.randn(400_000, k, generator=g).norm(dim=1).median())
    return _CHI_MEDIAN[k]


def _entropy_kl_uniform(counts: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    p = counts.to(torch.float64)
    p = p / p.sum().clamp_min(eps) + eps
    p = p / p.sum()
    q = torch.full_like(p, 1.0 / p.numel())
    kl = float((p * (p / q).log()).sum())
    m = 0.5 * (p + q)
    js = float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())
    return {"kl": kl, "js": js}


# =========================
#  Coverage
# =========================

@torch.no_grad()
def coverage(toy: SparseMixedToy, x: torch.Tensor, m_hat: torch.Tensor, dist: torch.Tensor,
             hq_sigmas: float = 3.0, min_count: int = 10, recall_frac: float = 0.5) -> Dict[str, float]:
    """
    modes: modes with >= min_count high-quality samples; hq: fraction of samples
    within hq_sigmas * sigma * sqrt(k) of their nearest center (full-dim
    distance, so smear on the 21 inactive dims *does* count against hq);
    mode_recall / hist_kl / hist_js: nearest-center mass balance, no threshold.
    """
    n = toy.n_modes
    radius = hq_sigmas * toy.sigma * math.sqrt(toy.k)
    hq = dist <= radius
    counts_hq = torch.bincount(m_hat[hq], minlength=n)
    counts = torch.bincount(m_hat, minlength=n).to(torch.float64)
    thresh = recall_frac * x.shape[0] / n
    div = _entropy_kl_uniform(counts)
    return {
        "modes": int((counts_hq >= min_count).sum()),
        "hq": float(hq.float().mean()),
        "mode_recall": float((counts >= thresh).to(torch.float64).mean()),
        "hist_kl": div["kl"],
        "hist_js": div["js"],
        "tail_frac_10sigma": float((dist > 10.0 * toy.sigma * math.sqrt(toy.k)).float().mean()),
    }


# =========================
#  Sparsity
# =========================

@torch.no_grad()
def sparsity(toy: SparseMixedToy, x: torch.Tensor, m_hat: torch.Tensor,
             eps_list: Sequence[float] = (1e-3, 1e-2), rec_sigmas: float = 10.0) -> Dict[str, float]:
    """
    Per assigned mode, the truly inactive / active dims are known, so:
      sparse_prec@eps: frac of |x_j| < eps over truly-inactive dims,
      sparse_rec:      frac of |x_j| > rec_sigmas * sigma over truly-active dims,
      exact_zero_frac: frac of inactive-dim coordinates that are *exactly* 0.0,
      smear:           mean |x_j| over inactive dims (the v0 failure in one number),
      active_bias:     mean |mean_j(x_j) - mu_j| over active dims, per mode, averaged.
    """
    act = toy.active[m_hat]                   # (N, d) bool
    inact = ~act
    ax = x.abs()
    out: Dict[str, float] = {}
    n_inact = inact.sum().clamp_min(1)
    n_act = act.sum().clamp_min(1)
    for eps in eps_list:
        tag = f"{eps:g}".replace("-", "m").replace(".", "p")
        out[f"sparse_prec@{tag}"] = float(((ax < eps) & inact).sum() / n_inact)
    out["sparse_rec"] = float(((ax > rec_sigmas * toy.sigma) & act).sum() / n_act)
    out["exact_zero_frac"] = float(((x == 0.0) & inact).sum() / n_inact)
    out["smear"] = float((ax * inact).sum() / n_inact)

    # active_bias: per-mode mean on active dims vs the true center.
    n_modes, d = toy.centers.shape
    sums = torch.zeros(n_modes, d, device=x.device, dtype=torch.float64).index_add_(0, m_hat, x.to(torch.float64))
    cnt = torch.bincount(m_hat, minlength=n_modes).to(torch.float64)
    have = cnt >= 5
    if bool(have.any()):
        means = sums[have] / cnt[have].unsqueeze(1)
        err = (means - toy.centers[have].to(torch.float64)).abs()
        a = toy.active[have]
        out["active_bias"] = float((err * a).sum() / a.sum().clamp_min(1))
    else:
        out["active_bias"] = float("nan")
    return out


# =========================
#  Conditioning
# =========================

@torch.no_grad()
def conditioning(toy: SparseMixedToy, x: torch.Tensor, m_hat: torch.Tensor, c_in: torch.Tensor,
                 x_real: torch.Tensor, c_real: torch.Tensor, n_proj: int = 64, seed: int = 0) -> Dict[str, float]:
    """
    cond_acc:        frac of samples whose nearest mode belongs to the requested class
                     (ground truth, independent of any D head),
    cond_sep_ratio:  mean pairwise sliced-W1 between per-class *fake* clouds divided by
                     the same quantity on *real* clouds. ~1 healthy; -> 0 is
                     conditioning collapse (every class emits the same cloud),
    cond_var_ratio:  between-class variance of x / total variance (fraction of the
                     output's variance that c explains). Real-data reference is
                     reported alongside as cond_var_ratio_real.
    """
    C = toy.n_classes
    out = {"cond_acc": float((toy.class_of_mode[m_hat] == c_in).float().mean())}

    def pairwise_sep(xx: torch.Tensor, cc: torch.Tensor) -> float:
        clouds = [xx[cc == c] for c in range(C)]
        n = min(cl.shape[0] for cl in clouds)
        if n < 8:
            return float("nan")
        vals = []
        for i in range(C):
            for j in range(i + 1, C):
                vals.append(sliced_w1(clouds[i][:n], clouds[j][:n], n_proj=n_proj, seed=seed + 100 * i + j))
        return float(sum(vals) / len(vals))

    sep_fake = pairwise_sep(x, c_in)
    sep_real = pairwise_sep(x_real, c_real)
    out["cond_sep_fake"] = sep_fake
    out["cond_sep_ratio"] = sep_fake / sep_real if sep_real and sep_real > 0 else float("nan")

    def var_ratio(xx: torch.Tensor, cc: torch.Tensor) -> float:
        xx = xx.to(torch.float64)
        total = xx.var(dim=0, unbiased=False).sum()
        mu = xx.mean(dim=0)
        between = torch.zeros((), dtype=torch.float64, device=xx.device)
        for c in range(C):
            sel = cc == c
            if sel.any():
                between = between + (sel.float().mean().to(torch.float64)) * ((xx[sel].mean(dim=0) - mu) ** 2).sum()
        return float(between / total.clamp_min(1e-12))

    out["cond_var_ratio"] = var_ratio(x, c_in)
    out["cond_var_ratio_real"] = var_ratio(x_real, c_real)
    return out


@torch.no_grad()
def particle_class_purity(particle_idx: torch.Tensor, c_in: torch.Tensor,
                          c_out: torch.Tensor, num_particles: int,
                          n_classes: int, min_count: int = 4) -> float:
    """Mean dominant-class share of each particle's correctly conditioned draws.

    Indices and classes must describe the same generated samples. Only particles
    with at least min_count correct draws enter the mean; no eligible particles
    gives NaN. This measures specialization, not generation quality: with a class
    embedding, one shared particle can correctly serve every requested class,
    yielding purity near 1 / n_classes under balanced sampling. Without class
    input, a particle that always generates one class can instead have purity 1.
    Finite sample counts bias the dominant-class share upward.
    """
    correct = c_out == c_in
    counts = torch.zeros(num_particles, n_classes, device=particle_idx.device)
    counts.index_put_((particle_idx[correct], c_in[correct]),
                      torch.ones_like(c_in[correct], dtype=counts.dtype), accumulate=True)
    totals = counts.sum(1)
    used = totals >= min_count
    if not bool(used.any()):
        return float("nan")
    return float((counts[used].max(1).values / totals[used]).mean())


# =========================
#  Symbols
# =========================

@torch.no_grad()
def symbols(toy: SparseMixedToy, logits: torch.Tensor, y: torch.Tensor, m_hat: torch.Tensor,
            c_in: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    sym_acc_mode: emitted symbol == the symbol of the mode the real part landed in
                  (joint coherence between the two heads),
    sym_acc_cond: emitted symbol has nonzero mass under the true p(s | c_in),
    sym_cond_kl:  mean_c KL( p(s|c) || emitted histogram(s|c) ), 0 iff the
                  per-class symbol distribution is right (this is what
                  separates 'right distribution' from 'right single symbol'
                  under symbol_map='split'),
    joint_acc:    class(m_hat) == c_in AND emitted symbol == symbol(m_hat),
    sym_hist_kl:  marginal symbol histogram vs the true marginal (uniform),
    sym_conf:     mean max softmax(logits): how decided the head is,
    y_hardness:   mean max(y) at eval (always 1.0 after the generator's hard
                  argmax read-out; this does not measure training saturation).
    """
    K, C = toy.n_symbols, toy.n_classes
    s_out = logits.argmax(dim=1)
    p_true = toy.p_symbol_given_class  # (C, K)
    out = {
        "sym_acc_mode": float((s_out == toy.symbol_of_mode[m_hat]).float().mean()),
        "sym_acc_cond": float((p_true[c_in, s_out] > 0).float().mean()),
        "joint_acc": float(((toy.class_of_mode[m_hat] == c_in) & (s_out == toy.symbol_of_mode[m_hat])).float().mean()),
        "sym_hist_kl": _entropy_kl_uniform(torch.bincount(s_out, minlength=K))["kl"],
        "sym_conf": float(torch.softmax(logits, dim=1).max(dim=1).values.mean()),
        "y_hardness": float(y.max(dim=1).values.mean()),
    }
    kls = []
    for c in range(C):
        sel = c_in == c
        if not bool(sel.any()):
            continue
        q = torch.bincount(s_out[sel], minlength=K).to(torch.float64)
        q = q / q.sum().clamp_min(1) + eps
        q = q / q.sum()
        p = p_true[c].to(torch.float64) + eps
        p = p / p.sum()
        kls.append(float((p * (p / q).log()).sum()))
    out["sym_cond_kl"] = float(sum(kls) / len(kls)) if kls else float("nan")
    return out


# =========================
#  Shape
# =========================

@torch.no_grad()
def core_ratio(toy: SparseMixedToy, x: torch.Tensor, m_hat: torch.Tensor, min_count: int = 30) -> Dict[str, float]:
    """
    Median-centered core width on the *active* dims of each assigned mode,
    inverted through median ||N(0, sigma^2 I_k)|| = sigma * chi_median(k), so
    1.0 is the data's width, < 1 a compressed core, > 1 a blurred one. Same
    estimator as lib.toy_metrics.per_mode_core_ratio, generalised to k dims.
    """
    counts = torch.bincount(m_hat, minlength=toy.n_modes)
    sig: List[float] = []
    factor = toy.sigma * chi_median(toy.k)
    for m in torch.nonzero(counts >= min_count).flatten().tolist():
        pts = x[m_hat == m][:, toy.active[m]].to(torch.float64)
        ctr = pts.quantile(0.5, dim=0)
        r = (pts - ctr).norm(dim=1)
        sig.append(float(r.quantile(0.5)) / factor)
    val = float(sum(sig) / len(sig)) if sig else float("nan")
    return {"core_ratio": val, "audited_modes": len(sig)}


# =========================
#  All together
# =========================

@torch.no_grad()
def evaluate_samples(
    toy: SparseMixedToy,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    c_in: torch.Tensor,
    x_real: torch.Tensor,
    c_real: torch.Tensor,
    seed: int = 0,
    with_shape: bool = True,
) -> Dict[str, float]:
    m_hat, dist = toy.assign(x)
    out: Dict[str, float] = {}
    out.update(coverage(toy, x, m_hat, dist))
    out.update(sparsity(toy, x, m_hat))
    out.update(conditioning(toy, x, m_hat, c_in, x_real, c_real, seed=seed))
    out.update(symbols(toy, logits, y, m_hat, c_in))
    n = min(x.shape[0], x_real.shape[0])
    out["sliced_w1"] = sliced_w1(x[:n], x_real[:n], n_proj=128, seed=seed + 7)
    if with_shape:
        out.update(core_ratio(toy, x, m_hat))
    return out


BAR = {
    # metric: (threshold, direction)
    "modes_full": None,  # handled specially: modes == n_modes
    "hq": (0.90, ">="),
    "cond_acc": (0.95, ">="),
    "sym_acc_mode": (0.95, ">="),
    "sparse_prec@0p01": (0.95, ">="),
}


def convergence_bar(row: Dict[str, float], n_modes: int) -> Dict[str, bool]:
    """Per-axis verdicts plus the overall bar (all axes at once)."""
    v = {
        "bar_coverage": row.get("modes", 0) == n_modes and row.get("hq", 0.0) >= BAR["hq"][0],
        "bar_cond": row.get("cond_acc", 0.0) >= BAR["cond_acc"][0],
        "bar_sym": row.get("sym_acc_mode", 0.0) >= BAR["sym_acc_mode"][0],
        "bar_sparse": row.get("sparse_prec@0p01", 0.0) >= BAR["sparse_prec@0p01"][0],
    }
    v["bar_all"] = all(v.values())
    return v
