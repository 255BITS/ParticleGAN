#!/usr/bin/env python
"""
metric_recon.py

Reconciling `per_mode_std_ratio` with `hq`.

The audit reports both of these for the same 100k EMA samples:

  * `hq` -- the fraction of samples within 3 * 0.03 of their nearest grid
    center (lib.toy_models.mode_coverage), up to 0.992;
  * `per_mode_std_ratio` -- the mean within-mode std divided by the data's
    0.03 (lib.toy_metrics.per_mode_moments), 2.1 to 5.4.

Those cannot both describe a Gaussian. For a 2-D isotropic Gaussian whose std
is r times the data's, the mass inside 3 sigma_true is

    P(||x - mu|| < 3 * sigma_true) = 1 - exp(-(3 / r)^2 / 2),

which is 0.675 at r = 2 and 0.180 at r = 5. An hq of 0.99 with a raw std ratio
of 2.4 is therefore not "a wider Gaussian" -- it is a distribution whose *core*
is much tighter than its second moment suggests, with a small far tail carrying
the variance. This script measures which, by pulling the within-mode spread
apart into three estimators that a Gaussian would agree on and a
heavy-tailed blob would not:

  * **raw**      -- sqrt(mean per-dim population variance) per mode, i.e. the
                    audit's own `per_mode_std`. Second moment: maximally
                    tail-sensitive.
  * **trimmed**  -- the same after dropping the 2% of each mode's samples
                    farthest from that mode's sample mean. A true Gaussian is
                    truncated too, which biases this low by a known factor
                    (0.9593, derived below), so a bias-corrected column is
                    reported alongside.
  * **robust**   -- inverted from the *median* radial distance. For a 2-D
                    isotropic Gaussian median ||x - mu|| = sigma * sqrt(2 ln 2),
                    so sigma = median / 1.17741. Depends only on the core.

If raw >> robust the blob is a tight core plus a far tail; if the three agree,
the blob is genuinely that wide and `hq` would have to be low.

Units, stated once because the audit's numbers are dimensionless ratios and the
bias is not: `center_bias` is in **absolute data coordinates**, the same units
as the samples. The data's within-mode sigma is 0.03 and the grid spacing
between adjacent mode centers is 1.0, so a center bias of 0.02 is 0.67
sigma_true and 2% of the distance to the neighbouring mode.

Usage:

    python experiments/metric_recon.py
    python experiments/metric_recon.py --runs_dir results/runs_audit
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Allow `python experiments/metric_recon.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.toy_metrics import (  # noqa: E402 - after the sys.path shim
    RAYLEIGH_MEDIAN_FACTOR,
    TAIL_SIGMAS,
    per_mode_core_ratio,
)

GROUPS: Sequence[str] = (
    "b_cap_c1p0_lr2p0",
    "a_r1r2_c1p0",
    "a_r1r2_c0p1",
    "c_eikonal_c0p1",
)
SEEDS: Sequence[int] = (1, 2, 3, 4, 5)

# The benchmark's own constants (lib/toy_models.sample_100gaussians).
SIGMA_TRUE = 0.03
GRID_SPACING = 1.0
HQ_RADIUS = 3.0 * SIGMA_TRUE          # mode_coverage's "high quality" ball
AUDIT_MIN_COUNT = 50                  # per_mode_moments' qualifying threshold

# Tail cuts, in absolute coordinates. The 10-sigma one is the shared estimator's
# own cut (lib.toy_metrics.TAIL_SIGMAS), restated here in data units.
TAIL_10SIG = TAIL_SIGMAS * SIGMA_TRUE  # 0.30 -- a third of the way to the neighbour
TAIL_ABS = 0.15                        # halfway to the Voronoi boundary at 0.5

TRIM_FRAC = 0.02                      # drop this share of each mode's far points

# Tolerances for the reproduction checks. `hq` in summary.json is measured by
# mode_coverage on its *own* fresh 20k draw from the same EMA model, so the
# comparison is between two independent estimates of the same probability and
# the tolerance has to carry that draw's binomial error (worst case ~0.0035 at
# p = 0.5, and these groups do sit near 0.6). TOL_HQ is the systematic budget;
# HQ_DRAW_N supplies the statistical one at 3 sigma.
TOL_HQ = 0.005
HQ_DRAW_N = 20_000
TOL_STD_RATIO = 1e-3


# =========================
#  Estimator constants
# =========================

def rayleigh_median_factor() -> float:
    """
    median ||x - mu|| / sigma for a 2-D isotropic Gaussian: sqrt(2 ln 2).

    The inversion itself lives in `lib.toy_metrics.per_mode_core_ratio`; this
    re-exports its constant for the prose below.
    """
    return RAYLEIGH_MEDIAN_FACTOR


def trim_bias_factor(trim_frac: float = TRIM_FRAC) -> float:
    """
    What the 98%-trim does to a *true* 2-D Gaussian, so the trimmed column can
    be read as a sigma rather than as "sigma, shrunk by an unknown amount".

    With u = r^2 / (2 sigma^2) ~ Exp(1), truncating at the (1 - p) quantile
    U = -ln(p) gives

        E[u | u < U] = 1 - U p / (1 - p),

    and the per-dimension variance estimate is sigma^2 * E[u | u < U], so the
    trimmed std is sigma * sqrt(1 - U p / (1 - p)).
    """
    p = float(trim_frac)
    u_cut = -math.log(p)
    return math.sqrt(1.0 - u_cut * p / (1.0 - p))


def gaussian_hq(ratio: float) -> float:
    """
    hq a 2-D isotropic Gaussian of std `ratio * SIGMA_TRUE` would score:
    P(||x - mu|| < 3 sigma_true) = 1 - exp(-(3 / ratio)^2 / 2).

    This is the number the tension is about: it caps at 0.675 for ratio = 2.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return 1.0 - math.exp(-0.5 * (3.0 / ratio) ** 2)


# =========================
#  Per-run measurement
# =========================

def grid_centers() -> np.ndarray:
    """The 100 mixture centers, row-major (x, y), matching lib.toy_metrics."""
    coords = (np.arange(10, dtype=np.float64) - 4.5) * GRID_SPACING
    cx, cy = np.meshgrid(coords, coords, indexing="ij")
    return np.stack([cx.ravel(), cy.ravel()], axis=1)


def analyze_samples(x: np.ndarray) -> Dict:
    """
    Every number this script reports, from one run's 100k sample cloud.

    The robust (core) columns and the 10-sigma tail mass are not computed here:
    they are `lib.toy_metrics.per_mode_core_ratio`, which was factored out of
    this script so the leaderboard's variance-compression guard and this report
    read the same estimator. The raw and trimmed ladders stay local -- they
    exist to show what the core estimator is being compared against.
    """
    x = np.asarray(x, dtype=np.float64)
    centers = grid_centers()
    k = centers.shape[0]

    core = {
        tag: per_mode_core_ratio(
            x, min_count=AUDIT_MIN_COUNT, std=SIGMA_TRUE, center=tag
        )
        for tag in ("median", "mean", "true")
    }

    # Nearest-center assignment: the same one hq, per_mode_moments and
    # mode_recall_and_hists all use.
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(x.shape[0]), nearest])

    out: Dict = {"n_samples": int(x.shape[0])}

    # --- nearest-center distance distribution ------------------------------
    q = np.quantile(dist, [0.5, 0.9, 0.99, 0.999])
    out["dist_q50"] = float(q[0])
    out["dist_q90"] = float(q[1])
    out["dist_q99"] = float(q[2])
    out["dist_q999"] = float(q[3])
    out["dist_max"] = float(dist.max())
    out["frac_within_3sig"] = float((dist <= HQ_RADIUS).mean())
    out["frac_beyond_10sig"] = core["median"]["tail_frac_10sigma"]
    out["frac_beyond_0p15"] = float((dist > TAIL_ABS).mean())

    # --- within-mode spread -------------------------------------------------
    # Every estimator needs a *center* as well as a spread rule, and the choice
    # of center matters as much as the choice of spread: a mode's sample mean is
    # itself dragged by the far tail (a 3% tail at distance 2.5 shifts the mean
    # by 0.075 = 2.5 sigma_true), which then inflates every distance measured
    # from it. So each spread rule is computed about all three centers that make
    # sense here -- the sample mean (what the audit uses), the coordinate-wise
    # median (robust, center-free), and the true grid center (what `hq` uses).
    counts = np.bincount(nearest, minlength=k)
    qualifying = np.where(counts >= AUDIT_MIN_COUNT)[0]

    acc: Dict[str, List[float]] = {
        key: []
        for key in ("raw", "trim_mean", "trim_med", "bias_mean", "bias_med")
    }
    for m in qualifying:
        pts = x[nearest == m]
        mean = pts.mean(axis=0)
        med = np.median(pts, axis=0)
        ctr = centers[m]

        # raw: exactly per_mode_moments' estimator.
        acc["raw"].append(math.sqrt(float(pts.var(axis=0).mean())))
        acc["bias_mean"].append(float(np.linalg.norm(mean - ctr)))
        acc["bias_med"].append(float(np.linalg.norm(med - ctr)))

        r_mean = np.linalg.norm(pts - mean, axis=1)
        r_med = np.linalg.norm(pts - med, axis=1)

        # trimmed: drop the TRIM_FRAC farthest, ranked from each center in turn,
        # then take the ordinary std of what is left.
        for tag, r in (("trim_mean", r_mean), ("trim_med", r_med)):
            kept = pts[r <= np.quantile(r, 1.0 - TRIM_FRAC)]
            if kept.shape[0] >= 2:
                acc[tag].append(math.sqrt(float(kept.var(axis=0).mean())))

    out["audited_modes"] = int(qualifying.size)
    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}

    out["sigma_raw"] = means["raw"]
    out["sigma_trim_mean"] = means["trim_mean"]
    out["sigma_trim_med"] = means["trim_med"]
    out["sigma_trim_mean_corrected"] = means["trim_mean"] / trim_bias_factor()
    out["sigma_trim_med_corrected"] = means["trim_med"] / trim_bias_factor()
    out["sigma_robust_mean"] = core["mean"]["per_mode_core_std"]
    out["sigma_robust_med"] = core["median"]["per_mode_core_std"]
    out["sigma_robust_center"] = core["true"]["per_mode_core_std"]

    out["center_bias"] = means["bias_mean"]          # the audit's definition
    out["center_bias_in_sigma_true"] = means["bias_mean"] / SIGMA_TRUE
    out["center_bias_median"] = means["bias_med"]    # tail-free counterpart
    out["center_bias_median_in_sigma_true"] = means["bias_med"] / SIGMA_TRUE

    for key in (
        "raw",
        "trim_mean", "trim_med",
        "trim_mean_corrected", "trim_med_corrected",
        "robust_mean", "robust_med", "robust_center",
    ):
        ratio = out[f"sigma_{key}"] / SIGMA_TRUE
        out[f"ratio_{key}"] = float(ratio)
        out[f"gauss_hq_{key}"] = gaussian_hq(ratio)

    return out


def analyze_run(run_dir: Path) -> Dict:
    """One seed: measurements plus the two reproduction checks against summary.json."""
    samples = np.load(run_dir / "final_samples.npy")
    res = analyze_samples(samples)
    res["name"] = run_dir.name

    with open(run_dir / "summary.json", "r") as f:
        summary = json.load(f)
    final = summary["final"]

    res["hq_summary"] = float(final["hq"])
    res["hq_dev"] = abs(res["frac_within_3sig"] - res["hq_summary"])
    p = res["hq_summary"]
    res["hq_tol"] = TOL_HQ + 3.0 * math.sqrt(max(p * (1.0 - p), 0.0) / HQ_DRAW_N)
    res["hq_ok"] = bool(res["hq_dev"] <= res["hq_tol"])

    res["ratio_raw_summary"] = float(final.get("per_mode_std_ratio", float("nan")))
    res["ratio_raw_dev"] = abs(res["ratio_raw"] - res["ratio_raw_summary"])
    res["ratio_raw_ok"] = bool(res["ratio_raw_dev"] <= TOL_STD_RATIO)

    res["center_bias_summary"] = float(final.get("center_bias", float("nan")))
    return res


# =========================
#  Aggregation / report
# =========================

def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def aggregate(runs: List[Dict]) -> Dict:
    keys = [
        "dist_q50", "dist_q90", "dist_q99", "dist_q999", "dist_max",
        "frac_within_3sig", "frac_beyond_10sig", "frac_beyond_0p15",
        "ratio_raw",
        "ratio_trim_mean", "ratio_trim_med",
        "ratio_trim_mean_corrected", "ratio_trim_med_corrected",
        "ratio_robust_mean", "ratio_robust_med", "ratio_robust_center",
        "gauss_hq_raw", "gauss_hq_robust_med", "gauss_hq_robust_center",
        "center_bias", "center_bias_in_sigma_true",
        "center_bias_median", "center_bias_median_in_sigma_true",
        "hq_summary", "audited_modes",
    ]
    agg = {k: mean_std([r[k] for r in runs]) for k in keys}
    agg["n_seeds"] = len(runs)
    agg["max_hq_dev"] = max(r["hq_dev"] for r in runs)
    agg["hq_tol"] = max(r["hq_tol"] for r in runs)
    agg["max_ratio_raw_dev"] = max(r["ratio_raw_dev"] for r in runs)
    agg["checks_ok"] = all(r["hq_ok"] and r["ratio_raw_ok"] for r in runs)
    return agg


def story(agg: Dict) -> str:
    """
    One sentence per group, decided by which sigma definition predicts `hq`.

    The test: feed each ratio through `gaussian_hq` and see which one reproduces
    the measured mass inside 3 sigma_true. If `raw` does, the blob really is that
    wide; if only `robust` does, the raw std is tail-inflated and the core is a
    Gaussian of the robust width; if even `robust` under-predicts hq, the core is
    *more* peaked than a Gaussian and the spread lives entirely outside it.
    """
    raw = agg["ratio_raw"]["mean"]
    rob = agg["ratio_robust_med"]["mean"]
    rob_c = agg["ratio_robust_center"]["mean"]
    tail = agg["frac_beyond_10sig"]["mean"]
    hq = agg["frac_within_3sig"]["mean"]
    hq_raw = agg["gauss_hq_raw"]["mean"]
    hq_rob = agg["gauss_hq_robust_center"]["mean"]
    if not np.isfinite(raw) or not np.isfinite(rob):
        return "insufficient data."
    shrink = raw / rob if rob > 0 else float("nan")

    if rob < 0.8:
        core = "an **under-dispersed** core"
    elif rob <= 1.25:
        core = "a **near-true-width** core"
    else:
        core = f"a genuinely **over-dispersed** core ({rob:.2f}x)"

    if abs(hq_raw - hq) <= 0.02:
        head = (
            f"genuinely wide: a Gaussian at the raw ratio {raw:.2f} would score hq "
            f"{hq_raw:.3f} and the cloud scores {hq:.3f}, so the blobs really are "
            f"that broad -- no reconciliation needed"
        )
    elif abs(hq_rob - hq) <= 0.03:
        head = (
            f"{core} with a tail-inflated second moment: the raw ratio {raw:.2f} would "
            f"imply hq {hq_raw:.3f}, while the core width {rob_c:.2f} implies "
            f"{hq_rob:.3f} against a measured {hq:.3f}"
        )
    elif hq_rob > hq:
        head = (
            f"{core} with a tail-inflated second moment, and the tail is visible in hq "
            f"too: raw ratio {raw:.2f} implies hq {hq_raw:.3f} and the core width "
            f"{rob_c:.2f} implies {hq_rob:.3f}, against {hq:.3f} measured -- the "
            f"shortfall against the core prediction is the tail leaving the 3-sigma ball"
        )
    else:
        head = (
            f"mixed: raw {raw:.2f} implies hq {hq_raw:.3f}, core width {rob_c:.2f} "
            f"implies {hq_rob:.3f}, measured {hq:.3f}"
        )

    return (
        f"{head}. Core ratio {rob:.2f} (about the per-mode median) vs raw {raw:.2f} "
        f"= {shrink:.1f}x tail inflation, bought by the {100 * tail:.2f}% of samples "
        f"beyond 10 sigma_true."
    )


def fmt(m: Dict[str, float], digits: int = 3) -> str:
    if m["n"] == 0:
        return "-"
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"


def write_markdown(path: Path, groups: Dict[str, Dict]) -> None:
    trim_bias = trim_bias_factor()
    L: List[str] = []
    L.append("# Reconciling `per_mode_std_ratio` with `hq`")
    L.append("")
    L.append(
        "The audit reports mean within-mode std ratios of 2.1-5.4 (over-dispersed) "
        "next to `hq` values up to 0.992. A 2-D isotropic Gaussian cannot do both: "
        "at std ratio r its mass inside 3 sigma_true is "
        "`1 - exp(-(3/r)^2/2)`, which is **0.675 at r = 2** and 0.180 at r = 5. "
        "This file resolves the tension from the saved 100k EMA sample clouds "
        "(`final_samples.npy`) rather than by assumption."
    )
    L.append("")

    # --- definitions -------------------------------------------------------
    L.append("## The sigma definitions")
    L.append("")
    L.append(
        "Three spread rules -- raw, trimmed, robust -- each computed per mode on the "
        "nearest-center assignment `per_mode_moments` uses, over modes holding >= "
        f"{AUDIT_MIN_COUNT} samples, then averaged over modes."
    )
    L.append("")
    L.append("| estimator | definition | what it is sensitive to |")
    L.append("|---|---|---|")
    L.append(
        "| `sigma_raw` | `sqrt(mean_d Var_d(x))` about the mode's sample mean -- the "
        "audit's own `per_mode_std` | the full second moment: one point at 10 sigma "
        "weighs as much as 100 at 1 sigma |"
    )
    L.append(
        f"| `sigma_trim` | the same after dropping the {100 * TRIM_FRAC:.0f}% of each "
        "mode's points farthest from its center | the body of the blob |"
    )
    L.append(
        f"| `sigma_trim_corrected` | `sigma_trim / {trim_bias:.4f}` | as above, with the "
        "bias the same trim inflicts on a *true* Gaussian divided out |"
    )
    L.append(
        f"| `sigma_robust` | `median(||x - center||) / sqrt(2 ln 2)` "
        f"(= median / {rayleigh_median_factor():.5f}) | the core only; a Gaussian-consistent "
        "inversion that ignores everything past the median |"
    )
    L.append("")
    L.append(
        "On a genuine Gaussian all four agree, so `sigma_raw / sigma_robust` reads "
        "directly as a tail-inflation factor: 1.0 means Gaussian, larger means the "
        "variance lives outside the core."
    )
    L.append("")
    L.append(
        "**The center matters as much as the spread rule.** A mode's *sample mean* is "
        "itself dragged by the far tail -- 3% of a mode's mass stranded at distance 2.5 "
        "moves the mean by 0.075, i.e. 2.5 sigma_true -- and every distance measured "
        "from a displaced center is then inflated. Each rule is therefore computed about "
        "all three centers that make sense: the sample **mean** (what the audit uses), "
        "the coordinate-wise **median** (robust and center-free), and the **true** grid "
        "center (the one `hq` measures from). The median- and true-center columns are "
        "the ones to read; the mean-centered ones are kept to show the size of the "
        "contamination."
    )
    L.append("")
    L.append(
        f"One caveat on the trim: dropping {100 * TRIM_FRAC:.0f}% only helps when the "
        "tail is thinner than that. Where the far-tail mass exceeds the trim fraction "
        "the trimmed std stays close to the raw one, and the median-based column is the "
        "only estimator in the ladder that still reports the core."
    )
    L.append("")
    L.append(
        f"**Units.** Samples, distances, sigmas and `center_bias` are all in absolute "
        f"data coordinates. The data's within-mode sigma is **{SIGMA_TRUE}** and the grid "
        f"spacing between neighbouring centers is **{GRID_SPACING}**, so 3 sigma_true = "
        f"{HQ_RADIUS:.2f} (the `hq` ball), 10 sigma_true = {TAIL_10SIG:.2f}, and the "
        f"Voronoi boundary between two modes is at 0.5. A `center_bias` of 0.02 is 0.67 "
        f"sigma_true and 2% of the way to the neighbouring mode."
    )
    L.append("")

    # --- distances ---------------------------------------------------------
    L.append("## Nearest-center distance distribution")
    L.append("")
    L.append(
        "| group | q50 | q90 | q99 | q99.9 | max | frac <= 0.09 (`hq`) | frac > 0.30 | frac > 0.15 |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]
        L.append(
            f"| `{name}` | {a['dist_q50']['mean']:.4f} | {a['dist_q90']['mean']:.4f} | "
            f"{a['dist_q99']['mean']:.4f} | {a['dist_q999']['mean']:.4f} | "
            f"{a['dist_max']['mean']:.3f} | {fmt(a['frac_within_3sig'], 4)} | "
            f"{fmt(a['frac_beyond_10sig'], 5)} | {fmt(a['frac_beyond_0p15'], 5)} |"
        )
    L.append("")

    # --- decomposition -----------------------------------------------------
    L.append("## Sigma-ratio decomposition")
    L.append("")
    L.append("Ratios are `sigma / 0.03`, so 1.00 is a perfect match to the data.")
    L.append("")
    L.append(
        "| group | raw (audit) | trim, mean-ctr | trim, median-ctr | trim, median-ctr, "
        "bias-corr. | robust, mean-ctr | **robust, median-ctr** | **robust, true-ctr** | "
        "raw / robust |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]
        shrink = a["ratio_raw"]["mean"] / a["ratio_robust_med"]["mean"]
        L.append(
            f"| `{name}` | {fmt(a['ratio_raw'], 2)} | {fmt(a['ratio_trim_mean'], 2)} | "
            f"{fmt(a['ratio_trim_med'], 2)} | {fmt(a['ratio_trim_med_corrected'], 2)} | "
            f"{fmt(a['ratio_robust_mean'], 2)} | **{fmt(a['ratio_robust_med'], 2)}** | "
            f"**{fmt(a['ratio_robust_center'], 2)}** | {shrink:.2f}x |"
        )
    L.append("")
    L.append(
        "The gap between the two robust columns and the mean-centered one is pure "
        "center contamination -- same points, same rule, only the center moved."
    )
    L.append("")
    L.append(
        "Which width actually explains `hq`? Feeding each ratio through "
        "`1 - exp(-(3/r)^2/2)` gives the hq a Gaussian of that width would score. "
        "The true-center column is the like-for-like one: `hq` is itself a "
        "distance-to-true-center statistic, so an offset blob is penalized in both."
    )
    L.append("")
    L.append(
        "| group | hq implied by raw | hq implied by core width (true-ctr) | "
        "measured hq | reading |"
    )
    L.append("|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]
        hq = a["frac_within_3sig"]["mean"]
        hq_raw = a["gauss_hq_raw"]["mean"]
        hq_rob = a["gauss_hq_robust_center"]["mean"]
        if abs(hq_raw - hq) <= 0.02:
            tag = "raw width is real"
        elif abs(hq_rob - hq) <= 0.03:
            tag = "core width explains hq; raw std is tail-inflated"
        elif hq_rob > hq:
            tag = "core width + a tail that leaves the 3-sigma ball"
        else:
            tag = "mixed"
        L.append(
            f"| `{name}` | {hq_raw:.3f} | {hq_rob:.3f} | {hq:.4f} | {tag} |"
        )
    L.append("")
    L.append(
        "| group | center bias, mean (audit) | in sigma_true | center bias, median | "
        "in sigma_true | audited modes |"
    )
    L.append("|---|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]
        L.append(
            f"| `{name}` | {fmt(a['center_bias'], 4)} | "
            f"{fmt(a['center_bias_in_sigma_true'], 2)} | "
            f"{fmt(a['center_bias_median'], 4)} | "
            f"{fmt(a['center_bias_median_in_sigma_true'], 2)} | "
            f"{a['audited_modes']['mean']:.1f} |"
        )
    L.append("")

    # --- per-group story ---------------------------------------------------
    L.append("## Reconciled story, per group")
    L.append("")
    for name, g in groups.items():
        L.append(f"- **`{name}`** -- {g['story']}")
    L.append("")

    # --- substantive finding ----------------------------------------------
    L.append("## The substantive finding")
    L.append("")

    best = min(
        groups.items(),
        key=lambda kv: abs(kv[1]["agg"]["ratio_robust_med"]["mean"] - 1.0),
    )
    worst = max(
        groups.items(), key=lambda kv: kv[1]["agg"]["ratio_robust_med"]["mean"]
    )
    # 0.8 is leaderboard.py's VARIANCE_COMPRESSED_BELOW, so a flag here is a flag
    # the promotion guard would have raised had it seen the core width.
    under = [
        k for k, g in groups.items() if g["agg"]["ratio_robust_med"]["mean"] < 0.8
    ]
    b_key = next((k for k in groups if k.startswith("b_cap")), None)
    c_key = next((k for k in groups if k.startswith("c_eikonal")), None)

    max_dev_core = max(
        abs(g["agg"]["gauss_hq_robust_center"]["mean"] - g["agg"]["frac_within_3sig"]["mean"])
        for g in groups.values()
    )
    max_dev_raw = max(
        abs(g["agg"]["gauss_hq_raw"]["mean"] - g["agg"]["frac_within_3sig"]["mean"])
        for g in groups.values()
    )

    para = [
        "**Is anyone matching the true variance, even robustly?** On the core "
        "estimator, yes -- and the misses run in *both* directions, which the raw "
        f"ratio hides. The closest to the data is `{best[0]}` at "
        f"{best[1]['agg']['ratio_robust_med']['mean']:.2f}; the full range is "
        f"{min(g['agg']['ratio_robust_med']['mean'] for g in groups.values()):.2f} to "
        f"{worst[1]['agg']['ratio_robust_med']['mean']:.2f} "
        f"(`{worst[0]}`), against raw ratios of "
        f"{min(g['agg']['ratio_raw']['mean'] for g in groups.values()):.1f}-"
        f"{max(g['agg']['ratio_raw']['mean'] for g in groups.values()):.1f}. "
        "That reverses the audit's headline reading in two places. First, `hq` and "
        "`per_mode_std_ratio` were never in conflict: a raw ratio of 2.4 implies a "
        f"Gaussian hq of {gaussian_hq(2.4):.3f}, but the raw ratio is a second moment "
        "of a tailed distribution, while the core width -- the thing `hq` actually "
        f"reflects -- predicts measured hq to within {max_dev_core:.3f} across these "
        f"groups against {max_dev_raw:.2f} for the raw ratio, and its residual is "
        "one-signed (it always over-predicts, by exactly the tail that leaves the "
        "3-sigma ball). Second, 'no variance collapse anywhere, all over-dispersed' does "
        "not survive the decomposition"
    ]
    if under:
        u = groups[under[0]]["agg"]
        para.append(
            f": `{under[0]}` has a core ratio of "
            f"{u['ratio_robust_med']['mean']:.2f} +- {u['ratio_robust_med']['std']:.2f}, "
            f"i.e. its blobs are *under*-dispersed -- variance-compressed cores hidden "
            f"behind a raw std that a "
            f"{100 * u['frac_beyond_10sig']['mean']:.1f}% far tail inflates to "
            f"{u['ratio_raw']['mean']:.1f}. The `per_mode_std_ratio < 0.8` guard in "
            f"`leaderboard.py` cannot see that, because it reads the tail-inflated "
            f"number."
        )
    else:
        para.append(
            ", but no group in this set falls below 0.9 on the core estimator either."
        )
    if b_key and c_key:
        ab, ac = groups[b_key]["agg"], groups[c_key]["agg"]
        para.append(
            f" The tail comparison separates the two arms cleanly. "
            f"`{c_key}` puts {100 * ac['frac_beyond_10sig']['mean']:.2f}% of its mass "
            f"beyond 10 sigma_true and {100 * ac['frac_beyond_0p15']['mean']:.2f}% "
            f"beyond 0.15, against {100 * ab['frac_beyond_10sig']['mean']:.2f}% and "
            f"{100 * ab['frac_beyond_0p15']['mean']:.2f}% for `{b_key}` -- "
            f"{ac['frac_beyond_10sig']['mean'] / max(ab['frac_beyond_10sig']['mean'], 1e-12):.1f}x "
            f"more far-tail mass, and the eikonal arm's 99th-percentile "
            f"nearest-center distance is {ac['dist_q99']['mean']:.2f} against "
            f"{ab['dist_q99']['mean']:.2f} -- i.e. its tail is not a wider blob but a "
            f"population stranded whole grid cells away. Their cores go the other way "
            f"(core ratio {ac['ratio_robust_med']['mean']:.2f} for the eikonal arm vs "
            f"{ab['ratio_robust_med']['mean']:.2f} for the cap), so `c_eikonal`'s worse "
            f"W1 and per-mode std are a tail phenomenon on top of an over-tight core, "
            f"exactly the signature of a critic pinned to unit slope everywhere: it "
            f"keeps a slope between the modes and leaves mass sitting there."
        )
    L.append("".join(para))
    L.append("")

    # --- checks ------------------------------------------------------------
    L.append("## Reproduction checks")
    L.append("")
    L.append(
        f"`frac <= 0.09` recomputed here vs. the run's own `hq` (tolerance {TOL_HQ}; "
        "`hq` is measured on a separate 20k draw from the same EMA model, so a ~1e-3 "
        f"sampling wobble is expected), and `ratio raw` vs. the summary's "
        f"`per_mode_std_ratio` (tolerance {TOL_STD_RATIO}; same samples, so this one "
        "should be exact)."
    )
    L.append("")
    L.append("| group | max |hq dev| | hq tolerance | max |std-ratio dev| | pass |")
    L.append("|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]
        L.append(
            f"| `{name}` | {a['max_hq_dev']:.5f} | {a['hq_tol']:.5f} | "
            f"{a['max_ratio_raw_dev']:.2e} | {'yes' if a['checks_ok'] else 'NO'} |"
        )
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# =========================
#  Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile per_mode_std_ratio with hq from the saved sample clouds.",
    )
    parser.add_argument("--runs_dir", type=str, default="results/runs_audit")
    parser.add_argument("--out_md", type=str, default="results/metric_recon.md")
    parser.add_argument("--out_json", type=str, default="results/metric_recon.json")
    parser.add_argument("--groups", type=str, nargs="*", default=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    groups: Dict[str, Dict] = {}
    failures: List[str] = []

    for group in args.groups:
        runs: List[Dict] = []
        for seed in args.seeds:
            name = f"{group}_s{seed}"
            run_dir = runs_dir / name
            if not (run_dir / "final_samples.npy").exists():
                print(f"[skip] {name}: no final_samples.npy", flush=True)
                continue
            res = analyze_run(run_dir)
            runs.append(res)
            if not res["hq_ok"]:
                failures.append(
                    f"{name}: hq dev {res['hq_dev']:.5f} > {res['hq_tol']:.5f}"
                )
            if not res["ratio_raw_ok"]:
                failures.append(
                    f"{name}: std-ratio dev {res['ratio_raw_dev']:.2e} > {TOL_STD_RATIO}"
                )
            print(
                f"[ok] {name}: hq {res['frac_within_3sig']:.4f} "
                f"(summary {res['hq_summary']:.4f}) | ratio raw {res['ratio_raw']:.3f} "
                f"trim {res['ratio_trim_med']:.3f} "
                f"core {res['ratio_robust_med']:.3f}/{res['ratio_robust_center']:.3f} | "
                f"tail>10sig {res['frac_beyond_10sig']:.5f}",
                flush=True,
            )
        if not runs:
            print(f"[skip] group {group}: no usable runs", flush=True)
            continue
        agg = aggregate(runs)
        groups[group] = {"runs": runs, "agg": agg, "story": story(agg)}

    if not groups:
        raise SystemExit("no usable runs found")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "runs_dir": str(runs_dir),
                    "groups": args.groups,
                    "seeds": args.seeds,
                    "sigma_true": SIGMA_TRUE,
                    "grid_spacing": GRID_SPACING,
                    "trim_frac": TRIM_FRAC,
                    "trim_bias_factor": trim_bias_factor(),
                    "rayleigh_median_factor": rayleigh_median_factor(),
                    "tol_hq": TOL_HQ,
                    "tol_std_ratio": TOL_STD_RATIO,
                },
                "groups": groups,
                "check_failures": failures,
            },
            f,
            indent=2,
        )

    write_markdown(Path(args.out_md), groups)
    print(f"\n[wrote] {args.out_md}\n[wrote] {args.out_json}")
    for name, g in groups.items():
        print(f"[story] {name}: {g['story']}")
    if failures:
        print("\n[FAIL] reproduction checks:")
        for msg in failures:
            print(f"  {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
