#!/usr/bin/env python
"""
analyze.py

Aggregation, statistics and plots for the GAN gradient-regularizer study.

Reads every ``<runs_dir>/*/summary.json`` (plus the matching metrics.jsonl
timeseries) and produces, under ``results/``:

  * ``table.csv`` / ``TABLE.md`` -- one row per (arm, coeff, variant) with
    seed-aggregated final quality, collapse rate, convergence speed, the
    spectral radius of the local game Jacobian, and oscillation diagnostics.
  * ``bootstrap.md`` -- 10k-resample CI95 on the difference in median final
    exact W1 between each main-stage (arm, coeff) and the best a_r1r2
    (baseline) coefficient, with a win/match/lose verdict.
  * ``plots/w1_curves.png``, ``plots/gradnorm_evolution.png``,
    ``plots/spectral.png``, ``plots/lr_sens.png``.

Everything here is defensive by design: the grid is large, runs die, and
metrics.jsonl is appended live, so the last line is routinely half-written.
Missing runs, missing spectral blocks and truncated timeseries are skipped
and counted rather than raised.

Usage:

    python experiments/analyze.py
    python experiments/analyze.py --runs_dir /tmp/fixture_runs --out_dir /tmp/out
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker as mticker  # noqa: E402

BASELINE_ARM = "a_r1r2"
CONTROL_ARM = "f_none"
ARM_ORDER = ["a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp", "f_none"]

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20250821

# Run directory names look like:
#   a_r1r2_c0p02_s3
#   c_eikonal_c0p1_lr2p0_s1
#   d_asym_c0p1_lazy16_s5   /   d_asym_c0p1_c16th_s5
RUN_NAME_RE = re.compile(
    r"^(?P<arm>[a-f]_[a-z0-9]+)"
    r"_c(?P<coeff>[0-9p]+)"
    r"(?:_(?P<variant>lr[0-9p]+|lazy16|c16th))?"
    r"_s(?P<seed>\d+)$"
)

# ---------------------------------------------------------------------------
#  Plot styling (validated categorical palette; baseline is always black)
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

# Slots 1..5 of the reference categorical palette. a_r1r2 (the baseline) is
# drawn in black in every figure, so the hues start at the second arm.
SERIES_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ARM_COLOR = {
    "a_r1r2": INK,
    "b_cap": SERIES_HEX[0],
    "c_eikonal": SERIES_HEX[1],
    "d_asym": SERIES_HEX[2],
    "e_interp": SERIES_HEX[3],
    "f_none": SERIES_HEX[4],
}


def style_axes(ax) -> None:
    """Recessive grid and axes; ink-colored text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One training run: identity, final scalars, and its eval timeseries."""

    name: str
    arm: str
    coeff: float
    coeff_label: str
    variant: str
    seed: int
    summary: Dict
    series: Dict[str, np.ndarray] = field(default_factory=dict)

    # --- convenience accessors, all None-tolerant -------------------------

    def final(self, key: str) -> Optional[float]:
        return _num(self.summary.get("final", {}).get(key))

    def mid(self, key: str) -> Optional[float]:
        return _num(self.summary.get("mid", {}).get(key))

    def osc(self, key: str) -> Optional[float]:
        return _num(self.summary.get("oscillation", {}).get(key))

    @property
    def collapse_events(self) -> float:
        v = _num(self.summary.get("collapse_events"))
        return 0.0 if v is None else v

    @property
    def wall_clock(self) -> Optional[float]:
        return _num(self.summary.get("wall_clock_sec"))

    @property
    def total_steps(self) -> int:
        v = self.summary.get("steps")
        if v is None:
            v = self.summary.get("config", {}).get("total_steps")
        try:
            return int(v)
        except (TypeError, ValueError):
            return 7000

    def spectral(self) -> Dict:
        """The spectral block, or {} if it is absent or an error record."""
        block = self.summary.get("spectral")
        if not isinstance(block, dict) or "error" in block:
            return {}
        return block

    def dominant_modulus(self) -> Optional[float]:
        return _num(self.spectral().get("dominant_modulus"))

    def top_eigs(self) -> List[Tuple[float, float]]:
        eigs = self.spectral().get("top_eigs")
        out: List[Tuple[float, float]] = []
        if isinstance(eigs, (list, tuple)):
            for e in eigs:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    re_, im_ = _num(e[0]), _num(e[1])
                    if re_ is not None and im_ is not None:
                        out.append((re_, im_))
        return out


def _num(x) -> Optional[float]:
    """Coerce to a finite float, or None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_run_name(name: str) -> Optional[Dict]:
    """
    Decompose a run directory name into (arm, coeff, variant, seed).

    The directory name is authoritative rather than summary["config"]: the
    ``c16th`` lazy variant deliberately carries the *nominal* coefficient in
    its name while its config holds coeff/16, and grouping the pair together
    is the whole point of that stage.
    """
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    coeff_label = m.group("coeff")
    try:
        coeff = float(coeff_label.replace("p", ".").replace("m", "-"))
    except ValueError:
        return None
    return {
        "arm": m.group("arm"),
        "coeff": coeff,
        "coeff_label": coeff_label,
        "variant": m.group("variant") or "main",
        "seed": int(m.group("seed")),
    }


def load_metrics(path: Path) -> Dict[str, np.ndarray]:
    """
    Parse a metrics.jsonl into column arrays.

    Tolerates a truncated final line (the file is appended during training)
    and any individual row that fails to parse.
    """
    rows: List[Dict] = []
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Truncated / interleaved write: drop the row, keep going.
                    continue
                if isinstance(obj, dict) and "step" in obj:
                    rows.append(obj)
    except OSError:
        return {}
    if not rows:
        return {}

    rows.sort(key=lambda r: _num(r.get("step")) or 0.0)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    cols: Dict[str, np.ndarray] = {}
    for key in keys:
        vals = [_num(r.get(key)) for r in rows]
        cols[key] = np.array(
            [np.nan if v is None else v for v in vals], dtype=float
        )
    return cols


def load_runs(runs_dir: Path) -> Tuple[List[Run], Dict[str, int]]:
    """Load every run under runs_dir; returns (runs, skip counters)."""
    runs: List[Run] = []
    skips = {"no_summary": 0, "bad_summary": 0, "unparsed_name": 0, "no_metrics": 0}
    if not runs_dir.is_dir():
        return runs, skips

    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        ident = parse_run_name(d.name)
        summary_path = d / "summary.json"
        if not summary_path.is_file():
            skips["no_summary"] += 1
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            skips["bad_summary"] += 1
            continue
        if not isinstance(summary, dict) or not isinstance(summary.get("final"), dict):
            skips["bad_summary"] += 1
            continue

        if ident is None:
            # Fall back to the config block if the directory name is custom.
            cfg = summary.get("config", {})
            arm = cfg.get("arm")
            coeff = _num(cfg.get("coeff"))
            seed = cfg.get("seed")
            if arm is None or coeff is None or seed is None:
                skips["unparsed_name"] += 1
                continue
            ident = {
                "arm": str(arm),
                "coeff": coeff,
                "coeff_label": f"{coeff:g}".replace(".", "p"),
                "variant": "main",
                "seed": int(seed),
            }

        series = load_metrics(d / "metrics.jsonl")
        if not series:
            skips["no_metrics"] += 1
        runs.append(
            Run(
                name=d.name,
                arm=ident["arm"],
                coeff=ident["coeff"],
                coeff_label=ident["coeff_label"],
                variant=ident["variant"],
                seed=ident["seed"],
                summary=summary,
                series=series,
            )
        )
    return runs, skips


# ---------------------------------------------------------------------------
#  Aggregation helpers
# ---------------------------------------------------------------------------


def agg(vals: Sequence[Optional[float]]) -> Tuple[float, float, int]:
    """(mean, std, n) over the non-None entries; (nan, nan, 0) if empty."""
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def med(vals: Sequence[Optional[float]]) -> float:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def group_key(run: Run) -> Tuple[str, str, str]:
    return (run.variant, run.arm, run.coeff_label)


def group_runs(runs: Sequence[Run]) -> Dict[Tuple[str, str, str], List[Run]]:
    groups: Dict[Tuple[str, str, str], List[Run]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    for v in groups.values():
        v.sort(key=lambda r: r.seed)
    return dict(groups)


def steps_to_threshold(run: Run, sliced_threshold: float) -> Tuple[float, bool]:
    """
    First eval step at which sliced_w1 drops to or below ``sliced_threshold``.

    NOTE ON CALIBRATION. The headline quality metric is the *exact* W1
    (`final.w1_exact`), but the per-step timeseries only records the sliced
    estimate, and sliced W1 is a different (lower, projection-averaged)
    quantity -- there is no exact-W1 trace to threshold against. Rather than
    fitting a per-run sliced->exact calibration (overkill, and noisy at the
    100-eval resolution here), the convergence-speed statistic is defined
    entirely within the sliced series: the threshold is 2x the best (minimum)
    mean *final sliced* W1 over the baseline arm's coefficients, and it is
    compared against the sliced trace. It is therefore a speed statistic on
    the sliced scale, consistent across arms, and NOT directly comparable to
    the 2x-best-exact-W1 threshold used for the exact-W1 columns.

    Returns (steps, censored). A run that never reaches the threshold is
    censored: its value is total_steps and the caller marks it with '>'.
    """
    steps = run.series.get("step")
    w1 = run.series.get("sliced_w1")
    if steps is None or w1 is None or steps.size == 0:
        return float(run.total_steps), True
    ok = np.isfinite(w1) & np.isfinite(steps) & (w1 <= sliced_threshold)
    if not ok.any():
        return float(run.total_steps), True
    return float(steps[np.argmax(ok)]), False


def curve_matrix(
    runs: Sequence[Run], key: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Stack ``key`` across seeds onto a common step grid.

    Uses the steps common to every run in the group (runs stop at different
    points when one dies early), which keeps the mean/std bands honest.
    """
    usable = [
        r
        for r in runs
        if "step" in r.series and key in r.series and r.series["step"].size
    ]
    if not usable:
        return None, None
    common = None
    for r in usable:
        s = r.series["step"]
        v = r.series[key]
        good = set(s[np.isfinite(s) & np.isfinite(v)].tolist())
        common = good if common is None else (common & good)
    if not common:
        return None, None
    grid = np.array(sorted(common), dtype=float)
    rows = []
    for r in usable:
        s = r.series["step"]
        v = r.series[key]
        idx = {float(step): i for i, step in enumerate(s)}
        rows.append([v[idx[float(g)]] for g in grid])
    return grid, np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
#  Table
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "variant",
    "arm",
    "coeff",
    "n_seeds",
    "w1_exact_mean",
    "w1_exact_std",
    "w1_exact_median",
    "w2_exact_mean",
    "w2_exact_std",
    "w1_sliced_mean",
    "mode_recall_mean",
    "mode_recall_std",
    "modes_mean",
    "hq_mean",
    "hist_kl_mean",
    "hist_js_mean",
    "nll_mean",
    "nll_std",
    "collapse_rate",
    "collapse_events_mean",
    "steps_to_thresh_mean",
    "steps_to_thresh_std",
    "steps_to_thresh_censored",
    "dominant_modulus_mean",
    "dominant_modulus_std",
    "dominant_modulus_max",
    "w1_windowed_std_mean",
    "dloss_fft_peak_mean",
    "med_nr_mid_mean",
    "med_nf_mid_mean",
    "wall_clock_mean",
]


def build_rows(
    groups: Dict[Tuple[str, str, str], List[Run]], sliced_threshold: float
) -> List[Dict]:
    """One aggregated row per (variant, arm, coeff)."""
    rows: List[Dict] = []
    for (variant, arm, coeff_label), runs in groups.items():
        w1 = [r.final("w1_exact") for r in runs]
        w1_mean, w1_std, n = agg(w1)
        w2_mean, w2_std, _ = agg([r.final("w2_exact") for r in runs])
        rec_mean, rec_std, _ = agg([r.final("mode_recall") for r in runs])
        nll_mean, nll_std, _ = agg([r.final("nll") for r in runs])
        dm_vals = [r.dominant_modulus() for r in runs]
        dm_mean, dm_std, dm_n = agg(dm_vals)
        dm_finite = [v for v in dm_vals if v is not None]

        s2t = [steps_to_threshold(r, sliced_threshold) for r in runs]
        s2t_vals = [v for v, _ in s2t]
        s2t_mean, s2t_std, _ = agg(s2t_vals)
        censored = sum(1 for _, c in s2t if c)

        collapse = [r.collapse_events for r in runs]
        rows.append(
            {
                "variant": variant,
                "arm": arm,
                "coeff": float(coeff_label.replace("p", ".").replace("m", "-")),
                "n_seeds": n if n else len(runs),
                "w1_exact_mean": w1_mean,
                "w1_exact_std": w1_std,
                "w1_exact_median": med(w1),
                "w2_exact_mean": w2_mean,
                "w2_exact_std": w2_std,
                "w1_sliced_mean": agg([r.final("w1_sliced") for r in runs])[0],
                "mode_recall_mean": rec_mean,
                "mode_recall_std": rec_std,
                "modes_mean": agg([r.final("modes") for r in runs])[0],
                "hq_mean": agg([r.final("hq") for r in runs])[0],
                "hist_kl_mean": agg([r.final("hist_kl") for r in runs])[0],
                "hist_js_mean": agg([r.final("hist_js") for r in runs])[0],
                "nll_mean": nll_mean,
                "nll_std": nll_std,
                "collapse_rate": (
                    float(np.mean([1.0 if c > 0 else 0.0 for c in collapse]))
                    if collapse
                    else float("nan")
                ),
                "collapse_events_mean": float(np.mean(collapse)) if collapse else float("nan"),
                "steps_to_thresh_mean": s2t_mean,
                "steps_to_thresh_std": s2t_std,
                "steps_to_thresh_censored": censored,
                "dominant_modulus_mean": dm_mean,
                "dominant_modulus_std": dm_std,
                "dominant_modulus_max": max(dm_finite) if dm_finite else float("nan"),
                "w1_windowed_std_mean": agg([r.osc("w1_windowed_std") for r in runs])[0],
                "dloss_fft_peak_mean": agg([r.osc("dloss_fft_peak") for r in runs])[0],
                "med_nr_mid_mean": agg([r.mid("med_nr") for r in runs])[0],
                "med_nf_mid_mean": agg([r.mid("med_nf") for r in runs])[0],
                "wall_clock_mean": agg([r.wall_clock for r in runs])[0],
                "_spectral_n": dm_n,
            }
        )

    # Sort by median final exact W1 *within* each variant group.
    variant_order = {"main": 0, "lazy16": 1, "c16th": 2}
    rows.sort(
        key=lambda r: (
            variant_order.get(r["variant"], 3),
            r["variant"],
            _nan_last(r["w1_exact_median"]),
        )
    )
    return rows


def _nan_last(x: float) -> float:
    return float("inf") if (x is None or not math.isfinite(x)) else x


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_pm(mean: float, std: float, nd: int = 4) -> str:
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(std):
        return fmt(mean, nd)
    return f"{fmt(mean, nd)} ± {fmt(std, nd)}"


def write_table(
    rows: List[Dict],
    out_dir: Path,
    skips: Dict[str, int],
    thresholds: Dict[str, float],
    n_runs: int,
) -> None:
    """Write table.csv and TABLE.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if isinstance(row.get(k), float) and not math.isfinite(row[k])
                        else row.get(k)
                    )
                    for k in CSV_COLUMNS
                }
            )

    header = (
        "| arm | coeff | n | final W1 (exact) | W2 (exact) | mode recall | modes | hq | "
        "NLL | collapse rate (mean ev) | steps→thresh | dom. modulus (max) | "
        "W1 win-std | dloss FFT pk | med_nr / med_nf @mid | wall (s) |"
    )
    sep = "|" + "---|" * 16

    lines: List[str] = []
    lines.append("# Regularizer study -- aggregate results\n")
    lines.append(
        f"{n_runs} runs loaded from disk. "
        f"Skipped: {skips.get('no_summary', 0)} without summary.json, "
        f"{skips.get('bad_summary', 0)} with unreadable/incomplete summary.json, "
        f"{skips.get('unparsed_name', 0)} with unparseable run names, "
        f"{skips.get('no_metrics', 0)} loaded but missing/empty metrics.jsonl "
        "(final-only; their convergence-speed entries are censored).\n"
    )

    by_variant: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    variant_titles = {
        "main": "Main grid (lr_mult 1.0, every-step penalty)",
        "lazy16": "Lazy variant: penalty every 16 steps at the nominal coeff (16x internal rescale)",
        "c16th": "Lazy control: penalty every step at coeff/16 (same nominal coeff in the label)",
    }
    section_order = {"main": 0, "lazy16": 1, "c16th": 2}
    for variant in sorted(by_variant, key=lambda v: (section_order.get(v, 3), v)):
        title = variant_titles.get(variant, f"LR sensitivity: lr_mult = {variant[2:].replace('p', '.')}")
        lines.append(f"\n## {title}\n")
        lines.append(header)
        lines.append(sep)
        for row in by_variant[variant]:
            s2t = fmt(row["steps_to_thresh_mean"], 0)
            if row["steps_to_thresh_censored"]:
                s2t = f">{s2t}"
            s2t = f"{s2t} ± {fmt(row['steps_to_thresh_std'], 0)}"
            if row["steps_to_thresh_censored"]:
                s2t += f" ({row['steps_to_thresh_censored']} censored)"
            dom = fmt_pm(row["dominant_modulus_mean"], row["dominant_modulus_std"], 3)
            if math.isfinite(row["dominant_modulus_max"]):
                dom += f" (max {fmt(row['dominant_modulus_max'], 3)})"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["arm"],
                        fmt(row["coeff"], 4).rstrip("0").rstrip(".") or "0",
                        str(row["n_seeds"]),
                        fmt_pm(row["w1_exact_mean"], row["w1_exact_std"]),
                        fmt_pm(row["w2_exact_mean"], row["w2_exact_std"]),
                        fmt_pm(row["mode_recall_mean"], row["mode_recall_std"], 3),
                        fmt(row["modes_mean"], 1),
                        fmt(row["hq_mean"], 3),
                        fmt_pm(row["nll_mean"], row["nll_std"], 3),
                        f"{fmt(row['collapse_rate'], 2)} ({fmt(row['collapse_events_mean'], 2)})",
                        s2t,
                        dom,
                        fmt(row["w1_windowed_std_mean"], 4),
                        fmt(row["dloss_fft_peak_mean"], 3),
                        f"{fmt(row['med_nr_mid_mean'], 3)} / {fmt(row['med_nf_mid_mean'], 3)}",
                        fmt(row["wall_clock_mean"], 0),
                    ]
                )
                + " |"
            )

    lines.append("\n## Footnotes\n")
    lines.append(
        "1. **Rows are sorted by median final exact W1 within each variant group** "
        "(best first). `± ` is the population std over seeds.\n"
    )
    lines.append(
        "2. **steps→thresh is measured on the *sliced* W1 series, against a sliced "
        "threshold.** The headline quality number is the exact W1, but metrics.jsonl "
        "only carries `sliced_w1` per eval step -- there is no exact-W1 trace to "
        "threshold. Rather than fitting a per-run sliced→exact calibration, the speed "
        "statistic is defined entirely on the sliced scale: "
        f"`sliced_threshold = 2 x min_coeff(mean final w1_sliced of {BASELINE_ARM}) = "
        f"{fmt(thresholds['sliced'], 5)}`. The corresponding exact-W1 threshold "
        f"(`2 x min_coeff(mean final w1_exact of {BASELINE_ARM})` = "
        f"{fmt(thresholds['exact'], 5)}) is reported for reference but is NOT used for "
        "the timeseries. The two are on different scales and must not be compared to "
        "each other.\n"
    )
    lines.append(
        "3. Seeds that never reach the threshold are counted as `total_steps` and the "
        "cell is prefixed with `>`; the count of such censored seeds is shown in "
        "parentheses. Means over censored groups are lower bounds.\n"
    )
    lines.append(
        "4. `collapse rate` is the fraction of seeds with `collapse_events > 0`; the "
        "parenthesized number is the mean event count per seed.\n"
    )
    lines.append(
        "5. `dom. modulus` is the modulus of the dominant eigenvalue of the local game "
        "Jacobian; > 1 indicates a locally divergent (non-contractive) equilibrium. "
        "Runs whose `spectral` block is missing or an error record are omitted from "
        "that column only.\n"
    )
    lines.append(
        "6. `med_nr / med_nf @mid` are the median discriminator gradient norms at real "
        "and fake samples, taken at the mid-training checkpoint recorded in "
        "`summary.mid`.\n"
    )

    with open(out_dir / "TABLE.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
#  Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_diff(
    arm_vals: Sequence[float],
    base_vals: Sequence[float],
    rng: np.random.Generator,
    n: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    """
    CI95 on median(arm) - median(baseline), resampling seeds within each group.

    Returns (point_estimate, lo, hi).
    """
    a = np.asarray([v for v in arm_vals if v is not None and math.isfinite(v)], float)
    b = np.asarray([v for v in base_vals if v is not None and math.isfinite(v)], float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(a) - np.median(b))
    ia = rng.integers(0, a.size, size=(n, a.size))
    ib = rng.integers(0, b.size, size=(n, b.size))
    draws = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def verdict(lo: float, hi: float, collapse: float, base_collapse: float) -> str:
    """
    An arm 'wins' only if the CI excludes 0 in its favour (lower W1 is better)
    AND it does not collapse more often than the baseline; it 'matches' if the
    CI includes 0 and its collapse rate is no worse.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "no data"
    collapse_ok = (
        math.isfinite(collapse)
        and math.isfinite(base_collapse)
        and collapse <= base_collapse + 1e-12
    )
    if hi < 0:
        return "**wins**" if collapse_ok else "lower W1 but collapses more"
    if lo > 0:
        return "loses"
    return "matches" if collapse_ok else "matches on W1 but collapses more"


def write_bootstrap(
    groups: Dict[Tuple[str, str, str], List[Run]],
    baseline_key: Optional[Tuple[str, str, str]],
    out_dir: Path,
) -> None:
    """Write bootstrap.md for the main-stage groups."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bootstrap.md"

    if baseline_key is None:
        with open(path, "w") as f:
            f.write(
                "# Bootstrap comparison vs. baseline\n\n"
                f"No `{BASELINE_ARM}` main-stage runs were found, so no baseline "
                "could be established. Nothing to compare.\n"
            )
        return

    base_runs = groups[baseline_key]
    base_vals = [r.final("w1_exact") for r in base_runs]
    base_collapse = float(
        np.mean([1.0 if r.collapse_events > 0 else 0.0 for r in base_runs])
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    main_keys = sorted(
        (k for k in groups if k[0] == "main"),
        key=lambda k: (ARM_ORDER.index(k[1]) if k[1] in ARM_ORDER else 99, k[2]),
    )

    lines = [
        "# Bootstrap comparison vs. the best baseline coefficient\n",
        f"Statistic: median final `w1_exact` of each main-stage (arm, coeff) minus the "
        f"median final `w1_exact` of the baseline `{baseline_key[1]}` @ coeff "
        f"`{baseline_key[2].replace('p', '.')}` (the {BASELINE_ARM} coefficient with the "
        "lowest mean final exact W1).\n",
        f"CI95 from {N_BOOTSTRAP:,} bootstrap resamples of the seeds *within each group* "
        f"(numpy default_rng, seed {BOOTSTRAP_SEED}). Negative differences favour the arm "
        "(lower W1 is better).\n",
        "**Verdict rule.** An arm *wins* only if its CI95 excludes 0 in its favour AND "
        f"its collapse rate is <= the baseline's ({fmt(base_collapse, 2)}). It *matches* "
        "if the CI includes 0 and its collapse rate is no worse. Otherwise it loses, or "
        "is flagged as buying W1 at the cost of stability.\n",
        "| arm | coeff | n | median W1 | Δ median W1 | CI95 | collapse rate | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for key in main_keys:
        runs = groups[key]
        vals = [r.final("w1_exact") for r in runs]
        point, lo, hi = bootstrap_diff(vals, base_vals, rng)
        collapse = float(
            np.mean([1.0 if r.collapse_events > 0 else 0.0 for r in runs])
        )
        tag = " (baseline)" if key == baseline_key else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    key[1] + tag,
                    key[2].replace("p", "."),
                    str(len(runs)),
                    fmt(med(vals)),
                    fmt(point),
                    f"[{fmt(lo)}, {fmt(hi)}]",
                    fmt(collapse, 2),
                    verdict(lo, hi, collapse, base_collapse),
                ]
            )
            + " |"
        )

    lines.append(
        "\nNote: the baseline row compares the baseline group against itself; its "
        "point estimate is 0 by construction and its CI reflects only seed noise, "
        "which is a useful scale reference for the other rows.\n"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------


def best_coeff_for_arm(
    groups: Dict[Tuple[str, str, str], List[Run]], arm: str, variant: str = "main"
) -> Optional[Tuple[str, str, str]]:
    """The (variant, arm, coeff) key with the lowest mean final exact W1."""
    best, best_val = None, float("inf")
    for key, runs in groups.items():
        if key[0] != variant or key[1] != arm:
            continue
        m, _, n = agg([r.final("w1_exact") for r in runs])
        if n and math.isfinite(m) and m < best_val:
            best, best_val = key, m
    return best


def plot_w1_curves(
    groups: Dict[Tuple[str, str, str], List[Run]], plots_dir: Path
) -> Optional[Path]:
    """Sliced W1 vs step, log-y, one panel per coefficient, one line per arm."""
    main = {k: v for k, v in groups.items() if k[0] == "main"}
    if not main:
        return None
    coeffs = sorted(
        {k[2] for k in main if k[1] != CONTROL_ARM},
        key=lambda c: float(c.replace("p", ".")),
    )
    if not coeffs:
        coeffs = sorted({k[2] for k in main})

    n = len(coeffs)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)

    handles: Dict[str, object] = {}
    for i, coeff in enumerate(coeffs):
        ax = axes[i // ncols][i % ncols]
        style_axes(ax)
        arms = [a for a in ARM_ORDER if (("main", a, coeff) in main or a == CONTROL_ARM)]
        for arm in arms:
            key = ("main", arm, coeff)
            if key not in main:
                # The unregularized control has a single coeff (0.0); repeat it
                # in every panel as a shared reference line.
                if arm == CONTROL_ARM:
                    ctrl = [k for k in main if k[1] == CONTROL_ARM]
                    if not ctrl:
                        continue
                    key = ctrl[0]
                else:
                    continue
            grid, mat = curve_matrix(main[key], "sliced_w1")
            if grid is None:
                continue
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            color = ARM_COLOR.get(arm, INK_SECONDARY)
            ls = "--" if arm == CONTROL_ARM else "-"
            ax.fill_between(
                grid, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            (line,) = ax.plot(grid, mean, color=color, linewidth=2.0, linestyle=ls)
            handles.setdefault(arm, line)
        ax.set_yscale("log")
        ax.set_title(f"coeff = {coeff.replace('p', '.')}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("sliced W1")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    order = [a for a in ARM_ORDER if a in handles]
    fig.legend(
        [handles[a] for a in order],
        [a + (" (baseline)" if a == BASELINE_ARM else "") for a in order],
        loc="lower center",
        ncol=min(len(order), 6),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle("Sliced W1 vs training step, by penalty coefficient", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = plots_dir / "w1_curves.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_gradnorm(
    groups: Dict[Tuple[str, str, str], List[Run]], plots_dir: Path
) -> Optional[Path]:
    """
    H2 mechanism plot: D gradient norms at real and fake samples over training,
    baseline vs. the eikonal arm, each at its own best coefficient.
    """
    keys = []
    for arm in (BASELINE_ARM, "c_eikonal"):
        key = best_coeff_for_arm(groups, arm)
        if key is not None:
            keys.append(key)
    if not keys:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, label in zip(
        axes[0], ("med_nr", "med_nf"), ("median ||∇D|| at real", "median ||∇D|| at fake")
    ):
        style_axes(ax)
        for key in keys:
            grid, mat = curve_matrix(groups[key], metric)
            if grid is None:
                continue
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            color = ARM_COLOR.get(key[1], INK_SECONDARY)
            ax.fill_between(
                grid, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            ax.plot(
                grid,
                mean,
                color=color,
                linewidth=2.0,
                label=f"{key[1]} @ {key[2].replace('p', '.')}",
            )
        # Reference levels: 0 (the R1/R2 zero-centred target) and 1 (the
        # eikonal / 1-Lipschitz target).
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.suptitle(
        "H2 mechanism: gradient-norm target reached by each penalty "
        "(dotted = 0, dashed = 1)",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plots_dir / "gradnorm_evolution.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_spectral(
    groups: Dict[Tuple[str, str, str], List[Run]],
    rows: List[Dict],
    plots_dir: Path,
) -> Optional[Path]:
    """Eigenvalues in the complex plane, plus dominant modulus per (arm, coeff)."""
    scatter_keys = []
    for arm in (BASELINE_ARM, "c_eikonal", "d_asym"):
        key = best_coeff_for_arm(groups, arm)
        if key is not None:
            scatter_keys.append(key)
    # All-pairs colour separation caps the scatter at three series.
    scatter_keys = scatter_keys[:3]

    bar_rows = [
        r
        for r in rows
        if r["variant"] == "main" and math.isfinite(r["dominant_modulus_mean"])
    ]
    if not scatter_keys and not bar_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0][0]
    style_axes(ax)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=INK_MUTED, linewidth=1.2, linestyle="--")
    any_eig = False
    for key in scatter_keys:
        pts = [e for r in groups[key] for e in r.top_eigs()]
        if not pts:
            continue
        any_eig = True
        arr = np.array(pts, dtype=float)
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            s=42,
            color=ARM_COLOR.get(key[1], INK_SECONDARY),
            edgecolors=SURFACE,
            linewidths=1.0,
            alpha=0.85,
            label=f"{key[1]} @ {key[2].replace('p', '.')}",
            zorder=3,
        )
    ax.axhline(0.0, color=GRID, linewidth=1.0)
    ax.axvline(0.0, color=GRID, linewidth=1.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_title("Top game-Jacobian eigenvalues (unit circle dashed)", fontsize=10)
    if any_eig:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    else:
        ax.text(
            0.5,
            0.5,
            "no spectral data",
            transform=ax.transAxes,
            ha="center",
            color=INK_MUTED,
            fontsize=10,
        )

    ax = axes[0][1]
    style_axes(ax)
    if bar_rows:
        bar_rows = sorted(
            bar_rows,
            key=lambda r: (
                ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99,
                r["coeff"],
            ),
        )
        xs = np.arange(len(bar_rows), dtype=float)
        vals = [r["dominant_modulus_mean"] for r in bar_rows]
        errs = [
            0.0 if not math.isfinite(r["dominant_modulus_std"]) else r["dominant_modulus_std"]
            for r in bar_rows
        ]
        colors = [ARM_COLOR.get(r["arm"], INK_SECONDARY) for r in bar_rows]
        ax.bar(
            xs,
            vals,
            width=0.72,
            color=colors,
            edgecolor=SURFACE,
            linewidth=2.0,
            zorder=2,
        )
        ax.errorbar(
            xs, vals, yerr=errs, fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=3
        )
        ax.axhline(1.0, color=INK, linewidth=1.4, linestyle="--", zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{r['arm']}\n{r['coeff']:g}" for r in bar_rows],
            rotation=0,
            fontsize=7,
        )
        ax.set_ylabel("dominant |λ|")
        ax.set_title("Dominant modulus per (arm, coeff), mean ± std", fontsize=10)
    else:
        ax.text(
            0.5, 0.5, "no spectral data", transform=ax.transAxes, ha="center",
            color=INK_MUTED, fontsize=10,
        )

    fig.tight_layout()
    path = plots_dir / "spectral.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_lr_sens(
    groups: Dict[Tuple[str, str, str], List[Run]], plots_dir: Path
) -> Optional[Path]:
    """Final exact W1 vs lr_mult per arm; only drawn if lr_sens runs exist."""
    lr_keys = [k for k in groups if k[0].startswith("lr")]
    if not lr_keys:
        return None

    # arm -> {lr_mult -> (mean, std)}. The lr_mult 1.0 point comes from the
    # matching main-grid coefficient.
    per_arm: Dict[str, Dict[float, Tuple[float, float]]] = defaultdict(dict)
    coeff_of_arm: Dict[str, str] = {}
    for key in lr_keys:
        variant, arm, coeff = key
        lr_mult = float(variant[2:].replace("p", "."))
        m, s, n = agg([r.final("w1_exact") for r in groups[key]])
        if n:
            per_arm[arm][lr_mult] = (m, s)
            coeff_of_arm[arm] = coeff
    for arm, coeff in coeff_of_arm.items():
        base_key = ("main", arm, coeff)
        if base_key in groups:
            m, s, n = agg([r.final("w1_exact") for r in groups[base_key]])
            if n:
                per_arm[arm][1.0] = (m, s)
    if not per_arm:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for arm in [a for a in ARM_ORDER if a in per_arm]:
        xs = sorted(per_arm[arm])
        ys = [per_arm[arm][x][0] for x in xs]
        es = [0.0 if not math.isfinite(per_arm[arm][x][1]) else per_arm[arm][x][1] for x in xs]
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            color=ARM_COLOR.get(arm, INK_SECONDARY),
            linewidth=2.0,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            capsize=3,
            label=f"{arm} @ {coeff_of_arm.get(arm, '?').replace('p', '.')}",
        )
    # Log x so 0.5 / 1.0 / 2.0 are evenly spaced; suppress the log minor
    # ticks, which otherwise print '6 x 10^-1' on top of the real labels.
    ax.set_xscale("log")
    ax.set_xticks(sorted({x for arm in per_arm for x in per_arm[arm]}))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("lr_mult")
    ax.set_ylabel("final W1 (exact)")
    ax.set_title("Learning-rate sensitivity", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    path = plots_dir / "lr_sens.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def compute_thresholds(groups: Dict[Tuple[str, str, str], List[Run]]) -> Dict[str, float]:
    """
    Thresholds derived from the baseline arm's best coefficient.

    Two are computed: one on the exact W1 scale (reported for reference) and
    one on the sliced W1 scale (the one actually applied to the timeseries).
    See `steps_to_threshold` for why they are kept separate.
    """
    best_exact, best_sliced = float("inf"), float("inf")
    for key, runs in groups.items():
        if key[0] != "main" or key[1] != BASELINE_ARM:
            continue
        m_e, _, n_e = agg([r.final("w1_exact") for r in runs])
        if n_e and math.isfinite(m_e):
            best_exact = min(best_exact, m_e)
        m_s, _, n_s = agg([r.final("w1_sliced") for r in runs])
        if n_s and math.isfinite(m_s):
            best_sliced = min(best_sliced, m_s)
    return {
        "exact": 2.0 * best_exact if math.isfinite(best_exact) else float("inf"),
        "sliced": 2.0 * best_sliced if math.isfinite(best_sliced) else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate, test and plot the regularizer study results."
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="results/runs",
        help="Directory holding one subdirectory per run (default: results/runs).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Where to write table.csv / TABLE.md / bootstrap.md / plots (default: results).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"

    runs, skips = load_runs(runs_dir)
    print(f"[analyze] loaded {len(runs)} runs from {runs_dir}")
    for k, v in skips.items():
        if v:
            print(f"[analyze]   skipped {v} ({k})")
    if not runs:
        print("[analyze] nothing to analyze")
        return

    groups = group_runs(runs)
    thresholds = compute_thresholds(groups)
    print(
        f"[analyze] thresholds: exact {thresholds['exact']:.5g} (reference), "
        f"sliced {thresholds['sliced']:.5g} (applied to the timeseries)"
    )

    rows = build_rows(groups, thresholds["sliced"])
    write_table(rows, out_dir, skips, thresholds, len(runs))
    print(f"[analyze] wrote {out_dir / 'table.csv'} and {out_dir / 'TABLE.md'} "
          f"({len(rows)} groups)")

    baseline_key = best_coeff_for_arm(groups, BASELINE_ARM)
    write_bootstrap(groups, baseline_key, out_dir)
    print(f"[analyze] wrote {out_dir / 'bootstrap.md'} (baseline {baseline_key})")

    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for fn in (
            lambda: plot_w1_curves(groups, plots_dir),
            lambda: plot_gradnorm(groups, plots_dir),
            lambda: plot_spectral(groups, rows, plots_dir),
            lambda: plot_lr_sens(groups, plots_dir),
        ):
            try:
                path = fn()
            except Exception as exc:  # noqa: BLE001 - a bad figure must not kill the run
                print(f"[analyze] WARN plot failed: {exc}")
                continue
            if path is not None:
                print(f"[analyze] wrote {path}")


if __name__ == "__main__":
    main()
