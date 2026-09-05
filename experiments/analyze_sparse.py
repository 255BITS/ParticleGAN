#!/usr/bin/env python
"""
analyze_sparse.py

Aggregate one stage of the sparse conditional mixed-output study.

Reads results/sparse/runs/<stage>/*/summary.json (and metrics.jsonl for the
curves), groups runs by name minus the `_s<seed>` suffix, and writes

    results/sparse/TABLE_<stage>.md      seed-aggregated table, one row per group
    results/sparse/plots/<stage>_curves.png   per-axis metric vs step, one line per group

and prints the table so the pipeline log carries it. Robust to missing /
half-written runs (they are counted, not raised).

Usage:
    python experiments/analyze_sparse.py --stage baseline
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS_ROOT = Path("results/sparse/runs")
OUT_ROOT = Path("results/sparse")

# (summary key, column header, format)
COLS = [
    ("modes", "modes", "{:.0f}"),
    ("hq", "hq", "{:.3f}"),
    ("cond_acc", "cond", "{:.3f}"),
    ("cond_sep_ratio", "sep", "{:.2f}"),
    ("sym_acc_mode", "sym", "{:.3f}"),
    ("joint_acc", "joint", "{:.3f}"),
    ("sym_cond_kl", "symKL", "{:.2f}"),
    ("sparse_prec@0p01", "sp@1e-2", "{:.3f}"),
    ("sparse_prec@0p001", "sp@1e-3", "{:.3f}"),
    ("sparse_rec", "rec", "{:.3f}"),
    ("exact_zero_frac", "zero", "{:.2f}"),
    ("smear", "smear", "{:.3f}"),
    ("core_ratio", "core", "{:.2f}"),
    ("sliced_w1", "w1", "{:.3f}"),
    ("ucd_acc_fake", "ucdF", "{:.2f}"),
    ("particle_class_purity", "ppur", "{:.2f}"),
]
CURVES = ["hq", "cond_acc", "sym_acc_mode", "joint_acc", "sparse_prec@0p01", "smear", "sliced_w1", "modes"]


def group_name(run_dir: str) -> str:
    return re.sub(r"_s\d+$", "", run_dir)


def legacy_x_only_gp(summary: Dict) -> bool:
    cfg = summary.get("config", {})
    return (cfg.get("gp_on_y") is False
            and cfg.get("arm") in ("e_interp", "g_interp_cap")
            and summary.get("implementation_versions", {}).get("x_only_gp", 1) < 2)


def load_stage(stage: str):
    runs: Dict[str, List[Dict]] = defaultdict(list)
    curves: Dict[str, List[List[Dict]]] = defaultdict(list)
    missing = 0
    for d in sorted((RUNS_ROOT / stage).glob("*")):
        if not d.is_dir():
            continue
        p = d / "summary.json"
        try:
            with open(p) as f:
                s = json.load(f)
            assert isinstance(s.get("final"), dict)
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        group = group_name(d.name)
        if legacy_x_only_gp(s):
            group += " [legacy x-only GP]"
        runs[group].append(s)
        rows = []
        try:
            with open(d / "metrics.jsonl") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
        except FileNotFoundError:
            pass
        curves[group].append(rows)
    return runs, curves, missing


def fmt_cell(vals: List[float], fmt: str) -> str:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "-"
    if v.size == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())}±{fmt.format(v.std())}"


def table(stage: str, runs: Dict[str, List[Dict]], missing: int) -> str:
    header = ["group", "n", "bar", "bar_step"] + [c[1] for c in COLS]
    lines = [f"# {stage}: {sum(len(v) for v in runs.values())} runs, {len(runs)} groups, {missing} incomplete", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    order = sorted(runs, key=lambda g: (-np.mean([s["bar_held"] for s in runs[g]]), -np.mean([s["final"]["joint_acc"] for s in runs[g]])))
    for g in order:
        ss = runs[g]
        n = len(ss)
        passed = sum(1 for s in ss if s["bar_held"])
        steps = [s["bar_step"] for s in ss if s["bar_step"] is not None]
        bar_step = f"{int(np.median(steps))}" + (f" ({len(steps)}/{n})" if len(steps) < n else "") if steps else "-"
        row = [g, str(n), f"{passed}/{n}", bar_step]
        for key, _, fmt in COLS:
            vals = [s["final"].get(key) for s in ss
                    if key != "particle_class_purity"
                    or s.get("implementation_versions", {}).get("particle_class_purity", 1) >= 2]
            row.append(fmt_cell(vals, fmt))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "bar = all seeds must hold: modes full & hq>=0.9 & cond>=0.95 & sym>=0.95 & sp@1e-2>=0.95 at the end; bar_step = median first crossing.",
              "cond = P(nearest mode has requested class); sep = per-class cloud separation vs real; sym = symbol matches the mode the real part landed in;",
              "joint = both; symKL = KL(true p(s|c) || emitted); sp@eps = P(|x|<eps on inactive dims); zero = exact 0.0 fraction; core = active-dim width ratio;",
              "ppur = mean dominant-class share per particle among correctly conditioned draws (>=4 draws per particle). Low purity can be healthy when G has a class embedding."]
    all_runs = [s for ss in runs.values() for s in ss]
    if any(s.get("implementation_versions", {}).get("particle_class_purity", 1) < 2 for s in all_runs):
        lines.append("Historical ppur values are omitted: older code paired samples with unrelated particle IDs; ppur uses only corrected runs, so its count may be smaller than n.")
    if any(legacy_x_only_gp(s) for s in all_runs):
        lines.append("Legacy x-only GP runs used endpoint penalties instead of interpolates. They are grouped separately and cannot establish the effect of excluding the y derivative; these results have not been recomputed.")
    return "\n".join(lines)


def plot_curves(stage: str, curves: Dict[str, List[List[Dict]]]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    groups = sorted(curves)
    cmap = plt.get_cmap("tab20")
    for ax, key in zip(axes.flat, CURVES):
        for gi, g in enumerate(groups):
            for si, rows in enumerate(curves[g]):
                xs = [r["step"] for r in rows if key in r]
                ys = [r[key] for r in rows if key in r]
                if xs:
                    ax.plot(xs, ys, color=cmap(gi % 20), alpha=0.8, lw=1.2, label=g if si == 0 else None)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{stage}: EMA read-out vs step (one line per seed)")
    fig.tight_layout()
    out = OUT_ROOT / "plots" / f"{stage}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    runs, curves, missing = load_stage(args.stage)
    if not runs:
        print(f"[analyze_sparse] no complete runs for stage {args.stage} ({missing} incomplete)")
        return
    md = table(args.stage, runs, missing)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / f"TABLE_{args.stage}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    p = plot_curves(args.stage, curves)
    print(f"[analyze_sparse] wrote {OUT_ROOT / f'TABLE_{args.stage}.md'} and {p}")


if __name__ == "__main__":
    main()
