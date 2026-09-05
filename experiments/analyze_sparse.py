#!/usr/bin/env python
"""
analyze_sparse.py

Aggregate one stage of the sparse conditional mixed-output study.

Reads results/sparse/runs/<stage>/*/summary.json (and metrics.jsonl for the
curves), groups runs by name minus the `_s<seed>` suffix, and writes

    results/sparse/TABLE_<stage>.md      seed-aggregated table, one row per group
    results/sparse/plots/<stage>_curves.png   per-axis metric vs step, one line per group

and prints the table so the pipeline log carries it. With --config_manifest,
only the listed requests are aggregated, their recorded full configs must match,
and missing/mismatched runs produce a nonzero exit. Without a manifest, historical
directory scanning remains available and counts missing / half-written runs.

Usage:
    python experiments/analyze_sparse.py --stage baseline
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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


def requested_runs(stage: str, config_manifest: Optional[Path] = None):
    """Return (output path, expected full config), or historical directory entries."""
    stage_root = (RUNS_ROOT / stage).resolve()
    if config_manifest is None:
        return [(d, None) for d in sorted(stage_root.glob("*"))
                if d.is_dir() and not d.name.startswith(".")]

    # Share the runner's literal-default resolution without importing the trainer.
    if __package__:
        from .run_grid import load_config, trainer_defaults
    else:
        from run_grid import load_config, trainer_defaults
    paths = json.loads(Path(config_manifest).read_text())
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ValueError("config manifest must be a nonempty JSON list of config paths")
    defaults = trainer_defaults(str(Path(__file__).with_name("train_sparse.py")))
    requests = []
    seen = set()
    for path in paths:
        cfg = load_config(path, defaults)
        out_dir = Path(cfg["out_dir"]).resolve()
        if out_dir.parent != stage_root or out_dir.name.startswith("."):
            raise ValueError(f"manifest output is not a run in stage {stage!r}: {out_dir}")
        if out_dir in seen:
            raise ValueError(f"duplicate output in config manifest: {out_dir}")
        seen.add(out_dir)
        requests.append((out_dir, cfg))
    return requests


def load_stage(stage: str, config_manifest: Optional[Path] = None):
    runs: Dict[str, List[Dict]] = defaultdict(list)
    curves: Dict[str, List[List[Dict]]] = defaultdict(list)
    missing = 0
    for d, expected in requested_runs(stage, config_manifest):
        p = d / "summary.json"
        try:
            with open(p) as f:
                s = json.load(f)
            assert isinstance(s.get("final"), dict)
            if expected is not None:
                recorded = json.dumps(s.get("config"), sort_keys=True, allow_nan=False)
                requested = json.dumps(expected, sort_keys=True, allow_nan=False)
                if not s["final"] or recorded != requested:
                    raise ValueError("summary does not match the current requested config")
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        runs[group_name(d.name)].append(s)
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
        curves[group_name(d.name)].append(rows)
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
            row.append(fmt_cell([s["final"].get(key) for s in ss], fmt))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "bar = all seeds must hold: modes full & hq>=0.9 & cond>=0.95 & sym>=0.95 & sp@1e-2>=0.95 at the end; bar_step = median first crossing.",
              "cond = P(nearest mode has requested class); sep = per-class cloud separation vs real; sym = symbol matches the mode the real part landed in;",
              "joint = both; symKL = KL(true p(s|c) || emitted); sp@eps = P(|x|<eps on inactive dims); zero = exact 0.0 fraction; core = active-dim width ratio; ppur = particle class purity."]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config_manifest", type=Path, help="Analyze only this JSON list of config paths; fail on missing/mismatched runs.")
    args = ap.parse_args()
    try:
        runs, curves, missing = load_stage(args.stage, args.config_manifest)
    except (OSError, ValueError, TypeError) as exc:
        ap.error(str(exc))
    if not runs:
        print(f"[analyze_sparse] no complete runs for stage {args.stage} ({missing} incomplete)")
        return int(args.config_manifest is not None)
    md = table(args.stage, runs, missing)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / f"TABLE_{args.stage}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    p = plot_curves(args.stage, curves)
    print(f"[analyze_sparse] wrote {OUT_ROOT / f'TABLE_{args.stage}.md'} and {p}")
    return int(args.config_manifest is not None and missing > 0)


if __name__ == "__main__":
    sys.exit(main())
