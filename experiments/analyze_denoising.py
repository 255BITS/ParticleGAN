#!/usr/bin/env python
"""Collect certified runs, per-cell results, paired effects, and learning curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from experiments.run_grid import has_valid_summary


METRICS = ["joint_hq", "modes", "cond_acc", "conditional_sw1", "conditional_mode_tv",
           "per_mode_core_ratio", "per_mode_cov_eig_min_ratio", "per_mode_cov_eig_max_ratio",
           "tail_10sigma", "posterior_sw1"]


def stats(values):
    values = np.array([x for x in values if x is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None}


def formatted(s, scale=1):
    if s["mean"] is None:
        return "—"
    text = f"{s['mean'] * scale:.3f}"
    return text + (f" ± {s['sd'] * scale:.3f}" if s["sd"] is not None else "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, missing = [], []
    for path in json.loads(Path(args.manifest).read_text()):
        cfg = read_config(path)
        run_dir = Path(cfg["out_dir"])
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if summary is None or not has_valid_summary(str(run_dir), cfg, summary.get("provenance")):
            missing.append(path)
            continue
        runs.append(summary)
    # Never silently pool training code revisions in a factorial.
    sources = {json.dumps(r["provenance"], sort_keys=True) for r in runs}
    if len(sources) > 1:
        raise ValueError("mixed source provenance: analyze revisions separately")
    groups = defaultdict(list)
    for run in runs:
        cfg = {k: v for k, v in run["config"].items() if k not in ("seed", "out_dir")}
        groups[json.dumps(cfg, sort_keys=True)].append(run)
    cells = []
    for key, rows in groups.items():
        cfg = json.loads(key)
        cell = {"config": cfg, "seeds": [r["config"]["seed"] for r in rows],
                "metrics": {m: stats([r["final"].get(m) for r in rows]) for m in METRICS},
                "train_seconds": stats([r["train_seconds"] for r in rows]),
                "sampling_ms_per_1000": stats([r["final"]["sampling_ms_per_1000"] for r in rows])}
        cells.append(cell)
    cells.sort(key=lambda c: c["metrics"]["conditional_sw1"]["mean"])
    contrasts = []
    for factor, a, b in [("model", "gan", "ddgan"), ("d_mode", "concat", "ucd"),
                         ("prior", "gaussian", "learned"), ("prior", "fixed", "learned"),
                         ("noise", "gaussian", "fixed"), ("noise", "fixed", "learned"),
                         ("noise", "gaussian", "learned")]:
        matched = defaultdict(dict)
        for run in runs:
            cfg = run["config"]
            if cfg[factor] not in (a, b):
                continue
            ignored = {factor, "out_dir"}
            # A Gaussian prior never samples its initialization table. Allow a
            # learned-table correction to reuse those exact Gaussian controls.
            if factor == "prior" and a == "gaussian":
                ignored.add("num_particles")
            key = json.dumps({k: v for k, v in cfg.items() if k not in ignored}, sort_keys=True)
            if cfg[factor] in matched[key]:
                raise ValueError(f"ambiguous {factor} comparison: multiple runs for the same context")
            matched[key][cfg[factor]] = run
        effects = defaultdict(list)
        for key, pair in matched.items():
            if set(pair) != {a, b}:
                continue
            context = json.loads(key)
            context.pop("seed")
            if factor == "prior" and a == "gaussian":
                context["prior_particle_counts"] = {v: pair[v]["config"]["num_particles"] for v in (a, b)}
            effects[json.dumps(context, sort_keys=True)].append({
                m: pair[b]["final"][m] - pair[a]["final"][m]
                for m in ("joint_hq", "conditional_sw1", "conditional_mode_tv", "cond_acc")})
        for context, paired in effects.items():
            contrasts.append({"factor": factor, "difference": f"{b} minus {a}", "context": json.loads(context),
                              "metrics": {m: stats([r[m] for r in paired]) for m in paired[0]}})
    result = {"manifest": args.manifest, "completed": len(runs), "missing": missing, "cells": cells, "paired_effects": contrasts}
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Denoising GAN comparison", "", f"Completed {len(runs)} / {len(runs) + len(missing)} runs.", "",
             "Sorted by conditional sliced W1 for inspection, not an automatic winner selection. "
             "Report coverage, class fidelity, shape, tails, and posterior error together. ± denotes sample standard deviation across seeds.", "",
             "| Model | D | Latent | Step noise | Updates | T | Classes | n | Joint HQ | Modes | Class acc | Conditional SW1 | Mode TV | Core | Cov min / max | Tail | Posterior SW1 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        c, m = cell["config"], cell["metrics"]
        d_label = c["d_mode"] + ("/" + c.get("ucd_target", "class") if c["d_mode"] == "ucd" else "")
        vals = [c["model"], d_label, c["prior"], c["noise"] if c["model"] == "ddgan" else "N/A",
                str(c["steps"]), str(len(c["alpha_bar"]) - 1) if c["model"] == "ddgan" else "0",
                str(c["classes"]), str(len(cell["seeds"]))]
        vals.extend(formatted(m[k]) for k in ("joint_hq", "modes", "cond_acc", "conditional_sw1", "conditional_mode_tv", "per_mode_core_ratio"))
        vals.append(formatted(m["per_mode_cov_eig_min_ratio"]) + " / " + formatted(m["per_mode_cov_eig_max_ratio"]))
        vals.extend(formatted(m[k]) for k in ("tail_10sigma", "posterior_sw1"))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "Full configs, paired effects, timing, and missing runs are in `comparison.json`.", ""])
    if missing:
        lines.append("Incomplete runs: " + ", ".join(missing))
    (out / "TABLE.md").write_text("\n".join(lines).rstrip() + "\n")
    if runs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        palette = plt.get_cmap("tab20")
        for i, (key, rows) in enumerate(groups.items()):
            cfg = json.loads(key)
            label = (f"{cfg['model']} {cfg['d_mode']}/{cfg.get('ucd_target', 'class')} z:{cfg['prior']} "
                     f"noise:{cfg['noise'] if cfg['model'] == 'ddgan' else 'N/A'} "
                     f"T:{len(cfg['alpha_bar']) - 1 if cfg['model'] == 'ddgan' else 0} "
                     f"c:{cfg['classes']} updates:{cfg['steps']}")
            for j, row in enumerate(rows):
                history = [json.loads(s) for s in (Path(row["config"]["out_dir"]) / "metrics.jsonl").read_text().splitlines()]
                for ax, metric in zip(axes, ("joint_hq", "conditional_sw1", "conditional_mode_tv")):
                    ax.plot([h["train_seconds"] for h in history], [h[metric] for h in history],
                            color=palette(i % 20), alpha=.75, label=label if j == 0 else None)
                    ax.set(xlabel="Training seconds (includes concurrent GPU contention)", ylabel=metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
        fig.tight_layout(rect=(0, .22, 1, 1))
        fig.savefig(out / "curves.png", dpi=150)
        plt.close(fig)
    print(f"{len(runs)} certified runs; {len(missing)} missing. {out / 'TABLE.md'}")


if __name__ == "__main__":
    main()
