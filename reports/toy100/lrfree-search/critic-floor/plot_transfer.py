#!/usr/bin/env python3
"""Plot frozen native100 transfer trajectories from saved runner JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "runs"
OLD = "h2-prior-handoff"
NEW = "h2-handoff-critic-floor"


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric(values):
    return np.array([np.nan if value is None else float(value) for value in values])


def plot_task(task: str) -> dict:
    runs = {}
    for label, run in [("prior handoff", OLD), ("critic floor", NEW)]:
        base = RUNS / run / task
        runs[label] = {
            "metrics": rows(base / "metrics.jsonl"),
            "rates": rows(base / "rates.jsonl"),
        }
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    ax = axs[0, 0]
    for label, style in [("prior handoff", "--"), ("critic floor", "-")]:
        rates = runs[label]["rates"]
        if not rates:
            continue
        steps = [row["step"] for row in rates]
        for name, idx, color in [("G", (0, 0), "tab:blue"), ("prior", (0, 1), "tab:orange"), ("D", (1, 0), "tab:green")]:
            ax.plot(steps, [row["lr"][idx[0]][idx[1]] for row in rates],
                    style, color=color, lw=1.1, label=f"{name} {label}")
    ax.set_yscale("log")
    ax.set_title("Applied learning rates")
    ax.legend(fontsize=7, ncol=2)

    fields = [
        (axs[0, 1], "precision", "Noisy precision", 0.97, ">="),
        (axs[0, 2], "acc_center_rms_sigma", "Center RMS / sigma", 0.20, "<="),
        (axs[0, 3], "min_cov_eig_ratio", "Minimum covariance eigenvalue ratio", 0.40, ">="),
        (axs[1, 0], "max_cov_eig_ratio", "Maximum covariance eigenvalue ratio", 1.70, "<="),
        (axs[1, 1], "acc_mass_tv", "Mode mass TV", 0.06, "<="),
        (axs[1, 2], "acc_abs_cov_trace_bias", "Absolute covariance trace bias", 0.10, "<="),
        (axs[1, 3], "acc_radial_ks", "Radial KS", 0.04, "<="),
    ]
    for panel, key, title, limit, direction in fields:
        for label, style, color in [("prior handoff", "--", "#8b8b8b"),
                                    ("critic floor", "-", "#0072b2")]:
            metrics = runs[label]["metrics"]
            panel.plot([row["step"] for row in metrics], numeric(row.get(key) for row in metrics),
                       style, color=color, marker="." if len(metrics) < 50 else None,
                       lw=1.5, ms=4, label=label)
        panel.axhline(limit, color="#d55e00", linestyle=":", lw=1.1,
                      label=f"gate {direction} {limit:g}")
        panel.set_title(title)
        panel.legend(fontsize=7)
    for panel in axs.flat:
        panel.grid(alpha=0.25)
        panel.set_xlim(0, 7000)
        panel.set_xlabel("Training update")
    latest = runs["critic floor"]["metrics"][-1]["step"] if runs["critic floor"]["metrics"] else 0
    fig.suptitle(f"{task}: critic floor vs prior handoff (new scored through update {latest})", fontsize=14)
    target = ROOT / f"{task}-trajectory.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    old_map = {row["step"]: row for row in runs["prior handoff"]["metrics"]}
    new_map = {row["step"]: row for row in runs["critic floor"]["metrics"]}
    common = sorted(old_map.keys() & new_map.keys())
    first_changed = next((step for step in common if any(
        old_map[step].get(key) != new_map[step].get(key)
        for key in ["precision", "acc_center_rms_sigma", "min_cov_eig_ratio", "max_cov_eig_ratio", "acc_abs_cov_trace_bias", "acc_radial_ks"]
    )), None)
    old_rates = {row["step"]: row for row in runs["prior handoff"]["rates"]}
    new_rates = {row["step"]: row for row in runs["critic floor"]["rates"]}
    first_d_changed = next((step for step in sorted(old_rates.keys() & new_rates.keys())
                            if not np.isclose(old_rates[step]["lr"][1][0],
                                              new_rates[step]["lr"][1][0], rtol=1e-7, atol=1e-10)), None)
    return {
        "task": task,
        "new_latest_scored_step": latest,
        "new_latest_applied_rate_step": runs["critic floor"]["rates"][-1]["step"] if runs["critic floor"]["rates"] else 0,
        "first_changed_d_rate_step": first_d_changed,
        "first_changed_scored_step": first_changed,
        "plot": str(target),
        "latest_common": {
            "step": common[-1] if common else None,
            "old": {key: old_map[common[-1]].get(key) for key in ["precision", "acc_center_rms_sigma", "min_cov_eig_ratio", "max_cov_eig_ratio", "acc_mass_tv", "acc_abs_cov_trace_bias", "acc_radial_ks"]} if common else {},
            "new": {key: new_map[common[-1]].get(key) for key in ["precision", "acc_center_rms_sigma", "min_cov_eig_ratio", "max_cov_eig_ratio", "acc_mass_tv", "acc_abs_cov_trace_bias", "acc_radial_ks"]} if common else {},
        },
    }


if __name__ == "__main__":
    summary = {task: plot_task(task) for task in ["rotated100", "staggered100"]}
    (ROOT / "trajectory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for item in summary.values():
        print(item["task"], "scored", item["new_latest_scored_step"],
              "first D change", item["first_changed_d_rate_step"],
              "first score change", item["first_changed_scored_step"])
