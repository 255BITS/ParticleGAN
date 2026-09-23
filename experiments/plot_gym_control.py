#!/usr/bin/env python
"""Render held-out control outcomes and validation checkpoint curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_dir):
    root = Path(report_dir)
    board = json.loads((root / "leaderboard.json").read_text())
    rows = {r["arm"]: r for r in board["rows"] if r["report_role"] == "selected"}
    order = [name for name in ("original", "imitation", "joint", "expert") if name in rows]
    labels = {"original": "Original prototype", "imitation": "Imitation only",
              "joint": "Joint three-generator", "expert": "Heuristic expert"}
    colors = {"original": "#8795a8", "imitation": "#478bc9", "joint": "#1b9e77", "expert": "#bc8d32"}
    fig, (land, ret) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    y = np.arange(len(order))
    for i, name in enumerate(order):
        s = rows[name]["summary"]
        rate = 100*s["landing_rate"]
        lo, hi = np.array(s["landing_rate_wilson95"])*100
        land.errorbar(rate, i, xerr=[[max(0, rate-lo)], [max(0, hi-rate)]],
                      fmt="o", capsize=4, color=colors[name], markersize=8)
        land.annotate(f"{s['landing_count']}/{s['episodes']}", (rate, i),
                      xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        values = [e["return"] for e in rows[name]["episodes"]]
        # Fixed deterministic jitter separates episode marks; no subsampling.
        jitter = (np.arange(len(values)) % 9 - 4)*.025
        ret.scatter(values, i+jitter, alpha=.4, s=14, color=colors[name])
        ret.plot(s["mean_return"], i, "D", color=colors[name], markersize=8,
                 markeredgecolor="black", markeredgewidth=.7)
    for ax in (land, ret):
        ax.set_yticks(y, [labels[n] for n in order])
        ax.set_ylim(len(order)-.5, -.6)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    land.set_xlim(-7, 107)
    land.set_xlabel("Successful landings (%) · 95% Wilson intervals")
    ret.set_xlabel("Episode return · diamond = mean, dots = all episodes")
    fig.suptitle("Lunar Lander: 50 paired held-out worlds\nCheckpoints selected on separate validation episodes", fontsize=13)
    fig.savefig(root / "control_outcomes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for name in ("imitation", "joint"):
        selection = json.loads((root / "selections" / f"{name}.json").read_text())
        candidates = sorted(selection["candidates"], key=lambda x: x["step"])
        steps = [r["step"] for r in candidates]
        for ax, key, scale in ((axes[0], "landing_rate", 100), (axes[1], "mean_return", 1)):
            ax.plot(steps, [r["summary"][key]*scale for r in candidates], "o-",
                    label=labels[name], color=colors[name])
            ax.set_xlabel("Training updates")
            ax.set_xticks(steps)
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation successful landings (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("Validation mean episode return")
    axes[1].legend()
    fig.suptitle("Checkpoint selection · 20 paired validation worlds")
    fig.savefig(root / "validation_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/gym/lunar_lander_control")
    plot(parser.parse_args().report_dir)
