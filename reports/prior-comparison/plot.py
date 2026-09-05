#!/usr/bin/env python
"""Rebuild curves.png from the portable data.json beside this script."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "data.json").read_text())
    colors = {"particles": "#047857", "frozen_gaussian": "#2563eb", "fresh_gaussian": "#b45309"}
    labels = {"particles": "Learned particles", "frozen_gaussian": "Frozen Gaussian table",
              "fresh_gaussian": "Fresh Gaussian noise"}
    styles = dict(zip(data["seeds"], ("-", "--", ":")))
    panels = (("modes", "Modes with ≥10 high-quality samples", 1),
              ("hq", "High-quality samples (%)", 100),
              ("sliced_w1", "Sliced W1 (log scale)", 1))
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for run in data["runs"]:
            rows = run["metrics"]
            for axis, (key, title, scale) in zip(axes, panels):
                axis.plot([row["step"] for row in rows], [scale * row[key] for row in rows],
                          color=colors[run["prior"]], linestyle=styles[run["seed"]],
                          linewidth=1.6, alpha=0.9)
        for axis, (_, title, _) in zip(axes, panels):
            axis.set_title(title, fontsize=11)
            axis.set_xlabel("Training step")
            axis.set_xlim(0, data["common_config"]["total_steps"])
            axis.axvline(4200, color="#64748b", linewidth=1, alpha=0.5)
            axis.grid(alpha=0.2)
        axes[0].set_ylim(0, 103)
        axes[1].set_ylim(0, 103)
        axes[2].set_yscale("log")
        figure.suptitle("Matched prior comparison · 7,000 steps · all three seeds", fontsize=14, y=0.995)
        prior_handles = [Line2D([], [], color=colors[k], linewidth=2, label=labels[k]) for k in colors]
        seed_handles = [Line2D([], [], color="#475569", linestyle=style, label=f"Seed {seed}")
                        for seed, style in styles.items()]
        figure.legend(handles=prior_handles + seed_handles, ncol=6, loc="lower center",
                      bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=9)
        figure.text(0.5, 0.09,
                    "EMA read-out every 100 steps; vertical line marks the start of learning-rate annealing.",
                    ha="center", fontsize=9, color="#475569")
        figure.tight_layout(rect=(0, 0.13, 1, 0.94))
        figure.savefig(directory / "curves.png", dpi=180, facecolor="white")
        plt.close(figure)


if __name__ == "__main__":
    main()
