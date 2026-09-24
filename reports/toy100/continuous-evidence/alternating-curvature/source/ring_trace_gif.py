"""Render a cold-ring particle trace as a GIF.

Gray: the eight ring mode centers with their HQ radius (3 sigma = .21).
Color: the twelve clean generator particles, with a short fading trail.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--trail", type=int, default=15)
    args = parser.parse_args()
    data = json.loads(args.trace.read_text())
    trace, means = np.array(data["trace"]), np.array(data["means"])
    hq = {o["step"]: o["hq"] for o in data["observations"]}
    frames = list(range(0, len(trace), args.every))
    colors = plt.cm.tab20(np.linspace(0, 1, trace.shape[1], endpoint=False))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    def draw(index):
        ax.clear()
        for center in means:
            ax.add_patch(plt.Circle(center, .21, color="0.8", zorder=0))
        ax.scatter(means[:, 0], means[:, 1], c="0.45", s=18, zorder=1)
        start = max(0, index - args.trail)
        for particle, color in enumerate(colors):
            path = trace[start:index + 1, particle]
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=.35, lw=1, zorder=2)
            ax.scatter(*trace[index, particle], color=color, s=28, edgecolors="k", linewidths=.4, zorder=3)
        step = index + 1
        latest = max((s for s in hq if s <= step), default=None)
        label = f"update {step}" + (f"   HQ@{latest} = {hq[latest]:.2f}" if latest else "")
        ax.set_title(f"{args.title}\n{label}", fontsize=9)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal"); ax.axis("off")
    animation = FuncAnimation(fig, draw, frames=frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=12))
    print(args.output)


if __name__ == "__main__":
    main()
