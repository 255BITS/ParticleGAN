"""Render the README hero GIF from a recorded toy100 grid100 run.

Every frame is a saved live-weight snapshot with its own recorded metrics.
Dense early checkpoints play quickly; the stable tail plays slower.

    python reports/readme-100gaussians/render_hero.py RUN_DIR 100gaussians.gif
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np

from benchmarks.toy100.gate import _events
from benchmarks.toy100.problems import evaluation_geometry

FAST_UNTIL = 1200  # dense checkpoints up to here
FAST_MS, SLOW_MS, HOLD_MS = 40, 80, 1000


def _frame(live, centers, step, budget, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(4.8, 5.2), facecolor="#ffffff")
    extent = 5.6
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.scatter(centers[:, 0], centers[:, 1], s=40, facecolors="none",
               edgecolors="#94a3b8", linewidths=.6)
    ax.scatter(live[:, 0], live[:, 1], s=1.4, alpha=.5, color="#087f8c",
               rasterized=True)
    fig.suptitle("ParticleGAN · 100 Gaussians", fontsize=13, color="#0f172a",
                 y=.975)
    ax.set_title(f"step {step:,}/{budget:,} · {metrics['modes']}/100 modes · "
                 f"{metrics['hq']:.1%} within 3σ",
                 fontsize=9.5, color="#334155")
    fig.tight_layout(rect=(0, 0, 1, .96))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("P", palette=Image.ADAPTIVE, colors=64)


def main(run_dir: Path, dest: Path) -> None:
    rows = {row["step"]: row for row in _events(run_dir / "events.jsonl")}
    budget = max(rows)
    steps = [s for s in sorted(rows)
             if (run_dir / "snapshots" / f"step_{s:06d}.npz").is_file()]
    centers = evaluation_geometry("grid100")[0].cpu().numpy()
    frames, durations = [], []
    for step in steps:
        live = np.load(run_dir / "snapshots" / f"step_{step:06d}.npz")["live"]
        frames.append(_frame(live, centers, step, budget, rows[step]["metrics"]))
        durations.append(FAST_MS if step <= FAST_UNTIL else SLOW_MS)
    durations[-1] = HOLD_MS
    frames[0].save(dest, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"{len(frames)} frames -> {dest}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
