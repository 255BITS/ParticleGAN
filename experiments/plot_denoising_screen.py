#!/usr/bin/env python
"""Compare saved screen samples at a prespecified seed; no training or GPU use."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=24002)
    args = parser.parse_args()
    configs = [yaml.safe_load(Path(p).read_text()) for p in json.loads(Path(args.manifest).read_text())]
    configs = [c for c in configs if c["seed"] == args.seed]
    cells = [("gan", "concat", "gaussian", "gaussian"),
             ("gan", "concat", "learned", "gaussian"),
             ("gan", "ucd", "learned", "gaussian"),
             ("ddgan", "concat", "gaussian", "gaussian"),
             ("ddgan", "concat", "learned", "gaussian"),
             ("ddgan", "ucd", "learned", "gaussian"),
             ("ddgan", "ucd", "learned", "learned")]
    rng = np.random.default_rng(723)
    coords = np.arange(10) - 4.5
    ix, iy = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    means = np.stack([coords[ix.ravel()], coords[iy.ravel()]], axis=1)
    labels = 2 * (ix.ravel() % 2) + iy.ravel() % 2
    ids = rng.integers(0, 100, 4000)
    target = means[ids] + rng.normal(0, .03, (4000, 2))
    panels = [("Target: 100 Gaussians, four classes", target, labels[ids], "Colors indicate requested class")]
    for cell in cells:
        cfg, = [c for c in configs if tuple(c[k] for k in ("model", "d_mode", "prior", "noise")) == cell]
        root = Path(cfg["out_dir"])
        data = np.load(root / "final_samples.npz")
        metrics = json.loads((root / "summary.json").read_text())["final"]
        select = rng.choice(len(data["x"]), 4000, replace=False)
        step_noise = "N/A" if cell[0] == "gan" else cell[3]
        title = f"{cell[0].upper()} / {cell[1]}\nlatent: {cell[2]}, step noise: {step_noise}"
        caption = f"Joint HQ {100 * metrics['joint_hq']:.1f}% | modes {metrics['modes']}/100"
        panels.append((title, data["x"][select], data["c"][select], caption))
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharex=True, sharey=True, layout="constrained")
    fig.set_constrained_layout_pads(h_pad=.12, w_pad=.06, hspace=.12, wspace=.03)
    for ax, (title, x, c, caption) in zip(axes.flat, panels):
        ax.scatter(means[:, 0], means[:, 1], s=18, facecolors="none", edgecolors="gray", linewidths=.4)
        ax.scatter(x[:, 0], x[:, 1], c=c, cmap="tab10", vmin=0, vmax=9, s=1.5, alpha=.5, rasterized=True)
        ax.set(title=title, xlabel=caption, xlim=(-5.2, 5.2), ylim=(-5.2, 5.2), aspect="equal")
    fig.suptitle(f"7000 updates; seed {args.seed} selected in advance; 4000 displayed samples per panel\n"
                 "Joint HQ requires the requested class and distance < 3σ from a mode; full metrics use 20,000 samples")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
