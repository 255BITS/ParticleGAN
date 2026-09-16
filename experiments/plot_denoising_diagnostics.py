#!/usr/bin/env python
"""Plot every variant at a specified seed, with target and local/mass metrics."""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import read_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=24002)
    args = p.parse_args()
    configs = [read_config(p) for p in json.loads(Path(args.manifest).read_text())]
    configs = [c for c in configs if c["seed"] == args.seed]
    rng = np.random.default_rng(723)
    ij = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 2)
    means, labels = ij - 4.5, 2 * (ij[:, 0] % 2) + ij[:, 1] % 2
    ids = rng.integers(0, 100, 4000)
    panels = [("Target (four-class coloring)", means[ids] + rng.normal(0, .03, (4000, 2)),
               labels[ids], "100 modes; σ = .03")]
    for cfg in configs:
        root = Path(cfg["out_dir"])
        data = np.load(root / "final_samples.npz")
        m = json.loads((root / "summary.json").read_text())["final"]
        select = rng.choice(len(data["x"]), 4000, replace=False)
        title = (f"{cfg['model'].upper()} / {cfg['d_mode']} ({cfg.get('ucd_target', 'class')}) / latent {cfg['prior']}\n"
                 f"{cfg['steps']} updates; {len(cfg['alpha_bar']) - 1 if cfg['model'] == 'ddgan' else 0} "
                 f"transitions; {cfg['classes']} classes\n"
                 f"step noise: {cfg['noise'] if cfg['model'] == 'ddgan' else 'N/A'}")
        caption = (f"Joint HQ {m['joint_hq']:.1%}; modes {m['modes']}/100\n"
                   f"Class TV {m['conditional_mode_tv']:.3f}; SW1 {m['conditional_sw1']:.3f}")
        panels.append((title, data["x"][select], data["c"][select], caption))
    rows = math.ceil(len(panels) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(17, 5.5 * rows), squeeze=False, layout="constrained")
    for ax, (title, x, c, caption) in zip(axes.flat, panels):
        ax.scatter(*means.T, s=14, facecolors="none", edgecolors="gray", linewidths=.4)
        ax.scatter(*x.T, c=c, cmap="tab10", vmin=0, vmax=9, s=1.5, alpha=.5, linewidths=0)
        ax.set(title=title, xlabel=caption, xlim=(-5.2, 5.2), ylim=(-5.2, 5.2), aspect="equal")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"Seed {args.seed}; 4000 samples shown per panel; metrics use 20,000\n"
                 "Sharpness, mode proportions, and class fidelity must be assessed together")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
