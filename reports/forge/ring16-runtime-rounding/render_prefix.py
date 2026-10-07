"""Render retained CUDA training samples; no model calls or random draws."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import torch

from experiments.forge.contracts import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.root / arm / "observations.pt" for arm in ("fresh_graph", "fresh_serial401")]
    observations = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    assert all(len(rows) == 24 for rows in observations)
    assert all(a["step"] == b["step"] and torch.equal(a["samples"], b["samples"])
               for a, b in zip(*observations))
    rows = observations[0]
    indices = np.linspace(0, len(rows) - 1, 9).round().astype(int).tolist()
    angles = np.arange(16) * (2 * np.pi / 16)
    centers = 3 * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    frames = []
    for index in indices:
        row = rows[index]
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), dpi=100)
        for axis in axes:
            axis.set(xlim=(-12, 12), ylim=(-12, 12), aspect="equal", xlabel="x", ylabel="y")
            axis.grid(alpha=.16)
        for x, y in centers:
            axes[0].add_patch(Circle((x, y), .2, color="#1e559c", fill=False, linewidth=1.3))
            axes[1].add_patch(Circle((x, y), .2, color="#1e559c", fill=False, alpha=.5))
        axes[0].scatter(*centers.T, s=10, color="#1e559c")
        axes[0].set_title("Target: 16 Gaussian modes\nradius 3; sigma .1; circles show 2 sigma")
        samples = row["samples"].numpy()
        axes[1].scatter(*samples.T, s=2, alpha=.28, color="#e18019", rasterized=True)
        metric = row["metrics"]
        axes[1].set_title(f"Actual clean live samples: update {row['step']}\n"
                          f"component cov error {metric['component_covariance_error']:.3f}; HQ {metric['hq']:.3f}")
        fig.suptitle("Shared ordinary prefix of both fresh CUDA arms; 4,096 retained samples", fontsize=11)
        fig.text(.5, .018, "Prefix through 400 only. Update-401 causal result is measured by gradient/state equality.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, .94))
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        frames.append(Image.open(buffer).convert("RGB"))
    args.output.mkdir(parents=True, exist_ok=True)
    gif = args.output / "shared-prefix.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=[350] * (len(frames) - 1) + [1400], optimize=False)
    atomic_json(args.output / "index.json", {
        "scope": "retained_actual_cuda_training_prefix", "qualification_input": False,
        "new_model_calls": 0, "new_scoring_draws": 0,
        "source_artifacts": {str(p): file_hash(p) for p in paths},
        "both_fresh_prefix_observations_bit_exact": True,
        "gif": {"file": gif.name, "sha256": file_hash(gif), "bytes": gif.stat().st_size,
                "frames": len(frames), "steps": [rows[i]["step"] for i in indices],
                "axes": [-12, 12], "sample_count_per_frame": 4096,
                "maximum_absolute_prefix_sample": max(float(row["samples"].abs().max()) for row in rows),
                "last_frame_update": 400, "boundary401_samples_claimed": False,
                "target_panel": "Analytic centers and 2-sigma contours; no target draw."}})
    print(f"Rendered {gif}; {len(frames)} actual saved frames; zero model calls.", flush=True)


if __name__ == "__main__":
    main()
