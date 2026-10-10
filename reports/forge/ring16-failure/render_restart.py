"""Render actual saved training observations; CPU metadata/rendering only."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(raw, destination):
    destination.mkdir(parents=True, exist_ok=False)
    result = json.loads((HERE / "reproduction-results.json").read_text())
    entries = []
    means = 3 * np.column_stack((np.cos(np.arange(16) * 2*np.pi/16), np.sin(np.arange(16) * 2*np.pi/16)))
    for arm in ("live", "restart", "archive-replay"):
        source = raw / arm / ("prefix-observations.pt" if arm == "live" else "observations.pt")
        samples = torch.load(source, map_location="cpu", weights_only=True)
        ids = sorted(set(np.linspace(0, len(samples)-1, 9).round().astype(int).tolist()))
        frames, steps = [], []
        for index in ids:
            row = samples[index]
            steps.append(row["step"])
            fig, (ax, plot) = plt.subplots(1, 2, figsize=(8, 3.8), dpi=100)
            values = row["samples"].numpy()
            ax.scatter(values[:, 0], values[:, 1], s=1, alpha=.3, color="#2457a7")
            for center in means:
                ax.add_patch(Circle(center, .3, fill=False, color="#ca7721", lw=1))
            ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel="x", ylabel="y", aspect="equal")
            curve = samples[:index+1]
            plot.plot([p["step"] for p in curve], [p["metrics"]["component_covariance_error"] for p in curve], color="#2457a7")
            plot.axhline(.85, color="#ca7721", linestyle="--", label="full covariance bound .85")
            plot.set(xlim=(0, 1600), ylim=(0, 12), xlabel="training update", ylabel="mean component covariance error")
            plot.legend(fontsize=7, loc="upper right")
            metrics = row["metrics"]
            scope = "saved prefix only; final live metrics FAIL" if arm == "live" else "final PASS; six terminal checks"
            fig.suptitle(f"{arm} — actual training at {row['step']}\n{scope}; cov={metrics['component_covariance_error']:.3f}, HQ={metrics['hq']:.3f}", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, .86))
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            with Image.open(buffer) as loaded:
                frames.append(loaded.convert("P", palette=Image.Palette.ADAPTIVE))
        target = destination / f"{arm}.gif"
        frames[0].save(target, save_all=True, append_images=frames[1:], duration=[500]*(len(frames)-1)+[1500], loop=0, optimize=False)
        with Image.open(target) as loaded:
            if loaded.n_frames != len(steps):
                raise ValueError("GIF frame count differs")
        entries.append({"arm": arm, "gif": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
            "frame_steps": steps, "saved_observations": len(samples), "source": str(source.relative_to(ROOT)),
            "source_sha256": sha(source), "execution_source_commit": json.loads((raw / arm / "source.json").read_text())["origin_commit"],
            "scope": "actual_training_prefix_only" if arm == "live" else "actual_training_full_reproduction",
            "prefix_origin": "fresh shared public-API live400" if arm != "archive-replay" else "original archived public-API prefix400"})
    manifest = {"schema_version": 1, "renderer_sha256": sha(Path(__file__)), "training_updates": 0,
                "model_forwards": 0, "sampling_draws": 0, "device": "CPU saved-tensor rendering only",
                "metrics_result_sha256": sha(HERE / "reproduction-results.json"), "entries": entries}
    (destination / "index.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"event": "saved_training_media_complete", "gifs": len(entries), "sampling_draws": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "runs/api/ring16-restart-diagnostic-v1")
    parser.add_argument("--output", type=Path, default=HERE / "media-reproduction")
    args = parser.parse_args()
    render(args.raw.resolve(), args.output.resolve())
