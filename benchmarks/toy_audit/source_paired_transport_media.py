"""Render source validation checkpoints with explicit row correspondence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .source_family_media import image_frame, save
from .source_routed_ring_training import sha, write


def render(receipt_path, output):
    receipt_path, output = Path(receipt_path), Path(output)
    report = json.loads(receipt_path.read_text())
    if "observation_archive" not in report:
        return dict(media=None, reason="Source failed before a saved training checkpoint")
    observed = report["observation_archive"]
    if sha(observed["path"]) != observed["sha256"]:
        raise ValueError("paired observation archive differs from source receipt")
    curve = report["curve_archive"]
    if sha(curve["path"]) != curve["sha256"]:
        raise ValueError("paired metric curve differs from source receipt")
    data = np.load(observed["path"])
    metrics = json.loads(Path(curve["path"]).read_text())
    target = data["target"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.9), dpi=95)
    fig.subplots_adjust(left=.055, right=.98, top=.77, bottom=.25, wspace=.35)
    frames = []
    for index, row in enumerate(metrics):
        for ax in axes:
            ax.clear()
        prediction = data["ema"][index]
        axes[0].scatter(target[:, 0], target[:, 1], s=4, alpha=.22, c="#137d69", label="paired target")
        axes[0].scatter(prediction[:, 0], prediction[:, 1], s=3, alpha=.28, c="#3274aa", label="EMA prediction")
        for k in range(0, len(target), 32):
            axes[0].plot([prediction[k, 0], target[k, 0]], [prediction[k, 1], target[k, 1]], color="#b94352", lw=.6, alpha=.65)
        combined = np.concatenate((target, data["ema"].reshape(-1, 2)))
        lo, hi = combined.min(0), combined.max(0)
        span = np.maximum(hi - lo, 1.)
        axes[0].set(xlim=(lo[0] - .06 * span[0], hi[0] + .06 * span[0]),
                    ylim=(lo[1] - .06 * span[1], hi[1] + .06 * span[1]), title="1,024 validation pairs · red links join same rows")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].legend(fontsize=7)
        seen = metrics[:index + 1]
        steps = [r["step"] for r in seen]
        for key, color, label in (("original_ema", "#3274aa", "EMA"), ("original_live", "#b94352", "live")):
            axes[1].plot(steps, [max(r[key]["nmse"], 1e-12) for r in seen], color=color, label=label)
            axes[2].plot(steps, [r[key]["p95_distance"] for r in seen], color=color, label=label)
        axes[1].set_yscale("log")
        axes[1].axhline(.01, color="#777777", ls=":", lw=1)
        axes[1].set(xlim=(0, 6000), xlabel="Completed source updates", title=f"Original NMSE · EMA {row['original_ema']['nmse']:.4g}")
        axes[1].legend(fontsize=7)
        axes[2].axhline(.2, color="#777777", ls=":", lw=1)
        axes[2].set(xlim=(0, 6000), xlabel="Completed source updates", title=f"Original p95 distance · EMA {row['original_ema']['p95_distance']:.4g}")
        axes[2].legend(fontsize=7)
        fig.suptitle(f"Paired {report['task']} · actual checkpoint {row['step']}/6000 · baseline/movable · {report['fresh_execution_status']}", y=.94, fontsize=13)
        for old in list(fig.texts):
            if old is not fig._suptitle:
                old.remove()
        fig.text(.5, .075, f"Added paired MSE / identity MSE: live {row['added_live']['relative_mse']:.4g}, EMA {row['added_ema']['relative_mse']:.4g} (gate ≤0.10).\n"
                 "Source validation sampling only; reporting MSE is not a training loss. Test selection requires all12 source jobs and is unqualified here.", ha="center", fontsize=8)
        frames.append(image_frame(fig))
    plt.close(fig)
    media = save(frames, output)
    return dict(media=media, actual_step_indices=data["steps"].tolist(), interpolation=False,
                renderer_sha256=sha(__file__), frame_helper_sha256=sha(Path(__file__).with_name("source_family_media.py")),
                training_receipt=dict(path=str(receipt_path), sha256=sha(receipt_path)),
                observation_archive=observed, curve_archive=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    args = parser.parse_args()
    write(args.media_receipt, render(args.receipt, args.output))


if __name__ == "__main__":
    main()
