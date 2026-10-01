"""GIFs from the source-family observer's actual captured states.

No coordinates, scores or optimizer updates are interpolated. Every frame is
bound to the raw observation NPZ and the external captured-metrics artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .source_routed_ring_training import sha, write

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fafafa", "axes.facecolor": "#fafafa"})
COLORS = ("#b94352", "#137d69")


def image_frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128)


def save(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[280] * (len(frames) - 1) + [2200], loop=0, optimize=True)
    frames[-1].convert("RGB").save(path.with_suffix(".png"))
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                frames=len(frames), interpolation=False, poster=str(path.with_suffix(".png")))


def routed(report, data, metrics, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.5), dpi=95)
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.22, wspace=.28, hspace=.52)
    target_edits = data["target"] - data["neutral"]
    model_edits = data["prediction"] - data["neutral"]
    limits = [min(float(target_edits.min()), float(model_edits.min())),
              max(float(target_edits.max()), float(model_edits.max()))]
    span = max(.1, limits[1] - limits[0])
    limits = (limits[0] - .06 * span, limits[1] + .06 * span)
    frames = []
    for index, metric in enumerate(metrics):
        for ax in axes.flat:
            ax.clear()
        step = int(data["steps"][index])
        model, target = model_edits[index], target_edits[index]
        flat_model, flat_target = model.reshape(-1, 2), target.reshape(-1, 2)
        stride = max(1, len(flat_model) // 1800)
        for coordinate, color in enumerate(COLORS):
            axes[0, 0].scatter(flat_target[::stride, coordinate], flat_model[::stride, coordinate],
                               s=5, alpha=.3, color=color, label=f"coordinate {coordinate + 1}")
        axes[0, 0].plot(limits, limits, color="#777777", lw=1, ls=":", label="exact paired edit")
        axes[0, 0].set(xlim=limits, ylim=limits, xlabel="Target edit at each held-out context",
                       ylabel="Clean model edit at the same context", title="Paired correspondence")
        axes[0, 0].legend(fontsize=7, loc="upper left")
        if model.ndim == 3:
            context = len(model) // 2
            for coordinate, color in enumerate(COLORS):
                axes[0, 1].plot(target[context, :, coordinate], color=color, ls=":", label=f"target {coordinate + 1}")
                axes[0, 1].plot(model[context, :, coordinate], color=color, label=f"model {coordinate + 1}")
            axes[0, 1].set(xlabel="Fixed token order", ylim=limits, title=f"Fixed held-out context {context} · {model.shape[1]} tokens")
        else:
            for coordinate, color in enumerate(COLORS):
                axes[0, 1].plot(target[:, coordinate], color=color, ls=":", lw=1, label=f"target {coordinate + 1}")
                axes[0, 1].plot(model[:, coordinate], color=color, lw=1, label=f"model {coordinate + 1}")
            axes[0, 1].set(xlabel="Fixed held-out context order", ylim=limits, title="All 180 held-out contexts")
        axes[0, 1].legend(fontsize=7, ncol=2)
        seen = metrics[:index + 1]
        steps = [r["step"] for r in seen]
        axes[1, 0].plot(steps, [r["original"]["heldout_rmse"] for r in seen], color="#3274aa")
        axes[1, 0].set(xlim=(0, report["updates"]), ylim=(0, max(r["original"]["heldout_rmse"] for r in metrics) * 1.05),
                       xlabel="Completed optimizer updates", title=f"Original RMSE {metric['original']['heldout_rmse']:.5f} · no absolute source gate")
        quality = [r["correspondence"]["relative_mse"] for r in seen]
        axes[1, 1].plot(steps, quality, color="#137d69")
        axes[1, 1].axhline(.1, color="#777777", ls=":", label="separate audit gate ≤ 0.10")
        axes[1, 1].set(xlim=(0, report["updates"]), ylim=(0, max(.12, max(r["correspondence"]["relative_mse"] for r in metrics)) * 1.05),
                       xlabel="Completed optimizer updates", title=f"MSE / neutral MSE {quality[-1]:.5f} · {'PASS' if metric['correspondence']['passed'] else 'FAIL'}")
        axes[1, 1].legend(fontsize=7)
        if report["fixture"] == "moving":
            for ax in axes[1]:
                for turn in (500, 1000):
                    ax.axvline(turn, color="#888888", ls="--", lw=.7)
        fig.suptitle(f"Routed {report['fixture']} · actual update {step}/{report['updates']} · served {metric['original']['served_source']}"
                     + (f" · target turn {metric['angle']:.0f}°" if report["fixture"] == "moving" else ""), fontsize=13, y=.95)
        text = "Clean deterministic routed forward; 180 held-out contexts; paired errors are never training signals."
        if report["fixture"] == "replay":
            text += "\n8-update activation replay smoke: software protocol only; full convergence untested."
        elif report["fixture"] == "support":
            text += f"\nDefault full bank: 128 particles, 128 tokens. {report['status']} at {report['completed_updates']}/1200; final captured update200."
        else:
            text += "\nOriginal source has no absolute trained gate. Stronger correspondence gate is a separate audit definition."
        for old in list(fig.texts):
            if old is not fig._suptitle:
                old.remove()
        fig.text(.5, .045, text, ha="center", fontsize=8)
        frames.append(image_frame(fig))
    plt.close(fig)
    return save(frames, output)


def ring(report, data, metrics, output):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.7), dpi=95)
    fig.subplots_adjust(left=.05, right=.98, top=.78, bottom=.24, wspace=.34)
    angles = np.arange(8) * 2 * np.pi / 8
    centers = 3 * np.stack((np.cos(angles), np.sin(angles)), 1)
    frames = []
    for index, metric in enumerate(metrics):
        for ax in axes:
            ax.clear()
        points = data["live"][index]
        for x, y in centers:
            axes[0].add_patch(plt.Circle((x, y), .21, color="#777777", fill=False, lw=1))
        axes[0].scatter(points[:, 0], points[:, 1], s=2, color="#b94352", alpha=.25)
        axes[0].set(xlim=(-4.8, 4.8), ylim=(-4.8, 4.8), title="All 4,096 live evaluator draws")
        axes[0].set_aspect("equal", adjustable="box")
        seen = metrics[:index + 1]
        steps = [r["step"] for r in seen]
        axes[1].plot(steps, [r["original"]["hq"] for r in seen], color="#b94352", label="HQ fraction")
        axes[1].plot(steps, [r["original"]["modes"] / 8 for r in seen], color="#137d69", label="quality modes / 8")
        axes[1].axhline(.90, color="#777777", ls=":", lw=1)
        axes[1].set(xlim=(0, 1200), ylim=(-.02, 1.04), xlabel="Completed updates", title=f"Original: {metric['original']['modes']}/8 modes · HQ {metric['original']['hq']:.3f}")
        axes[1].legend(fontsize=7)
        axes[2].plot(steps, [r["gaussian_law"]["mass_tv"] for r in seen], color="#3274aa", label="mass TV")
        axes[2].plot(steps, [r["gaussian_law"]["max_radial_ks"] for r in seen], color="#b94352", label="max radial KS")
        axes[2].axhline(.075, color="#3274aa", ls=":", lw=1)
        axes[2].axhline(.10, color="#b94352", ls=":", lw=1)
        axes[2].set(xlim=(0, 1200), ylim=(-.02, 1.02), xlabel="Completed updates",
                    title=f"Separate Gaussian-law gate: {'PASS' if metric['gaussian_law']['passed'] else 'FAIL'}")
        axes[2].legend(fontsize=7)
        fig.suptitle(f"Source ring acquisition · actual update {metric['step']}/1200 · constant rates · {report['status']} terminal", fontsize=13, y=.94)
        for old in list(fig.texts):
            if old is not fig._suptitle:
                old.remove()
        fig.text(.5, .08, "Source default config via its LegacyRecipe bridge; fixed evaluation indices, declared output-noisy live law.\n"
                 "Failed acquisition stops hold and matched shift/frozen-control phases. No training extension or favorable checkpoint selection.", ha="center", fontsize=8)
        frames.append(image_frame(fig))
    plt.close(fig)
    return save(frames, output)


def render(report_path, output):
    report = json.loads(Path(report_path).read_text())
    observed = report.get("observations")
    if observed is None:
        return dict(fixture=report["fixture"], media=None, reason="Source did not reach an observable training boundary")
    artifact = Path(observed["path"])
    if sha(artifact) != observed["sha256"]:
        raise ValueError("captured-state artifact differs from training receipt")
    data = np.load(artifact)
    metrics_path = artifact.parent / "captured-metrics.json"
    metrics = json.loads(metrics_path.read_text())
    media = ring(report, data, metrics, output) if report["fixture"] == "ring" else routed(report, data, metrics, output)
    return dict(fixture=report["fixture"], catalog_id=report["catalog_id"], media=media,
                renderer_sha256=sha(__file__),
                training_receipt=dict(path=str(report_path), sha256=sha(report_path)),
                raw_observations=observed, captured_metrics=dict(path=str(metrics_path), sha256=sha(metrics_path)),
                frames_are_actual_states=True, interpolation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    args = parser.parse_args()
    write(args.media_receipt, render(args.receipt, args.output))


if __name__ == "__main__":
    main()
