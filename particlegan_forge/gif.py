"""Actual-training GIF renderer for public-API and Tier 1 tasks in particlegan_forge."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _safe_float(v: object) -> float:
    """Safely convert a metric value to float; non-finite, None, and non-numeric map to nan."""
    if v is None:
        return float("nan")
    try:
        val = float(v)
        return val if math.isfinite(val) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _tile_images(images: np.ndarray, max_cols: int = 8) -> np.ndarray:
    """Tile (N, C, H, W) or (N, H, W) float images in [0, 1] into a 2D/3D canvas."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    n, c, h, w = arr.shape
    cols = min(max_cols, max(1, n))
    rows = (n + cols - 1) // cols
    pad = 1
    canvas = np.full((c, rows * (h + pad) + pad, cols * (w + pad) + pad), 0.18, dtype=np.float32)
    for idx in range(n):
        r, col = divmod(idx, cols)
        y0 = pad + r * (h + pad)
        x0 = pad + col * (w + pad)
        canvas[:, y0 : y0 + h, x0 : x0 + w] = np.clip(arr[idx], 0.0, 1.0)
    if c == 1:
        return canvas[0]
    return np.transpose(canvas[:3], (1, 2, 0))


def render_training_gif(
    frames: list[dict],
    output_path: Path | str,
    *,
    task_id: str,
    candidate_id: str,
    verdict: str,
    primary_metric: str | None = None,
    duration_ms: int = 180,
) -> dict:
    """Render observed training states (target vs. generated samples + metric curve) to a GIF.

    Each element of ``frames`` is a dict:
      - ``step``: int
      - ``passed``: bool
      - ``metrics``: dict[str, float]
      - ``target_images``: optional (N_t, C, H, W) array/tensor
      - ``sample_images``: optional (N_s, C, H, W) array/tensor
      - ``current_stream_image``: optional (1, C, H, W) array/tensor (for single-stream tasks)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames: list[Image.Image] = []

    steps = [f["step"] for f in frames]
    if primary_metric is None and frames and frames[0].get("metrics"):
        primary_metric = sorted(frames[0]["metrics"].keys())[0]

    metric_vals = (
        [_safe_float((f.get("metrics") or {}).get(primary_metric)) for f in frames]
        if primary_metric
        else []
    )

    for idx, frame in enumerate(frames):
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), dpi=100)
        fig.patch.set_facecolor("#111418")

        # Panel 0: Target / Stream input
        ax0 = axes[0]
        ax0.set_facecolor("#111418")
        if "current_stream_image" in frame and frame["current_stream_image"] is not None:
            stream_im = np.asarray(frame["current_stream_image"])
            target_im = np.asarray(frame.get("target_images", stream_im))
            combined = np.concatenate([stream_im[:1], target_im], axis=0)
            grid0 = _tile_images(combined, max_cols=4)
            ax0.set_title("Stream x_t [0] + Target Modes", color="#e6edf3", fontsize=9)
        elif "target_images" in frame and frame["target_images"] is not None:
            grid0 = _tile_images(frame["target_images"], max_cols=4)
            ax0.set_title("Target Law", color="#e6edf3", fontsize=9)
        else:
            grid0 = np.zeros((8, 8), dtype=np.float32)
            ax0.set_title("Target", color="#e6edf3", fontsize=9)
        ax0.imshow(grid0, cmap="viridis" if grid0.ndim == 2 else None, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax0.axis("off")

        # Panel 1: Generated Samples / Rollout
        ax1 = axes[1]
        ax1.set_facecolor("#111418")
        if "sample_images" in frame and frame["sample_images"] is not None:
            grid1 = _tile_images(frame["sample_images"], max_cols=8)
        else:
            grid1 = np.zeros((8, 8), dtype=np.float32)
        ax1.imshow(grid1, cmap="viridis" if grid1.ndim == 2 else None, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax1.set_title("Served Output / Particles", color="#e6edf3", fontsize=9)
        ax1.axis("off")

        # Panel 2: Metric trajectory
        ax2 = axes[2]
        ax2.set_facecolor("#181c22")
        if primary_metric and metric_vals:
            ax2.plot(steps[: idx + 1], metric_vals[: idx + 1], color="#58a6ff", marker="o", markersize=3, linewidth=1.5)
            ax2.set_xlim(min(steps), max(max(steps), min(steps) + 1))
            finite_vals = [v for v in metric_vals if math.isfinite(v)]
            if finite_vals:
                ymin, ymax = min(finite_vals), max(finite_vals)
                pad = max(0.05, 0.1 * (ymax - ymin))
                ax2.set_ylim(ymin - pad, ymax + pad)
            else:
                ax2.set_ylim(0.0, 1.0)
            val_str = (
                f"{metric_vals[idx]:.4f}"
                if (idx < len(metric_vals) and math.isfinite(metric_vals[idx]))
                else "nan"
            )
            ax2.set_title(f"{primary_metric}: {val_str}", color="#e6edf3", fontsize=9)
        ax2.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color("#30363d")

        step_status = "PASS" if frame.get("passed", False) else "FAIL"
        color = "#3fb950" if step_status == "PASS" else "#f85149"
        fig.suptitle(
            f"{task_id} | {candidate_id} | step {frame['step']} ({step_status}) | final: {verdict}",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    return {
        "path": str(output_path),
        "frames": len(pil_frames),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest() if output_path.is_file() else None,
    }
