"""Render traceable GIFs from saved generated samples, never inferred frames."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .gate import _events, score_run
from .problems import PROBLEM_NAMES, evaluation_geometry


def _frame_steps(steps: list[int], max_frames: int = 14) -> list[int]:
    """Keep initialization and early dynamics, then span the remaining budget."""
    if len(steps) <= max_frames:
        return steps
    early = [step for step in steps if step in (0, 1, 10, 25, 50, 100)]
    remaining = [step for step in steps if step not in early]
    slots = max(2, max_frames - len(early))
    indices = np.linspace(0, len(remaining) - 1, min(slots, len(remaining)), dtype=int)
    return sorted(set(early + [remaining[index] for index in indices]))


def _run(output: Path, name: str):
    run_dir = output / name
    verdict = score_run(run_dir, name)
    if verdict["status"] in ("MISSING", "INVALID", "ERROR"):
        raise ValueError(f"cannot render {name}: {verdict['reason']}")
    config = json.loads((run_dir / "config.json").read_text())
    rows = {row["step"]: row for row in _events(run_dir / "events.jsonl")}
    return {"name": name, "dir": run_dir, "config": config, "rows": rows,
            "steps": sorted(rows), "verdict": verdict}


def _snapshot(run: dict, step: int):
    path = run["dir"] / "snapshots" / f"step_{step:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if "live" not in archive or "target" not in archive:
            raise ValueError(f"{path} is missing live or target samples")
        live, target = archive["live"], archive["target"]
    for label, points in (("live", live), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError(f"{path}: invalid {label} samples")
    return live, target


def _style_axes(ax, *, name: str, step: int, metrics: dict | None, target: bool):
    centers, _ = evaluation_geometry(name)
    centers = centers.cpu().numpy()
    extent = 6.9  # One fixed scale for all three geometries and every frame.
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_facecolor("#ffffff")
    ax.scatter(centers[:, 0], centers[:, 1], s=32, facecolors="none",
               edgecolors="#475569", linewidths=.6, alpha=.8)
    ax.tick_params(colors="#64748b", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if target:
        ax.set_title(f"{name} · target", fontsize=11, color="#0f172a")
    else:
        modes = metrics.get("modes", "?") if metrics else "?"
        hq = metrics.get("hq") if metrics else None
        hq_text = f"{hq:.1%}" if isinstance(hq, (int, float)) else "?"
        ax.set_title(f"live · step {step:,} · {modes}/100 modes · HQ {hq_text}",
                     fontsize=10, color="#0f172a")


def _make_frame(runs: list[dict], step: int):
    # Imports stay local: the read-only gate can run without plot extras.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    columns = len(runs)
    fig, axes = plt.subplots(2, columns, figsize=(4.4 * columns, 8.0), squeeze=False,
                             facecolor="#f8fafc")
    for column, run in enumerate(runs):
        live, target = _snapshot(run, step)
        target_ax, live_ax = axes[:, column]
        target_ax.scatter(target[:, 0], target[:, 1], s=1.6, alpha=.46,
                          color="#64748b", rasterized=True)
        live_ax.scatter(live[:, 0], live[:, 1], s=1.6, alpha=.46,
                        color="#087f8c", rasterized=True)
        _style_axes(target_ax, name=run["name"], step=step, metrics=None, target=True)
        _style_axes(live_ax, name=run["name"], step=step,
                    metrics=run["rows"][step]["metrics"], target=False)
    label = runs[0]["config"].get("name", "toy100")
    budget = runs[0]["config"]["steps"]
    fig.suptitle(f"100-Gaussian toys · {label} · step {step:,}/{budget:,}",
                 fontsize=15, color="#0f172a", y=.99)
    fig.text(.5, .015, "Recorded target and generated samples at each checkpoint · "
             "live weights · fixed axes · per-mode quality scored separately",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .035, 1, .955))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def _save_gif(path: Path, runs: list[dict], steps: list[int]):
    frames = [_make_frame(runs, step) for step in steps]
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [500] * (len(frames) - 1) + [2500]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    for frame in frames:
        frame.close()
    sources = {}
    for run in runs:
        sources[run["name"]] = {
            "events_sha256": hashlib.sha256((run["dir"] / "events.jsonl").read_bytes()).hexdigest(),
            "snapshots": [
                {"step": step, "path": str(run["dir"] / "snapshots" / f"step_{step:06d}.npz"),
                 "sha256": hashlib.sha256((run["dir"] / "snapshots" / f"step_{step:06d}.npz").read_bytes()).hexdigest()}
                for step in steps
            ],
        }
    provenance = {"model": "live", "frames": len(steps), "steps": steps,
                  "axes": {"xlim": [-6.9, 6.9], "ylim": [-6.9, 6.9]},
                  "gif_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "sources": sources}
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return path


def render_progress(output: Path | str, *, problem: str | None = None) -> list[Path]:
    """Render one diagnostic GIF per problem and an aligned all-problem GIF."""
    output = Path(output)
    if problem is not None and problem not in PROBLEM_NAMES:
        raise ValueError(f"unknown problem: {problem}")
    names = (problem,) if problem else PROBLEM_NAMES
    runs = [_run(output, name) for name in names]
    paths = []
    for run in runs:
        steps = _frame_steps(run["steps"])
        paths.append(_save_gif(run["dir"] / "progress.gif", [run], steps))
    if problem is None:
        schedules = [run["steps"] for run in runs]
        if any(schedule != schedules[0] for schedule in schedules[1:]):
            raise ValueError("combined animation requires identical recorded checkpoint schedules")
        steps = _frame_steps(schedules[0])
        paths.append(_save_gif(output / "toy100-progress.gif", runs, steps))
    return paths
