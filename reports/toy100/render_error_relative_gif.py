"""Replay gain=.5 error-relative cross-only and draw the cold trajectory.

Logging is a no-grad read of the clean generator after each accepted update.
The solver, gates, and host are unchanged.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GAIN = 0.5
TRAIL = 5
COLORS = (
    "#e11d48", "#ea580c", "#ca8a04", "#65a30d", "#059669", "#0891b2",
    "#2563eb", "#4f46e5", "#7c3aed", "#c026d3", "#db2777", "#44403c",
)


def _snapshot(recorder, local, frames):
    import torch
    from benchmarks.locked_shared.trajectory import trajectories

    step = recorder.outer_steps
    rejects = [row for row in recorder.guard_rejections
               if row.get("outer_step") == step and row.get("guard") == "error_relative_motion"]
    accepted = [row for row in recorder.solves if row.get("outer_step") == step and row.get("accepted")]
    scale = accepted[-1]["scale"] if accepted else None
    generator, prior = local.get("generator"), local.get("prior")
    slow_ref, fast = trajectories()
    clean = getattr(generator, "model", generator)
    with torch.no_grad():
        pred = clean(slow_ref, prior.z).detach().reshape(-1, 8, 2).cpu()
    mse = float((pred.reshape(len(pred), -1) - fast.reshape(len(fast), -1)).pow(2).mean())
    frames.append(dict(
        update=step, mse=mse, scale=scale, rejections=len(rejects),
        pred=pred.numpy(), fast=fast.reshape(-1, 8, 2).numpy(),
    ))
    print(json.dumps(dict(event="FRAME", update=step, traj_mse=round(mse, 6),
                          scale=scale, error_relative_rejections=len(rejects))), flush=True)


def run_cold(gain):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    from reports.toy100.cross_competitive_scratch import CrossCompetitiveRecorder, cross_competitive

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="cross_competitive_response", lr_floor=1.0, lr_anneal_start=0.0)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "trajectory")
    frames = []
    with cross_competitive(task="trajectory", error_relative_gain=gain) as (recorder, _source):
        original = CrossCompetitiveRecorder.phases

        def phases(step, opt_d, opt_g, local):
            yield from original(recorder, step, opt_d, opt_g, local)
            _snapshot(recorder, local, frames)

        recorder.phases = phases
        result, _context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    return frames, result


def _subtitle(frame, host_mse):
    scale = frame["scale"]
    scale_text = "n/a" if scale is None else f"{scale:.3g}"
    if frame["rejections"]:
        bound = f"error-relative bound rejected x{frame['rejections']}, scaled to {scale_text}"
    else:
        bound = "error-relative bound held"
    text = f"update {frame['update']}/400 · traj MSE {frame['mse']:.5f} · {bound}"
    if host_mse is not None:
        text += f" · gate MSE {host_mse:.5f}"
    return text


def render(frames, path, host_by_step, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    stride = 2 if len(frames) > 220 else 1
    chosen = frames[::stride]
    if chosen[-1] is not frames[-1]:
        chosen.append(frames[-1])
    images = []
    fast = frames[0]["fast"]
    for index, frame in enumerate(chosen):
        fig, ax = plt.subplots(figsize=(7.2, 7.5), dpi=100, facecolor="#f8fafc")
        ax.set_facecolor("#ffffff")
        for arc in fast:
            ax.plot(arc[:, 0], arc[:, 1], color="#94a3b8", lw=1.1, alpha=.85, zorder=1)
            ax.scatter(arc[:, 0], arc[:, 1], s=14, c="#cbd5e1", zorder=1)
        cursor = next(i for i, row in enumerate(frames) if row["update"] == frame["update"])
        history = frames[max(0, cursor - TRAIL):cursor + 1]
        for age, past in enumerate(history[:-1]):
            alpha = 0.10 + 0.10 * age
            for identity, arc in enumerate(past["pred"]):
                ax.plot(arc[:, 0], arc[:, 1], color=COLORS[identity % len(COLORS)],
                        lw=.6, alpha=alpha, zorder=2)
        for identity, arc in enumerate(frame["pred"]):
            color = COLORS[identity % len(COLORS)]
            ax.plot(arc[:, 0], arc[:, 1], color=color, lw=1.6, alpha=.95, zorder=3)
            ax.scatter(arc[:, 0], arc[:, 1], s=18, c=color, zorder=4)
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal")
        ax.set_title("error-relative cross-only · gain 0.5 · cold trajectory", fontsize=12, color="#0f172a")
        ax.set_xlabel(_subtitle(frame, host_by_step.get(frame["update"])), fontsize=9, color="#0f172a")
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buffer.seek(0)
        with Image.open(buffer) as image:
            images.append(image.convert("RGB"))
        if index % 20 == 0:
            print(json.dumps(dict(event="DRAW", frame=index, of=len(chosen))), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(round(1000 / fps))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration,
                   loop=0, optimize=True, disposal=2)
    print(json.dumps(dict(event="GIF", path=str(path), frames=len(images),
                          fps=fps, seconds=round(len(images) / fps, 2),
                          bytes=path.stat().st_size)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/error-relative-gain05-cold-traj.gif")
    parser.add_argument("--fps", type=float, default=14)
    parser.add_argument("--replay", type=Path, default=None)
    args = parser.parse_args()
    if args.replay is None:
        frames, result = run_cold(GAIN)
        cache = args.output.with_suffix(".npz")
        import numpy as np
        np.savez_compressed(
            cache,
            update=np.array([row["update"] for row in frames]),
            mse=np.array([row["mse"] for row in frames]),
            scale=np.array([(-1 if row["scale"] is None else row["scale"]) for row in frames]),
            rejections=np.array([row["rejections"] for row in frames]),
            pred=np.stack([row["pred"] for row in frames]),
            fast=frames[0]["fast"],
            host_mse=result["live"].get("identity_mse"),
            observations=json.dumps(result.get("observations", [])),
        )
        print(json.dumps(dict(event="TRAIN_DONE", live=result["live"],
                              checks=result.get("convergence"), seconds=result.get("seconds"),
                              frames=len(frames))), flush=True)
        host_by_step = {row["step"]: row.get("identity_mse") for row in result.get("observations", [])}
    else:
        import numpy as np
        archive = np.load(args.replay, allow_pickle=False)
        frames = []
        for index, update in enumerate(archive["update"]):
            scale = float(archive["scale"][index])
            frames.append(dict(update=int(update), mse=float(archive["mse"][index]),
                               scale=None if scale < 0 else scale,
                               rejections=int(archive["rejections"][index]),
                               pred=archive["pred"][index], fast=archive["fast"]))
        host_by_step = {row["step"]: row.get("identity_mse")
                        for row in json.loads(str(archive["observations"]))}
    render(frames, args.output, host_by_step, args.fps)


if __name__ == "__main__":
    main()
