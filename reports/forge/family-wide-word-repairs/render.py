"""Render actual saved API metric observations against frozen task gate targets.

This performs zero training updates and zero sampling draws. These are labelled
numerical goal GIFs, not reconstructed particle/image frames. The full numerical
grade retains every check; only visual frames are selected here.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, __version__ as pillow_version


def identity(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request_path, raw_path = args.attempt / "request.json", args.attempt / "raw-result.json"
    resolved, raw = [json.loads(path.read_text()) for path in (request_path, raw_path)]
    request, job = resolved["request"], resolved["job"]
    task = request["tasks"][job["task_id"]]
    raw = raw.get("task_results", {}).get(job["task_id"], raw)
    rows = raw.get("evidence", {}).get("observations", [])
    if not rows:
        raise ValueError("This task requires its existing saved observer-output renderer; no sampling fallback")
    bounds = task["evaluation"]["thresholds"]
    indices = np.rint(np.linspace(0, len(rows) - 1, min(9, len(rows)))).astype(int)
    frames = []
    for index in indices:
        row = rows[index]
        fig, axes = plt.subplots(1, len(bounds), figsize=(5 * len(bounds), 4), squeeze=False)
        for ax, (key, operator, bound) in zip(axes.flat, bounds):
            values = [float(point[key]) for point in rows]
            steps = [point["step"] for point in rows]
            ax.plot(steps[:index + 1], values[:index + 1], color="#2255aa", label="Actual API observation")
            ax.scatter([row["step"]], [row[key]], color="#2255aa", s=25)
            ax.axhline(bound, color="#228833", linestyle="--", label=f"Target: {operator} {bound:g}")
            low, high = min(min(values), bound), max(max(values), bound)
            pad = max((high - low) * .1, abs(bound) * .03, .01)
            ax.set(xlim=(0, task["execution"]["steps"]), ylim=(low - pad, high + pad),
                   xlabel="Completed updates", ylabel=key,
                   title=f"{key}\nActual {row[key]:.6g}; required {operator} {bound:g}")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(f"{request['candidate']['id']} · {task['id']} · update {row['step']}\n"
                     "Actual-training numerical goal observations; fixed target/gate bounds")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=85)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=350, loop=0)
    receipt = {"schema_version": 1, "kind": "actual_training_numerical_goal_gif",
        "candidate": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
        "task": task["id"], "thresholds": bounds, "observation_count": len(rows),
        "selected_observation_indices": indices.tolist(), "updates": [rows[i]["step"] for i in indices],
        "observations_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                                         allow_nan=False).encode()).hexdigest(),
        "raw_result": {"path": str(raw_path.resolve()), **identity(raw_path)},
        "resolved_request": {"path": str(request_path.resolve()), **identity(request_path)},
        "gif": {"path": str(args.output), **identity(args.output)},
        "renderer": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "pillow": pillow_version, "source_sha256": identity(Path(__file__))["sha256"]},
        "optimizer_updates": 0, "sampling_draws": 0,
        "scope": "Saved API metric outputs against declared gate targets; no invented samples or parameter interpolation. Numerical grader retains all recorded observations."}
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gif": str(args.output), "optimizer_updates": 0, "sampling_draws": 0}))


if __name__ == "__main__":
    main()
