"""Run from the repository root after the 16/64-step unclipped studies finish."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.autonomous_memory import trajectory_metrics


def directions(paths):
    late = [trajectory_metrics(path[None, 128:]) for path in paths]
    circles = [m for m in late if m["circle_like_fraction"] == 1]
    return {"late_circle_count": len(circles),
            "late_circle_ccw": sum(m["signed_angular_speed"] > 0 for m in circles),
            "late_circle_cw": sum(m["signed_angular_speed"] < 0 for m in circles)}


def main():
    diagnostics = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for row, variant in enumerate(("shared", "frozen_writer")):
        for horizon in (16, 64):
            source = (Path("runs/memory_path/autonomous_unclipped_2k") if horizon == 16
                      else Path("runs/memory_path/autonomous_h64_2k") / f"{variant}_run")
            records = [json.loads(line) for line in (source / variant / "metrics.jsonl").read_text().splitlines()]
            with np.load(source / variant / "trajectories.npz") as saved:
                diagnostics[f"{variant}_h{horizon}"] = directions(saved["generated"])
                if row == 0 and horizon == 16:
                    diagnostics["real_noisy"] = directions(saved["real_noisy"])
            for ax, key in zip(axes[row], ("d", "g")):
                ax.plot([r["step"] for r in records], [r[key] for r in records], label=f"train {horizon}")
                ax.set(title=f"{variant}: {key.upper()} adversarial loss", xlabel="update", yscale="log")
                ax.legend()
    fig.suptitle("Unclipped training: sampled losses; scores alone do not measure trajectory quality")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("learning_curves.png"), dpi=140)
    plt.close(fig)
    Path(__file__).with_name("directions.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
