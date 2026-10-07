"""Render only saved numerical diagnostics; performs no model draws or updates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.raw / "results.json").read_text())
    if result["protocol_sha256"] != digest(DEST / "protocol.json"):
        raise ValueError("analysis protocol mismatch")
    if result["analysis_source_sha256"] != digest(ROOT / "benchmarks/toy_audit/gaussian1d_diagnosis.py"):
        raise ValueError("analysis source mismatch")
    shutil.copyfile(args.raw / "results.json", DEST / "results.json")
    rows = json.loads((args.raw / "observations-analysis.json").read_text())
    plt.rcParams.update({"svg.hashsalt": "gaussian1d-saved-diagnosis-v1", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    steps = [row["step"] for row in rows]
    for ax, metric, label, lower, upper in zip(axes, ["mean", "std", "cdf_ks"],
                                             ["Mean", "Standard deviation", "CDF KS"],
                                             [1.9, .4, 0.], [2.1, .6, .05]):
        ax.axhspan(lower, upper, color="#dff0de", label="Passing metric range")
        ax.plot(steps, [row["metrics"][metric] for row in rows], color="#295888", lw=1.5,
                label="Saved trained outputs (4,096 samples)")
        if metric == "cdf_ks":
            ax.plot(steps, [row["fitted_normal_ks"] for row in rows], color="#aa5633", lw=1,
                    alpha=.8, label="Shape KS after fitting mean/std")
            ax.set_ylim(0., .4)
            ax.text(.02, .92, "Initial target KS = .99992 (outside plotted range)",
                    transform=ax.transAxes, fontsize=8)
            passed = [row for row in rows[1:] if not row["failed_bounds"]]
            ax.scatter([row["step"] for row in passed], [row["metrics"][metric] for row in passed],
                       color="#287634", marker="o", s=32, zorder=3, label="All bounds pass: 3 isolated checks")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title("The Gaussian is acquired intermittently, then drifts")
    axes[-1].set_xlabel("Completed training updates")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.savefig(DEST / "trajectory.svg", metadata={"Date": None})
    plt.close(fig)
    svg = DEST / "trajectory.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print("Published compact results and trajectory from saved arrays; zero model draws/updates")


if __name__ == "__main__":
    main()
