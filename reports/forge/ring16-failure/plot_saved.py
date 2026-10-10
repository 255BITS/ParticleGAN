"""Plot the frozen Ring16 observations already certified by analyze_saved.py."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(results, output):
    receipt = json.loads(results.read_text())
    paths = {"historical": next(Path(p) for p in receipt["inputs"]
                                if p.endswith("mog100-n256-ring16_acquisition/curve.json")),
             "current": next(Path(p) for p in receipt["inputs"]
                             if p.endswith(receipt["attempt_id"] + "/result.json"))}
    curves = {}
    for label, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["inputs"][str(path)]["sha256"]:
            raise ValueError("Changed certified plot input: " + str(path))
        saved = json.loads(path.read_text())
        curves[label] = ([{"step": p["step"], **p["metrics"]} for p in saved] if label == "historical"
                         else saved["task_results"][0]["evidence"]["observations"])
    matplotlib.rcParams["svg.hashsalt"] = "ring16-failure-report-v1"
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    colors = {"historical": "#247447", "current": "#b64132"}
    titles = {"historical": "Archived restore/extend", "current": "Current uninterrupted"}
    for label, curve in curves.items():
        steps = [p["step"] for p in curve]
        for ax, metric in zip(axes[:2], ["component_covariance_error", "hq"]):
            ax.plot(steps, [p[metric] for p in curve], color=colors[label], label=titles[label], linewidth=1.6)
    axes[0].axhline(.85, color="#555555", linestyle="--", label="Required ≤ 0.85")
    axes[0].set_yscale("log")
    axes[0].set_title("Full component covariance error")
    axes[1].axhline(.85, color="#555555", linestyle="--", label="Required ≥ 0.85")
    axes[1].set_title("Fraction within target 3σ")
    axes[1].set_ylim(0, 1.02)
    for ax in axes[:2]:
        ax.axvline(400, color="#888888", linestyle=":")
        ax.set_xlabel("Completed updates")
        ax.legend(fontsize=8, loc="best")
    for offset, label in [(-.18, "historical"), (.18, "current")]:
        errors = receipt["endpoints"][label]["component_covariance_errors"]
        axes[2].bar([i + offset for i in range(16)], errors, width=.35, color=colors[label], label=titles[label])
    axes[2].set_title("Endpoint covariance error by component")
    axes[2].set_xlabel("Target component (zero based)")
    axes[2].set_xticks([0, 4, 8, 11, 15])
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Ring16: coverage improves, but the uninterrupted path retains a far tail", fontsize=12)
    fig.text(.5, .01, "Same recipe and 96 saved observations; distinct execution histories. No fresh training or model draws.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .93))
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)
    if output.suffix.lower() == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.output)
