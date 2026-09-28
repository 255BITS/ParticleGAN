"""Plot saved native100 rate and quality trajectories; no candidate run is inferred."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "h2-prior-handoff"
OUT = Path(__file__).resolve().parent
TASKS = ("grid100", "rotated100", "staggered100")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rate_data(task):
    lines = rows(RUNS / task / "rates.jsonl")
    t = np.array([line["step"] for line in lines])
    g = np.array([line["lr"][0][0] / 0.00425 for line in lines])
    p = np.array([line["lr"][0][1] / 0.00850 for line in lines])
    d = np.array([line["lr"][1][0] / 0.00425 for line in lines])
    # The critic tester takes dyadic values; its later payoff factor is
    # reconstructed from the recorded applied rate (0.988..1 in these runs).
    raw_d = np.exp2(np.ceil(np.log2(d) - 1e-10))
    payoff = d / raw_d
    floor = np.maximum.reduce((raw_d, g, p)) * payoff
    return t, g, p, d, floor, payoff


def quality_data(task):
    lines = rows(RUNS / task / "metrics.jsonl")
    def col(key):
        return np.array([np.nan if line[key] is None else line[key] for line in lines])
    return (col("step"), col("precision"), col("acc_center_rms_sigma"),
            col("acc_cov_trace_bias"), col("acc_radial_ks"), col("min_cov_eig_ratio"))


def plot_rates():
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        t, g, p, d, floor, payoff = rate_data(task)
        ax.fill_between(t, 1/256, 1.15, where=floor > d * 1.001,
                        color="#f4cccc", alpha=.28, step="post")
        ax.step(t, g, where="post", color="#247ba0", lw=1.45, label="G applied scale")
        ax.step(t, p, where="post", color="#df8e1d", lw=1.45, label="prior applied scale")
        ax.step(t, d, where="post", color="#383838", lw=1.65, label="D applied scale: measured")
        ax.step(t, floor, where="post", color="#c43131", ls="--", lw=1.5,
                label="D floor on old trajectory: counterfactual")
        ax.set_yscale("log", base=2)
        ax.set_ylim(1/256, 1.15)
        ax.set_ylabel(f"{task}\nLR / base LR")
        ax.grid(alpha=.20)
        affected = int(np.count_nonzero(floor > d * 1.001))
        ax.text(.99, .08, f"floor active on {affected:,}/7,000 old updates",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    axes[-1].set_xlabel("Training update")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Native100 learning-rate dynamics\nRed dashed line is predicted from saved runs, not a measured new run", fontsize=13)
    fig.savefig(OUT / "native_rate_dynamics.png", dpi=180)
    fig.savefig(OUT / "native_rate_dynamics.svg")
    plt.close(fig)


def plot_quality():
    fig, axes = plt.subplots(3, 5, figsize=(17, 8.2), sharex=True,
                             constrained_layout=True)
    specs = (
        ("Noisy precision", 1, .970, "above"),
        ("Centre RMS / data sigma", 2, .20, "below"),
        ("Covariance trace bias", 3, .10, "band"),
        ("Radial KS", 4, .040, "below"),
        ("Minimum covariance eig ratio", 5, .40, "above"),
    )
    for i, task in enumerate(TASKS):
        data = quality_data(task)
        t = data[0]
        for j, (title, k, limit, gate) in enumerate(specs):
            ax = axes[i, j]
            v = data[k]
            ax.plot(t, v, color="#2c5f8a", marker="o", ms=2.4, lw=1.2)
            if gate == "band":
                ax.axhspan(-limit, limit, color="#d8edd8", alpha=.58)
                ax.axhline(-limit, color="#497d49", ls=":", lw=.9)
                ax.axhline(limit, color="#497d49", ls=":", lw=.9)
            else:
                ax.axhline(limit, color="#af3a3a", ls="--", lw=1)
            ax.axvspan(6000, 7000, color="#eee", alpha=.55)
            ax.grid(alpha=.18)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(task)
            if i == 2:
                ax.set_xlabel("Update")
            ax.set_xlim(0, 7000)
    fig.suptitle("Measured quality on the same three frozen native100 runs\nGrey: required terminal window; horizontal lines/band: frozen accuracy limits", fontsize=13)
    fig.savefig(OUT / "native_quality_dynamics.png", dpi=180)
    fig.savefig(OUT / "native_quality_dynamics.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_rates()
    plot_quality()
