"""Draw the proposed benchmark and its exact posteriors; no learned models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    rng = np.random.default_rng(24001)
    ij = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 2)
    means = ij.astype(float) - 4.5
    labels = 2 * (ij[:, 0] % 2) + ij[:, 1] % 2
    palette = np.array(["#dc6255", "#347bb0", "#43a07c", "#a17ac0"])
    observation = np.array([-0.5, -0.5])
    sigma = 0.03
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True)
    for row, allowed in enumerate((np.ones(100, dtype=bool), labels == 0)):
        ids = np.flatnonzero(allowed)
        selected = rng.choice(ids, 5000)
        samples = means[selected] + sigma * rng.normal(size=(len(selected), 2))
        axes[row, 0].scatter(*samples.T, c=palette[labels[selected]], s=3, alpha=0.55, linewidths=0)
        axes[row, 0].set_title("Target: all 100 modes" if row == 0 else "Target: class 0 (25 modes)")
        for col, tau in enumerate((0.15, 0.8, 2.0), start=1):
            # Observation u = x0 + tau * epsilon, equivalent to xt / sqrt(alpha_bar).
            variance = sigma**2 + tau**2
            log_weight = -np.square(means[ids] - observation).sum(1) / (2 * variance)
            weight = np.exp(log_weight - log_weight.max())
            weight /= weight.sum()
            posterior_mean = means[ids] + sigma**2 / variance * (observation - means[ids])
            posterior_std = np.sqrt(sigma**2 * tau**2 / variance)
            local_ids = rng.choice(len(ids), 5000, p=weight)
            samples = posterior_mean[local_ids] + posterior_std * rng.normal(size=(5000, 2))
            ax = axes[row, col]
            ax.scatter(*samples.T, c=palette[labels[ids[local_ids]]], s=3, alpha=0.5, linewidths=0)
            visible = weight > 0.002
            ax.scatter(*posterior_mean[visible].T, s=1400 * weight[visible],
                       facecolors="none", edgecolors="#333333", linewidths=0.8)
            ax.scatter(*observation, c="black", marker="x", s=65, linewidths=1.8, zorder=5)
            ax.set_title(f"Exact posterior: noise SD {tau:g}")
        for ax in axes[row]:
            ax.scatter(*means.T, c="#bbbbbb", marker="+", s=13, linewidths=0.6, zorder=0)
            ax.set_aspect("equal")
            ax.set_xlim(-5.2, 5.2)
            ax.set_ylim(-5.2, 5.2)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("No class condition\ny")
    axes[1, 0].set_ylabel("Class 0 requested\ny")
    for ax in axes[1]:
        ax.set_xlabel("x")
    fig.suptitle("100 Gaussians: distinguish coverage, class fidelity, and denoising uncertainty", fontsize=16)
    fig.text(0.5, 0.015,
             "Illustration of ground truth, not training results. Black x: fixed noisy observation u = (-0.5, -0.5). "
             "Circle area: posterior mode probability.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(Path(__file__).with_name("preview.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
