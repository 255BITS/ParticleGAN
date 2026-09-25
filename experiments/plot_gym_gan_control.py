#!/usr/bin/env python
"""Plot the GAN-only paired control experiment and its validation curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_dir):
    root = Path(report_dir)
    board = json.loads((root / "leaderboard.json").read_text())
    rows = {r["arm"]: r for r in board["rows"] if r["report_role"] == "selected"}
    order = [name for name in ("joint", "marginals", "legacy_joint") if name in rows]
    labels = {"joint": "5 labeled episodes · joint GAN", "marginals": "5 labeled episodes · joint + marginal GANs",
              "legacy_joint": "47 labeled episodes · legacy joint GAN", "expert": "Heuristic expert"}
    colors = {"joint": "#478bc9", "marginals": "#1b9e77", "legacy_joint": "#926ac1", "expert": "#bc8d32"}
    fig, (land, ret) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    y = np.arange(len(order))
    for i, name in enumerate(order):
        s = rows[name]["summary"]
        rate = 100*s["landing_rate"]
        lo, hi = np.array(s["landing_rate_wilson95"])*100
        land.errorbar(rate, i, xerr=[[max(0, rate-lo)], [max(0, hi-rate)]],
                      fmt="o", capsize=4, color=colors[name], markersize=8)
        land.annotate(f"{s['landing_count']}/{s['episodes']}", (rate, i),
                      xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        values = [e["return"] for e in rows[name]["episodes"]]
        # Fixed deterministic jitter separates episode marks; no subsampling.
        jitter = (np.arange(len(values)) % 9 - 4)*.025
        ret.scatter(values, i+jitter, alpha=.4, s=14, color=colors[name])
        ret.plot(s["mean_return"], i, "D", color=colors[name], markersize=8,
                 markeredgecolor="black", markeredgewidth=.7)
    for ax in (land, ret):
        ax.set_yticks(y, [labels[n] for n in order])
        ax.set_ylim(len(order)-.5, -.6)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    land.set_xlim(-7, 107)
    land.set_xlabel("Successful landings (%) · 95% Wilson intervals")
    ret.set_xlabel("Episode return · diamond = mean, dots = all episodes")
    fig.suptitle("Three-generator GAN controllers · 50 paired held-out worlds\nCheckpoints selected on separate validation episodes", fontsize=13)
    fig.savefig(root / "control_outcomes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for name in ("joint", "marginals"):
        selection = json.loads((root / "selections" / f"{name}.json").read_text())
        candidates = sorted(selection["candidates"], key=lambda x: x["step"])
        steps = [r["step"] for r in candidates]
        for ax, key, scale in ((axes[0], "landing_rate", 100), (axes[1], "mean_return", 1)):
            ax.plot(steps, [r["summary"][key]*scale for r in candidates], "o-",
                    label=labels[name], color=colors[name])
            ax.set_xlabel("Training updates")
            ax.set_xticks(steps)
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation successful landings (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("Validation mean episode return")
    axes[1].legend()
    fig.suptitle("Checkpoint selection · 20 paired validation worlds")
    fig.savefig(root / "validation_curve.png", dpi=180)
    plt.close(fig)

    # These expert records are identical across the two arms; learner-visited
    # state distributions differ and therefore stay in the detailed readout.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    names = ("joint", "marginals")
    for i, name in enumerate(names):
        score = rows[name]["expert_diagnostics"]
        axes[0].bar(i, score["standardized_action_mse"], color=colors[name])
        axes[1].bar(i, score["next_state"]["standardized_continuous_mse"], color=colors[name])
    persistence = rows["joint"]["expert_diagnostics"]["persistence_next_state"]["standardized_continuous_mse"]
    axes[1].axhline(persistence, color="#333333", linestyle="--", label="Persistence on same records")
    for ax in axes:
        ax.set_xticks([0, 1], ["Joint GAN", "Joint + marginals"])
        ax.set_ylabel("Standardized MSE (lower is better)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("G2: expert action error")
    axes[1].set_title("G3: expert successor error · 6 continuous coordinates")
    axes[1].legend(fontsize=8)
    fig.suptitle("Action fitting and successor prediction · same held-out expert records")
    fig.savefig(root / "auxiliary_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    for col, name in enumerate(("joint", "marginals")):
        history = [json.loads(line) for line in
                   (root / "training" / name / "metrics.jsonl").read_text().splitlines()]
        steps = [row["step"] for row in history]
        for key, label in (("g_adversarial_loss", "G adversarial objective"),
                           ("action_loss", "Paired action MSE")):
            axes[0, col].plot(steps, [row[key] for row in history], "o-", label=label, markersize=3)
        for suffix, label in (("_gan", "D adversarial objectives (sum)"),
                              ("_penalty", "D gradient penalties (sum)")):
            values = [sum(v for k, v in row.items() if k.startswith("d_") and k.endswith(suffix))
                      for row in history]
            axes[1, col].plot(steps, values, "o-", label=label, markersize=3)
        axes[0, col].set_title("Joint GAN" if name == "joint" else "Joint + marginal GANs")
        for ax in axes[:, col]:
            ax.set_xlabel("Training updates")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GAN and paired losses during training\nDifferent critic counts give different loss scales; control outcomes select checkpoints")
    fig.savefig(root / "adversarial_training.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/gym/lunar_lander_gan_control")
    plot(parser.parse_args().report_dir)
