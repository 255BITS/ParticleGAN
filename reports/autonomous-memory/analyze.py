"""Rebuild diagnostics: python reports/autonomous-memory/analyze.py RUN_DIRECTORY ... [--out REPORT_DIRECTORY]."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.autonomous_memory import trajectory_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    out = args.out
    directories = {}
    for variant in ("shared", "frozen_writer", "no_memory"):
        matches = [source / variant for source in args.source
                   if (source / variant / "trajectories.npz").exists()]
        if len(matches) > 1:
            parser.error(f"multiple runs for variant {variant}")
        if matches:
            directories[variant] = matches[0]
    variants = tuple(directories)
    if not variants:
        parser.error("no completed trajectories in source directory")
    summaries = {v: json.loads((directories[v] / "summary.json").read_text()) for v in variants}
    horizons = {r["config"]["train_length"] for r in summaries.values()}
    if len(horizons) != 1:
        parser.error("analyze matching training horizons together")
    horizon = horizons.pop()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, axes = plt.subplots(len(variants) + 1, 5, figsize=(15, 2.75 * (len(variants) + 1)))
    for row, variant in enumerate((*variants, "real_noisy")):
        is_generated = variant != "real_noisy"
        with np.load(directories[variant if is_generated else variants[0]] / "trajectories.npz") as saved:
            paths = saved["generated" if is_generated else "real_noisy"]
        steps = np.linalg.norm(np.diff(paths, axis=1), axis=-1)
        late = steps[:, -64:].mean(1)
        early = steps[:, :15].mean(1)
        metrics = {
            "full": trajectory_metrics(paths),
            # Separately labelled: refitting late paths tests settled dynamics,
            # and must not replace the cold-start/early-reference score above.
            "last_128_refit": trajectory_metrics(paths[:, -128:]),
            "first_16_mean_step": float(early.mean()),
            "last_64_mean_step": float(late.mean()),
            "last_64_stationary_fraction": float((late < .01).mean()),
            "prefix_circle_like": {str(t): trajectory_metrics(paths[:, :t])["circle_like_fraction"]
                                   for t in (16, 32, 64, 128, 256)},
        }
        if is_generated:
            metrics["training"] = summaries[variant]
        results[variant] = metrics
        for ax, path in zip(axes[row, :4], paths[:4]):
            ax.plot(*path.T, color="0.65", linewidth=1, label="all 256")
            ax.plot(*path[:horizon].T, color="tab:blue", linewidth=2, label=f"first {horizon}")
            ax.plot(*path[-64:].T, color="tab:orange", linewidth=1, label="last 64")
            ax.scatter(*path[0], color="green", s=15)
            ax.scatter(*path[-1], color="red", s=15)
            ax.set_aspect("equal", adjustable="datalim")
        axes[row, 0].set_ylabel(variant)
        ax = axes[row, 4]
        q = np.quantile(steps, [.1, .5, .9], axis=0)
        ax.fill_between(np.arange(1, 256), q[0], q[2], alpha=.2)
        ax.plot(np.arange(1, 256), q[1])
        ax.axvline(horizon - 1, color="black", linestyle=":")
        ax.set(xlabel="transition", ylabel="step distance", ylim=(0, max(.4, q[2].max())))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Cold autonomous rollouts: first four particles, no selection; motion median and 10–90% range")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=140)
    plt.close(fig)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if len(args.source) == 1:
        (out / "provenance.json").write_bytes((args.source[0] / "provenance.json").read_bytes())
    else:
        provenance = {str(source): json.loads((source / "provenance.json").read_text())
                      for source in args.source}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: {m: v for m, v in r.items() if m != "training"} for k, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
