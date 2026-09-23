"""Plot recorded held-out LR actions; requires the optional matplotlib extra."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("reports/learned_lr/full_suite/results.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/learned_lr/lr_actions.svg"))
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    colors = {"constant": "#888888", "cosine": "#0072B2", "learned": "#D55E00", "time_only": "#009E73"}
    labels = {"constant": "Constant", "cosine": "Cosine", "learned": "Learned feedback", "time_only": "Feedback zeroed"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    budget = report["protocol"]["budgets"]["mode_hold"]
    for axis, role, title in zip(axes, ("g", "d"), ("Generator / prior", "Discriminator")):
        for row in report["rows"]:
            kind = row["controller"]
            trace = [p for p in row["toys"]["mode_hold"]["controller_trace"] if p["role"] == role]
            axis.plot([p["step"] / budget * 100 for p in trace], [p["multiplier"] for p in trace],
                      color=colors[kind], label=labels[kind], lw=2,
                      linestyle="--" if kind in ("constant", "time_only") else "-")
        axis.set_title(title)
        axis.set_xlabel("Training budget completed (%)")
        axis.set_yscale("log")
        axis.set_ylim(.045, 2.1)
        axis.set_yticks([.05, .1, .2, .5, 1, 2], labels=["0.05", "0.1", "0.2", "0.5", "1", "2"])
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Multiplier of initial learning rate")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Frozen policy on the held-out eight-mode ring", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})


if __name__ == "__main__":
    main()
