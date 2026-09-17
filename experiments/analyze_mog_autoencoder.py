#!/usr/bin/env python
"""Export the routing scout leaderboard and learning curves."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/mog-autoencoder"))
    parser.add_argument("--allow-source-differences", action="store_true",
                        help="Allow reviewed training-source differences; retain and verify each source hash")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.run_dir / "leaderboard.json").read_text())
    configs = {m["arm"]: json.loads((args.run_dir / m["arm"] / "config.json").read_text()) for m in rows}
    for arm, config in configs.items():
        source = (args.run_dir / arm / "source.py").read_bytes()
        assert hashlib.sha256(source).hexdigest() == config["source_sha256"], f"source hash mismatch: {arm}"
    ignored = {"arm", "arms", "out", "source_sha256"}
    shared = [{k: v for k, v in c.items() if k not in ignored} for c in configs.values()]
    assert all(c == shared[0] for c in shared), "shared configurations differ"
    if len({c["source_sha256"] for c in configs.values()}) != 1:
        assert args.allow_source_differences, "training sources differ; review before allowing"
        print("Reviewed source differences allowed; original hashes retained in configs.json.")
    assert all(m["sigma"] == configs[m["arm"]]["sigma"] for m in rows), "sigma changed"
    assert all(m["step"] == configs[m["arm"]]["steps"] for m in rows), "incomplete training budget"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [("hq", "Unconditional high-quality fraction", rows[0]["reference_hq"]),
             ("modes", "Modes covered / 100", 100),
             ("recon_mse", "Held-out reconstruction MSE", None),
             ("width_ratio", "Mode width / real width", 1)]
    for index, row in enumerate(rows):
        arm = row["arm"]
        history = [json.loads(line) for line in (args.run_dir / arm / "history.jsonl").read_text().splitlines()]
        for ax, (key, title, target) in zip(axes.flat, specs):
            valid = [m for m in history if m[key] is not None]
            if valid:
                ax.plot([m["step"] for m in valid], [m[key] for m in valid],
                        marker="o", label=arm, color=f"C{index % 10}")
            ax.set(title=title, xlabel="Updates")
            ax.grid(alpha=.2)
    for ax, (_, _, target) in zip(axes.flat, specs):
        if target is not None:
            ax.axhline(target, color="gray", ls="--", alpha=.5)
    axes[0, 0].legend()
    axes[1, 0].set_yscale("log")
    fig.suptitle("Fixed sigma, one shared seed, 400 particles, 2D latent")
    fig.tight_layout()
    fig.savefig(args.out / "learning_curves.png", dpi=160)
    plt.close(fig)
    for name in ("LEADERBOARD.md", "leaderboard.json"):
        shutil.copyfile(args.run_dir / name, args.out / name)
    for row in rows:
        shutil.copyfile(args.run_dir / row["arm"] / "samples.png", args.out / (row["arm"] + ".png"))
    (args.out / "configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    print((args.out / "LEADERBOARD.md").read_text())


if __name__ == "__main__":
    main()
