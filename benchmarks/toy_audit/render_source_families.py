"""Render source-family GIFs from real captured training-state observations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


COLORS = ("#b94b5a", "#187e70", "#7664aa")
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fafafa",
                     "axes.facecolor": "#fafafa", "axes.spines.top": False,
                     "axes.spines.right": False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames_for(family, rows, arrays, receipt):
    arms = ("collapsed", "paired", "supervised") if family == "sign" else ("baseline", "combined", "supervised")
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    steps = [r["step"] for r in by_arm[arms[0]]]
    assert all([r["step"] for r in values] == steps for values in by_arm.values())
    max_step = 200 if family == "sign" else 250
    names = ("Joint sign flip", "Paired GAN", "Supervised · rejected") if family == "sign" else (
        "GAN-only slow expert", "GAN + safe-fast", "Safe-fast only · rejected")
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.5), dpi=100)
    fig.subplots_adjust(left=.07, right=.98, bottom=.20, top=.84, wspace=.36, hspace=.43)
    frames = []
    for index, step in enumerate(steps):
        for ax in axes.flat:
            ax.clear(); ax.grid(alpha=.18)
        for column, (arm, name, color) in enumerate(zip(arms, names, COLORS)):
            row = by_arm[arm][index]
            metric, prefix = row["metrics"], row["array_prefix"]
            if family == "sign":
                positions = arrays[prefix + "__position"]
                # Same 12 fixed rows at each state; complete 80-row gate above.
                for k in range(12):
                    axes[0, column].plot(positions[:, k, 0], positions[:, k, 1], color=color, alpha=.45, lw=1)
                axes[0, column].scatter(positions[0, :12, 0], positions[0, :12, 1], s=10, color="#777")
                axes[0, column].scatter(positions[-1, :12, 0], positions[-1, :12, 1], s=13, color=color)
                axes[0, column].add_patch(plt.Circle((0, 0), .15, fill=False, color="#777", ls="--"))
                beta_label = f" · β {metric['beta']:.4f}" if "beta" in metric else ""
                axes[0, column].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel="True-plant x", ylabel="True-plant y",
                                    title=f"{name}\nlandings {metric['landings']:.1%} · α {metric['alpha']:.4f}{beta_label}")
                prediction, target = arrays[prefix + "__prediction"], arrays[prefix + "__target"]
                axes[1, column].scatter(target.reshape(-1), prediction.reshape(-1), s=8, alpha=.3, color=color)
                axes[1, column].plot([-1, 1], [-1, 1], ls="--", color="#777", lw=1)
                axes[1, column].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="Required paired action", ylabel="Actual clean action",
                                    title=f"Relative paired MSE {metric['strict_paired']['relative_mse']:.6g}")
            else:
                states = arrays[prefix + "__states"]
                landed = arrays[prefix + "__landed"]
                crashed = arrays[prefix + "__crashed"]
                for k in range(16):
                    first_contact = next((i for i, state in enumerate(states[1:, k], 1)
                                          if state[1] <= 0 or abs(state[0]) > 1.6 or state[1] > 2.4), len(states) - 1)
                    path = states[:first_contact + 1, k]
                    axes[0, column].plot(path[:, 0], path[:, 1], color=color, alpha=.4, lw=1)
                    axes[0, column].scatter(path[-1, 0], path[-1, 1], s=12,
                                            color="#187e70" if landed[k] else "#b94b5a" if crashed[k] else "#777")
                axes[0, column].plot([-.35, .35], [0, 0], color="#333", lw=4)
                axes[0, column].set(xlim=(-.5, .5), ylim=(-.1, 2.5), xlabel="True-plant lateral position", ylabel="Altitude",
                                    title=f"{name}\nlandings {metric['landings']:.1%} · sink {metric['sink']:.3f}")
                prefix_rows = by_arm[arm][:index + 1]
                x = [r["step"] for r in prefix_rows]
                axes[1, column].plot(x, [r["metrics"]["strict_landing"]["restricted_mean_steps"] for r in prefix_rows],
                                     color=color, marker=".", lw=1.5, label="All starts · restricted mean")
                axes[1, column].plot(x, [r["metrics"]["mean_steps"] for r in prefix_rows],
                                     color="#777", ls="--", lw=1, label="Successful starts only")
                axes[1, column].axhline(28, ls=":", color="#333", lw=1)
                axes[1, column].set(xlim=(0, max_step), ylim=(15, 49), xlabel="Completed updates", ylabel="Plant steps",
                                    title=f"Crashes {metric['crash_rate']:.1%} · timeouts {metric['strict_landing']['timeout_rate']:.1%}")
                if column == 0:
                    axes[1, column].legend(fontsize=7, loc="lower right")
        if family == "sign":
            fig.suptitle(f"Sign correspondence · actual training state {step} / 200\n"
                         "Can paired residual training resolve a marginally invisible wrong action sign?", y=.98, fontsize=14)
            footer = ("80 fixed starts × 40 plant steps; paths show the same 12 starts. Divergent paths extend outside the ±1.2 view.\n"
                      "Action panels include all 256 held-out states × two coordinates. Target diagonal is fixed.\n"
                      "Only the paired GAN is an accepted controller fix; supervised success is deliberately rejected. Final declared protocol: PASS.")
        else:
            fig.suptitle(f"Safe-fast landing · actual training state {step} / 250\n"
                         "Can a plant-cost term improve a slow expert match while the GAN remains active?", y=.98, fontsize=14)
            footer = ("200 fixed starts × 48 plant steps; paths show the same 16 starts and stop at their first actual terminal event.\n"
                      "All-start restricted means charge crashes/timeouts the full horizon; dotted line is the new 28-step gate.\n"
                      "The plant law and original acceptance gate are unchanged. Supervised-only success is rejected. Final declared protocol: PASS.")
        note = fig.text(.5, .055, footer, ha="center", fontsize=9)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128))
        note.remove()
    plt.close(fig)
    return frames, steps


def native_frames(rows, arrays, receipt):
    arms = ("current", "fixed")
    pre = [r for r in rows if r["arm"] == "pretrain"]
    grouped = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    assert [r["step"] for r in grouped["current"]] == [r["step"] for r in grouped["fixed"]]
    timeline = [("pretrain", r["step"], r) for r in pre]
    timeline.extend(("finetune", r["step"], index) for index, r in enumerate(grouped["current"]))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), dpi=100)
    fig.subplots_adjust(left=.085, right=.97, top=.83, bottom=.19, wspace=.32, hspace=.52)
    frames = []
    for phase, step, key in timeline:
        for ax in axes.flat:
            ax.clear(); ax.grid(alpha=.18)
        if phase == "pretrain":
            row = key
            prefix = row["array_prefix"]
            action, prediction, paired = (arrays[prefix + "__" + field] for field in (
                "target", "live_prediction", "paired_prediction"))
            for column, (values, name) in enumerate(((paired, "Supervised current-action reconstruction"),
                                                    (prediction, "Same encoder given previous command"))):
                axes[0, column].scatter(action[::8, 0], values[::8, 0], s=8, alpha=.25, color=COLORS[column])
                axes[0, column].plot([-1, 1], [-1, 1], color="#777", ls="--", lw=1)
                axes[0, column].set(xlim=(-1.1, 1.1), ylim=(-1.7, 1.7), xlabel="Expert action", ylabel="Actual action prediction", title=name)
            prefix_rows = [r for r in pre if r["step"] <= step]
            for column, (field, title) in enumerate((("paired_reconstruction_mse", "Paired reconstruction MSE"),
                                                    ("live_action_mse", "Wrong-context controller action MSE"))):
                axes[1, column].plot([r["step"] for r in prefix_rows], [r["metrics"][field] for r in prefix_rows],
                                     color=COLORS[column], marker=".")
                axes[1, column].set(xlim=(0, 250), ylim=(0, 2.2), xlabel="Completed shared pretraining updates", ylabel="MSE", title=title)
            subtitle = "One shared supervised reconstruction stage, before either adversarial arm"
        else:
            for column, arm in enumerate(arms):
                row, prefix_rows = grouped[arm][key], grouped[arm][:key + 1]
                prefix = row["array_prefix"]
                previous = arrays[prefix + "__previous"][:, 0]
                order = np.argsort(previous)
                target = arrays[prefix + "__target"][:, 0]
                axes[0, column].plot(previous[order], target[order], color="#333", lw=1.2, label="Expert target")
                for field, color, label in (("live_prediction", COLORS[column], "Live"),
                                             ("ema_prediction", "#7664aa", "EMA")):
                    axes[0, column].scatter(previous[::8], arrays[prefix + "__" + field][::8, 0],
                                            s=7, color=color, alpha=.27, label=label)
                axes[0, column].set(xlim=(-1, 1), ylim=(-1.7, 1.7), xlabel="Previous command", ylabel="Predicted action",
                                    title="Observation critics · all owners" if arm == "current" else "Live latent joint · scoped owners")
                axes[0, column].legend(fontsize=7, loc="upper right")
                for field, color, label in (("live_action_mse", COLORS[column], "Live"),
                                             ("ema_action_mse", "#7664aa", "EMA")):
                    axes[1, column].plot([r["step"] for r in prefix_rows], [r["metrics"][field] for r in prefix_rows],
                                         color=color, marker=".", lw=1.5, label=label)
                axes[1, column].axhline(.18, ls="--", color="#333", lw=1, label="Original fixed ≤ .18")
                axes[1, column].axhline(row["metrics"]["strict_paired_ema"]["neutral_mse"] * .10,
                                       ls=":", color="#777", lw=1, label="New paired relative ≤ .10")
                axes[1, column].set(xlim=(0, 400), ylim=(0, 2.2), xlabel="Completed finetune updates", ylabel="Paired action MSE",
                                    title=f"Live {row['metrics']['live_action_mse']:.4f} · EMA {row['metrics']['ema_action_mse']:.4f}")
                axes[1, column].legend(fontsize=7)
            subtitle = "Matched initial host and original input streams; arms also change trainable ownership"
        fig.suptitle(f"Particle adapter toy · {phase} actual state {step}\n{subtitle}", y=.98, fontsize=13)
        note = fig.text(.5, .052, "2,048 fixed held-out contexts; plots display every eighth actual prediction; metrics include every row.\n"
                        "250 shared pretraining updates + 400 updates per adversarial arm. CPU1 is distinct from the source CPU4 CLI.\n"
                        "Full original research gate: FAIL (fixed EMA .221889 > .18). Protocol software checks do not establish convergence.",
                        ha="center", fontsize=9)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128))
        note.remove()
    plt.close(fig)
    return frames, [f"{phase}:{step}" for phase, step, _ in timeline]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for family in ("sign", "landing", "native"):
        root = args.artifacts / family
        receipt = json.loads((root / "receipt.json").read_text())
        assert receipt["status"] == "COMPLETE"
        assert sha(root / "observations.npz") == receipt["observations_sha256"]
        rows = [json.loads(line) for line in (root / "observations.jsonl").read_text().splitlines()]
        assert len(rows) == receipt["observation_count"] and all(r["observation_state_and_rng_pure"] for r in rows)
        arrays = np.load(root / "observations.npz", allow_pickle=False)
        frames, steps = (native_frames(rows, arrays, receipt) if family == "native" else frames_for(family, rows, arrays, receipt))
        path = args.out / (family + ".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, optimize=True,
                       duration=[320] * (len(frames) - 1) + [2400])
        poster = path.with_suffix(".png")
        frames[-1].convert("RGB").save(poster)
        metadata[family] = {"gif": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
                            "poster": poster.name, "poster_sha256": sha(poster),
                            "frames": len(frames), "actual_states": steps, "interpolation": False,
                            "source_observations_sha256": receipt["observations_sha256"],
                            "sampling": receipt["sampling"]}
        print(json.dumps({"family": family, "frames": len(frames), "bytes": path.stat().st_size}), flush=True)
    (args.out / "media.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
