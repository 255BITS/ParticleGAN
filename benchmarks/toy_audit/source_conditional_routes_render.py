"""Render genuine conditional-route source states, including capped attempts.

Only saved training predictions are rendered. No optimizer or source model is
loaded and no intermediate state is interpolated from endpoint metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


BLUE, GREEN, GRAY = "#2875b9", "#18846d", "#8e8e8e"
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fafafa",
                     "axes.facecolor": "#fafafa", "axes.spines.top": False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limits(values):
    flat = np.concatenate([v.reshape(-1) for v in values])
    assert np.isfinite(flat).all()
    lo, hi = float(flat.min()), float(flat.max())
    pad = max(.05, .08 * (hi - lo))
    return lo - pad, hi + pad


def saved(rows, arrays, row, name):
    return arrays[row["array_prefix"] + "__" + name]


def make_frames(rows, arrays, receipt):
    # Original held-out scene 0, midpoint tick 31; each class is a separate group.
    group_ids = (2, 7)
    first = rows[0]
    groups = saved(rows, arrays, first, "group")
    masks = [groups == group for group in group_ids]
    assert all(mask.sum() == 512 for mask in masks)
    for mask, c in zip(masks, (0, 1)):
        assert np.all(saved(rows, arrays, first, "c")[mask] == c)
        assert np.all(saved(rows, arrays, first, "tick")[mask] == 31)
    extents = [[], []]
    for row in rows:
        for field in ("reference", "live_prior", "ema_prior"):
            values = saved(rows, arrays, row, field)
            for mask in masks:
                x = values[mask]
                for points in (x[:, :2], x[:, :2] + x[:, 2:4], x[:, 4:]):
                    for dim in (0, 1):
                        extents[dim].append(points[:, dim])
    bounds = [limits(v) for v in extents]
    paired_bounds = limits([saved(rows, arrays, row, field)[:, 4:]
                            for row in rows for field in ("reference", "live_encoded", "ema_encoded")])
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.3), dpi=100)
    fig.subplots_adjust(left=.065, right=.955, bottom=.20, top=.84, wspace=.36, hspace=.44)
    frames = []
    for index, row in enumerate(rows):
        for ax in axes.flat:
            ax.clear(); ax.grid(alpha=.15)
        for cls, mask in enumerate(masks):
            for column, (label, color) in enumerate((("live", BLUE), ("ema", GREEN))):
                ax = axes[cls, column]
                real = saved(rows, arrays, row, "reference")[mask]
                # The fixed first 32 draws of each source panel, without resampling.
                x = saved(rows, arrays, row, label + "_prior")[mask][:32]
                state, expected, nxt = x[:, :2], x[:, :2] + x[:, 2:4], x[:, 4:]
                ax.scatter(real[:128, 0], real[:128, 1], s=7, color=GRAY, alpha=.35)
                ax.quiver(real[:32, 0], real[:32, 1], real[:32, 2], real[:32, 3],
                          angles="xy", scale_units="xy", scale=1, width=.003, color=GRAY, alpha=.5)
                ax.scatter(state[:, 0], state[:, 1], s=13, marker=".", color=color)
                ax.quiver(state[:, 0], state[:, 1], x[:, 2], x[:, 3],
                          angles="xy", scale_units="xy", scale=1, width=.004, color=color, alpha=.7)
                ax.scatter(expected[:, 0], expected[:, 1], s=22, facecolors="none", edgecolors=color, linewidths=.8)
                ax.scatter(nxt[:, 0], nxt[:, 1], s=20, marker="^", color=color, alpha=.55)
                ax.plot(np.stack([expected[:, 0], nxt[:, 0]]), np.stack([expected[:, 1], nxt[:, 1]]),
                        color=color, lw=.65, alpha=.38, ls=":")
                ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="Physical x", ylabel="Physical y",
                       title=f"{label.upper()} joint draw · class {cls} (upper p={.8 if cls == 0 else .3})")
        ax = axes[0, 2]
        target = saved(rows, arrays, row, "reference")[::64, 4:].reshape(-1)
        for label, color in (("live", BLUE), ("ema", GREEN)):
            prediction = saved(rows, arrays, row, label + "_encoded")[::64, 4:].reshape(-1)
            ax.scatter(target, prediction, s=8, alpha=.22, color=color, label=label.upper())
        ax.plot(paired_bounds, paired_bounds, color=GRAY, ls="--", lw=1)
        ax.set(xlim=paired_bounds, ylim=paired_bounds, xlabel="Actual analytic next coordinate",
               ylabel="Actual encoded prediction", title="Paired inference from observed state/action")
        ax.legend(fontsize=8)
        ax = axes[1, 2]
        window = rows[:index + 1]
        for label, color in (("live", BLUE), ("ema", GREEN)):
            ax.plot([r["step"] for r in window], [r["metrics"][label + "_prior"]["consistency_mean"] for r in window],
                    color=color, marker=".", label=label.upper())
        max_error = max(r["metrics"][label + "_prior"]["consistency_mean"]
                        for r in rows for label in ("live", "ema"))
        ax.set(xlim=(0, max(1, steps[-1])), ylim=(0, 1.1 * max_error), xlabel="Completed updates (actual saved states)",
               ylabel="Mean physical ‖next − state − action‖", title="Full held-out joint consistency · 20,480 rows")
        ax.legend(fontsize=8)
        m = row["metrics"]
        fig.suptitle(f"Route transitions · {receipt['geometry']} training geometry · actual state {row['step']} / 28,000\n"
                     "Can shared latent branches learn a joint state/action/next-state law on held-out scenes?",
                     y=.98, fontsize=14)
        sw = fig.text(.5, .875, f"Full-test normalized joint SW1: live {m['live_prior']['joint_sw1']:.4f}, EMA {m['ema_prior']['joint_sw1']:.4f}"
                      f"  ·  paired next MSE: live {m['live_encoded']['next_state_mse']:.5f}, EMA {m['ema_encoded']['next_state_mse']:.5f}",
                      ha="center", fontsize=10)
        legend = fig.legend(handles=[Line2D([], [], color=GRAY, marker=".", ls="", label="Analytic states/actions"),
                                     Line2D([], [], color=BLUE, marker=".", ls="-", label="Actual state + action arrow"),
                                     Line2D([], [], color=BLUE, marker="o", markerfacecolor="none", ls="", label="state + action"),
                                     Line2D([], [], color=BLUE, marker="^", ls="", label="Independent next branch")],
                            loc="lower center", bbox_to_anchor=(.5, .115), ncol=4, frameon=False, fontsize=9)
        note = fig.text(.5, .042, f"{receipt['fresh_execution_status']}: 120-second entry cap; {receipt['training']['completed_updates']} completed / 28,000 required updates. No convergence acceptance gate declared.\n"
                        "Joint panels: scene (0, 0, .27), tick 31; first 32 actual MoG draws. Dotted gaps expose violated joint relationships.\n"
                        "Paired panel: every 64th held-out row, both coordinates. All metrics use 40 contexts × 512 rows with original served MoG noise.\n"
                        "Each frame is a captured training state. Connected metric dots do not create interpolated model states; no qualification credit.",
                        ha="center", fontsize=8.5)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128))
        sw.remove(); legend.remove(); note.remove()
    plt.close(fig)
    return frames, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for catalog_id in ("source-family-06", "source-family-07"):
        root = args.artifacts / catalog_id
        receipt = json.loads((root / "receipt.json").read_text())
        training = receipt["training"]
        data = root / "training"
        for name in ("observations.npz", "observations.jsonl"):
            assert sha(data / name) == training[name + "_sha256"]
        rows = [json.loads(line) for line in (data / "observations.jsonl").read_text().splitlines()]
        assert len(rows) == training["observer_states"]
        assert all(r["complete_owner_and_rng_pure"] for r in rows)
        arrays = np.load(data / "observations.npz", allow_pickle=False)
        frames, steps = make_frames(rows, arrays, receipt)
        path = args.out / (catalog_id + ".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, optimize=True,
                       duration=[600] * (len(frames) - 1) + [2400])
        poster = path.with_suffix(".png")
        frames[-1].convert("RGB").save(poster)
        decoded = Image.open(path)
        assert decoded.n_frames == len(frames)
        frame_hashes = []
        for i in range(decoded.n_frames):
            decoded.seek(i)
            frame_hashes.append(hashlib.sha256(np.asarray(decoded.convert("RGB")).tobytes()).hexdigest())
        assert len(set(frame_hashes)) == len(frames)
        metadata[catalog_id] = dict(gif=path.name, sha256=sha(path), bytes=path.stat().st_size,
                                   poster=poster.name, poster_sha256=sha(poster), frames=len(frames),
                                   actual_states=steps, interpolation=False, decoded_rgb_sha256=frame_hashes,
                                   observations_npz_sha256=sha(data / "observations.npz"),
                                   observations_jsonl_sha256=sha(data / "observations.jsonl"),
                                   training_status=receipt["fresh_execution_status"], sampling=rows[0]["metrics"]["panel"],
                                   renderer_sha256=sha(__file__))
        print(json.dumps(dict(catalog_id=catalog_id, frames=len(frames), bytes=path.stat().st_size)), flush=True)
    (args.out / "media.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
