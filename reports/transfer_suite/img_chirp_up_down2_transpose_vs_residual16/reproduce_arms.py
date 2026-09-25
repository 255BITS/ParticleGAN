#!/usr/bin/env python3
"""Break: published transpose12 FAIL vs residual16 PASS on novel chirp_up_down2."""
from __future__ import annotations
import gzip, json, sys, time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch

REPO = Path(__file__).resolve().parents[3]
if not (REPO / "benchmarks").exists():
    REPO = Path("/home/mikkel/sliders-outscore/ParticleGAN")
sys.path.insert(0, str(REPO))

from benchmarks.transfer_suite import image_solvability as solv
from benchmarks.transfer_suite import image_tasks as host

EP = Path(__file__).resolve().parent
TIP = (EP / "tip_sha.txt").read_text().strip() if (EP / "tip_sha.txt").exists() else "UNKNOWN"
CARDS = json.loads((EP / "cards.json").read_text())
_ORIG = host.templates
bg, fg = 0.05, 0.9


def make_task():
    task = deepcopy(next(t for t in host.TASKS if t["name"] == "img_intensity2"))
    task.update(
        name="img_chirp_up_down2",
        pattern="chirp_up_down2",
        modes=2,
        family="image_chirp_spectrogram",
        importance_reason=(
            "Novel application: DSP spectrogram chirp polarity on 8x8 grayscale — "
            "rising frequency ridge (y grows with x) vs falling ridge. Product-adjacent "
            "to audio/sonogram UIs and radar chirp polarity. Distinct from moiré beat, "
            "wave polarity, and glyph farms."
        ),
        limitations="Finite 32-particle prior; 8x8 grayscale custom template; not natural-image fidelity.",
        thresholds=dict(
            hq_min=0.9,
            modes=2,
            quality_rmse=0.05,
            min_mode_fraction=0.5 / 2,
            observations=24,
            minimum_stable_checks=5,
        ),
    )
    return task


def templates_chirp(spec):
    if spec["pattern"] != "chirp_up_down2":
        return _ORIG(spec)
    images = torch.zeros(spec["modes"], 1, 8, 8)
    images[0, 0] = bg
    images[1, 0] = bg
    for x in range(8):
        y_up = int(round(1 + x * 5 / 7))
        y_dn = int(round(6 - x * 5 / 7))
        for dy in (-1, 0, 1):
            yu, yd = y_up + dy, y_dn + dy
            if 0 <= yu < 8:
                images[0, 0, yu, x] = fg if dy == 0 else 0.55
            if 0 <= yd < 8:
                images[1, 0, yd, x] = fg if dy == 0 else 0.55
    return images


def main():
    task = make_task()
    summary = {
        "task": task["name"],
        "tip_sha": TIP,
        "principal": "chirp_up_down2 rising vs falling spectrogram ridge: published transpose12 vs residual16",
        "published_winner": "image suite baseline transpose12 (RpGAN+b_cap+particles)",
        "arms": {},
    }
    index_arms = {}
    with patch.object(host, "templates", templates_chirp):
        centers = templates_chirp(task)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            for ax, img, title in zip(axes, centers, ["rising chirp", "falling chirp"]):
                ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle("chirp_up_down2 templates (DSP spectrogram)")
            fig.tight_layout()
            fig.savefig(EP / "templates.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            (EP / "templates_error.txt").write_text(str(exc))

        for card in CARDS:
            name = card["name"]
            print("START", name, task["name"], flush=True)
            t0 = time.perf_counter()
            torch.set_num_threads(1)
            result = solv.episode(task, card)
            secs = time.perf_counter() - t0
            sustained_bool, modes, hq = solv.quality(result)
            conv = result.get("convergence") or {}
            live = result.get("live") or {}
            obs = result.get("observations") or []
            last = obs[-1] if obs else {}
            quality = {
                "sustained": bool(sustained_bool),
                "confirmed_step": conv.get("confirmed_step"),
                "modes": live.get("modes", modes),
                "hq": live.get("hq", hq),
                "mean_rmse": last.get("mean_rmse"),
                "error": result.get("error"),
                "seconds": result.get("seconds", secs),
                "last_obs": {
                    "hq": last.get("hq"),
                    "mean_rmse": last.get("mean_rmse"),
                    "modes": last.get("modes"),
                },
            }
            verdict = "PASS" if sustained_bool else "FAIL"
            slim = {k: v for k, v in result.items() if k not in ("actions", "protocol")}
            raw = (json.dumps(slim, sort_keys=True, allow_nan=False, default=str) + "\n").encode()
            (EP / f"{name}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
            arm_summary = {
                "arm": name,
                "verdict": verdict,
                "modes": quality["modes"],
                "hq": quality["hq"],
                "mean_rmse": quality["mean_rmse"],
                "confirmed_step": quality["confirmed_step"],
                "seconds": quality["seconds"],
            }
            (EP / f"{name}.summary.json").write_text(json.dumps(arm_summary, indent=2) + "\n")
            summary["arms"][name] = {"card": card, "verdict": verdict, "quality": quality, "seconds": secs}
            index_arms[name] = arm_summary
            print(json.dumps(arm_summary, default=str), flush=True)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                samples = None
                for key in ("samples", "generated", "particles", "x"):
                    if key in live and live[key] is not None:
                        samples = live[key]
                        break
                if samples is None and obs:
                    for key in ("samples", "generated", "particles"):
                        if key in last and last[key] is not None:
                            samples = last[key]
                            break
                if samples is not None:
                    arr = np.asarray(samples)
                    if arr.ndim >= 3:
                        n = min(arr.shape[0], 16)
                        cols = 4
                        rows = (n + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
                        axes = np.atleast_1d(axes).ravel()
                        for i in range(rows * cols):
                            axes[i].axis("off")
                            if i < n:
                                img = arr[i]
                                if img.ndim == 3:
                                    img = img[0]
                                axes[i].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        fig.suptitle(f"{name} samples")
                        fig.tight_layout()
                        fig.savefig(EP / f"{name}_scatter.png", dpi=120)
                        plt.close(fig)
            except Exception as exc:
                (EP / f"{name}_scatter_error.txt").write_text(str(exc))

    hit = (
        summary["arms"]["baseline_transpose12"]["verdict"] == "FAIL"
        and summary["arms"]["residual16"]["verdict"] == "PASS"
    )
    summary["hit"] = hit
    summary["soft_miss"] = not hit
    (EP / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (EP / "index.json").write_text(json.dumps({"tip_sha": TIP, "hit": hit, "arms": index_arms}, indent=2) + "\n")
    print("SUMMARY", json.dumps({"hit": hit, "arms": {k: v["verdict"] for k, v in summary["arms"].items()}}), flush=True)


if __name__ == "__main__":
    main()
