#!/usr/bin/env python3
"""Break: published transpose12 FAIL vs residual16 PASS on novel vh_bars2."""
from __future__ import annotations
import gzip, json, sys, time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch

REPO = Path("/home/mikkel/sliders-outscore/ParticleGAN")
sys.path.insert(0, str(REPO))

from benchmarks.transfer_suite import image_solvability as solv
from benchmarks.transfer_suite import image_tasks as host

EP = Path(__file__).resolve().parent
TIP = (EP / "tip_sha.txt").read_text().strip() if (EP / "tip_sha.txt").exists() else "UNKNOWN"
CARDS = json.loads((EP / "cards.json").read_text())
_ORIG = host.templates


def make_task():
    task = deepcopy(next(t for t in host.TASKS if t["name"] == "img_intensity2"))
    task.update(
        name="img_vh_bars2",
        pattern="vh_bars2",
        modes=2,
        family="image_vh_bars_unpaired",
        importance_reason=(
            "Novel application: unpaired domain-translation-style toy — vertical-bar domain vs "
            "horizontal-bar domain on 8x8 grayscale. Same intensity mass, orthogonal bar orientation. "
            "Product-adjacent to unpaired translation / domain transfer demos; distinct from "
            "mask_inpaint2, sparse_obs2, soft_ring2, intensity2, diag_ramp2, colorize_lr2, "
            "digit/symbol topology, and edge-conditioned fills."
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


def templates_vh_bars(spec):
    if spec["pattern"] != "vh_bars2":
        return _ORIG(spec)
    images = torch.zeros(spec["modes"], 1, 8, 8)
    # mode0: two vertical bars; mode1: two horizontal bars (unpaired V↔H domains)
    images[0, 0, :, 1:3] = 0.9
    images[0, 0, :, 5:7] = 0.9
    images[1, 0, 1:3, :] = 0.9
    images[1, 0, 5:7, :] = 0.9
    return images


def main():
    task = make_task()
    summary = {
        "task": task["name"],
        "tip_sha": TIP,
        "principal": "vh_bars2 unpaired V vs H bars: published transpose12 vs residual16",
        "published_winner": "image suite baseline transpose12 (RpGAN+b_cap+particles)",
        "arms": {},
    }
    index_arms = {}
    with patch.object(host, "templates", templates_vh_bars):
        centers = templates_vh_bars(task)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            for ax, img, title in zip(axes, centers, ["vertical bars", "horizontal bars"]):
                ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle("vh_bars2 templates (unpaired V↔H)")
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
            print(json.dumps({"arm": name, **arm_summary}, default=str), flush=True)

    winner = summary["arms"]["baseline_transpose12"]["verdict"]
    control = summary["arms"]["residual16"]["verdict"]
    summary["hit"] = winner == "FAIL" and control == "PASS"
    summary["soft_miss"] = not summary["hit"]
    (EP / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    index = {
        "arms": index_arms,
        "device": "cpu",
        "hit": summary["hit"],
        "principal": summary["principal"],
        "tip_sha": TIP,
    }
    (EP / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    sys.exit(0 if summary["hit"] else 1)


if __name__ == "__main__":
    main()
