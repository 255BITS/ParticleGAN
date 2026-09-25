#!/usr/bin/env python3
"""Break: published transpose12 FAIL vs residual16 PASS on novel barcode_quiet_lr2."""
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
BAR_BITS = [1, 0, 1, 1, 0, 1]


def make_task():
    task = deepcopy(next(t for t in host.TASKS if t["name"] == "img_intensity2"))
    task.update(
        name="img_barcode_quiet_lr2",
        pattern="barcode_quiet_lr2",
        modes=2,
        family="image_barcode_quiet_zone",
        importance_reason=(
            "Novel application: retail barcode quiet-zone LEFT vs RIGHT of a shared "
            "UPC-style bar pattern on 8x8 grayscale. Product-adjacent to barcode "
            "scanners and margin/quiet-zone discrimination. Distinct from sonar TOF, "
            "chirp ridges, diffraction-order LR, moiré beats, and Manchester edges."
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


def templates_barcode(spec):
    if spec["pattern"] != "barcode_quiet_lr2":
        return _ORIG(spec)
    images = torch.zeros(spec["modes"], 1, 8, 8)
    images[:] = bg
    # Mode 0: quiet zone left (cols 0-1 blank), bars start at col 2
    col = 2
    for bit in BAR_BITS:
        if bit:
            images[0, 0, 2:6, col] = fg
        col += 1
    # Mode 1: quiet zone right (cols 6-7 blank), bars start at col 0
    col = 0
    for bit in BAR_BITS:
        if bit:
            images[1, 0, 2:6, col] = fg
        col += 1
    return images


def main():
    task = make_task()
    summary = {
        "task": task["name"],
        "tip_sha": TIP,
        "principal": "barcode_quiet_lr2 quiet-zone left vs right: published transpose12 vs residual16",
        "published_winner": "image suite baseline transpose12 (RpGAN+b_cap+particles)",
        "arms": {},
    }
    index_arms = {}
    with patch.object(host, "templates", templates_barcode):
        centers = templates_barcode(task)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            for ax, img, title in zip(axes, centers, ["quiet LEFT", "quiet RIGHT"]):
                ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle("barcode_quiet_lr2 templates")
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
