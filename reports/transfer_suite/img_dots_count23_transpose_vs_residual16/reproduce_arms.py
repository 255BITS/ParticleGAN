#!/usr/bin/env python3
"""Break: published transpose12 FAIL vs residual16 PASS on novel dots_count23."""
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
        name="img_dots_count23",
        pattern="dots_count23",
        modes=2,
        family="image_dots_count",
        importance_reason=(
            "Novel application: 2 corner dots vs 3 triangle dots on 8x8 "
            "grayscale — discrete cardinality / sparse count discrimination. "
            "Product-adjacent to counting under a shared GAN formulation. "
            "Distinct from sparse_obs2, sparse_pts_field2, blob_count2, "
            "corner_pair_diag2, L_chirality2, solid_hollow2."
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


def templates_dots(spec):
    if spec["pattern"] != "dots_count23":
        return _ORIG(spec)
    images = torch.zeros(spec["modes"], 1, 8, 8)
    bg, fg = 0.05, 0.9
    m0 = torch.full((8, 8), bg)
    m1 = torch.full((8, 8), bg)
    # 2 dots: NW and SE
    m0[1:3, 1:3] = fg
    m0[5:7, 5:7] = fg
    # 3 dots: NW, NE, S-center (triangle)
    m1[1:3, 1:3] = fg
    m1[1:3, 5:7] = fg
    m1[5:7, 3:5] = fg
    images[0, 0], images[1, 0] = m0, m1
    return images


def main():
    task = make_task()
    summary = {
        "task": task["name"],
        "tip_sha": TIP,
        "principal": "dots_count23 2 vs 3 dots: published transpose12 vs residual16",
        "published_winner": "image suite baseline transpose12 (RpGAN+b_cap+particles)",
        "arms": {},
    }
    index_arms = {}
    with patch.object(host, "templates", templates_dots):
        centers = templates_dots(task)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            for ax, img, title in zip(axes, centers, ["2 dots", "3 dots"]):
                ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle("dots_count23 templates (discrete cardinality)")
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
