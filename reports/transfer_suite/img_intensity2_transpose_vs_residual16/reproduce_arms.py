#!/usr/bin/env python
"""Break: published transpose12 FAIL vs residual16 PASS on img_intensity2."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.transfer_suite import image_solvability as solv
from benchmarks.transfer_suite import image_tasks as host

EP = Path(__file__).resolve().parent
OUT = EP / "runs"
OUT.mkdir(exist_ok=True)

TASK_NAME = "img_intensity2"
CARDS = json.loads((EP / "cards.json").read_text())
task = next(t for t in host.TASKS if t["name"] == TASK_NAME)
summary = {
    "task": TASK_NAME,
    "tip_sha": (EP / "tip_sha.txt").read_text().strip(),
    "arms": {},
}

for card in CARDS:
    name = card["name"]
    print(f"START {name} {TASK_NAME}", flush=True)
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
        "solv_quality": [bool(sustained_bool), modes, hq],
    }
    verdict = "PASS" if sustained_bool else "FAIL"
    slim = {k: v for k, v in result.items() if k not in ("actions", "protocol")}
    (OUT / f"{name}.json").write_text(json.dumps(slim, indent=2, sort_keys=True, default=str) + "\n")
    summary["arms"][name] = {"card": card, "verdict": verdict, "quality": quality, "seconds": secs}
    print(json.dumps({"arm": name, "verdict": verdict, "quality": quality}, default=str), flush=True)

winner = summary["arms"]["baseline_transpose12"]["verdict"]
control = summary["arms"]["residual16"]["verdict"]
summary["hit"] = winner == "FAIL" and control == "PASS"
summary["soft_miss"] = not summary["hit"]
(EP / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("SUMMARY", json.dumps(summary, indent=2), flush=True)
sys.exit(0 if summary["hit"] else 1)
