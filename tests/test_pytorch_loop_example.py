"""examples/pytorch_loop.py: the documented caller-owned loop runs and scores itself."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/pytorch_loop.py"


def test_loop_sets_no_learning_rates():
    # The recipe-built optimizers own the schedule; the user loop never touches LR.
    source = EXAMPLE.read_text()
    assert not re.search(r"\[[\"']lr[\"']\]\s*=|learning_rate_scale|base_lr", source)
    assert "torch.optim.Adam(" not in source


def test_loop_smoke_reports_verdict():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-u", str(EXAMPLE), "--steps", "5", "--batch-size", "16"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=300).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    train = [row for row in rows if row["event"] == "train"]
    # The critic optimizer applied the recipe's decay inside step(): step 5 runs below the base rate.
    assert train[-1]["step"] == 5 and train[-1]["lr"] < train[0]["lr"]
    complete = rows[-1]
    assert complete["event"] == "complete" and complete["step"] == 5
    assert complete["verdict"] in {"PASS", "FAIL"} and 0 <= complete["modes"] <= 100 and 0 <= complete["hq"] <= 1
