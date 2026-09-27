"""examples/quickstart_gan.py: the documented stop/resume path matches an uninterrupted run."""
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "quickstart_gan.py"


def _run(*args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(SCRIPT), "--device", "cpu", *args], cwd=ROOT, env=env,
                   check=True, capture_output=True, text=True)


def test_quickstart_resume_matches_uninterrupted_run(tmp_path):
    full, part = tmp_path / "full.pt", tmp_path / "part.pt"
    _run("--steps", "4", "--output", str(full))
    _run("--steps", "4", "--stop-after", "2", "--output", str(part))
    _run("--steps", "4", "--resume", str(part), "--output", str(part))
    a = torch.load(full, map_location="cpu", weights_only=True)["trainer"]
    b = torch.load(part, map_location="cpu", weights_only=True)["trainer"]
    # The stopped run drew its samples once more, so only the eval stream may differ.
    a["streams"].pop("eval_generator"), b["streams"].pop("eval_generator")
    assert a["completed_steps"] == b["completed_steps"] == 4
    torch.testing.assert_close(a["models"], b["models"], rtol=0, atol=0)
    torch.testing.assert_close(a["streams"], b["streams"], rtol=0, atol=0)
    for opt_a, opt_b in zip(a["optimizers"], b["optimizers"]):
        torch.testing.assert_close(opt_a["state"], opt_b["state"], rtol=0, atol=0)
        assert opt_a["lr_schedule"] == opt_b["lr_schedule"]
        assert opt_a["lr_schedule"]["completed_steps"] == 4
        assert [g["lr"] for g in opt_a["param_groups"]] == [g["lr"] for g in opt_b["param_groups"]]
