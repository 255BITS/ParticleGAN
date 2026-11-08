"""Executable teaching examples exercise domain selection and continuation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from examples.bispf_categorical import TARGET, score


ROOT = Path(__file__).resolve().parents[1]


def run_example(name, directory, *arguments):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), *map(str, arguments)],
        cwd=directory, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_domain_walkthrough_runs_outside_checkout(tmp_path):
    output = run_example("bispf_input_domains.py", tmp_path)
    result = json.loads(output.splitlines()[-1])
    assert result["passed"] and len(result["checks"]) == 9
    cases = {(row["width"], row["input_domain"]): row for row in result["checks"]}
    # Width alone misclassifies wide vectors and misses small vocabularies.
    assert cases[1024, "auto"]["penalty_branch"] == "categorical"
    assert cases[1024, "continuous"]["mean_grad_norm"] == pytest.approx(0.2)
    assert cases[4, "auto"]["penalty_branch"] == "continuous"
    assert cases[4, "categorical"]["floor_pen"] == 0
    assert cases[4, "categorical"]["gauge_loss"] == 0


def test_categorical_scorer_oracle_and_destructive_controls():
    oracle = F.one_hot(torch.repeat_interleave(torch.arange(4), torch.tensor([10, 20, 30, 40])), 4).float()
    assert score(oracle)["quality_pass"]
    assert not score(F.one_hot(torch.zeros(100, dtype=torch.long), 4).float())["quality_pass"]
    # Correct mean category probabilities are insufficient: soft outputs must
    # also concentrate toward the one-hot support of the real observations.
    assert not score(TARGET.repeat(100, 1))["quality_pass"]
    assert not score(torch.full((100, 4), 0.25))["quality_pass"]
    with pytest.raises(ValueError, match="probability rows"):
        score(torch.full((5, 4), float("nan")))


def test_categorical_training_resume_and_actual_training_gif(tmp_path):
    full, split, resumed = [tmp_path / name for name in ("full", "split", "resumed")]
    run_example("bispf_categorical.py", tmp_path, "--steps", 4, "--output", full)
    run_example("bispf_categorical.py", tmp_path, "--steps", 4,
                "--stop-after", 2, "--output", split)
    run_example("bispf_categorical.py", tmp_path, "--steps", 4,
                "--resume", split / "checkpoint.pt", "--output", resumed)
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert_same(expected, actual)
    assert expected["trainer"]["penalty_options"] == {"input_domain": "categorical"}
    assert expected["history"][0]["step"] == 0
    assert expected["history"][-1]["step"] == 4
    report = json.loads((resumed / "metrics.json").read_text())
    assert not report["quality_pass"]  # Four updates never qualify a full-budget run.
    assert report["last_penalty_stats"]["floor_pen"] == 0
    assert report["last_penalty_stats"]["gauge_loss"] == 0
    from PIL import Image
    with Image.open(resumed / "training.gif") as animation:
        assert animation.n_frames >= 2
