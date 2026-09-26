"""Diag traces must match the ring's mode rule and must not move weights."""
import torch

from benchmarks.locked_shared.mode_hold import (
    SIGMA, ModeHoldRecipe, diversity, ring_means, train_mode_hold,
)
from benchmarks.toy100 import diag_traj
from benchmarks.toy100.diag_traj import assign


def test_assign_matches_diversity_counts():
    means = ring_means()
    samples = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0], [2.9, 0.05]])
    detailed = diversity(samples, means, detailed=True)
    row = assign(samples, means, SIGMA)
    assert row["hq_counts"] == detailed["hq_counts"]
    assert row["nearest_counts"] == detailed["nearest_counts"]
    assert row["modes"] == detailed["modes"]


def test_diag_trace_does_not_change_a_short_run(tmp_path, monkeypatch):
    recipe = ModeHoldRecipe(steps=2)
    monkeypatch.delenv("K3P_DIAG_TRAJ", raising=False)
    torch.manual_seed(0)
    plain = train_mode_hold(recipe, diagnostics=True)
    diag_traj._FILE = None
    diag_traj._STASH.clear()
    path = tmp_path / "diag.jsonl"
    monkeypatch.setenv("K3P_DIAG_TRAJ", str(path))
    torch.manual_seed(0)
    traced = train_mode_hold(recipe, diagnostics=True)
    assert path.read_text().count("\n") == 3
    assert plain["support"]["points"] == traced["support"]["points"]
    assert plain["hq_counts"] == traced["hq_counts"]
