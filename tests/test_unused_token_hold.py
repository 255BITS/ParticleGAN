"""unused_token_hold is a problem-only toy on the shared runner."""
from pathlib import Path

import pytest
import torch

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.hosts import unused_token_hold as uth
from benchmarks.toy_runner import ToyRun, run

ROOT = Path(__file__).resolve().parents[1]


def test_unused_token_hold_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/hosts/unused_token_hold.py").read_text()
    for banned in ("torch.optim", "GANLoss", "GradientPenalty", "noise_policy", '["lr"]',
                   "learning_rate_scale", "schedule_optimizer", "manual_seed"):
        assert banned not in source, banned


def test_runs_on_recipe_deterministically_and_logs_named_losses(tmp_path):
    recipe = uth.UnusedTokenHold().recipe().replace(total_steps=12)
    log = tmp_path / "uth.log"
    first = run(uth.UnusedTokenHold(), recipe=recipe, observe_every=4, log_path=log)
    again = run(uth.UnusedTokenHold(), recipe=recipe, observe_every=4)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert first["verdict"] == uth.verdict(first["live"])
    assert '"unused_hold"' in log.read_text().splitlines()[0]
    assert first["live"]["scale0_err"] == 0.0


def test_hold_loss_pins_the_unused_slot_and_arms_differ():
    recipe = uth.UnusedTokenHold().recipe().replace(total_steps=30)
    held = run(uth.UnusedTokenHold(), recipe=recipe)["live"]
    free = run(uth.UnusedTokenHold(hold_weight=0.0), recipe=recipe)["live"]
    assert held["unused_dist"] < free["unused_dist"]
    with pytest.raises(ValueError):
        uth.UnusedTokenHold(pairing="nope")


def test_resume_is_exact():
    problem = uth.UnusedTokenHold(fm_weight=0.5)
    recipe = problem.recipe().replace(total_steps=10)
    full = ToyRun(problem, recipe=recipe)
    for _ in range(6):
        full.step()
    part = ToyRun(uth.UnusedTokenHold(fm_weight=0.5), recipe=recipe)
    for _ in range(3):
        part.step()
    resumed = ToyRun(uth.UnusedTokenHold(fm_weight=0.5), recipe=recipe)
    resumed.load_state_dict(part.state_dict())
    for _ in range(3):
        resumed.step()
    for a, b in zip(full.nets.generator.parameters(), resumed.nets.generator.parameters()):
        assert torch.equal(a, b)


def test_baseline_routes_to_the_runner_and_refuses_harness_noise():
    result = baseline.run_toy("unused_token_hold", baseline.Candidate("locked_shared"))
    assert set(result["live"]) == {"unused_hold", "concept_move"}
    assert len(result["observations"]) == 24
    with pytest.raises(ValueError, match="recipe"):
        baseline.run_toy("unused_token_hold", baseline.Candidate("locked_shared"), noise_policy=object())
