"""Check migration/protocol integrity without running a qualification experiment."""
from copy import deepcopy
import json

import pytest
import torch

from particlegan import Recipe, get_recipe
from reports.toy100 import public_default_baseline as baseline


def test_prepare_records_defaults_without_constructing_a_learner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only must never construct or train a learner")
    monkeypatch.setattr(baseline, "make_trainer", forbidden)
    path = tmp_path / "declaration"
    baseline.run("shift", path, "cuda:0", prepare_only=True)
    saved = json.loads((path / "declaration.json").read_text())
    assert Recipe(**saved["recipe"]) == get_recipe(total_steps=3600)
    assert saved["status"] == "NOT_RUN"
    assert not saved["historical_scores_inherited"]
    assert "particlegan/k3p.py" in saved["source_sha256"]
    assert not (path / "metrics.jsonl").exists()
    with pytest.raises(FileExistsError):
        baseline.run("shift", path, "cpu", prepare_only=True)


def test_shift_requires_every_window_and_complete_negative_control():
    live = [{"step": t, "modes": 8, "hq": .99} for t in range(10, 3601, 10)]
    frozen = [{"step": t, "modes": 0, "hq": 0.} for t in range(2410, 3601, 10)]
    result = baseline.shift_summary(live, frozen)
    assert result["status"] == "PASS"
    assert result["stationary"]["checks"] == 5
    assert result["prehold"]["checks"] == 120
    assert result["deadline"]["checks"] == 81
    assert baseline.shift_summary(live, frozen[:-1])["status"] == "FAIL"
    assert baseline.shift_summary(live[:-1], frozen)["status"] == "FAIL"
    broken = deepcopy(live)
    next(p for p in broken if p["step"] == 1800)["hq"] = .89
    assert baseline.shift_summary(broken, frozen)["status"] == "FAIL"
    assert baseline.shift_summary(live, live)["status"] == "FAIL"


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_sampling_and_control_fork_preserve_complete_learner_state(monkeypatch):
    # A tiny integration check includes the active anchor and sparse-row history.
    monkeypatch.setattr(baseline.mode_hold, "HIDDEN", 8)
    monkeypatch.setattr(baseline.mode_hold, "EVAL_N", 16)
    torch.set_num_threads(1)
    recipe = get_recipe(batch_size=4, num_particles=32, total_steps=8,
                        network_lr_horizon_cap=4, d_guard_min_steps=0)
    trainer = baseline.make_trainer(recipe, "cpu")
    real = torch.randn(4, 2)
    for _ in range(6):
        trainer.step(real)
    state = trainer.state_dict()
    assert trainer.opt_d.record.anchor_started
    baseline.measure(trainer, baseline.mode_hold.ring_means())
    baseline.measure(trainer, baseline.mode_hold.ring_means(), ema=True)
    assert_equal(state, trainer.state_dict())
    control = baseline.make_trainer(recipe, "cpu")
    control.load_state_dict(state)
    assert_equal(state, control.state_dict())
    trainer.step(real)
    baseline.measure(control, baseline.mode_hold.ring_means())
    # Global RNG belongs to the process, not the frozen learner.
    frozen_state = control.state_dict()
    for key in ("models", "optimizers", "streams", "completed_steps"):
        assert_equal(state[key], frozen_state[key])
