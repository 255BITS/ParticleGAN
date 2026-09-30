"""The R1 configuration (configs/100gaussians/r1-rotation.json): E22 plus the optimizer-surprise re-open
(reopen_signal="optimizer") and the KA2 anchor release (reopen_anchor="release"). The re-open fires on a target
shift, the defaults keep E22 unchanged, and a checkpoint taken after a fire resumes bit-exactly."""
import json
import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

import particlegan.ka2
from particlegan.continuous import OptimizerSurprise
from particlegan import GANTrainer, get_recipe

CONFIGS = Path(__file__).parents[1] / "configs/100gaussians"
E22 = json.loads((CONFIGS / "e22-noout.json").read_text())
R1 = json.loads((CONFIGS / "r1-rotation.json").read_text())


def _recipe(options):
    return get_recipe(**dict(options, num_particles=512, z_dim=2, batch_size=64))


def _trainer(recipe, seed=0):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    return GANTrainer(recipe, G, D, seed=seed, optimizer_options={"foreach": False})


def _reals(n, shift_at, batch=64, seed=1):
    # A target that jumps once: the network-side gradients shock, which is what the re-open reads.
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, 2, generator=rng) * 2 + (8. if i >= shift_at else 0.) for i in range(n)]


def _run(trainer, reals):
    return [trainer.step(real, collect_stats=True) for real in reals]


def test_r1_config_is_e22_plus_two_fields():
    assert {k: v for k, v in R1.items() if R1[k] != E22.get(k)} == {"reopen_signal": "optimizer", "reopen_anchor": "release"}
    assert _recipe(E22).reopen_anchor == "hold"
    assert _trainer(_recipe(E22)).policy.surprise is None
    with pytest.raises(ValueError, match="requires reopen_signal optimizer"):
        _recipe(dict(E22, reopen_anchor="release"))


def test_r1_fires_on_shift_and_resumes_exactly_after_the_fire(monkeypatch):
    monkeypatch.setattr(particlegan.ka2, "WARMUP_CALLS", 20)  # the release acts on KA2's blended anchor
    # This small CPU model never settles far enough for the native-host threshold (a 2x jump); the shift here
    # lifts the ratio to ~1.8, so the threshold is lowered to exercise the fire, the latch and resume.
    monkeypatch.setattr(OptimizerSurprise, "RISE", 1.5)
    recipe = _recipe(R1)
    reals = _reals(140, shift_at=80)
    full = _trainer(recipe)
    full_out = _run(full, reals)
    surprise = full.policy.surprise
    assert surprise.fires >= 1 and surprise.anchor_events >= 1
    fire_step = surprise.log[0][0]
    assert fire_step > 80
    split = fire_step + 3
    first = _trainer(recipe)
    _run(first, reals[:split])
    resumed = _trainer(recipe, seed=5)
    resumed.load_state_dict(first.state_dict())
    for a, b in zip(_run(resumed, reals[split:]), full_out[split:]):
        for key in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(a[key], b[key]), key
    assert resumed.policy.surprise.state_dict() == surprise.state_dict()


ROUTED = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "e22_routed_moving.py"))


def test_r1_routed_reopens_on_a_turn_and_recovers_faster():
    # One 30-degree turn of the paired edit after 400 updates, once E22's learning rates have settled. E22-routed
    # never re-opens and 100 updates later is at ~.025 held-out RMSE; R1-routed fires and is back at ~.0015.
    def trace(r1):
        lines = []
        loop, periods = ROUTED["run"](turn_every=400, turns=1, degrees=30., r1=r1, log_every=50, emit=lines.append)
        rmse = {row["step"]: row["heldout_rmse"] for row in map(json.loads, lines) if "event" not in row}
        return loop, periods, rmse

    e22_loop, e22, e22_rmse = trace(False)
    r1_loop, r1, r1_rmse = trace(True)
    assert r1[0] == e22[0]  # identical until the first fire
    assert ROUTED["reopens"](e22_loop.policy) == 0 < ROUTED["reopens"](r1_loop.policy)
    assert r1_rmse[500] < e22_rmse[500] / 5
    assert r1[1] < e22[1]
