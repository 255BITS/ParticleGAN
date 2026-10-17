"""R1, now E22's default (get_recipe("e22"), configs/100gaussians/e22-noout.json): the optimizer-surprise re-open
(reopen_signal="optimizer") and the KA2 anchor release (reopen_anchor="release"). The re-open fires on a target
shift, the defaults keep E22 unchanged, and a checkpoint taken after a fire resumes bit-exactly."""
import json
import math
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
R1 = E22
PRE_R1 = dict(E22, reopen_signal="none", reopen_anchor="hold")  # E22 before R1 became its default


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


def test_r1_is_the_e22_default_and_can_be_turned_off():
    assert get_recipe("e22").reopen_signal == "optimizer" and get_recipe("e22").reopen_anchor == "release"
    assert get_recipe("e22_routed").reopen_signal == "optimizer"
    assert _trainer(_recipe(R1)).policy.surprise is not None
    assert _trainer(_recipe(PRE_R1)).policy.surprise is None
    with pytest.raises(ValueError, match="requires reopen_signal optimizer"):
        _recipe(dict(PRE_R1, reopen_anchor="release"))


def _feed(surprise, qs):
    fired = []
    for step, q in enumerate(qs):
        surprise.pending = {"g": torch.tensor(q)}
        if surprise.decide(step):
            fired.append(step)
    return fired


def test_surprise_fires_on_a_jump_not_on_a_ramp_and_waits_after_a_fire():
    K = OptimizerSurprise.K
    settled = [1.] * 200
    # A step change: q jumps 5x and stays. Fires once, K updates after the fast average passes RISE.
    fired = _feed(OptimizerSurprise(), settled + [5.] * 100)
    assert len(fired) == 1 and 200 + K < fired[0] <= 200 + 2 * K
    # A slow ramp to the same level: the ratio creeps up for many windows before it crosses RISE.
    ramp = [math.exp(math.log(5.) * min(1., i / 400)) for i in range(600)]
    assert _feed(OptimizerSurprise(), settled + ramp) == []
    # Two jumps 60 updates apart: the second lands inside the refractory horizon (8K) and does not fire;
    # a jump after the horizon, from calm, does.
    two = settled + [5.] * 30 + [1.] * 30 + [25.] * 60
    assert len(_feed(OptimizerSurprise(), two)) == 1
    later = settled + [5.] * 30 + [1.] * 300 + [25.] * 60
    assert len(_feed(OptimizerSurprise(), later)) == 2


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


def test_r1_routed_resumes_exactly_after_a_fire(monkeypatch):
    # KA2's warm-up is shortened so the anchor release latch is exercised before the turn.
    monkeypatch.setattr(particlegan.ka2, "WARMUP_CALLS", 20)
    paired = ROUTED["paired"]

    def loop_until(steps, state=None):
        loop = paired["make_loop"](recipe_overrides=ROUTED["R1_OVERRIDES"])
        target = ROUTED["MovingTarget"](loop)
        if state is not None:
            paired["restore"](loop, state)
            if loop.policy.completed_steps > 400:
                target.turn_to(math.radians(30.))
        outputs = []
        while loop.policy.completed_steps < steps:
            if loop.policy.completed_steps == 400:
                target.turn_to(math.radians(30.))
            outputs.append(paired["update"](loop))
        return loop, outputs

    full, full_out = loop_until(480)
    surprise = full.policy.surprise
    assert surprise.fires == 1 and surprise.anchor_events == 1
    split = surprise.log[0][0] + 3
    assert 400 < split < 480
    first, _ = loop_until(split)
    state = paired["checkpoint"](first)
    resumed, resumed_out = loop_until(480, state)
    assert [(o["loss_d"], o["loss_g"]) for o in resumed_out] == [(o["loss_d"], o["loss_g"]) for o in full_out[split:]]
    assert resumed.policy.surprise.state_dict() == surprise.state_dict()
