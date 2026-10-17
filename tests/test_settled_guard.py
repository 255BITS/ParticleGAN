"""Acquisition, loss-epoch, and checkpoint contracts for the opt-in R1 guard."""
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path

import pytest
import torch

from particlegan.continuous import OptimizerSurprise, SettledReopenGuard
from particlegan.particle_prior import ParticlePrior
from particlegan.recipes import Recipe
from particlegan.training import GANTrainer
import particlegan.ka2

CONFIG = Path(os.environ.get("PARTICLEGAN_RA13_CONTRACT_CONFIG",
    str(Path(__file__).resolve().parents[1] / "configs/100gaussians/ra13-settled.json")))
BASE = json.loads(CONFIG.read_text())
ROLES = [["generator", "table", "noise"], ["critic"]]
NETWORK = {"0.0": {"role": "generator", "scale": .5},
           "1.0": {"role": "critic", "scale": .25}}
KEYS = ("0.0", "0.1", "0.2", "1.0")


@pytest.fixture(autouse=True)
def isolated_cpu_contract():
    threads, device, dtype = torch.get_num_threads(), torch.get_default_device(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng, cuda_initialized = torch.get_rng_state().clone(), torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    try:
        yield
        assert torch.cuda.is_initialized() == cuda_initialized
    finally:
        torch.set_num_threads(threads)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)


def make(*, guard="settled", reg_every=1):
    torch.manual_seed(314159)
    G = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    D = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    prior = ParticlePrior(32, 2, generator=torch.Generator().manual_seed(314160))
    config = dict(BASE, num_particles=32, z_dim=2, batch_size=32,
                  birth_death_backend="knn", reopen_guard=guard, reg_every=reg_every)
    for key in ("birth_death_cells", "birth_death_metric_rank", "birth_death_chunk",
                "birth_death_parent_policy"):
        config.pop(key)
    return GANTrainer(Recipe(**config), G, D, prior=prior, seed=314159,
                      serial_backward=True, optimizer_options={"foreach": False})


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


def consume(detector, guard, step, q, network):
    detector.pending = {key: torch.tensor(value) for key, value in zip(KEYS, q)}
    return detector.decide(step, guard=guard, network=network)


def test_default_recipe_and_surprise_packet_are_legacy():
    trainer = make(guard=None)
    state = trainer.state_dict()
    assert "reopen_guard" not in state and "reopen_guard" not in state["recipe"]
    assert trainer.policy.reopen_guard is None
    assert set(OptimizerSurprise().state_dict()) == {
        "fast", "slow", "pending", "armed", "streak", "fires", "since_calm", "since_fire",
        "last_ratio", "last_ratios", "log", "anchor_event", "anchor_events"}
    restored = make(guard=None)
    restored.load_state_dict(state)
    equal(state, restored.state_dict())
    with pytest.raises(ValueError, match="reopen_guard"):
        trainer.recipe.replace(reopen_guard="settled", reopen_signal="none", reopen_anchor="hold")


def test_full_rate_startup_shock_tracks_without_reopen():
    legacy, detector, guard = OptimizerSurprise(), OptimizerSurprise(), SettledReopenGuard()
    for step in range(300):
        q = (1.,) * 4 if step < 200 else (5.,) * 4
        legacy.pending = {key: torch.tensor(value) for key, value in zip(KEYS, q)}
        legacy.decide(step)
        assert not consume(detector, guard, step, q, {})
    assert legacy.fires == 1
    assert detector.fires == 0 and detector.log == [] and detector.streak == 0
    assert detector.anchor_event is None and detector.anchor_events == 0
    assert detector.last_ratio < detector.RISE


def test_contracted_excursion_keeps_witness_after_ladder_release():
    legacy, detector, guard = OptimizerSurprise(), OptimizerSurprise(), SettledReopenGuard()
    fires = []
    for step in range(260):
        q = (1.,) * 4 if step < 200 else (5., 5., .5, 5.)
        network = NETWORK if step < 205 else {}
        legacy.pending = {key: torch.tensor(value) for key, value in zip(KEYS, q)}
        legacy.decide(step)
        if consume(detector, guard, step, q, network): fires.append(step)
        if 205 <= step < 215:
            assert guard.excursion["network"]["1.0"]["role"] == "critic"
    assert len(fires) == 1 and fires[0] > 205
    # All groups, including the table and scalar noise, retain the original math.
    equal(legacy.state_dict(), detector.state_dict())


def test_current_explicit_network_scales_own_eligibility():
    trainer = make()
    policy = trainer.policy
    for row, roles in zip(policy.lr_settle.testers, policy.roles):
        for tester, role in zip(row, roles):
            tester.s = .25 if role in ("table", "noise") else 1.
    for group in trainer.opt_d.param_groups: group["lr"] *= .001
    assert policy._contracted_network() == {}
    policy.lr_settle.testers[1][0].s = .25
    assert policy._contracted_network() == {"1.0": {"role": "critic", "scale": .25}}


def test_epoch_rebase_preserves_real_fire_cooldown_and_anchor_latch():
    detector, guard = OptimizerSurprise(), SettledReopenGuard()
    guard.observe_epoch(False, detector)
    detector.fast = {"0.0": 1.}; detector.slow = {"0.0": .1}
    detector.pending = {"0.0": torch.tensor(4.)}
    detector.streak, detector.since_calm = 7, 9
    detector.armed, detector.fires, detector.since_fire = False, 2, 3
    detector.log, detector.anchor_event, detector.anchor_events = [[90, 2.5]], [True], 1
    assert guard.observe_epoch(True, detector)
    assert guard.epoch_rebases == 1 and detector.fast == detector.slow == detector.pending == {}
    assert detector.streak == detector.since_calm == 0
    assert not detector.armed and detector.fires == 2 and detector.since_fire == 3
    assert detector.log == [[90, 2.5]] and detector.anchor_event == [True] and detector.anchor_events == 1
    assert not guard.observe_epoch(True, detector)


def test_ka2_epoch_uses_actual_applied_lazy_calls(monkeypatch):
    monkeypatch.setattr(particlegan.ka2, "WARMUP_CALLS", 3)
    trainer = make(reg_every=2)
    real = torch.arange(64, dtype=torch.float32).reshape(32, 2).sin()
    for _ in range(5):
        trainer.step(real)
        assert not trainer.opt_d.record.anchor_started
        assert trainer.policy.reopen_guard.epoch_rebases == 0
    trainer.step(real)
    assert trainer.completed_steps == 6 and trainer.opt_d.record.calls == 3
    assert trainer.opt_d.record.anchor_started and trainer.policy.reopen_guard.epoch_rebases == 1
    assert trainer.policy.surprise.fast == trainer.policy.surprise.slow == {}
    assert trainer.policy.surprise.pending and trainer.policy.surprise.fires == 0
    saved = trainer.state_dict()
    restored = make(reg_every=2); restored.load_state_dict(saved)
    equal(saved, restored.state_dict())
    trainer.step(real); restored.step(real)
    equal(trainer.state_dict(), restored.state_dict())


def test_pending_and_cooldown_restore_from_native_and_cpu_map():
    detector, guard = OptimizerSurprise(), SettledReopenGuard()
    guard.observe_epoch(False, detector)
    step = 0
    while detector.fires == 0:
        q = (1.,) * 4 if step < 200 else (5.,) * 4
        consume(detector, guard, step, q, NETWORK)
        step += 1
    for _ in range(3):
        consume(detector, guard, step, (5.,) * 4, {})
        step += 1
    detector.pending = {key: torch.tensor(5.) for key in KEYS}
    assert not detector.armed and 0 < detector.since_fire < 8 * detector.K
    payload = {"surprise": detector.state_dict(), "guard": guard.state_dict()}
    buffer = io.BytesIO(); torch.save(payload, buffer)
    branches = []
    for location in (None, "cpu"):
        buffer.seek(0)
        saved = torch.load(buffer, map_location=location, weights_only=True)
        d, g = OptimizerSurprise(), SettledReopenGuard()
        d.load_state_dict(saved["surprise"])
        g.load_state_dict(saved["guard"], roles=ROLES, completed_steps=step, anchor_started=False)
        branches.append((d, g))
    for update in range(step, step + 400):
        for d, g in ((detector, guard), *branches):
            # First update consumes the checkpoint's owned pending observation.
            if update != step:
                q = 1. if update < step + 300 else 25.
                d.pending = {key: torch.tensor(q) for key in KEYS}
            d.decide(update, guard=g, network=NETWORK if update >= step + 160 else {})
        for d, g in branches:
            equal(detector.state_dict(), d.state_dict()); equal(guard.state_dict(), g.state_dict())
    assert detector.fires == 2


@pytest.mark.parametrize("mode", ["trainer", "policy"])
@pytest.mark.parametrize("mutation", ["schema", "epoch", "counter", "role", "scale", "future"])
def test_guard_rejects_malformed_state_before_leased_model_mutation(mode, mutation):
    trainer = make()
    trainer.policy._table_tester().last_decisive = -1
    with torch.no_grad():
        for parameter in trainer.ema_G.parameters(): parameter.add_(.125)
    owner = trainer if mode == "trainer" else trainer.policy
    valid = owner.state_dict()
    invalid = deepcopy(valid)
    guard = invalid["reopen_guard"]
    if mutation == "schema": guard["schema"] = True
    elif mutation == "epoch": guard["ka2_anchor_started"] = True
    elif mutation == "counter": guard["epoch_rebases"] = True
    elif mutation == "role": guard["calm_network"] = {"0.1": {"role": "table", "scale": .5}}
    elif mutation == "scale": guard["calm_network"] = {"1.0": {"role": "critic", "scale": True}}
    elif mutation == "future": guard["excursion"] = {"step": 1, "network": NETWORK}
    trainer._serve_apply()
    assert trainer._fast is not None
    parameters = [p for module in trainer.policy._training_modules().values() for p in module.parameters()]
    values, versions = [p.detach().clone() for p in parameters], [p._version for p in parameters]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="guard|witness|excursion"):
        owner.load_state_dict(invalid)
    assert trainer._fast is not None
    assert [p._version for p in parameters] == versions
    assert torch.equal(torch.get_rng_state(), rng)
    for actual, expected in zip(parameters, values):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
