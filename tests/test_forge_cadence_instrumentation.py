"""Software-only public API cadence, passive telemetry and exact continuation.

These bounded numerical checks establish implementation behavior, not a trained
quality gate or scientific seed comparison.
"""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.bispf import _LEGACY_THERMAL_CHANNEL
from particlegan.init import deterministic_orthogonal_
from experiments.forge.mechanisms import MechanismAudit
from experiments.forge.state import state_digest


@pytest.fixture(autouse=True)
def isolated_runtime():
    threads, thermal = torch.get_num_threads(), dict(_LEGACY_THERMAL_CHANNEL)
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)
    _LEGACY_THERMAL_CHANNEL.clear()
    _LEGACY_THERMAL_CHANNEL.update(thermal)


def batch(step):
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    return torch.sin(values * .31 + step * .17)


def trainer(k, updates):
    # Constructor draws remain isolated; all learned parameters use the public
    # initializer. This fixture intentionally runs on CPU with the preset's
    # documented FP32 fallback, without changing its recipe execution policy.
    with torch.random.fork_rng(devices=[]):
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        recipe = get_recipe("bispf_optimized", z_dim=2, num_particles=16,
                            batch_size=8, total_steps=updates, reg_every=k)
        prior = recipe.make_prior()
    for component in (generator, critic, prior):
        deterministic_orthogonal_(component, seed=0)
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_public_bispf_lazy_schedule_scales_penalty_and_critic_gradients(k):
    eager, lazy = trainer(1, 32), trainer(k, 32)
    # Match the same pre-update state, including its applied-call clock. This
    # isolates the coefficient law from the real trajectory differences.
    eager.opt_d.record.observed_steps = lazy.opt_d.record.observed_steps = k - 1
    real, fake = batch(2), batch(3)
    eager.penalty.collect_stats = lazy.penalty.collect_stats = True
    before = state_digest(lazy.opt_d.record.state_dict())
    if k > 1:
        lazy.opt_d.record.observed_steps = k - 2
        skipped = lazy.penalty(lazy.D, real, fake)
        assert float(skipped) == 0. and not skipped.requires_grad
        assert lazy.penalty.last_stats["applied"] is False
        lazy.opt_d.record.observed_steps = k - 1
        assert state_digest(lazy.opt_d.record.state_dict()) == before
    one = eager.penalty(eager.D, real, fake)
    many = lazy.penalty(lazy.D, real, fake)
    torch.testing.assert_close(many, k * one, rtol=0, atol=0)
    assert lazy.penalty.last_stats["applied"] is True
    assert eager.opt_d.record.calls == lazy.opt_d.record.calls == 1
    one.backward()
    many.backward()
    compared = 0
    for left, right in zip(eager.D.parameters(), lazy.D.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(right.grad, k * left.grad, rtol=0, atol=0)
            compared += 1
    assert compared > 0
    assert [{key: value for key, value in group.items() if key != "params"}
            for group in eager.opt_d.param_groups] == [
        {key: value for key, value in group.items() if key != "params"}
        for group in lazy.opt_d.param_groups]
    assert eager.recipe.prior_reg == lazy.recipe.prior_reg
    assert eager.recipe.prior_update == lazy.recipe.prior_update


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_audit_is_passive_and_checkpoint_resets_only_its_local_span(k):
    updates, split = 2 * k + 1, max(1, k - 1)
    baseline = trainer(k, updates)
    for step in range(1, updates + 1):
        baseline.step(batch(step), collect_stats=True)
    expected = state_digest(baseline.state_dict())

    live = trainer(k, updates)
    audit = MechanismAudit(live.recipe, live.opt_d, [live.opt_g])
    for step in range(1, updates + 1):
        result = live.step(batch(step), collect_stats=True)
        audit.observe_penalty(result["penalty_stats"])
        if step == split:
            checkpoint = live.state_dict()
    assert state_digest(live.state_dict()) == expected
    rng = torch.get_rng_state().clone()
    state = state_digest(live.state_dict())
    receipt = audit.receipt()["penalty_cadence"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert state_digest(live.state_dict()) == state
    assert receipt == audit.receipt()["penalty_cadence"]
    assert receipt["observations"] == dict(calls=updates, applied=updates // k,
                                           skipped=updates - updates // k, unavailable=0)
    assert receipt["configured"]["effective_coefficient"] == k * live.recipe.reg_coeff
    assert receipt["optimizer"] == dict(observed_steps_start=0, observed_steps_end=updates,
        observed_steps_in_window=updates, applied_record_calls_start=0,
        applied_record_calls_end=updates // k, applied_record_calls_in_window=updates // k)
    assert receipt["first_application_update"] == k
    assert receipt["last_application_update"] == (updates // k) * k
    assert receipt["phase_counts"] == {"bispf": updates // k}
    assert receipt["ranges"]["s_thermal"]["count"] == updates
    json.dumps(receipt, allow_nan=False)

    restored = trainer(k, updates)
    resumed_audit = MechanismAudit(restored.recipe, restored.opt_d, [restored.opt_g])
    restored.load_state_dict(checkpoint)
    for step in range(split + 1, updates + 1):
        result = restored.step(batch(step), collect_stats=True)
        resumed_audit.observe_penalty(result["penalty_stats"])
    assert state_digest(restored.state_dict()) == expected
    resumed = resumed_audit.receipt()["penalty_cadence"]
    assert resumed["scope"] == "audit_window"
    assert resumed["start_reason"] == "controller_state_restored"
    assert resumed["resets"] == 1
    assert resumed["observations"]["calls"] == updates - split
    assert resumed["optimizer"]["observed_steps_start"] == split
    assert resumed["optimizer"]["applied_record_calls_start"] == split // k
    assert resumed["optimizer"]["applied_record_calls_in_window"] == updates // k - split // k
    assert resumed["first_application_update"] == ((split // k) + 1) * k


@pytest.mark.parametrize("observe_before_step", [True, False])
def test_application_updates_are_correct_for_both_host_observation_orders(observe_before_step):
    host = trainer(4, 8)
    audit = MechanismAudit(host.recipe, host.opt_d, [host.opt_g])
    host.penalty.collect_stats = True
    for step in range(1, 9):
        real, fake = batch(step), batch(step + 10)
        value = host.penalty(host.D, real, fake)
        if observe_before_step:
            audit.observe_penalty(host.penalty.last_stats)
        host.opt_d.zero_grad()
        (host.D(real).mean() + value).backward()
        host.opt_d.step()
        if not observe_before_step:
            audit.observe_penalty(host.penalty.last_stats)
    receipt = audit.receipt()["penalty_cadence"]
    assert receipt["first_application_update"] == 4
    assert receipt["last_application_update"] == 8
    assert receipt["application_updates_unavailable"] == 0


def test_missing_stats_are_unavailable_and_ranges_never_sync_tensors():
    host = trainer(4, 8)
    audit = MechanismAudit(host.recipe, host.opt_d, [host.opt_g])
    for stats in (None, {}, {"applied": False, "s_thermal": float("nan")},
                  {"ema_residual": torch.tensor(.2)}, {"s_thermal": float("inf")}):
        audit.observe_penalty(stats)
    receipt = audit.receipt()["penalty_cadence"]
    assert receipt["observations"] == dict(calls=5, applied=0, skipped=1, unavailable=4)
    assert receipt["optimizer"]["applied_record_calls_in_window"] == 0
    assert receipt["first_application_update"] is None
    assert receipt["ranges"]["s_thermal"]["nonfinite"] == 2
    assert receipt["ranges"]["ema_residual"]["unavailable"] == 1
    assert receipt["ranges"]["s_thermal"]["last"] == 1.
    json.dumps(receipt, allow_nan=False)


def test_copied_optimizer_validation_does_not_reset_live_audit():
    host = trainer(1, 4)
    audit = MechanismAudit(host.recipe, host.opt_d, [host.opt_g])
    result = host.step(batch(1), collect_stats=True)
    audit.observe_penalty(result["penalty_stats"])
    before = audit.receipt()["penalty_cadence"]
    copied = deepcopy(host.opt_d)
    copied.load_state_dict(deepcopy(host.opt_d.state_dict()))
    assert audit.receipt()["penalty_cadence"] == before
