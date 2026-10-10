"""Behavioral contracts of the GAN-only, conditional FiLM diagnostic."""
from copy import deepcopy
import io
import math

import pytest
import torch

from benchmarks.routed_conditioning.film_damping import (
    checkpoint, evaluate, main, make_loop, restore, update,
)
from particlegan import init


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def test_named_init_consumes_no_global_rng_and_only_shift_rows_differ():
    rng = torch.get_rng_state().clone()
    original = make_loop()
    neutral = make_loop("shift_zero_native")
    bypass = make_loop("shift_zero_g_bypass")
    assert torch.equal(torch.get_rng_state(), rng)
    exact(neutral.policy.state_dict(), bypass.policy.state_dict())
    for name, parameter in original.policy.G.named_parameters():
        other = dict(neutral.policy.G.named_parameters())[name]
        if name == "condition.weight":
            exact(parameter[:16], other[:16])
            assert other[16:].count_nonzero() == 0 and parameter[16:].count_nonzero() > 0
        else:
            exact(parameter, other)
    for role in ("encoder", "router", "critic"):
        exact(original.metadata["initial_hashes"][role], neutral.metadata["initial_hashes"][role])
    exact(original.policy.table, neutral.policy.table)
    assert original.metadata["data_hashes"] == neutral.metadata["data_hashes"]
    for model in original.policy._training_modules().values():
        assert all(spec is not None for spec in init.declarations(model).values())
    # The signed curvature head starts neutral, with no fixed negative prior.
    assert original.policy.D.quadratic.weight.count_nonzero() == 0


def test_full_native_roles_dense_table_and_frozen_host_survive_update():
    loop = make_loop("shift_zero_native")
    p = loop.policy
    frozen = deepcopy(p.G.host.state_dict())
    assert p.roles == [["generator", "encoder", "table", "noise"], ["critic"]]
    assert p.row_evidence is not None and p.birth_death is p.routed_control
    assert p.lr_settle.testers[0][0] is not None
    assert p.penalty.regularizer.record is p.opt_d.record
    assert p.opt_d.continuous_controller is p.controller
    owners = [parameter for optimizer in p.optimizers for group in optimizer.param_groups
              for parameter in group["params"]]
    assert len(owners) == len({id(parameter) for parameter in owners})
    row = update(loop)
    assert row["dense_gradient_rows"] == 128
    assert row["gradient_energy"]["encoder"] > 0 and row["gradient_energy"]["table"] > 0
    assert row["gradient_energy"]["critic"] > 0
    assert p.completed_steps == p.opt_d.record.observed_steps == 1
    assert p.penalty.last_stats["applied"]
    assert p.G.host.training is False
    exact(frozen, p.G.host.state_dict())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in p.G.host.parameters())
    assert p.D.quadratic.weight.count_nonzero() > 0


@pytest.mark.parametrize("profile", ["shift_zero_g_bypass", "shift_zero_antithetic_g_bypass"])
def test_g_bypass_keeps_native_observer_and_supplies_actual_applied_ratio(monkeypatch, profile):
    loop = make_loop(profile)
    p = loop.policy
    tester = p.lr_settle.testers[0][0]
    # Synthetic scale only exercises the software contract, never experiment evidence.
    tester.s = .25
    calls = []
    native_observe = tester.observe

    def observed(parameters, ratio, step=None):
        calls.append(ratio)
        return native_observe(parameters, ratio, step=step)

    monkeypatch.setattr(tester, "observe", observed)
    row = update(loop)
    assert p.lr_settle.testers[0][0] is tester and tester.s == .25
    assert row["proposed_rates"]["generator"]["applied_ratio"] == .25
    assert row["applied_rates"]["generator"]["applied_ratio"] == 1.
    assert calls == [1.]
    assert tester.blocks_in_window == 1 and tester.tau == 0
    for role in ("encoder", "table", "noise", "critic"):
        exact(row["proposed_rates"][role], row["applied_rates"][role])
    assert p.surprise.pending and p.routed_control.probe_clock["observed_updates"] == 1


def test_evaluation_changes_neither_policy_nor_named_rngs():
    loop = make_loop()
    update(loop)
    before = deepcopy(checkpoint(loop))
    global_rng = torch.get_rng_state().clone()
    result = evaluate(loop)
    assert result["live_mse"] >= 0 and result["served_mse"] >= 0
    exact(before, checkpoint(loop))
    exact(global_rng, torch.get_rng_state())


@pytest.mark.parametrize("profile", ["original_native", "shift_zero_native", "shift_zero_g_bypass",
                                     "shift_zero_antithetic", "shift_zero_antithetic_g_bypass"])
def test_named_stream_checkpoint_replays_exactly(profile):
    loop = make_loop(profile)
    for _ in range(3):
        update(loop)
    buffer = io.BytesIO()
    torch.save(checkpoint(loop), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    expected = [update(loop) for _ in range(2)]
    final = checkpoint(loop)
    restored = make_loop(profile)
    restore(restored, saved)
    for row in expected:
        exact(row, update(restored))
    exact(final, checkpoint(restored))
    exact(evaluate(loop), evaluate(restored))


def test_checkpoint_rejects_other_profile():
    original = make_loop()
    neutral = make_loop("shift_zero_native")
    with pytest.raises(ValueError, match="profile/fixture"):
        restore(neutral, checkpoint(original))


def test_antithetic_g_coupling_preserves_d_update_and_all_named_draws():
    ordinary = make_loop("shift_zero_native")
    antithetic = make_loop("shift_zero_antithetic")
    ordinary_row, antithetic_row = update(ordinary), update(antithetic)
    exact(ordinary.policy.D.state_dict(), antithetic.policy.D.state_dict())
    exact(ordinary.batch_rng.get_state(), antithetic.batch_rng.get_state())
    exact(ordinary.paired_noise_rng.get_state(), antithetic.paired_noise_rng.get_state())
    for name in ordinary.policy._STREAMS:
        exact(getattr(ordinary.policy, name).get_state(), getattr(antithetic.policy, name).get_state())
    exact(ordinary_row["loss_d"], antithetic_row["loss_d"])
    assert ordinary_row["loss_g"] != antithetic_row["loss_g"]
    assert antithetic.policy.completed_steps == 1


def test_antithetic_bypass_matches_native_before_a_rate_cut():
    native = make_loop("shift_zero_antithetic")
    bypass = make_loop("shift_zero_antithetic_g_bypass")
    exact(native.policy.state_dict(), bypass.policy.state_dict())
    for _ in range(3):
        exact(update(native), update(bypass))
    exact(native.policy.state_dict(), bypass.policy.state_dict())


def test_nonempty_artifact_directory_is_preserved(tmp_path, monkeypatch):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("prior evidence\n")
    monkeypatch.setattr("sys.argv", ["film_damping", "--output", str(tmp_path), "--steps", "1"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert sentinel.read_text() == "prior evidence\n"


def test_nonfinite_loss_stops_before_an_optimizer_update():
    loop = make_loop("shift_zero_native")
    before = deepcopy(loop.policy.G.state_dict())
    with torch.no_grad():
        loop.policy.D.score.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite critic loss"):
        update(loop)
    assert loop.policy.completed_steps == 0
    exact(before, loop.policy.G.state_dict())
