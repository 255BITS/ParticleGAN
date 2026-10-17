"""Native generator-ladder regression distilled from Supra's causal LR release.

``--runxfail`` exposes the current stability failure. The strict xfail makes a
future fix visible as XPASS so the marker must then be removed. These tests
do not require monotonically improving held-out error or prohibit LR rises.
"""

import importlib
import io
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def toy():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
        yield importlib.import_module("e22_stiff_game_reopen")


@pytest.fixture(autouse=True)
def cpu_isolation():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.cuda.is_initialized() == cuda_initialized
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def test_contracted_reference_remains_in_the_stationary_game_neighborhood(toy):
    fixture, rows = toy.run(cancel_first_release=True)
    toy.assert_game_stable(rows)
    assert fixture.generator.bias[1] < toy.INITIAL_RESIDUAL[1]


def test_safe_geometry_allows_native_learning_rate_release(toy):
    fixture, rows = toy.run(fixture=toy.make_fixture(contracted_stiff_factor=.8))
    toy.assert_game_stable(rows)
    assert any(row["next_scale"] > row["previous_scale"] for row in rows)
    assert fixture.generator.bias[1] < toy.INITIAL_RESIDUAL[1]


@pytest.mark.parametrize("factor", [1.6, .8])
def test_prepared_moments_have_an_actual_native_adam_history(toy, factor):
    fixture = toy.make_fixture(contracted_stiff_factor=factor)
    expected = fixture.optimizer.state[fixture.generator.bias]
    beta2 = fixture.optimizer.param_groups[0]["betas"][1]
    gradient = (expected["max_exp_avg_sq"] /
                (1. - beta2 ** (toy.PAST_ADAM_STEPS - 1))).sqrt()
    # Reconstruct the moment history with a fresh native optimizer. Its
    # parameter values are immaterial to this gradient-program witness; the
    # specified generator snapshot does not claim this was its past dataset.
    witness = toy.make_fixture(contracted_stiff_factor=factor)
    witness.optimizer.state.clear()
    for _ in range(toy.PAST_ADAM_STEPS - 1):
        witness.generator.bias.grad = gradient.clone()
        witness.optimizer.step()
    witness.generator.bias.grad = torch.zeros_like(gradient)
    witness.optimizer.step()
    actual = witness.optimizer.state[witness.generator.bias]
    assert float(actual["step"]) == toy.PAST_ADAM_STEPS
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        torch.testing.assert_close(actual[name], expected[name], rtol=5e-14, atol=0)


@pytest.mark.xfail(strict=True, reason=(
    "native G displacement-coherence DRIFT releases a settled stiff/soft game "
    "outside its stable step interval; Supra's isolated G release also regresses"))
def test_native_ladder_keeps_this_stationary_paired_game_stable(toy):
    _, rows = toy.run()
    toy.assert_game_stable(rows)


@pytest.mark.parametrize("cancel,factor", [(False, 1.6), (True, 1.6), (False, .8)])
def test_native_snapshot_cpu_recovery_replays_the_release_boundary_exactly(toy, cancel, factor):
    full, expected = toy.run(cancel_first_release=cancel,
                             fixture=toy.make_fixture(contracted_stiff_factor=factor))
    prefix, prefix_rows = toy.run(cancel_first_release=cancel, stop=toy.CANCEL_STEP - 1,
                                 fixture=toy.make_fixture(contracted_stiff_factor=factor))
    equal(prefix_rows, expected[:toy.CANCEL_STEP - 1])
    snapshot = cpu_roundtrip(prefix.state_dict())
    resumed = toy.make_fixture(contracted_stiff_factor=factor)
    resumed.load_state_dict(snapshot)
    equal(resumed.state_dict(), snapshot)
    resumed, actual = toy.run(cancel_first_release=cancel, fixture=resumed,
                              start=toy.CANCEL_STEP, stop=toy.HORIZON)
    equal(actual, expected[toy.CANCEL_STEP - 1:])
    equal(resumed.state_dict(), full.state_dict())
