"""A distinct H/b initialization law; native particles and controls remain live."""
from copy import deepcopy
import io
import math

import pytest
import torch

from examples import e22_routed_convergence_neutral as neutral

base = neutral.baseline


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def single_threaded():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@pytest.fixture(scope="module")
def data():
    original = base.make_data()
    return original, neutral.make_neutral_data(original)


def test_only_initial_h_and_bias_change_before_fast_and_ema_construction(data):
    original, modified = data
    reference = base.make_loop(neutral.ARM, original)
    candidate = neutral.make_neutral_loop(modified)
    rng = torch.get_rng_state().clone()
    expected_native, actual_native = reference.policy.state_dict(), candidate.policy.state_dict()
    for family in ("models", "averages"):
        for name, value in actual_native[family]["generator"].items():
            expected = expected_native[family]["generator"][name].clone()
            if name.endswith("bridge.weight"):
                assert expected[:, :base.RANK].count_nonzero() > 0
                assert value[:, base.RANK:].count_nonzero() > 0
                expected[:, :base.RANK].zero_()
            elif name.endswith("bridge.bias"):
                assert expected.count_nonzero() > 0
                expected.zero_()
            assert_tree_equal(value, expected)
        actual_native[family].pop("generator")
        expected_native[family].pop("generator")
    assert_tree_equal(actual_native, expected_native)
    assert_tree_equal(candidate.policy.G.state_dict(), candidate.policy.ema_G.state_dict())
    for key in original:
        if key not in ("initial_particle", "digest"):
            assert_tree_equal(original[key], modified[key])
    assert_tree_equal(torch.get_rng_state(), rng)
    assert candidate.law["task"] == neutral.TASK and candidate.law["task"] != reference.law["task"]
    assert modified["digest"] != original["digest"]


def test_neutral_start_is_exact_common_base_and_retains_live_c_h_bias_and_bank(data):
    original, modified = data
    loop = neutral.make_neutral_loop(modified)
    ordinary = base.make_loop("ordinary_native_game", original)
    with torch.no_grad():
        for pool in base.SPLITS:
            context = modified[pool]["context"]
            for i in range(0, len(context), base.BATCH_SIZE):
                actual = base.forward(loop, context[i:i + base.BATCH_SIZE])
                assert_tree_equal(actual, base.forward(ordinary, context[i:i + base.BATCH_SIZE]))
                assert_tree_equal(actual, modified[pool]["base"][i:i + base.BATCH_SIZE])
    rows = [base.update(loop) for _ in range(4)]
    assert rows[0]["bank_gradient_norm"] == 0
    assert all(row["bank_gradient_rows"] == 128 and row["bank_gradient_norm"] > 0
               and row["query_gradient_norm"] > 0 for row in rows[1:])
    for site in ("first", "second"):
        bridge = getattr(loop.G, site).bridge
        assert bridge.weight.requires_grad and bridge.bias.requires_grad
        assert bridge.weight[:, base.RANK:].count_nonzero() > 0
        assert bridge.weight.grad[:, base.RANK:].norm() > 0
        assert bridge.weight.grad[:, :base.RANK].norm() > 0 and bridge.bias.grad.norm() > 0
    assert loop.policy.recipe.to_dict() == base.make_loop(neutral.ARM, original).policy.recipe.to_dict()
    assert loop.policy.routed_control.spec.max_context_harm == 0
    assert not loop.policy.routed_control.spec.output_error_guard


def test_neutral_checkpoint_restores_two_real_updates_and_rejects_other_law(data):
    original, modified = data
    loop = neutral.make_neutral_loop(modified, bindings={"source": "neutral-fixture"})
    for _ in range(3):
        base.update(loop)
    buffer = io.BytesIO()
    torch.save(base.checkpoint(loop), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=False)
    expected_rows = [base.update(loop) for _ in range(2)]
    expected = base.checkpoint(loop)
    restored = neutral.make_neutral_loop(modified, bindings={"source": "neutral-fixture"})
    base.restore(restored, saved)
    assert_tree_equal(base.checkpoint(restored), saved)
    assert_tree_equal([base.update(restored) for _ in range(2)], expected_rows)
    assert_tree_equal(base.checkpoint(restored), expected)
    ordinary_law = base.checkpoint(base.make_loop(neutral.ARM, original))
    before = base.checkpoint(restored)
    with pytest.raises(ValueError, match="match"):
        base.restore(restored, ordinary_law)
    assert_tree_equal(base.checkpoint(restored), before)
    bad = deepcopy(modified)
    bad["intervention"]["id"] = "unknown-initialization-law"
    with pytest.raises(ValueError, match="law"):
        neutral.make_neutral_loop(bad)


def test_unresolved_initial_checkpoint_uses_fresh_policy_before_resolved_checkpoint(data):
    _, modified = data
    loop = neutral.make_neutral_loop(modified, bindings={"source": "neutral-phase-fixture"})
    initial = base.checkpoint(loop)
    assert initial["training"]["backend_selection"]["output_shape"] is None
    for _ in range(4):
        base.update(loop)
    resolved = base.checkpoint(loop)
    assert resolved["training"]["backend_selection"]["output_shape"] == [base.TOKENS, base.WIDTH]
    with pytest.raises(ValueError, match="shape"):
        base.restore(loop, initial)
    assert_tree_equal(base.checkpoint(loop), resolved)
    fresh = neutral.make_neutral_loop(modified, bindings={"source": "neutral-phase-fixture"})
    base.restore(fresh, initial)
    assert_tree_equal(base.checkpoint(fresh), initial)
    base.restore(fresh, resolved)
    assert_tree_equal(base.checkpoint(fresh), resolved)
