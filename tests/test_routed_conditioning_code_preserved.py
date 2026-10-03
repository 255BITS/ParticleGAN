"""Time-only additive initialization retains the actual public conditioning path."""
from copy import deepcopy

import pytest
import torch

from benchmarks.routed_conditioning import film_damping as common
from benchmarks.routed_conditioning.spatial_damping import make_loop


@pytest.fixture(autouse=True)
def single_threaded():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("width", [4, 16])
def test_only_additive_time_columns_change_and_code_path_remains_live(width):
    original = make_loop(width=width)
    neutral = make_loop("shift_zero_native", width=width)
    candidate = make_loop("shift_time_zero_native", width=width)
    for role in ("critic", "encoder", "router", "table"):
        assert original.metadata["initial_hashes"][role] == candidate.metadata["initial_hashes"][role]
    assert original.metadata["data_hashes"] == candidate.metadata["data_hashes"]
    original_params = dict(original.policy.G.named_parameters())
    for name, value in candidate.policy.G.named_parameters():
        expected = original_params[name]
        if name == "condition.weight":
            assert torch.equal(value[:width], expected[:width])
            assert torch.equal(value[width:, :4], expected[width:, :4])
            assert value[width:, :4].count_nonzero() > 0
            assert value[width:, 4:].count_nonzero() == 0
            assert expected[width:, 4:].count_nonzero() > 0
        else:
            assert torch.equal(value, expected)
    assert candidate.metadata["initial_code_jacobian_frobenius"] > neutral.metadata["initial_code_jacobian_frobenius"]
    a, b = common.update(neutral), common.update(candidate)
    assert b["gradient_energy"]["encoder"] > a["gradient_energy"]["encoder"] > 0
    assert b["displacement_energy"]["encoder"] > 0
    assert b["dense_gradient_rows"] == 128
    for stream in ("batch_rng", "paired_noise_rng"):
        assert torch.equal(getattr(neutral, stream).get_state(), getattr(candidate, stream).get_state())
    p = candidate.policy
    assert p.reopen_guard is not None and p.row_evidence is not None
    assert p.recipe.total_steps is None and p.completed_steps == 1
    assert p.penalty.last_stats["applied"] and p.opt_d.record.observed_steps == 1


def test_code_preserved_checkpoint_replays_exactly_and_refuses_whole_zero_owner():
    loop = make_loop("shift_time_zero_native")
    for _ in range(3):
        common.update(loop)
    state = deepcopy(common.checkpoint(loop))
    expected = common.update(loop)
    restored = make_loop("shift_time_zero_native")
    common.restore(restored, state)
    assert common.json_safe(common.update(restored)) == common.json_safe(expected)
    torch.testing.assert_close(restored.policy.table, loop.policy.table, rtol=0, atol=0)
    other = make_loop("shift_zero_native")
    with pytest.raises(ValueError, match="profile/fixture"):
        common.restore(other, state)
