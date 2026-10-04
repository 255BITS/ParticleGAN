"""Model-free public diagnostic-health controls; no scientific grade or run."""
from copy import deepcopy
import math

import pytest
import torch

from experiments.forge.policy_adapters import finite_policy_state, typed_state_digest
from particlegan.birth_death import _levina_bickel


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def public_skip_state():
    # Actual public statistic on predetermined tied distances, without a model,
    # sampler, optimizer, birth/death owner, or maybe_apply callback invocation.
    undefined, valid_rows = _levina_bickel(torch.zeros(11, 5))
    assert math.isnan(undefined) and valid_rows == 0
    return {"completed_steps": 9,
            "models": {"generator": {"weight": torch.zeros(2, 2)}},
            "optimizers": [{"state": {0: {"step": torch.tensor(9.),
                                           "exp_avg": torch.zeros(2, 2)}}}],
            "birth_death": {"config": {"table_shape": (11, 2)},
                "counters": {"evals": 9, "dim_skips": 2, "moves": 4},
                "moved_rows": None,
                "last": {"step": 9, "k": 5, "d_R": undefined,
                         "d_F": .926260882020859, "skip": "dimension undefined"}}}


def envelope(state, root):
    return state if root is None else {root: state}


@pytest.mark.parametrize("root", [None, "trainer", "policy"])
def test_health_accepts_only_public_clocked_no_move_skip_and_preserves_bytes(root):
    state = envelope(public_skip_state(), root)
    before = typed_state_digest(state)
    assert finite_policy_state(state)
    assert typed_state_digest(state) == before
    policy = state if root is None else state[root]
    assert math.isnan(policy["birth_death"]["last"]["d_R"])
    # Prior moves remain history; a previous skip can persist until turnover.
    policy["completed_steps"] = 10
    assert finite_policy_state(state)


@pytest.mark.parametrize("change", [
    "no_marker", "wrong_marker", "extra_last_field", "no_completed_clock", "bool_completed_clock",
    "future_last_clock", "zero_last_clock", "float_last_clock", "bool_last_clock",
    "wrong_k", "float_k", "no_shape", "list_shape", "bad_shape", "huge_shape",
    "no_counters", "zero_skips", "skips_exceed_evals", "evals_exceed_completed", "float_evals",
    "missing_moved_rows", "active_move", "claimed_zero_moves", "nan_fake_dimension", "inf_fake_dimension",
])
def test_health_rejects_forged_or_incomplete_undefined_dimension_metadata(change):
    state = public_skip_state(); birth = state["birth_death"]; last = birth["last"]
    if change == "no_marker": last.pop("skip")
    elif change == "wrong_marker": last["skip"] = "another branch"
    elif change == "extra_last_field": last["unknown"] = 0.
    elif change == "no_completed_clock": state.pop("completed_steps")
    elif change == "bool_completed_clock": state["completed_steps"] = True
    elif change == "future_last_clock": last["step"] = 10
    elif change == "zero_last_clock": last["step"] = 0
    elif change == "float_last_clock": last["step"] = 9.
    elif change == "bool_last_clock": last["step"] = True
    elif change == "wrong_k": last["k"] = 4
    elif change == "float_k": last["k"] = 5.
    elif change == "no_shape": birth["config"].pop("table_shape")
    elif change == "list_shape": birth["config"]["table_shape"] = [11, 2]
    elif change == "bad_shape": birth["config"]["table_shape"] = (0, 2)
    elif change == "huge_shape": birth["config"]["table_shape"] = (10 ** 1000, 2)
    elif change == "no_counters": birth.pop("counters")
    elif change == "zero_skips": birth["counters"]["dim_skips"] = 0
    elif change == "skips_exceed_evals": birth["counters"]["dim_skips"] = 10
    elif change == "evals_exceed_completed": birth["counters"]["evals"] = 10
    elif change == "float_evals": birth["counters"]["evals"] = 9.
    elif change == "missing_moved_rows": birth.pop("moved_rows")
    elif change == "active_move": birth["moved_rows"] = torch.tensor([0], dtype=torch.long)
    elif change == "claimed_zero_moves": last["moves"] = 0
    elif change == "nan_fake_dimension": last["d_F"] = math.nan
    elif change == "inf_fake_dimension": last["d_F"] = math.inf
    assert not finite_policy_state({"policy": state})


@pytest.mark.parametrize("value", [math.inf, -math.inf, torch.tensor(math.nan)])
def test_health_rejects_nonpublic_reference_dimension_values(value):
    state = public_skip_state(); state["birth_death"]["last"]["d_R"] = value
    assert not finite_policy_state({"policy": state})


@pytest.mark.parametrize("root", ["unknown", "models", "optimizers", "controller"])
def test_health_rejects_same_sentinel_under_unknown_or_learned_paths(root):
    assert not finite_policy_state({root: public_skip_state()})
    assert not finite_policy_state({"policy": {root: public_skip_state()}})


@pytest.mark.parametrize("field", ["models", "optimizers", "averages", "table", "output_noise", "birth_tensor", "unknown"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_health_skip_never_exempts_nonfinite_learned_or_unknown_state(field, value):
    state = public_skip_state()
    if field == "birth_tensor": state["birth_death"]["S"] = torch.tensor([value])
    else: state[field] = {"bad": torch.tensor([value])}
    assert not finite_policy_state({"policy": state})


def test_health_finite_ordinary_birth_diagnostics_keep_existing_behavior():
    state = public_skip_state()
    state["birth_death"]["last"] = {"step": 9, "k": 5, "d_R": 1., "d_F": 1., "moves": 1}
    state["birth_death"]["moved_rows"] = torch.tensor([0], dtype=torch.long)
    assert finite_policy_state({"policy": state})


def test_health_existing_lr_exception_and_birth_skip_are_both_read_only():
    state = public_skip_state()
    state["lr_settle"] = [[{"r_b": [torch.tensor(math.nan)],
                            "last": {"log_bf_b": -math.inf}}]]
    before = typed_state_digest(state)
    assert finite_policy_state({"policy": state})
    assert typed_state_digest(state) == before
    bad = deepcopy(state); bad["lr_settle"][0][0]["r_b"][0] = torch.tensor(math.inf)
    assert not finite_policy_state({"policy": bad})
