"""An API-initialized bank-dependent task, with explicit benchmark overrides."""
import io
from pathlib import Path
import runpy

import pytest
import torch

from test_e22_routed_sites import assert_tree_equal


@pytest.fixture(scope="module")
def api():
    examples = Path(__file__).resolve().parents[1] / "examples"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(examples))
        yield runpy.run_path(str(examples / "e22_routed_support.py"))


@pytest.fixture(autouse=True)
def serial():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(before)


def test_neutral_branches_public_initialization_and_matched_controls(api, monkeypatch):
    original, calls = api["init"].deterministic_orthogonal_, []

    def record(module, **kwargs):
        calls.append((module, kwargs.get("seed", 0)))
        return original(module, **kwargs)

    monkeypatch.setattr(api["init"], "deterministic_orthogonal_", record)
    loops = [api["make_loop"](mode=mode, tokens=8) for mode in api["MODES"]]
    expected = api["initial_weights"](loops[0])
    assert len(calls) == 15
    for loop in loops:
        policy = loop.policy
        assert_tree_equal(api["initial_weights"](loop), expected)
        assert loop.config["initialization"] == "api"
        assert loop.config["penalty_units"] == "token"
        assert policy.routed_control.spec.max_context_harm == 0
        assert not policy.routed_control.spec.output_error_guard
        assert torch.count_nonzero(policy.G.first_output.weight) == 0
        assert torch.count_nonzero(policy.G.second_output.weight) == 0
        assert policy.G.first_input.weight.std() > 0
        assert policy.G.second_input.weight.std() > 0
        assert_tree_equal(policy.served_model().routed_forward(loop.test_context),
                          api["neutral"](loop.test_context))
        assert policy.G.first_host.weight.dtype == torch.bfloat16
        assert not policy.G.first_host.weight.requires_grad
    assert not loops[0].policy.table.requires_grad
    assert loops[1].policy.table.requires_grad and loops[2].policy.table.requires_grad
    traces = [[api["update"](loop) for _ in range(3)] for loop in loops]
    for trace in traces[1:]:
        for left, right in zip(trace, traces[0]):
            for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest"):
                assert_tree_equal(left[key], right[key])


def test_real_move_with_explicit_allowance_exact_resume_and_coherent_serving(api):
    # This is an integration-style benchmark allowance, not RoutedRows' zero
    # default. The output metric is only evaluated after the controller runs.
    options = dict(tokens=8, max_context_harm=1e-4)
    native = api["make_loop"](**options)
    previous = None
    for _ in range(240):
        previous = api["checkpoint"](native)
        row = api["update"](native)
        if row["move"] and row["move"].get("moves"):
            break
    else:
        pytest.fail("the public-initialized capacity task must exercise an accepted move")
    assert native.policy.routed_control.spec.max_context_harm == 1e-4
    assert native.policy.row_evidence.counters["updates"] > 0
    assert row["move"]["guard_gain"] > 0
    assert max(row["move"]["max_context_harm"], row["move"]["average_max_context_harm"]) <= 1e-4 + 1e-12
    stream = io.BytesIO()
    torch.save(previous, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    recovered = api["make_loop"](**options)
    api["restore"](recovered, saved)
    assert_tree_equal(api["update"](recovered), row)
    for _ in range(5):
        assert_tree_equal(api["update"](recovered), api["update"](native))
    assert_tree_equal(api["checkpoint"](recovered), api["checkpoint"](native))
    assert_tree_equal(native.policy.served_snapshot(), recovered.policy.served_snapshot())
    assert_tree_equal(native.policy.served_model().routed_forward(native.test_context),
                      recovered.policy.served_model().routed_forward(recovered.test_context))
    assert api["evaluate"](native)["heldout_rmse"] < native.initial_rmse


def test_fit_guard_test_are_disjoint_and_validation_is_read_only(api):
    loop = api["make_loop"](tokens=8)
    pools = [set(tuple(row.flatten().tolist()) for row in context)
             for context in (loop.fit_context, loop.guard_context, loop.test_context)]
    assert all(pools[a].isdisjoint(pools[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    before = api["checkpoint"](loop)
    api["evaluate"](loop)
    assert_tree_equal(api["checkpoint"](loop), before)


def test_noisy_game_observations_leave_accepted_moves_and_training_unchanged(api, capsys):
    options = dict(tokens=8, max_context_harm=1e-4)
    native, observed = api["make_loop"](**options), api["make_loop"](**options)
    ordinary = api["train"](native, 40, log_every=20)
    diagnostic = api["train"](observed, 40, log_every=20, game_diagnostics=True)
    capsys.readouterr()
    assert diagnostic["move_game_observations"], "observations must cover an actual accepted row move"
    assert_tree_equal(api["checkpoint"](native), api["checkpoint"](observed))
    assert ordinary["trace_sha256"] == diagnostic["trace_sha256"]
    assert_tree_equal(ordinary["row_diagnostics"], diagnostic["row_diagnostics"])
    assert ordinary["heldout_rmse"] == diagnostic["heldout_rmse"]
    for observation in diagnostic["move_game_observations"]:
        assert observation["clean_decision"]["accepted"]
        assert observation["game"]["probe"] == "fixed_critic_gradient_response"
        assert observation["game"]["bandwidth_mode"] == "current"
        assert observation["prospective"]["bandwidth_mode"] == "prospective"
        assert len(observation["game"]["draws"]) == 4
        assert "output_mse_log" in observation["game"]["mean_delta"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_default_128_by_4_spatial_move_recovers_exactly_with_frozen_bf16(api):
    # The benchmark's zero feature-harm default accepts a real move. No
    # initializer rewrite, tolerance override or synthetic controller priming.
    native = api["make_loop"](device="cuda:0")
    previous, row = None, None
    for _ in range(32):
        previous = api["checkpoint"](native)
        row = api["update"](native)
        if row["move"] and row["move"].get("moves"):
            break
    else:
        pytest.fail("the default 128x4 CUDA spatial task must accept a real move")
    assert native.config["max_context_harm"] == 0.
    assert not native.policy.routed_control.spec.output_error_guard
    assert row["dense_gradient_rows"] == 128
    assert max(row["move"]["max_context_harm"], row["move"]["average_max_context_harm"]) <= 1e-12
    stream = io.BytesIO()
    torch.save(previous, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    recovered = api["make_loop"](device="cuda:0")
    api["restore"](recovered, saved)
    assert_tree_equal(api["update"](recovered), row)
    for _ in range(5):
        assert_tree_equal(api["update"](recovered), api["update"](native))
    assert_tree_equal(api["checkpoint"](native), api["checkpoint"](recovered))
    assert_tree_equal(native.policy.served_model().routed_forward(native.test_context),
                      recovered.policy.served_model().routed_forward(recovered.test_context))
    for owner in (native.policy.G, recovered.policy.G):
        assert owner.first_host.weight.dtype == owner.second_host.weight.dtype == torch.bfloat16
        assert not owner.first_host.weight.requires_grad and not owner.second_host.weight.requires_grad
