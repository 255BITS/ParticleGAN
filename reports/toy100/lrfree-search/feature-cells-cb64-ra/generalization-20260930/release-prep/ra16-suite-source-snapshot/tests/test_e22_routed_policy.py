"""Conformance for one fixed conditional paired-error training trajectory."""
import io
import math
from pathlib import Path
import runpy

import pytest
import torch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "e22_routed_paired.py"


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
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


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def api():
    return runpy.run_path(str(EXAMPLE))


def run_trajectory(api, *, device="cpu", steps=160):
    loop = api["make_loop"](device=device)
    frozen = loop.policy.G.host.weight.detach().clone()
    initial_noise = loop.policy.log_output_sigma.detach().clone()
    rows = []
    before_move = None
    with torch.autograd.set_multithreading_enabled(False):
        for _ in range(steps):
            previous = api["checkpoint"](loop)
            stats = api["update"](loop)
            rows.append(stats)
            if stats["move"] and stats["move"].get("moves", 0):
                before_move = previous
    assert before_move is not None, "the real conditional controller must accept a structural move"
    return {"loop": loop, "rows": rows, "before_move": cpu_roundtrip(before_move),
            "final": api["checkpoint"](loop), "metrics": api["evaluate"](loop),
            "frozen": frozen, "initial_noise": initial_noise}


@pytest.fixture(scope="module")
def trajectory(api):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield run_trajectory(api)
    finally:
        torch.set_num_threads(previous)


def test_paired_game_trains_all_dense_rows_and_preserves_frozen_bf16_host(trajectory):
    loop = trajectory["loop"]
    policy = loop.policy
    assert policy.recipe.row_policy == "routed_paired"
    assert policy.recipe.row_evidence_gate and policy.recipe.particle_birth_death
    assert policy.row_evidence is not None and policy.birth_death is not None
    assert all(row["dense_gradient_rows"] == len(policy.table) for row in trajectory["rows"])
    assert any(row["move"] and row["move"].get("moves", 0) for row in trajectory["rows"])
    for row in trajectory["rows"]:
        event = row["move"]
        if event and event.get("moves", 0):
            assert event["accepted"] and event["moves"] == 2
            assert event["guard_gain"] >= policy.routed_control.spec.improvement_margin
            assert event["average_guard_gain"] >= -1e-12
            assert max(event["max_context_harm"], event["average_max_context_harm"]) <= 1e-4 + 1e-12
    diagnostics = policy.birth_death.diagnostics()
    assert diagnostics["counters"]["moves"] > 0
    assert diagnostics["counters"]["probes"] > 0
    assert diagnostics["counters"]["guard_rejections"] > 0
    assert diagnostics["rows"]["law"] == "conditional_paired_diagnostic"
    assert diagnostics["rows"]["counters"]["updates"] == len(trajectory["rows"])
    assert diagnostics["rows"]["counters"]["flagged_rows_sum"] > 0
    assert diagnostics["rows"]["counters"].get("hold_steps", 0) > 0
    assert trajectory["metrics"]["heldout_rmse"] < .5 * trajectory["metrics"]["initial_rmse"]
    assert trajectory["metrics"]["output_dtype"] == "torch.float32"
    assert policy.G.host.weight.dtype == torch.bfloat16
    assert not policy.G.host.weight.requires_grad
    assert_tree_equal(policy.G.host.weight, trajectory["frozen"])
    assert_tree_equal(policy.ema_G.host.weight, trajectory["frozen"])
    for module in (policy.G, policy.encoder, policy.router, policy.D):
        assert all(p.dtype == torch.float32 for p in module.parameters() if p.requires_grad)
    assert policy.table.dtype == torch.float32
    assert not torch.equal(policy.table, trajectory["before_move"]["policy"]["table"])
    assert not torch.equal(policy.log_output_sigma.detach(), trajectory["initial_noise"])
    assert policy.opt_g.state[policy.log_output_sigma]["exp_avg_sq"].gt(0)


def test_fit_guard_and_final_context_grids_are_disjoint_and_final_is_never_observed(trajectory):
    loop = trajectory["loop"]
    fit = {tuple(row) for row in loop.fit_context.tolist()}
    guard = {tuple(row) for row in loop.guard_context.tolist()}
    final = {tuple(row) for row in loop.test_context.tolist()}
    assert fit.isdisjoint(guard) and fit.isdisjoint(final) and guard.isdisjoint(final)
    # Final validation is a clean, read-only served forward. Full-state equality
    # detects hidden reservoir writes or advances of any training/global RNG.
    before = loop.policy.state_dict()
    served = loop.policy.served_model()
    prediction = served.routed_forward(loop.test_context)
    again = served.routed_forward(loop.test_context)
    assert_tree_equal(prediction, again)
    assert_tree_equal(before, loop.policy.state_dict())


def test_cpu_deserialized_exact_resume_replays_accepted_move_and_served_validation(api, trajectory):
    saved = trajectory["before_move"]
    resumed = api["make_loop"]()
    api["restore"](resumed, saved)
    start = resumed.policy.completed_steps
    assert trajectory["rows"][start]["move"]["moves"] > 0
    with torch.autograd.set_multithreading_enabled(False):
        for expected in trajectory["rows"][start:]:
            assert_tree_equal(api["update"](resumed), expected)
    assert_tree_equal(api["checkpoint"](resumed), trajectory["final"])
    assert_tree_equal(api["evaluate"](resumed), trajectory["metrics"])
    assert_tree_equal(resumed.policy.served_snapshot(), trajectory["loop"].policy.served_snapshot())
    assert_tree_equal(resumed.policy.served_model().routed_forward(resumed.test_context),
                      trajectory["loop"].policy.served_model().routed_forward(resumed.test_context))


@pytest.mark.parametrize("decisive, source", [(1, "fast"), (-1, "averaged")])
def test_clean_serving_uses_one_consistent_frozen_routed_bank(api, trajectory, decisive, source):
    loop = api["make_loop"]()
    api["restore"](loop, cpu_roundtrip(trajectory["final"]))
    loop.policy._table_tester().last_decisive = decisive
    before = loop.policy.state_dict()
    served = loop.policy.served_model()
    assert served.source == source
    assert all(not p.requires_grad for model in served.models.values() for p in model.parameters())
    assert not served.table.requires_grad
    prediction = served.routed_forward(loop.test_context)
    # Inference must retain its E/router/table together after caller modules move.
    with torch.no_grad():
        loop.policy.G.adapter.bias.add_(3.)
        loop.policy.encoder.projection.weight.add_(2.)
        loop.policy.router.query.bias.sub_(1.)
        loop.policy.router.log_mass[0].add_(4.)
        loop.policy.table.add_(5.)
    assert_tree_equal(prediction, served.routed_forward(loop.test_context))
    # Reload the boundary and compare a separately reconstructed snapshot.
    loop.policy.load_state_dict(before)
    assert_tree_equal(prediction, loop.policy.served_model().routed_forward(loop.test_context))
    with pytest.raises(ValueError, match="independent|routed"):
        served.sample(2)


def test_tied_key_and_value_paths_both_receive_dense_gradients(api):
    loop = api["make_loop"]()
    policy = loop.policy
    models = policy._training_modules()
    candidate = policy.routed_control.spec.candidate_for(models, policy.table)
    weights = api["route"](models, loop.fit_context[:16], candidate)
    key_gradient = torch.autograd.grad(weights[:, 0].sum(), policy.table, retain_graph=True)[0]
    value_gradient = torch.autograd.grad((weights.detach() @ policy.table).sum(), policy.table)[0]
    assert key_gradient.norm(dim=-1).gt(0).all()
    assert value_gradient.norm(dim=-1).gt(0).all()


def test_decoder_uses_supplied_mixed_code_after_policy_perturbation(api):
    loop = api["make_loop"]()
    policy = loop.policy
    models, context = policy._training_modules(), loop.fit_context[:8]
    spec = policy.routed_control.spec
    candidate = spec.candidate_for(models, policy.table)
    weights = api["route"](models, context, candidate)
    codes = weights @ policy.table
    offset = codes.new_tensor([.3, -.1])
    expected = policy.G(context, policy.encoder(context), codes + offset)
    actual = spec.forward(models, context, candidate, perturb_fn=lambda code: code + offset)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, spec.forward(models, context, candidate))
    # Keys and routing weights stay at their actual table values.
    torch.testing.assert_close(weights, api["route"](models, context, candidate), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_host_routed_resume_replays_a_real_accepted_move_after_cpu_load(api):
    # Same device and execution order; this is a second device conformance run,
    # not a different seed or a hyperparameter search.
    receipt = run_trajectory(api, device="cuda", steps=24)
    resumed = api["make_loop"](device="cuda")
    api["restore"](resumed, receipt["before_move"])
    start = resumed.policy.completed_steps
    assert receipt["rows"][start]["move"]["moves"] > 0
    with torch.autograd.set_multithreading_enabled(False):
        for expected in receipt["rows"][start:]:
            assert_tree_equal(api["update"](resumed), expected)
    assert_tree_equal(api["checkpoint"](resumed), receipt["final"])
    assert_tree_equal(api["evaluate"](resumed), receipt["metrics"])
    assert_tree_equal(resumed.policy.served_model().routed_forward(resumed.test_context),
                      receipt["loop"].policy.served_model().routed_forward(resumed.test_context))
