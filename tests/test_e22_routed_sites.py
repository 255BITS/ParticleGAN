"""End-to-end conformance for sequential token sites sharing one E22 bank."""
from dataclasses import replace
import io
import math
from pathlib import Path
import runpy

import pytest
import torch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "e22_routed_sites.py"


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


def run_trajectory(api, *, device="cpu", steps=60):
    loop = api["make_loop"](device=device)
    frozen = {key: parameter.detach().clone() for key, parameter in loop.policy.G.named_parameters()
              if not parameter.requires_grad}
    rows, before_move = [], None
    with torch.autograd.set_multithreading_enabled(False):
        for _ in range(steps):
            previous = api["checkpoint"](loop)
            row = api["update"](loop)
            rows.append(row)
            if row["move"] and row["move"].get("moves", 0):
                before_move = previous
    assert before_move is not None, "the actual two-site controller must accept a structural move"
    return dict(loop=loop, rows=rows, frozen=frozen, before_move=cpu_roundtrip(before_move),
                final=api["checkpoint"](loop), metrics=api["evaluate"](loop))


@pytest.fixture(scope="module")
def trajectory(api):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield run_trajectory(api)
    finally:
        torch.set_num_threads(previous)


def test_two_site_paired_training_has_dense_bank_gradients_real_evidence_and_guarded_moves(trajectory):
    loop, rows = trajectory["loop"], trajectory["rows"]
    p = loop.policy
    assert p.routed_control.spec.sites == ("first", "second")
    assert p.recipe.row_evidence_gate and p.recipe.particle_birth_death
    assert all(row["dense_gradient_rows"] == len(p.table) for row in rows)
    assert trajectory["metrics"]["heldout_rmse"] < .5 * trajectory["metrics"]["initial_rmse"]
    diagnostics = p.routed_control.diagnostics()
    assert diagnostics["counters"]["splits"] > 0
    assert diagnostics["counters"]["guard_rejections"] > 0
    assert diagnostics["rows"]["counters"]["updates"] == len(rows)
    assert diagnostics["rows"]["counters"]["flagged_rows_sum"] > 0
    assert diagnostics["rows"]["counters"].get("hold_steps", 0) > 0
    for row in rows:
        move = row["move"]
        if move and move.get("moves", 0):
            assert move["accepted"] and move["moves"] == 2
            assert move["guard_gain"] >= p.routed_control.spec.improvement_margin
            assert move["average_guard_gain"] >= -1e-12
            assert max(move["max_context_harm"], move["average_max_context_harm"]) <= 1e-4 + 1e-12
    for key, parameter in p.G.named_parameters():
        if key in trajectory["frozen"]:
            assert parameter.dtype == torch.bfloat16 and not parameter.requires_grad
            assert_tree_equal(parameter, trajectory["frozen"][key])
            assert_tree_equal(dict(p.ema_G.named_parameters())[key], trajectory["frozen"][key])
        else:
            assert parameter.dtype == torch.float32
    # Token and site multiplicity do not turn one conditioning sample into
    # multiple diagnostic observations. Counts describe current pool contexts.
    evidence = p.row_evidence
    assert (evidence.effective_contexts <= p.routed_control.fit_fill + 1e-9).all()
    assert (evidence.effect_contexts <= p.routed_control.fit_fill).all()


def test_counterfactual_recomputes_downstream_queries_and_measures_final_token_outputs(api):
    loop = api["make_loop"]()
    p, context = loop.policy, loop.fit_context[:8]
    models, spec = p._training_modules(), p.routed_control.spec
    candidate = spec.candidate_for(models, p.table, copy=True)
    second_inputs = []
    handle = p.router.second_query.register_forward_pre_hook(
        lambda module, inputs: second_inputs.append(inputs[0].detach().clone()))
    try:
        actual, usage = spec.forward_with_usage(models, context, candidate)
        mass = candidate.log_mass.clone()
        mass[0] = -torch.inf
        deleted = replace(candidate, log_mass=mass, row_state={**candidate.row_state, "log_mass": mass})
        changed, changed_usage = spec.forward_with_usage(models, context, deleted)
    finally:
        handle.remove()
    assert actual.shape == changed.shape == (len(context), loop.config["tokens"], 2)
    assert usage.shape == changed_usage.shape == (len(context), len(p.table))
    assert len(second_inputs) == 2
    assert not torch.equal(second_inputs[0], second_inputs[1])
    assert not torch.equal(actual, changed)
    assert changed_usage[:, 0].eq(0).all()
    torch.testing.assert_close(changed_usage.sum(dim=-1), torch.ones(len(context)))
    features = api["paired_features"](models, context, changed, loop.fit_targets[:8])
    assert features.shape == (len(context), loop.config["tokens"] * 8)


def test_exact_resume_replays_trained_two_site_move_and_clean_served_final_outputs(api, trajectory):
    resumed = api["make_loop"]()
    api["restore"](resumed, trajectory["before_move"])
    start = resumed.policy.completed_steps
    assert start > 0 and trajectory["rows"][start]["move"]["moves"] > 0
    with torch.autograd.set_multithreading_enabled(False):
        for expected in trajectory["rows"][start:]:
            assert_tree_equal(api["update"](resumed), expected)
    assert_tree_equal(api["checkpoint"](resumed), trajectory["final"])
    assert_tree_equal(api["evaluate"](resumed), trajectory["metrics"])
    assert_tree_equal(resumed.policy.served_model().routed_forward(resumed.test_context),
                      trajectory["loop"].policy.served_model().routed_forward(resumed.test_context))


@pytest.mark.parametrize("decisive, source", [(1, "fast"), (-1, "averaged")])
def test_serving_binds_both_sites_to_one_frozen_fast_or_average_bank(api, trajectory, decisive, source):
    loop = api["make_loop"]()
    api["restore"](loop, cpu_roundtrip(trajectory["final"]))
    loop.policy._table_tester().last_decisive = decisive
    before = loop.policy.state_dict()
    served = loop.policy.served_model()
    assert served.source == source
    assert not served.table.requires_grad
    assert all(not parameter.requires_grad for model in served.models.values() for parameter in model.parameters())
    prediction = served.routed_forward(loop.test_context)
    assert_tree_equal(before, loop.policy.state_dict())
    with torch.no_grad():
        loop.policy.G.first_adapter.bias.add_(3.)
        loop.policy.G.second_adapter.weight.sub_(2.)
        loop.policy.router.second_query.bias.add_(4.)
        loop.policy.table.add_(5.)
        loop.policy.router.log_mass[0].add_(1.)
    assert_tree_equal(prediction, served.routed_forward(loop.test_context))
    loop.policy.load_state_dict(before)
    assert_tree_equal(prediction, loop.policy.served_model().routed_forward(loop.test_context))


def test_comparison_modes_share_initial_parameters_batches_and_paired_base_noise(api):
    loops = [api["make_loop"](mode=mode) for mode in api["MODES"]]
    initial_table = loops[0].policy.table.detach().clone()
    for loop in loops[1:]:
        assert_tree_equal(loop.policy.table, initial_table)
        for name, module in loop.policy._training_modules().items():
            assert_tree_equal(module.state_dict(), loops[0].policy._training_modules()[name].state_dict())
    traces = []
    with torch.autograd.set_multithreading_enabled(False):
        for loop in loops:
            trace = [api["update"](loop) for _ in range(5)]
            traces.append([(row["batch_indices"], row["base_noise_sums"],
                            row["paired_rng_digest"], row["dv12_rng_digest"]) for row in trace])
    assert traces[0] == traces[1] == traces[2]
    for loop in loops[1:]:
        assert_tree_equal(loop.paired_noise_rng.get_state(), loops[0].paired_noise_rng.get_state())
        assert_tree_equal(loop.data_rng.get_state(), loops[0].data_rng.get_state())
    frozen, no_rows, full = [loop.policy for loop in loops]
    assert not frozen.table.requires_grad and frozen.table.grad is None
    assert_tree_equal(frozen.table, initial_table)
    assert not torch.equal(no_rows.table, initial_table)
    for policy in (frozen, no_rows):
        assert policy.row_evidence is None and policy.birth_death is None
        assert policy.routed_control.fit_context is None and policy.routed_control.guard_context is None
        assert policy.routed_control.evidence.counters["updates"] == 0
        assert policy.routed_control.counters["evals"] == policy.routed_control.counters["proposals"] == 0
    assert full.routed_control.counters["evals"] > 0


@pytest.mark.parametrize("mode", ["frozen", "no_rows", "full"])
def test_untimed_warmup_restores_entire_initial_checkpoint_modes_and_gradients(api, mode):
    loop = api["make_loop"](mode=mode)
    p = loop.policy
    p.D.eval()
    p.G.first_host.eval()
    p.G.first_adapter.weight.grad = torch.ones_like(p.G.first_adapter.weight)
    roots = [*p._training_modules().values(), *p._average_modules().values(), p.opt_d.ema_critic]
    modes = {module: module.training for root in roots for module in root.modules()}
    initial = api["checkpoint"](loop)
    initial_gradient = p.G.first_adapter.weight.grad.clone()
    api["warmup"](loop, 1)
    assert_tree_equal(api["checkpoint"](loop), initial)
    assert {module: module.training for root in roots for module in root.modules()} == modes
    assert_tree_equal(p.G.first_adapter.weight.grad, initial_gradient)
    assert p.table.grad is None and p.log_output_sigma.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_two_site_resume_crosses_accepted_move_after_cpu_checkpoint_load(api):
    receipt = run_trajectory(api, device="cuda", steps=16)
    resumed = api["make_loop"](device="cuda")
    api["restore"](resumed, receipt["before_move"])
    start = resumed.policy.completed_steps
    assert start > 0 and receipt["rows"][start]["move"]["moves"] > 0
    with torch.autograd.set_multithreading_enabled(False):
        for expected in receipt["rows"][start:]:
            assert_tree_equal(api["update"](resumed), expected)
    assert_tree_equal(api["checkpoint"](resumed), receipt["final"])
    assert_tree_equal(api["evaluate"](resumed), receipt["metrics"])
    assert_tree_equal(resumed.policy.served_model().routed_forward(resumed.test_context),
                      receipt["loop"].policy.served_model().routed_forward(receipt["loop"].test_context))
