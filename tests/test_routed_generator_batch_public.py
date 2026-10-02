"""Plumbing contracts only; the fixed500-update scientific gate is separate."""
import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from particlegan import init

DRIVER = Path(__file__).resolve().parents[1] / "examples" / "routed_generator_batch.py"


@pytest.fixture(scope="module")
def api():
    return runpy.run_path(str(DRIVER))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            tree_equal(a, b)
    else:
        assert left == right


def test_capacity_target_and_symmetric_hidden_nuisance(api):
    rng = torch.get_rng_state().clone()
    data = api["fixture"]()
    assert torch.equal(rng, torch.get_rng_state())
    assert data["fit_rows"].shape == (256, 3, 8, 8)
    assert data["guard_rows"].shape == (64, 3, 8, 8)
    assert data["population_context"].shape == (64, 3, 8, 8)
    torch.testing.assert_close(data["nuisance"][:8], -data["nuisance"][8:], rtol=0, atol=0)
    torch.testing.assert_close(data["nuisance"].square().mean((1, 2, 3)),
                               torch.full((16,), .15 ** 2), rtol=1e-6, atol=1e-8)
    for name in ("fit", "guard"):
        target = data[name + "_target"].reshape(-1, 16, 2, 8, 8)
        torch.testing.assert_close(target.mean(1), data[name + "_mean"], rtol=0, atol=3e-8)
        # No hidden nuisance identity reaches the source/time input.
        for mode in range(16):
            torch.testing.assert_close(data[name + "_rows"][mode::16], data[name + "_context"], rtol=0, atol=0)


def test_matched_public_init_live_ownership_and_independent_ema(api):
    rng = torch.get_rng_state().clone()
    a, b = api["make_loop"](16), api["make_loop"](64)
    assert torch.equal(rng, torch.get_rng_state())
    assert a.initial_hashes == b.initial_hashes
    assert api["caller_stream_hashes"](a) == api["caller_stream_hashes"](b)
    for loop in (a, b):
        p = loop.policy
        api["assert_ownership"](p)
        assert p.recipe.row_policy == "routed_paired"
        assert p.controller is p.opt_d.continuous_controller
        assert p.birth_death is p.routed_control and p.row_evidence is not None
        assert p.reopen_guard is not None and p.penalty is not None
        assert p.D.quadratic.weight.count_nonzero() == 0
        assert init.declarations(p.D)["quadratic.weight"] is init.KEEP
        assert float(p.D.global_gain) == 0 and float(p.D.local_gain) == .0625
        assert p.G.host.prefix.weight.dtype == p.G.host.head.weight.dtype == torch.bfloat16
        assert all(not parameter.requires_grad for parameter in p.G.host.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in p.G.parameters() if parameter.requires_grad)
        assert p.table.dtype == torch.float32
        for module in (p.G, p.encoder, p.router, p.D):
            assert all(spec is not None for spec in init.declarations(module).values())
        assert all(group["betas"] == (0., .999) for optimizer in p.optimizers for group in optimizer.param_groups)
        assert not ({id(parameter) for parameter in p.D.parameters()}
                    & {id(parameter) for parameter in p.opt_d.ema_critic.parameters()})


def test_one_native_step_keeps_d16_and_matched_common_caller_panel(api):
    a, b = api["make_loop"](16), api["make_loop"](64)
    for loop in (a, b):
        row = api["update"](loop)
        assert row["step"] == 1 and row["d_batch"] == 16 and row["g_batch"] == loop.g_batch
        assert row["dense_rows"] == 128
        assert loop.policy.opt_d.record.observed_steps == 1
        assert loop.policy.penalty.last_stats["applied"]
        assert loop.policy.routed_control.diagnostics()["rows"]["counters"]["updates"] == 1
        assert all(parameter.grad is None for parameter in loop.policy.G.host.parameters())
        assert loop.policy.D.quadratic.weight.count_nonzero() > 0
    assert a.caller_history == b.caller_history
    assert api["caller_stream_hashes"](a) == api["caller_stream_hashes"](b)
    # Native DV12 private consumption can differ with G batch; no assertion.


def test_g64_extends_exact_d16_examples_by48_caller_draws(api):
    a, b = api["make_loop"](16), api["make_loop"](64)
    for _ in range(3):
        first, second = api["draw_panel"](a), api["draw_panel"](b)
        tree_equal(first, second)
        torch.testing.assert_close(first["g_indices"][:16], first["d_indices"], rtol=0, atol=0)
        assert first["g_indices"].shape == (64,) and first["g_indices"][16:].shape == (48,)
    expected = torch.Generator().manual_seed(api["SEEDS"]["g_data"])
    torch.randint(256, (48,), generator=expected)
    torch.randint(256, (48,), generator=expected)
    torch.randint(256, (48,), generator=expected)
    torch.testing.assert_close(a.streams["g_data"].get_state(), expected.get_state(), rtol=0, atol=0)


def test_public_resume_and_evaluation_preserve_callers_and_full_native_state(api):
    loop = api["make_loop"](16)
    api["update"](loop)
    boundary = deepcopy(api["checkpoint"](loop))
    before = deepcopy(loop.policy.state_dict())
    callers = api["caller_stream_hashes"](loop)
    first, second = api["evaluate"](loop), api["evaluate"](loop)
    tree_equal(first, second)
    tree_equal(before, loop.policy.state_dict())
    assert callers == api["caller_stream_hashes"](loop)
    expected = api["update"](loop)
    resumed = api["make_loop"](16)
    api["restore"](resumed, boundary)
    tree_equal(expected, api["update"](resumed))
    tree_equal(api["checkpoint"](loop), api["checkpoint"](resumed))
    tree_equal(api["evaluate"](loop), api["evaluate"](resumed))
    with pytest.raises(ValueError, match="cohort"):
        api["restore"](api["make_loop"](64), boundary)


def test_fixed_gate_rejects_no_gain_even_when_both_converge(api):
    arm = {"steps": 500, "health": {"finite_steps": 500, "min_dense_rows": 128,
           "ka2_applied_calls": 1, "ownership": True, "frozen_host": True},
           "control": {"rows": {"counters": {"updates": 500}}, "counters": {"evals": 1, "probes": 1}},
           "initial": {"live_excess_error": 1.}, "endpoint": {"live_excess_error": .5}}
    checks = api["gates"]({"G16": deepcopy(arm), "G64": deepcopy(arm)}, streams_match=True, source_match=True)
    assert not checks["G64_improves_10_percent"]
    assert checks["G16_converges_10_percent"] and checks["G64_converges_10_percent"]
