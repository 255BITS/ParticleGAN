"""Math and frozen-host checks for the cross-only response with own-curvature bound."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.cross_curvature_scratch import CrossCurvatureRecorder, cross_curvature


def _run_quadratic_game(a, b, recorder, lr=.1):
    # D=y, G=x. F_y = -x + b y, F_x = y + a x. Own blocks a, b.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=lr, betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=lr, betas=(0., .9), eps=1e-12)
    draws = []
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        draws.append(torch.rand(3))
        y.grad = (-x + b * y).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = (y + a * x).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    return x, y, opt_g, opt_d, draws


def _exact_cross_step(a, b, lr=.1):
    fy, fx = -1. + b * 2., 2. + a * 1.
    p = torch.diag(torch.tensor([lr / abs(fy), lr / abs(fx)], dtype=torch.float64))
    k = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    field = torch.tensor([fy, fx], dtype=torch.float64)
    delta = -torch.linalg.solve(torch.eye(2, dtype=torch.float64) + p @ k, p @ field)
    # Own-curvature ratio along each player's step is P_p * |own block|.
    factor_y = min(1., 1. / (float(p[0, 0]) * abs(b))) if b else 1.
    factor_x = min(1., 1. / (float(p[1, 1]) * abs(a))) if a else 1.
    return 1. + factor_x * float(delta[1]), 2. + factor_y * float(delta[0])


def test_pure_bilinear_matches_full_implicit_solution_and_replays_rng():
    recorder = CrossCurvatureRecorder(krylov_dim=4, linear_tolerance=1e-8)
    host_rng_before = torch.get_rng_state().clone()
    x, y, opt_g, opt_d, draws = _run_quadratic_game(0., 0., recorder)
    # Without own curvature, the cross-only system equals the full implicit one.
    assert x.item() == pytest.approx(.895522388, abs=2e-7)
    assert y.item() == pytest.approx(2.089552239, abs=2e-7)
    assert all(torch.equal(draws[0], draw) for draw in draws[1:])
    torch.set_rng_state(host_rng_before)
    torch.rand(3)
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.solves[-1]["accepted"]
    for row in recorder.receipt()["optimizers"]:
        assert row["calls"] == 1 + len(recorder.queries)


def test_own_curvature_is_ignored_by_cross_only_response():
    a, b = 3., -2.
    recorder = CrossCurvatureRecorder(krylov_dim=4, linear_tolerance=1e-8)
    x, y, *_ = _run_quadratic_game(a, b, recorder)
    expected_x, expected_y = _exact_cross_step(a, b)
    assert x.item() == pytest.approx(expected_x, abs=1e-6)
    assert y.item() == pytest.approx(expected_y, abs=1e-6)
    kinds = {query["kind"] for query in recorder.queries}
    assert kinds == {"cross_jvp_d", "cross_jvp_g", "own_curvature_d", "own_curvature_g"}
    assert recorder.curvature[-1]["g"]["factor"] == recorder.curvature[-1]["d"]["factor"] == 1.
    summary = recorder.summary()
    assert summary["moment_updates_per_outer_step"] == 1
    assert summary["gradient_evaluations_per_player"] == 1 + len(recorder.queries)


def test_large_own_curvature_scales_only_that_players_step():
    a, b, lr = 3., -.01, 10.
    recorder = CrossCurvatureRecorder(krylov_dim=4, linear_tolerance=1e-8)
    x, y, *_ = _run_quadratic_game(a, b, recorder, lr=lr)
    expected_x, expected_y = _exact_cross_step(a, b, lr=lr)
    row = recorder.curvature[-1]
    assert row["g"]["rho"] == pytest.approx(lr / 5. * 3., rel=1e-4)
    assert row["g"]["factor"] == pytest.approx(1. / row["g"]["rho"])
    assert row["d"]["factor"] == 1.
    assert x.item() == pytest.approx(expected_x, abs=1e-6)
    assert y.item() == pytest.approx(expected_y, abs=1e-6)


def test_zero_game_field_stays_still():
    x = torch.nn.Parameter(torch.tensor([0.]))
    y = torch.nn.Parameter(torch.tensor([0.]))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9))
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9))
    recorder = CrossCurvatureRecorder()
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = -x.detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = y.detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    assert x.item() == y.item() == 0
    assert recorder.queries == []
    assert recorder.solves[0]["zero_field"]


def test_inactive_adapter_preserves_frozen_host_noise_and_rng():
    torch.set_num_threads(1)

    def run():
        policy = NoisePolicy(.029, .5, .1, 1200)
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
                                           noise_policy=policy, diagnostics=True)
        return result, policy.receipt(), torch.get_rng_state().clone(), \
            policy.input_stream.get_state().clone()

    ordinary = run()
    with cross_curvature(start_step=1000) as (recorder, _):
        wrapped = run()
    assert ordinary[:2] == wrapped[:2]
    assert all(torch.equal(left, right) for left, right in zip(ordinary[2:], wrapped[2:]))
    assert recorder.outer_steps == 0


def test_active_host_consumes_one_training_batch_and_one_moment_update():
    torch.set_num_threads(1)

    def run(context=None):
        policy = NoisePolicy(.029, .5, .1, 1200)
        if context is None:
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1), noise_policy=policy)
            recorder = None
        else:
            with context as (recorder, _):
                mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1), noise_policy=policy)
        return (policy.receipt(), torch.get_rng_state().clone(),
                policy.input_stream.get_state().clone(), recorder)

    ordinary = run()
    active = run(cross_curvature(start_step=0, krylov_dim=2))
    assert ordinary[0]["step_calls"] == active[0]["step_calls"] == 1
    assert all(torch.equal(left, right) for left, right in zip(ordinary[1:3], active[1:3]))
    recorder = active[3]
    assert recorder.outer_steps == 1
    assert recorder.rng_replay_verified == len(recorder.queries)
    assert all(query["kind"].startswith(("cross_jvp_", "own_curvature_")) for query in recorder.queries)
    assert len(recorder.curvature) == 1
    for row in recorder.receipt()["optimizers"]:
        assert row["calls"] == 1 + len(recorder.queries)
        assert all(step == 1 for group in row["groups"] for step in group["moment_steps"])


def test_amplification_bound_caps_each_players_step_at_its_explicit_adam_step():
    # Exact bilinear CGD moves G by -.1045 versus explicit -.1 and D by .0896
    # versus explicit .1, so only G is clamped.
    recorder = CrossCurvatureRecorder(krylov_dim=4, linear_tolerance=1e-8, amplification_bound=1.)
    x, y, *_ = _run_quadratic_game(0., 0., recorder)
    assert x.item() == pytest.approx(.9, abs=1e-7)
    assert y.item() == pytest.approx(2.089552239, abs=2e-7)
    row = recorder.curvature[-1]
    assert row["g_amplification"]["clamped"] and not row["d_amplification"]["clamped"]
    assert row["g_amplification"]["proposed_to_explicit"] == pytest.approx(1.044776, abs=1e-5)


def test_explicit_mode_is_plain_adam_until_own_curvature_bound_binds():
    recorder = CrossCurvatureRecorder(explicit=True)
    x, y, *_ = _run_quadratic_game(0., 0., recorder)
    assert x.item() == pytest.approx(.9, abs=1e-9)
    assert y.item() == pytest.approx(2.1, abs=1e-9)
    assert {q["kind"] for q in recorder.queries} == {"own_curvature_d", "own_curvature_g"}
    a, b, lr = 3., -.01, 10.
    recorder = CrossCurvatureRecorder(explicit=True)
    x, y, *_ = _run_quadratic_game(a, b, recorder, lr=lr)
    rho = lr / 5. * 3.
    assert recorder.curvature[-1]["g"]["rho"] == pytest.approx(rho, rel=1e-4)
    assert x.item() == pytest.approx(1. - lr / 5. * 5. / rho, rel=1e-4)
    assert y.item() == pytest.approx(2. - lr / abs(-1.02) * -1.02, rel=1e-6)
