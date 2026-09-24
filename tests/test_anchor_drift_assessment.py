"""Algebra and artifact-integrity checks; no training or random draws."""

import pytest
import torch

from reports.toy100.anchor_drift_assessment import (
    affine_nullspace_example, horizontal_rectangle, optimizer_metrics, verify_bytes, sha,
    fixed_readout_chart,
)


def test_fitting_after_native_retains_nullspace_while_same_target_pre_start_rests():
    row = affine_nullspace_example()
    assert row["post_output"] == pytest.approx(row["target_delta"], abs=1e-14)
    assert row["pre_output"] == pytest.approx(row["target_delta"], abs=1e-14)
    assert row["inherited_nullspace_displacement"] == pytest.approx([1., -1., 0.], abs=1e-14)
    assert row["post_parameter_norm"] > 6 * row["pre_parameter_norm"]
    assert row["zero_target_pre_state_fit_displacement"] == [0., 0., 0.]


def test_bias_corrected_metric_and_finite_moments_are_audited_from_stored_state():
    group = dict(params=[0], betas=(0., .9), lr=.01, eps=0.)
    state = {0: dict(step=torch.tensor(2.), exp_avg=torch.tensor([.2, .4], dtype=torch.float64),
                     exp_avg_sq=torch.tensor([.19, .76], dtype=torch.float64))}
    snapshot = {name: dict(param_groups=[group], state=state) for name in ("optimizer_d", "optimizer_g")}
    receipt = optimizer_metrics(snapshot)["d"]
    assert receipt["bias_corrected_denominator"]["min"] == pytest.approx(1.)
    assert receipt["bias_corrected_denominator"]["max"] == pytest.approx(2.)
    assert receipt["coordinate_metric"]["max"] == pytest.approx(.01)
    assert receipt["last_moment_derived_unbounded_adam_proposal_l2"] == pytest.approx(2**.5*.002)
    state[0]["exp_avg_sq"][0] = float("nan")
    with pytest.raises(FloatingPointError):
        optimizer_metrics(snapshot)


def test_changed_snapshot_bytes_are_not_silently_assessed(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"complete saved state")
    expected = sha(path.read_bytes())
    assert verify_bytes(path, expected) == b"complete saved state"
    path.write_bytes(b"different saved state")
    with pytest.raises(RuntimeError, match="hash changed"):
        verify_bytes(path, expected)


def test_horizontal_velocity_is_the_exact_minimum_norm_output_lift():
    for x in (0., .5, 1.):
        for z in (0., 1., 10.):
            j = torch.tensor([[1., 0., 0.], [z, 1., x]], dtype=torch.float64)
            kernel = torch.tensor([0., -x, 1.], dtype=torch.float64)
            for dx, du in ((1., 0.), (0., 1.)):
                velocity = torch.tensor([dx, (du-z*dx)/(1+x*x), x*(du-z*dx)/(1+x*x)], dtype=j.dtype)
                output = torch.tensor([dx, du], dtype=j.dtype)
                assert torch.allclose(j@velocity, output, atol=1e-14, rtol=0)
                assert abs(float(kernel@velocity)) < 1e-14
                assert torch.allclose(velocity, torch.linalg.pinv(j)@output, atol=1e-13, rtol=0)


def test_closed_bounded_output_loop_can_accumulate_unbounded_parameter_motion():
    row = horizontal_rectangle()
    assert row["output_vertices"][0] == row["output_vertices"][-1] == (0., 0.)
    assert all(0 <= x <= 1 and 0 <= u <= 1 for x, u in row["output_vertices"])
    assert row["fiber_shift_per_loop"] == pytest.approx(1/2**.5)
    assert row["final_z_after_repeated_loops"] == pytest.approx(100/2**.5)
    next_loop = horizontal_rectangle(initial_z=row["fiber_shift_per_loop"], repetitions=1)
    assert next_loop["final_z_after_repeated_loops"] == pytest.approx(2/2**.5)


def test_fixed_affine_readout_has_path_independent_reference_relative_projection():
    features = torch.tensor([[1., 0., 1.], [0., 1., 1.]], dtype=torch.float64)
    reference = torch.tensor([[.1, -.2], [.3, .2], [.4, -.1]], dtype=torch.float64)
    start = features@reference
    inverse = torch.linalg.pinv(features)
    moving = reference.clone()
    previous = start.clone()
    for delta in (torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64),
                  torch.tensor([[1., 1.], [0., 0.]], dtype=torch.float64),
                  torch.tensor([[0., 1.], [0., 0.]], dtype=torch.float64),
                  torch.zeros_like(start)):
        target = start+delta
        moving += inverse@(target-previous)
        direct = reference+inverse@(target-start)
        assert torch.allclose(moving, direct, atol=1e-14, rtol=0)
        assert torch.allclose(features@direct, target, atol=1e-14, rtol=0)
        previous = target
    assert torch.allclose(moving, reference, atol=1e-14, rtol=0)


def test_feature_chart_is_copied_and_does_not_consume_caller_rng():
    from benchmarks.locked_shared.mlp import SimpleMLPGenerator
    from benchmarks.locked_shared import mode_hold
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
        snapshot = dict(generator=model.state_dict(), prior={"z": torch.randn(12, mode_hold.Z_DIM)})
    before = torch.get_rng_state().clone()
    result = fixed_readout_chart(snapshot)
    assert result["saved_state_unchanged"] and result["caller_rng_unchanged"]
    assert torch.equal(before, torch.get_rng_state())
    assert result["rows"] == 12 and result["columns"] == 97
    assert result["rank"] == 12
    assert result["reconstruction_max_abs_error"] < 1e-6
