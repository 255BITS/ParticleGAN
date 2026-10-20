"""Three-level solvability audit for grid100_mlp16 and grid100_res16_tanh tasks.

Level 1: Oracle 200/200 pass verification with >= 2x headroom above finite-sample noise floor.
Level 2: Constructive capacity witness:
         - 16-layer MLP identity representation theorem and pass state.
         - 16-layer Residual MLP with zero-residual identity blocks and pass state.
Level 3: Destructive negative controls deterministically failing on:
         - Missing modes (< 100 modes)
         - Collapsed mode width (sigma = 0.005 vs 0.05)
         - Shifted centers (> 0.40 sigma)
         - Tanh rail saturation injection (> 0.001)
         - Non-finite outputs (NaN / Inf)
"""
import pytest
import torch
import numpy as np

from benchmarks.toy_audit import api_vectors as api
from benchmarks.toy_audit.api_vectors import ResidualMLPGenerator
from benchmarks.toy100 import accuracy, problems
from lib.toy_models import SimpleMLPGenerator


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_level1_oracle_grid100_mlp16_200_trials():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    center_max_records = []
    center_rms_records = []
    mass_tv_records = []
    cov_trace_records = []
    radial_ks_records = []

    for seed in range(200):
        rng = torch.Generator().manual_seed(seed + 100)
        oracle_sample = api._target(case, case["eval_samples"], rng, case["default_steps"])
        score = api.score_case(case, oracle_sample, completed_steps=case["default_steps"])
        assert score["passed"] is True, f"Oracle failed on seed {seed}: {score.get('failed_bounds')}"
        m = score["metrics"]
        center_max_records.append(m["accuracy_center_max_sigma"])
        center_rms_records.append(m["accuracy_center_rms_sigma"])
        mass_tv_records.append(m["accuracy_mass_tv"])
        cov_trace_records.append(m["accuracy_abs_cov_trace_bias"])
        radial_ks_records.append(m["accuracy_radial_ks"])

    # Verify >= 2x headroom between finite-sample oracle noise floor and declared limits
    mean_center_rms = float(np.mean(center_rms_records))
    mean_mass_tv = float(np.mean(mass_tv_records))
    mean_cov_trace = float(np.mean(cov_trace_records))
    mean_radial_ks = float(np.mean(radial_ks_records))

    assert accuracy.LIMITS["center_rms_sigma"] >= 2.0 * mean_center_rms, (
        f"center_rms_sigma limit {accuracy.LIMITS['center_rms_sigma']} lacks 2x headroom over mean {mean_center_rms}"
    )
    assert accuracy.LIMITS["mass_tv"] >= 2.0 * mean_mass_tv, (
        f"mass_tv limit {accuracy.LIMITS['mass_tv']} lacks 2x headroom over mean {mean_mass_tv}"
    )
    assert accuracy.LIMITS["abs_cov_trace_bias"] >= 2.0 * mean_cov_trace, (
        f"abs_cov_trace_bias limit {accuracy.LIMITS['abs_cov_trace_bias']} lacks 2x headroom over mean {mean_cov_trace}"
    )
    assert accuracy.LIMITS["radial_ks"] >= 2.0 * mean_radial_ks, (
        f"radial_ks limit {accuracy.LIMITS['radial_ks']} lacks 2x headroom over mean {mean_radial_ks}"
    )


def test_level1_oracle_grid100_res16_tanh_200_trials():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-res16-tanh")
    sat_fractions = []

    for seed in range(200):
        rng = torch.Generator().manual_seed(seed + 300)
        oracle_sample = api._target(case, case["eval_samples"], rng, case["default_steps"])
        score = api.score_case(case, oracle_sample, completed_steps=case["default_steps"])
        assert score["passed"] is True, f"Oracle failed on seed {seed}: {score.get('failed_bounds')}"
        sat_fractions.append(score["metrics"]["saturated_fraction"])

    max_sat = max(sat_fractions)
    assert max_sat == 0.0, f"Expected 0.0 saturated fraction for oracle, got {max_sat}"
    # Declared limit is 0.001; oracle is 0.0, so headroom is strictly >= 2x above noise floor
    assert case.get("max_saturated_fraction", 0.001) >= 2.0 * max(max_sat, 1e-5)


def test_level2_constructive_capacity_mlp16_identity_representation():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    gen = SimpleMLPGenerator(z_dim=2, hidden_dim=128, n_hidden=16, out_dim=2)

    # Verify structural depth: 16 hidden layers (17 linear layers total)
    linears = [m for m in gen.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 17, f"Expected 17 linear layers (16 hidden + 1 head), got {len(linears)}"

    # Construct exact identity map across both positive and negative quadrants:
    # LeakyReLU(x, 0.2) - LeakyReLU(-x, 0.2) = 1.2 * x identically for all x in R.
    with torch.no_grad():
        for p in gen.parameters():
            p.zero_()
        # Stem: Linear(2, 128)
        linears[0].weight[0, 0] = 1.0
        linears[0].weight[1, 1] = 1.0
        linears[0].weight[2, 0] = -1.0
        linears[0].weight[3, 1] = -1.0
        # 15 hidden layers: Linear(128, 128)
        for l in linears[1:-1]:
            l.weight[0, 0] = 1.0 / 1.2
            l.weight[0, 2] = -1.0 / 1.2
            l.weight[1, 1] = 1.0 / 1.2
            l.weight[1, 3] = -1.0 / 1.2
            l.weight[2, 0] = -1.0 / 1.2
            l.weight[2, 2] = 1.0 / 1.2
            l.weight[3, 1] = -1.0 / 1.2
            l.weight[3, 3] = 1.0 / 1.2
        # Head: Linear(128, 2)
        linears[-1].weight[0, 0] = 1.0 / 1.2
        linears[-1].weight[0, 2] = -1.0 / 1.2
        linears[-1].weight[1, 1] = 1.0 / 1.2
        linears[-1].weight[1, 3] = -1.0 / 1.2

    # Verify algebraic reconstruction error on arbitrary inputs across [-5, 5]^2
    torch.manual_seed(99)
    test_z = (torch.rand(5000, 2) - 0.5) * 10.0
    with torch.no_grad():
        reconstructed = gen(test_z)
    max_reconstruction_error = float(torch.max(torch.abs(reconstructed - test_z)))
    assert max_reconstruction_error < 1e-5, f"Identity representation error {max_reconstruction_error} >= 1e-5"

    # Evaluated on grid100 modes, this identity generator perfectly preserves modes and passes
    rng = torch.Generator().manual_seed(42)
    grid_inputs = api._target(case, case["eval_samples"], rng, case["default_steps"])
    with torch.no_grad():
        grid_outputs = gen(grid_inputs)
    score = api.score_case(case, grid_outputs)
    assert score["passed"] is True, f"Constructive identity generator failed to pass: {score.get('failed_bounds')}"
    assert score["metrics"]["modes"] == 100


def test_level2_constructive_capacity_res16_tanh():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-res16-tanh")
    scale = case.get("scale", 5.25)
    gen = ResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=scale)

    # Verify structural depth: 1 stem + 7 blocks * 2 linears + 1 head = 16 linear layers
    linears = [m for m in gen.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 16, f"Expected 16 linear layers, got {len(linears)}"

    # Configure residual blocks to identity (linear2 = 0 implies h + 0 = h identically)
    # Stem maps [z0, z1] to [z0, z1, -z0, -z1]
    # Head maps [h0, h1, h2, h3] to (h0 - h2)/1.2, (h1 - h3)/1.2
    with torch.no_grad():
        for p in gen.parameters():
            p.zero_()
        gen.stem[0].weight[0, 0] = 1.0
        gen.stem[0].weight[1, 1] = 1.0
        gen.stem[0].weight[2, 0] = -1.0
        gen.stem[0].weight[3, 1] = -1.0
        gen.head.weight[0, 0] = 1.0 / 1.2
        gen.head.weight[0, 2] = -1.0 / 1.2
        gen.head.weight[1, 1] = 1.0 / 1.2
        gen.head.weight[1, 3] = -1.0 / 1.2

    # Given inputs pre-warped by 5.25 * atanh(y / 5.25), output is 5.25 * tanh(head / 5.25) = y
    rng = torch.Generator().manual_seed(42)
    grid_targets = api._target(case, case["eval_samples"], rng, case["default_steps"])
    clamped_ratio = torch.clamp(grid_targets / scale, -0.999, 0.999)
    warped_z = scale * torch.atanh(clamped_ratio)

    with torch.no_grad():
        gen_outputs = gen(warped_z)

    score = api.score_case(case, gen_outputs)
    assert score["passed"] is True, f"Residual tanh generator failed to pass: {score.get('failed_bounds')}"
    assert score["metrics"]["modes"] == 100
    assert score["metrics"]["saturated_fraction"] <= 0.001


def test_level3_destructive_missing_mode():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    rng = torch.Generator().manual_seed(42)
    target_pts = api._target(case, case["eval_samples"], rng, case["default_steps"])

    # Drop mode at corner (-4.5, -4.5) and duplicate points from another mode
    corner = torch.tensor([-4.5, -4.5])
    dists = torch.norm(target_pts - corner, dim=1)
    pts_without_mode = target_pts[dists > 0.2]
    replacement = target_pts[dists <= 0.2] + torch.tensor([1.0, 0.0])
    corrupted_pts = torch.cat([pts_without_mode, replacement], dim=0)

    score = api.score_case(case, corrupted_pts)
    assert score["passed"] is False
    assert any("modes" in bound for bound in score["failed_bounds"])


def test_level3_destructive_collapsed_mode_width():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    centers = problems._centers("grid100", device="cpu", dtype=torch.float32)
    torch.manual_seed(77)
    idx = torch.randint(0, 100, (case["eval_samples"],))
    # Collapsed width: sigma = 0.005 instead of nominal 0.05
    collapsed_pts = centers[idx] + torch.randn(case["eval_samples"], 2) * 0.005

    score = api.score_case(case, collapsed_pts)
    assert score["passed"] is False
    assert any("min_cov_eig_ratio" in b or "min_radial_median_ratio" in b for b in score["failed_bounds"])


def test_level3_destructive_shifted_centers():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    rng = torch.Generator().manual_seed(42)
    target_pts = api._target(case, case["eval_samples"], rng, case["default_steps"])

    # Shift centers by 0.35 * sigma along diagonal (effective displacement 0.35 * sqrt(2) * sigma ~ 0.495 sigma > 0.40)
    shifted_pts = target_pts + 0.35 * 0.05 * torch.tensor([1.0, 1.0])

    score = api.score_case(case, shifted_pts)
    assert score["passed"] is False
    assert any("accuracy_center_rms_sigma" in b or "precision" in b for b in score["failed_bounds"])


def test_level3_destructive_tanh_saturation_injection():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-res16-tanh")
    rng = torch.Generator().manual_seed(42)
    target_pts = api._target(case, case["eval_samples"], rng, case["default_steps"])

    # Inject 2% (400 / 20000) saturated points at rail boundaries (threshold is 0.001 = 0.1%)
    sat_pts = target_pts.clone()
    sat_pts[:400] = 5.25

    score = api.score_case(case, sat_pts)
    assert score["passed"] is False
    assert "saturated_fraction <= 0.001" in score["failed_bounds"]


def test_level3_destructive_non_finite_outputs():
    case = next(c for c in api.list_cases() if c["id"] == "api-grid100-mlp16")
    rng = torch.Generator().manual_seed(42)
    target_pts = api._target(case, case["eval_samples"], rng, case["default_steps"])

    nan_pts = target_pts.clone()
    nan_pts[0, 0] = float("nan")
    score_nan = api.score_case(case, nan_pts)
    assert score_nan["passed"] is False
    assert "all_finite == true" in score_nan["failed_bounds"]

    inf_pts = target_pts.clone()
    inf_pts[0, 0] = float("inf")
    score_inf = api.score_case(case, inf_pts)
    assert score_inf["passed"] is False
    assert "all_finite == true" in score_inf["failed_bounds"]
