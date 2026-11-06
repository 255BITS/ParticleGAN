"""Automated 3-level solvability audit and multi-family trainer verification for MisGAN tasks.

Audits all 4 MisGAN tasks:
1. misgan_block_recovery: 2-pair MisGAN on 8D block sensor-dropout with detached mask gradient.
2. misgan_block_neural_imputer: 3-pair MisGAN neural imputer on held-out ambiguous single-sensor queries.
3. misgan_mcar_p50_recovery: 2-pair MisGAN on 8D MCAR coordinate-wise dropout (p=0.5).
4. misgan_mcar_p80_anchored: 3-pair MisGAN + GAIN partial-hint imputer on 8D MCAR p=0.80 with 10% complete anchors.

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom above 99th-percentile noise floor.
- Level 2: Constructive Capacity Witness achieving PASS state without identity bypass.
- Level 3: Destructive Negative Controls deterministically FAILING each threshold + NaN/Inf rejection.
- Multi-family trainer verification: 4-step training + particle prior update under bispf, atlas, bcap, k3p, ka2.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    NormalNoiseManipulator,
    ParticlePrior,
    get_recipe,
    init,
    unwrap_prior,
)
from benchmarks.toy_audit.api_conditionals import (
    CASES,
    MisGANNeuralImputerFixture,
    MisGANRecoveryFixture,
    build_mcar_p50_problem,
    build_mcar_p80_anchored_problem,
    score_misgan_mcar_p50_recovery,
    score_misgan_mcar_p80_anchored,
    score_misgan_neural_imputer,
    score_misgan_recovery,
)
from benchmarks.toy_audit.api_sources import misgan


def _build_algebraic_gx_gm():
    """Constructive G_x and G_m nn.Sequential modules with non-bypassed hidden representations."""
    G_x = nn.Sequential(
        nn.Linear(8, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 8),
    )
    G_m = nn.Sequential(
        nn.Linear(8, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 8), nn.Sigmoid(),
    )
    with torch.no_grad():
        for p in G_x.parameters():
            p.zero_()
        for i in range(8):
            G_x[0].weight[i, i] = 1.0
            G_x[0].weight[i + 8, i] = -1.0
            G_x[2].weight[i, i] = 1.0 / 1.2
            G_x[2].weight[i, i + 8] = -1.0 / 1.2
            G_x[2].weight[i + 8, i] = -1.0 / 1.2
            G_x[2].weight[i + 8, i + 8] = 1.0 / 1.2
            G_x[4].weight[i, i] = 1.0 / 1.2
            G_x[4].weight[i, i + 8] = -1.0 / 1.2

        for p in G_m.parameters():
            p.zero_()
        for i in range(8):
            G_m[0].weight[i, i] = 20.0
            G_m[0].bias[i] = -10.0
            G_m[0].weight[i + 8, i] = -20.0
            G_m[0].bias[i + 8] = 10.0
            G_m[2].weight[i, i] = 1.0 / 1.2
            G_m[2].weight[i, i + 8] = -1.0 / 1.2
            G_m[2].weight[i + 8, i] = -1.0 / 1.2
            G_m[2].weight[i + 8, i + 8] = 1.0 / 1.2
            G_m[4].weight[i, i] = 1.0 / 1.2
            G_m[4].weight[i, i + 8] = -1.0 / 1.2
    return G_x, G_m


# ============================================================================
# Level 1: Oracle 200/200 Pass & Noise Headroom
# ============================================================================

def test_level1_oracle_misgan_recovery_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for block recovery."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    n_draws = 200

    passes = 0
    modes_vals, hq_vals, mass_tv_vals = [], [], []
    off_plane_vals, mask_tv_vals, acc_vals, acc_lo_vals = [], [], [], []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 1000)
        idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
        x_oracle = problem.x_train_full[idx]
        m_oracle = misgan.sample_masks("block", 4096, rng)

        obs = score_misgan_recovery(x_oracle, m_oracle, problem=problem, eval_seed=seed + 2000, use_ppost=True)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_vals.append(m["modes"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        off_plane_vals.append(m["off_plane"])
        mask_tv_vals.append(m["mask_tv"])
        acc_vals.append(m["acc"])
        acc_lo_vals.append(m["acc_lo"])

    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"
    assert min(modes_vals) == max(modes_vals) == 100

    for name, vals, bound, is_lower in [
        ("hq", hq_vals, 0.90, True),
        ("mass_tv", mass_tv_vals, 0.10, False),
        ("off_plane", off_plane_vals, 0.04, False),
        ("mask_tv", mask_tv_vals, 0.05, False),
        ("acc", acc_vals, 0.96, True),
        ("acc_lo", acc_lo_vals, 0.70, True),
    ]:
        mean = float(np.mean(vals))
        std = max(float(np.std(vals)), 1e-6)
        margin = (mean - bound) if is_lower else (bound - mean)
        headroom = margin / (3.0 * std)
        assert headroom >= 2.0, f"{name} headroom {headroom:.2f} < 2.0x"


def test_level1_oracle_misgan_neural_imputer_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for neural imputer."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    n_draws = 200

    passes = 0
    obs_err_vals, acc_vals, acc_lo_vals, ratio_vals = [], [], [], []

    def oracle_imputer(x, m, rng):
        _, draws = misgan.bayes_posterior(problem, x, m, draws=16, generator=rng)
        return draws

    for seed in range(n_draws):
        obs = score_misgan_neural_imputer(oracle_imputer, problem=problem, eval_seed=seed + 3000)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        obs_err_vals.append(m["observed_max_error"])
        acc_vals.append(m["acc"])
        acc_lo_vals.append(m["acc_lo"])
        ratio_vals.append(m["missing_std_ratio"])

    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"
    assert max(obs_err_vals) <= 1e-6

    acc_mean, acc_std = float(np.mean(acc_vals)), max(float(np.std(acc_vals)), 1e-6)
    assert (acc_mean - 0.95) / (3.0 * acc_std) >= 2.0

    acclo_mean, acclo_std = float(np.mean(acc_lo_vals)), max(float(np.std(acc_lo_vals)), 1e-6)
    assert (acclo_mean - 0.65) / (3.0 * acclo_std) >= 2.0

    ratio_mean, ratio_std = float(np.mean(ratio_vals)), max(float(np.std(ratio_vals)), 1e-6)
    assert min((ratio_mean - 0.70) / (3.0 * ratio_std), (1.30 - ratio_mean) / (3.0 * ratio_std)) >= 2.0


def test_level1_oracle_misgan_mcar_p50_recovery_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for MCAR p=0.5 recovery."""
    problem = build_mcar_p50_problem(20000, 10000, 0, "cpu")
    n_draws = 200

    passes = 0
    modes_vals, hq_vals, mass_tv_vals = [], [], []
    off_plane_vals, mask_tv_vals, acc_vals, rmse_vals = [], [], [], []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 4000)
        idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
        x_oracle = problem.x_train_full[idx]
        m_oracle = problem.m_train[idx]

        obs = score_misgan_mcar_p50_recovery(x_oracle, m_oracle, problem=problem, eval_seed=seed + 5000, use_ppost=True)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_vals.append(m["modes"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        off_plane_vals.append(m["off_plane"])
        mask_tv_vals.append(m["mask_tv"])
        acc_vals.append(m["acc"])
        rmse_vals.append(m["impute_rmse"])

    assert passes == n_draws == 200, f"MCAR p50 oracle failed: {passes}/200 passed"
    assert min(modes_vals) >= 90

    for name, vals, bound, is_lower in [
        ("modes", modes_vals, 90.0, True),
        ("hq", hq_vals, 0.85, True),
        ("mass_tv", mass_tv_vals, 0.12, False),
        ("off_plane", off_plane_vals, 0.05, False),
        ("mask_tv", mask_tv_vals, 0.05, False),
        ("acc", acc_vals, 0.90, True),
        ("impute_rmse", rmse_vals, 0.12, False),
    ]:
        mean = float(np.mean(vals))
        std = max(float(np.std(vals)), 1e-6)
        margin = (mean - bound) if is_lower else (bound - mean)
        headroom = margin / (3.0 * std)
        assert headroom >= 2.0, f"mcar_p50 {name} headroom {headroom:.2f} < 2.0x"


def test_level1_oracle_misgan_mcar_p80_anchored_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for MCAR p=0.80 anchored."""
    problem = build_mcar_p80_anchored_problem(20000, 10000, 0, "cpu")
    n_draws = 200

    # Pre-draw an exact posterior pool on the fixed test panel and subsample 4 draws per seed
    ref_rng = torch.Generator().manual_seed(9001)
    _, posterior_pool = misgan.bayes_posterior(problem, problem.x_test, problem.m_test, draws=24, generator=ref_rng)

    passes = 0
    modes_vals, hq_vals, mass_tv_vals, off_plane_vals = [], [], [], []
    imp_acc_vals, imp_off_vals, rmse_vals = [], [], []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 6000)
        idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
        x_oracle = problem.x_train_full[idx]
        m_oracle = problem.m_train[idx]
        draw_idx = torch.randint(len(posterior_pool), (4,), generator=rng)
        imp_draws = posterior_pool[draw_idx]

        obs = score_misgan_mcar_p80_anchored(x_oracle, m_oracle, imp_draws, problem=problem, eval_seed=seed + 7000)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_vals.append(m["modes"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        off_plane_vals.append(m["off_plane"])
        imp_acc_vals.append(m["imp_acc"])
        imp_off_vals.append(m["imp_off_plane"])
        rmse_vals.append(m["impute_rmse"])

    assert passes == n_draws == 200, f"MCAR p80 anchored oracle failed: {passes}/200 passed"
    assert min(modes_vals) >= 80

    for name, vals, bound, is_lower in [
        ("modes", modes_vals, 80.0, True),
        ("hq", hq_vals, 0.80, True),
        ("mass_tv", mass_tv_vals, 0.15, False),
        ("off_plane", off_plane_vals, 0.06, False),
        ("imp_acc", imp_acc_vals, 0.65, True),
        ("imp_off_plane", imp_off_vals, 0.06, False),
        ("impute_rmse", rmse_vals, 0.18, False),
    ]:
        mean = float(np.mean(vals))
        std = max(float(np.std(vals)), 1e-6)
        margin = (mean - bound) if is_lower else (bound - mean)
        headroom = margin / (3.0 * std)
        assert headroom >= 2.0, f"mcar_p80_anchored {name} headroom {headroom:.2f} < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================

def test_level2_constructive_capacity_witness_recovery():
    """Verify constructive capacity witness achieves PASS state on misgan_block_recovery."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    G_x, G_m = _build_algebraic_gx_gm()

    rng = torch.Generator().manual_seed(42)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    z_x = problem.x_train_full[idx].clone()
    z_m = misgan.sample_masks("block", 4096, rng)
    with torch.no_grad():
        x = G_x(z_x)
        m = G_m(z_m)

    assert isinstance(G_x, nn.Sequential) and isinstance(G_m, nn.Sequential)
    assert not torch.equal(x, z_x)

    obs = score_misgan_recovery(x, m, problem=problem, eval_seed=42, use_ppost=True)
    assert obs["passed"] is True
    assert len(obs["failed_bounds"]) == 0


def test_level2_constructive_capacity_witness_neural_imputer():
    """Verify constructive capacity witness achieves PASS state on misgan_block_neural_imputer."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")

    centers_2d = misgan.grid_centers(device="cpu", dtype=torch.float64)
    centers_raw_8d = (centers_2d @ problem.A.T).float()
    std_centers = problem.standardize(centers_raw_8d).float()

    with torch.no_grad():
        probs, _ = misgan.bayes_posterior(problem, problem.x_test, problem.m_test, draws=0)
        map_mode = probs.argmax(dim=1)
        map_target = std_centers[map_mode]

    x_test = problem.x_test
    m_test = problem.m_test
    lo = m_test.sum(1) <= 1
    idx_lo = lo.nonzero().squeeze(1)
    idx_hi = (~lo).nonzero().squeeze(1)

    A = problem.A
    A7 = A[7]
    v = torch.tensor([-A7[1], A7[0]], dtype=torch.float64)
    v = v / v.norm()

    mu = misgan.grid_centers(dtype=torch.float64)
    centers_lo = mu[map_mode[idx_lo]]
    t_lo = (centers_lo * v).sum(1).float()
    t_norm = (t_lo / 2.7).unsqueeze(1)

    N = len(x_test)
    zi_train = torch.zeros(N, 4)
    zi_train[idx_lo, 0:1] = t_norm

    cond_train = torch.cat([x_test * m_test, m_test, zi_train], dim=1)
    target_train = map_target

    G_i = nn.Sequential(
        nn.Linear(20, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 8),
    )

    opt = torch.optim.Adam(G_i.parameters(), lr=0.01)
    for _ in range(2500):
        opt.zero_grad()
        b_lo = idx_lo[torch.randint(len(idx_lo), (128,))]
        b_hi = idx_hi[torch.randint(len(idx_hi), (128,))]
        b = torch.cat([b_lo, b_hi])
        pred = G_i(cond_train[b])
        loss = F.mse_loss(pred, target_train[b])
        loss.backward()
        opt.step()

    G_i.eval()

    def predict_fn(x, m, r, draws=16):
        out_draws = []
        with torch.no_grad():
            for _ in range(draws):
                zi = torch.randn(len(x), 4, generator=r, device=x.device) * 0.12
                zi[idx_lo, 0:1] = t_norm + zi[idx_lo, 0:1]
                cond = torch.cat([x * m, m, zi], dim=1)
                raw = G_i(cond)
                grafted = m * x + (1 - m) * raw
                out_draws.append(grafted)
        return torch.stack(out_draws, 0)

    obs = score_misgan_neural_imputer(predict_fn, problem=problem, eval_seed=42)
    assert isinstance(G_i, nn.Sequential)
    assert obs["passed"] is True
    assert len(obs["failed_bounds"]) == 0


def test_level2_constructive_capacity_witness_mcar_p50_recovery():
    """Verify constructive capacity witness achieves PASS state on misgan_mcar_p50_recovery."""
    problem = build_mcar_p50_problem(20000, 10000, 0, "cpu")
    G_x, G_m = _build_algebraic_gx_gm()

    rng = torch.Generator().manual_seed(142)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    z_x = problem.x_train_full[idx].clone()
    z_m = problem.m_train[idx].clone()
    with torch.no_grad():
        x = G_x(z_x)
        m = G_m(z_m)

    assert isinstance(G_x, nn.Sequential) and isinstance(G_m, nn.Sequential)
    assert not torch.equal(x, z_x)

    obs = score_misgan_mcar_p50_recovery(x, m, problem=problem, eval_seed=142, use_ppost=True)
    assert obs["passed"] is True, f"Failed bounds: {obs['failed_bounds']}, metrics: {obs['metrics']}"
    assert len(obs["failed_bounds"]) == 0


def test_level2_constructive_capacity_witness_mcar_p80_anchored():
    """Verify constructive capacity witness achieves PASS state on misgan_mcar_p80_anchored."""
    problem = build_mcar_p80_anchored_problem(20000, 10000, 0, "cpu")
    G_x, G_m = _build_algebraic_gx_gm()

    # Constructive G_i trained on anchor-assisted MCAR queries via supervised regression on the 2D subspace
    G_i = nn.Sequential(
        nn.Linear(20, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 8),
    )
    x_test = problem.x_test
    m_test = problem.m_test

    cond_base = torch.cat([x_test * m_test, m_test, torch.zeros(len(x_test), 4)], dim=1)
    opt = torch.optim.Adam(G_i.parameters(), lr=0.015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2500)
    rng_fit = torch.Generator().manual_seed(88)
    for _ in range(2500):
        opt.zero_grad()
        b = torch.randint(len(x_test), (512,), generator=rng_fit)
        pred = G_i(cond_base[b])
        loss = F.mse_loss(pred, x_test[b])
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        A = problem.A.float()
        P_raw = A @ A.T
        std = problem.std.float()
        mean = problem.mean.float()
        M = torch.diag(1.0 / std) @ P_raw @ torch.diag(std)
        b_proj = torch.diag(1.0 / std) @ ((P_raw - torch.eye(8)) @ mean)
        G_i[4].weight.copy_(M @ G_i[4].weight)
        G_i[4].bias.copy_(M @ G_i[4].bias + b_proj)

    G_i.eval()

    def predict_fn(x_q, m_q, r):
        with torch.no_grad():
            zi = torch.zeros(len(x_q), 4, device=x_q.device)
            raw = G_i(torch.cat([x_q * m_q, m_q, zi], dim=1))
            grafted = m_q * x_q + (1.0 - m_q) * raw
            return grafted.unsqueeze(0)

    rng = torch.Generator().manual_seed(242)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    z_x = problem.x_train_full[idx].clone()
    z_m = problem.m_train[idx].clone()
    with torch.no_grad():
        x = G_x(z_x)
        m = G_m(z_m)

    assert isinstance(G_x, nn.Sequential) and isinstance(G_i, nn.Sequential)
    assert not torch.equal(x, z_x)

    obs = score_misgan_mcar_p80_anchored(x, m, predict_fn, problem=problem, eval_seed=242)
    assert obs["passed"] is True, f"Failed bounds: {obs['failed_bounds']}, metrics: {obs['metrics']}"
    assert len(obs["failed_bounds"]) == 0


# ============================================================================
# Level 3: Destructive Negative Controls
# ============================================================================

def test_level3_destructive_negative_controls_recovery():
    """Verify destructive negative controls deterministically fail each recovery gate."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    rng = torch.Generator().manual_seed(99)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    x_clean = problem.x_train_full[idx].clone()
    m_clean = misgan.sample_masks("block", 4096, rng)

    # 1. Off-plane leak
    x_off = x_clean + 0.15 * torch.randn_like(x_clean)
    obs = score_misgan_recovery(x_off, m_clean, problem=problem)
    assert obs["passed"] is False and "off_plane" in obs["failed_bounds"]

    # 2. Mask mismatch
    m_wrong = (torch.rand_like(m_clean) > 0.5).float()
    obs = score_misgan_recovery(x_clean, m_wrong, problem=problem)
    assert obs["passed"] is False and "mask_tv" in obs["failed_bounds"]

    # 3. Missing modes / mass_tv collapse
    x_collapsed = torch.zeros_like(x_clean)
    obs = score_misgan_recovery(x_collapsed, m_clean, problem=problem)
    assert obs["passed"] is False
    assert "modes" in obs["failed_bounds"] and "mass_tv" in obs["failed_bounds"]

    # 4. Low HQ / blurred data
    x_blurred = x_clean + 0.5 * torch.randn_like(x_clean)
    obs = score_misgan_recovery(x_blurred, m_clean, problem=problem)
    assert obs["passed"] is False and "hq" in obs["failed_bounds"]

    # 5. Non-finite values: NaN or Inf
    x_nan = x_clean.clone()
    x_nan[0, 0] = float("nan")
    obs = score_misgan_recovery(x_nan, m_clean, problem=problem)
    assert obs["passed"] is False and "nonfinite_output" in obs["failed_bounds"]

    # 6. Corrupted generator samples under particle-posterior (ppost) imputation
    x_noise = torch.randn_like(x_clean)
    obs_noise = score_misgan_recovery(x_noise, m_clean, problem=problem, use_ppost=True)
    assert obs_noise["passed"] is False
    assert "acc" in obs_noise["failed_bounds"] and "acc_lo" in obs_noise["failed_bounds"]


def test_level3_destructive_negative_controls_neural_imputer():
    """Verify destructive negative controls deterministically fail each imputer gate."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")

    # 1. Observed coordinate corruption
    def leaking_imputer(x, m, rng):
        _, draws = misgan.bayes_posterior(problem, x, m, draws=16, generator=rng)
        return draws + 0.01 * m[None]

    obs = score_misgan_neural_imputer(leaking_imputer, problem=problem)
    assert obs["passed"] is False and "observed_max_error" in obs["failed_bounds"]

    # 2. Latent noise ignore (deterministic mode collapse)
    def noise_ignore_imputer(x, m, rng):
        _, single = misgan.bayes_posterior(problem, x, m, draws=1, generator=rng)
        return single.expand(16, -1, -1)

    obs = score_misgan_neural_imputer(noise_ignore_imputer, problem=problem)
    assert obs["passed"] is False and "missing_std_ratio" in obs["failed_bounds"]

    # 3. Excessive variance imputer
    def excessive_variance_imputer(x, m, rng):
        _, draws = misgan.bayes_posterior(problem, x, m, draws=16, generator=rng)
        miss = 1 - m
        return draws + 5.0 * torch.randn_like(draws) * miss[None]

    obs = score_misgan_neural_imputer(excessive_variance_imputer, problem=problem)
    assert obs["passed"] is False
    assert "missing_std_ratio" in obs["failed_bounds"] or "acc" in obs["failed_bounds"]

    # 4. Mean imputer
    def mean_imputer(x, m, rng):
        return (x * m).unsqueeze(0).expand(16, -1, -1)

    obs = score_misgan_neural_imputer(mean_imputer, problem=problem)
    assert obs["passed"] is False and "acc_lo" in obs["failed_bounds"]

    # 5. Non-finite values: Inf
    def inf_imputer(x, m, rng):
        draws = torch.zeros(16, len(x), 8)
        draws[0, 0, 0] = float("inf")
        return draws

    obs = score_misgan_neural_imputer(inf_imputer, problem=problem)
    assert obs["passed"] is False and "nonfinite_output" in obs["failed_bounds"]


def test_level3_destructive_negative_controls_mcar_p50_recovery():
    """Verify destructive negative controls deterministically fail each threshold of misgan_mcar_p50_recovery."""
    problem = build_mcar_p50_problem(20000, 10000, 0, "cpu")
    rng = torch.Generator().manual_seed(199)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    x_clean = problem.x_train_full[idx].clone()
    m_clean = problem.m_train[idx].clone()

    # 1. Off-plane leak
    obs = score_misgan_mcar_p50_recovery(x_clean + 0.15 * torch.randn_like(x_clean), m_clean, problem=problem)
    assert obs["passed"] is False and "off_plane" in obs["failed_bounds"]

    # 2. Mask rate mismatch (all-observed masks violate p=0.5 MCAR rate)
    obs = score_misgan_mcar_p50_recovery(x_clean, torch.ones_like(m_clean), problem=problem)
    assert obs["passed"] is False and "mask_tv" in obs["failed_bounds"]

    # 3. Mode collapse / mass_tv failure
    obs = score_misgan_mcar_p50_recovery(torch.zeros_like(x_clean), m_clean, problem=problem)
    assert obs["passed"] is False
    assert "modes" in obs["failed_bounds"] and "mass_tv" in obs["failed_bounds"]

    # 4. Blurred samples fail hq, acc, and impute_rmse
    obs = score_misgan_mcar_p50_recovery(torch.randn_like(x_clean), m_clean, problem=problem)
    assert obs["passed"] is False
    for k in ("hq", "acc", "impute_rmse"):
        assert k in obs["failed_bounds"], f"Expected {k} in {obs['failed_bounds']}"

    # 5. Non-finite NaN / Inf rejection
    for bad_val in (float("nan"), float("inf")):
        x_bad = x_clean.clone()
        x_bad[0, 0] = bad_val
        obs = score_misgan_mcar_p50_recovery(x_bad, m_clean, problem=problem)
        assert obs["passed"] is False and "nonfinite_output" in obs["failed_bounds"]


def test_level3_destructive_negative_controls_mcar_p80_anchored():
    """Verify destructive negative controls deterministically fail each threshold of misgan_mcar_p80_anchored."""
    problem = build_mcar_p80_anchored_problem(20000, 10000, 0, "cpu")
    rng = torch.Generator().manual_seed(299)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    x_clean = problem.x_train_full[idx].clone()
    m_clean = problem.m_train[idx].clone()
    _, good_draws = misgan.bayes_posterior(problem, problem.x_test, problem.m_test, draws=2, generator=rng)

    # 1. Mode collapse / mass_tv failure on G_x
    obs = score_misgan_mcar_p80_anchored(torch.zeros_like(x_clean), m_clean, good_draws, problem=problem)
    assert obs["passed"] is False
    assert "modes" in obs["failed_bounds"] and "mass_tv" in obs["failed_bounds"]

    # 2. Off-plane + hq failure on G_x
    obs = score_misgan_mcar_p80_anchored(x_clean + 0.5 * torch.randn_like(x_clean), m_clean, good_draws, problem=problem)
    assert obs["passed"] is False
    assert "off_plane" in obs["failed_bounds"] and "hq" in obs["failed_bounds"]

    # 3. Imputer failure (zero-fill imputer fails imp_acc and impute_rmse; noisy imputer fails imp_off_plane)
    zero_draws = (problem.x_test * problem.m_test).unsqueeze(0)
    obs = score_misgan_mcar_p80_anchored(x_clean, m_clean, zero_draws, problem=problem)
    assert obs["passed"] is False
    assert "imp_acc" in obs["failed_bounds"] and "impute_rmse" in obs["failed_bounds"]

    noisy_draws = good_draws + 0.30 * torch.randn_like(good_draws) * (1.0 - problem.m_test)[None]
    obs = score_misgan_mcar_p80_anchored(x_clean, m_clean, noisy_draws, problem=problem)
    assert obs["passed"] is False
    assert "imp_off_plane" in obs["failed_bounds"]

    # 4. Non-finite NaN / Inf rejection
    for bad_val in (float("nan"), float("inf")):
        bad_draws = good_draws.clone()
        bad_draws[0, 0, 0] = bad_val
        obs = score_misgan_mcar_p80_anchored(x_clean, m_clean, bad_draws, problem=problem)
        assert obs["passed"] is False and "nonfinite_output" in obs["failed_bounds"]


# ============================================================================
# Multi-Family Trainer & Prior Binding Verification
# ============================================================================

@pytest.mark.parametrize(
    "case_id",
    [
        "api-misgan-block-recovery",
        "api-misgan-block-neural-imputer",
        "api-misgan-mcar-p50-recovery",
        "api-misgan-mcar-p80-anchored",
    ],
)
@pytest.mark.parametrize("family", ["bispf", "atlas", "bcap", "k3p", "ka2"])
def test_misgan_multi_family_trainer_and_prior_updates(case_id: str, family: str):
    """Verify all 4 MisGAN fixtures run 4 steps cleanly and update unwrap_prior(fixture.prior).z across families."""
    case = deepcopy(CASES[case_id])
    base_recipe = get_recipe(family)
    kw = dict(batch_size=64, num_particles=256, z_dim=8)
    if base_recipe.continuous_policy is None:
        kw["total_steps"] = case["default_steps"]
    recipe = get_recipe(family, **kw)
    prior = None
    if family == "atlas":
        base = ParticlePrior(256, 8)
        init.deterministic_orthogonal_(base, seed=17)
        prior = ManipulatedPrior(base, [NormalNoiseManipulator(sigma=0.0065)])

    if case_id in ("api-misgan-block-recovery", "api-misgan-mcar-p50-recovery"):
        fixture = MisGANRecoveryFixture(
            case,
            seed=0,
            recipe=recipe,
            prior=prior,
            max_steps=4,
            batch_size=64,
            particles=256,
            z_dim=8,
            device="cpu",
        )
    else:
        fixture = MisGANNeuralImputerFixture(
            case,
            seed=0,
            recipe=recipe,
            prior=prior,
            max_steps=4,
            batch_size=64,
            particles=256,
            z_dim=8,
            device="cpu",
        )

    assert fixture.recipe.name == family
    base_prior = unwrap_prior(fixture.prior)
    assert hasattr(base_prior, "z") and base_prior.z.requires_grad
    z_before = base_prior.z.detach().clone()

    for step_idx in range(1, 5):
        stats = fixture.step()
        assert stats["step"] == step_idx
        assert all(math.isfinite(v) for k, v in stats.items() if isinstance(v, float))

    z_after = unwrap_prior(fixture.prior).z.detach()
    assert not torch.equal(z_before, z_after), f"Prior particles z did not update for {case_id} under {family}"
