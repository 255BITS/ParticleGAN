"""Automated 3-level solvability audit for MisGAN tasks.

Audits both:
1. misgan_block_recovery (R4): 2-pair MisGAN on 8D block sensor-dropout with detached mask gradient.
2. misgan_block_neural_imputer (R4): 3-pair MisGAN neural imputer on held-out ambiguous single-sensor queries.

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom above 99th-percentile noise floor.
- Level 2: Constructive Capacity Witness achieving PASS state.
- Level 3: Destructive Negative Controls deterministically FAILING.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.toy_audit.api_conditionals import (
    score_misgan_recovery,
    score_misgan_neural_imputer,
    misgan_ppost_indices,
)
from benchmarks.toy_audit.api_sources import misgan


# ============================================================================
# Level 1: Oracle 200/200 Pass & Noise Headroom
# ============================================================================

def test_level1_oracle_misgan_recovery_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for recovery."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    n_draws = 200
    
    passes = 0
    modes_vals = []
    hq_vals = []
    mass_tv_vals = []
    off_plane_vals = []
    mask_tv_vals = []
    acc_vals = []
    acc_lo_vals = []

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

    # 1. 200/200 passes
    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"

    # 2. Modes is strictly 100
    assert min(modes_vals) == max(modes_vals) == 100

    # 3. Headroom calculations: (bound_margin) / (3 * std) >= 2.0
    # hq >= 0.90
    hq_mean = float(np.mean(hq_vals))
    hq_std = max(float(np.std(hq_vals)), 1e-6)
    hq_headroom = (hq_mean - 0.90) / (3.0 * hq_std)
    assert hq_headroom >= 2.0, f"hq headroom {hq_headroom:.2f} < 2.0x"

    # mass_tv <= 0.10
    mass_mean = float(np.mean(mass_tv_vals))
    mass_std = max(float(np.std(mass_tv_vals)), 1e-6)
    mass_headroom = (0.10 - mass_mean) / (3.0 * mass_std)
    assert mass_headroom >= 2.0, f"mass_tv headroom {mass_headroom:.2f} < 2.0x"

    # off_plane <= 0.04
    off_mean = float(np.mean(off_plane_vals))
    off_std = max(float(np.std(off_plane_vals)), 1e-6)
    off_headroom = (0.04 - off_mean) / (3.0 * off_std)
    assert off_headroom >= 2.0, f"off_plane headroom {off_headroom:.2f} < 2.0x"

    # mask_tv <= 0.05
    mtv_mean = float(np.mean(mask_tv_vals))
    mtv_std = max(float(np.std(mask_tv_vals)), 1e-6)
    mtv_headroom = (0.05 - mtv_mean) / (3.0 * mtv_std)
    assert mtv_headroom >= 2.0, f"mask_tv headroom {mtv_headroom:.2f} < 2.0x"

    # acc >= 0.96
    acc_mean = float(np.mean(acc_vals))
    acc_std = max(float(np.std(acc_vals)), 1e-6)
    acc_headroom = (acc_mean - 0.96) / (3.0 * acc_std)
    assert acc_headroom >= 2.0, f"acc headroom {acc_headroom:.2f} < 2.0x"

    # acc_lo >= 0.70
    acclo_mean = float(np.mean(acc_lo_vals))
    acclo_std = max(float(np.std(acc_lo_vals)), 1e-6)
    acclo_headroom = (acclo_mean - 0.70) / (3.0 * acclo_std)
    assert acclo_headroom >= 2.0, f"acc_lo headroom {acclo_headroom:.2f} < 2.0x"


def test_level1_oracle_misgan_neural_imputer_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for neural imputer."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    n_draws = 200

    passes = 0
    obs_err_vals = []
    acc_vals = []
    acc_lo_vals = []
    ratio_vals = []

    # Oracle imputer: draws directly from analytic Bayes posterior
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

    # acc >= 0.95
    acc_mean = float(np.mean(acc_vals))
    acc_std = max(float(np.std(acc_vals)), 1e-6)
    acc_headroom = (acc_mean - 0.95) / (3.0 * acc_std)
    assert acc_headroom >= 2.0, f"acc headroom {acc_headroom:.2f} < 2.0x"

    # acc_lo >= 0.65
    acclo_mean = float(np.mean(acc_lo_vals))
    acclo_std = max(float(np.std(acc_lo_vals)), 1e-6)
    acclo_headroom = (acclo_mean - 0.65) / (3.0 * acclo_std)
    assert acclo_headroom >= 2.0, f"acc_lo headroom {acclo_headroom:.2f} < 2.0x"

    # missing_std_ratio in [0.70, 1.30]
    ratio_mean = float(np.mean(ratio_vals))
    ratio_std = max(float(np.std(ratio_vals)), 1e-6)
    ratio_lo_headroom = (ratio_mean - 0.70) / (3.0 * ratio_std)
    ratio_hi_headroom = (1.30 - ratio_mean) / (3.0 * ratio_std)
    assert min(ratio_lo_headroom, ratio_hi_headroom) >= 2.0, f"std ratio headroom < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================

def test_level2_constructive_capacity_witness_recovery():
    """Verify constructive capacity witness achieves PASS state on misgan_block_recovery."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")
    
    # Constructive witness: actual G_x and G_m network instances from api_conditionals.py
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

    # Algebraic constructive weights:
    # G_x implements exact identity mapping on R^8 using LeakyReLU(u) - LeakyReLU(-u) = 1.2u
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

        # G_m maps binary mask latents {0, 1}^8 through pre-activation 20.0 * z - 10.0 to Sigmoid
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

    rng = torch.Generator().manual_seed(42)
    idx = torch.randint(len(problem.x_train_full), (4096,), generator=rng)
    z_x = problem.x_train_full[idx].clone()
    z_m = misgan.sample_masks("block", 4096, rng)
    with torch.no_grad():
        x = G_x(z_x)
        m = G_m(z_m)

    assert isinstance(G_x, nn.Sequential)
    assert isinstance(G_m, nn.Sequential)
    assert not torch.equal(x, z_x)

    obs = score_misgan_recovery(x, m, problem=problem, eval_seed=42, use_ppost=True)
    assert obs["passed"] is True
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["modes"] == 100
    assert obs["metrics"]["hq"] >= 0.90
    assert obs["metrics"]["mass_tv"] <= 0.10
    assert obs["metrics"]["off_plane"] <= 0.04
    assert obs["metrics"]["mask_tv"] <= 0.05
    assert obs["metrics"]["acc"] >= 0.96
    assert obs["metrics"]["acc_lo"] >= 0.70


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

    # Genuine neural imputer G_i network instance from api_conditionals.py
    G_i = nn.Sequential(
        nn.Linear(20, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 8),
    )

    opt = torch.optim.Adam(G_i.parameters(), lr=0.01)
    for step in range(2500):
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
            for d in range(draws):
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
    assert obs["metrics"]["observed_max_error"] <= 1e-6
    assert obs["metrics"]["acc"] >= 0.95
    assert obs["metrics"]["acc_lo"] >= 0.65
    assert 0.70 <= obs["metrics"]["missing_std_ratio"] <= 1.30


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

    # 1. Off-plane leak: perturb orthogonal to the subspace
    # Adds high off-plane noise (0.10 > 0.04 gate)
    x_off = x_clean + 0.15 * torch.randn_like(x_clean)
    obs = score_misgan_recovery(x_off, m_clean, problem=problem)
    assert obs["passed"] is False
    assert "off_plane" in obs["failed_bounds"]

    # 2. Mask mismatch: random uniform masks rather than block dropout
    m_wrong = (torch.rand_like(m_clean) > 0.5).float()
    obs = score_misgan_recovery(x_clean, m_wrong, problem=problem)
    assert obs["passed"] is False
    assert "mask_tv" in obs["failed_bounds"]

    # 3. Missing modes: collapse all data to a single mode
    x_collapsed = torch.zeros_like(x_clean)
    obs = score_misgan_recovery(x_collapsed, m_clean, problem=problem)
    assert obs["passed"] is False
    assert "modes" in obs["failed_bounds"] or "hq" in obs["failed_bounds"]

    # 4. Low HQ / blurred data
    x_blurred = x_clean + 0.5 * torch.randn_like(x_clean)
    obs = score_misgan_recovery(x_blurred, m_clean, problem=problem)
    assert obs["passed"] is False
    assert "hq" in obs["failed_bounds"]

    # 5. Non-finite values: NaN or Inf
    x_nan = x_clean.clone()
    x_nan[0, 0] = float("nan")
    obs = score_misgan_recovery(x_nan, m_clean, problem=problem)
    assert obs["passed"] is False
    assert "nonfinite_output" in obs["failed_bounds"]

    # 6. Corrupted generator samples under particle-posterior (ppost) imputation:
    # Pure Gaussian noise x_noise ~ N(0, I) must deterministically fail acc and acc_lo
    x_noise = torch.randn_like(x_clean)
    obs_noise = score_misgan_recovery(x_noise, m_clean, problem=problem, use_ppost=True)
    assert obs_noise["passed"] is False
    assert "acc" in obs_noise["failed_bounds"]
    assert "acc_lo" in obs_noise["failed_bounds"]
    assert obs_noise["metrics"]["acc"] < 0.20
    assert obs_noise["metrics"]["acc_lo"] < 0.05

    # 7. Mode collapsed generator samples x = 0 under ppost imputation:
    x_zero = torch.zeros_like(x_clean)
    obs_zero = score_misgan_recovery(x_zero, m_clean, problem=problem, use_ppost=True)
    assert obs_zero["passed"] is False
    assert "acc" in obs_zero["failed_bounds"]
    assert "acc_lo" in obs_zero["failed_bounds"]


def test_level3_destructive_negative_controls_neural_imputer():
    """Verify destructive negative controls deterministically fail each imputer gate."""
    problem = misgan.Problem("block", 20000, 10000, 0, "cpu")

    # 1. Observed coordinate corruption: leaks modification into observed coordinates
    def leaking_imputer(x, m, rng):
        _, draws = misgan.bayes_posterior(problem, x, m, draws=16, generator=rng)
        # Corrupt observed coordinates
        draws = draws + 0.01 * m[None]
        return draws

    obs = score_misgan_neural_imputer(leaking_imputer, problem=problem)
    assert obs["passed"] is False
    assert "observed_max_error" in obs["failed_bounds"]

    # 2. Latent noise ignore (deterministic mode collapse): ignores z_i
    def noise_ignore_imputer(x, m, rng):
        # Always return identical constant imputation across all draws
        _, single = misgan.bayes_posterior(problem, x, m, draws=1, generator=rng)
        return single.expand(16, -1, -1)

    obs = score_misgan_neural_imputer(noise_ignore_imputer, problem=problem)
    assert obs["passed"] is False
    assert "missing_std_ratio" in obs["failed_bounds"]
    assert obs["metrics"]["missing_std_ratio"] < 0.70

    # 3. Excessive variance imputer: injects huge random noise into missing coordinates
    def excessive_variance_imputer(x, m, rng):
        _, draws = misgan.bayes_posterior(problem, x, m, draws=16, generator=rng)
        miss = 1 - m
        draws = draws + 5.0 * torch.randn_like(draws) * miss[None]
        return draws

    obs = score_misgan_neural_imputer(excessive_variance_imputer, problem=problem)
    assert obs["passed"] is False
    assert "missing_std_ratio" in obs["failed_bounds"] or "acc" in obs["failed_bounds"]

    # 4. Mean imputer (imputes 0 for missing values): fails accuracy gates
    def mean_imputer(x, m, rng):
        # In standardized coordinates, mean is 0
        imputed = x * m  # missing filled with 0
        return imputed.unsqueeze(0).expand(16, -1, -1)

    obs = score_misgan_neural_imputer(mean_imputer, problem=problem)
    assert obs["passed"] is False
    assert "acc_lo" in obs["failed_bounds"]

    # 5. Non-finite values: Inf
    def inf_imputer(x, m, rng):
        draws = torch.zeros(16, len(x), 8)
        draws[0, 0, 0] = float("inf")
        return draws

    obs = score_misgan_neural_imputer(inf_imputer, problem=problem)
    assert obs["passed"] is False
    assert "nonfinite_output" in obs["failed_bounds"]
