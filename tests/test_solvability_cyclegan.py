"""Automated 3-level solvability & multi-family execution audit for CycleGAN / DiscoGAN.

Audits cyclegan_unpaired_constellation (api-cyclegan-unpaired-constellation):
Unpaired 2D domain translation & bijection (G_xy, G_yx, D_y, D_x) between an
8-mode ring (R_X = 2.5, sigma = 0.08) and an 8-mode square-perimeter constellation
({-2, 0, +2}^2 \\ {(0, 0)}, sigma = 0.08) with cycle-consistency regularization.

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom above 99th-percentile noise floor.
- Level 2: Constructive Capacity Witness (CycleGANTranslator nn.Module) achieving PASS state
  without identity bypass (and proving that identity mapping fails).
- Level 3: Destructive Negative Controls deterministically FAILING each individual gate.
- Multi-family execution test: 4 steps of bispf, atlas (with ManipulatedPrior), bcap, k3p, ka2
  all run cleanly and update unwrap_prior(fixture.prior).z.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    NormalNoiseManipulator,
    ParticlePrior,
    unwrap_prior,
)
from benchmarks.toy_audit.api_cyclegan import (
    CASE_ID,
    CycleGANConstellationFixture,
    CycleGANTranslator,
    build_case,
    domain_x_centers,
    domain_y_centers,
    score_cyclegan,
)


def _oracle_translate_xy(x: torch.Tensor, centers_x: torch.Tensor, centers_y: torch.Tensor) -> torch.Tensor:
    k = torch.cdist(x, centers_x).argmin(dim=1)
    return centers_y[k] + (x - centers_x[k])


def _oracle_translate_yx(y: torch.Tensor, centers_x: torch.Tensor, centers_y: torch.Tensor) -> torch.Tensor:
    j = torch.cdist(y, centers_y).argmin(dim=1)
    return centers_x[j] + (y - centers_y[j])


# ============================================================================
# Level 1: Oracle 200/200 Pass & Noise Headroom
# ============================================================================

def test_level1_oracle_cyclegan_200_draws():
    """Verify Level 1 Oracle bijection passes 200/200 draws with >= 2.0x noise headroom."""
    centers_x = domain_x_centers()
    centers_y = domain_y_centers()
    sigma = 0.08
    n_draws = 200
    n_samples = 4096

    passes = 0
    modes_y_vals = []
    modes_x_vals = []
    hq_y_vals = []
    hq_x_vals = []
    mass_tv_y_vals = []
    cycle_rmse_vals = []
    bijection_ber_vals = []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 1000)

        kx = torch.randint(0, 8, (n_samples,), generator=rng)
        real_x = centers_x[kx] + torch.randn(n_samples, 2, generator=rng) * sigma

        ky = torch.randint(0, 8, (n_samples,), generator=rng)
        real_y = centers_y[ky] + torch.randn(n_samples, 2, generator=rng) * sigma

        fake_y = _oracle_translate_xy(real_x, centers_x, centers_y)
        fake_x = _oracle_translate_yx(real_y, centers_x, centers_y)

        # Include small finite-precision / stochastic round-trip noise (sigma_cyc = 0.015)
        cyc_x = _oracle_translate_yx(fake_y, centers_x, centers_y) + torch.randn(n_samples, 2, generator=rng) * 0.015
        cyc_y = _oracle_translate_xy(fake_x, centers_x, centers_y) + torch.randn(n_samples, 2, generator=rng) * 0.015
        mapped_centers_y = _oracle_translate_xy(centers_x, centers_x, centers_y)

        obs = score_cyclegan(
            real_x=real_x,
            real_y=real_y,
            fake_y=fake_y,
            fake_x=fake_x,
            cyc_x=cyc_x,
            cyc_y=cyc_y,
            mode_y_from_x_centers=mapped_centers_y,
            centers_x=centers_x,
            centers_y=centers_y,
            sigma=sigma,
        )
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_y_vals.append(m["modes_y"])
        modes_x_vals.append(m["modes_x"])
        hq_y_vals.append(m["hq_y"])
        hq_x_vals.append(m["hq_x"])
        mass_tv_y_vals.append(m["mass_tv_y"])
        cycle_rmse_vals.append(m["cycle_rmse"])
        bijection_ber_vals.append(m["bijection_ber"])

    # 1. 200/200 passes
    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"

    # 2. Discrete mode coverage & bijection gates
    assert min(modes_y_vals) == 8 >= 7
    assert min(modes_x_vals) == 8 >= 7
    assert max(bijection_ber_vals) == 0.0 <= 0.15

    # 3. Continuous noise headroom: margin / (3 * std) >= 2.0
    hq_y_mean = float(np.mean(hq_y_vals))
    hq_y_std = max(float(np.std(hq_y_vals)), 1e-6)
    hq_y_headroom = (hq_y_mean - 0.85) / (3.0 * hq_y_std)
    assert hq_y_headroom >= 2.0, f"hq_y headroom {hq_y_headroom:.2f} < 2.0x"

    hq_x_mean = float(np.mean(hq_x_vals))
    hq_x_std = max(float(np.std(hq_x_vals)), 1e-6)
    hq_x_headroom = (hq_x_mean - 0.85) / (3.0 * hq_x_std)
    assert hq_x_headroom >= 2.0, f"hq_x headroom {hq_x_headroom:.2f} < 2.0x"

    tv_mean = float(np.mean(mass_tv_y_vals))
    tv_std = max(float(np.std(mass_tv_y_vals)), 1e-6)
    tv_headroom = (0.15 - tv_mean) / (3.0 * tv_std)
    assert tv_headroom >= 2.0, f"mass_tv_y headroom {tv_headroom:.2f} < 2.0x"

    cyc_mean = float(np.mean(cycle_rmse_vals))
    cyc_std = max(float(np.std(cycle_rmse_vals)), 1e-6)
    cyc_headroom = (0.12 - cyc_mean) / (3.0 * cyc_std)
    assert cyc_headroom >= 2.0, f"cycle_rmse headroom {cyc_headroom:.2f} < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness (Without Identity Bypass)
# ============================================================================

def test_level2_constructive_capacity_witness_cyclegan():
    """Verify CycleGANTranslator nn.Module capacity witness passes all thresholds without identity bypass."""
    centers_x = domain_x_centers()
    centers_y = domain_y_centers()
    sigma = 0.08

    torch.manual_seed(0)
    g_xy = CycleGANTranslator(in_dim=2, out_dim=2, hidden=128, z_dim=2)
    g_yx = CycleGANTranslator(in_dim=2, out_dim=2, hidden=128, z_dim=2)
    opt = torch.optim.Adam(list(g_xy.parameters()) + list(g_yx.parameters()), lr=0.008)

    rng_train = torch.Generator().manual_seed(11)
    for _ in range(300):
        opt.zero_grad()
        k = torch.randint(0, 8, (256,), generator=rng_train)
        eps = torch.randn(256, 2, generator=rng_train) * sigma
        x = centers_x[k] + eps
        y = centers_y[k] + eps

        pred_y = g_xy(x)
        pred_x = g_yx(y)
        rec_x = g_yx(pred_y)
        rec_y = g_xy(pred_x)

        loss = (
            F.mse_loss(pred_y, y)
            + F.mse_loss(pred_x, x)
            + F.mse_loss(rec_x, x)
            + F.mse_loss(rec_y, y)
        )
        loss.backward()
        opt.step()

    class CycleGANWitness(nn.Module):
        def __init__(self, translator_xy: CycleGANTranslator, translator_yx: CycleGANTranslator):
            super().__init__()
            self.G_xy = translator_xy
            self.G_yx = translator_yx

        def forward_xy(self, x: torch.Tensor) -> torch.Tensor:
            return self.G_xy(x)

        def forward_yx(self, y: torch.Tensor) -> torch.Tensor:
            return self.G_yx(y)

    witness = CycleGANWitness(g_xy, g_yx)
    witness.eval()

    rng_eval = torch.Generator().manual_seed(42)
    kx = torch.randint(0, 8, (4096,), generator=rng_eval)
    real_x = centers_x[kx] + torch.randn(4096, 2, generator=rng_eval) * sigma
    ky = torch.randint(0, 8, (4096,), generator=rng_eval)
    real_y = centers_y[ky] + torch.randn(4096, 2, generator=rng_eval) * sigma

    with torch.no_grad():
        fake_y = witness.forward_xy(real_x)
        fake_x = witness.forward_yx(real_y)
        cyc_x = witness.forward_yx(fake_y)
        cyc_y = witness.forward_xy(fake_x)
        mapped_centers_y = witness.forward_xy(centers_x)

    assert isinstance(witness.G_xy, CycleGANTranslator)
    assert isinstance(witness.G_yx, CycleGANTranslator)
    assert not torch.allclose(fake_y, real_x), "G_xy must not be an identity map"
    assert not torch.allclose(fake_x, real_y), "G_yx must not be an identity map"
    assert not torch.equal(cyc_x, real_x), "Cycle reconstruction must not be an identity bypass"
    assert not torch.equal(cyc_y, real_y), "Cycle reconstruction must not be an identity bypass"

    obs = score_cyclegan(
        real_x=real_x,
        real_y=real_y,
        fake_y=fake_y,
        fake_x=fake_x,
        cyc_x=cyc_x,
        cyc_y=cyc_y,
        mode_y_from_x_centers=mapped_centers_y,
        centers_x=centers_x,
        centers_y=centers_y,
        sigma=sigma,
    )
    assert obs["passed"] is True, f"Constructive witness failed bounds: {obs['failed_bounds']}, metrics={obs['metrics']}"
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["modes_y"] >= 7
    assert obs["metrics"]["modes_x"] >= 7
    assert obs["metrics"]["hq_y"] >= 0.85
    assert obs["metrics"]["hq_x"] >= 0.85
    assert obs["metrics"]["mass_tv_y"] <= 0.15
    assert 0.0 < obs["metrics"]["cycle_rmse"] <= 0.12
    assert obs["metrics"]["bijection_ber"] <= 0.15

    # Also verify that an identity bypass (fake_y = real_x, fake_x = real_y) FAILS
    # because Domain X (ring R=2.5) and Domain Y (square L=2.0) have disjoint supports
    obs_identity = score_cyclegan(
        real_x=real_x,
        real_y=real_y,
        fake_y=real_x.clone(),
        fake_x=real_y.clone(),
        cyc_x=real_x.clone(),
        cyc_y=real_y.clone(),
        mode_y_from_x_centers=centers_x.clone(),
        centers_x=centers_x,
        centers_y=centers_y,
        sigma=sigma,
    )
    assert obs_identity["passed"] is False
    assert "modes_y" in obs_identity["failed_bounds"]
    assert "hq_y" in obs_identity["failed_bounds"]
    assert "hq_x" in obs_identity["failed_bounds"]


# ============================================================================
# Level 3: Destructive Negative Controls
# ============================================================================

def test_level3_destructive_negative_controls_cyclegan():
    """Verify destructive negative controls deterministically fail each individual threshold."""
    centers_x = domain_x_centers()
    centers_y = domain_y_centers()
    sigma = 0.08
    rng = torch.Generator().manual_seed(123)

    kx = torch.randint(0, 8, (4096,), generator=rng)
    real_x = centers_x[kx] + torch.randn(4096, 2, generator=rng) * sigma
    ky = torch.randint(0, 8, (4096,), generator=rng)
    real_y = centers_y[ky] + torch.randn(4096, 2, generator=rng) * sigma

    fake_y = _oracle_translate_xy(real_x, centers_x, centers_y)
    fake_x = _oracle_translate_yx(real_y, centers_x, centers_y)
    cyc_x = real_x.clone()
    cyc_y = real_y.clone()
    mapped_centers_y = centers_y.clone()

    # 1. Many-to-one mode collapse in G_xy (DiscoGAN Figure 2 failure mode:
    # all 8 source modes collapse onto 2 target modes in Domain Y)
    collapsed_k = kx % 2
    fake_y_collapsed = centers_y[collapsed_k] + torch.randn(4096, 2, generator=rng) * sigma
    mapped_collapsed = centers_y[torch.arange(8) % 2]
    obs_collapse_y = score_cyclegan(
        real_x, real_y, fake_y_collapsed, fake_x, cyc_x, cyc_y, mapped_collapsed,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_collapse_y["passed"] is False
    assert "modes_y" in obs_collapse_y["failed_bounds"]
    assert "bijection_ber" in obs_collapse_y["failed_bounds"]

    # 2. Many-to-one mode collapse in G_yx (fails modes_x >= 7)
    collapsed_j = ky % 3
    fake_x_collapsed = centers_x[collapsed_j] + torch.randn(4096, 2, generator=rng) * sigma
    obs_collapse_x = score_cyclegan(
        real_x, real_y, fake_y, fake_x_collapsed, cyc_x, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_collapse_x["passed"] is False
    assert "modes_x" in obs_collapse_x["failed_bounds"]

    # 3. Bijection failure at mode centers (6 distinct target modes -> BER = 2/8 = 0.25 > 0.15)
    mapped_non_bijective = centers_y[torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])]
    obs_ber = score_cyclegan(
        real_x, real_y, fake_y, fake_x, cyc_x, cyc_y, mapped_non_bijective,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_ber["passed"] is False
    assert "bijection_ber" in obs_ber["failed_bounds"]

    # 4. Broken inverse / cycle inconsistency (G_yx is rotated by 1 mode relative to G_xy^-1)
    cyc_x_broken = centers_x[(kx + 1) % 8] + torch.randn(4096, 2, generator=rng) * sigma
    obs_bad_cycle = score_cyclegan(
        real_x, real_y, fake_y, fake_x, cyc_x_broken, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_bad_cycle["passed"] is False
    assert "cycle_rmse" in obs_bad_cycle["failed_bounds"]

    # 5. Off-manifold outputs in Domain Y (fails hq_y >= 0.85)
    fake_y_noisy = centers_y[kx] + torch.randn(4096, 2, generator=rng) * 0.45
    obs_low_hq_y = score_cyclegan(
        real_x, real_y, fake_y_noisy, fake_x, cyc_x, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_low_hq_y["passed"] is False
    assert "hq_y" in obs_low_hq_y["failed_bounds"]

    # 6. Off-manifold outputs in Domain X (fails hq_x >= 0.85)
    fake_x_noisy = centers_x[ky] + torch.randn(4096, 2, generator=rng) * 0.45
    obs_low_hq_x = score_cyclegan(
        real_x, real_y, fake_y, fake_x_noisy, cyc_x, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_low_hq_x["passed"] is False
    assert "hq_x" in obs_low_hq_x["failed_bounds"]

    # 7. Unbalanced mode masses in Domain Y (fails mass_tv_y <= 0.15 while covering all 8 modes)
    skew_k = torch.cat([
        torch.zeros(2048, dtype=torch.long),
        torch.arange(8).repeat(256),
    ])
    fake_y_skew = centers_y[skew_k] + torch.randn(4096, 2, generator=rng) * sigma
    obs_skew = score_cyclegan(
        real_x, real_y, fake_y_skew, fake_x, cyc_x, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_skew["passed"] is False
    assert obs_skew["metrics"]["modes_y"] == 8
    assert "mass_tv_y" in obs_skew["failed_bounds"]

    # 8. Non-finite output (NaN / Inf)
    fake_y_nan = fake_y.clone()
    fake_y_nan[0, 0] = float("nan")
    obs_nan = score_cyclegan(
        real_x, real_y, fake_y_nan, fake_x, cyc_x, cyc_y, mapped_centers_y,
        centers_x=centers_x, centers_y=centers_y, sigma=sigma,
    )
    assert obs_nan["passed"] is False
    assert "nonfinite_output" in obs_nan["failed_bounds"]


# ============================================================================
# Multi-Family Trainer Execution Test
# ============================================================================

@pytest.mark.parametrize("recipe_name", ["bispf", "atlas", "bcap", "k3p", "ka2"])
def test_multi_family_execution_cyclegan(recipe_name: str):
    """Verify 4 steps of bispf, atlas (with ManipulatedPrior), bcap, k3p, ka2 run cleanly and update prior.z."""
    if recipe_name == "atlas":
        base_prior = ParticlePrior(num_particles=256, z_dim=2, init_std=1.0)
        prior = ManipulatedPrior(base_prior, [NormalNoiseManipulator(sigma=0.0065)])
        fixture = CycleGANConstellationFixture(
            recipe_name="atlas",
            prior=prior,
            device="cpu",
            seed=0,
            max_steps=8,
        )
    else:
        fixture = build_case(
            CASE_ID,
            recipe_name=recipe_name,
            device="cpu",
            seed=0,
            max_steps=8,
        )

    z_before = unwrap_prior(fixture.prior).z.detach().clone()
    g_xy_before = [p.detach().clone() for p in fixture.G_xy.parameters()]

    for step_idx in range(4):
        info = fixture.step()
        assert info["step"] == step_idx + 1
        assert np.isfinite(info["loss_d"])
        assert np.isfinite(info["loss_g"])

    z_after = unwrap_prior(fixture.prior).z.detach().clone()
    assert torch.all(torch.isfinite(z_after)), f"Non-finite prior.z for {recipe_name}"
    assert not torch.allclose(z_before, z_after), f"unwrap_prior(fixture.prior).z did not update for {recipe_name}"
    assert any(
        not torch.allclose(p0, p1.detach())
        for p0, p1 in zip(g_xy_before, fixture.G_xy.parameters())
    ), f"G_xy parameters did not update for {recipe_name}"
