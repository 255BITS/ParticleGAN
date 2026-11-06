"""Automated 3-level solvability audit and multi-family verification for CramerGAN.

Audits cramer_energy_overlap (api-cramer-energy-overlap):
Bellemare et al. (NeurIPS 2017) multi-sample unbiased Cramer energy distance on
the 9-mode 3x3 overlapping heterogeneous-mass 2D Gaussian mixture.

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom (margin / (3 * std) >= 2.0).
- Level 2: Constructive nn.Module capacity witness achieving PASS state without identity bypass.
- Level 3: Destructive negative controls deterministically failing each individual gate.
- Multi-family execution test: 4 steps across bispf, atlas (ManipulatedPrior), bcap, k3p, ka2.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    NormalNoiseManipulator,
    ParticlePrior,
    unwrap_prior,
)
from benchmarks.toy_audit.api_cramergan import (
    CRAMER_SIGMA,
    CramerCritic,
    CramerGANOverlapFixture,
    CramerGenerator,
    cramer_centers,
    cramer_weights,
    sample_cramer_target,
    score_cramergan,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ============================================================================
# Level 1: Oracle 200/200 Pass & >= 2.0x Noise Headroom
# ============================================================================


def test_level1_oracle_cramergan_200_draws():
    """Verify 200/200 Oracle draws pass all thresholds with >= 2.0x noise headroom."""
    centers = cramer_centers()
    weights = cramer_weights()
    n_draws = 200

    passes = 0
    modes_vals = []
    hq_vals = []
    mass_tv_vals = []
    energy_dist_vals = []
    min_cov_eig_vals = []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 1000)
        samples = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
        target = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)

        obs = score_cramergan(samples, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_vals.append(m["modes"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        energy_dist_vals.append(m["energy_distance"])
        min_cov_eig_vals.append(m["min_cov_eig_ratio"])

    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"
    assert min(modes_vals) == max(modes_vals) == 9

    # hq >= 0.88
    hq_mean = float(np.mean(hq_vals))
    hq_std = max(float(np.std(hq_vals)), 1e-6)
    hq_headroom = (hq_mean - 0.88) / (3.0 * hq_std)
    assert hq_headroom >= 2.0, f"hq headroom {hq_headroom:.2f} < 2.0x"

    # mass_tv <= 0.10
    tv_mean = float(np.mean(mass_tv_vals))
    tv_std = max(float(np.std(mass_tv_vals)), 1e-6)
    tv_headroom = (0.10 - tv_mean) / (3.0 * tv_std)
    assert tv_headroom >= 2.0, f"mass_tv headroom {tv_headroom:.2f} < 2.0x"

    # energy_distance <= 0.06
    ed_mean = float(np.mean(energy_dist_vals))
    ed_std = max(float(np.std(energy_dist_vals)), 1e-6)
    ed_headroom = (0.06 - ed_mean) / (3.0 * ed_std)
    assert ed_headroom >= 2.0, f"energy_distance headroom {ed_headroom:.2f} < 2.0x"

    # min_cov_eig_ratio >= 0.35
    eig_mean = float(np.mean(min_cov_eig_vals))
    eig_std = max(float(np.std(min_cov_eig_vals)), 1e-6)
    eig_headroom = (eig_mean - 0.35) / (3.0 * eig_std)
    assert eig_headroom >= 2.0, f"min_cov_eig_ratio headroom {eig_headroom:.2f} < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================


def test_level2_constructive_capacity_witness_cramergan():
    """Verify constructive nn.Module capacity witness passes all CramerGAN gates without identity bypass."""
    centers = cramer_centers()
    weights = cramer_weights()

    torch.manual_seed(7)
    gen = CramerGenerator(z_dim=2, out_dim=2, hidden=128)
    critic = CramerCritic(in_dim=2, embed_dim=16, hidden=128)
    opt = torch.optim.Adam(gen.parameters(), lr=0.008)

    rng_train = torch.Generator().manual_seed(77)
    for _ in range(250):
        opt.zero_grad()
        x = sample_cramer_target(256, generator=rng_train, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
        loss = F.mse_loss(gen(x), x)
        loss.backward()
        opt.step()

    class CramerWitness(nn.Module):
        def __init__(self, generator: CramerGenerator, discriminator: CramerCritic):
            super().__init__()
            self.G = generator
            self.D = discriminator

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.G(z)

    witness = CramerWitness(gen, critic)
    witness.eval()

    rng_eval = torch.Generator().manual_seed(42)
    z_eval = sample_cramer_target(4096, generator=rng_eval, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    target = sample_cramer_target(4096, generator=rng_eval, centers=centers, weights=weights, sigma=CRAMER_SIGMA)

    with torch.no_grad():
        samples = witness(z_eval)
        emb = witness.D.embed(samples[:64])
        logits = witness.D(samples[:64])

    assert isinstance(witness.G, CramerGenerator)
    assert isinstance(witness.D, CramerCritic)
    assert emb.shape == (64, 16)
    assert logits.shape == (64,)
    assert not torch.equal(samples, z_eval), "Generator output must not be an identity bypass"

    obs = score_cramergan(samples, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs["passed"] is True, f"Constructive witness failed: {obs['failed_bounds']} ({obs['metrics']})"
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["modes"] == 9
    assert obs["metrics"]["hq"] >= 0.88
    assert obs["metrics"]["mass_tv"] <= 0.10
    assert obs["metrics"]["energy_distance"] <= 0.06
    assert obs["metrics"]["min_cov_eig_ratio"] >= 0.35


# ============================================================================
# Level 3: Destructive Negative Controls
# ============================================================================


def test_level3_destructive_negative_controls_cramergan():
    """Verify destructive negative controls deterministically fail each individual threshold."""
    centers = cramer_centers()
    weights = cramer_weights()
    rng = torch.Generator().manual_seed(123)
    target = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    clean_samples = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=CRAMER_SIGMA)

    # 1. Mode drop: drop mode 8 (0 samples in mode 8 -> modes == 8 != 9)
    w_drop = weights.clone()
    w_drop[8] = 0.0
    w_drop = w_drop / w_drop.sum()
    samples_drop = sample_cramer_target(4096, generator=rng, centers=centers, weights=w_drop, sigma=CRAMER_SIGMA)
    obs_drop = score_cramergan(samples_drop, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_drop["passed"] is False
    assert "modes" in obs_drop["failed_bounds"]
    assert obs_drop["metrics"]["modes"] == 8

    # 2. Off-support noise: excessive dispersion sigma=0.75 >> 0.16 fails hq >= 0.88
    samples_noisy = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=0.75)
    obs_noisy = score_cramergan(samples_noisy, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_noisy["passed"] is False
    assert "hq" in obs_noisy["failed_bounds"]
    assert obs_noisy["metrics"]["hq"] < 0.88

    # 3. Uniform mode weights: biased gradient equalizing masses fails mass_tv <= 0.10
    w_uniform = torch.full((9,), 1.0 / 9.0)
    samples_uniform = sample_cramer_target(4096, generator=rng, centers=centers, weights=w_uniform, sigma=CRAMER_SIGMA)
    obs_uniform = score_cramergan(samples_uniform, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_uniform["passed"] is False
    assert "mass_tv" in obs_uniform["failed_bounds"]
    assert obs_uniform["metrics"]["mass_tv"] > 0.10

    # 4. Shifted distribution: global shift (+0.35, +0.35) fails energy_distance <= 0.06
    samples_shifted = clean_samples + torch.tensor([0.35, 0.35])
    obs_shifted = score_cramergan(samples_shifted, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_shifted["passed"] is False
    assert "energy_distance" in obs_shifted["failed_bounds"]
    assert obs_shifted["metrics"]["energy_distance"] > 0.06

    # 5. Intra-mode variance collapse: sigma=0.02 << 0.16 fails min_cov_eig_ratio >= 0.35
    samples_collapsed = sample_cramer_target(4096, generator=rng, centers=centers, weights=weights, sigma=0.02)
    obs_collapsed = score_cramergan(samples_collapsed, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_collapsed["passed"] is False
    assert "min_cov_eig_ratio" in obs_collapsed["failed_bounds"]
    assert obs_collapsed["metrics"]["min_cov_eig_ratio"] < 0.35

    # 6. Non-finite values: NaN and Inf deterministically fail
    samples_nan = clean_samples.clone()
    samples_nan[0, 0] = float("nan")
    obs_nan = score_cramergan(samples_nan, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_nan["passed"] is False
    assert "nonfinite_output" in obs_nan["failed_bounds"]

    samples_inf = clean_samples.clone()
    samples_inf[0, 1] = float("inf")
    obs_inf = score_cramergan(samples_inf, target, centers=centers, weights=weights, sigma=CRAMER_SIGMA)
    assert obs_inf["passed"] is False
    assert "nonfinite_output" in obs_inf["failed_bounds"]


# ============================================================================
# Multi-Family Trainer Execution Test
# ============================================================================


@pytest.mark.parametrize("recipe_name", ["bispf", "atlas", "bcap", "k3p", "ka2"])
def test_multi_family_execution_cramergan(recipe_name: str):
    """Verify 4 steps of bispf, atlas (ManipulatedPrior), bcap, k3p, ka2 run cleanly and update prior.z."""
    if recipe_name == "atlas":
        base = ParticlePrior(num_particles=256, z_dim=2, init_std=1.0)
        prior = ManipulatedPrior(base, [NormalNoiseManipulator(sigma=0.0065)])
    elif recipe_name == "ka2":
        prior = MoGParticlePrior(num_particles=256, z_dim=2, sigma=0.10, standardize=False)
    else:
        prior = None

    fixture = CramerGANOverlapFixture(recipe_name=recipe_name, prior=prior, device="cpu", seed=0, max_steps=16)
    z_initial = unwrap_prior(fixture.prior).z.detach().clone()

    for step_idx in range(1, 5):
        info = fixture.step()
        assert info["step"] == step_idx
        assert math.isfinite(info["loss_d"])
        assert math.isfinite(info["loss_g"])
        assert math.isfinite(info["loss_cramer_d"])
        assert math.isfinite(info["loss_cramer_g"])

    z_after = unwrap_prior(fixture.prior).z.detach()
    assert torch.all(torch.isfinite(z_after))
    assert not torch.equal(z_initial, z_after), f"Prior table z was not updated under {recipe_name}"

    state = fixture.state_dict()
    fixture.load_state_dict(state)
    obs = fixture.observe(n=256, seed=10000)
    assert isinstance(obs["passed"], bool)
    assert set(obs["metrics"]) == {"modes", "hq", "mass_tv", "energy_distance", "min_cov_eig_ratio"}
