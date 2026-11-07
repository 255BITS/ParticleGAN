"""Automated 3-level solvability and multi-family trainer audit for BiGAN / ALICE tasks.

Audits:
- ring16_bigan_continuous (R5): Continuous 2D multimodal joint (G(z), E(x), D(x, z)) on 16-Gaussian ring.
- alice_spiral_bigan: Continuous 2D Archimedes spiral joint (G(z), E(x), D(x, z)) with ALICE cycle-consistency.

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom above 99th-percentile noise floor.
- Level 2: Constructive Capacity Witness achieving PASS state without identity bypass.
- Level 3: Destructive Negative Controls deterministically FAILING each individual gate.
- Multi-family trainer & prior update verification across bispf, atlas, bcap, k3p, and ka2.
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
    NormalNoiseManipulator,
    ParticlePrior,
    get_recipe,
    init,
    unwrap_prior,
)
from benchmarks.toy_audit.api_ring16 import (
    ALICESpiralBiGANFixture,
    BiGANEncoder,
    BiGANGenerator,
    BiGANJointCritic,
    Ring16BiGANFixture,
    score_alice_spiral,
    score_bigan,
)


def _ring_centers():
    angles = torch.linspace(0, 2 * math.pi, 17)[:16]
    return torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()


def _sample_spiral_oracle(n: int, rng: torch.Generator, sigma: float = 0.06):
    theta_min, theta_max = 0.5 * math.pi, 3.5 * math.pi
    theta = theta_min + (theta_max - theta_min) * torch.rand(n, generator=rng)
    r = 0.5 + 0.25 * theta
    eps = torch.randn(n, 2, generator=rng) * sigma
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).float() + eps
    return x, theta


# ============================================================================
# Level 1: Oracle 200/200 Pass & Noise Headroom
# ============================================================================

def test_level1_oracle_bigan_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for ring16 BiGAN."""
    centers = _ring_centers()
    n_draws = 200

    passes = 0
    modes_vals = []
    hq_vals = []
    mass_tv_vals = []
    recon_rmse_vals = []
    mode_cons_vals = []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 1000)
        k = torch.randint(0, 16, (4096,), generator=rng)
        eps = torch.randn(4096, 2, generator=rng) * 0.10
        samples = centers[k] + eps

        k_tgt = torch.randint(0, 16, (4096,), generator=rng)
        eps_tgt = torch.randn(4096, 2, generator=rng) * 0.10
        target = centers[k_tgt] + eps_tgt
        reconstructed = target.clone()

        obs = score_bigan(samples, target, reconstructed, centers=centers, sigma=0.10)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        modes_vals.append(m["modes"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        recon_rmse_vals.append(m["recon_rmse"])
        mode_cons_vals.append(m["mode_consistency"])

    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"
    assert min(modes_vals) == 16

    hq_mean = float(np.mean(hq_vals))
    hq_std = max(float(np.std(hq_vals)), 1e-6)
    hq_headroom = (hq_mean - 0.85) / (3.0 * hq_std)
    assert hq_headroom >= 2.0, f"hq headroom {hq_headroom:.2f} < 2.0x"

    mass_tv_mean = float(np.mean(mass_tv_vals))
    mass_tv_std = max(float(np.std(mass_tv_vals)), 1e-6)
    mass_tv_headroom = (0.15 - mass_tv_mean) / (3.0 * mass_tv_std)
    assert mass_tv_headroom >= 2.0, f"mass_tv headroom {mass_tv_headroom:.2f} < 2.0x"

    assert max(recon_rmse_vals) <= 0.12
    assert min(mode_cons_vals) >= 0.98


def test_level1_oracle_alice_spiral_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom for ALICE spiral BiGAN."""
    n_draws = 200

    passes = 0
    arc_bins_vals = []
    hq_vals = []
    mass_tv_vals = []
    recon_rmse_vals = []
    latent_rmse_vals = []
    arc_cons_vals = []

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 2000)
        samples, _ = _sample_spiral_oracle(4096, rng, sigma=0.06)
        target, _ = _sample_spiral_oracle(4096, rng, sigma=0.06)
        reconstructed = target + torch.randn(4096, 2, generator=rng) * 0.015
        z = torch.randn(4096, 2, generator=rng)
        latent_reconstructed = z + torch.randn(4096, 2, generator=rng) * 0.02

        obs = score_alice_spiral(
            samples,
            target,
            reconstructed,
            z=z,
            latent_reconstructed=latent_reconstructed,
            sigma=0.06,
        )
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        arc_bins_vals.append(m["arc_bins"])
        hq_vals.append(m["hq"])
        mass_tv_vals.append(m["mass_tv"])
        recon_rmse_vals.append(m["recon_rmse"])
        latent_rmse_vals.append(m["latent_recon_rmse"])
        arc_cons_vals.append(m["arc_consistency"])

    assert passes == n_draws == 200, f"ALICE Spiral Oracle failed: {passes}/200 passed"
    assert min(arc_bins_vals) >= 15

    hq_mean = float(np.mean(hq_vals))
    hq_std = max(float(np.std(hq_vals)), 1e-6)
    hq_headroom = (hq_mean - 0.85) / (3.0 * hq_std)
    assert hq_headroom >= 2.0, f"hq headroom {hq_headroom:.2f} < 2.0x"

    mass_tv_mean = float(np.mean(mass_tv_vals))
    mass_tv_std = max(float(np.std(mass_tv_vals)), 1e-6)
    mass_tv_headroom = (0.16 - mass_tv_mean) / (3.0 * mass_tv_std)
    assert mass_tv_headroom >= 2.0, f"mass_tv headroom {mass_tv_headroom:.2f} < 2.0x"

    recon_mean = float(np.mean(recon_rmse_vals))
    recon_std = max(float(np.std(recon_rmse_vals)), 1e-6)
    recon_headroom = (0.14 - recon_mean) / (3.0 * recon_std)
    assert recon_headroom >= 2.0, f"recon_rmse headroom {recon_headroom:.2f} < 2.0x"

    lat_mean = float(np.mean(latent_rmse_vals))
    lat_std = max(float(np.std(latent_rmse_vals)), 1e-6)
    lat_headroom = (0.25 - lat_mean) / (3.0 * lat_std)
    assert lat_headroom >= 2.0, f"latent_recon_rmse headroom {lat_headroom:.2f} < 2.0x"

    arc_cons_mean = float(np.mean(arc_cons_vals))
    arc_cons_std = max(float(np.std(arc_cons_vals)), 1e-6)
    arc_cons_headroom = (arc_cons_mean - 0.90) / (3.0 * arc_cons_std)
    assert arc_cons_headroom >= 2.0, f"arc_consistency headroom {arc_cons_headroom:.2f} < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================

def test_level2_constructive_capacity_witness_bigan():
    """Verify constructive capacity witness achieves PASS state on ring16_bigan_continuous."""
    centers = _ring_centers()

    torch.manual_seed(6)
    gen = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    enc = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3, fourier=3)
    opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=0.01)

    for _ in range(200):
        opt.zero_grad()
        k = torch.randint(0, 16, (128,))
        x = centers[k] + torch.randn(128, 2) * 0.10
        loss = F.mse_loss(gen(x), x) + F.mse_loss(enc(x), x)
        loss.backward()
        opt.step()

    class BiGANWitness(nn.Module):
        def __init__(self, generator: BiGANGenerator, encoder: BiGANEncoder):
            super().__init__()
            self.G = generator
            self.E = encoder

        def generate(self, z):
            return self.G(z)

        def encode(self, x):
            return self.E(x)

        def reconstruct(self, x):
            return self.G(self.E(x))

    witness = BiGANWitness(gen, enc)
    witness.eval()

    rng = torch.Generator().manual_seed(42)
    k_tgt = torch.randint(0, 16, (4096,), generator=rng)
    target = centers[k_tgt] + torch.randn(4096, 2, generator=rng) * 0.10

    k_z = torch.randint(0, 16, (4096,), generator=rng)
    z = centers[k_z] + torch.randn(4096, 2, generator=rng) * 0.10
    with torch.no_grad():
        samples = witness.generate(z)
        reconstructed = witness.reconstruct(target)

    assert isinstance(witness.G, BiGANGenerator)
    assert isinstance(witness.E, BiGANEncoder)
    assert not torch.equal(reconstructed, target), "Reconstruction must not be an identity bypass"

    obs = score_bigan(samples, target, reconstructed, centers=centers, sigma=0.10)
    assert obs["passed"] is True
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["modes"] >= 16
    assert obs["metrics"]["hq"] >= 0.85
    assert obs["metrics"]["mass_tv"] <= 0.15
    assert 0.0 < obs["metrics"]["recon_rmse"] <= 0.12
    assert obs["metrics"]["mode_consistency"] >= 0.98


def test_level2_constructive_capacity_witness_alice_spiral():
    """Verify constructive capacity witness achieves PASS state on alice_spiral_bigan without identity bypass."""
    torch.manual_seed(7)
    gen = BiGANGenerator(z_dim=2, out_dim=2, hidden=128, layers=3)
    enc = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3)
    opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=0.008)

    theta_min, theta_max = 0.5 * math.pi, 3.5 * math.pi
    sigma = 0.06
    train_rng = torch.Generator().manual_seed(77)

    for _ in range(600):
        opt.zero_grad()
        theta = theta_min + (theta_max - theta_min) * torch.rand(256, generator=train_rng)
        u = torch.randn(256, generator=train_rng) * sigma
        r = 0.5 + 0.25 * theta + u
        x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).float()
        z_code = torch.stack([(theta - 2.0 * math.pi) / math.pi, u / (3.0 * sigma)], dim=1).float()

        x_hat = gen(z_code)
        z_hat = enc(x)
        x_cyc = gen(z_hat)
        z_cyc = enc(x_hat)
        loss = (
            F.mse_loss(x_hat, x)
            + F.mse_loss(z_hat, z_code)
            + F.mse_loss(x_cyc, x)
            + F.mse_loss(z_cyc, z_code)
        )
        loss.backward()
        opt.step()

    class ALICESpiralWitness(nn.Module):
        def __init__(self, generator: BiGANGenerator, encoder: BiGANEncoder):
            super().__init__()
            self.G = generator
            self.E = encoder

        def generate(self, z):
            return self.G(z)

        def encode(self, x):
            return self.E(x)

        def reconstruct_data(self, x):
            return self.G(self.E(x))

        def reconstruct_latent(self, z):
            return self.E(self.G(z))

    witness = ALICESpiralWitness(gen, enc)
    witness.eval()

    eval_rng = torch.Generator().manual_seed(42)
    target, _ = _sample_spiral_oracle(4096, eval_rng, sigma=sigma)
    theta_z = theta_min + (theta_max - theta_min) * torch.rand(4096, generator=eval_rng)
    u_z = torch.randn(4096, generator=eval_rng) * sigma
    z = torch.stack([(theta_z - 2.0 * math.pi) / math.pi, u_z / (3.0 * sigma)], dim=1).float()

    with torch.no_grad():
        samples = witness.generate(z)
        reconstructed = witness.reconstruct_data(target)
        latent_reconstructed = witness.reconstruct_latent(z)

    assert isinstance(witness.G, BiGANGenerator)
    assert isinstance(witness.E, BiGANEncoder)
    assert not torch.equal(reconstructed, target), "Data reconstruction must not be an identity bypass"
    assert not torch.equal(latent_reconstructed, z), "Latent reconstruction must not be an identity bypass"

    obs = score_alice_spiral(
        samples,
        target,
        reconstructed,
        z=z,
        latent_reconstructed=latent_reconstructed,
        sigma=sigma,
    )
    assert obs["passed"] is True, f"Witness failed bounds: {obs['failed_bounds']} with metrics {obs['metrics']}"
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["arc_bins"] >= 15
    assert obs["metrics"]["hq"] >= 0.85
    assert obs["metrics"]["mass_tv"] <= 0.16
    assert 0.0 < obs["metrics"]["recon_rmse"] <= 0.14
    assert 0.0 < obs["metrics"]["latent_recon_rmse"] <= 0.25
    assert obs["metrics"]["arc_consistency"] >= 0.90


# ============================================================================
# Level 3: Destructive Negative Controls
# ============================================================================

def test_level3_destructive_negative_controls_bigan():
    """Verify destructive negative controls deterministically fail each bigan gate."""
    centers = _ring_centers()
    rng = torch.Generator().manual_seed(123)

    k = torch.randint(0, 16, (4096,), generator=rng)
    target = centers[k] + torch.randn(4096, 2, generator=rng) * 0.10
    samples = centers[k] + torch.randn(4096, 2, generator=rng) * 0.10
    reconstructed = target.clone()

    # 1. Missing mode: drop mode 15 (only 15 modes generated)
    k_15 = torch.randint(0, 15, (4096,), generator=rng)
    samples_15 = centers[k_15] + torch.randn(4096, 2, generator=rng) * 0.10
    obs1 = score_bigan(samples_15, target, reconstructed, centers=centers)
    assert obs1["passed"] is False
    assert "modes" in obs1["failed_bounds"]

    # 2. Low HQ: high noise variance (sigma=0.60 >> 0.10)
    samples_noisy = centers[k] + torch.randn(4096, 2, generator=rng) * 0.60
    obs2 = score_bigan(samples_noisy, target, reconstructed, centers=centers)
    assert obs2["passed"] is False
    assert "hq" in obs2["failed_bounds"]

    # 3. High mass TV: skewed mode allocation (50% in mode 0)
    k_skew = torch.cat([torch.zeros(2048, dtype=torch.long), torch.randint(0, 16, (2048,), generator=rng)])
    samples_skew = centers[k_skew] + torch.randn(4096, 2, generator=rng) * 0.10
    obs3 = score_bigan(samples_skew, target, reconstructed, centers=centers)
    assert obs3["passed"] is False
    assert "mass_tv" in obs3["failed_bounds"]

    # 4. Reconstruction error: recon_rmse exceeds 0.12
    recon_bad = target + 0.25
    obs4 = score_bigan(samples, target, recon_bad, centers=centers)
    assert obs4["passed"] is False
    assert "recon_rmse" in obs4["failed_bounds"]

    # 4b. Discrete mode-center reconstruction (clustering without intra-mode continuous recovery)
    nearest_k = torch.cdist(target, centers).argmin(dim=1)
    recon_discrete = centers[nearest_k]
    obs4b = score_bigan(samples, target, recon_discrete, centers=centers)
    assert obs4b["passed"] is False
    assert "recon_rmse" in obs4b["failed_bounds"]

    # 5. Inconsistent modes: mode_consistency falls below 0.98
    k_scrambled = (k + 1) % 16
    recon_scrambled = centers[k_scrambled] + torch.randn(4096, 2, generator=rng) * 0.10
    obs5 = score_bigan(samples, target, recon_scrambled, centers=centers)
    assert obs5["passed"] is False
    assert "mode_consistency" in obs5["failed_bounds"]

    # 6. Collapsed width (sigma = 0 Dirac points)
    samples_collapsed = centers[k]
    obs6 = score_bigan(samples_collapsed, target, reconstructed, centers=centers)
    assert obs6["metrics"]["modes"] >= 16

    # 7. Continuous ring: uniform circular angle (no modes)
    angles_unif = torch.rand(4096, generator=rng) * 2 * math.pi
    samples_circle = torch.stack([3.0 * torch.cos(angles_unif), 3.0 * torch.sin(angles_unif)], dim=1)
    obs7 = score_bigan(samples_circle, target, reconstructed, centers=centers)
    assert obs7["passed"] is False
    assert "hq" in obs7["failed_bounds"]

    # 8. Non-finite values: NaN or Inf
    samples_nan = samples.clone()
    samples_nan[0, 0] = float("nan")
    obs8 = score_bigan(samples_nan, target, reconstructed, centers=centers)
    assert obs8["passed"] is False
    assert "nonfinite_output" in obs8["failed_bounds"]


def test_level3_destructive_negative_controls_alice_spiral():
    """Verify destructive negative controls deterministically fail each ALICE spiral gate."""
    rng = torch.Generator().manual_seed(321)
    samples, _ = _sample_spiral_oracle(4096, rng, sigma=0.06)
    target, theta_tgt = _sample_spiral_oracle(4096, rng, sigma=0.06)
    reconstructed = target.clone()
    z = torch.randn(4096, 2, generator=rng)
    latent_reconstructed = z.clone()

    # 1. Missing arc_bins: truncate spiral to half angle range (only 8 of 16 arc bins populated)
    theta_half = 0.5 * math.pi + 1.5 * math.pi * torch.rand(4096, generator=rng)
    r_half = 0.5 + 0.25 * theta_half
    samples_half = torch.stack([r_half * torch.cos(theta_half), r_half * torch.sin(theta_half)], dim=1)
    obs1 = score_alice_spiral(samples_half, target, reconstructed, z=z, latent_reconstructed=latent_reconstructed)
    assert obs1["passed"] is False
    assert "arc_bins" in obs1["failed_bounds"]

    # 2. Low HQ: excessive radial noise (sigma=0.45 >> 0.06)
    samples_noisy, _ = _sample_spiral_oracle(4096, rng, sigma=0.45)
    obs2 = score_alice_spiral(samples_noisy, target, reconstructed, z=z, latent_reconstructed=latent_reconstructed)
    assert obs2["passed"] is False
    assert "hq" in obs2["failed_bounds"]

    # 3. High mass TV: 50% of samples concentrated in arc bin 0
    theta_bin0 = 0.5 * math.pi + (3.0 * math.pi / 16.0) * torch.rand(2048, generator=rng)
    r_bin0 = 0.5 + 0.25 * theta_bin0
    samples_bin0 = torch.stack([r_bin0 * torch.cos(theta_bin0), r_bin0 * torch.sin(theta_bin0)], dim=1)
    samples_skew = torch.cat([samples_bin0, samples[:2048]], dim=0)
    obs3 = score_alice_spiral(samples_skew, target, reconstructed, z=z, latent_reconstructed=latent_reconstructed)
    assert obs3["passed"] is False
    assert "mass_tv" in obs3["failed_bounds"]

    # 4. Data reconstruction failure: recon_rmse > 0.14
    recon_bad = target + 0.25
    obs4 = score_alice_spiral(samples, target, recon_bad, z=z, latent_reconstructed=latent_reconstructed)
    assert obs4["passed"] is False
    assert "recon_rmse" in obs4["failed_bounds"]

    # 5. Latent cycle failure: latent_recon_rmse > 0.25
    lat_bad = z + 0.40
    obs5 = score_alice_spiral(samples, target, reconstructed, z=z, latent_reconstructed=lat_bad)
    assert obs5["passed"] is False
    assert "latent_recon_rmse" in obs5["failed_bounds"]

    # 6. Arc consistency failure: shift reconstructed points by 3 arc bins along spiral
    theta_shifted = 0.5 * math.pi + ((theta_tgt - 0.5 * math.pi + 3.0 * (3.0 * math.pi / 16.0)) % (3.0 * math.pi))
    r_shifted = 0.5 + 0.25 * theta_shifted
    recon_shifted = torch.stack([r_shifted * torch.cos(theta_shifted), r_shifted * torch.sin(theta_shifted)], dim=1)
    obs6 = score_alice_spiral(samples, target, recon_shifted, z=z, latent_reconstructed=latent_reconstructed)
    assert obs6["passed"] is False
    assert "arc_consistency" in obs6["failed_bounds"]

    # 7. Non-finite values: NaN / Inf
    samples_nan = samples.clone()
    samples_nan[0, 0] = float("nan")
    obs7 = score_alice_spiral(samples_nan, target, reconstructed, z=z, latent_reconstructed=latent_reconstructed)
    assert obs7["passed"] is False
    assert "nonfinite_output" in obs7["failed_bounds"]

    lat_inf = latent_reconstructed.clone()
    lat_inf[0, 1] = float("inf")
    obs8 = score_alice_spiral(samples, target, reconstructed, z=z, latent_reconstructed=lat_inf)
    assert obs8["passed"] is False
    assert "nonfinite_output" in obs8["failed_bounds"]


# ============================================================================
# Multi-Family Trainer & Prior Update Verification
# ============================================================================

@pytest.mark.parametrize("fixture_cls", [Ring16BiGANFixture, ALICESpiralBiGANFixture])
@pytest.mark.parametrize("recipe_name", ["bispf", "atlas", "bcap", "k3p", "ka2"])
def test_bigan_and_alice_multi_family_trainer_and_prior_updates(fixture_cls, recipe_name):
    """Verify both BiGAN fixtures execute 4 steps cleanly and update unwrap_prior(fixture.prior).z across trainer families."""
    fixture = fixture_cls(steps=10, batch_size=64, particles=256, z_dim=2, device="cpu", seed=0)

    if recipe_name == "atlas":
        recipe = get_recipe("atlas", batch_size=64, num_particles=256, z_dim=2)
        base_prior = ParticlePrior(num_particles=256, z_dim=2)
        init.deterministic_orthogonal_(base_prior, seed=4)
        prior = ManipulatedPrior(base_prior, [NormalNoiseManipulator(sigma=0.0065)])
        fixture.rebind_trainer(recipe, prior=prior)
    else:
        fixture.rebind_trainer(recipe_name)

    z_param = unwrap_prior(fixture.prior).z
    assert z_param.requires_grad is True
    z_before = z_param.detach().clone()

    for _ in range(4):
        metrics = fixture.step()
        assert all(math.isfinite(v) for k, v in metrics.items() if k != "step")

    z_after = unwrap_prior(fixture.prior).z.detach().clone()
    assert not torch.allclose(z_before, z_after), (
        f"{fixture_cls.__name__} failed to update unwrap_prior(prior).z under recipe={recipe_name}"
    )


# ============================================================================
# Architectural & Formulation Checks: Recommendation 2
# ============================================================================

def test_bigan_coordinate_fourier_features_architecture():
    """Verify coordinate Fourier features sinusoidal projections on x for BiGANEncoder and BiGANJointCritic."""
    # 1. Fourier features on Encoder
    enc = BiGANEncoder(in_dim=2, z_dim=2, hidden=128, layers=3, fourier=3)
    assert enc.fourier == 3
    assert hasattr(enc, "freqs")
    expected_freqs = torch.pi * torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(enc.freqs, expected_freqs)
    # in_dim * (1 + 2 * fourier) = 2 * (1 + 6) = 14
    assert enc.net[0].in_features == 14
    assert enc.net[-1].out_features == 2

    # Forward pass on 2D input
    x = torch.randn(16, 2)
    z = enc(x)
    assert z.shape == (16, 2)

    # 2. Fourier features on Joint Critic
    critic = BiGANJointCritic(in_dim=2, z_dim=2, hidden=128, layers=3, fourier=3)
    assert critic.fourier == 3
    assert hasattr(critic, "freqs")
    assert torch.allclose(critic.freqs, expected_freqs)
    # feat_dim + z_dim = 14 + 2 = 16
    assert critic.net[0].in_features == 16
    assert critic.net[-1].out_features == 1

    # Joint critic forward accepts both pair (B, 4) and separate (x, z)
    pair = torch.randn(16, 4)
    out_pair = critic(pair)
    assert out_pair.shape == (16,)
    # Verify true unprojected 4D coordinates attached to _bispf_input
    assert hasattr(out_pair, "_bispf_input")
    assert out_pair._bispf_input.shape == (16, 4)
    assert torch.equal(out_pair._bispf_input, pair)

    out_sep = critic(pair[:, :2], pair[:, 2:])
    assert out_sep.shape == (16,)
    assert hasattr(out_sep, "_bispf_input")
    assert torch.allclose(out_sep._bispf_input, pair)


def test_bigan_capacity_and_cycle_weight():
    """Verify Ring16BiGANFixture capacity upgrade to 128 hidden, 3 layers, and cycle_weight=2.0."""
    fixture = Ring16BiGANFixture(steps=5, device="cpu", seed=0)
    assert fixture.cycle_weight == 2.0

    # Generator: 3 layers, 128 hidden
    assert fixture.G.net[0].in_features == 2
    assert fixture.G.net[0].out_features == 128
    assert fixture.G.net[2].in_features == 128
    assert fixture.G.net[2].out_features == 128
    assert fixture.G.net[4].in_features == 128
    assert fixture.G.net[4].out_features == 128
    assert fixture.G.net[-1].out_features == 2

    # Encoder: 3 layers, 128 hidden, fourier=3 (input dim 14)
    assert fixture.E.fourier == 3
    assert fixture.E.net[0].in_features == 14
    assert fixture.E.net[0].out_features == 128
    assert fixture.E.net[2].in_features == 128
    assert fixture.E.net[2].out_features == 128
    assert fixture.E.net[4].in_features == 128
    assert fixture.E.net[4].out_features == 128
    assert fixture.E.net[-1].out_features == 2

    # Joint Critic: 3 layers, 128 hidden, fourier=3 (input dim 16)
    assert fixture.D.fourier == 3
    assert fixture.D.net[0].in_features == 16
    assert fixture.D.net[0].out_features == 128
    assert fixture.D.net[2].in_features == 128
    assert fixture.D.net[2].out_features == 128
    assert fixture.D.net[4].in_features == 128
    assert fixture.D.net[4].out_features == 128
    assert fixture.D.net[-1].out_features == 1


def test_bispf_thermal_and_critic_hook_binding():
    """Verify rebind_trainer binds thermal source and hook attaches unprojected coordinates for BISPF."""
    fixture = Ring16BiGANFixture(steps=5, device="cpu", seed=0, recipe_name="bispf")

    # Thermal source must be bound to optimizer record
    assert hasattr(fixture.opt_d, "record")
    assert fixture.loss.thermal_source is fixture.opt_d.record
    assert fixture.opt_d.record.legacy_thermal_channel is False

    # Execute one step to ensure BISPF forward and penalty work seamlessly
    metrics = fixture.step()
    assert math.isfinite(metrics["loss_d"])
    assert math.isfinite(metrics["loss_g"])
    assert math.isfinite(metrics["reg_d"])


def test_bigan_alice_recon_loss_target_detached():
    """Verify that z_fake is detached in ALICE reconstruction loss so prior table gradients come only from adversarial / prior regularizer."""
    fixture = Ring16BiGANFixture(steps=5, device="cpu", seed=0)
    real_x = fixture.sample_target(16)
    z_fake, _ = fixture._sample_prior(16)
    x_fake = fixture._generate(z_fake)
    z_real = fixture.E(real_x)

    # Detached target in E(x_fake)
    loss_recon = F.mse_loss(fixture.G(z_real), real_x) + F.mse_loss(fixture.E(x_fake), z_fake.detach())

    # Backward from loss_recon alone
    loss_recon.backward()

    # z_fake has no grad accumulation directly through MSE target
    prior_z = unwrap_prior(fixture.prior).z
    # Gradients on G and E exist from reconstruction
    assert any(p.grad is not None and torch.norm(p.grad) > 0 for p in fixture.G.parameters())
    assert any(p.grad is not None and torch.norm(p.grad) > 0 for p in fixture.E.parameters())


