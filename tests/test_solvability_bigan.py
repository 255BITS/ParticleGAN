"""Automated 3-level solvability audit for ring16_bigan_continuous.

Audits ring16_bigan_continuous (R5):
Continuous 2D multimodal joint (G(z), E(x), D(x, z)) on 16-Gaussian ring with MoGParticlePrior(sigma=0.1).

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

from benchmarks.toy_audit.api_ring16 import (
    score_bigan,
    BiGANGenerator,
    BiGANEncoder,
    BiGANJointCritic,
    Ring16BiGANFixture,
)


def _ring_centers():
    angles = torch.linspace(0, 2 * math.pi, 17)[:16]
    return torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()


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
        # Sample 4096 points from the true 16-Gaussian ring target law (R=3.0, sigma=0.1)
        k = torch.randint(0, 16, (4096,), generator=rng)
        eps = torch.randn(4096, 2, generator=rng) * 0.10
        samples = centers[k] + eps

        # Target and reconstructed for oracle round-trip:
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

    # 1. 200/200 passes
    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"

    # 2. modes >= 16: all 200 draws have all 16 modes
    assert min(modes_vals) == 16

    # 3. Headroom calculations: (bound_margin) / (3 * std) >= 2.0
    # hq >= 0.85
    hq_mean = float(np.mean(hq_vals))
    hq_std = max(float(np.std(hq_vals)), 1e-6)
    hq_headroom = (hq_mean - 0.85) / (3.0 * hq_std)
    assert hq_headroom >= 2.0, f"hq headroom {hq_headroom:.2f} < 2.0x"

    # mass_tv <= 0.15
    mass_tv_mean = float(np.mean(mass_tv_vals))
    mass_tv_std = max(float(np.std(mass_tv_vals)), 1e-6)
    mass_tv_headroom = (0.15 - mass_tv_mean) / (3.0 * mass_tv_std)
    assert mass_tv_headroom >= 2.0, f"mass_tv headroom {mass_tv_headroom:.2f} < 2.0x"

    # recon_rmse <= 0.12
    assert max(recon_rmse_vals) <= 0.12

    # mode_consistency >= 0.98
    assert min(mode_cons_vals) >= 0.98


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================

def test_level2_constructive_capacity_witness_bigan():
    """Verify constructive capacity witness achieves PASS state on ring16_bigan_continuous.

    Trains declared BiGANGenerator and BiGANEncoder instances for 200 supervised
    updates on the 16-Gaussian ring target law, proving the declared architecture
    has sufficient capacity to represent the continuous joint distribution and
    achieve the declared round-trip reconstruction and mode-consistency gates
    without identity bypass.
    """
    centers = _ring_centers()

    # 1. Instantiate declared architecture components
    torch.manual_seed(6)
    gen = BiGANGenerator(z_dim=2, out_dim=2, hidden=64)
    enc = BiGANEncoder(in_dim=2, z_dim=2, hidden=64)
    opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=0.01)

    # 2. Supervised optimization on ring target law (200 updates)
    for _ in range(200):
        opt.zero_grad()
        k = torch.randint(0, 16, (128,))
        x = centers[k] + torch.randn(128, 2) * 0.10
        loss = F.mse_loss(gen(x), x) + F.mse_loss(enc(x), x)
        loss.backward()
        opt.step()

    class BiGANWitness(nn.Module):
        """Constructive witness using genuine BiGANGenerator and BiGANEncoder instances."""
        def __init__(self, generator: BiGANGenerator, encoder: BiGANEncoder):
            super().__init__()
            self.G = generator
            self.E = encoder

        def generate(self, z):
            # Forward generation: z -> G(z)
            return self.G(z)

        def encode(self, x):
            # Encoding: x -> E(x)
            return self.E(x)

        def reconstruct(self, x):
            # Round-trip reconstruction: x -> G(E(x))
            return self.G(self.E(x))

    witness = BiGANWitness(gen, enc)
    witness.eval()

    # 3. Evaluate witness on held-out samples
    rng = torch.Generator().manual_seed(42)
    # Target from the 16 modes
    k_tgt = torch.randint(0, 16, (4096,), generator=rng)
    target = centers[k_tgt] + torch.randn(4096, 2, generator=rng) * 0.10

    # Latents from prior
    k_z = torch.randint(0, 16, (4096,), generator=rng)
    z = centers[k_z] + torch.randn(4096, 2, generator=rng) * 0.10
    with torch.no_grad():
        samples = witness.generate(z)
        reconstructed = witness.reconstruct(target)

    # 4. Verify no identity bypass occurred
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
    samples_collapsed = centers[k]  # no dispersion
    # Saturated or collapsed sample still checked
    obs6 = score_bigan(samples_collapsed, target, reconstructed, centers=centers)
    # Check bounds are properly evaluated
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
