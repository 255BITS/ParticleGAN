"""Automated 3-level solvability audit and multi-family execution suite for cgan_overlapping_classes.

Audits api-cgan-overlapping-classes (cgan_overlapping_classes):
Conditional GAN with Projection Discriminator D(x, y) = psi(phi(x)) + <phi(x), V y>
(Miyato & Koyama, ICLR 2018) and UCD (particlegan/conditioning.py) on 4 overlapping
2D Gaussian classes (centers at (+-0.50, +-0.50), sigma=0.35), testing within-class
variance preservation and overlap density against AC-GAN boundary repulsion
(Odena, Olah & Shlens, ICML 2017).

Levels:
- Level 1: Oracle 200/200 Pass with >= 2.0x noise headroom (margin / (3 * std) >= 2.0).
- Level 2: Constructive nn.Module Capacity Witness achieving PASS state without identity bypass.
- Level 3: Destructive Negative Controls deterministically FAILING each individual threshold
  (including AC-GAN boundary-repulsion / variance-truncation control and non-finite NaN/Inf).
- Multi-family execution test: 4 steps of bispf, atlas (with ManipulatedPrior), bcap, k3p, ka2.
"""
from __future__ import annotations

import json
from pathlib import Path
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
    unwrap_prior,
)
from benchmarks.toy_audit.api_cgan_overlap import (
    CASE_ID,
    SIGMA,
    CGANOverlapFixture,
    ConditionalGenerator,
    ProjectionConditionalDiscriminator,
    build_case,
    class_centers,
    score_cgan_overlap,
)


# ============================================================================
# Level 1: Oracle 200/200 Pass & >= 2.0x Noise Headroom
# ============================================================================

def test_level1_oracle_cgan_overlap_200_draws():
    """Verify Level 1 Oracle passes 200/200 times with >= 2.0x noise headroom across all metrics."""
    centers = class_centers(device="cpu", dtype=torch.float32)
    n_draws = 200
    n_samples = 4096

    passes = 0
    covered_vals = []
    mass_tv_vals = []
    center_err_vals = []
    min_eig_vals = []
    max_eig_vals = []
    overlap_vals = []

    labels = torch.arange(n_samples, dtype=torch.long) % 4

    for seed in range(n_draws):
        rng = torch.Generator().manual_seed(seed + 1000)
        samples = centers[labels] + SIGMA * torch.randn(n_samples, 2, generator=rng)
        target = centers[labels] + SIGMA * torch.randn(n_samples, 2, generator=rng)

        obs = score_cgan_overlap(samples, labels, target=target, target_labels=labels, centers=centers, sigma=SIGMA)
        if obs["passed"]:
            passes += 1

        m = obs["metrics"]
        covered_vals.append(m["classes_covered"])
        mass_tv_vals.append(m["class_mass_tv"])
        center_err_vals.append(m["mean_center_err"])
        min_eig_vals.append(m["min_cov_eig_ratio"])
        max_eig_vals.append(m["max_cov_eig_ratio"])
        overlap_vals.append(m["overlap_density_ratio"])

    # 1. 200/200 passes
    assert passes == n_draws == 200, f"Oracle failed: {passes}/200 passed"

    # 2. classes_covered == 4 on all 200 draws
    assert min(covered_vals) == max(covered_vals) == 4

    # 3. Noise headroom checks: margin / (3 * std) >= 2.0
    # class_mass_tv <= 0.08
    tv_mean = float(np.mean(mass_tv_vals))
    tv_std = max(float(np.std(mass_tv_vals)), 1e-6)
    tv_headroom = (0.08 - tv_mean) / (3.0 * tv_std)
    assert tv_headroom >= 2.0, f"class_mass_tv headroom {tv_headroom:.2f} < 2.0x"

    # mean_center_err <= 0.12
    err_mean = float(np.mean(center_err_vals))
    err_std = max(float(np.std(center_err_vals)), 1e-6)
    err_headroom = (0.12 - err_mean) / (3.0 * err_std)
    assert err_headroom >= 2.0, f"mean_center_err headroom {err_headroom:.2f} < 2.0x"

    # min_cov_eig_ratio >= 0.60
    min_eig_mean = float(np.mean(min_eig_vals))
    min_eig_std = max(float(np.std(min_eig_vals)), 1e-6)
    min_eig_headroom = (min_eig_mean - 0.60) / (3.0 * min_eig_std)
    assert min_eig_headroom >= 2.0, f"min_cov_eig_ratio headroom {min_eig_headroom:.2f} < 2.0x"

    # max_cov_eig_ratio <= 1.50
    max_eig_mean = float(np.mean(max_eig_vals))
    max_eig_std = max(float(np.std(max_eig_vals)), 1e-6)
    max_eig_headroom = (1.50 - max_eig_mean) / (3.0 * max_eig_std)
    assert max_eig_headroom >= 2.0, f"max_cov_eig_ratio headroom {max_eig_headroom:.2f} < 2.0x"

    # overlap_density_ratio >= 0.55
    ov_mean = float(np.mean(overlap_vals))
    ov_std = max(float(np.std(overlap_vals)), 1e-6)
    ov_headroom = (ov_mean - 0.55) / (3.0 * ov_std)
    assert ov_headroom >= 2.0, f"overlap_density_ratio headroom {ov_headroom:.2f} < 2.0x"


# ============================================================================
# Level 2: Constructive Capacity Witness
# ============================================================================

def test_level2_constructive_capacity_witness_cgan_overlap():
    """Verify constructive nn.Module capacity witness passes all thresholds without identity bypass."""
    centers = class_centers(device="cpu", dtype=torch.float32)

    torch.manual_seed(7)
    gen = ConditionalGenerator(z_dim=2, num_classes=4, out_dim=2, hidden=128)
    disc = ProjectionConditionalDiscriminator(x_dim=2, num_classes=4, hidden=128)
    opt = torch.optim.Adam(gen.parameters(), lr=0.008)

    train_rng = torch.Generator().manual_seed(77)
    for _ in range(300):
        opt.zero_grad()
        y = torch.randint(0, 4, (256,), generator=train_rng)
        z = torch.randn(256, 2, generator=train_rng)
        y_oh = F.one_hot(y, num_classes=4).float()
        target_x = centers[y] + SIGMA * z
        pred_x = gen(z, y_oh)
        loss = F.mse_loss(pred_x, target_x)
        loss.backward()
        opt.step()

    class CGANWitness(nn.Module):
        def __init__(self, generator: ConditionalGenerator, discriminator: ProjectionConditionalDiscriminator):
            super().__init__()
            self.G = generator
            self.D = discriminator

        def sample(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y_oh = F.one_hot(y, num_classes=4).float()
            return self.G(z, y_oh)

    witness = CGANWitness(gen, disc)
    witness.eval()

    eval_rng = torch.Generator().manual_seed(42)
    labels = torch.arange(4096, dtype=torch.long) % 4
    z_eval = torch.randn(4096, 2, generator=eval_rng)
    exact_reference = centers[labels] + SIGMA * z_eval
    with torch.no_grad():
        samples = witness.sample(z_eval, labels)
        scores = witness.D(samples, labels)

    assert isinstance(witness.G, ConditionalGenerator)
    assert isinstance(witness.D, ProjectionConditionalDiscriminator)
    assert scores.shape == (4096,)
    assert not torch.equal(samples, z_eval), "Witness must not be an identity bypass on z"
    assert not torch.equal(samples, exact_reference), "Witness must be a learned neural map, not an oracle copy"

    obs = score_cgan_overlap(samples, labels, target=exact_reference, target_labels=labels, centers=centers, sigma=SIGMA)
    assert obs["passed"] is True, f"Constructive witness failed bounds: {obs['failed_bounds']}, metrics={obs['metrics']}"
    assert len(obs["failed_bounds"]) == 0
    assert obs["metrics"]["classes_covered"] == 4
    assert obs["metrics"]["class_mass_tv"] <= 0.08
    assert obs["metrics"]["mean_center_err"] <= 0.12
    assert 0.60 <= obs["metrics"]["min_cov_eig_ratio"] <= obs["metrics"]["max_cov_eig_ratio"] <= 1.50
    assert obs["metrics"]["overlap_density_ratio"] >= 0.55


# ============================================================================
# Level 3: Destructive Negative Controls
# ============================================================================

def test_level3_destructive_negative_controls_cgan_overlap():
    """Verify destructive negative controls deterministically fail each individual threshold."""
    centers = class_centers(device="cpu", dtype=torch.float32)
    rng = torch.Generator().manual_seed(123)
    n = 4096
    labels = torch.arange(n, dtype=torch.long) % 4
    clean_samples = centers[labels] + SIGMA * torch.randn(n, 2, generator=rng)

    # 1. Missing class coverage (class 3 collapsed far outside its 3.5*sigma support)
    samples_missing_class = clean_samples.clone()
    samples_missing_class[labels == 3] = torch.tensor([-3.0, -3.0])
    obs_missing = score_cgan_overlap(samples_missing_class, labels, centers=centers, sigma=SIGMA)
    assert obs_missing["passed"] is False
    assert "classes_covered" in obs_missing["failed_bounds"]

    # 2. Skewed class mass (class 0 receives 50% of labels -> TV = 0.25 > 0.08)
    skewed_labels = torch.cat([
        torch.zeros(2048, dtype=torch.long),
        torch.ones(682, dtype=torch.long),
        torch.full((683,), 2, dtype=torch.long),
        torch.full((683,), 3, dtype=torch.long),
    ])
    samples_skewed = centers[skewed_labels] + SIGMA * torch.randn(n, 2, generator=rng)
    obs_skew = score_cgan_overlap(samples_skewed, skewed_labels, centers=centers, sigma=SIGMA)
    assert obs_skew["passed"] is False
    assert "class_mass_tv" in obs_skew["failed_bounds"]

    # 3. Mean center error failure (all class centers displaced by +0.25)
    samples_shifted = clean_samples + 0.25
    obs_shifted = score_cgan_overlap(samples_shifted, labels, centers=centers, sigma=SIGMA)
    assert obs_shifted["passed"] is False
    assert "mean_center_err" in obs_shifted["failed_bounds"]

    # 4. AC-GAN boundary-repulsion & variance-truncation control (Odena et al. 2017 / Miyato & Koyama 2018):
    # Samples are pushed outward away from the decision boundaries x=0, y=0 to (+-0.85, +-0.85)
    # with truncated within-class std (0.12 << 0.35), hollowing out [-0.35, 0.35]^2.
    repelled_centers = centers * 1.70  # (+-0.85, +-0.85)
    samples_acgan = repelled_centers[labels] + 0.12 * torch.randn(n, 2, generator=rng)
    obs_acgan = score_cgan_overlap(samples_acgan, labels, centers=centers, sigma=SIGMA)
    assert obs_acgan["passed"] is False
    assert "min_cov_eig_ratio" in obs_acgan["failed_bounds"]
    assert "overlap_density_ratio" in obs_acgan["failed_bounds"]

    # 5. Pure variance truncation at true centers (sigma_trunc = 0.20 -> eig_ratio ~ 0.326 < 0.60)
    samples_trunc = centers[labels] + 0.20 * torch.randn(n, 2, generator=rng)
    obs_trunc = score_cgan_overlap(samples_trunc, labels, centers=centers, sigma=SIGMA)
    assert obs_trunc["passed"] is False
    assert "min_cov_eig_ratio" in obs_trunc["failed_bounds"]

    # 6. Variance explosion (sigma_wide = 0.50 -> eig_ratio ~ 2.04 > 1.50)
    samples_wide = centers[labels] + 0.50 * torch.randn(n, 2, generator=rng)
    obs_wide = score_cgan_overlap(samples_wide, labels, centers=centers, sigma=SIGMA)
    assert obs_wide["passed"] is False
    assert "max_cov_eig_ratio" in obs_wide["failed_bounds"]

    # 7. Hollowed-out central overlap box [-0.35, 0.35]^2 without collapsing outer variance
    samples_hollow = clean_samples.clone()
    in_box = (samples_hollow[:, 0].abs() <= 0.35) & (samples_hollow[:, 1].abs() <= 0.35)
    samples_hollow[in_box] = centers[labels[in_box]] * 1.5
    obs_hollow = score_cgan_overlap(samples_hollow, labels, centers=centers, sigma=SIGMA)
    assert obs_hollow["passed"] is False
    assert "overlap_density_ratio" in obs_hollow["failed_bounds"]

    # 8. Non-finite values (NaN and Inf)
    samples_nan = clean_samples.clone()
    samples_nan[0, 0] = float("nan")
    obs_nan = score_cgan_overlap(samples_nan, labels, centers=centers, sigma=SIGMA)
    assert obs_nan["passed"] is False
    assert "nonfinite_output" in obs_nan["failed_bounds"]

    samples_inf = clean_samples.clone()
    samples_inf[0, 1] = float("inf")
    obs_inf = score_cgan_overlap(samples_inf, labels, centers=centers, sigma=SIGMA)
    assert obs_inf["passed"] is False
    assert "nonfinite_output" in obs_inf["failed_bounds"]


# ============================================================================
# Multi-Family Trainer Execution & Task Configuration Tests
# ============================================================================

@pytest.mark.parametrize("family", ["bispf", "bispf_optimized", "atlas", "e22", "bcap", "bcap_adam", "k3p", "ka2"])
def test_multi_family_execution_updates_prior(family: str):
    """Verify 4 steps of all 8 families (including bispf and atlas with ManipulatedPrior) run cleanly and update z."""
    if family in ("atlas", "e22"):
        base_prior = ParticlePrior(num_particles=256, z_dim=2, init_std=0.5)
        prior = ManipulatedPrior(base_prior, [NormalNoiseManipulator(sigma=0.0065)])
    else:
        prior = None

    fixture = CGANOverlapFixture(recipe_name=family, prior=prior, device="cpu", seed=0, max_steps=10)
    z_before = unwrap_prior(fixture.prior).z.detach().clone()

    for step_idx in range(1, 5):
        out = fixture.step()
        assert out["step"] == step_idx
        assert all(np.isfinite(v) for v in out.values())

    z_after = unwrap_prior(fixture.prior).z.detach().clone()
    assert not torch.equal(z_before, z_after), f"Family {family} did not update unwrap_prior(fixture.prior).z"

    obs = fixture.observe(n=512, seed=100)
    assert isinstance(obs["passed"], bool)
    assert len(obs["views"]) == 3


def test_rebind_trainer_and_ucd_conditioning():
    """Verify rebind_trainer switches recipes and UCD conditioning loss executes cleanly."""
    fixture = build_case(CASE_ID, device="cpu", seed=0, recipe_name="bispf", max_steps=10)
    ucd_recipe = get_recipe(
        "bispf",
        batch_size=256,
        total_steps=3000,
        z_dim=2,
        num_particles=256,
        conditioning="ucd",
        num_classes=4,
        ucd_weight=0.05,
        input_noise_std=0.0,
        output_noise_std=0.0,
    )
    fixture.rebind_trainer(ucd_recipe)
    z0 = unwrap_prior(fixture.prior).z.detach().clone()
    for _ in range(2):
        stats = fixture.step()
        assert np.isfinite(stats["loss_d"]) and np.isfinite(stats["loss_g"])
    assert not torch.equal(z0, unwrap_prior(fixture.prior).z.detach())

    state = fixture.state_dict()
    restored = build_case(CASE_ID, device="cpu", seed=0, recipe=ucd_recipe, max_steps=10)
    restored.load_state_dict(state)
    assert restored.completed_steps == 2


def test_forge_task_json_contract():
    """Verify configs/forge/tasks/cgan_overlapping_classes.json matches the declared contract."""
    task_path = Path(__file__).resolve().parents[1] / "configs/forge/tasks/cgan_overlapping_classes.json"
    assert task_path.is_file()
    task = json.loads(task_path.read_text())
    assert task["id"] == "cgan_overlapping_classes"
    assert task["adapter"] == "transfer_vector"
    assert task["execution"]["prior_contract"] == "recipe_owned_v1"
    assert task["execution"]["steps"] == 3000
    assert task["evaluation"]["eval_samples"] == 4096
    expected_thresholds = [
        ["classes_covered", "==", 4],
        ["class_mass_tv", "<=", 0.08],
        ["mean_center_err", "<=", 0.12],
        ["min_cov_eig_ratio", ">=", 0.60],
        ["max_cov_eig_ratio", "<=", 1.50],
        ["overlap_density_ratio", ">=", 0.55],
    ]
    assert task["evaluation"]["thresholds"] == expected_thresholds
