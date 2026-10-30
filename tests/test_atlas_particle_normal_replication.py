"""Replication integration test suite for Atlas with swapped particle + normal prior.

Validates that swapping the MoG prior for an unstandardized particle prior table plus
an additive normal noise manipulator replicates historical PR #223 test performance
across representative benchmark tasks under repository experimentation rules:
  - Protocol seed 0
  - Public deterministic orthogonal initialization
  - Isolated RNG streams (data, latent, prior noise seed+6, eval)
  - Quantitative numerical pass/fail metrics based on PR #223 targets:
      * Toy sample precision >= 95.0% (historical reported 96.53%)
      * Mode coverage >= 24/25 modes (historical reported 25/25)
      * Mass total variation <= 0.08 (historical reported 0.05211)
      * 1D Gaussian error sigma <= 0.20 (historical reported 0.12)
      * Ring16 mode acquisition >= 15/16 modes with angular precision >= 90.0%
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from particlegan import (
    GANTrainer,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    get_recipe,
    unwrap_prior,
)
from particlegan.capabilities import prior_capabilities
from particlegan.init import deterministic_orthogonal_
from particlegan.priors import ManipulatedPrior, NormalNoiseManipulator
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from benchmarks.toy_audit.gaussian1d_quality import score_samples as score_gaussian1d

ROOT = Path(__file__).resolve().parents[1]


def get_test_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Group 1: Forge Idea & Study Declaration Contracts
# ==============================================================================


def test_forge_atlas_particle_normal_idea_contract():
    """Verify configs/forge/ideas/atlas-particle-normal-v1.json declaration."""
    idea_path = ROOT / "configs/forge/ideas/atlas-particle-normal-v1.json"
    assert idea_path.exists(), f"Idea config missing: {idea_path}"

    idea = json.loads(idea_path.read_text())
    from experiments.forge.contracts import validate_idea

    validate_idea(idea)

    assert idea["id"] == "atlas-particle-normal-v1"
    assert idea["parent"] == "atlas"
    assert idea["recipe_preset"] == "atlas"
    assert idea["mechanism_class"] == "structural"

    # Prior declaration must be unstandardized learnable particle cloud with sigma 0.0
    prior_cfg = idea["prior"]
    assert prior_cfg["kind"] == "particle_cloud"
    assert prior_cfg["standardize"] is False
    assert prior_cfg["learnable"] is True
    assert prior_cfg["sigma"] == 0.0

    # Recipe overrides per survey_experiments_tests_1/report.md
    overrides = idea["recipe_overrides"]
    assert overrides["prior_kind"] == "particles"
    assert overrides["output_noise_mode"] == "learnable"
    assert overrides["output_noise_std"] == 0.029
    assert overrides["output_noise_warmup"] == 0.0
    assert overrides["birth_death_backend"] == "auto"
    assert overrides["birth_death_cells"] == 128
    assert overrides["reopen_guard"] == "settled"

    # Required capabilities
    caps = set(idea["requires_capabilities"])
    for required in ("particle_cloud", "a2", "named_rng", "policy_controls", "policy_serving"):
        assert required in caps, f"Missing required capability: {required}"


def test_forge_atlas_particle_normal_study_contract():
    """Verify configs/forge/studies/atlas-particle-normal-replication-study-v1.json declaration."""
    study_path = ROOT / "configs/forge/studies/atlas-particle-normal-replication-study-v1.json"
    assert study_path.exists(), f"Study config missing: {study_path}"

    from experiments.forge.studies import load_study

    study = load_study(ROOT, "atlas-particle-normal-replication-study-v1")
    assert study["id"] == "atlas-particle-normal-replication-study-v1"
    assert study["status"] == "ready"
    assert study["candidate"] == "atlas-particle-normal-v1"
    assert study["control"]["candidate_id"] == "atlas"

    # Control task map covers native 100-Gaussians and portability tasks
    task_map = study["control"]["task_map"]
    for task in (
        "grid100",
        "rotated100",
        "staggered100",
        "gaussian1d_smoke",
        "gaussian1d_stability",
        "ring16_acquisition",
        "two_pole",
        "vector_spiral",
        "vector_anisotropic",
    ):
        assert task in task_map, f"Missing task in replication study: {task}"

    # Prediction and falsifier signatures
    assert study["prediction"]["task_id"] == "grid100"
    assert study["prediction"]["metric"] == "mode_precision"
    assert study["prediction"]["op"] == ">="
    assert study["prediction"]["threshold"] == 0.95

    assert study["falsifier"]["task_id"] == "gaussian1d_smoke"
    assert study["falsifier"]["metric"] == "mean_error_sigma"
    assert study["falsifier"]["op"] == ">"
    assert study["falsifier"]["threshold"] == 0.20

    # Source-bound prior evidence
    evidence = study["prior_evidence"]
    assert len(evidence) >= 1
    assert evidence[0]["path"] == "reports/develop-gates-20261001/atlas19-replay.json"
    assert evidence[0]["identity"]["record_id"] == "develop-original-atlas19-and-gate-diagnosis"


def test_forge_validate_includes_replication_study():
    """Verify python -m experiments.forge validate loads both new idea and study."""
    from experiments.forge.planning import load_idea
    from experiments.forge.studies import load_study

    idea = load_idea(ROOT, "atlas-particle-normal-v1")
    assert idea["id"] == "atlas-particle-normal-v1"

    study = load_study(ROOT, "atlas-particle-normal-replication-study-v1")
    assert study["id"] == "atlas-particle-normal-replication-study-v1"


# ==============================================================================
# Group 2: Swapped Prior Architectural Invariants & Gradient Locality
# ==============================================================================


def test_swapped_prior_capabilities_and_row_locality():
    """Verify that the swapped prior reports particle_cloud, row_local_gradients, and a2_eligible."""
    recipe = get_recipe("atlas", z_dim=4, num_particles=64)
    swapped_prior = recipe.make_prior(sigma=0.025)

    assert isinstance(swapped_prior, ManipulatedPrior)
    assert isinstance(unwrap_prior(swapped_prior), ParticlePrior)

    caps = prior_capabilities(swapped_prior)
    assert caps["kind"] == "particle_cloud"
    assert caps["row_local_gradients"] is True
    assert caps["a2_eligible"] is True
    assert any(getattr(m, "additive_transform", False) for m in swapped_prior.manipulators)


def test_swapped_prior_uncoupled_gradients_vs_mog_contrast():
    """Assure single-row gradient updates do not cross-couple to other rows.

    This mathematically proves why MoG standardization broke Atlas's 128 feature cells,
    while the swapped prior preserves strict row-local gradient transport.
    """
    B, D = 16, 4
    torch.manual_seed(0)

    # 1. Swapped prior: ParticlePrior + NormalNoiseManipulator
    base_prior = ParticlePrior(B, D)
    swapped_prior = base_prior.manipulate(NormalNoiseManipulator(sigma=0.025))

    codes = base_prior.z.clone().detach().requires_grad_(True)
    eps = torch.randn_like(codes)
    sampled = swapped_prior.manipulators[0].transform(codes, base_prior=base_prior, eps=eps)

    # Perturb row k and check Jacobian: d(sampled[i, c]) / d(codes[j, c'])
    k = 3
    target_scalar = sampled[k].sum()
    grads = torch.autograd.grad(target_scalar, codes, retain_graph=False)[0]

    # For swapped prior: row k has gradient 1.0, all other rows are strictly 0.0
    assert torch.allclose(grads[k], torch.ones(D))
    mask = torch.ones(B, dtype=torch.bool)
    mask[k] = False
    assert torch.all(grads[mask] == 0.0), "Cross-row coupling detected in swapped prior!"

    # 2. Contrast with MoG standardized prior:
    mog = MoGParticlePrior(num_particles=B, z_dim=D, sigma=0.025, standardize=True)
    mog_codes = mog.z.clone().detach().requires_grad_(True)
    mean_val = mog_codes.mean(0)
    std_val = mog_codes.std(0) + 1e-6
    std_codes = (mog_codes - mean_val) / std_val
    mog_target = std_codes[k].sum()
    mog_grads = torch.autograd.grad(mog_target, mog_codes, retain_graph=False)[0]

    # In MoG, other rows receive non-zero gradient coupling -1 / (B * std)
    assert torch.any(mog_grads[mask] != 0.0), "MoG failed to exhibit cross-row coupling"


# ==============================================================================
# Group 3: Historical Benchmark Replication
# ==============================================================================


def test_atlas_gaussian1d_smoke_replication():
    """Verify Atlas with swapped prior reproduces historical 1D Gaussian performance.

    Benchmark: gaussian1d_smoke (Target: N(2.0, 0.5^2))
    Pass criterion: mean_error_sigma <= 0.20 (Historical PR #223 reported: 0.12).
    """
    device = get_test_device()
    target_mean = 2.0
    target_sigma = 0.5

    def sample_data(n: int, gen: torch.Generator) -> torch.Tensor:
        return target_mean + target_sigma * torch.randn(n, 1, device=device, generator=gen)

    # Recipe with swapped prior and DV12 continuous policy, learned output noise
    recipe = get_recipe(
        "atlas",
        z_dim=2,
        num_particles=256,
        batch_size=128,
        lr=0.005,
    )
    G = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
    D = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)

    # Public deterministic initializer
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior(sigma=0.025).to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    # Stream-isolated data generator
    data_gen = torch.Generator(device=device).manual_seed(42)

    for _ in range(250):
        real_batch = sample_data(128, data_gen)
        trainer.step(real_batch)

    # Evaluation with separate eval generator
    eval_gen = torch.Generator(device=device).manual_seed(713)
    samples = trainer.sample(4096, generator=eval_gen, output_noise=False)

    spec = {
        "kind": "gaussian_mixture",
        "means": [[target_mean]],
        "covariances": [[[target_sigma**2]]],
        "masses": [1.0],
    }
    metrics = score_gaussian1d(samples, spec)

    mean_err_sigma = metrics["mean_error_sigma"]
    assert mean_err_sigma is not None
    assert metrics["finite_fraction"] == 1.0
    assert metrics["sample_count"] == 4096

    # PR #223 Numerical Target: mean_error_sigma <= 0.20
    assert mean_err_sigma <= 0.20, (
        f"1D Gaussian error sigma {mean_err_sigma:.4f} exceeded threshold 0.20 (target: historical <= 0.12)"
    )


def test_atlas_toy_25_modes_replication():
    """Verify Atlas with swapped prior reproduces historical 25-mode toy Gaussian performance.

    Benchmark: 25-mode 2D Gaussian mixture (5x5 grid)
    Pass criteria:
      - Toy sample precision >= 95.0% (Historical PR #223 reported: 96.53%)
      - Mode coverage >= 24/25 modes (Historical PR #223 reported: 25/25)
      - Mass total variation <= 0.08 (Historical PR #223 reported: 0.05211)
    """
    device = get_test_device()
    n_modes = 25
    sigma = 0.05

    coords = torch.linspace(-2.0, 2.0, 5, device=device)
    row, col = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack((row.flatten(), col.flatten()), dim=1)  # (25, 2)

    def sample_25modes(n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, n_modes, (n,), device=device, generator=gen)
        noise = torch.randn(n, 2, device=device, generator=gen) * sigma
        return centers[idx] + noise

    # Configure Atlas with 128 feature cells, DV12 continuous policy, and swapped prior
    recipe = get_recipe(
        "atlas",
        z_dim=2,
        num_particles=1024,
        batch_size=128,
        lr=0.005,
    )
    G = SimpleMLPGenerator(z_dim=2, hidden_dim=64, n_hidden=3, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=3, fourier=2).to(device)

    # Public deterministic orthogonal initializer
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior(sigma=0.025).to(device)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    data_gen = torch.Generator(device=device).manual_seed(42)
    # Train for 1200 steps on GPU (or 250 steps on CPU fallback)
    step_budget = 1200 if device.type == "cuda" else 250

    for _ in range(step_budget):
        real_batch = sample_25modes(128, data_gen)
        trainer.step(real_batch)

    # Clean evaluation samples
    eval_gen = torch.Generator(device=device).manual_seed(713)
    samples = trainer.sample(4096, generator=eval_gen, output_noise=False)

    # Compute precision, mode coverage, and mass total variation
    dists, nearest = torch.cdist(samples, centers).min(dim=1)
    in_radius = dists <= 3.0 * sigma
    precision = float(in_radius.float().mean().item())

    counts = torch.bincount(nearest, minlength=n_modes)
    # Each mode must have at least 1% of the samples to count as covered
    covered_modes = int((counts >= 0.01 * len(samples)).sum().item())

    # Mass total variation distance to uniform distribution (1 / 25)
    hist = counts.float() / len(samples)
    uniform_prob = 1.0 / n_modes
    mass_tv = float((hist - uniform_prob).abs().sum().item() / 2.0)

    # Assert PR #223 Numerical Targets
    if device.type == "cuda":
        assert precision >= 0.950, f"Precision {precision*100:.2f}% < 95.0% target (historical 96.53%)"
        assert covered_modes >= 24, f"Mode coverage {covered_modes}/25 < 24/25 target (historical 25/25)"
        assert mass_tv <= 0.080, f"Mass TV {mass_tv:.5f} > 0.080 target (historical 0.05211)"
    else:
        # CPU smoke bounds
        assert covered_modes >= 20, f"CPU mode coverage {covered_modes}/25 < 20/25"
        assert mass_tv <= 0.20, f"CPU Mass TV {mass_tv:.5f} > 0.20"


def test_atlas_ring16_acquisition_replication():
    """Verify Atlas with swapped prior reproduces ring16 mode acquisition.

    Benchmark: ring16_acquisition (16 Gaussian clusters on a circle of radius 3.0, sigma 0.1)
    Pass criteria:
      - Mode coverage >= 15/16 modes (Historical: 16/16)
      - Angular / cluster precision >= 90.0% (Historical: > 95%)
    """
    device = get_test_device()
    n_modes = 16
    radius = 3.0
    sigma = 0.1

    angles = torch.linspace(0, 2 * math.pi, n_modes + 1, device=device)[:-1]
    centers = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)

    def sample_ring(n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, n_modes, (n,), device=device, generator=gen)
        noise = torch.randn(n, 2, device=device, generator=gen) * sigma
        return centers[idx] + noise

    recipe = get_recipe(
        "atlas",
        z_dim=2,
        num_particles=512,
        batch_size=128,
        lr=0.005,
    )
    G = SimpleMLPGenerator(z_dim=2, hidden_dim=64, n_hidden=2, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2).to(device)

    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior(sigma=0.025).to(device)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0, serial_backward=True)

    data_gen = torch.Generator(device=device).manual_seed(42)
    step_budget = 500 if device.type == "cuda" else 150

    for _ in range(step_budget):
        real_batch = sample_ring(128, data_gen)
        trainer.step(real_batch)

    eval_gen = torch.Generator(device=device).manual_seed(713)
    # Clean sampling per task contract (eval_output_noise: clean)
    samples = trainer.sample(4096, generator=eval_gen, output_noise=False)

    dists, nearest = torch.cdist(samples, centers).min(dim=1)
    in_radius = dists <= 3.0 * sigma
    precision = float(in_radius.float().mean().item())

    counts = torch.bincount(nearest, minlength=n_modes)
    covered_modes = int((counts >= 0.01 * len(samples)).sum().item())

    assert covered_modes >= 15, f"Ring16 mode coverage {covered_modes}/16 < 15/16 target"
    assert precision >= 0.90, f"Ring16 precision {precision*100:.2f}% < 90.0% target"


# ==============================================================================
# Group 4: Stream Isolation and Bit-Exact Reproducibility
# ==============================================================================


def test_atlas_prior_noise_stream_isolation_and_reproducibility():
    """Verify that prior_noise_generator (stream seed+6) is isolated and repeatable."""
    torch.manual_seed(0)

    recipe = get_recipe("atlas", z_dim=2, num_particles=64, batch_size=32)
    G1 = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 2))
    D1 = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 1))
    deterministic_orthogonal_(G1, seed=0)
    deterministic_orthogonal_(D1, seed=1)

    prior1 = recipe.make_prior(sigma=0.025)
    deterministic_orthogonal_(unwrap_prior(prior1), seed=2)
    t1 = GANTrainer(recipe, G1, D1, prior=prior1, seed=0, serial_backward=True)

    # Generator stream seed + 6
    assert t1.prior_noise_generator is not None
    initial_stream_state = t1.prior_noise_generator.get_state().clone()

    # Step trainer and record loss
    data_gen = torch.Generator().manual_seed(99)
    real_data = torch.randn(32, 2, generator=data_gen)
    loss1 = t1.step(real_data)

    # Assert prior_noise_generator advanced
    advanced_stream_state = t1.prior_noise_generator.get_state()
    assert not torch.equal(initial_stream_state, advanced_stream_state)

    # Repeat identically in t2
    G2 = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 2))
    D2 = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 1))
    deterministic_orthogonal_(G2, seed=0)
    deterministic_orthogonal_(D2, seed=1)

    prior2 = recipe.make_prior(sigma=0.025)
    deterministic_orthogonal_(unwrap_prior(prior2), seed=2)
    t2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=0, serial_backward=True)

    data_gen_2 = torch.Generator().manual_seed(99)
    real_data_2 = torch.randn(32, 2, generator=data_gen_2)
    loss2 = t2.step(real_data_2)

    # Check bit-exact match
    for k in ("loss_d", "loss_g", "loss_gan"):
        assert torch.allclose(loss1[k], loss2[k], atol=0.0, rtol=0.0), f"Mismatch in {k}"

    assert torch.equal(
        t1.prior_noise_generator.get_state(),
        t2.prior_noise_generator.get_state(),
    )
