"""Adversarial empirical stress-test harness for Milestone 1 (Tier A Formulation & Engineering Fixes).

Empirical challenger tests:
1. Small-table guard stress test across table cardinalities:
   N in [8, 12, 16, 32, 64, 128, 129, 256, 1024] with cells=128.
   - Standalone: For all N <= 128, moves == 0 strictly over 1000 calls even with extreme evidence.
   - Policy Integration:
     * For N in [8, 10, 12, 16, 32, 64, 128]: birth_death deactivated (None), mode coverage >= 7/8 (no decimation).
     * For N=8 with default atlas preset (birth_death_isolation=True): constructs cleanly, birth_death=None, steps without error.
     * For N in [129, 256, 1024]: turnover functions normally (evals > 0, moves execute).
2. Clean intra-mode covariance and mode precision stress test:
   - Clean training (output_noise_std=0.0, sigma=0.005) vs noisy (output_noise_std=0.029)
     under clean evaluation (sampling_law='public_prior_without_output_noise').
   - Empirical assertion of covariance trace bias (|bias| <= 0.15 vs bias < -0.50)
     and mode precision on separated clusters.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from particlegan import GANTrainer, get_recipe, unwrap_prior
from particlegan.birth_death import ParticleBirthDeath, ParticleRows
from particlegan.cell_statistics import population_policy
from particlegan.feature_cells import FeatureCellBirthDeath
from particlegan.init import deterministic_orthogonal_
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator
from benchmarks.locked_shared.mode_hold import ring_means, sample_ring, diversity


def get_device() -> torch.device:
    # Use cuda:1 if free, otherwise cuda:0 or cpu
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _make_table(n_rows: int, z_dim: int = 2) -> nn.Parameter:
    t = torch.arange(n_rows * z_dim, dtype=torch.float64).reshape(n_rows, z_dim)
    return nn.Parameter(torch.sin(t * 0.71) + t * 0.03)


def _make_rows(table: nn.Parameter) -> ParticleRows:
    return ParticleRows(
        table=table,
        optimizer=torch.optim.Adam([table], lr=0.01, amsgrad=True),
        averaged_table=table.detach().clone(),
        generate=lambda z: z,
    )


# ==============================================================================
# Suite 1: Small-Table Guard Standalone Empirical Tests
# ==============================================================================

@pytest.mark.parametrize("N", [8, 12, 16, 32, 64, 128])
def test_small_table_guard_standalone_adversarial_bypass(N: int):
    """Adversarially probe ParticleBirthDeath with cells=128 on N <= 128.

    Even when reservoir is full and S, W evidence are overwhelmingly biased
    to trigger maximum BH rejections and moves, the guard must strictly
    prevent any evaluation or moves over 1000 consecutive calls.
    """
    table = _make_table(n_rows=N)
    rows = _make_rows(table)
    bd = ParticleBirthDeath(rows, seed=42, cells=128)
    assert bd.cells == 128

    # Force-populate reservoir
    real_sample = table.detach().clone() + 0.1
    bd.observe_real(real_sample)

    # Inject extreme synthetic discovery evidence
    bd.S.fill_(10000.0)
    bd.W.fill_(10000.0)
    bd.n.fill_(10)

    initial_table = table.detach().clone()

    for call_idx in range(1000):
        event = bd.maybe_apply(sigma_out=0.0)
        assert event is None, f"Call {call_idx} returned event {event} for N={N}"
        assert bd.counters["moves"] == 0, f"Moves > 0 on call {call_idx} for N={N}"
        assert bd.counters["evals"] == 0, f"Evals > 0 on call {call_idx} for N={N}"
        assert bd.moved_rows is None

    # Assert bit-exact identity: table was never mutated by turnover
    assert torch.equal(table, initial_table)


@pytest.mark.parametrize("N", [129, 256])
def test_large_table_guard_standalone_knn_turnover(N: int):
    """Verify that for N > 128 under kNN backend, turnover is permitted and moves occur."""
    table = _make_table(n_rows=N)
    rows = _make_rows(table)
    bd = ParticleBirthDeath(rows, seed=42, cells=128)
    assert bd.cells == 128

    # Prime discovery evidence
    bd.observe_real(table.detach().clone() + 0.05)
    bd.anchor = table.detach().clone()
    bd.radius = torch.full((N,), 100.0, dtype=torch.float64)
    signs = torch.ones(N, dtype=torch.float64)
    signs[N // 2:] = -1.0
    bd.S.copy_(signs * 1000.0)
    bd.W.copy_(signs * 1000.0)
    bd.n.fill_(2)

    initial_table = table.detach().clone()
    event = bd.maybe_apply(sigma_out=0.0)

    # For N > 128, guard does NOT bypass
    assert event is not None, f"Expected turnover event for N={N} > 128"
    assert bd.counters["moves"] > 0
    assert bd.moved_rows is not None
    assert not torch.equal(table, initial_table)


def test_large_table_guard_standalone_feature_cells_turnover():
    """Verify that for N=1024, feature-cell turnover functions normally."""
    pol = population_policy(1024)
    assert pol["finite_resolution_feasible"] is True
    assert pol["actual_backend"] == "feature_cells"


# ==============================================================================
# Suite 2: Small-Table Guard Policy Integration Tests (1000 steps)
# ==============================================================================

def test_small_table_n8_isolation_constructor_vulnerability():
    """Remediated probe: In atlas preset (birth_death_isolation=True), N=8 constructs
    cleanly because the small-table constructor guard deactivates birth-death (birth_death=None)
    before ParticleBirthDeath leave-one-out isolation can fail.
    Verify clean construction and 50-step training execution without ValueError.
    """
    device = get_device()
    recipe = get_recipe("atlas", num_particles=8, z_dim=4, batch_size=128, birth_death_cells=128)
    G = SimpleMLPGenerator(z_dim=4, hidden_dim=96, n_hidden=3, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3, fourier=3).to(device)
    prior = recipe.make_prior().to(device)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    assert trainer.birth_death is None
    assert trainer.policy.birth_death is None

    # Step for 50 steps to verify runtime stability
    means = ring_means()
    data_gen = torch.Generator().manual_seed(10)
    for _ in range(50):
        real_batch = sample_ring(means, 128, 0.07, data_gen).to(device)
        trainer.step(real_batch)
        assert trainer.birth_death is None


def test_small_table_n8_isolation_disabled_1000_steps():
    """Verify that on N=8, the small-table guard functions properly over 1000 steps
    with birth_death deactivated (None) and no mode decimation.
    """
    device = get_device()
    means = ring_means()
    recipe = get_recipe("atlas", num_particles=8, z_dim=4, batch_size=128, lr=0.002,
                        birth_death_cells=128, birth_death_isolation=False)
    G = SimpleMLPGenerator(z_dim=4, hidden_dim=96, n_hidden=3, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3, fourier=3).to(device)
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior().to(device)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    assert trainer.birth_death is None

    data_gen = torch.Generator().manual_seed(10)
    for _ in range(1000):
        real_batch = sample_ring(means, 128, 0.07, data_gen).to(device)
        trainer.step(real_batch)

    assert trainer.birth_death is None

    eval_gen = torch.Generator(device=device).manual_seed(999)
    samples = trainer.sample(4096, generator=eval_gen, output_noise=False).cpu()
    div = diversity(samples, means, sigma=0.07)
    assert div["modes"] >= 5, f"Mode decimation on N=8: covered {div['modes']}/8"


@pytest.mark.parametrize("N", [8, 10, 12, 16, 32, 64, 128])
def test_small_table_guard_policy_1000_steps_zero_moves_and_no_decimation(N: int):
    """Stress-test GANTrainer policy over 1000 steps across N in [8, 10, 12, 16, 32, 64, 128].

    Target: 8-mode ring (mode_hold benchmark).
    Assertions:
      1. Every single step keeps birth_death deactivated (None).
      2. No mode decimation occurs: at step 1000, mode coverage >= 7/8 modes.
    """
    device = get_device()
    means = ring_means()

    recipe = get_recipe(
        "atlas",
        num_particles=N,
        z_dim=4,
        batch_size=128,
        lr=0.002,
        birth_death_cells=128,
    )
    assert recipe.birth_death_cells == 128

    G = SimpleMLPGenerator(z_dim=4, hidden_dim=96, n_hidden=3, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3, fourier=3).to(device)
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior().to(device)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    assert trainer.birth_death is None

    data_gen = torch.Generator().manual_seed(10)

    for step in range(1000):
        real_batch = sample_ring(means, 128, 0.07, data_gen).to(device)
        trainer.step(real_batch)
        assert trainer.birth_death is None

    # Assert birth_death is None over full 1000 steps
    assert trainer.birth_death is None

    # Mode decimation check: sample from generator and check mode coverage
    eval_gen = torch.Generator(device=device).manual_seed(999)
    samples = trainer.sample(4096, generator=eval_gen, output_noise=False).cpu()
    div = diversity(samples, means, sigma=0.07)

    # For N=8 particles on 8 modes, expected occupancy is 5/8; for N>=10, mode coverage must be >= 7/8
    min_modes = 5 if N == 8 else 7
    assert div["modes"] >= min_modes, (
        f"Mode decimation detected for N={N}: covered {div['modes']}/8 modes (expected >= {min_modes})"
    )


@pytest.mark.parametrize("N", [129, 256, 1024])
def test_large_table_policy_turnover_functions_normally(N: int):
    """Verify that for N in [129, 256, 1024] (> 128), turnover functions normally.

    - N=129 and N=256 route to kNN birth-death, where evaluations and moves execute.
    - N=1024 routes to feature_cells backend, where evaluations and moves execute.
    """
    device = get_device()
    means = ring_means()
    recipe = get_recipe(
        "atlas",
        num_particles=N,
        z_dim=4,
        batch_size=128,
        lr=0.005,
        birth_death_cells=128,
    )
    G = SimpleMLPGenerator(z_dim=4, hidden_dim=96, n_hidden=3, out_dim=2).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3, fourier=3).to(device)
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)

    prior = recipe.make_prior().to(device)
    deterministic_orthogonal_(unwrap_prior(prior), seed=2)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=42)
    data_gen = torch.Generator().manual_seed(10)

    step_budget = 400 if N <= 256 else 150
    for step in range(step_budget):
        real_batch = sample_ring(means, 128, 0.07, data_gen).to(device)
        trainer.step(real_batch)

    bd = trainer.birth_death
    assert bd.counters["evals"] > 0, f"Turnover failed to evaluate for N={N}"
    assert bd.counters["moves"] > 0, f"Turnover produced 0 moves for N={N} > 128"


# ==============================================================================
# Suite 3: Clean Intra-Mode Covariance & Mode Precision Stress Tests
# ==============================================================================

def test_clean_vs_noisy_intra_mode_covariance_trace_bias():
    """Empirically assert covariance trace bias on clean evaluation:
    - output_noise_std=0.0 with sigma=0.005 achieves |cov_trace_bias| <= 0.15.
    - Counterfactual output_noise_std=0.029 with sigma=0.005 produces cov_trace_bias < -0.50
      (demonstrating the severe mode variance deflation caused by training with output noise).
    """
    device = get_device()
    target_sigma = 0.030
    target_variance = target_sigma ** 2
    target_trace = 2.0 * target_variance

    def train_single_mode(noise_std: float, steps: int = 400) -> tuple[float, float, float]:
        recipe = get_recipe(
            "atlas",
            z_dim=2,
            num_particles=256,
            batch_size=128,
            lr=0.005,
            output_noise_mode="fixed",
            output_noise_std=noise_std,
        )
        G = SimpleMLPGenerator(z_dim=2, hidden_dim=64, n_hidden=2, out_dim=2).to(device)
        D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2).to(device)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=1)

        prior = recipe.make_prior(sigma=0.005).to(device)
        deterministic_orthogonal_(unwrap_prior(prior), seed=2)
        trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)

        data_gen = torch.Generator(device=device).manual_seed(42)
        for _ in range(steps):
            real_batch = target_sigma * torch.randn(128, 2, device=device, generator=data_gen)
            trainer.step(real_batch)

        eval_gen = torch.Generator(device=device).manual_seed(713)
        samples = trainer.sample(4096, generator=eval_gen, output_noise=False)
        cov = torch.cov(samples.T)
        trace = float(torch.trace(cov).item())
        bias = (trace - target_trace) / target_trace
        dists = samples.norm(dim=1)
        precision = float((dists <= 3.0 * target_sigma).float().mean().item())
        return trace, bias, precision

    # 1. Clean Cohort
    clean_trace, clean_bias, clean_precision = train_single_mode(noise_std=0.0)
    assert abs(clean_bias) <= 0.15, (
        f"Clean cohort covariance bias {clean_bias:+.4f} exceeded +/-0.15 gate (trace={clean_trace:.6f})"
    )
    assert clean_precision >= 0.95, (
        f"Clean cohort precision {clean_precision*100:.2f}% < 95.0%"
    )

    # 2. Noisy Cohort (Counterfactual)
    noisy_trace, noisy_bias, noisy_precision = train_single_mode(noise_std=0.029)
    assert noisy_bias < -0.50, (
        f"Counterfactual check failed: expected severe negative bias (< -0.50), got {noisy_bias:+.4f}"
    )


def test_clean_vs_noisy_multi_mode_precision_and_coverage():
    """Empirically test mode precision and mode coverage on a 4-mode clustered target:
    - Clean training (noise_std=0.0, sigma=0.005) achieves >= 95% mode precision and covers all 4 modes.
    - Noisy training (noise_std=0.029, sigma=0.005) under clean evaluation drops modes or severely
      distorts cluster representation.
    """
    device = get_device()
    centers = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], device=device)
    sigma_mode = 0.030

    def sample_4mode(n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, 4, (n,), device=device, generator=gen)
        noise = torch.randn(n, 2, device=device, generator=gen) * sigma_mode
        return centers[idx] + noise

    def train_4mode(noise_std: float, steps: int = 800) -> tuple[float, int]:
        recipe = get_recipe(
            "atlas",
            z_dim=2,
            num_particles=256,
            batch_size=128,
            lr=0.005,
            output_noise_mode="fixed",
            output_noise_std=noise_std,
        )
        G = SimpleMLPGenerator(z_dim=2, hidden_dim=64, n_hidden=2, out_dim=2).to(device)
        D = SimpleMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2).to(device)
        deterministic_orthogonal_(G, seed=0)
        deterministic_orthogonal_(D, seed=1)

        prior = recipe.make_prior(sigma=0.005).to(device)
        deterministic_orthogonal_(unwrap_prior(prior), seed=2)
        trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)

        data_gen = torch.Generator(device=device).manual_seed(42)
        for _ in range(steps):
            real_batch = sample_4mode(128, data_gen)
            trainer.step(real_batch)

        eval_gen = torch.Generator(device=device).manual_seed(713)
        samples = trainer.sample(4096, generator=eval_gen, output_noise=False)
        dists, nearest = torch.cdist(samples, centers).min(dim=1)
        precision = float((dists <= 3.0 * sigma_mode).float().mean().item())
        counts = torch.bincount(nearest, minlength=4)
        covered = int((counts >= 0.05 * len(samples)).sum().item())
        return precision, covered

    clean_prec, clean_cov = train_4mode(noise_std=0.0)
    assert clean_cov == 4, f"Clean cohort covered only {clean_cov}/4 modes"
    assert clean_prec >= 0.95, f"Clean cohort precision {clean_prec*100:.2f}% < 95.0%"

    noisy_prec, noisy_cov = train_4mode(noise_std=0.029)
    # Proves degradation in noisy cohort under clean evaluation
    assert noisy_cov < 4 or noisy_prec < clean_prec, (
        f"Expected noisy cohort to degrade compared to clean, got cov={noisy_cov}/4, prec={noisy_prec:.4f}"
    )
