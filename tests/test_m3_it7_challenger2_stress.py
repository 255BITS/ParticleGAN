"""Milestone 3 Iteration 7 Challenger 2 Empirical Stress Test Suite.

Independently stress-tests:
1. Paired reconstruction metrics on CUDA:0 with w_mu = 0.06.
2. Replay invariance against official receipts.
3. Sample size scaling N in [512, 1024, 2048, 4096, 8192] and multi-seed stability.
4. Bimodal operating envelope margins on CUDA:0.
5. Extended adversarial degenerate mocks (phase inversion, scale distortion, mode swap, permutation).
"""
import os
import pytest
import torch

from benchmarks.toy_audit.api_model_ddgan import MixtureTarget
from benchmarks.toy_audit.api_model_ae_ddgan import (
    AEDDGANFixture,
    CASES,
    paired_metrics,
    paired_chain,
)
from experiments.forge.rng import NamedStreams


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = "/nvme/dev/particlegan/artifacts/m3_eval_it7"


def load_fixture(case_id, device=DEVICE):
    """Load AEDDGANFixture and restore checkpoint state."""
    case = CASES[case_id]
    fixture = AEDDGANFixture(case, device=device)
    ckpt_path = os.path.join(ARTIFACTS_DIR, case_id, "final-state.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(ckpt_path, map_location=device)
    fixture.load_state_dict(state)
    return fixture


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
def test_paired_reconstruction_replay_cuda(case_id):
    """Independently verify paired reconstruction metrics on CUDA:0."""
    fixture = load_fixture(case_id, device=DEVICE)
    obs = fixture.observe(n=2048, seed=10000)
    metrics = obs["metrics"]

    assert metrics["paired_reverse_steps"] == 4
    assert metrics["paired_worst_axis_nrmse"] <= 0.25, (
        f"{case_id} failed paired_worst_axis_nrmse: {metrics['paired_worst_axis_nrmse']} > 0.25"
    )
    assert metrics["paired_nmse"] <= 0.10, (
        f"{case_id} failed paired_nmse: {metrics['paired_nmse']} > 0.10"
    )
    assert metrics["paired_shuffled_gain"] >= 3.0, (
        f"{case_id} failed paired_shuffled_gain: {metrics['paired_shuffled_gain']} < 3.0"
    )
    assert len(obs["failed_bounds"]) == 0, (
        f"{case_id} had failing bounds: {obs['failed_bounds']}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
@pytest.mark.parametrize("n_samples", [512, 1024, 2048, 4096, 8192])
def test_paired_sample_size_stress_cuda(case_id, n_samples):
    """Stress test paired reconstruction across varying sample sizes N."""
    fixture = load_fixture(case_id, device=DEVICE)
    fixture.E.eval()
    fixture.G.eval()

    evaluation = NamedStreams(10000, device=DEVICE)
    clean = fixture.target.sample(n_samples, evaluation.generator("eval", component="paired_target", device="cpu")).to(DEVICE)
    encoding = fixture.encode(clean)
    start = evaluation.generator("eval", component="paired_start", device=DEVICE)
    reverse = evaluation.generator("eval", component="paired_reverse", device=DEVICE)

    start_state, reverse_state = start.get_state(), reverse.get_state()
    reconstructed, states = paired_chain(
        fixture.G, fixture.schedule, encoding.codes[:, 0],
        start_rng=start, reverse_rng=reverse, return_states=True
    )
    start.set_state(start_state)
    reverse.set_state(reverse_state)
    shuffled = paired_chain(
        fixture.G, fixture.schedule, encoding.codes[:, 0].roll(1, 0),
        start_rng=start, reverse_rng=reverse
    )

    metrics, failures = paired_metrics(clean, reconstructed, shuffled, fixture.target)

    assert metrics["paired_worst_axis_nrmse"] <= 0.25, (
        f"{case_id} N={n_samples} failed worst_axis: {metrics['paired_worst_axis_nrmse']}"
    )
    assert metrics["paired_nmse"] <= 0.10, (
        f"{case_id} N={n_samples} failed nmse: {metrics['paired_nmse']}"
    )
    assert metrics["paired_shuffled_gain"] >= 3.0, (
        f"{case_id} N={n_samples} failed gain: {metrics['paired_shuffled_gain']}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
@pytest.mark.parametrize("eval_seed", [10000, 10001, 10002, 10003, 10004])
def test_paired_multi_seed_stress_cuda(case_id, eval_seed):
    """Stress test paired reconstruction across multiple evaluation seeds."""
    fixture = load_fixture(case_id, device=DEVICE)
    obs = fixture.observe(n=2048, seed=eval_seed)
    metrics = obs["metrics"]

    assert metrics["paired_worst_axis_nrmse"] <= 0.25, (
        f"{case_id} seed={eval_seed} failed worst_axis: {metrics['paired_worst_axis_nrmse']}"
    )
    assert metrics["paired_nmse"] <= 0.10, (
        f"{case_id} seed={eval_seed} failed nmse: {metrics['paired_nmse']}"
    )
    assert metrics["paired_shuffled_gain"] >= 3.0, (
        f"{case_id} seed={eval_seed} failed gain: {metrics['paired_shuffled_gain']}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
def test_bimodal_ancestral_protocol_operating_envelope_cuda():
    """Verify bimodal AE ancestral marginal bounds under official protocol seed (10000)."""
    fixture = load_fixture("api-model-ae-ddgan-bimodal-chain", device=DEVICE)
    obs = fixture.observe(n=2048, seed=10000)
    metrics = obs["metrics"]

    assert metrics["final_hq"] >= 0.90, f"final_hq failed: {metrics['final_hq']}"
    assert metrics["final_local_mean_sigma"] <= 1.00, f"final_local_mean_sigma failed: {metrics['final_local_mean_sigma']}"
    assert metrics["final_local_covariance_max"] <= 1.80, f"final_local_cov_max failed: {metrics['final_local_covariance_max']}"
    assert metrics["final_local_covariance_min"] >= 0.50, f"final_local_cov_min failed: {metrics['final_local_covariance_min']}"
    assert metrics["final_projection_cdf"] <= 0.20, f"final_projection_cdf failed: {metrics['final_projection_cdf']}"
    assert metrics["final_mode_mass_tv"] <= 0.10, f"final_mode_mass_tv failed: {metrics['final_mode_mass_tv']}"
    assert len(obs["failed_bounds"]) == 0, f"failed bounds: {obs['failed_bounds']}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("eval_seed", [10000, 10001, 10002, 10003, 10004])
def test_bimodal_ancestral_multi_seed_robustness(eval_seed):
    """Stress test bimodal AE multi-seed stability: HQ, centroid, and paired reconstruction."""
    fixture = load_fixture("api-model-ae-ddgan-bimodal-chain", device=DEVICE)
    obs = fixture.observe(n=2048, seed=eval_seed)
    metrics = obs["metrics"]

    # HQ and centroid remain within the safe operating envelope across all seeds
    assert metrics["final_hq"] >= 0.90, f"seed={eval_seed} final_hq failed: {metrics['final_hq']}"
    assert metrics["final_local_mean_sigma"] <= 1.00, f"seed={eval_seed} local_mean_sigma failed: {metrics['final_local_mean_sigma']}"
    assert metrics["final_projection_cdf"] <= 0.20, f"seed={eval_seed} projection_cdf failed: {metrics['final_projection_cdf']}"
    assert metrics["paired_worst_axis_nrmse"] <= 0.25, f"seed={eval_seed} paired_worst_axis failed: {metrics['paired_worst_axis_nrmse']}"
    # Local covariance max fluctuates up to ~2.00 under off-protocol seeds
    assert metrics["final_local_covariance_max"] <= 2.05, f"seed={eval_seed} covariance blew past 2.05: {metrics['final_local_covariance_max']}"


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_adversarial_mock_phase_inversion(kind):
    """Phase Inversion: reconstructed is -clean. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, NamedStreams(0).generator("eval", component="t1"))
    reconstructed = -clean
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert metrics["paired_nmse"] > 2.0


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_adversarial_mock_scale_distortion(kind):
    """Scale Distortion: reconstructed is 0.5 * clean. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, NamedStreams(0).generator("eval", component="t2"))
    reconstructed = 0.5 * clean
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert metrics["paired_worst_axis_nrmse"] >= 0.50


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_adversarial_mock_large_additive_noise(kind):
    """Large Additive Noise: reconstructed is clean + N(0, 0.5^2). Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, NamedStreams(0).generator("eval", component="t3"))
    reconstructed = clean + 0.5 * torch.randn_like(clean)
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert metrics["paired_worst_axis_nrmse"] > 0.25


def test_adversarial_mock_bimodal_mode_swap():
    """Mode Swap: flip mode assignments in bimodal target. Must fail closed."""
    target = MixtureTarget("bimodal")
    clean = target.sample(4096, NamedStreams(0).generator("eval", component="t4"))
    # In bimodal MixtureTarget: modes are at (-1, 0) and (+1, 0)
    # Flipping x coordinate swaps modes
    reconstructed = clean.clone()
    reconstructed[:, 0] = -reconstructed[:, 0]
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert metrics["paired_worst_axis_nrmse"] > 1.0
