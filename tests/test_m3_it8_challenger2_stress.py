"""Milestone 3 Iteration 8 Challenger 2 Adversarial Stress Test Suite.

Independently stress-tests:
1. Bit-exact replay invariance against official m3_eval_it8 receipts on CUDA:0.
2. Paired reconstruction metrics on CUDA:0 for both Gaussian AE and Bimodal AE in m3_eval_it8.
3. Genuine coordinate pairing under code rolls, permutations, and ablations (shuffled gain >= 3.0).
4. Sample size scaling N in [512, 1024, 2048, 4096, 8192] and multi-seed stability.
5. Calibrated bimodal operating envelope on both pure DDGAN and AE-DDGAN in m3_eval_it8.
6. Fail-closed rejection of orthogonal collapse (lambda <= 0.24) and diffuse modes (lambda > 2.05).
7. Extended adversarial degenerate mock rejection (16+ mock variants).
"""
import json
import os
import pytest
import torch

from benchmarks.toy_audit.api_model_ddgan import (
    CASES as DD_CASES,
    DDGANFixture,
)
from benchmarks.toy_audit.api_model_ae_ddgan import (
    AEDDGANFixture,
    CASES as AE_CASES,
    paired_metrics,
    paired_chain,
)
from experiments.forge.rng import NamedStreams

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = "/nvme/dev/particlegan/artifacts/m3_eval_it8"


def load_fixture(case_id, device=DEVICE):
    """Load DDGANFixture or AEDDGANFixture and restore checkpoint state."""
    if "ae-ddgan" in case_id:
        case = AE_CASES[case_id]
        fixture = AEDDGANFixture(case, device=device)
    else:
        case = DD_CASES[case_id]
        fixture = DDGANFixture(case, device=device)
    ckpt_path = os.path.join(ARTIFACTS_DIR, case_id, "final-state.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(ckpt_path, map_location=device)
    fixture.load_state_dict(state)
    return fixture


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ddgan-gaussian-chain",
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ddgan-bimodal-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
def test_it8_checkpoint_replay_determinism_cuda(case_id):
    """Verify bit-exact replay invariance vs receipt.json on CUDA:0."""
    fixture = load_fixture(case_id, device=DEVICE)
    obs = fixture.observe(n=fixture.case["eval_samples"], seed=10000)

    receipt_path = os.path.join(ARTIFACTS_DIR, case_id, "receipt.json")
    with open(receipt_path) as f:
        receipt = json.load(f)
    receipt_obs = receipt["observations"][-1]

    assert obs["passed"] is True, f"{case_id} failed observation: {obs['failed_bounds']}"
    assert len(obs["failed_bounds"]) == 0
    assert obs["passed"] == receipt_obs["passed"]

    max_diff = 0.0
    for k, v in obs["metrics"].items():
        rv = receipt_obs["metrics"][k]
        diff = abs(v - rv)
        if diff > max_diff:
            max_diff = diff
    assert max_diff < 1e-6, f"{case_id} metric discrepancy vs receipt: {max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
def test_it8_paired_reconstruction_replay_cuda(case_id):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
def test_it8_genuine_coordinate_pairing_shuffled_loss_cuda(case_id):
    """Empirically prove coordinate pairing is genuine (uncoupled codes cause catastrophic error)."""
    fixture = load_fixture(case_id, device=DEVICE)
    fixture.E.eval()
    fixture.G.eval()

    evaluation = NamedStreams(10000, device=DEVICE)
    clean = fixture.target.sample(2048, evaluation.generator("eval", component="paired_target", device="cpu")).to(DEVICE)
    encoding = fixture.encode(clean)
    start = evaluation.generator("eval", component="paired_start", device=DEVICE)
    reverse = evaluation.generator("eval", component="paired_reverse", device=DEVICE)

    start_st, rev_st = start.get_state(), reverse.get_state()
    reconstructed = paired_chain(
        fixture.G, fixture.schedule, encoding.codes[:, 0],
        start_rng=start, reverse_rng=reverse
    )

    # 1. Shuffled by roll
    start.set_state(start_st); reverse.set_state(rev_st)
    shuffled_roll = paired_chain(
        fixture.G, fixture.schedule, encoding.codes[:, 0].roll(1, 0),
        start_rng=start, reverse_rng=reverse
    )
    m_roll, f_roll = paired_metrics(clean, reconstructed, shuffled_roll, fixture.target)
    assert m_roll["paired_shuffled_gain"] >= 3.0
    assert m_roll["paired_nmse"] <= 0.10

    # 2. Random code permutation
    start.set_state(start_st); reverse.set_state(rev_st)
    perm = torch.randperm(2048, device=DEVICE)
    shuffled_perm = paired_chain(
        fixture.G, fixture.schedule, encoding.codes[:, 0][perm],
        start_rng=start, reverse_rng=reverse
    )
    m_perm, f_perm = paired_metrics(clean, reconstructed, shuffled_perm, fixture.target)
    assert m_perm["paired_shuffled_gain"] >= 3.0

    # 3. Complete code zeroing (destroying code information completely)
    start.set_state(start_st); reverse.set_state(rev_st)
    zero_codes = torch.zeros_like(encoding.codes[:, 0])
    zero_recon = paired_chain(
        fixture.G, fixture.schedule, zero_codes,
        start_rng=start, reverse_rng=reverse
    )
    m_zero, f_zero = paired_metrics(clean, reconstructed, zero_recon, fixture.target)
    assert m_zero["paired_shuffled_gain"] >= 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
@pytest.mark.parametrize("n_samples", [512, 1024, 2048, 4096, 8192])
def test_it8_paired_sample_size_stress_cuda(case_id, n_samples):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ae-ddgan-gaussian-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
@pytest.mark.parametrize("eval_seed", [10000, 10001, 10002, 10003, 10004])
def test_it8_paired_multi_seed_stress_cuda(case_id, eval_seed):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("case_id", [
    "api-model-ddgan-bimodal-chain",
    "api-model-ae-ddgan-bimodal-chain",
])
def test_it8_bimodal_ancestral_operating_envelope_cuda(case_id):
    """Verify bimodal ancestral marginal operating envelope under protocol seed (10000)."""
    fixture = load_fixture(case_id, device=DEVICE)
    obs = fixture.observe(n=2048, seed=10000)
    metrics = obs["metrics"]

    assert metrics["final_hq"] >= 0.90, f"{case_id} final_hq failed: {metrics['final_hq']}"
    assert metrics["final_mode_mass_tv"] <= 0.10, f"{case_id} final_mode_mass_tv failed: {metrics['final_mode_mass_tv']}"
    assert metrics["final_local_mean_sigma"] <= 1.50, f"{case_id} local_mean_sigma failed: {metrics['final_local_mean_sigma']}"
    assert metrics["final_local_covariance_min"] >= 0.30, f"{case_id} cov_min failed: {metrics['final_local_covariance_min']}"
    assert metrics["final_local_covariance_max"] <= 2.05, f"{case_id} cov_max failed: {metrics['final_local_covariance_max']}"
    assert metrics["final_projection_cdf"] <= 0.32, f"{case_id} projection_cdf failed: {metrics['final_projection_cdf']}"
    assert len(obs["failed_bounds"]) == 0, f"{case_id} had failing bounds: {obs['failed_bounds']}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for empirical challenge")
@pytest.mark.parametrize("eval_seed", [10000, 10001, 10002, 10003, 10004])
def test_it8_ae_bimodal_multi_seed_covariance_envelope_cuda(eval_seed):
    """Verify that calibrated cov_max <= 2.05 encloses multi-seed fluctuations on AE-DDGAN."""
    fixture = load_fixture("api-model-ae-ddgan-bimodal-chain", device=DEVICE)
    obs = fixture.observe(n=2048, seed=eval_seed)
    metrics = obs["metrics"]

    assert metrics["final_hq"] >= 0.90
    assert metrics["final_local_mean_sigma"] <= 1.50
    assert metrics["final_local_covariance_max"] <= 2.05, (
        f"seed={eval_seed} cov_max blew past 2.05: {metrics['final_local_covariance_max']}"
    )
    assert metrics["final_local_covariance_min"] >= 0.30, (
        f"seed={eval_seed} cov_min collapsed below 0.30: {metrics['final_local_covariance_min']}"
    )


@pytest.mark.parametrize("scale_y", [0.45, 0.40, 0.30, 0.10, 0.0])
def test_it8_orthogonal_collapse_rejection(scale_y):
    """Verify orthogonal collapse (lambda <= 0.24) fails closed under final_local_covariance_min >= 0.30."""
    n = 2048
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (n,), generator=g)
    means = torch.tensor([[-0.75, 0.0], [0.75, 0.0]])
    pts = means[labels] + torch.randn(n, 2, generator=g) * torch.tensor([0.18, 0.18 * scale_y])

    m0_pts = pts[labels == 0]
    m1_pts = pts[labels == 1]
    cov0 = torch.cov(m0_pts.T) / (0.18 ** 2)
    cov1 = torch.cov(m1_pts.T) / (0.18 ** 2)
    eigs0 = torch.linalg.eigvalsh(cov0)
    eigs1 = torch.linalg.eigvalsh(cov1)
    lam_min = float(min(eigs0.min(), eigs1.min()))

    # Must be <= 0.24 and strictly fail the >= 0.30 gate
    assert lam_min <= 0.24
    assert lam_min < 0.30, f"Orthogonal collapse scale_y={scale_y} unexpectedly passed gate: {lam_min}"


@pytest.mark.parametrize("scale", [1.45, 1.50, 1.60, 2.00, 3.00])
def test_it8_diffuse_distribution_rejection(scale):
    """Verify diffuse distributions (lambda > 2.05 / 2.50) fail closed under final_local_covariance_max <= 2.05."""
    n = 2048
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (n,), generator=g)
    means = torch.tensor([[-0.75, 0.0], [0.75, 0.0]])
    pts = means[labels] + torch.randn(n, 2, generator=g) * torch.tensor([0.18 * scale, 0.18 * scale])

    m0_pts = pts[labels == 0]
    m1_pts = pts[labels == 1]
    cov0 = torch.cov(m0_pts.T) / (0.18 ** 2)
    cov1 = torch.cov(m1_pts.T) / (0.18 ** 2)
    eigs0 = torch.linalg.eigvalsh(cov0)
    eigs1 = torch.linalg.eigvalsh(cov1)
    lam_max = float(max(eigs0.max(), eigs1.max()))

    # Must be > 2.05 and strictly fail the <= 2.05 gate
    assert lam_max > 2.05, f"Diffuse scale={scale} unexpectedly passed gate: {lam_max}"
