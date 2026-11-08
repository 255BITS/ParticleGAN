"""Tier 5 Adversarial Coverage Hardening on ParticleGAN Public-API Fixtures and Runners.

Adversarially stress-tests:
1. Malformed inputs and NaN/Inf tensor handling in metrics across all Tier 1, 3, and 4 suites.
2. Checkpoint state dict serialization / deserialization round-trip across all 7 new fixtures,
   plus corruption/tampering rejection.
3. Particle posterior evaluation edge cases with zero/single particles, single/zero-sample batches,
   extreme sigmas, and boundary masks.
4. BiGAN continuous reconstruction fidelity under perturbed latents (Lipschitz continuity,
   degradation profiles, adversarial FGSM perturbations, and non-finite propagation).
5. Public-API runner (api_run) bounds checking, wall-cap timeouts, and non-finite JSON serialization.
"""
from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.toy_audit import api_contract as contract
from benchmarks.toy_audit import api_run
from benchmarks.toy_audit import api_vectors as api_vec
from benchmarks.toy_audit import api_conditionals as api_cond
from benchmarks.toy_audit import api_ring16 as api_ring
from benchmarks.toy_audit.api_sources import misgan
from particlegan import MoGParticlePrior


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ============================================================================
# 1. Malformed inputs & Nonfinite (NaN / +Inf / -Inf) Metric Handling
# ============================================================================

class TestNonfiniteAndMalformedMetrics:
    """Stress-test metric evaluators against NaN, +Inf, -Inf, and malformed tensor inputs."""

    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    def test_score_case_zero_tanh_smoke_nonfinite(self, corrupt_val):
        case = contract.discover()["api-zero-tanh-smoke"]
        samples = torch.zeros(100, 2)
        samples[0, 0] = corrupt_val
        result = api_vec.score_case(case, samples)
        assert result["passed"] is False
        assert result["metrics"]["all_finite"] is False
        assert math.isinf(result["metrics"]["mean_abs_error"])
        assert math.isinf(result["metrics"]["rms_error"])
        assert math.isinf(result["metrics"]["p99_abs_error"])
        assert result["failed_bounds"] == ["all_finite == true"]

    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("step", [500, 1500])
    def test_score_case_zero_tanh_rail_escape_nonfinite(self, corrupt_val, step):
        case = contract.discover()["api-zero-tanh-rail-escape"]
        samples = torch.zeros(100, 2)
        samples[42, 1] = corrupt_val
        result = api_vec.score_case(case, samples, completed_steps=step)
        assert result["passed"] is False
        assert result["metrics"]["all_finite"] is False
        assert math.isinf(result["metrics"]["mean_abs_error"])
        assert result["failed_bounds"] == ["all_finite == true"]

    def test_score_case_zero_tanh_malformed_shapes(self):
        case = contract.discover()["api-zero-tanh-smoke"]
        # Empty tensor
        with pytest.raises(ValueError, match="needs nonempty, finite row-major points"):
            api_vec.score_case(case, torch.empty(0, 2))
        # 1D tensor
        with pytest.raises(ValueError, match="needs nonempty, finite row-major points"):
            api_vec.score_case(case, torch.zeros(10))
        # 3D tensor
        with pytest.raises(ValueError, match="needs nonempty, finite row-major points"):
            api_vec.score_case(case, torch.zeros(5, 5, 2))
        # Wrong column dimension (3 instead of 2)
        with pytest.raises(ValueError, match="zero_tanh points need two coordinates"):
            api_vec.score_case(case, torch.zeros(10, 3))

    @pytest.mark.parametrize("case_id", ["api-grid100-mlp16", "api-grid100-res16-tanh"])
    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    def test_score_case_grid100_suites_nonfinite(self, case_id, corrupt_val):
        case = contract.discover()[case_id]
        samples = torch.zeros(200, 2)
        samples[10, 0] = corrupt_val
        result = api_vec.score_case(case, samples)
        assert result["passed"] is False
        assert result["metrics"]["all_finite"] is False
        assert result["failed_bounds"] == ["all_finite == true"]

    def test_score_case_grid100_malformed_shapes(self):
        case = contract.discover()["api-grid100-mlp16"]
        with pytest.raises(ValueError, match="needs nonempty, finite row-major points"):
            api_vec.score_case(case, torch.empty(0, 2))
        with pytest.raises(ValueError, match="100-mode points need two coordinates"):
            api_vec.score_case(case, torch.zeros(50, 4))

    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    def test_score_misgan_recovery_nonfinite(self, corrupt_val):
        x = torch.zeros(100, 8)
        m = torch.ones(100, 8)
        # Corrupt x
        x_corrupt = x.clone()
        x_corrupt[3, 2] = corrupt_val
        obs_x = api_cond.score_misgan_recovery(x_corrupt, m)
        assert obs_x["passed"] is False
        assert "nonfinite_output" in obs_x["failed_bounds"]
        assert math.isinf(obs_x["metrics"]["off_plane"])

        # Corrupt m
        m_corrupt = m.clone()
        m_corrupt[5, 1] = corrupt_val
        obs_m = api_cond.score_misgan_recovery(x, m_corrupt)
        assert obs_m["passed"] is False
        assert "nonfinite_output" in obs_m["failed_bounds"]

    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    def test_score_misgan_neural_imputer_nonfinite(self, corrupt_val):
        # Neural imputer expects (draws, N, 8) or (N, 8)
        draws = torch.zeros(2, 50, 8)
        draws[0, 12, 4] = corrupt_val
        obs = api_cond.score_misgan_neural_imputer(draws)
        assert obs["passed"] is False
        assert "nonfinite_output" in obs["failed_bounds"]
        assert math.isinf(obs["metrics"]["observed_max_error"])
        assert obs["metrics"]["acc"] == 0.0

    @pytest.mark.parametrize("corrupt_val", [float("nan"), float("inf"), float("-inf")])
    def test_score_bigan_nonfinite(self, corrupt_val):
        samples = torch.zeros(100, 2)
        target = torch.zeros(100, 2)
        reconstructed = torch.zeros(100, 2)

        # Corrupt samples
        s_corrupt = samples.clone()
        s_corrupt[0, 0] = corrupt_val
        obs_s = api_ring.score_bigan(s_corrupt, target, reconstructed)
        assert obs_s["passed"] is False
        assert "nonfinite_output" in obs_s["failed_bounds"]
        assert math.isinf(obs_s["metrics"]["recon_rmse"])

        # Corrupt reconstructed
        r_corrupt = reconstructed.clone()
        r_corrupt[0, 1] = corrupt_val
        obs_r = api_ring.score_bigan(samples, target, r_corrupt)
        assert obs_r["passed"] is False
        assert "nonfinite_output" in obs_r["failed_bounds"]
        assert math.isinf(obs_r["metrics"]["recon_rmse"])

    def test_api_run_json_value_nonfinite_sanitization(self):
        """api_run.json_value must serialize NaN/Inf to strings without crashing atomic_json."""
        raw = {
            "scalar_nan": float("nan"),
            "scalar_inf": float("inf"),
            "scalar_ninf": float("-inf"),
            "nested": {"array": [1.0, float("nan"), float("inf")]},
            "tensor": torch.tensor([float("nan"), 2.0]),
        }
        sanitized = api_run.json_value(raw)
        assert sanitized["scalar_nan"] == "nan"
        assert sanitized["scalar_inf"] == "inf"
        assert sanitized["scalar_ninf"] == "-inf"
        assert sanitized["nested"]["array"] == [1.0, "nan", "inf"]
        assert sanitized["tensor"] == ["nan", 2.0]


# ============================================================================
# 2. Checkpoint Serialization / Deserialization Round-Trip Across All Fixtures
# ============================================================================

class TestCheckpointStateDictRoundTrip:
    """Stress-test checkpoint serialization, deserialization, and corruption rejection."""

    ALL_TARGET_CASES = [
        "api-zero-tanh-smoke",
        "api-zero-tanh-rail-escape",
        "api-grid100-mlp16",
        "api-grid100-res16-tanh",
        "api-misgan-block-recovery",
        "api-misgan-block-neural-imputer",
        "api-ring16-bigan-continuous",
    ]

    @pytest.mark.parametrize("case_id", ALL_TARGET_CASES)
    def test_checkpoint_roundtrip_parameter_fidelity(self, case_id):
        """Verify full byte-level state_dict serialization and exact parameter restoration."""
        cases = contract.discover()
        case = cases[case_id]

        # 1. Build initial fixture and run 2 steps
        fixture1 = contract.build(case, max_steps=10)
        for _ in range(2):
            fixture1.step()
        assert fixture1.completed_steps == 2

        # 2. Serialize state dict to bytes and deserialize
        sd = fixture1.state_dict()
        buffer = io.BytesIO()
        torch.save(sd, buffer)
        buffer.seek(0)
        sd_loaded = torch.load(buffer)

        # 3. Build fresh fixture and restore state
        fixture2 = contract.build(case, max_steps=10)
        assert fixture2.completed_steps == 0
        fixture2.load_state_dict(sd_loaded)
        assert fixture2.completed_steps == 2

        # 4. Confirm bitwise identical parameter weights across all model components
        for mname in ("G", "D", "E", "prior", "generator", "critic", "encoder",
                      "G_x", "G_m", "G_i", "D_x", "D_m", "D_i"):
            m1 = getattr(fixture1, mname, None)
            m2 = getattr(fixture2, mname, None)
            if m1 is not None and isinstance(m1, nn.Module):
                for (pname, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                    assert torch.equal(p1, p2), f"{case_id}.{mname}.{pname} parameter mismatch"

        # 5. Step 3 on restored fixture continues cleanly to step 3 with finite losses
        stats2 = fixture2.step()
        assert stats2["step"] == 3
        for k, v in stats2.items():
            if isinstance(v, float):
                assert math.isfinite(v)

    VECTOR_CASES = [
        "api-zero-tanh-smoke",
        "api-zero-tanh-rail-escape",
        "api-grid100-mlp16",
        "api-grid100-res16-tanh",
    ]

    @pytest.mark.parametrize("case_id", VECTOR_CASES)
    def test_vector_fixtures_deterministic_rng_continuation(self, case_id):
        """VectorFixture preserves data_rng and generator_data_rng; step 3 losses match exactly."""
        cases = contract.discover()
        case = cases[case_id]

        fixture1 = contract.build(case, max_steps=10)
        fixture1.step()
        fixture1.step()

        sd = fixture1.state_dict()
        fixture2 = contract.build(case, max_steps=10)
        fixture2.load_state_dict(sd)

        stats1 = fixture1.step()
        stats2 = fixture2.step()
        assert stats1["step"] == stats2["step"] == 3
        for k in stats1:
            if isinstance(stats1[k], float):
                assert math.isclose(stats1[k], stats2[k], rel_tol=1e-5, abs_tol=1e-6)

    @pytest.mark.parametrize("case_id", [
        "api-misgan-block-recovery",
        "api-misgan-block-neural-imputer",
        "api-ring16-bigan-continuous",
    ])
    def test_misgan_and_bigan_uncheckpointed_rng_streams_finding(self, case_id):
        """Empirical verification: MisGAN and BiGAN omit data_rng and latent_rng from state_dict().

        Restoring from checkpoint does not resume the consumed RNG stream, replaying data
        batches from initial offset 0 unless RNG state is manually transferred.
        """
        cases = contract.discover()
        case = cases[case_id]

        fixture1 = contract.build(case, max_steps=10)
        fixture1.step()
        fixture1.step()

        sd = fixture1.state_dict()
        assert "data_rng" not in sd, f"{case_id} unexpectedly serialized data_rng"
        assert "latent_rng" not in sd, f"{case_id} unexpectedly serialized latent_rng"

        fixture2 = contract.build(case, max_steps=10)
        fixture2.load_state_dict(sd)

        # Without RNG sync, the batches drawn at step 3 differ (initial offset vs 2-step offset)
        # When RNG is manually synced, BiGAN matches bitwise identically
        if case_id == "api-ring16-bigan-continuous":
            fixture2.data_rng.set_state(fixture1.data_rng.get_state())
            fixture2.latent_rng.set_state(fixture1.latent_rng.get_state())
            s1 = fixture1.step()
            s2 = fixture2.step()
            assert s1 == s2, "BiGAN step 3 should match bitwise when RNG streams are synchronized"

    def test_vector_fixture_checkpoint_tampering_rejection(self):
        """VectorFixture must reject invalid or forged checkpoints."""
        case = contract.discover()["api-zero-tanh-smoke"]
        fixture = contract.build(case, max_steps=10)
        fixture.step()
        sd = fixture.state_dict()

        # Negative completed_steps
        corrupt_sd = {**sd, "completed_steps": -1}
        with pytest.raises(ValueError, match="checkpoint exceeds this execution prefix"):
            fixture.load_state_dict(corrupt_sd)

        # Exceeds max_steps
        corrupt_sd = {**sd, "completed_steps": 9999}
        with pytest.raises(ValueError, match="checkpoint exceeds this execution prefix"):
            fixture.load_state_dict(corrupt_sd)

        # Mismatched case_id
        corrupt_sd = {**sd, "case_id": "api-wrong-case-id"}
        with pytest.raises(ValueError, match="checkpoint identity/recipe differs"):
            fixture.load_state_dict(corrupt_sd)

        # Mismatched version
        corrupt_sd = {**sd, "version": "invalid-version-99"}
        with pytest.raises(ValueError, match="checkpoint identity/recipe differs"):
            fixture.load_state_dict(corrupt_sd)

        # Corrupted batch_sequence_sha256 digest
        corrupt_sd = {**sd, "batch_sequence_sha256": "bad_digest"}
        with pytest.raises(ValueError, match="checkpoint batch sequence digest differs"):
            fixture.load_state_dict(corrupt_sd)

    def test_misgan_and_bigan_checkpoint_missing_keys_rejection(self):
        """MisGAN and BiGAN fixtures must raise KeyError if required keys are missing."""
        # MisGAN recovery
        case_rec = contract.discover()["api-misgan-block-recovery"]
        fix_rec = contract.build(case_rec, max_steps=10)
        fix_rec.step()
        sd_rec = fix_rec.state_dict()
        del sd_rec["G_x"]
        with pytest.raises(KeyError):
            fix_rec.load_state_dict(sd_rec)

        # MisGAN neural imputer
        case_imp = contract.discover()["api-misgan-block-neural-imputer"]
        fix_imp = contract.build(case_imp, max_steps=10)
        fix_imp.step()
        sd_imp = fix_imp.state_dict()
        del sd_imp["G_i"]
        with pytest.raises(KeyError):
            fix_imp.load_state_dict(sd_imp)

        # Ring16 BiGAN
        case_bi = contract.discover()["api-ring16-bigan-continuous"]
        fix_bi = contract.build(case_bi, max_steps=10)
        fix_bi.step()
        sd_bi = fix_bi.state_dict()
        del sd_bi["generator"]
        with pytest.raises(KeyError):
            fix_bi.load_state_dict(sd_bi)


# ============================================================================
# 3. Particle Posterior Evaluation Edge Cases (Zero/Single & Extremes)
# ============================================================================

class TestParticlePosteriorEvaluationEdgeCases:
    """Stress-test Bayesian particle posterior imputation and MoG prior boundary conditions."""

    def test_misgan_ppost_indices_zero_particle_table_rejection(self):
        """misgan_ppost_indices must reject empty candidate particle tables."""
        x = torch.randn(10, 8)
        m = torch.ones(10, 8)
        with pytest.raises(ValueError, match="candidate particle table cannot be empty"):
            api_cond.misgan_ppost_indices(torch.empty(0, 8), x, m)

    def test_misgan_ppost_indices_single_particle_candidate_table(self):
        """Table with exactly 1 particle must deterministically assign all draws to index 0."""
        table = torch.randn(1, 8)
        x = torch.randn(16, 8)
        m = (torch.rand(16, 8) > 0.5).float()
        indices = api_cond.misgan_ppost_indices(table, x, m, draws=3)
        assert indices.shape == (3, 16)
        assert torch.all(indices == 0)

    def test_misgan_ppost_indices_single_sample_batch(self):
        """Single-sample batch (N=1) must return shape (draws, 1) without dimensional collapse."""
        table = torch.randn(64, 8)
        x = torch.randn(1, 8)
        m = torch.ones(1, 8)
        indices = api_cond.misgan_ppost_indices(table, x, m, draws=5)
        assert indices.shape == (5, 1)
        assert torch.all(indices >= 0) and torch.all(indices < 64)

    def test_misgan_ppost_indices_zero_sample_batch(self):
        """Zero-sample batch (N=0) must return shape (draws, 0) gracefully without error."""
        table = torch.randn(32, 8)
        x = torch.empty(0, 8)
        m = torch.empty(0, 8)
        indices = api_cond.misgan_ppost_indices(table, x, m, draws=4)
        assert indices.shape == (4, 0)

    def test_misgan_ppost_indices_extreme_sigmas_and_underflow_safety(self):
        """Test underflow-inducing ultra-small sigma and flat ultra-large sigma."""
        table = torch.randn(20, 8)
        x = torch.randn(10, 8)
        m = torch.ones(10, 8)

        # Ultra-small sigma: distance * scale causes extreme negative logits, triggering nan_mask protection
        idx_small = api_cond.misgan_ppost_indices(table, x, m, sigma=1e-15, draws=4)
        assert idx_small.shape == (4, 10)
        assert torch.all(idx_small >= 0) and torch.all(idx_small < 20)

        # Ultra-large sigma: distances become negligible, softmax approaches uniform
        idx_large = api_cond.misgan_ppost_indices(table, x, m, sigma=1e15, draws=4)
        assert idx_large.shape == (4, 10)
        assert torch.all(idx_large >= 0) and torch.all(idx_large < 20)

    def test_misgan_ppost_indices_extreme_masks(self):
        """All-observed (m=1) vs all-missing (m=0) masks."""
        table = torch.randn(30, 8)
        x = torch.randn(15, 8)

        # All-zero mask: all coordinates missing, distance is identically 0
        m_zero = torch.zeros(15, 8)
        idx_zero = api_cond.misgan_ppost_indices(table, x, m_zero, draws=2)
        assert idx_zero.shape == (2, 15)
        assert torch.all(idx_zero >= 0) and torch.all(idx_zero < 30)

        # All-one mask: full coordinates observed
        m_one = torch.ones(15, 8)
        idx_one = api_cond.misgan_ppost_indices(table, x, m_one, draws=2)
        assert idx_one.shape == (2, 15)

    def test_mog_particle_prior_particle_count_constraints(self):
        """MoGParticlePrior must enforce positive particle counts and standardization rules."""
        with pytest.raises(ValueError, match="num_particles must be positive"):
            MoGParticlePrior(num_particles=0, sigma=0.1)

        with pytest.raises(ValueError, match="num_particles must be positive"):
            MoGParticlePrior(num_particles=-10, sigma=0.1)

        # Standardization with 1 particle must fail (requires variance across >= 2 particles)
        with pytest.raises(ValueError, match="standardized means require at least two particles"):
            MoGParticlePrior(num_particles=1, sigma=0.1, standardize=True)

        # Non-standardized with 1 particle is valid
        prior1 = MoGParticlePrior(num_particles=1, sigma=0.1, standardize=False)
        assert prior1.num_particles == 1

    def test_mog_particle_prior_sampling_batch_size_constraints(self):
        """MoGParticlePrior sampling must validate batch sizes."""
        prior = MoGParticlePrior(num_particles=32, z_dim=4, sigma=0.1, standardize=False)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            prior.sample(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            prior.sample(batch_size=-1)

        # Single-sample batch returns shape (1, z_dim)
        z, idx = prior.sample(batch_size=1)
        assert z.shape == (1, 4)
        assert idx.shape == (1,)


# ============================================================================
# 4. BiGAN Continuous Reconstruction Fidelity Under Perturbed Latents
# ============================================================================

class TestBiGANContinuousReconstructionUnderPerturbation:
    """Stress-test continuous multimodal BiGAN round-trip reconstruction under latent perturbations."""

    @pytest.fixture
    def bigan_witness(self):
        """Trained constructive witness architecture for ring16 continuous BiGAN."""
        angles = torch.linspace(0, 2 * math.pi, 17)[:16]
        centers = torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()

        torch.manual_seed(101)
        gen = api_ring.BiGANGenerator(z_dim=2, out_dim=2, hidden=64)
        enc = api_ring.BiGANEncoder(in_dim=2, z_dim=2, hidden=64)
        opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=0.01)

        for _ in range(60):
            opt.zero_grad()
            k = torch.randint(0, 16, (128,))
            x = centers[k] + torch.randn(128, 2) * 0.10
            loss = F.mse_loss(gen(x), x) + F.mse_loss(enc(x), x)
            loss.backward()
            opt.step()

        gen.eval()
        enc.eval()
        return gen, enc, centers

    def test_bigan_latent_perturbation_lipschitz_continuity(self, bigan_witness):
        """Verify generator displacement ||G(z + delta) - G(z)|| is continuous and locally Lipschitz."""
        gen, enc, centers = bigan_witness
        rng = torch.Generator().manual_seed(202)
        k = torch.randint(0, 16, (256,), generator=rng)
        target = centers[k] + torch.randn(256, 2, generator=rng) * 0.10

        z = enc(target)
        recon_clean = gen(z)

        # Evaluate across increasing perturbation radii eps
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.10]
        displacements = []
        for eps in epsilons:
            delta = torch.randn_like(z)
            delta = delta / delta.norm(dim=1, keepdim=True) * eps
            recon_pert = gen(z + delta)
            disp = (recon_pert - recon_clean).norm(dim=1).mean().item()
            displacements.append(disp)

        # Displacements must increase monotonically with epsilon without discontinuity
        for i in range(len(displacements) - 1):
            assert displacements[i] < displacements[i + 1]

        # Empirical Lipschitz constant (disp / eps) must remain bounded
        lipschitz_ratios = [d / e for d, e in zip(displacements, epsilons)]
        assert max(lipschitz_ratios) < 20.0

    def test_bigan_latent_perturbation_degradation_spectrum(self, bigan_witness):
        """Perturbing latents must smoothly degrade reconstruction until gate violation."""
        gen, enc, centers = bigan_witness
        rng = torch.Generator().manual_seed(303)
        k = torch.randint(0, 16, (1024,), generator=rng)
        target = centers[k] + torch.randn(1024, 2, generator=rng) * 0.10

        z = enc(target)

        # Baseline clean reconstruction
        obs_clean = api_ring.score_bigan(gen(z), target, gen(z), centers=centers)
        rmse_clean = obs_clean["metrics"]["recon_rmse"]
        cons_clean = obs_clean["metrics"]["mode_consistency"]
        assert cons_clean >= 0.98

        # Moderate noise (sigma=0.02): reconstruction error slightly increases but stays well-conditioned
        z_mod = z + 0.02 * torch.randn_like(z)
        obs_mod = api_ring.score_bigan(gen(z), target, gen(z_mod), centers=centers)
        assert obs_mod["metrics"]["recon_rmse"] >= rmse_clean - 0.01

        # Severe noise (sigma=1.0): breaks reconstruction gate (RMSE > 0.12)
        z_severe = z + 1.0 * torch.randn_like(z)
        obs_severe = api_ring.score_bigan(gen(z), target, gen(z_severe), centers=centers)
        assert obs_severe["metrics"]["recon_rmse"] > 0.12
        assert obs_severe["passed"] is False

    def test_bigan_adversarial_fgsm_latent_perturbation(self, bigan_witness):
        """Adversarial latent perturbation delta = eps * sign(grad_z L) causes greater distortion than random noise."""
        gen, enc, centers = bigan_witness
        torch.manual_seed(404)
        k = torch.randint(0, 16, (128,))
        target = centers[k] + torch.randn(128, 2) * 0.10

        z = enc(target).detach().requires_grad_(True)
        recon = gen(z)
        loss = F.mse_loss(recon, target)
        loss.backward()

        eps = 0.05
        # 1. Adversarial FGSM perturbation
        delta_adv = eps * z.grad.sign()
        recon_adv = gen(z.detach() + delta_adv)
        loss_adv = F.mse_loss(recon_adv, target).item()

        # 2. Random perturbation with same Linf norm
        delta_rand = eps * torch.sign(torch.randn_like(z))
        recon_rand = gen(z.detach() + delta_rand)
        loss_rand = F.mse_loss(recon_rand, target).item()

        # Adversarial attack must inflict greater reconstruction error
        assert loss_adv > loss_rand

    def test_bigan_nonfinite_latent_propagation(self, bigan_witness):
        """Nonfinite latents in BiGAN generator must propagate nonfinite values and fail score_bigan."""
        gen, enc, centers = bigan_witness
        z_nan = torch.zeros(32, 2)
        z_nan[0, 0] = float("nan")
        recon = gen(z_nan)
        assert not torch.all(torch.isfinite(recon))

        target = centers[:32]
        samples = gen(torch.randn(32, 2))
        obs = api_ring.score_bigan(samples, target, recon, centers=centers)
        assert obs["passed"] is False
        assert "nonfinite_output" in obs["failed_bounds"]


# ============================================================================
# 5. Public-API Runner (api_run) Bounds & Timeout Handling
# ============================================================================

class TestApiRunnerSafetyAndValidation:
    """Stress-test benchmarks.toy_audit.api_run parameter constraints and wall-cap timeouts."""

    def test_api_run_eval_samples_must_be_positive_int(self):
        case = contract.discover()["api-zero-tanh-smoke"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            with pytest.raises(ValueError, match="evaluation sample count must be positive"):
                api_run.run_case(case, out / "run_eval0", eval_samples=0)
            with pytest.raises(ValueError, match="evaluation sample count must be positive"):
                api_run.run_case(case, out / "run_eval_neg", eval_samples=-10)

    def test_api_run_wall_cap_must_be_finite_positive(self):
        case = contract.discover()["api-zero-tanh-smoke"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            with pytest.raises(ValueError, match="wall cap must be finite positive seconds"):
                api_run.run_case(case, out / "run_cap_neg", wall_cap_seconds=-1.0)
            with pytest.raises(ValueError, match="wall cap must be finite positive seconds"):
                api_run.run_case(case, out / "run_cap_zero", wall_cap_seconds=0.0)
            with pytest.raises(ValueError, match="wall cap must be finite positive seconds"):
                api_run.run_case(case, out / "run_cap_nan", wall_cap_seconds=float("nan"))
            with pytest.raises(ValueError, match="wall cap must be finite positive seconds"):
                api_run.run_case(case, out / "run_cap_bool", wall_cap_seconds=True)

    def test_api_run_wall_cap_timeout_graceful_handling(self):
        """A timeout during run_case must write a valid receipt with status='INCOMPLETE' and verdict='FAIL'."""
        case = contract.discover()["api-zero-tanh-smoke"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "timeout_run"
            receipt = api_run.run_case(case, out, steps=10, wall_cap_seconds=0.0001)
            assert receipt["status"] == "INCOMPLETE"
            assert receipt["verdict"] == "FAIL"
            assert receipt["passed"] is False
            assert (out / "receipt.json").is_file()
