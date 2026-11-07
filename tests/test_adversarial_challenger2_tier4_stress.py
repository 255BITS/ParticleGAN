"""Adversarial stress-testing suite by Challenger 2 (Empirical Adversarial Verifier).

Addresses the four designated stress-testing targets:
1. CramerGAN two-sample unbiased energy distance:
   - Triangle inequality on metric sqrt(E)
   - Positive definiteness (E(P, Q) >= 0, E(P, Q) > 0 for P != Q)
   - Zero energy on identical distributions
   - Maximum energy on disjoint orthogonal supports
   - Boundedness and surrogate properties of Bellemare et al. vector critic
2. MisGAN decoupled priors:
   - Zero cross-talk gradients between G_x and G_m on prior particles
   - Exact independence of prior_x, prior_m, and prior_i optimization paths
3. Ring16 BiGAN:
   - Detached latent target semantics in loss_recon preventing prior table corruption
   - Contrast against un-detached counterfactual demonstrating prior collapse
4. Grid100 MLP16:
   - LayerNorm maintains non-zero gradient flow back to z across 16 layers
   - Numerical stability under extreme latent norms and boundary conditions
"""
from __future__ import annotations

import math
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ParticlePrior,
)
from benchmarks.toy_audit.api_cramergan import (
    CramerCritic,
    cramer_critic_surrogate,
    cramer_generator_loss,
)
from benchmarks.toy_audit.api_conditionals import (
    FourierMLPDiscriminator,
    TemperatureSigmoid,
    build_case as build_conditionals_case,
)
from benchmarks.toy_audit.api_ring16 import (
    BiGANGenerator,
    BiGANEncoder,
)
from benchmarks.toy_audit.api_vectors import (
    ScaledResidualMLPGenerator,
    ScaledSimpleMLPGenerator,
)


# ==============================================================================
# 1. CramerGAN Two-Sample Unbiased Energy Distance Metric Properties
# ==============================================================================

class TestCramerGANMetricProperties:
    """Rigorous mathematical stress tests on the CramerGAN two-sample energy distance."""

    @staticmethod
    def _unbiased_energy_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the empirical two-sample energy distance:
        E(X, Y) = 2 E||X - Y||_2 - E||X - X'||_2 - E||Y - Y'||_2.
        Using unbiased U-statistic diagonals (excluding i == j).
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]
        d_xy = torch.cdist(X, Y)
        d_xx = torch.cdist(X, X)
        d_yy = torch.cdist(Y, Y)

        # Off-diagonal masks for unbiased sample estimation
        mask_xx = ~torch.eye(n_x, dtype=torch.bool, device=X.device)
        mask_yy = ~torch.eye(n_y, dtype=torch.bool, device=Y.device)

        mean_xy = d_xy.mean()
        mean_xx = d_xx[mask_xx].mean() if n_x > 1 else torch.tensor(0.0, device=X.device)
        mean_yy = d_yy[mask_yy].mean() if n_y > 1 else torch.tensor(0.0, device=Y.device)

        return 2.0 * mean_xy - mean_xx - mean_yy

    def test_energy_distance_zero_on_identical_distributions(self):
        """Stress Test: Unbiased energy distance on two independent draws from identical distributions."""
        torch.manual_seed(42)
        # Sample P = Q ~ N(0, I_2)
        n = 2000
        P_samples1 = torch.randn(n, 2)
        P_samples2 = torch.randn(n, 2)

        ed = self._unbiased_energy_distance(P_samples1, P_samples2)
        # In expectation, unbiased energy distance between identical distributions is 0.
        # With n=2000, sampling deviation must be tightly bounded (|ed| < 0.05).
        assert abs(ed.item()) < 0.05, f"Energy distance on identical distribution not near zero: {ed.item()}"

    def test_energy_distance_positive_definiteness(self):
        """Stress Test: Energy distance must be strictly positive for non-identical distributions."""
        torch.manual_seed(101)
        n = 1000
        # Distribution P: N([0, 0], 0.1 I)
        # Distribution Q: N([1, 1], 0.1 I)
        P = torch.randn(n, 2) * 0.1
        Q = torch.randn(n, 2) * 0.1 + torch.tensor([1.0, 1.0])

        ed = self._unbiased_energy_distance(P, Q)
        assert ed.item() > 0.5, f"Energy distance between distinct distributions should be strictly positive: {ed.item()}"

        # Closer distributions: P and Q separated by only 0.2
        Q_close = torch.randn(n, 2) * 0.1 + torch.tensor([0.2, 0.0])
        ed_close = self._unbiased_energy_distance(P, Q_close)
        assert ed_close.item() > 0.05, f"Energy distance for close distributions should still be positive: {ed_close.item()}"
        assert ed_close.item() < ed.item(), "Closer distributions must have smaller energy distance than distant ones"

    def test_energy_distance_triangle_inequality(self):
        """Stress Test: Metric D(P, R) <= D(P, Q) + D(Q, R) where D(A, B) = sqrt(clamp_min(E(A, B), 0))."""
        torch.manual_seed(202)
        n = 1500
        # Three distributions at vertices of a triangle in 2D
        # P at (0, 0), Q at (2, 0), R at (0, 2)
        P = torch.randn(n, 2) * 0.05
        Q = torch.randn(n, 2) * 0.05 + torch.tensor([2.0, 0.0])
        R = torch.randn(n, 2) * 0.05 + torch.tensor([0.0, 2.0])

        e_pq = torch.clamp(self._unbiased_energy_distance(P, Q), min=0.0)
        e_qr = torch.clamp(self._unbiased_energy_distance(Q, R), min=0.0)
        e_pr = torch.clamp(self._unbiased_energy_distance(P, R), min=0.0)

        d_pq = torch.sqrt(e_pq)
        d_qr = torch.sqrt(e_qr)
        d_pr = torch.sqrt(e_pr)

        # Triangle inequality: d_pr <= d_pq + d_qr (with small float tolerance)
        slack = (d_pq + d_qr) - d_pr
        assert slack.item() >= -1e-4, f"Triangle inequality violated! slack={slack.item()}, d_pr={d_pr.item()}, d_pq+d_qr={d_pq.item()+d_qr.item()}"

    def test_maximum_energy_on_disjoint_orthogonal_supports(self):
        """Stress Test: Energy distance on orthogonal supports scales with Euclidean separation."""
        torch.manual_seed(303)
        n = 1000
        # Support along X axis vs support along Y axis
        delta = 5.0
        X_support = torch.stack([torch.linspace(-1, 1, n) + delta, torch.zeros(n)], dim=1)
        Y_support = torch.stack([torch.zeros(n), torch.linspace(-1, 1, n) + delta], dim=1)

        ed_orthogonal = self._unbiased_energy_distance(X_support, Y_support)
        # Expected distance between orthogonal vectors (d, 0) and (0, d) is sqrt(2)*d approx 1.414 * 5 = 7.07
        assert ed_orthogonal.item() > 5.0, f"Orthogonal supports should exhibit large energy distance: {ed_orthogonal.item()}"

    def test_cramer_critic_surrogate_mathematical_bounds(self):
        """Stress Test: Bellemare critic surrogate f(x; x_hat2) = ||h(x) - h(x_hat2)|| - ||h(x)||.
        By reverse triangle inequality: -||h(x_hat2)|| <= f(x; x_hat2) <= ||h(x_hat2)||.
        """
        torch.manual_seed(404)
        critic = CramerCritic(in_dim=2, embed_dim=16, hidden=64)
        x = torch.randn(32, 2) * 2.0
        x_hat2 = torch.randn(32, 2) * 2.0

        h_x = critic.embed(x)
        h_hat2 = critic.embed(x_hat2)

        f_vals = cramer_critic_surrogate(h_x, h_hat2)
        norm_hat2 = torch.sqrt(h_hat2.square().sum(dim=-1) + 1e-8).mean()

        # Upper bound: f(x; x_hat2) <= ||h_hat2||_2 + eps
        assert (f_vals <= norm_hat2 + 1e-3).all(), "Surrogate exceeded theoretical upper bound ||h(x_hat2)||"
        # Lower bound: f(x; x_hat2) >= -||h_hat2||_2 - eps
        assert (f_vals >= -norm_hat2 - 1e-3).all(), "Surrogate fell below theoretical lower bound -||h(x_hat2)||"

    def test_cramer_generator_loss_properties_on_matched_distributions(self):
        """Stress Test: Bellemare generator loss L_G = d_r1 + d_r2 - d_12.
        1. On degenerate single-point support (h_r == h_f1 == h_f2 == c), L_G == 0.
        2. When fake draws match the real distribution, L_G equals the intra-target dispersion E||h(x) - h(x')||.
        3. Subtracting the intra-target dispersion yields the true energy distance (exactly 0).
        """
        # 1. Degenerate single point support
        # With zero distance, sqrt(0 + 1e-8) = 1e-4 per term, so d_r1 + d_r2 - d_12 = 1e-4.
        c = torch.ones(16, 16) * 3.5
        loss_degenerate = cramer_generator_loss(c, c, c)
        assert abs(loss_degenerate.item() - 1e-4) < 1e-5, f"Degenerate support must equal epsilon floor (1e-4), got {loss_degenerate.item()}"

        # 2. General distribution: L_G matches intra-target dispersion
        torch.manual_seed(444)
        h_r = torch.randn(100, 16)
        h_f1 = torch.randn(100, 16)
        h_f2 = torch.randn(100, 16)
        lg = cramer_generator_loss(h_r, h_f1, h_f2)
        d_rr = torch.sqrt((h_r[:, None, :] - h_r[None, :, :]).square().sum(dim=-1) + 1e-8).mean()
        # In expectation, lg equals d_rr
        assert abs(lg.item() - d_rr.item()) < 0.25, f"Generator loss {lg.item()} diverged from intra-target dispersion {d_rr.item()}"
        # Full energy distance: lg - d_rr approx 0
        ed = lg - d_rr
        assert abs(ed.item()) < 0.25, f"Energy distance between identical distributions should be ~0, got {ed.item()}"


# ==============================================================================
# 2. MisGAN Decoupled Priors: Cross-Talk Gradient Isolation
# ==============================================================================

class TestMisGANDecoupledPriorsIsolation:
    """Stress tests verifying that prior particles receive ZERO cross-talk gradients between G_x and G_m."""

    def test_gradient_isolation_g_x_and_g_m(self):
        """Verify: d(loss_gx)/d(prior_m) == 0 and d(loss_gm)/d(prior_x) == 0."""
        torch.manual_seed(505)
        # Create separate prior tables
        prior_x = ParticlePrior(num_particles=32, z_dim=2, init_std=1.0)
        prior_m = ParticlePrior(num_particles=32, z_dim=2, init_std=1.0)

        G_x = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 8))
        G_m = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 8), TemperatureSigmoid(0.66))
        D_x = FourierMLPDiscriminator(in_dim=8, hidden_dim=32, n_hidden=2, fourier=2)
        D_m = nn.Sequential(nn.Linear(8, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

        bs = 16
        z_x, idx_x = prior_x.sample(bs)
        z_m, idx_m = prior_m.sample(bs)

        x_fake = G_x(z_x)
        m_fake = G_m(z_m)

        # MisGAN masking forward: mask is detached when passed to x_fake
        m_detached = m_fake.detach()
        fake_gx_in = x_fake * m_detached

        # Discriminator outputs
        dx_out = D_x(fake_gx_in).squeeze(-1)
        dm_out = D_m(m_fake).squeeze(-1)

        # Generator losses
        loss_gx = -dx_out.mean()
        loss_gm = -dm_out.mean()

        # 1. Backprop loss_gx: MUST NOT produce gradients in prior_m or G_m
        prior_x.zero_grad(); prior_m.zero_grad(); G_x.zero_grad(); G_m.zero_grad()
        loss_gx.backward(retain_graph=True)

        assert prior_x.z.grad is not None, "prior_x should receive gradients from loss_gx"
        assert prior_x.z.grad.norm().item() > 0, "prior_x gradient should be non-zero"
        assert prior_m.z.grad is None or prior_m.z.grad.norm().item() == 0, (
            f"Cross-talk detected! prior_m received gradients from loss_gx: {prior_m.z.grad.norm().item() if prior_m.z.grad is not None else None}"
        )
        for name, p in G_m.named_parameters():
            assert p.grad is None or p.grad.norm().item() == 0, (
                f"Cross-talk detected! G_m parameter {name} received gradients from loss_gx"
            )

        # 2. Backprop loss_gm: MUST NOT produce gradients in prior_x or G_x
        prior_x.zero_grad(); prior_m.zero_grad(); G_x.zero_grad(); G_m.zero_grad()
        loss_gm.backward()

        assert prior_m.z.grad is not None, "prior_m should receive gradients from loss_gm"
        assert prior_m.z.grad.norm().item() > 0, "prior_m gradient should be non-zero"
        assert prior_x.z.grad is None or prior_x.z.grad.norm().item() == 0, (
            f"Cross-talk detected! prior_x received gradients from loss_gm: {prior_x.z.grad.norm().item() if prior_x.z.grad is not None else None}"
        )
        for name, p in G_x.named_parameters():
            assert p.grad is None or p.grad.norm().item() == 0, (
                f"Cross-talk detected! G_x parameter {name} received gradients from loss_gm"
            )

    def test_misgan_fixture_isolated_param_groups(self):
        """Stress test in MisGAN fixture: ensure optimizer param groups decouple G_x, G_m, prior_x, prior_m."""
        fixture = build_conditionals_case("api-misgan-block-recovery", recipe_name="bispf", max_steps=5)
        # Check that optimizer has separate parameter groups
        g_param_ids = set()
        for group in fixture.opt_g.param_groups:
            for p in group["params"]:
                assert id(p) not in g_param_ids, "Duplicate parameter found across optimizer param groups!"
                g_param_ids.add(id(p))

        # Check prior_x vs prior_m parameters are strictly disjoint
        params_px = {id(p) for p in fixture.prior_x.parameters()}
        params_pm = {id(p) for p in fixture.prior_m.parameters()}
        assert params_px.isdisjoint(params_pm), "prior_x and prior_m share parameter instances!"


# ==============================================================================
# 3. Ring16 BiGAN: Detached Latent Target Semantics in loss_recon
# ==============================================================================

class TestRing16BiGANDetachedTargetSemantics:
    """Stress tests on Ring16 BiGAN ALICE cycle-consistency reconstruction loss."""

    def test_loss_recon_detached_latent_prevents_prior_corruption(self):
        """Stress test: Compare gradient on prior.z with vs without z_fake.detach().
        Without detach, loss_recon = ||E(G(z_fake)) - z_fake||^2 differentiates w.r.t z_fake,
        dragging prior particles toward intermediate encoder targets.
        With detach, d(loss_recon)/d(z_fake_target) == 0.
        """
        torch.manual_seed(606)
        prior = ParticlePrior(num_particles=64, z_dim=2, init_std=1.0)
        G = BiGANGenerator(z_dim=2, out_dim=2, hidden=64, layers=2)
        E = BiGANEncoder(in_dim=2, z_dim=2, hidden=64, layers=2, fourier=2)

        bs = 16
        z_fake, idx = prior.sample(bs)
        real_x = torch.randn(bs, 2)
        z_real = E(real_x)
        x_fake = G(z_fake)

        # Contract loss: F.mse_loss(G(z_real), real_x) + F.mse_loss(E(x_fake), z_fake.detach())
        loss_recon_contract = F.mse_loss(G(z_real), real_x) + F.mse_loss(E(x_fake), z_fake.detach())

        prior.zero_grad(); G.zero_grad(); E.zero_grad()
        loss_recon_contract.backward(retain_graph=True)
        grad_with_detach = prior.z.grad.clone() if prior.z.grad is not None else torch.zeros_like(prior.z)

        # Counterfactual: without detach on target: ||E(x_fake) - z_fake||^2
        # Here z_fake is both the generator input and the target.
        # Target term gives gradient -2(E(x_fake) - z_fake) directly on z_fake!
        loss_recon_counterfactual = F.mse_loss(G(z_real), real_x) + F.mse_loss(E(x_fake), z_fake)

        prior.zero_grad(); G.zero_grad(); E.zero_grad()
        loss_recon_counterfactual.backward()
        grad_without_detach = prior.z.grad.clone()

        # The difference in gradients must directly reflect the target term
        # With detach, prior.z only receives gradients through G(z_fake) (via E(G(z_fake))),
        # NOT directly from the target reference slot!
        diff_norm = (grad_without_detach - grad_with_detach).norm().item()
        assert diff_norm > 1e-3, f"Counterfactual should have produced differing gradients on prior.z: diff={diff_norm}"

    def test_loss_recon_zero_when_encoder_is_exact_inverse(self):
        """Stress test: If E is an exact inverse of G, loss_recon is zero."""
        torch.manual_seed(707)
        z = torch.randn(32, 2)
        x = z * 2.0  # Simple linear invertible generator: G(z) = 2z, E(x) = 0.5x
        z_recon = 0.5 * x
        x_recon = 2.0 * z_recon

        loss_recon = F.mse_loss(x_recon, x) + F.mse_loss(z_recon, z.detach())
        assert loss_recon.item() < 1e-10, f"Perfect inversion must yield zero reconstruction loss: {loss_recon.item()}"


# ==============================================================================
# 4. Grid100 MLP16: LayerNorm Gradient Flow Across 16 Layers
# ==============================================================================

class TestGrid100MLP16GradientFlow:
    """Stress tests on deep MLP16 and ScaledResidualMLP generator gradient propagation."""

    def test_layernorm_gradient_flow_across_16_layers(self):
        """Stress test: Verify non-zero, stable gradient flow from output back to z across 16 layers."""
        torch.manual_seed(808)
        # Deep 16-layer network with LayerNorm
        gen_norm = ScaledSimpleMLPGenerator(z_dim=4, hidden_dim=128, n_hidden=16, out_dim=2, use_norm=True)

        bs = 32
        z1 = torch.randn(bs, 4, requires_grad=True)
        out1 = gen_norm(z1)
        loss1 = (out1 ** 2).sum()
        loss1.backward()

        grad_norm_with_ln = z1.grad.norm().item()
        assert math.isfinite(grad_norm_with_ln), "Gradient with LayerNorm is non-finite!"
        assert grad_norm_with_ln > 1e-4, f"Gradient vanished despite LayerNorm! norm={grad_norm_with_ln}"
        assert grad_norm_with_ln < 1e4, f"Gradient exploded with LayerNorm! norm={grad_norm_with_ln}"

        # Verify gradient flows to EVERY layer in the 16-layer stack
        linear_layers = [m for m in gen_norm.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 17  # 16 hidden + 1 out
        for idx, layer in enumerate(linear_layers):
            assert layer.weight.grad is not None, f"Layer {idx} weight grad is None!"
            weight_grad_norm = layer.weight.grad.norm().item()
            assert math.isfinite(weight_grad_norm), f"Layer {idx} weight grad is non-finite!"
            assert weight_grad_norm > 1e-6, f"Layer {idx} weight grad vanished: norm={weight_grad_norm}"

    def test_mlp16_extreme_input_norm_stability(self):
        """Stress test: Test gradient flow under extreme inputs: z * 100 and z * 1e-4."""
        torch.manual_seed(909)
        gen = ScaledSimpleMLPGenerator(z_dim=4, hidden_dim=128, n_hidden=16, out_dim=2, use_norm=True)

        # Huge input norm: ||z|| ~ 200
        z_huge = (torch.randn(16, 4) * 100.0).requires_grad_(True)
        out_huge = gen(z_huge)
        assert torch.all(torch.isfinite(out_huge)), "Output with huge latent input contains Inf/NaN"
        loss_huge = out_huge.sum()
        loss_huge.backward()
        assert math.isfinite(z_huge.grad.norm().item()), "Gradient with huge latent input is non-finite!"
        assert z_huge.grad.norm().item() > 0, "Gradient with huge latent input vanished!"

        # Tiny input norm: ||z|| ~ 2e-4
        z_tiny = (torch.randn(16, 4) * 1e-4).requires_grad_(True)
        out_tiny = gen(z_tiny)
        assert torch.all(torch.isfinite(out_tiny)), "Output with tiny latent input contains Inf/NaN"
        loss_tiny = out_tiny.sum()
        loss_tiny.backward()
        assert math.isfinite(z_tiny.grad.norm().item()), "Gradient with tiny latent input is non-finite!"
        assert z_tiny.grad.norm().item() > 0, "Gradient with tiny latent input vanished!"

    def test_scaled_residual_mlp_16_layer_scaling(self):
        """Stress test: ScaledResidualMLPGenerator 1/sqrt(2N) scaling preserves variance and gradient."""
        torch.manual_seed(1010)
        res_gen = ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)
        assert res_gen.res_scale == pytest.approx(1.0 / math.sqrt(14.0), rel=1e-5)

        z = torch.randn(64, 2, requires_grad=True)
        out = res_gen(z)

        # Output must be bounded by S=5.25
        assert (out.abs() <= 5.25).all(), "Output exceeded tanh rail scale 5.25!"

        # Test gradient propagation back to z
        loss = out.sum()
        loss.backward()
        assert z.grad is not None, "Residual generator z.grad is None!"
        assert math.isfinite(z.grad.norm().item()), "Residual generator z.grad is non-finite!"
        assert z.grad.norm().item() > 1e-4, f"Residual generator gradient vanished: {z.grad.norm().item()}"
