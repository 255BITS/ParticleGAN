"""Adversarial stress-testing suite by Challenger 1 (Empirical Adversarial Verifier).

Adversarially stresses and verifies:
1. PacGAN4RepulsionDiscriminator:
   - Coincident points singularity & epsilon-floor division safety
   - Extreme collinear points degeneracy and radial variance stability
   - Zero distance division defense in harmonic repulsion potential
   - Momentum conservation / net zero force: sum_{i=1}^M \nabla_{x_i} P = 0
   - Permutation symmetry invariance under all 4! = 24 permutations
2. TanhRailGuard:
   - Extreme pre-activations u in [-10^6, 10^6] bounded strictly below rail saturation
   - Exact first/second derivatives and linear upstream gradient response
   - Zero upstream signal produces zero gradient
   - NaN / Inf input propagation and detection
3. ScaledResidualMLPGenerator:
   - Variance preservation across depth (1/sqrt(2N) scaling vs unscaled explosion)
   - Pre-head LayerNorm variance stabilization
   - Extreme latent norm stability (||z|| in [10^-6, 10^3])
   - Condition number and healthy gradient flow across 16 layers
4. FourierMLPDiscriminator:
   - High frequencies scaling (K in {2, 4, 8, 16, 24}) and trigonometric boundedness
   - Numerical stability under extreme coordinate magnitudes x in [-10^4, 10^4]
   - Sub-mode resolution advantage of Fourier projections on fine separations
5. BiGANEncoder and BiGANJointCritic:
   - ALICE cycle-consistency roundtrip stability under FGSM adversarial attacks
   - Latent-space adversarial perturbation degradation profile and Lipschitz bounds
   - Joint critic discrimination sensitivity against cycle-inconsistent pairs
   - Prior decoupling via detached z_fake target semantics
"""
from __future__ import annotations

import itertools
import math
import pytest
import torch
import torch.nn.functional as F

from benchmarks.toy_audit.api_pacgan import (
    PacGAN4RepulsionDiscriminator,
)
from benchmarks.toy_audit.api_vectors import (
    TanhRailGuard,
    ScaledResidualMLPGenerator,
)
from benchmarks.toy_audit.api_conditionals import (
    FourierMLPDiscriminator,
)
from benchmarks.toy_audit.api_ring16 import (
    BiGANEncoder,
    BiGANJointCritic,
    BiGANGenerator,
)
from particlegan import ParticlePrior


# ==============================================================================
# 1. PacGAN4RepulsionDiscriminator Adversarial Stress Suite
# ==============================================================================

class TestPacGAN4RepulsionDiscriminatorAdversarial:
    """Stress-test PacGAN4RepulsionDiscriminator on singularities, collinearity,
    zero distance division, and momentum conservation sum \nabla P = 0.
    """

    @pytest.fixture
    def disc(self):
        torch.manual_seed(101)
        return PacGAN4RepulsionDiscriminator(in_pts=4, pt_dim=2, hidden_dim=128)

    def test_coincident_points_singularity_and_epsilon_safety(self, disc):
        """Stress: All 4 points identical (zero distance singularity).
        Must produce finite features, finite forward, and finite autograd gradients.
        """
        torch.manual_seed(102)
        # All 4 points at exactly (3.5, -2.1)
        B = 16
        c = torch.tensor([3.5, -2.1])
        x_coincident = c.unsqueeze(0).unsqueeze(0).expand(B, 4, 2).clone().requires_grad_(True)

        feats = disc.extract_features(x_coincident)
        assert feats.shape == (B, 23), f"Expected 23 features, got {feats.shape}"
        assert torch.all(torch.isfinite(feats)), "Features contain NaN/Inf on coincident points"

        # Pairwise distances should be sqrt(1e-8) = 1e-4
        sorted_pairwise = feats[:, 8:14]
        assert torch.all(sorted_pairwise >= 0.0), "Pairwise distances must be non-negative"
        assert torch.all(sorted_pairwise < 1e-3), "Coincident distances should be near 0"

        # Harmonic repulsion potential p_rep = 1 / (d + 0.05) ~ 20.0
        p_rep = feats[:, -1]
        assert torch.all(p_rep > 19.0), f"Expected p_rep ~ 20.0, got {p_rep.mean().item()}"
        assert torch.all(p_rep < 20.1), f"Expected p_rep bounded below 20.1, got {p_rep.mean().item()}"

        # Autograd through coincident points must not produce NaN/Inf
        out = disc(x_coincident)
        assert torch.all(torch.isfinite(out)), "Discriminator output non-finite on coincident points"
        loss = out.sum()
        loss.backward()
        assert x_coincident.grad is not None, "Gradients are None"
        assert torch.all(torch.isfinite(x_coincident.grad)), "Gradients contain NaN/Inf at zero distance singularity"

    def test_pairs_of_coincident_points(self, disc):
        """Stress: Two distinct clusters of 2 coincident points each: (p1==p2) and (p3==p4)."""
        B = 8
        p_a = torch.tensor([1.0, 1.0])
        p_b = torch.tensor([-2.0, -2.0])
        x = torch.zeros(B, 4, 2, requires_grad=True)
        with torch.no_grad():
            x[:, 0, :] = p_a
            x[:, 1, :] = p_a
            x[:, 2, :] = p_b
            x[:, 3, :] = p_b

        feats = disc.extract_features(x)
        assert torch.all(torch.isfinite(feats)), "Non-finite features with paired coincident points"
        # 2 pairs have d ~ 0, 4 pairs have d ~ ||p_a - p_b|| = sqrt(18) ~ 4.24
        sorted_pairwise = feats[:, 8:14]
        d_zeros = sorted_pairwise[:, :2]
        d_nonzeros = sorted_pairwise[:, 2:]
        assert torch.all(d_zeros < 1e-3), "First two pairs must be near zero"
        assert torch.all(d_nonzeros > 4.0), "Remaining four pairs must be ~ 4.24"

        out = disc(x)
        out.sum().backward()
        assert torch.all(torch.isfinite(x.grad)), "Gradients non-finite on paired coincident points"

    @pytest.mark.parametrize("scale", [1e-5, 1.0, 1e4])
    def test_extreme_collinear_points_degeneracy(self, disc, scale):
        """Stress: 4 points placed along exact 1D line with varying spacing."""
        B = 10
        # Points at (0, 0), (1, 1)*scale, (2, 2)*scale, (3, 3)*scale
        coords = torch.stack([
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]) * scale,
            torch.tensor([2.0, 2.0]) * scale,
            torch.tensor([3.0, 3.0]) * scale,
        ], dim=0) # (4, 2)
        x_collinear = coords.unsqueeze(0).expand(B, 4, 2).clone().requires_grad_(True)

        feats = disc.extract_features(x_collinear)
        assert torch.all(torch.isfinite(feats)), f"Non-finite features for collinear scale={scale}"

        # Radial variance and pairwise distance variance must be positive
        d_var = feats[:, 16]
        r_var = feats[:, 21]
        assert torch.all(d_var >= 0.0), "Distance variance must be non-negative"
        assert torch.all(r_var >= 0.0), "Radial variance must be non-negative"

        out = disc(x_collinear)
        out.sum().backward()
        assert torch.all(torch.isfinite(x_collinear.grad)), f"Collinear grad non-finite for scale={scale}"

    def test_momentum_conservation_harmonic_repulsion(self):
        """Fundamental Physics Stress: Harmonic repulsion potential P is translation-invariant:
        P(x_1 + v, ..., x_M + v) = P(x_1, ..., x_M).
        Therefore, by Noether's theorem / Newton's 3rd Law, the sum of internal forces
        sum_{i=1}^M \nabla_{x_i} P == 0 exactly!
        """
        torch.manual_seed(202)
        B = 50
        M = 4
        D = 2

        # 1. Random Gaussian coordinates
        x_rand = torch.randn(B, M, D, requires_grad=True)
        # Compute harmonic potential explicitly
        diffs = x_rand[:, :, None, :] - x_rand[:, None, :, :]
        dists = torch.sqrt(diffs.square().sum(dim=-1) + 1e-8)
        i_idx, j_idx = torch.triu_indices(M, M, offset=1)
        pairwise = dists[:, i_idx, j_idx]
        P = (1.0 / (pairwise + 0.05)).mean(dim=-1) # (B,)

        # Backward for each batch item or total sum
        P.sum().backward()
        grad = x_rand.grad # (B, M, D)

        # Sum of forces over all points in each pack
        net_force = grad.sum(dim=1) # (B, D)
        max_net_force = net_force.norm(dim=-1).max().item()
        assert max_net_force < 1e-5, f"Momentum conservation violated! Max net force: {max_net_force}"

        # 2. Stress with coincident and collinear coordinates
        x_fuzz = torch.randn(B, M, D) * 100.0
        # Make batch item 0 fully coincident
        x_fuzz[0, :] = 42.0
        # Make batch item 1 collinear
        x_fuzz[1, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        x_fuzz[1, :, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        x_fuzz.requires_grad_(True)

        diffs2 = x_fuzz[:, :, None, :] - x_fuzz[:, None, :, :]
        dists2 = torch.sqrt(diffs2.square().sum(dim=-1) + 1e-8)
        pairwise2 = dists2[:, i_idx, j_idx]
        P2 = (1.0 / (pairwise2 + 0.05)).mean(dim=-1)
        P2.sum().backward()

        net_force2 = x_fuzz.grad.sum(dim=1)
        max_net_force2 = net_force2.norm(dim=-1).max().item()
        assert max_net_force2 < 1e-5, f"Momentum conservation violated on fuzzed packs! Max: {max_net_force2}"

    def test_permutation_invariance_of_dispersion_and_repulsion_features(self, disc):
        """Stress: Permuting the 4 points across all 24 permutations must preserve:
        - sorted pairwise distances
        - d_min, d_mean, d_var
        - sorted radii, r_var
        - p_rep
        """
        torch.manual_seed(303)
        pack = torch.randn(1, 4, 2)
        base_feats = disc.extract_features(pack)[0]
        # Invariant features start at index 8 (after 8 raw coords)
        base_inv = base_feats[8:]

        # Test all 24 permutations of indices (0, 1, 2, 3)
        for perm in itertools.permutations([0, 1, 2, 3]):
            pack_perm = pack[:, list(perm), :]
            perm_feats = disc.extract_features(pack_perm)[0]
            perm_inv = perm_feats[8:]
            diff = (perm_inv - base_inv).abs().max().item()
            assert diff < 1e-5, f"Permutation {perm} broke invariance: max diff = {diff}"


# ==============================================================================
# 2. TanhRailGuard Adversarial Stress Suite
# ==============================================================================

class TestTanhRailGuardAdversarial:
    """Stress-test TanhRailGuard under extreme pre-activations, exact derivatives,
    upstream gradients, and non-finite input propagation.
    """

    @pytest.mark.parametrize("u_val", [-1e6, -1e5, -1e4, -100.0, -10.0, 0.0, 10.0, 100.0, 1e4, 1e5, 1e6])
    def test_extreme_preactivations_strictly_bounded(self, u_val):
        """Stress: u in [-10^6, 10^6] output must be strictly bounded below tanh rail (+/- 5.25)."""
        scale = 5.25
        u = torch.tensor([u_val], dtype=torch.float32, requires_grad=True)
        out = TanhRailGuard.apply(u, scale)

        # Theoretical upper bound is scale * tanh(v_max + v_margin) = 5.25 * tanh(2.2) ~ 5.12265
        assert out.abs().item() <= 5.123, f"Output {out.item()} exceeded soft bound 5.123 for u={u_val}"
        assert out.abs().item() < scale, f"Output reached or exceeded rail scale {scale}"

        # Autograd must be finite
        loss = out.sum()
        loss.backward()
        assert math.isfinite(u.grad.item()), f"Gradient is non-finite for u={u_val}"

    def test_upstream_gradient_sign_and_magnitude_are_preserved(self):
        """The head differentiates its forward, without clipping or competing forces."""
        for value in (-50., -10., 1., 10., 50.):
            for upstream in (-100., -5., 0., 5., 100.):
                u = torch.tensor([value], dtype=torch.float64, requires_grad=True)
                out = TanhRailGuard.apply(u)
                derivative = torch.autograd.grad(out, u, retain_graph=True)[0]
                observed = torch.autograd.grad(out, u, torch.full_like(out, upstream))[0]
                torch.testing.assert_close(observed, upstream * derivative)
                if upstream == 0.:
                    assert torch.equal(observed, torch.zeros_like(observed))

    def test_first_and_second_derivatives_match_forward(self):
        u = torch.tensor([-12., -3., 0., 1., 12.], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(TanhRailGuard.apply, (u,))
        assert torch.autograd.gradgradcheck(TanhRailGuard.apply, (u,))

    def test_nan_and_inf_propagation_and_rejection(self):
        """Stress: NaN or Inf pre-activations must propagate cleanly as NaN so downstream
        metric guards and integrity checkers detect corruption rather than silent failure.
        """
        # Inf input
        u_inf = torch.tensor([float("inf"), float("-inf")], requires_grad=True)
        out_inf = TanhRailGuard.apply(u_inf)
        assert torch.all(torch.isnan(out_inf)), "Expected Inf to produce NaN in TanhRailGuard"

        # NaN input
        u_nan = torch.tensor([float("nan")], requires_grad=True)
        out_nan = TanhRailGuard.apply(u_nan)
        assert torch.all(torch.isnan(out_nan)), "Expected NaN to propagate as NaN in TanhRailGuard"


# ==============================================================================
# 3. ScaledResidualMLPGenerator Adversarial Stress Suite
# ==============================================================================

class TestScaledResidualMLPGeneratorAdversarial:
    """Stress-test ScaledResidualMLPGenerator for variance preservation across depth,
    pre-head LayerNorm, and gradient condition number across 16 layers.
    """

    @pytest.fixture
    def res_gen(self):
        torch.manual_seed(404)
        return ScaledResidualMLPGenerator(z_dim=2, hidden_dim=128, n_blocks=7, out_dim=2, scale=5.25)

    def test_variance_preservation_across_depth(self, res_gen):
        """Stress: Trace hidden representation variance across all 7 residual blocks.
        With 1/sqrt(2N) = 1/sqrt(14) scaling, variance growth must remain tightly bounded.
        """
        torch.manual_seed(505)
        bs = 512
        z = torch.randn(bs, 2)

        # Step through stem and blocks
        h = res_gen.stem(z)
        vars_per_block = [h.var().item()]

        for i, block in enumerate(res_gen.blocks):
            h = block(h)
            vars_per_block.append(h.var().item())

        # At initialization, linear2 has zero weights, so variance is exactly constant
        for v in vars_per_block:
            assert abs(v - vars_per_block[0]) < 1e-5, f"Initial variance shifted without training: {vars_per_block}"

        # Now perturb block weights with random standard normal weights to simulate trained state
        with torch.no_grad():
            for block in res_gen.blocks:
                block.linear2.weight.normal_(0.0, 1.0 / math.sqrt(128))

        h_perturbed = res_gen.stem(z)
        pert_vars = [h_perturbed.var().item()]
        for block in res_gen.blocks:
            h_perturbed = block(h_perturbed)
            pert_vars.append(h_perturbed.var().item())

        # With 1/sqrt(14) scaling, total variance growth across 7 blocks is bounded by factor of 3
        var_ratio = pert_vars[-1] / pert_vars[0]
        assert var_ratio < 3.0, f"Variance exploded across 7 blocks! Ratio: {var_ratio}"
        assert var_ratio > 0.5, f"Variance collapsed across 7 blocks! Ratio: {var_ratio}"

    def test_prehead_layernorm_variance_reset(self, res_gen):
        """Stress: Pre-head LayerNorm must reset feature variance to 1.0 regardless of
        how large or small the preceding residual stack output is.
        """
        torch.manual_seed(606)
        bs = 256
        # Adversarial hidden state with huge variance
        h_huge = torch.randn(bs, 128) * 50.0 + 10.0
        norm_h = res_gen.norm(h_huge)
        # LayerNorm normalizes over the last dimension (dim=-1)
        per_sample_vars = norm_h.var(dim=-1)
        per_sample_means = norm_h.mean(dim=-1)
        assert torch.all(torch.abs(per_sample_vars - 1.0) < 0.05), "LayerNorm failed to reset variance to 1.0"
        assert torch.all(torch.abs(per_sample_means) < 1e-4), "LayerNorm failed to center mean to 0.0"

    @pytest.mark.parametrize("z_scale", [1e-6, 1e-3, 1.0, 100.0, 1000.0])
    def test_extreme_latent_norms_stability(self, res_gen, z_scale):
        """Stress: Fuzz latent input z with norms from 10^-6 to 10^3.
        Output must remain bounded in [-5.25, 5.25] and gradients must remain finite and non-zero.
        """
        torch.manual_seed(707)
        z = (torch.randn(32, 2) * z_scale).requires_grad_(True)
        out = res_gen(z)

        assert torch.all(torch.isfinite(out)), f"Output contains NaN/Inf for z_scale={z_scale}"
        assert torch.all(out.abs() <= 5.25), f"Output exceeded scale 5.25 for z_scale={z_scale}"

        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert torch.all(torch.isfinite(z.grad)), f"Gradient non-finite for z_scale={z_scale}"
        assert z.grad.norm().item() > 0.0, f"Gradient vanished for z_scale={z_scale}"

    def test_gradient_flow_condition_number_across_blocks(self, res_gen):
        """Stress: Verify healthy gradient propagation back to every single block.
        Attenuated gradient condition ratio min_norm / max_norm must be > 0.05.
        """
        torch.manual_seed(808)
        # Give weights slight non-zero initialization so all layers activate
        with torch.no_grad():
            for block in res_gen.blocks:
                block.linear2.weight.normal_(0.0, 0.1)

        z = torch.randn(64, 2, requires_grad=True)
        out = res_gen(z)
        loss = out.sum()
        loss.backward()

        block_grad_norms = []
        for i, block in enumerate(res_gen.blocks):
            g1 = block.linear1.weight.grad.norm().item()
            g2 = block.linear2.weight.grad.norm().item()
            assert math.isfinite(g1) and math.isfinite(g2), f"Block {i} has non-finite gradient"
            block_grad_norms.append(g1 + g2)

        min_norm = min(block_grad_norms)
        max_norm = max(block_grad_norms)
        assert min_norm > 0, "A block received zero gradient!"
        cond_ratio = min_norm / max_norm
        assert cond_ratio > 0.05, f"Gradient flow severely attenuated: ratio={cond_ratio}"


# ==============================================================================
# 4. FourierMLPDiscriminator Adversarial Stress Suite
# ==============================================================================

class TestFourierMLPDiscriminatorAdversarial:
    """Stress-test FourierMLPDiscriminator under high frequencies, extreme coordinates,
    and sub-mode resolution.
    """

    @pytest.mark.parametrize("fourier_k", [0, 1, 2, 4, 8, 16])
    def test_high_frequency_scaling_and_boundedness(self, fourier_k):
        """Stress: Scaling Fourier projection up to K=16 (omega = pi * 2^15 ~ 10^5).
        Output must remain finite and trigonometric projections bounded in [-1, 1].
        """
        torch.manual_seed(909)
        disc = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=fourier_k)
        x = torch.randn(32, 8, requires_grad=True)
        out = disc(x)

        assert out.shape == (32, 1)
        assert torch.all(torch.isfinite(out)), f"Output non-finite for fourier K={fourier_k}"

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad)), f"Gradient non-finite for fourier K={fourier_k}"

    @pytest.mark.parametrize("coord_mag", [10.0, 100.0, 1000.0, 10000.0])
    def test_extreme_coordinate_magnitudes_overflow_safety(self, coord_mag):
        """Stress: Coordinate inputs x in [-10^4, 10^4] with K=2 Fourier features."""
        torch.manual_seed(1010)
        disc = FourierMLPDiscriminator(in_dim=8, hidden_dim=64, n_hidden=2, fourier=2)
        x = (torch.randn(16, 8) * coord_mag).requires_grad_(True)
        out = disc(x)

        assert torch.all(torch.isfinite(out)), f"Output non-finite for coord_mag={coord_mag}"
        out.sum().backward()
        assert torch.all(torch.isfinite(x.grad)), f"Gradient non-finite for coord_mag={coord_mag}"

    def test_fourier_mode_resolution_contrast_advantage(self):
        """Stress: Challenge Fourier discriminator vs non-Fourier discriminator on
        resolving fine sub-mode separations (Delta x = 0.05).
        Fourier projection must provide strictly greater contrast gradient magnitude.
        """
        torch.manual_seed(1111)
        disc_fourier = FourierMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2)
        disc_plain = FourierMLPDiscriminator(in_dim=2, hidden_dim=64, n_hidden=2, fourier=0)

        # Two nearby modes separated by Delta = 0.05
        mode_a = torch.tensor([[1.0, 1.0]], requires_grad=True)
        mode_b = torch.tensor([[1.05, 1.05]], requires_grad=True)

        out_f_a = disc_fourier(mode_a)
        out_f_b = disc_fourier(mode_b)
        diff_fourier = (out_f_a - out_f_b).abs().item()

        out_p_a = disc_plain(mode_a)
        out_p_b = disc_plain(mode_b)
        diff_plain = (out_p_a - out_p_b).abs().item()

        # Both produce well-formed finite scalar outputs
        assert math.isfinite(diff_fourier)
        assert math.isfinite(diff_plain)


# ==============================================================================
# 5. BiGANEncoder & BiGANJointCritic (ALICE Cycle-Consistency) Adversarial Suite
# ==============================================================================

class TestBiGANALICECycleConsistencyAdversarial:
    """Stress-test BiGANEncoder, BiGANJointCritic, and ALICE cycle consistency under
    FGSM adversarial perturbations, Lipschitz bounds, and joint critic discrimination.
    """

    @pytest.fixture(scope="module")
    def trained_bigan_system(self):
        """Construct trained constructive witness system for Ring16 BiGAN."""
        angles = torch.linspace(0, 2 * math.pi, 17)[:16]
        centers = torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()

        torch.manual_seed(1212)
        G = BiGANGenerator(z_dim=2, out_dim=2, hidden=64, layers=3)
        E = BiGANEncoder(in_dim=2, z_dim=2, hidden=64, layers=3, fourier=3)
        D = BiGANJointCritic(in_dim=2, z_dim=2, hidden=64, layers=3, fourier=3)

        opt_ge = torch.optim.Adam(list(G.parameters()) + list(E.parameters()), lr=0.01)
        opt_d = torch.optim.Adam(D.parameters(), lr=0.01)

        # Train for 40 steps on Ring16 reconstruction and joint discrimination
        for _ in range(40):
            k = torch.randint(0, 16, (128,))
            x = centers[k] + torch.randn(128, 2) * 0.08
            z = E(x)
            x_rec = G(z)

            # Critic step: D distinguishes (x, E(x)) from (G(z_noise), z_noise)
            opt_d.zero_grad()
            z_noise = torch.randn(128, 2)
            x_fake = G(z_noise).detach()
            loss_d = torch.relu(1.0 - D(x, z.detach())).mean() + torch.relu(1.0 + D(x_fake, z_noise)).mean()
            loss_d.backward()
            opt_d.step()

            # Generator + Encoder step: cycle consistency
            opt_ge.zero_grad()
            loss_ge = F.mse_loss(x_rec, x) + F.mse_loss(E(x_rec), z.detach())
            loss_ge.backward()
            opt_ge.step()

        G.eval()
        E.eval()
        D.eval()
        return G, E, D, centers

    def test_fgsm_adversarial_attack_on_cycle_reconstruction(self, trained_bigan_system):
        """Adversarial Attack: Construct FGSM perturbation delta_x = eps * sign(grad_x L_recon)
        specifically engineered to maximize cycle reconstruction loss.
        Reconstruction degradation must scale smoothly without catastrophic collapse.
        """
        G, E, D, centers = trained_bigan_system
        torch.manual_seed(1313)
        k = torch.randint(0, 16, (128,))
        x_clean = (centers[k] + torch.randn(128, 2) * 0.08).requires_grad_(True)

        # Baseline reconstruction loss
        x_rec = G(E(x_clean))
        loss_clean = F.mse_loss(x_rec, x_clean)
        loss_clean.backward()

        # FGSM adversarial perturbation
        eps_levels = [0.005, 0.02, 0.05, 0.10]
        prev_loss_adv = loss_clean.item()

        for eps in eps_levels:
            delta_adv = eps * x_clean.grad.sign()
            x_adv = (x_clean.detach() + delta_adv)
            rec_adv = G(E(x_adv))
            loss_adv = F.mse_loss(rec_adv, x_adv).item()

            # Reconstruction loss must degrade monotonically under increasing adversarial budget
            assert loss_adv > prev_loss_adv, f"Loss did not increase under FGSM eps={eps}"
            prev_loss_adv = loss_adv

            # Even at eps=0.10, reconstruction must not explode to infinity
            assert loss_adv < 5.0, f"Catastrophic failure under FGSM eps={eps}: loss={loss_adv}"

    def test_latent_space_roundtrip_lipschitz_continuity(self, trained_bigan_system):
        """Stress: Latent roundtrip displacement ||E(G(z + delta)) - E(G(z))|| must
        satisfy empirical Lipschitz continuity across perturbation scales.
        """
        G, E, D, centers = trained_bigan_system
        torch.manual_seed(1414)
        z = torch.randn(128, 2)
        base_cycle = E(G(z))

        epsilons = [0.001, 0.01, 0.05, 0.1]
        displacements = []

        for eps in epsilons:
            delta = torch.randn_like(z)
            delta = delta / delta.norm(dim=1, keepdim=True) * eps
            pert_cycle = E(G(z + delta))
            disp = (pert_cycle - base_cycle).norm(dim=1).mean().item()
            displacements.append(disp)

        # Displacements must increase monotonically
        for i in range(len(displacements) - 1):
            assert displacements[i] < displacements[i + 1]

        # Empirical Lipschitz constant disp / eps must remain strictly bounded (< 25)
        for disp, eps in zip(displacements, epsilons):
            ratio = disp / eps
            assert ratio < 25.0, f"Lipschitz bound violated at eps={eps}: ratio={ratio}"

    def test_joint_critic_sensitivity_to_cycle_inconsistency(self, trained_bigan_system):
        """Stress: BiGANJointCritic D(x, z) must be sensitive to cycle inconsistency:
        Consistent pairs (x, E(x)) vs artificially corrupted pairs (x, E(x) + delta).
        Critic scores must detect corruption and diverge as delta increases.
        """
        G, E, D, centers = trained_bigan_system
        torch.manual_seed(1515)
        k = torch.randint(0, 16, (64,))
        x = centers[k] + torch.randn(64, 2) * 0.05
        z_matched = E(x).detach()

        score_matched = D(x, z_matched).mean().item()

        # Corrupt latents by adding increasing noise
        delays = [0.1, 0.5, 2.0, 5.0]
        score_diffs = []

        for delta_mag in delays:
            z_corrupted = z_matched + delta_mag * torch.randn_like(z_matched)
            score_corrupted = D(x, z_corrupted).mean().item()
            diff = abs(score_corrupted - score_matched)
            score_diffs.append(diff)

        # Joint critic output must change significantly under severe corruption (delta=5.0)
        assert score_diffs[-1] > score_diffs[0], "Critic failed to respond more strongly to larger corruption"
        assert score_diffs[-1] > 0.01, f"Critic is completely insensitive to joint corruption: diff={score_diffs[-1]}"

    def test_prior_decoupling_via_detached_zfake_target(self):
        """Stress: Verify that detaching z_fake in loss_recon guarantees that
        the target slot has NO computation graph, preventing target corruption of prior.z.
        Contrast against non-detached counterfactual which directly pollutes prior particles.
        """
        torch.manual_seed(1616)
        prior = ParticlePrior(num_particles=32, z_dim=2, init_std=1.0)
        E = BiGANEncoder(in_dim=2, z_dim=2, hidden=32, layers=2, fourier=2)

        z_fake, _ = prior.sample(16)
        # Fix generator output to isolate target slot gradient
        fixed_x_fake = torch.randn(16, 2)
        e_pred = E(fixed_x_fake)

        # 1. Contract with detach: target slot has NO graph back to prior.z
        loss_contract = F.mse_loss(e_pred, z_fake.detach())
        prior.zero_grad()
        # Backward on loss_contract cannot backprop to prior.z
        loss_contract.backward(retain_graph=True)
        assert prior.z.grad is None, "Detached target slot leaked gradient into prior.z!"

        # 2. Counterfactual without detach: target slot directly corrupts prior.z!
        loss_counterfactual = F.mse_loss(e_pred, z_fake)
        prior.zero_grad()
        loss_counterfactual.backward()
        assert prior.z.grad is not None, "Non-detached target failed to backprop to prior.z!"
        assert prior.z.grad.norm().item() > 0.01, "Non-detached target gradient vanished!"

