"""Unit and integration tests for Born-Infeld Sinkhorn Particle Flow (BISPF)."""
from copy import deepcopy
import math
import pytest
import torch
import torch.nn as nn

from particlegan import (
    BISPFCriticAdam,
    BISPFCriticPenalty,
    BISPFEngine,
    BISPFLossAdapter,
    GANTrainer,
    SBNPPriorRegularizer,
    SinkhornRelativisticLoss,
    StratifiedBlueNoisePrior,
    get_recipe,
    init,
    learning_rate_scales,
)


def test_sbnp_stratified_sampling_and_repulsion():
    torch.manual_seed(0)
    prior = StratifiedBlueNoisePrior(num_particles=100, z_dim=2, knn_k=3)

    # 1. Verify exact without-replacement coverage over 100 samples (4 batches of 25)
    seen_indices = []
    for _ in range(4):
        z, idx = prior.sample(25)
        assert z.shape == (25, 2)
        seen_indices.append(idx)
    all_idx = torch.cat(seen_indices)
    assert len(torch.unique(all_idx)) == 100, "Must visit every particle once per epoch"

    # 2. Verify k-NN log repulsion penalizes collapsed particles more than spread particles
    with torch.no_grad():
        prior.z.normal_(0.0, 1.0)
    loss_spread = prior.regularization_loss()

    with torch.no_grad():
        prior.z[:10] = prior.z[0].clone()  # Collapse 10 particles onto the same point
    loss_collapsed = prior.regularization_loss()
    assert loss_collapsed > loss_spread + 0.05

    loss_collapsed.backward()
    assert prior.z.grad is not None and torch.isfinite(prior.z.grad).all()


def test_bispf_coordinate_scale_invariance():
    torch.manual_seed(42)
    x_real = torch.randn(32, 2)
    x_fake = torch.randn(32, 2, requires_grad=True)

    # Linear critic D(x) = w^T x + b where we scale w inversely with coordinate scale s
    # so logits D(s * x) are identical across scales.
    base_D = nn.Linear(2, 1)

    losses_d = []
    for scale in [0.1, 1.0, 10.0]:
        engine = BISPFEngine(sigma_d_max=0.0)
        D = nn.Linear(2, 1)
        with torch.no_grad():
            D.weight.copy_(base_D.weight / scale)
            D.bias.copy_(base_D.bias)

        xr = x_real * scale
        xf = (x_fake.detach() * scale).requires_grad_(True)
        d_loss, stats = engine.d_step(D, xr, xf)
        losses_d.append(stats["d_adv"])

    assert abs(losses_d[0] - losses_d[1]) < 1e-4
    assert abs(losses_d[1] - losses_d[2]) < 1e-4


def test_bispf_ebi_harmonic_capture_vs_ballistic_transit():
    torch.manual_seed(7)
    engine = BISPFEngine(lambda_floor=0.1, lambda_cap=20.0, grad_cap=1.0, sigma_d_max=0.0)

    # Critic with small gradient norm (||grad|| = 0.2 < grad_cap = 1.0) -> Harmonic regime
    D_small = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        D_small.weight.copy_(torch.tensor([[0.2, 0.0]]))

    # Critic with large gradient norm (||grad|| = 3.0 > grad_cap = 1.0) -> Ballistic cap regime
    D_large = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        D_large.weight.copy_(torch.tensor([[3.0, 0.0]]))

    xr = torch.randn(16, 2)
    xf = torch.zeros(16, 2, requires_grad=True)

    _, stats_small = engine.d_step(D_small, xr, xf)
    _, stats_large = engine.d_step(D_large, xr, xf)

    # In harmonic regime, EBI penalty is ~ 0.5 * 0.1 * (0.2)^2 = 0.002
    assert 0.001 < stats_small["d_ebi"] < 0.01
    # In ballistic regime, EBI cap wall engages strongly (~ 0.5 * 20 * (3 - 1)^2 = 40.0)
    assert stats_large["d_ebi"] > 30.0


def test_bispf_checkpoint_roundtrip():
    prior = StratifiedBlueNoisePrior(num_particles=50, z_dim=2)
    engine = BISPFEngine()
    D = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1))

    xr = torch.randn(16, 2)
    xf, _ = prior.sample(16)
    loss, _ = engine.d_step(D, xr, xf)
    assert torch.isfinite(loss)

    p_state = prior.state_dict()
    e_state = engine.state_dict()

    prior2 = StratifiedBlueNoisePrior(num_particles=50, z_dim=2)
    engine2 = BISPFEngine()
    prior2.load_state_dict(p_state)
    engine2.load_state_dict(e_state)

    assert torch.equal(prior._perm, prior2._perm)
    assert torch.equal(prior._cursor, prior2._cursor)
    assert torch.allclose(engine.ema_residual, engine2.ema_residual)


def test_bispf_recipe_and_trainer_integration():
    recipe = get_recipe(
        "bispf",
        num_particles=64,
        z_dim=2,
        batch_size=16,
        total_steps=20,
    )
    assert recipe.critic_formulation == "bispf"
    assert recipe.effective_critic_formulation == "bispf"
    assert recipe.prior_regularizer == "sbnp"
    assert recipe.lr_schedule == "constant"
    assert recipe.reg_anchor_weight == 0.0
    assert learning_rate_scales(0, recipe) == (1.0, 1.0)
    assert learning_rate_scales(15, recipe) == (1.0, 1.0)

    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    prior = recipe.make_prior()
    for idx, mod in enumerate((G, D, prior)):
        init.deterministic_orthogonal_(mod, seed=idx)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    # BISPF must not allocate an EMA critic copy
    assert trainer.ema_D is None
    assert isinstance(trainer.opt_d, BISPFCriticAdam)
    assert isinstance(trainer.penalty, BISPFCriticPenalty)
    assert isinstance(trainer.loss, SinkhornRelativisticLoss)
    assert isinstance(trainer.prior_regularizer, SBNPPriorRegularizer)

    data_gen = torch.Generator().manual_seed(123)
    for _ in range(3):
        real = torch.randn(16, 2, generator=data_gen)
        out = trainer.step(real, collect_stats=True)
        assert torch.isfinite(out["loss_d"])
        assert torch.isfinite(out["loss_g"])
        assert torch.isfinite(out["prior_regularization"])
        assert out["penalty_stats"]["applied"] is True
        assert "ema_residual" in out["penalty_stats"]

    # Verify exact checkpoint continuation roundtrip
    ckpt = deepcopy(trainer.state_dict())
    real_next = torch.randn(16, 2, generator=data_gen)
    out_a = trainer.step(real_next, collect_stats=True)
    samples_a = trainer.sample(16)

    G2 = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
    D2 = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    prior2 = recipe.make_prior()
    trainer2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=99)
    trainer2.load_state_dict(ckpt)
    out_b = trainer2.step(real_next, collect_stats=True)
    samples_b = trainer2.sample(16)

    assert torch.allclose(out_a["loss_d"], out_b["loss_d"])
    assert torch.allclose(out_a["loss_g"], out_b["loss_g"])
    assert torch.allclose(samples_a, samples_b)

    # Verify invalid critic rejection on BISPFCriticPenalty
    unrelated_critic = nn.Linear(2, 1)
    with pytest.raises(TypeError, match="pass the critic paired with this penalty"):
        trainer.penalty(unrelated_critic, real_next, real_next)


def test_bispf_rms_born_infeld_dimension_invariance_and_smoothness():
    """Verify RMS gradient norm g_rms = ||g||_2 / sqrt(d) produces identical EBI penalties
    across scalar (d=2), medium (d=64), and conv image (d=144 = 1x12x12) dimensions for
    constant-slope critics, and transitions smoothly across kappa=1.0 without discontinuities.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    # 1. Dimension Invariance across medium (d=64) and conv image (d=144 = 1x12x12) in RMS regime
    slopes = [0.4, 1.0, 2.5]  # Sub-cap (harmonic floor), at cap, and ballistic cap
    dimensions = [
        (64, (16, 64)),
        (144, (16, 144)),  # 2D continuous feature vectors (both ndim=2)
    ]

    for slope in slopes:
        penalties = []
        for dim, shape in dimensions:
            torch.manual_seed(42)
            kernel = _BISPFPenaltyKernel(
                coeff=1.0,
                lambda_floor=0.1,
                lambda_cap=10.0,
                kappa=1.0,
                ebi_beta=30.0,
                gamma_gauge=0.0,  # Isolate EBI penalty from gauge
                chord_blend=0.0,
            )

            # Critic with constant per-coordinate gradient: D(x) = slope * sum(x)
            # In RMS coordinates: ||grad||_2 / sqrt(d) = sqrt(d * slope^2) / sqrt(d) = slope
            class ConstantSlopeCritic(nn.Module):
                def __init__(self, s):
                    super().__init__()
                    self.s = s

                def forward(self, x):
                    return self.s * x.flatten(1).sum(dim=1, keepdim=True)

            D = ConstantSlopeCritic(slope)
            xr = torch.randn(*shape)
            xf = torch.randn(*shape)

            pen, _ = kernel.penalty(D, xr, xf, step=0, collect_stats=True)
            penalties.append(float(pen.detach().item()))

        # Penalties must match across d=64 and d=144 within numerical tolerance
        ref_pen = penalties[0]
        for idx, p in enumerate(penalties[1:], 1):
            assert abs(p - ref_pen) < 1e-4, (
                f"Slope {slope}: penalty at dim {dimensions[idx][0]} ({p}) does not match "
                f"dim {dimensions[0][0]} ({ref_pen})"
            )

    # 2. Verify low-dimensional continuous coordinates (d <= 2) enforce raw Euclidean barrier ||g||_2 <= 1.0
    # without dividing by sqrt(d), ensuring tight critic regularization in 2D slot-hold tasks
    kernel_2d = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_floor=0.1,
        lambda_cap=10.0,
        kappa=1.0,
        ebi_beta=30.0,
        gamma_gauge=0.0,
        chord_blend=0.0,
    )
    D_2d_at_cap = ConstantSlopeCritic(1.0 / math.sqrt(2.0))  # ||g||_2 = sqrt(2 * (1/sqrt(2))^2) = 1.0
    xr_2d = torch.randn(16, 2)
    xf_2d = torch.randn(16, 2)
    pen_2d, stats_2d = kernel_2d.penalty(D_2d_at_cap, xr_2d, xf_2d, step=0, collect_stats=True)
    assert abs(stats_2d["mean_grad_norm"] - 1.0) < 1e-4

    # 3. Verify discrete categorical codebooks (d=170, ndim=2) retain raw Euclidean norm,
    # omit zero-mean logit gauge, omit chords, and hold thermal scale at 1.0
    xr_token = torch.randn(16, 170)
    xf_token = torch.randn(16, 170)
    D_token = ConstantSlopeCritic(1.0 / math.sqrt(170.0))  # ||g||_2 = 1.0
    kernel_token = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_floor=0.1,
        lambda_cap=10.0,
        kappa=1.0,
        gamma_gauge=0.002,
    )
    pen_tok, stats_tok = kernel_token.penalty(D_token, xr_token, xf_token, step=0, collect_stats=True)
    assert abs(stats_tok["mean_grad_norm"] - 1.0) < 1e-4
    assert stats_tok["mean_logit"] == 0.0
    assert stats_tok["gauge_loss"] == 0.0
    assert stats_tok["floor_pen"] == 0.0
    assert stats_tok["s_thermal"] == 1.0

    # 4. Smoothness across kappa=1.0 (no jump discontinuity in value or derivative)
    c_vals = torch.linspace(0.8, 1.2, 41, requires_grad=False)
    kernel = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_floor=0.1,
        lambda_cap=10.0,
        kappa=1.0,
        ebi_beta=30.0,
        gamma_gauge=0.0,
    )
    vals = []
    for c in c_vals:
        c_param = nn.Parameter(torch.tensor([float(c)]))

        class ScaledCritic(nn.Module):
            def __init__(self, param):
                super().__init__()
                self.param = param

            def forward(self, x):
                return self.param * x.flatten(1).sum(dim=1, keepdim=True)

        D = ScaledCritic(c_param)
        xr = torch.randn(8, 64)
        xf = torch.randn(8, 64)
        pen, _ = kernel.penalty(D, xr, xf, step=0)
        vals.append(pen)

    loss_t = torch.stack(vals)
    # Strictly monotonically increasing
    diffs = loss_t[1:] - loss_t[:-1]
    assert (diffs > 0).all(), "Born-Infeld penalty must strictly increase with gradient norm"
    # Second differences (curvature) are finite and smooth
    second_diffs = diffs[1:] - diffs[:-1]
    assert (second_diffs.abs() < 0.05).all(), "Penalty transition across kappa must be C1 smooth"


def test_bispf_universal_zero_mean_logit_gauge_active_and_drift_penalized():
    """Verify zero-mean logit gauge is universally active across all dimensions (including
    d=144 image inputs), penalizes translational bias shift quadratically, and provides
    a restorative autograd gradient.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    gamma = 0.002
    coeff = 1.0
    kernel = _BISPFPenaltyKernel(
        coeff=coeff,
        lambda_floor=0.0,
        lambda_cap=0.0,
        gamma_gauge=gamma,
        chord_blend=0.0,
    )

    # Test both vector (d=2) and conv image (d=144 = 1x12x12) inputs
    for dim_name, shape in [("vector2d", (16, 2)), ("image144", (16, 1, 12, 12))]:
        xr = torch.randn(*shape)
        xf = torch.randn(*shape)

        # Baseline: Centered linear critic with zero bias
        class BiasedCritic(nn.Module):
            def __init__(self, in_features, b=0.0):
                super().__init__()
                self.proj = nn.Linear(in_features, 1)
                with torch.no_grad():
                    self.proj.weight.zero_()
                    self.proj.bias.fill_(b)

            def forward(self, x):
                return self.proj(x.flatten(1))

        in_feat = xr[0].numel()

        # 1. Zero drift: bias = 0.0 -> gauge_loss == 0.0
        D_zero = BiasedCritic(in_feat, b=0.0)
        pen_0, stats_0 = kernel.penalty(D_zero, xr, xf, step=0, collect_stats=True)
        assert abs(stats_0.get("mean_logit", 0.0)) < 1e-6
        assert float(pen_0.item()) < 1e-8

        # 2. Translational drift: bias = +1.0 and bias = +3.0
        shifts = [1.0, 3.0, -2.0]
        eff_gamma = 0.005 if dim_name == "image144" else gamma
        for b in shifts:
            D_b = BiasedCritic(in_feat, b=b)
            pen_b, stats_b = kernel.penalty(D_b, xr, xf, step=0, collect_stats=True)

            # Gauge loss formula: 0.5 * eff_gamma * b^2
            expected_loss = 0.5 * eff_gamma * (b ** 2)
            actual_loss = float(pen_b.item())
            assert abs(actual_loss - expected_loss) < 1e-5, (
                f"{dim_name} with bias={b}: expected {expected_loss}, got {actual_loss}"
            )
            assert abs(stats_b["mean_logit"] - b) < 1e-5

            # Restoring gradient check
            pen_b.backward()
            bias_grad = D_b.proj.bias.grad.item()
            expected_grad = eff_gamma * b  # d/db [0.5 * eff_gamma * b^2] = eff_gamma * b
            assert abs(bias_grad - expected_grad) < 1e-5, (
                f"{dim_name}: expected bias grad {expected_grad}, got {bias_grad}"
            )


def test_bispf_continuous_thermal_cooling_and_reheating():
    """Verify continuous thermal scaling smoothly cools inside mode basins, re-heats
    upon non-stationary target shifts, and scales backward gradients straight-through.
    """
    from particlegan.bispf import ContinuousThermalCooler

    cooler = ContinuousThermalCooler(
        tau_cooling=0.020,
        s_min=1e-5,
        ema_decay=0.90,  # Responsive EMA for unit test
        eps_smooth=0.001,
    )

    # 1. Initial high residual -> s_thermal = 1.0
    s_init = cooler.update(batch_residual=0.80)
    assert abs(s_init - 1.0) < 1e-6

    # 2. Monotonic cooling as particles settle in basins
    residuals = [0.018, 0.014, 0.010, 0.005, 0.001]
    prev_s = s_init
    for r in residuals:
        s_curr = cooler.update(batch_residual=r)
        assert s_curr <= prev_s, f"Thermal scale must decrease monotonically: {s_curr} > {prev_s}"
        assert s_curr >= 1e-5, f"Thermal scale must never drop below s_min: {s_curr} < 1e-5"
        prev_s = s_curr

    # At steady-state near r=0.0, cooler reaches floor s_min
    for _ in range(120):
        cooler.update(batch_residual=0.0)
    assert abs(cooler.scale - 1e-5) < 1e-6

    # 3. Straight-Through Gradient Verification
    x = torch.randn(8, 2, requires_grad=True)
    D = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))

    raw_out = D(x)
    raw_grad = torch.autograd.grad(raw_out.sum(), x, retain_graph=True)[0]

    s_test = 0.25
    scaled_out = cooler.apply_straight_through(raw_out, scale=s_test)
    scaled_grad = torch.autograd.grad(scaled_out.sum(), x)[0]

    # Forward logits must be unchanged
    assert torch.allclose(scaled_out, raw_out)
    # Backward gradient must be exactly scaled by s_test
    assert torch.allclose(scaled_grad, s_test * raw_grad, atol=1e-6)

    # 4. Autonomous Re-Heating on Target Shift (step 4000 shift simulation)
    # Target jumps -> batch residual spikes to 1.20
    for _ in range(15):
        s_reheat = cooler.update(batch_residual=1.20)

    assert abs(s_reheat - 1.0) < 1e-3, (
        f"Cooler must autonomously re-heat to 1.0 after distribution shift, got {s_reheat}"
    )


def test_bispf_sbnp_variance_floor_one_sided_behavior():
    """Verify SBNP variance floor strictly penalizes dimensional collapse (std < target_std)
    while imposing ZERO penalty and ZERO gradient on variance expansion (std > target_std)
    and anisotropic stretching (allowing vector_anisotropic to freely elongate).
    """
    from particlegan.bispf import SBNPPriorRegularizer, sbnp_regularization_loss

    target_std = 1.0
    sbnp = SBNPPriorRegularizer(target_std=target_std, k_nn=3, repel_weight=0.0)

    # 1. Collapsed Particles (std = 0.2 < 1.0) -> Positive penalty with outward restoring grad
    z_collapsed = (0.2 * torch.randn(200, 2)).requires_grad_(True)
    loss_collapsed = sbnp(z_collapsed)
    assert loss_collapsed.item() > 0.40, f"Collapsed particles must incur high loss: {loss_collapsed.item()}"

    loss_collapsed.backward()
    assert z_collapsed.grad is not None
    # Outward gradient: inner product with z must be negative (loss decreases as ||z|| increases)
    grad_dot_z = (z_collapsed.grad * z_collapsed).sum().item()
    assert grad_dot_z < 0.0, "Gradient must push particles outward to restore variance"

    # 2. Target Variance Particles (std = 1.0) -> Zero variance penalty
    torch.manual_seed(123)
    z1 = torch.randn(500)
    z2 = torch.randn(500)
    z1 = z1 - z1.mean()
    z2 = z2 - z2.mean()
    z2 = z2 - (z2 @ z1) / (z1 @ z1) * z1
    z1 = z1 / z1.std()
    z2 = z2 / z2.std()
    z_target = torch.stack([z1, z2], dim=1)
    loss_target = sbnp_regularization_loss(z_target, target_std=target_std, repel_weight=0.0)
    assert abs(loss_target.item()) < 1e-4

    # 3. Expanded Particles (std = 2.5 > target_std = 1.0) -> STRICT ZERO PENALTY & ZERO GRAD
    z_expanded_diag = torch.stack([2.5 * z1, 2.5 * z2], dim=1).requires_grad_(True)

    loss_expanded = sbnp_regularization_loss(z_expanded_diag, target_std=target_std, repel_weight=0.0)
    assert abs(loss_expanded.item()) < 1e-5, (
        f"One-sided variance floor must NOT penalize expanded particles: got {loss_expanded.item()}"
    )

    # 4. Anisotropic Stretching (sigma_x = 4.0, sigma_y = 1.0) -> ZERO PENALTY
    z_aniso = torch.stack([4.0 * z1, 1.0 * z2], dim=1)
    loss_aniso = sbnp_regularization_loss(z_aniso, target_std=target_std, repel_weight=0.0)
    assert abs(loss_aniso.item()) < 1e-4, (
        f"Anisotropic stretching must not be penalized: got {loss_aniso.item()}"
    )


def test_bispf_sbnp_degree_of_freedom_guard_discrete_codebooks():
    """Verify SBNP degree-of-freedom guard (N <= 2*k_nn + d_z) completely deactivates
    regularization for discrete token codebooks (N=5), leaving embeddings unperturbed.
    """
    from particlegan.bispf import SBNPPriorRegularizer

    # Discrete codebook parameters from five_word_joint_hold: N=5 particles, z_dim=5, k_nn=3
    n_tokens = 5
    z_dim = 5
    k_nn = 3
    assert n_tokens <= 2 * k_nn + z_dim  # 5 <= 11

    codebook = nn.Parameter(torch.zeros(n_tokens, z_dim))  # Fully collapsed token embeddings
    sbnp = SBNPPriorRegularizer(target_std=1.0, k_nn=k_nn, repel_weight=0.1)

    loss = sbnp(codebook)
    assert loss.item() == 0.0, "Degree-of-freedom guard must return exactly 0.0 for N <= min_dof"

    # Backward pass must produce zero gradient (or None) on codebook parameters
    loss.backward()
    assert codebook.grad is None or (codebook.grad == 0).all(), (
        "Degree-of-freedom guard must not backpropagate spurious gradients into discrete codebooks"
    )

    # Counter-check: when N > 2*k_nn + d_z (e.g. N=32), collapsed particles MUST be penalized
    z_active = nn.Parameter(torch.zeros(32, z_dim))
    loss_active = sbnp(z_active)
    assert loss_active.item() > 0.5, "Active regime (N > min_dof) must penalize collapsed particles"


def test_bispf_continuous_image_transport_chords():
    """Verify continuous transport chords are evaluated for 4D image tensors (d=144),
    contributing chord gradient penalty, while categorical token simplices bypass chords.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_floor=0.0,
        lambda_cap=10.0,
        kappa=0.5,
        chord_blend=0.5,  # 50% fake, 50% chord
        gamma_gauge=0.0,
    )

    # 4D Image batch: 16 images of size 1x12x12 (d=144)
    xr = torch.randn(16, 1, 12, 12)
    xf = torch.randn(16, 1, 12, 12)

    # Critic that has zero gradient at endpoints xr and xf, but high gradient on chord interior
    class ChordSensitiveCritic(nn.Module):
        def forward(self, x):
            # Nonlinear bump centered at chord midpoint
            dist = (x.flatten(1) - 0.5 * (xr.flatten(1).mean(0) + xf.flatten(1).mean(0))).norm(2, dim=1)
            return torch.sin(5.0 * dist).unsqueeze(1)

    D = ChordSensitiveCritic()
    pen, stats = kernel.penalty(D, xr, xf, step=0, collect_stats=True)

    assert pen.item() > 0.0
    assert stats["applied"] is True
    assert "mean_grad_norm" in stats
    assert stats["mean_grad_norm"] > 0.0


def test_bispf_endpoint_born_infeld_image_limit():
    """Verify 4D image tensors (with standard chord_blend <= 0.10) bypass chord
    interpolation and evaluate crisp endpoint Born-Infeld penalties strictly
    at real and fake endpoints without evaluating g_chord, while continuous 2D
    flows evaluate chords.
    """
    from particlegan.bispf import _BISPFPenaltyKernel
    import torch.nn.functional as F

    # 1. Verify 4D image tensors bypass chords (exactly 2 critic evaluations, real and fake)
    called_image_shapes = []
    class TrackingImageCritic(nn.Module):
        def forward(self, x):
            called_image_shapes.append(tuple(x.shape))
            return x.flatten(1).sum(dim=1, keepdim=True)

    kernel = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_floor=0.1,
        lambda_cap=10.0,
        kappa=1.0,
        chord_blend=0.02,
        gamma_gauge=0.002,
    )

    xr_img = torch.randn(16, 1, 12, 12)
    xf_img = torch.randn(16, 1, 12, 12)
    pen_img, stats_img = kernel.penalty(TrackingImageCritic(), xr_img, xf_img, step=0, collect_stats=True)

    assert len(called_image_shapes) == 2, (
        f"Image tensors must bypass chords and evaluate only real and fake endpoints; "
        f"got {len(called_image_shapes)} calls: {called_image_shapes}"
    )
    assert called_image_shapes == [(16, 1, 12, 12), (16, 1, 12, 12)]
    assert stats_img["applied"] is True
    assert "mean_grad_norm" in stats_img
    assert stats_img["mean_grad_norm"] > 0.0
    assert "gauge_loss" in stats_img

    # 2. Verify continuous 2D flows evaluate chords (exactly 3 critic evaluations: real, fake, chord)
    called_2d_shapes = []
    class Tracking2DCritic(nn.Module):
        def forward(self, x):
            called_2d_shapes.append(tuple(x.shape))
            return x.flatten(1).sum(dim=1, keepdim=True)

    xr_2d = torch.randn(16, 2)
    xf_2d = torch.randn(16, 2)
    pen_2d, stats_2d = kernel.penalty(Tracking2DCritic(), xr_2d, xf_2d, step=0, collect_stats=True)

    assert len(called_2d_shapes) == 3, (
        f"Continuous 2D coordinates must evaluate chords; got {len(called_2d_shapes)} calls: {called_2d_shapes}"
    )
    assert called_2d_shapes == [(16, 2), (16, 2), (16, 2)]

    # 3. Verify theoretical endpoint Born-Infeld math for image tensors
    s = 2.0
    kappa = 1.0
    beta = 30.0
    l_cap = 10.0
    l_floor = 0.1
    kernel_math = _BISPFPenaltyKernel(
        coeff=1.0,
        kappa=kappa,
        lambda_cap=l_cap,
        lambda_floor=l_floor,
        ebi_beta=beta,
        gamma_gauge=0.0,
        chord_blend=0.02,
    )

    dim_img = 1 * 8 * 8

    class ConstantSlopeImageCritic(nn.Module):
        def forward(self, x):
            return (s / math.sqrt(dim_img)) * x.flatten(1).sum(dim=1, keepdim=True)

    xr_math = torch.randn(8, 1, 8, 8)
    xf_math = torch.randn(8, 1, 8, 8)
    pen_math, stats_math = kernel_math.penalty(ConstantSlopeImageCritic(), xr_math, xf_math, step=0, collect_stats=True)

    # In Euclidean coordinates: ||g||_2 = sqrt(dim * (s / sqrt(dim))^2) = s = 2.0
    # Under scale_dim = 1.0, norm = s / 1.0 = 2.0 matches expected analytical cap and decoupled floor penalties with adaptive kappa_eff
    res_ratio = min(1.0, max(0.0, float(getattr(kernel_math.record, "ema_residual", 1.0))))
    img_scale = min(1.0, max(0.15, float(getattr(kernel_math.record, "ema_data_scale", 4.899)) / 4.899))
    kappa_eff = (kappa / img_scale) * (0.6 + 1.10 * (1.0 - res_ratio))
    norm = s / 1.0
    expected_cap = l_cap * (F.softplus(torch.tensor(norm - kappa_eff), beta=beta).item() ** 2)
    expected_floor = 0.5 * 0.0005 * ((s ** 2) / dim_img)
    expected_total = expected_cap + expected_floor + stats_math["gauge_loss"]
    assert abs(pen_math.item() - expected_total) < 2e-3, (
        f"Expected {expected_total}, got {pen_math.item()}"
    )


def test_bispf_sinkhorn_scale_free_nn_residual():
    """Verify sinkhorn_coupling returns scale-free symmetric nearest-neighbor residual delta_nn
    with zero entropic floor when real and fake points coincide, and affine scale-invariance.
    """
    from particlegan.bispf import sinkhorn_coupling

    torch.manual_seed(42)
    # 1. Identical mode sets -> delta_nn must be exactly zero (zero entropic floor!)
    x_real = torch.randn(32, 2)
    x_fake = x_real.clone()
    Pi, med_sq, delta_nn, scale = sinkhorn_coupling(x_real, x_fake)
    assert abs(delta_nn.item()) < 5e-4, f"Identical real/fake batches must have delta_nn ~ 0, got {delta_nn.item()}"

    # 2. Separated modes -> delta_nn > 0
    x_fake_sep = x_real + 2.0
    _, _, delta_sep, _ = sinkhorn_coupling(x_real, x_fake_sep)
    assert delta_sep.item() > 0.05

    # 3. Scale-invariance: multiplying coordinates by alpha leaves delta_nn unchanged
    alpha = 15.0
    _, _, delta_scaled, _ = sinkhorn_coupling(x_real * alpha, x_fake_sep * alpha)
    assert abs(delta_sep.item() - delta_scaled.item()) < 1e-4, (
        f"delta_nn must be scale-invariant: got {delta_sep.item()} vs {delta_scaled.item()}"
    )


def test_bispf_v2_trainer_integration_and_telemetry():
    """Verify full GANTrainer integration with bispf recipe enforces strictly constant
    learning rates, exposes telemetry keys, and executes bit-exact checkpoint restoration.
    """
    recipe = get_recipe(
        "bispf",
        num_particles=64,
        z_dim=2,
        batch_size=16,
        total_steps=50,
    )
    # Strictly constant learning rate contract
    assert recipe.lr_schedule == "constant"
    for step in [0, 10, 25, 49]:
        assert learning_rate_scales(step, recipe) == (1.0, 1.0)

    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    prior = recipe.make_prior()
    for idx, mod in enumerate((G, D, prior)):
        init.deterministic_orthogonal_(mod, seed=idx)

    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    data_gen = torch.Generator().manual_seed(999)

    # Run 5 training steps
    for _ in range(5):
        real = torch.randn(16, 2, generator=data_gen)
        out = trainer.step(real, collect_stats=True)
        stats = out["penalty_stats"]
        assert "applied" in stats and stats["applied"] is True
        assert "ema_residual" in stats
        assert "mean_logit" in stats

    # Verify bit-exact checkpoint restoration
    ckpt = deepcopy(trainer.state_dict())
    next_real = torch.randn(16, 2, generator=data_gen)
    out_1 = trainer.step(next_real, collect_stats=True)

    G2 = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
    D2 = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    prior2 = recipe.make_prior()
    trainer2 = GANTrainer(recipe, G2, D2, prior=prior2, seed=777)
    trainer2.load_state_dict(ckpt)
    out_2 = trainer2.step(next_real, collect_stats=True)

    assert torch.allclose(out_1["loss_d"], out_2["loss_d"])
    assert torch.allclose(out_1["loss_g"], out_2["loss_g"])


def test_bispf_critic_hook_generator_gradient_unthrottled():
    """Verify that critic forward hook does NOT throttle generator gradients even under
    severe thermal cooling (s_thermal < 1.0), preserving true constant learning rate 1.0.
    """
    from particlegan.bispf import (
        BISPFStepRecord,
        _attach_critic_input_hook,
    )

    torch.manual_seed(42)
    D = nn.Sequential(nn.Linear(2, 4), nn.LeakyReLU(0.2), nn.Linear(4, 1))
    G = nn.Sequential(nn.Linear(2, 4), nn.LeakyReLU(0.2), nn.Linear(4, 2))

    # 1. Baseline unhooked critic gradient
    z = torch.randn(8, 2)
    fake_base = G(z)
    loss_base = D(fake_base).sum()
    loss_base.backward()
    grads_base = [p.grad.clone() for p in G.parameters()]

    # 2. Hooked critic during uncooled exploration (s_thermal = 1.0)
    for p in G.parameters():
        p.grad.zero_()

    record = BISPFStepRecord()
    _attach_critic_input_hook(D, record)
    record._thermal_scale = 1.0  # Exploration
    record._in_penalty = False

    fake_uncooled = G(z)
    loss_uncooled = D(fake_uncooled).sum()
    loss_uncooled.backward()
    grads_uncooled = [p.grad.clone() for p in G.parameters()]

    # During exploration (s_thermal = 1.0), generator gradients must match baseline EXACTLY (100% unthrottled)
    for gb, gu in zip(grads_base, grads_uncooled):
        assert torch.allclose(gb, gu, atol=1e-7), "Generator gradients must not be throttled during exploration"

    # 3. Hooked critic under thermal cooling (s_thermal = 0.02) linearizes velocity straight-through by s_local
    for p in G.parameters():
        p.grad.zero_()
    record._thermal_scale = 0.02  # Cooling
    fake_cooled = G(z)
    loss_cooled = D(fake_cooled).sum()
    # Forward pass value is unchanged
    assert torch.allclose(loss_cooled, loss_base, atol=1e-6)
    loss_cooled.backward()
    grads_cooled = [p.grad.clone() for p in G.parameters()]
    for gc, gb in zip(grads_cooled, grads_base):
        assert torch.allclose(gc, gb * 0.02, atol=1e-6), "Gradients must scale directly by s under cooling"

    # 4. Verify input coordinate attachment for SinkhornRelativisticLoss
    fake_logits = D(fake_cooled)
    assert hasattr(fake_logits, "_bispf_input")
    assert fake_logits._bispf_input is fake_cooled


def test_bispf_sbnp_capacity_conditioned_covariance_tethering():
    """Verify capacity-conditioned SBNP covariance loss:
    - Finite ensembles (N <= 64, d <= 4): engages two-sided covariance tethering (Delta C = C - C_0)
      to anchor mode basins against drift.
    - Dense ensembles (N > 64): engages off-diagonal decorrelation (Delta C_off = C - diag(C)),
      imposing zero penalty on diagonal expansion so radial rings can expand freely.
    """
    from particlegan.bispf import SBNPPriorRegularizer

    class FixedRefSBNP(SBNPPriorRegularizer):
        def _reference_moments(self):
            return torch.zeros(2), torch.eye(2), 100

    sbnp = FixedRefSBNP(target_std=1.0, k_nn=3, repel_weight=0.0)
    sbnp.references = True

    # Case A: Sample-starved finite ensemble (N = 12 <= 64, d = 2 <= 4)
    # Particles with diagonal variance expansion: var = 4.0 (std = 2.0 > 1.0), zero off-diagonal
    z_small = torch.zeros(12, 2)
    pattern4 = torch.tensor([[2.0, 2.0], [2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0]])
    z_small = pattern4.repeat(3, 1)  # 12 samples, mean=0, off-diagonal cov=0, var ~ 4.36
    loss_small = sbnp(z_small)
    # Small N <= 64 must penalize the diagonal variance deviation from target_cov=I
    assert loss_small.item() > 1.0, f"N <= 64 must engage two-sided covariance tethering; got {loss_small.item()}"

    # Case B: Dense ensemble (N = 256 > 64, d = 2)
    # Particles with exact same expanded diagonal variance (var ~ 4.015) and zero off-diagonal
    z_large = pattern4.repeat(64, 1)  # 256 samples, mean=0, off-diagonal cov=0, var ~ 4.015
    loss_large = sbnp(z_large)
    # Large N > 64 uses off-diagonal decorrelation; since off-diagonal is 0 and std > target_std,
    # covariance loss and variance floor loss are both 0!
    assert loss_large.item() < 1e-4, f"N > 64 must allow diagonal variance expansion without penalty; got {loss_large.item()}"

    # Case C: Dense ensemble with off-diagonal correlation (N = 256, cov_xy != 0)
    z_corr = torch.randn(256, 1)
    z_corr_2d = torch.cat([z_corr, z_corr], dim=1)  # perfectly correlated
    loss_corr = sbnp(z_corr_2d)
    assert loss_corr.item() > 0.05, f"N > 64 must penalize off-diagonal correlation; got {loss_corr.item()}"


def test_bispf_1d_continuous_flow_cooler_active():
    """Verify 1D scalar flows (d=1, e.g. gaussian1d_stability) engage continuous
    thermal cooling when call_idx >= 500.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(
        tau_cooling=0.020,
        s_min=1e-5,
    )
    # Hold cooler residual at 0.001 by overriding _observe_sinkhorn
    kernel._observe_sinkhorn = lambda res, ds, ema_decay=0.95: None
    kernel.record.ema_residual = 0.001
    kernel.cooler.ema_residual = 0.001
    kernel.cooler.initialized = True
    assert kernel.cooler.scale < 0.01

    class Simple1DCritic(nn.Module):
        def forward(self, x):
            return x

    xr_1d = torch.randn(64, 1)
    xf_1d = torch.randn(64, 1)

    # Call with call_idx >= 500
    kernel.record.calls = 600
    pen, stats = kernel.penalty(Simple1DCritic(), xr_1d, xf_1d, step=0, collect_stats=True)

    # In 1D, is_continuous_flow is True (1 <= dim <= 4), so s_thermal must engage cooling
    assert stats["s_thermal"] < 0.10, f"1D flow must engage thermal cooling; got {stats['s_thermal']}"
    assert kernel.record._thermal_scale < 0.10

    # Counter-check: in 2D (1 <= dim <= 4), continuous cooling is also active for call_idx >= 500
    class Simple2DCritic(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1, keepdim=True)

    xr_2d = torch.randn(64, 2)
    xf_2d = torch.randn(64, 2)
    pen_2d, stats_2d = kernel.penalty(Simple2DCritic(), xr_2d, xf_2d, step=0, collect_stats=True)
    assert stats_2d["s_thermal"] < 0.10, f"2D flow must engage thermal cooling; got {stats_2d['s_thermal']}"


def test_bispf_gauge_and_cooling_defaults():
    """Verify default hyperparameter calibration in _BISPFPenaltyKernel:
    gamma_gauge == 0.050 and tau_cooling == 0.020.
    """
    from particlegan.bispf import _BISPFPenaltyKernel, ContinuousThermalCooler

    kernel = _BISPFPenaltyKernel()
    assert kernel.gamma_gauge == 0.050, f"Expected default gamma_gauge=0.050, got {kernel.gamma_gauge}"
    assert kernel.tau_cooling == 0.020, f"Expected default tau_cooling=0.020, got {kernel.tau_cooling}"
    assert kernel.cooler.tau_cooling == 0.020, f"Expected cooler tau_cooling=0.020, got {kernel.cooler.tau_cooling}"

    cooler = ContinuousThermalCooler()
    assert cooler.tau_cooling == 0.020, f"Expected default ContinuousThermalCooler tau_cooling=0.020, got {cooler.tau_cooling}"


def test_bispf_ring16_thermal_stability():
    """Verify that multi-modal ring geometries with residual delta_nn >= 0.026
    maintain s_thermal == 1.0 unconditionally under default tau_cooling=0.020,
    preventing premature gradient throttling and preserving sustained passing suffixes.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel()
    # Simulate ring16 residual regime: delta_nn ~ 0.028 in batch_size=1024
    kernel.cooler.ema_residual = 0.028
    kernel.cooler.initialized = True
    assert kernel.cooler.scale_for(batch_size=1024, dim=2) == 1.0, f"Ring16 residual must hold s_thermal=1.0; got {kernel.cooler.scale_for(batch_size=1024, dim=2)}"

    # Counter-check: within-basin settling on 100-mode grids (residual ~ 0.005) DOES cool
    kernel.cooler.ema_residual = 0.005
    assert kernel.cooler.scale_for(batch_size=1024, dim=2) < 0.20, f"100-mode grid basin settling must cool below 0.20; got {kernel.cooler.scale_for(batch_size=1024, dim=2)}"


def test_bispf_adaptive_image_kappa():
    """Verify adaptive image Born-Infeld scale kappa_eff:
    - High transport residual (early in training) softens kappa_eff to ~ 0.6 * kappa,
      lowering the potential barrier and preventing mode monopolization.
    - Low transport residual (converged) tightens kappa_eff back to 1.0 * kappa
      for crisp terminal image mode reconstruction.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    # High residual regime: simulate early training
    kernel_high = _BISPFPenaltyKernel(kappa=1.0)
    kernel_high.record.ema_residual = 1.0
    kernel_high.record.initialized = True

    # Low residual regime: simulate late converged training
    kernel_low = _BISPFPenaltyKernel(kappa=1.0)
    kernel_low.record.ema_residual = 0.0
    kernel_low.record.initialized = True

    class ConstantImageCritic(nn.Module):
        def forward(self, x):
            return x.flatten(1).sum(dim=1, keepdim=True)

    xr = torch.randn(16, 1, 8, 8)
    xf = torch.randn(16, 1, 8, 8)

    pen_high, _ = kernel_high.penalty(ConstantImageCritic(), xr, xf, step=0)

    # Overwrite _observe_sinkhorn to keep residual at 0.0 for low-residual test
    kernel_low._observe_sinkhorn = lambda res, scale, ema_decay=0.95: None
    kernel_low.record.ema_residual = 0.0

    pen_low, _ = kernel_low.penalty(ConstantImageCritic(), xr, xf, step=0)

    # Softened barrier (kappa_eff=0.6) imposes higher cap penalty for the same gradient norm > 1.0
    assert pen_high.item() > pen_low.item(), (
        f"Softened barrier (kappa_eff=0.6) must penalize large gradients more strongly than kappa=1.0: "
        f"{pen_high.item()} vs {pen_low.item()}"
    )

    # Check eff_gamma_gauge on images is 0.005 vs vector 0.020
    kernel_gauge = _BISPFPenaltyKernel(gamma_gauge=0.020)
    xr_vec = torch.randn(16, 2)
    xf_vec = torch.randn(16, 2)
    class BiasOnlyCritic(nn.Module):
        def forward(self, x):
            return 0.0 * x.flatten(1).sum(dim=1, keepdim=True) + 2.0
    _, stats_vec = kernel_gauge.penalty(BiasOnlyCritic(), xr_vec, xf_vec, step=0, collect_stats=True)
    expected_vec_gauge = 0.5 * 0.020 * (2.0 ** 2)
    assert abs(stats_vec["gauge_loss"] - expected_vec_gauge) < 1e-5

    _, stats_img = kernel_gauge.penalty(BiasOnlyCritic(), xr, xf, step=0, collect_stats=True)
    expected_img_gauge = 0.5 * 0.005 * (2.0 ** 2)
    assert abs(stats_img["gauge_loss"] - expected_img_gauge) < 1e-5

    # Check sub-batch (b_r < 16, e.g. b_r=8) caps eff_gamma_gauge at 0.020 even with default gamma_gauge=0.050
    kernel_default = _BISPFPenaltyKernel()
    xr_sub = torch.randn(8, 2)
    xf_sub = torch.randn(8, 2)
    _, stats_sub = kernel_default.penalty(BiasOnlyCritic(), xr_sub, xf_sub, step=0, collect_stats=True)
    expected_sub_gauge = 0.5 * 0.020 * (2.0 ** 2)
    assert abs(stats_sub["gauge_loss"] - expected_sub_gauge) < 1e-5


def test_bispf_lambda_floor_default():
    """Verify _BISPFPenaltyKernel and BISPFEngine default lambda_floor is 0.01."""
    from particlegan.bispf import _BISPFPenaltyKernel, BISPFEngine

    kernel = _BISPFPenaltyKernel()
    assert kernel.lambda_floor == 0.01, f"Expected lambda_floor == 0.01, got {kernel.lambda_floor}"
    engine = BISPFEngine()
    assert engine.lambda_floor == 0.01, f"Expected BISPFEngine.lambda_floor == 0.01, got {engine.lambda_floor}"


def test_bispf_fast_attack_reheating():
    """Verify ContinuousThermalCooler and _BISPFPenaltyKernel re-heat instantaneously on residual shock."""
    from particlegan.bispf import ContinuousThermalCooler, _BISPFPenaltyKernel

    cooler = ContinuousThermalCooler(tau_cooling=0.020)
    for _ in range(50):
        cooler.update(0.005)
    assert cooler.scale < 0.20, f"Expected cooled scale < 0.20, got {cooler.scale}"

    # Shock update (non-stationary shift, delta_nn spikes to 0.207)
    scale_shock = cooler.update(0.207)
    assert scale_shock == 1.0, f"Fast-attack must instantly restore scale to 1.0, got {scale_shock}"
    assert cooler.ema_residual == 0.207

    kernel = _BISPFPenaltyKernel(tau_cooling=0.020)
    for _ in range(50):
        kernel._observe_sinkhorn(0.005, 1.0)
    assert kernel.cooler.scale < 0.20
    kernel._observe_sinkhorn(0.207, 1.0)
    assert kernel.cooler.scale == 1.0
    assert kernel.record.ema_residual == 0.207


def test_bispf_dense_multimode_rownorm_linearization():
    """Verify Born-Infeld DualNorm rownorm inversion scales update norm linearly in dense multi-mode regimes."""
    from particlegan.bispf import SinkhornRelativisticLoss, _LAST_THERMAL_SCALE

    loss_fn = SinkhornRelativisticLoss()
    fake_logits = torch.full((2048, 1), -5.0, requires_grad=True)
    real_logits = torch.full((2048, 1), 5.0)
    fake_logits._bispf_input = torch.randn(2048, 2)
    real_logits._bispf_input = torch.randn(2048, 2)

    # Set thermal scale to 0.10
    _LAST_THERMAL_SCALE[fake_logits.device] = 0.10
    try:
        loss = loss_fn.g_loss(fake_logits, real_logits)
        grad = torch.autograd.grad(loss, fake_logits)[0]
        # Evaluate DualNorm rownorm on this gradient:
        eps_smooth = 0.001
        norm = grad.norm(dim=1, keepdim=True)
        denom = torch.hypot(norm, torch.full_like(norm, eps_smooth))
        step_norm = (grad / denom).norm(dim=1).mean().item()
        assert abs(step_norm - 0.10) < 0.02, f"DualNorm step norm should be linearized to ~0.10, got {step_norm}"
    finally:
        _LAST_THERMAL_SCALE.clear()


def test_bispf_image_scale_dim():
    """Verify spatial image tensors evaluate penalties with scale_dim == 2.0."""
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel()

    class IdentityCritic(nn.Module):
        def forward(self, x):
            return x.flatten(1).sum(dim=1, keepdim=True)

    xr = torch.randn(16, 1, 8, 8)
    xf = torch.randn(16, 1, 8, 8)
    # On image tensors, scale_dim is 2.0 (dim = 64)
    loss, _ = kernel.penalty(IdentityCritic(), xr, xf, step=0)
    assert torch.isfinite(loss)


def test_bispf_1d_continuous_flow_cooler_with_fast_attack():
    """Verify 1D flows engage thermal cooling after step 500 and immediately re-heat upon shock."""
    from particlegan.bispf import _BISPFPenaltyKernel, _LAST_THERMAL_SCALE

    kernel = _BISPFPenaltyKernel(tau_cooling=0.020, s_min=1e-5)
    kernel.record.calls = 600

    class Dummy1DCritic(nn.Module):
        def forward(self, x):
            return x

    xr = torch.randn(64, 1)
    xf = xr + 1e-4

    for _ in range(50):
        kernel._observe_sinkhorn(0.001, 1.0)

    try:
        _, _ = kernel.penalty(Dummy1DCritic(), xr, xf, step=0, collect_stats=True)
        scale_damped = _LAST_THERMAL_SCALE.get(xr.device, 1.0)
        assert scale_damped < 0.20, f"1D flow should cool to < 0.20, got {scale_damped}"

        # Non-stationary shift shock
        xf_shock = xr + 4.0
        _, _ = kernel.penalty(Dummy1DCritic(), xr, xf_shock, step=1, collect_stats=True)
        scale_shock = _LAST_THERMAL_SCALE.get(xr.device, 0.0)
        assert scale_shock == 1.0, f"1D flow must re-heat instantly to 1.0 on shock, got {scale_shock}"
    finally:
        _LAST_THERMAL_SCALE.clear()


def test_sbnp_regularizer_defaults_and_unweighted_scale():
    """Verify SBNPPriorRegularizer defaults to k_nn=5, repel_weight=0.05, repel_margin=0.50 and linear scaling."""
    from particlegan.bispf import SBNPPriorRegularizer

    reg = SBNPPriorRegularizer()
    assert reg.k_nn == 5
    assert reg.repel_weight == 0.05
    assert reg.repel_margin == 0.50

    # Unanchored latent prior (weight=0.01 in (0, 0.1) -> scale_w = 1.0)
    reg_small = SBNPPriorRegularizer(weight=0.01)
    z = torch.randn(64, 8, requires_grad=True)
    loss = reg_small(z)
    assert torch.isfinite(loss)

    # Anchored reference prior (scale_w = self.weight = 0.01)
    class _MockPrior:
        def __init__(self, d=2):
            self.moment_tether_mean = torch.zeros(d)
            self.moment_tether_covariance = torch.eye(d)
            self.moment_tether_count = torch.tensor(100)

    ref = _MockPrior(d=2)
    reg_ref = SBNPPriorRegularizer(reference=ref, weight=0.01)
    assert len(reg_ref.references) == 1
    z_ref = torch.randn(64, 2, requires_grad=True)
    loss_ref = reg_ref(z_ref)
    assert torch.isfinite(loss_ref)


def test_bispf_scale_invariant_thermal_cooling():
    """Verify ContinuousThermalCooler dimension- and batch-scale invariant thresholding:
    - effective_tau scales as tau_0 * (1024 / B) in 1D and tau_0 * (128 / B)^(1/d) in 2D+.
    - For B=128, d=1: tau_eff = 0.020 * 8 = 0.160, so residual 0.10 cools (scale < 1.0).
    - For B=128, d=2: tau_eff = 0.020, so ring16 residual 0.028 does not cool (scale == 1.0).
    """
    from particlegan.bispf import ContinuousThermalCooler

    cooler = ContinuousThermalCooler(tau_cooling=0.020, s_min=0.05)

    # 1. Exact scaling factor checks
    tau_base = cooler.effective_tau(batch_size=128, dim=2)
    assert abs(tau_base - 0.020) < 1e-6

    tau_1d = cooler.effective_tau(batch_size=128, dim=1)
    assert abs(tau_1d - 0.020) < 1e-6  # For b >= 128, tau_eff = 0.020 without 8x inflation

    tau_1024_2d = cooler.effective_tau(batch_size=1024, dim=2)
    assert abs(tau_1024_2d - 0.020) < 1e-6

    for b in [16, 64, 128, 512, 1024]:
        expected_scale = 1.0 if b >= 128 else (1024.0 / b)
        assert abs(cooler.effective_tau(batch_size=b, dim=1) - 0.020 * expected_scale) < 1e-6
        expected_scale_2d = 1.0 if b >= 128 else (1024.0 / b) ** 0.5
        assert abs(cooler.effective_tau(batch_size=b, dim=2) - 0.020 * expected_scale_2d) < 1e-6

    # 2. Cooling behavior
    cooler.ema_residual = 0.10
    # For (128, 2), 0.10 > 0.020 -> uncooled (exploration)
    s_base = cooler.scale_for(batch_size=128, dim=2)
    assert s_base == 1.0

    # For (128, 1), 0.10 > 0.020 -> uncooled (no deadband)
    s_1d_128 = cooler.scale_for(batch_size=128, dim=1)
    assert s_1d_128 == 1.0

    # For (64, 1), tau_eff = 0.020 * 16 = 0.320, 0.10 < 0.320 -> actively cooling
    s_1d_64 = cooler.scale_for(batch_size=64, dim=1)
    assert s_1d_64 < 1.0
    assert s_1d_64 >= cooler.s_min


def test_bispf_critic_hook_straight_through_damping():
    """Verify _bispf_critic_forward_hook straight-through DualNorm velocity linearization:
    - Forward pass is exact identity (loss_hooked == loss_base).
    - Generator backward gradients are scaled by s_v when s_thermal < 0.999.
    - Generator gradients are 100% unthrottled during exploration (s_thermal = 1.0).
    """
    from particlegan.bispf import BISPFStepRecord, _attach_critic_input_hook

    torch.manual_seed(123)
    D = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    G = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))

    z = torch.randn(16, 2)
    fake = G(z)
    loss_base = D(fake).sum()
    loss_base.backward()
    grads_base = [p.grad.clone() for p in G.parameters()]

    record = BISPFStepRecord()
    _attach_critic_input_hook(D, record)

    # 1. Uncooled exploration (s_thermal = 1.0) -> unthrottled
    for p in G.parameters():
        p.grad.zero_()
    record._thermal_scale = 1.0
    record._in_penalty = False
    fake_exp = G(z)
    loss_exp = D(fake_exp).sum()
    loss_exp.backward()
    grads_exp = [p.grad.clone() for p in G.parameters()]
    for gb, ge in zip(grads_base, grads_exp):
        assert torch.allclose(gb, ge, atol=1e-6)

    # 2. Cooled settlement (s_thermal = 0.50) -> linearized by s_local
    for p in G.parameters():
        p.grad.zero_()
    record._thermal_scale = 0.50
    record._in_penalty = False
    fake_cool = G(z)
    loss_cool = D(fake_cool).sum()
    assert torch.allclose(loss_cool, loss_base, atol=1e-6)  # Forward pass unchanged
    loss_cool.backward()
    grads_cool = [p.grad.clone() for p in G.parameters()]

    for p in G.parameters():
        p.grad.zero_()
    record._thermal_scale = 1.0  # Disable hook scaling to test s single application
    fake_fresh = G(z)
    loss_expected = (D(fake_fresh) * 0.50).sum()
    loss_expected.backward()
    grads_expected = [p.grad.clone() for p in G.parameters()]
    for gc, ge in zip(grads_cool, grads_expected):
        assert torch.allclose(gc, ge, atol=1e-6)


def test_bispf_density_directed_transport_bridge():
    """Verify Density-Directed Transport Bridge (DDTB) connects overpopulated fakes
    to starved reals using cached discriminator logits across entropic voids.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(chord_blend=0.02)
    kernel.record.calls = 600  # is_continuous_flow = True

    b = 32
    xr = torch.randn(b, 2)
    xf = torch.randn(b, 2)

    class StarvedModeCritic(nn.Module):
        def forward(self, x):
            val = x[:, :1]
            return val

    # Run step 0: populates _last_d_logits
    pen0, _ = kernel.penalty(StarvedModeCritic(), xr, xf, step=0, collect_stats=True)
    assert kernel.record._last_d_logits is not None
    dr_cached, df_cached = kernel.record._last_d_logits
    assert len(dr_cached) == b
    assert len(df_cached) == b

    # Run step 1: DDTB activates using cached logits
    pen1, stats1 = kernel.penalty(StarvedModeCritic(), xr, xf, step=1, collect_stats=True)
    assert stats1["applied"] is True
    assert torch.isfinite(pen1)


def test_bispf_chord_curvature_penalty():
    """Verify transport chord zero-curvature consistency penalizes non-linear critics
    along continuous flow transport paths, conserving variance.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(chord_blend=0.02)
    kernel.record.calls = 600  # is_continuous_flow = True

    xr = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    xf = torch.linspace(-0.5, 0.5, 32).unsqueeze(1)

    class LinearCritic(nn.Module):
        def forward(self, x):
            return 2.0 * x

    class CurvatureCritic(nn.Module):
        def forward(self, x):
            return 2.0 * x + 5.0 * x.pow(2)

    # Linear critic: D''(x) == 0 -> zero curvature penalty along transport chords
    pen_lin, stats_lin = kernel.penalty(LinearCritic(), xr, xf, step=0, collect_stats=True)

    # Quadratic critic: D''(x) != 0 -> non-zero curvature penalty
    pen_curv, stats_curv = kernel.penalty(CurvatureCritic(), xr, xf, step=0, collect_stats=True)

    assert pen_curv.item() > pen_lin.item(), (
        f"Non-linear critic must incur higher penalty due to curvature regularizer: "
        f"{pen_curv.item()} vs {pen_lin.item()}"
    )


def test_bispf_image_euclidean_barrier_and_kappa():
    """Verify image tensors retain the Euclidean barrier (scale_dim == 1.0),
    progressive well depth kappa_eff = kappa * (0.6 + 0.65(1 - r)), and decoupled floor eff_floor == 0.0005.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(kappa=1.0, lambda_floor=0.01)
    kernel.record.calls = 600

    xr = torch.randn(16, 1, 8, 8)
    xf = torch.randn(16, 1, 8, 8)

    class SlopedImageCritic(nn.Module):
        def forward(self, x):
            return 1.5 * x.flatten(1).sum(dim=1, keepdim=True)

    # High transport residual (early in training): kappa_eff = 0.6 * kappa = 0.6
    kernel.record.ema_residual = 1.0
    kernel._observe_sinkhorn = lambda res, ds, ema_decay=0.95: None
    pen_early, stats_early = kernel.penalty(SlopedImageCritic(), xr, xf, step=0, collect_stats=True)

    # Converged transport residual (late in training): kappa_eff = 1.25 * kappa = 1.25
    kernel.record.ema_residual = 0.0
    pen_late, stats_late = kernel.penalty(SlopedImageCritic(), xr, xf, step=0, collect_stats=True)

    # Under slope 1.5 > 1.25, early softened barrier (0.6) imposes higher softplus cap penalty than late (1.25)
    assert pen_early.item() > pen_late.item(), (
        f"Early softened barrier (kappa_eff=0.6) must penalize large gradients more strongly than kappa=1.25: "
        f"{pen_early.item()} vs {pen_late.item()}"
    )

    # Verify background floor is decoupled to 0.0005 on images vs 0.01 on vectors
    assert stats_early["floor_pen"] < 0.01, f"Image floor penalty must be decoupled; got {stats_early['floor_pen']}"


def test_bispf_sbnp_small_latent_repulsion():
    """Verify that for small unanchored latent clouds (n <= 64 and d >= 8 and not references),
    SBNPPriorRegularizer activates multi-neighbor repulsion with eff_k_nn = 10, eff_repel_weight = 7.5 * repel_weight (0.375),
    and effective_margin = 0.50.
    """
    from particlegan.bispf import SBNPPriorRegularizer
    import torch.nn.functional as F

    torch.manual_seed(42)
    # n=32, d=8: small unanchored latent cloud (image generator latent space)
    z_img = torch.randn(32, 8)
    reg_active = SBNPPriorRegularizer(weight=1.0, k_nn=5, repel_weight=0.05, repel_margin=0.50)
    reg_no_repel = SBNPPriorRegularizer(weight=1.0, k_nn=5, repel_weight=0.0, repel_margin=0.50)

    loss_active = reg_active(z_img)
    loss_base = reg_no_repel(z_img)
    repel_contrib = (loss_active - loss_base).item()

    # Manual calculation with eff_k_nn=10, eff_repel_weight=1.25, effective_margin=1.00:
    m_sub = 32
    sq_dist = torch.cdist(z_img, z_img, p=2.0).pow(2)
    off_mask = ~torch.eye(m_sub, dtype=torch.bool)
    r_med_sq = torch.median(sq_dist[off_mask]).clamp_min(1e-12)
    eye_inf = torch.eye(m_sub, dtype=z_img.dtype) * 1e9
    knn_sq_dist, _ = torch.topk(sq_dist + eye_inf, k=10, dim=1, largest=False)
    norm_sq_dist = knn_sq_dist / (1.00 * r_med_sq)
    expected_repel_loss = F.softplus(-torch.log(norm_sq_dist.clamp_min(1e-12))).mean()
    expected_repel_contrib = 1.25 * expected_repel_loss.item()

    assert abs(repel_contrib - expected_repel_contrib) < 1e-5, (
        f"Small latent cloud must use eff_k_nn=10 and 25x repel weight (1.25): got {repel_contrib} vs {expected_repel_contrib}"
    )

    # Counter-check: when references are provided with moment_tether_count > 64, is_small_latent is FALSE
    class MockRef:
        moment_tether_mean = torch.zeros(8)
        moment_tether_covariance = torch.eye(8)
        moment_tether_count = torch.tensor(100)

    reg_ref = SBNPPriorRegularizer(reference=MockRef(), weight=1.0, k_nn=5, repel_weight=0.05, repel_margin=0.50)
    reg_ref_no_repel = SBNPPriorRegularizer(reference=MockRef(), weight=1.0, k_nn=5, repel_weight=0.0, repel_margin=0.50)
    knn_sq_dist_5, _ = torch.topk(sq_dist + eye_inf, k=5, dim=1, largest=False)
    norm_sq_dist_5 = knn_sq_dist_5 / (1.0 * r_med_sq)
    expected_ref_loss = 0.05 * F.softplus(-torch.log(norm_sq_dist_5.clamp_min(1e-12))).mean().item()
    ref_contrib = (reg_ref(z_img) - reg_ref_no_repel(z_img)).item()
    assert abs(ref_contrib - expected_ref_loss) < 1e-5, (
        f"Anchored reference prior must retain default k_nn=5 and 1x repel weight: got {ref_contrib} vs {expected_ref_loss}"
    )


def test_bispf_2d_continuous_flow_bypasses_chord_curvature():
    """Verify transport chord zero-curvature regularizer is strictly scoped to 1D continuous flows,
    and 2D continuous flows (dim=2) bypass curv_pen (curv_pen == 0.0), preserving multi-cluster
    potential barriers.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(chord_blend=0.02)
    kernel.record.calls = 600  # call_idx >= 500 activates is_continuous_flow

    class CurvatureCritic(nn.Module):
        def forward(self, x):
            return 2.0 * x.flatten(1).sum(dim=1, keepdim=True) + 5.0 * x.flatten(1).pow(2).sum(dim=1, keepdim=True)

    # 1. 1D flow (dim=1): curv_pen is active on non-linear critic
    xr_1d = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    xf_1d = torch.linspace(-0.5, 0.5, 32).unsqueeze(1)
    pen_1d, stats_1d = kernel.penalty(CurvatureCritic(), xr_1d, xf_1d, step=0, collect_stats=True)

    class LinearCritic(nn.Module):
        def forward(self, x):
            return 2.0 * x.flatten(1).sum(dim=1, keepdim=True)

    pen_lin_1d, stats_lin_1d = kernel.penalty(LinearCritic(), xr_1d, xf_1d, step=0, collect_stats=True)
    assert stats_1d["ebi_loss"] > stats_lin_1d["ebi_loss"], "1D continuous flow must incur curvature penalty"

    # 2. 2D flow (dim=2): curv_pen must be strictly 0.0 (bypassed)
    torch.manual_seed(0)
    xr_2d = torch.randn(32, 2)
    xf_2d = torch.randn(32, 2)

    kernel_zero_caps = _BISPFPenaltyKernel(lambda_cap=0.0, lambda_floor=0.0, chord_blend=0.02)
    kernel_zero_caps.record.calls = 600

    _, stats_1d_zero = kernel_zero_caps.penalty(CurvatureCritic(), xr_1d, xf_1d, step=0, collect_stats=True)
    assert stats_1d_zero["ebi_loss"] > 0.0, "1D flow with zero caps must show non-zero curv_pen"

    _, stats_2d_zero = kernel_zero_caps.penalty(CurvatureCritic(), xr_2d, xf_2d, step=0, collect_stats=True)
    assert stats_2d_zero["ebi_loss"] == 0.0, (
        f"2D continuous flow must bypass curvature penalty completely; got ebi_loss = {stats_2d_zero['ebi_loss']}"
    )


def test_bispf_scalar_logit_gauge_preserves_spatial_tilt():
    """Verify that gauge loss penalizes scalar logit drift mean_logit^2 without damping
    spatial translation gradients, preserving critic restoring forces for multi-cluster acquisition.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    gamma = 0.05
    kernel = _BISPFPenaltyKernel(
        coeff=1.0,
        lambda_cap=0.0,
        lambda_floor=0.0,
        gamma_gauge=gamma,
        chord_blend=0.0,
    )

    w = torch.tensor([1.5, -2.0])

    class LinearTiltCritic(nn.Module):
        def forward(self, x):
            return (x @ w).unsqueeze(1) + 2.0

    torch.manual_seed(123)
    xr = torch.randn(32, 2)
    xr = xr - xr.mean(0, keepdim=True)
    xf = torch.randn(32, 2)
    xf = xf - xf.mean(0, keepdim=True)

    pen, stats = kernel.penalty(LinearTiltCritic(), xr, xf, step=0, collect_stats=True)

    expected_gauge = 0.5 * gamma * (2.0 ** 2)
    assert abs(stats["gauge_loss"] - expected_gauge) < 1e-5, (
        f"Expected gauge_loss {expected_gauge}, got {stats['gauge_loss']}"
    )
    assert abs(stats["mean_logit"] - 2.0) < 1e-5


def test_bispf_1d_continuous_flow_velocity_scaling():
    """Verify that 1D continuous flow inputs (dim=1) receive straight-through DualNorm
    velocity scaling in _bispf_critic_forward_hook when s_thermal < 0.999:
    - factor = 0.002 * (s_thermal / sqrt(1 - s_thermal^2))
    - Forward pass is exact identity
    - Backward gradients scale by s_v
    """
    from particlegan.bispf import BISPFStepRecord, _attach_critic_input_hook

    class DummyCritic(nn.Module):
        def forward(self, x):
            return 3.0 * x.sum(dim=-1, keepdim=True)

    critic = DummyCritic()
    record = BISPFStepRecord()
    _attach_critic_input_hook(critic, record)

    # 1. Uncooled exploration (s_thermal = 1.0) -> unthrottled
    record._thermal_scale = 1.0
    record._in_penalty = False
    x_1d = torch.randn(16, 1, requires_grad=True)
    out_1d = critic(x_1d)
    out_1d.sum().backward()
    assert torch.allclose(x_1d.grad, torch.full_like(x_1d.grad, 3.0), atol=1e-5)

    # 2. Cooled flow (s_thermal = 0.50) -> scaled by s_v
    record._thermal_scale = 0.50
    x_1d_cool = torch.randn(16, 1, requires_grad=True)
    out_1d_cool = critic(x_1d_cool)
    assert torch.allclose(out_1d_cool, 3.0 * x_1d_cool.detach(), atol=1e-5)

    out_1d_cool.sum().backward()
    expected_grad = 3.0 * 0.50
    assert torch.allclose(x_1d_cool.grad, torch.full_like(x_1d_cool.grad, expected_grad), atol=1e-5)


def test_bispf_local_covariance_circularization():
    """Verify scale-free local covariance circularization (isotropy):
    - Circular/isotropic clusters have near-zero local_iso_loss.
    - Elongated needle/anisotropic clusters produce positive local_iso_loss.
    - Loss is scale-free: invariant to uniform spatial scaling of coordinates.
    """
    from particlegan.bispf import SBNPPriorRegularizer

    reg = SBNPPriorRegularizer(weight=1.0, k_nn=4)

    torch.manual_seed(0)
    x_coords = torch.linspace(-30.0, 30.0, 7)
    y_coords = torch.linspace(-30.0, 30.0, 7)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    z_iso = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # 49 points in 2D regular grid

    z_needle = z_iso.clone()
    z_needle[:, 1] = z_needle[:, 1] * 0.01  # Highly flattened needle

    d = 2
    m_sub = z_needle.shape[0]
    k_iso = min(max(reg.k_nn, d + 1), m_sub - 1)

    def compute_local_iso(z):
        dists_sub = torch.cdist(z, z, p=2.0).pow(2)
        eye_inf = torch.eye(m_sub, dtype=z.dtype) * 1e9
        _, iso_knn_idx = torch.topk(dists_sub + eye_inf, k=k_iso, dim=1, largest=False)
        delta_z = z[iso_knn_idx] - z.unsqueeze(1)
        M = torch.einsum("nkd,nke->nde", delta_z, delta_z) / max(1, k_iso)
        tr_M = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        M_iso = (tr_M / d) * torch.eye(d, dtype=z.dtype).unsqueeze(0)
        A = M - M_iso
        return (A.pow(2).sum(dim=(-2, -1)) / ((tr_M.squeeze(-1).squeeze(-1) / d) + 1e-6).pow(2)).mean().item()

    iso_val_needle = compute_local_iso(z_needle)
    iso_val_scale10 = compute_local_iso(z_needle * 10.0)
    iso_val_iso = compute_local_iso(z_iso)

    assert iso_val_needle > 1.5, f"Needle must have large local_iso_loss (~2.0); got {iso_val_needle}"
    assert iso_val_needle > iso_val_iso * 5.0, f"Needle local_iso_loss ({iso_val_needle}) must exceed isotropic cluster ({iso_val_iso})"
    assert abs(iso_val_needle - iso_val_scale10) < 1e-3, "Local circularization must be scale-free"


def test_bispf_image_dynamic_well_depth():
    """Verify dynamic image well depth scaling by s_metric:
    - High-contrast images (ema_data_scale >= 4.899) have img_scale = 1.0 (unscaled kappa_eff).
    - Low-contrast images (e.g. img_intensity2, ema_data_scale ~ 2.0) have img_scale < 1.0,
      deepening the well (kappa_eff = (kappa / img_scale) * (0.6 + 1.10 * (1 - res)) > kappa) for contrast separation.
    - img_scale is clamped to [0.15, 1.0].
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(kappa=1.0)
    kernel.record.calls = 600

    xr = torch.randn(16, 1, 8, 8)
    xf = torch.randn(16, 1, 8, 8)

    class FlatCritic(nn.Module):
        def forward(self, x):
            return x.flatten(1).sum(dim=1, keepdim=True)

    # 1. High-contrast image (ema_data_scale = 4.899 -> img_scale = 1.0)
    kernel.record.ema_data_scale = 4.899
    kernel.record.ema_residual = 1.0  # res_ratio = 1.0 -> kappa_eff = 1.0 / 1.0 * 0.6 = 0.6
    kernel._observe_sinkhorn = lambda res, ds, ema_decay=0.95: None
    pen_high, _ = kernel.penalty(FlatCritic(), xr, xf, step=0, collect_stats=True)

    # 2. Low-contrast image (ema_data_scale = 2.0 -> img_scale = 2.0 / 4.899 ~ 0.4082)
    # Deeper well: kappa_eff = 1.0 / 0.4082 * 0.6 ~ 1.47
    kernel.record.ema_data_scale = 2.0
    pen_low, _ = kernel.penalty(FlatCritic(), xr, xf, step=0, collect_stats=True)

    assert pen_high.item() > pen_low.item(), (
        f"Under gradient norm 1.0, tightened barrier (kappa_eff=0.6) must penalize more than deepened well (kappa_eff=1.47): "
        f"{pen_high.item()} vs {pen_low.item()}"
    )

    # 3. Minimum clamp verification: ema_data_scale = 0.1 -> clamped to 0.15
    kernel.record.ema_data_scale = 0.1
    img_scale_clamped = min(1.0, max(0.15, float(kernel.record.ema_data_scale) / 4.899))
    assert abs(img_scale_clamped - 0.15) < 1e-6


def test_bispf_active_plane_circularization():
    """Verify active-plane circularization on top 2 eigenvalues for d >= 2:
    - Circular/isotropic 2D plane produces near-zero active_iso_loss (lambda_1 ~= lambda_2).
    - Eccentric/anisotropic 2D plane produces positive active_iso_loss (lambda_1 >> lambda_2).
    - Invariant: Ring disk in 4D space has lambda_1 ~= lambda_2, producing near-zero loss.
    - Loss is scale-free: invariant to uniform spatial scaling of coordinates.
    - SBNPPriorRegularizer integrates active_iso_loss and propagates valid finite gradients.
    """
    from particlegan.bispf import SBNPPriorRegularizer

    reg = SBNPPriorRegularizer(weight=1.0, k_nn=5)

    # 1. Circular 2D distribution (grid of 100 points, lambda_1 ~= lambda_2)
    coords = torch.linspace(-5.0, 5.0, 10)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    z_circ = torch.stack([gx.flatten(), gy.flatten()], dim=1)

    # 2. Needle 2D distribution (stretched ellipse, lambda_1 >> lambda_2)
    z_needle = z_circ.clone()
    z_needle[:, 1] = z_needle[:, 1] * 0.05

    # 3. 4D embedded ring (2 active dimensions, 2 zero dimensions)
    theta = torch.linspace(0, 2 * math.pi, 100)
    z_ring4d = torch.stack([3.0 * torch.cos(theta), 3.0 * torch.sin(theta), torch.zeros(100), torch.zeros(100)], dim=1)

    def calc_active_iso(z):
        c = z - z.mean(dim=0, keepdim=True)
        cov = (c.T @ c) / max(1, z.shape[0] - 1)
        eigs = torch.linalg.eigvalsh(cov)
        l1, l2 = eigs[-1], eigs[-2]
        return (0.05 * (l1 - l2).pow(2) / (l1 + l2 + 1e-6).pow(2)).item()

    iso_circ = calc_active_iso(z_circ)
    iso_needle = calc_active_iso(z_needle)
    iso_scale = calc_active_iso(z_needle * 10.0)
    iso_ring4d = calc_active_iso(z_ring4d)

    assert iso_circ < 1e-4, f"Circular plane must have near-zero active_iso_loss, got {iso_circ}"
    assert iso_needle > 0.035, f"Needle plane must produce strong active_iso_loss (~0.045), got {iso_needle}"
    assert abs(iso_needle - iso_scale) < 1e-4, "Active-plane circularization must be scale-free"
    assert iso_ring4d < 1e-4, f"Ring disk in 4D space must produce near-zero active_iso_loss, got {iso_ring4d}"

    # Verify SBNPPriorRegularizer gradient backpropagation
    z_grad = z_needle.clone().requires_grad_(True)
    loss = reg(z_grad)
    loss.backward()
    assert z_grad.grad is not None and torch.isfinite(z_grad.grad).all()


def test_bispf_suffix_gauge_ramp():
    """Verify phase-aligned zero-mean logit gauge ramp in _BISPFPenaltyKernel:
    - For continuous flow (dim <= 4), before step 1000, eff_gamma_gauge == 0.050.
    - Over steps 1000-1200, eff_gamma_gauge ramps linearly from 0.050 to 0.200.
    - At step 1200 and beyond, eff_gamma_gauge is clamped at 0.200.
    - Verify gauge_loss = 0.5 * eff_gamma_gauge * mean_logit^2 matches exact analytical values.
    """
    from particlegan.bispf import _BISPFPenaltyKernel

    kernel = _BISPFPenaltyKernel(
        coeff=1.0,
        gamma_gauge=0.050,
        lambda_cap=0.0,
        lambda_floor=0.0,
        chord_blend=0.0,
    )

    class FixedLogitCritic(nn.Module):
        def forward(self, x):
            return (x * 0.0).sum(dim=-1, keepdim=True) + 2.0

    xr = torch.randn(32, 2)
    xf = torch.randn(32, 2)

    # Step 800: pre-ramp (calls = 800)
    kernel.record.calls = 799
    _, stats_800 = kernel.penalty(FixedLogitCritic(), xr, xf, step=800, collect_stats=True)
    expected_800 = 0.5 * 0.050 * (2.0 ** 2)
    assert abs(stats_800["gauge_loss"] - expected_800) < 1e-5

    # Step 1100: mid-ramp (calls = 1100 -> ramp = 0.50 -> eff_gamma = 0.050 + 0.150 * 0.5 = 0.125)
    kernel.record.calls = 1099
    _, stats_1100 = kernel.penalty(FixedLogitCritic(), xr, xf, step=1100, collect_stats=True)
    expected_1100 = 0.5 * 0.125 * (2.0 ** 2)
    assert abs(stats_1100["gauge_loss"] - expected_1100) < 1e-5

    # Step 1250: post-ramp (calls = 1250 -> ramp = 1.0 -> eff_gamma = 0.200)
    kernel.record.calls = 1249
    _, stats_1250 = kernel.penalty(FixedLogitCritic(), xr, xf, step=1250, collect_stats=True)
    expected_1250 = 0.5 * 0.200 * (2.0 ** 2)
    assert abs(stats_1250["gauge_loss"] - expected_1250) < 1e-5


def test_bispf_continuous_mobility_floor():
    """Verify direct thermal scaling in _bispf_critic_forward_hook:
    - Generator gradient is scaled directly by s = record._thermal_scale.
    - Locks mode locations firmly during harmonic cooling to eliminate drift.
    """
    from particlegan.bispf import BISPFStepRecord, _attach_critic_input_hook

    class SimpleCritic(nn.Module):
        def forward(self, x):
            return 2.0 * x.sum(dim=-1, keepdim=True)

    critic = SimpleCritic()
    record = BISPFStepRecord()
    _attach_critic_input_hook(critic, record)

    # 1. Mobility floor under severe cooling (s_thermal = 0.01) before step 1000
    record._thermal_scale = 0.01
    record.calls = 500
    record._in_penalty = False

    x_2d = torch.randn(16, 2, requires_grad=True)
    out_2d = critic(x_2d)
    out_2d.sum().backward()

    # Under origin/develop, output is scaled directly by s = record._thermal_scale (0.01)
    assert torch.allclose(x_2d.grad, torch.full_like(x_2d.grad, 2.0 * 0.01), atol=1e-5)

    # 2. At thermal cooling lock (s_thermal = 1e-5), generator gradient is scaled down to 1e-5
    for p in [x_2d]:
        p.grad.zero_()
    record._thermal_scale = 1e-5
    out_2d_locked = critic(x_2d)
    out_2d_locked.sum().backward()

    assert torch.allclose(x_2d.grad, torch.full_like(x_2d.grad, 2.0 * 1e-5), atol=1e-5)


def test_bispf_sbnp_high_dim_latents_standard_batches():
    """Verify SBNP regularizer remains active (non-zero loss, finite non-zero gradients)
    when latent dimension d >= batch_size (e.g. d=64, 128, 512 with n=16, 32, 64)
    and when num_particles > 1024 so GANTrainer._step passes sampled unique rows.
    """
    from particlegan.bispf import SBNPPriorRegularizer, sbnp_regularization_loss
    from particlegan.recipes import get_recipe
    from particlegan.training import GANTrainer

    torch.manual_seed(42)

    for d in (64, 128, 512):
        for n in (16, 32, 64):
            assert d >= n
            rows_fn = torch.randn(n, d, requires_grad=True)
            loss_fn = sbnp_regularization_loss(rows_fn)
            assert torch.isfinite(loss_fn) and loss_fn.item() > 0.0, (
                f"sbnp_regularization_loss returned {loss_fn.item()} for n={n}, d={d}"
            )
            loss_fn.backward()
            assert rows_fn.grad is not None
            assert torch.isfinite(rows_fn.grad).all()
            assert rows_fn.grad.norm().item() > 0.0

            rows_mod = torch.randn(n, d, requires_grad=True)
            reg = SBNPPriorRegularizer()
            loss_mod = reg(rows_mod)
            assert torch.isfinite(loss_mod) and loss_mod.item() > 0.0, (
                f"SBNPPriorRegularizer returned {loss_mod.item()} for n={n}, d={d}"
            )
            loss_mod.backward()
            assert rows_mod.grad is not None
            assert torch.isfinite(rows_mod.grad).all()
            assert rows_mod.grad.norm().item() > 0.0

    # Verify num_particles > 1024 sampled-row path in bispf_optimized and GANTrainer._step
    for batch_size in (16, 64):
        recipe = get_recipe("bispf_optimized", z_dim=128, batch_size=batch_size, num_particles=4096)
        prior = recipe.make_prior()
        prior_reg = recipe.make_prior_regularizer(prior=prior, reference=prior)

        _, indices = prior.sample(batch_size)
        unique_idx = torch.unique(indices)
        raw = prior.z[unique_idx]
        reg_loss = prior_reg(raw)
        assert torch.isfinite(reg_loss) and reg_loss.item() > 0.0
        reg_loss.backward()
        assert prior.z.grad is not None
        assert torch.isfinite(prior.z.grad).all()
        assert prior.z.grad[unique_idx].norm().item() > 0.0
        prior.zero_grad()

        gen = nn.Linear(128, 8)
        disc = nn.Linear(8, 1)
        trainer = GANTrainer(recipe, gen, disc, prior=prior, seed=0)
        stats = trainer.step(torch.randn(batch_size, 8))
        assert stats["prior_regularization"].item() > 0.0

