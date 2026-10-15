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
        (144, (16, 1, 12, 12)),  # 4D conv image tensor
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
        for b in shifts:
            D_b = BiasedCritic(in_feat, b=b)
            pen_b, stats_b = kernel.penalty(D_b, xr, xf, step=0, collect_stats=True)

            # Gauge loss formula: 0.5 * gamma * b^2
            expected_loss = 0.5 * gamma * (b ** 2)
            actual_loss = float(pen_b.item())
            assert abs(actual_loss - expected_loss) < 1e-5, (
                f"{dim_name} with bias={b}: expected {expected_loss}, got {actual_loss}"
            )
            assert abs(stats_b["mean_logit"] - b) < 1e-5

            # Restoring gradient check
            pen_b.backward()
            bias_grad = D_b.proj.bias.grad.item()
            expected_grad = gamma * b  # d/db [0.5 * gamma * b^2] = gamma * b
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

    class ConstantSlopeImageCritic(nn.Module):
        def forward(self, x):
            return s * x.flatten(1).sum(dim=1, keepdim=True)

    xr_math = torch.randn(8, 1, 8, 8)
    xf_math = torch.randn(8, 1, 8, 8)
    pen_math, _ = kernel_math.penalty(ConstantSlopeImageCritic(), xr_math, xf_math, step=0, collect_stats=True)

    # In RMS coordinates: g_rms = ||g||_2 / sqrt(d) = sqrt(d * s^2) / sqrt(d) = s = 2.0
    # Endpoint Born-Infeld cap penalty: 0.5 * l_cap * (softplus(s - kappa)^2 + softplus(s - kappa)^2)
    expected_cap = l_cap * (F.softplus(torch.tensor(s - kappa), beta=beta).item() ** 2)
    expected_floor = 0.5 * l_floor * (s ** 2)
    expected_total = expected_cap + expected_floor
    assert abs(pen_math.item() - expected_total) < 1e-4, (
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

