"""Unit and semantic-control tests for Relativistic Hellinger-Born-Infeld Transport (RHBI)."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import (
    GANTrainer,
    InputNoise,
    ParticlePrior,
    RHBICriticAdam,
    RHBICriticPenalty,
    RHBIGANLoss,
    RHBILoss,
    bind_rhbi_record,
    get_recipe,
    init,
)


def _make_mlp_pair(z_dim: int = 2, x_dim: int = 2, hidden: int = 32, seed: int = 0):
    p_rng = torch.Generator().manual_seed(seed + 2)
    G = nn.Sequential(nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, x_dim))
    D = nn.Sequential(nn.Linear(x_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
    init.deterministic_orthogonal_(G, seed=seed)
    init.deterministic_orthogonal_(D, seed=seed + 1)
    prior = ParticlePrior(num_particles=64, z_dim=z_dim)
    with torch.no_grad():
        prior.z.normal_(0.0, 1.0, generator=p_rng)
    return G, D, prior


def test_invariant_1_zero_strength_identity():
    """weight=0.0 leaves buffers untouched and matches pure baseline trajectory bit-for-bit."""
    op = RHBILoss(weight=0.0)
    x_fake = torch.randn(16, 4, requires_grad=True)
    y_real = torch.randn(16, 4)
    d_fake = torch.randn(16, requires_grad=True)
    d_real = torch.randn(16)

    loss, metrics = op(x_fake, y_real, d_fake, d_real)
    assert not bool(op.ell_initialized.item())
    assert float(op.ell_ema.item()) == 0.0
    assert loss.item() == 0.0
    assert metrics["rhbi_active_rate"] == 0.0

    # 10-step trajectory equivalence: bcap vs bcap with rhbi_weight=0.0
    data_rng_a = torch.Generator().manual_seed(42)
    data_rng_b = torch.Generator().manual_seed(42)
    G_a, D_a, P_a = _make_mlp_pair(seed=10)
    G_b, D_b, P_b = deepcopy(G_a), deepcopy(D_a), deepcopy(P_a)

    rec_base = get_recipe("bcap", total_steps=20, batch_size=16, num_particles=64, z_dim=2)
    rec_zero = get_recipe("bcap", total_steps=20, batch_size=16, num_particles=64, z_dim=2, rhbi_weight=0.0)

    tr_a = GANTrainer(rec_base, G_a, D_a, prior=P_a, seed=0)
    tr_b = GANTrainer(rec_zero, G_b, D_b, prior=P_b, seed=0)
    for _ in range(10):
        batch_a = torch.randn(16, 2, generator=data_rng_a)
        batch_b = torch.randn(16, 2, generator=data_rng_b)
        tr_a.step(batch_a)
        tr_b.step(batch_b)

    for pa, pb in zip(G_a.parameters(), G_b.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(D_a.parameters(), D_b.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(P_a.z, P_b.z)


def test_invariant_2_rng_isolation():
    """RHBILoss forward + backward consumes zero global CPU or CUDA RNG draws."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        op = RHBILoss(weight=1.0).to(dev)
        x_fake = torch.randn(32, 3, device=dev, requires_grad=True)
        y_real = torch.randn(32, 3, device=dev)
        d_fake = torch.randn(32, device=dev, requires_grad=True)
        d_real = torch.randn(32, device=dev)

        cpu_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

        loss, _ = op(x_fake, y_real, d_fake, d_real)
        loss.backward()

        cpu_after = torch.get_rng_state()
        assert torch.equal(cpu_before, cpu_after)
        if torch.cuda.is_available():
            cuda_after = torch.cuda.get_rng_state_all()
            for sb, sa in zip(cuda_before, cuda_after):
                assert torch.equal(sb, sa)


def test_invariant_3_permutation_equivariance_and_logit_shift_invariance():
    """Row permutation equivariance and +1000.0 constant critic logit shift invariance."""
    gen = torch.Generator().manual_seed(7)
    x_base = torch.randn(24, 5, generator=gen)
    y_real = torch.randn(24, 5, generator=gen)
    df_base = torch.randn(24, generator=gen)
    dr_base = torch.randn(24, generator=gen)
    perm = torch.randperm(24, generator=gen)

    op1 = RHBILoss(weight=1.0)
    x1 = x_base.clone().requires_grad_(True)
    df1 = df_base.clone().requires_grad_(True)
    loss1, _ = op1(x1, y_real, df1, dr_base)
    loss1.backward()

    # Permuted rows + constant +1000.0 shift on all critic logits
    op2 = RHBILoss(weight=1.0)
    x2 = x_base[perm].clone().requires_grad_(True)
    df2 = (df_base[perm] + 1000.0).clone().requires_grad_(True)
    dr2 = dr_base + 1000.0
    loss2, _ = op2(x2, y_real, df2, dr2)
    loss2.backward()

    assert torch.allclose(loss1, loss2, rtol=1e-4, atol=5e-4)
    assert torch.allclose(x1.grad[perm], x2.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(df1.grad[perm], df2.grad, rtol=1e-4, atol=1e-4)


def test_invariant_4_outlier_coincident_and_singleton_finiteness():
    """float32 finiteness at 1e30 outliers, coincident points x == y, and B = 1."""
    # Case A: Extreme outlier 1e30 in float32
    op = RHBILoss(weight=1.0)
    y_real = torch.randn(16, 4, dtype=torch.float32)
    x_out = y_real.clone()
    x_out[0] = 1e30
    x_out.requires_grad_(True)
    d_fake = torch.zeros(16, dtype=torch.float32, requires_grad=True)
    d_real = torch.zeros(16, dtype=torch.float32)

    loss_out, _ = op(x_out, y_real, d_fake, d_real)
    loss_out.backward()
    assert torch.isfinite(loss_out)
    assert torch.isfinite(x_out.grad).all()
    assert torch.isfinite(d_fake.grad).all()
    assert abs(float(d_fake.grad[0].item())) > 0.01

    # Case A2: Distant tunneling outlier has active transport velocity strictly in (0.5, 1.0]
    op_tun = RHBILoss(weight=1.0)
    x_tun = y_real.clone()
    x_tun[0] = 50.0
    x_tun.requires_grad_(True)
    df_tun = torch.zeros(16, dtype=torch.float32)
    df_tun[0] = -20.0
    df_tun.requires_grad_(True)
    dr_tun = torch.full((16,), 20.0, dtype=torch.float32)
    loss_tun, _ = op_tun(x_tun, y_real, df_tun, dr_tun)
    loss_tun.backward()
    assert 0.5 < float(x_tun.grad[0].norm().item()) * 16.0 <= 1.0

    # Case B: Exact coincident points x_fake == y_real (no 0/0 NaNs)
    op_coin = RHBILoss(weight=1.0)
    x_same = y_real.clone().requires_grad_(True)
    d_f2 = torch.zeros(16, dtype=torch.float32, requires_grad=True)
    loss_same, _ = op_coin(x_same, y_real, d_f2, d_real)
    loss_same.backward()
    assert torch.isfinite(loss_same)
    assert torch.isfinite(x_same.grad).all()

    # Case C: Singleton batch B = 1 with hollow self-coupling
    op_one = RHBILoss(weight=1.0)
    x_one = torch.tensor([[1.0, -2.0]], requires_grad=True)
    y_one = torch.tensor([[0.5, 0.5]])
    df_one = torch.tensor([0.2], requires_grad=True)
    dr_one = torch.tensor([-0.1])
    loss_one, _ = op_one(x_one, y_one, df_one, dr_one)
    loss_one.backward()
    assert torch.isfinite(loss_one)
    assert torch.isfinite(x_one.grad).all()


def test_invariant_5_singleton_equilibrium_bypass_100_gaussians():
    """On 100-Gaussians at B=64 with flat critic, >85% of unmatched singleton mass routes to slack."""
    gen = torch.Generator().manual_seed(0)
    coords = torch.linspace(-9.0, 9.0, 10)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [100, 2], spacing 2.0

    B = 64
    idx_r = torch.randint(0, 100, (B,), generator=gen)
    idx_f = torch.randint(0, 100, (B,), generator=gen)
    y_real = centers[idx_r] + 0.05 * torch.randn(B, 2, generator=gen)
    x_fake = (centers[idx_f] + 0.05 * torch.randn(B, 2, generator=gen)).requires_grad_(True)
    d_fake = torch.zeros(B, requires_grad=True)
    d_real = torch.zeros(B)

    op = RHBILoss(weight=1.0)
    # Warm ell_t for a few real draws from the same 100-mode law so ell_t settles on intra-mode scale
    for _ in range(15):
        idx_w = torch.randint(0, 100, (B,), generator=gen)
        y_w = centers[idx_w] + 0.05 * torch.randn(B, 2, generator=gen)
        op._update_ell(y_w)

    loss, _ = op(x_fake, y_real, d_fake, d_real)
    loss.backward()

    with torch.no_grad():
        nn_to_real = torch.cdist(x_fake.detach(), y_real).amin(dim=1)
        unmatched = nn_to_real > 1.0  # No real draw in the same mode
        assert int(unmatched.sum().item()) >= 15
        # Each row's total marginal is 1/B; slack fraction for row i is B * P_xy[i, B].
        slack_frac = B * op.last_p_xy_slack[unmatched]
        assert float(slack_frac.mean().item()) > 0.85


@pytest.mark.parametrize("family", ["dualnorm", "adam", "formulation"])
def test_exact_checkpoint_continuation_and_tamper_rejection(family: str):
    """Exact bit-for-bit continuation across ell_initialized boundary + tamper rejection."""
    if family == "dualnorm":
        recipe = get_recipe("rhbi", total_steps=20, batch_size=16, num_particles=64, z_dim=2)
    elif family == "adam":
        recipe = get_recipe(
            "bcap_adam",
            critic_formulation="rhbi",
            reg_arm=None,
            rhbi_weight=1.0,
            ebi_beta=4.0,
            total_steps=20,
            batch_size=16,
            num_particles=64,
            z_dim=2,
        )
    else:
        recipe = get_recipe(
            "bispf",
            critic_formulation="rhbi",
            prior_regularizer="moment_tether",
            rhbi_weight=1.0,
            ebi_beta=4.0,
            total_steps=20,
            batch_size=16,
            num_particles=64,
            z_dim=2,
        )

    G1, D1, P1 = _make_mlp_pair(seed=21)
    G2, D2, P2 = _make_mlp_pair(seed=99)  # Distinct initial weights before load_state_dict

    batches = [torch.randn(16, 2, generator=torch.Generator().manual_seed(100 + k)) for k in range(6)]

    tr1 = GANTrainer(recipe, G1, D1, prior=P1, seed=0)
    for b in batches[:3]:
        tr1.step(b)
    ckpt = tr1.state_dict()
    assert tr1.opt_d.record.rhbi_ell_initialized is True
    ell_mid = tr1.opt_d.record.rhbi_ell_ema
    assert ell_mid > 0.0

    for b in batches[3:]:
        tr1.step(b)

    tr2 = GANTrainer(recipe, G2, D2, prior=P2, seed=55)
    # Tampered checkpoint missing RHBI observer/controller fields must be rejected
    bad_ckpt = deepcopy(ckpt)
    if family == "formulation":
        bad_ckpt["optimizers"][1]["regularizer"]["record"]["formulation"] = "bispf"
    else:
        bad_ckpt["optimizers"][1]["regularizer"]["record"].pop("rhbi", None)
    with pytest.raises(ValueError):
        tr2.load_state_dict(bad_ckpt)

    tr2.load_state_dict(ckpt)
    assert tr2.opt_d.record.rhbi_ell_ema == ell_mid
    for b in batches[3:]:
        tr2.step(b)

    for p1, p2 in zip(G1.parameters(), G2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(D1.parameters(), D2.parameters()):
        assert torch.equal(p1, p2)
    assert torch.equal(P1.z, P2.z)
    assert tr1.opt_d.record.rhbi_ell_ema == tr2.opt_d.record.rhbi_ell_ema


def test_hook_capture_views_input_noise_joint_bigan_and_discrete_cond():
    """Verify ring capture across .squeeze(-1).detach(), InputNoise, joint_g_loss, and class conditioning."""
    recipe = get_recipe("rhbi", total_steps=10, batch_size=8, num_particles=32, z_dim=2)

    class CondJointCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(6, 1)
            self.emb = nn.Embedding(4, 2)

        def forward(self, x, z, cls):
            return self.net(torch.cat([x, z, self.emb(cls)], dim=-1))

    D = CondJointCritic()
    opt_d = recipe.make_critic_optimizer(D)
    loss_fn = recipe.make_loss()
    bind_rhbi_record(opt_d.record, loss_fn)
    assert isinstance(loss_fn, RHBIGANLoss)

    # 1. InputNoise records clean x rather than noisy x
    noisy_D = InputNoise(D, std=0.5, generator=torch.Generator().manual_seed(3))
    x_clean = torch.ones(8, 2, requires_grad=True)
    z_clean = torch.zeros(8, 2, requires_grad=True)
    cls_a = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = noisy_D(x_clean, z_clean, cls_a).squeeze(-1)
    x_cap, _, cond_cap, _, _ = loss_fn._resolve_pair(out, out.detach())
    assert torch.equal(x_cap[:, :2], x_clean)
    assert torch.equal(cond_cap.squeeze(-1), cls_a)

    # 2. joint_g_loss delivers finite nonzero gradients to both G and E streams,
    #    even when logits are .squeeze(-1) views
    G = nn.Linear(2, 2)
    E = nn.Linear(2, 2)
    z = torch.randn(8, 2)
    real_x = torch.randn(8, 2)
    fake_x = G(z)
    enc_z = E(real_x)
    fake_logits = D(fake_x, z, cls_a).squeeze(-1)
    real_logits = D(real_x, enc_z, cls_a).squeeze(-1)
    j_loss = loss_fn.joint_g_loss(fake_logits, real_logits)
    j_loss.backward()
    assert G.weight.grad is not None and float(G.weight.grad.abs().sum()) > 0.0
    assert E.weight.grad is not None and float(E.weight.grad.abs().sum()) > 0.0

    # 3. Disjoint discrete class labels route 100% of active coupling into slack
    op = RHBILoss(weight=1.0)
    _, _ = op(
        torch.randn(8, 2, requires_grad=True),
        torch.randn(8, 2),
        torch.zeros(8, requires_grad=True),
        torch.zeros(8),
        cond_fake=torch.zeros(8, dtype=torch.long),
        cond_real=torch.ones(8, dtype=torch.long),
    )
    assert torch.allclose(op.last_p_xy_slack * 8.0, torch.ones(8), atol=1e-5)


def test_semantic_learning_and_zero_preset_hash_drift():
    """Verify RHBI learns a 2D target, deliberate damage degrades quality, and existing presets have zero drift."""
    for preset in ("ka2", "k3p", "bcap", "bispf", "bispf_optimized", "e22", "atlas", "halloween"):
        saved = get_recipe(preset).to_dict()
        assert not any(k.startswith("rhbi_") for k in saved), f"{preset} leaked rhbi_* into to_dict()"

    # Train RHBI briefly on a shifted 2D target and verify deliberate weight damage degrades MSE
    target_mean = torch.tensor([2.5, -2.0])
    G, D, prior = _make_mlp_pair(seed=5)
    recipe = get_recipe("rhbi", total_steps=250, batch_size=32, num_particles=64, z_dim=2)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
    data_rng = torch.Generator().manual_seed(11)
    for _ in range(250):
        batch = target_mean + 0.15 * torch.randn(32, 2, generator=data_rng)
        trainer.step(batch)

    eval_rng = torch.Generator().manual_seed(999)
    samples = trainer.sample(256, generator=eval_rng)
    trained_err = float((samples.mean(dim=0) - target_mean).norm().item())
    assert trained_err < 0.6

    # Deliberate model damage check (AGENTS.md semantic control)
    with torch.no_grad():
        for p in trainer.G.parameters():
            p.mul_(-3.0).add_(4.0)
    damaged_samples = trainer.sample(256, generator=torch.Generator().manual_seed(999))
    damaged_err = float((damaged_samples.mean(dim=0) - target_mean).norm().item())
    assert damaged_err > trained_err + 1.5
    assert isinstance(RHBICriticAdam, type)
    assert isinstance(RHBICriticPenalty, type)


def test_polar_cpu_dense_profile_checkpoint_and_parity():
    """Verify polar_cpu_v1 matches exact CPU polar updates and enforces checkpoint profile identity."""
    from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor

    torch.manual_seed(7)
    w1 = nn.Parameter(torch.randn(16, 8))
    b1 = nn.Parameter(torch.randn(16))
    w2 = nn.Parameter(w1.detach().clone())
    b2 = nn.Parameter(b1.detach().clone())

    opt_prof = NormalizedOptimizer(
        [w1, b1], lr=0.05, smoothing=0.001, dense_profile="polar_cpu_v1"
    )
    opt_ref = NormalizedOptimizer(
        [w2, b2], lr=0.05, smoothing=0.001, svd_backend="cpu"
    )
    g_w = torch.randn_like(w1)
    g_b = torch.randn_like(b1)
    w1.grad, b1.grad = g_w.clone(), g_b.clone()
    w2.grad, b2.grad = g_w.clone(), g_b.clone()
    opt_prof.step()
    opt_ref.step()
    assert torch.equal(w1, w2)
    assert torch.equal(b1, b2)
    assert opt_prof.state_dict()["dualnorm"]["dense_profile"] == "polar_cpu_v1"
    with pytest.raises(ValueError, match="dense profile"):
        opt_ref.load_state_dict(opt_prof.state_dict())
    assert polar_factor(g_w, smoothing=0.001, backend="cpu").shape == w1.shape
