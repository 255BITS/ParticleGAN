"""Unit, continuation, and semantic-control tests for Exponential Moving Prototype Flow (EMPF)."""
from copy import deepcopy
import io
import math
import pytest
import torch
from torch import nn

from particlegan import (
    DEFAULT_RECIPE,
    EMPFCriticAdam,
    EMPFCriticPenalty,
    EMPFPriorRegularizer,
    EMPFSinkhornRelativisticLoss,
    EMPFStepRecord,
    GANTrainer,
    get_recipe,
    init,
)
from particlegan.empf import (
    emeans_effective_alpha,
    mass_weighted_sinkhorn,
    update_emeans_prototypes,
)


def test_default_recipe_and_incumbent_presets_unchanged():
    assert DEFAULT_RECIPE == "bispf_optimized"
    empf = get_recipe("empf")
    assert empf.effective_critic_formulation == "empf"
    assert empf.prior_regularizer == "empf"
    assert empf.optimizer_family == "dualnorm"
    assert empf.reg_anchor_weight == 0.0
    assert empf.lr_floor == 1.0 and empf.network_lr_floor == 1.0

    with pytest.raises(ValueError, match="reg_anchor_weight=0"):
        get_recipe("empf", optimizer_family="formulation", optimizer_smoothing=0.0,
                   optimizer_convolution="none", optimizer_convolution_rule="polar",
                   optimizer_svd_backend="native", training_precision="float32",
                   constraint_geometry_mode="none", reg_anchor_weight=0.5)
    with pytest.raises(ValueError, match="reg_arm"):
        get_recipe("empf", reg_arm="b_cap", optimizer_family="adam",
                   optimizer_smoothing=0.0, optimizer_convolution="none",
                   optimizer_convolution_rule="polar", optimizer_svd_backend="native",
                   training_precision="float32", constraint_geometry_mode="none")


def test_emeans_hit_rate_scaling_and_mass_conserving_revival():
    alpha_small = emeans_effective_alpha(64, 64)
    alpha_large = emeans_effective_alpha(2048, 256)
    assert math.isclose(alpha_small, 0.08, rel_tol=1e-6)
    assert alpha_large > alpha_small
    assert 0.03 <= alpha_small <= 0.35 and 0.03 <= alpha_large <= 0.35

    rng = torch.Generator().manual_seed(0)
    x_init = -5.0 + 0.05 * torch.randn(32, 2, generator=rng)
    c0, w0, p_sum, p_count, phase, revivals = update_emeans_prototypes(
        x_init, None, None, 0.0, 0, max_prototypes=16
    )
    assert c0.shape == (16, 2)
    assert math.isclose(float(w0.sum()), 1.0, rel_tol=1e-6)
    initial_total_mass = float(p_count.sum())
    assert revivals == 0

    # Shift target abruptly to (+5, +5) so 15 of 16 prototypes starve and trigger proportional splits.
    for _ in range(45):
        x_shifted = 5.0 + 0.05 * torch.randn(32, 2, generator=rng)
        c_now, w_now, p_sum, p_count, phase, revivals = update_emeans_prototypes(
            x_shifted, p_sum, p_count, phase, revivals, max_prototypes=16
        )
    assert revivals > 0, "Expected dead prototypes to be revived via proportional donor splitting"
    assert math.isclose(float(p_count.sum()), initial_total_mass, rel_tol=1e-5)
    assert math.isclose(float(w_now.sum()), 1.0, rel_tol=1e-6)
    assert float((c_now - 5.0).norm(dim=1).max()) < 0.5


def test_mass_weighted_sinkhorn_preserves_rare_mode_marginals():
    prototypes = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    weights = torch.tensor([0.98, 0.02], dtype=torch.float32)
    rng = torch.Generator().manual_seed(7)
    x_fake = torch.randn(128, 2, generator=rng)

    pi, norm_cost, residual, data_scale = mass_weighted_sinkhorn(
        prototypes, weights, x_fake, sinkhorn_eps=0.15, sinkhorn_iters=25
    )
    assert pi.shape == (2, 128)
    assert torch.isfinite(norm_cost).all() and torch.isfinite(residual) and float(data_scale) > 0.0
    row_sums = pi.sum(dim=1)
    col_sums = pi.sum(dim=0)
    assert torch.allclose(row_sums, weights, atol=1e-3)
    assert torch.allclose(col_sums, torch.full((128,), 1.0 / 128.0), atol=1e-3)


def test_empf_prior_regularizer_dof_bypass_and_dead_particle_tether():
    reg = EMPFPriorRegularizer(None, weight=1.0, k_nn=5)

    # 1. Tiny unanchored discrete codebook returns exact zero; anchored codebook
    # returns zero when well-separated and repels only colliding prototype pairs.
    tiny = torch.randn(5, 2, requires_grad=True)
    tiny_loss = reg(tiny)
    assert float(tiny_loss.detach()) == 0.0

    recipe5 = get_recipe("empf", num_particles=5, z_dim=2)
    prior5 = recipe5.make_prior()
    init.deterministic_orthogonal_(prior5, seed=0)
    recipe5.apply_prior_policy(prior5)
    reg_anchored = recipe5.make_prior_regularizer(prior=prior5, reference=prior5)
    separated = torch.tensor(
        [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0], [0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    sep_loss = reg_anchored(separated)
    assert float(sep_loss.detach()) == 0.0

    colliding = separated.detach().clone()
    colliding[1] = colliding[0] + torch.tensor([1e-3, 0.0])
    colliding.requires_grad_(True)
    col_loss = reg_anchored(colliding)
    assert float(col_loss.detach()) > 0.0
    col_loss.backward()
    assert colliding.grad is not None and torch.isfinite(colliding.grad).all()
    assert float(colliding.grad[0, 0]) > 0.0 and float(colliding.grad[1, 0]) < 0.0
    assert torch.all(colliding.grad[2:] == 0.0)

    # 2. Spread bimodal cloud vs stranded rogue outlier particle.
    rng = torch.Generator().manual_seed(11)
    left = torch.tensor([-1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    right = torch.tensor([1.5, 0.0]) + 0.05 * torch.randn(16, 2, generator=rng)
    clean_cloud = torch.cat([left, right], dim=0).requires_grad_(True)
    loss_clean = reg(clean_cloud)

    stranded_cloud_data = clean_cloud.detach().clone()
    stranded_cloud_data[0] = torch.tensor([12.0, 0.0])  # Stranded far from both modes
    stranded_cloud = stranded_cloud_data.requires_grad_(True)
    loss_stranded = reg(stranded_cloud)
    assert float(loss_stranded.detach()) > float(loss_clean.detach())
    loss_stranded.backward()
    assert stranded_cloud.grad is not None and torch.isfinite(stranded_cloud.grad).all()
    # Gradient on stranded x=12.0 particle points positively in loss -> negative descent pulls it left toward modes!
    assert float(stranded_cloud.grad[0, 0]) > 0.0


def test_conditional_critic_penalty_preserves_fiber_and_skips_marginal_bank():
    recipe = get_recipe("empf")

    class CondCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))

        def forward(self, x, *, cond):
            return self.net(torch.cat([x, cond], dim=1)).squeeze(-1)

    critic = CondCritic()
    init.deterministic_orthogonal_(critic, seed=0)
    opt_d = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    assert isinstance(penalty, EMPFCriticPenalty)

    rng = torch.Generator().manual_seed(3)
    cond = torch.randn(32, 2, generator=rng)
    real = cond + 0.1 * torch.randn(32, 2, generator=rng)
    fake = cond - 0.1 * torch.randn(32, 2, generator=rng)

    loss = penalty(critic, real, fake, cond=cond)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in critic.parameters())
    # Conditional call must not pollute an unconditional marginal prototype bank.
    assert opt_d.record.proto_initialized is False
    assert opt_d.record.proto_sum is None


@pytest.mark.parametrize("optimizer_family", ["dualnorm", "formulation", "adam"])
def test_empf_checkpoint_continuation_and_corruption_rejection(optimizer_family):
    overrides = dict(
        critic_formulation="empf",
        prior_regularizer="empf",
        prior_reg=0.02,
        optimizer_family=optimizer_family,
        lr=0.004,
        d_lr_mult=1.0,
        prior_lr_mult=0.05 if optimizer_family == "dualnorm" else 1.0,
        lr_floor=1.0,
        network_lr_floor=1.0,
        num_particles=32,
        z_dim=2,
        batch_size=16,
        total_steps=30,
    )
    if optimizer_family != "dualnorm":
        overrides.update(
            optimizer_smoothing=0.0,
            optimizer_convolution="none",
            optimizer_convolution_rule="polar",
            optimizer_svd_backend="native",
            constraint_geometry_mode="none",
        )
    else:
        overrides.update(
            optimizer_smoothing=0.001,
            optimizer_convolution="per_offset",
            optimizer_convolution_rule="quintic",
            constraint_geometry_mode="direction_blend",
        )
    recipe = get_recipe("empf", **overrides)

    def make_models():
        g = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
        d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
        p = recipe.make_prior()
        for idx, mod in enumerate((g, d, p)):
            init.deterministic_orthogonal_(mod, seed=idx)
        return g, d, p

    g1, d1, p1 = make_models()
    g2, d2, p2 = deepcopy(g1), deepcopy(d1), deepcopy(p1)

    t1 = GANTrainer(recipe, g1, d1, prior=p1, seed=0)
    if optimizer_family == "formulation":
        assert isinstance(t1.opt_d, EMPFCriticAdam)
        assert isinstance(t1.opt_d.record, EMPFStepRecord)

    rng1 = torch.Generator().manual_seed(19)
    for step in range(12):
        shift = -2.5 if step < 5 else 2.5
        t1.step(shift + 0.1 * torch.randn(16, 2, generator=rng1))

    # Sampling must not mutate trainer/record state.
    state_before_sample = deepcopy(t1.state_dict())
    _ = t1.sample(24)
    state_after_sample = t1.state_dict()
    assert torch.equal(
        state_before_sample["optimizers"][1]["regularizer"]["record"]["empf"]["proto_sum"],
        state_after_sample["optimizers"][1]["regularizer"]["record"]["empf"]["proto_sum"],
    )
    assert (
        state_before_sample["optimizers"][1]["regularizer"]["record"]["calls"]
        == state_after_sample["optimizers"][1]["regularizer"]["record"]["calls"]
    )

    buf = io.BytesIO()
    torch.save({"trainer": state_after_sample, "rng": rng1.get_state()}, buf)
    buf.seek(0)
    saved = torch.load(buf, weights_only=True)

    for _ in range(8):
        t1.step(2.5 + 0.1 * torch.randn(16, 2, generator=rng1))

    t2 = GANTrainer(recipe, g2, d2, prior=p2, seed=77)
    # Corrupted EMPF packet must be rejected before mutating live state.
    bad_state = deepcopy(saved["trainer"])
    bad_state["optimizers"][1]["regularizer"]["record"]["empf"]["thermal_scale"] = 2.5
    with pytest.raises(ValueError, match="thermal"):
        t2.opt_d.load_state_dict(bad_state["optimizers"][1])
    with pytest.raises(ValueError, match="invalid checkpoint optimizer state|thermal"):
        t2.load_state_dict(bad_state)

    t2.load_state_dict(saved["trainer"])
    rng2 = torch.Generator()
    rng2.set_state(saved["rng"])
    for _ in range(8):
        t2.step(2.5 + 0.1 * torch.randn(16, 2, generator=rng2))

    for pa, pb in zip(t1.G.parameters(), t2.G.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(t1.D.parameters(), t2.D.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(t1.sample(32), t2.sample(32))


def test_empf_relativistic_loss_option():
    recipe = get_recipe("empf", loss="relativistic")
    loss_fn = recipe.make_loss()
    assert isinstance(loss_fn, EMPFSinkhornRelativisticLoss)
    r = torch.randn(16, 1, requires_grad=True)
    f = torch.randn(16, 1, requires_grad=True)
    dl = loss_fn.d_loss(r, f)
    gl = loss_fn.g_loss(f, r.detach())
    assert torch.isfinite(dl) and torch.isfinite(gl)


def test_empf_deliberate_damage_and_instance_isolation():
    """Verify zero cross-instance thermal/prototype leakage and genuine generator serving."""
    recipe = get_recipe(
        "empf",
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=80,
        lr=0.008,
        prior_lr_mult=0.02,
    )

    def build_pair(seed_offset: int):
        g = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
        d = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        p = recipe.make_prior()
        for idx, mod in enumerate((g, d, p)):
            init.deterministic_orthogonal_(mod, seed=seed_offset + idx)
        return GANTrainer(recipe, g, d, prior=p, seed=seed_offset)

    trainer_main = build_pair(0)
    trainer_other = build_pair(100)

    target_center = torch.tensor([2.0, -1.5])
    init_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())

    rng = torch.Generator().manual_seed(42)
    for _ in range(60):
        trainer_main.step(target_center + 0.08 * torch.randn(32, 2, generator=rng))
        # Interleave steps on a wild second trainer to detect any global state leakage.
        trainer_other.step(torch.tensor([-50.0, 50.0]) + 5.0 * torch.randn(32, 2, generator=rng))

    rec_main = trainer_main.opt_d.record
    rec_other = trainer_other.opt_d.record
    assert rec_main is not rec_other
    assert not torch.allclose(rec_main.proto_sum, rec_other.proto_sum)
    assert rec_main.ema_data_scale != rec_other.ema_data_scale

    trained_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert trained_err < 0.5 * init_err, f"Expected EMPF training to reduce mean error ({trained_err} vs {init_err})"

    # Deliberate damage to served generator weights must degrade sample() immediately
    # (proving sample() evaluates G(prior) and never returns critic prototypes).
    with torch.no_grad():
        for param in trainer_main.G.parameters():
            param.fill_(25.0)
    damaged_err = float((trainer_main.sample(128).mean(dim=0) - target_center).norm())
    assert damaged_err > 10.0 * trained_err


def test_empf_fresh_process_import_order_and_incumbent_hashes():
    """Verify import order leaves shared defaults and incumbent recipe digests untouched."""
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[1])
    probe_a = (
        "import json, particlegan.empf, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}}))"
    )
    probe_b = (
        "import json, particlegan.recipes as r; "
        "from experiments.forge.contracts import stable_hash; "
        "from experiments.forge.techniques import technique_signature; "
        "import particlegan.empf; "
        "print(json.dumps({'default': r.DEFAULT_RECIPE, "
        "'hashes': {k: stable_hash(r.get_recipe(k).to_dict()) for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}, "
        "'sigs': {k: technique_signature(r.get_recipe(k))['digest'] for k in ('bispf_optimized', 'bispf', 'bcap', 'empf')}}))"
    )
    out_a = json.loads(subprocess.check_output([sys.executable, "-c", probe_a], cwd=root, text=True))
    out_b = json.loads(subprocess.check_output([sys.executable, "-c", probe_b], cwd=root, text=True))
    assert out_a == out_b
    assert out_a["default"] == "bispf_optimized"
    assert out_a["hashes"]["bispf_optimized"].startswith("a59d78806e67")
    assert out_a["hashes"]["bispf"].startswith("6bb9c45dfbdb")
    assert out_a["hashes"]["bcap"].startswith("1c0aeefa46b0")
    assert out_a["sigs"]["empf"] != out_a["sigs"]["bispf_optimized"]
