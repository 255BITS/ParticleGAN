"""Unit and integration tests for Born-Infeld Sinkhorn Particle Flow (BISPF)."""
from copy import deepcopy
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
