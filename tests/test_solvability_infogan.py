"""Three-level metric solvability and multi-family trainer audit for InfoGAN (Track D).

Verifies:
- Level 1 (Oracle Statistical Headroom): 200 independent draws of N=4096 from the
  8-mode rotated anisotropic Gaussian ring pass all thresholds with >= 2.0x 3-sigma
  noise headroom.
- Level 2 (Constructive Capacity Witness): Genuine InfoGANGenerator and
  InfoGANRecognition nn.Module instances trained without identity bypass pass all
  thresholds in score_infogan.
- Level 3 (Destructive Negative Controls): Deterministic failure on each individual
  threshold (mode drop -> modes, off-manifold noise -> hq, unbalanced weights -> mass_tv,
  random code-to-mode permutation -> discrete_nmi/q_acc, ignored continuous code ->
  continuous_r2, and NaN/Inf -> nonfinite_output).
- Multi-family execution test: 4 steps of bispf, atlas (with ManipulatedPrior +
  NormalNoiseManipulator), bcap, k3p, and ka2 all run cleanly and update
  unwrap_prior(fixture.prior).z.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    NormalNoiseManipulator,
    ParticlePrior,
    get_recipe,
    init,
    unwrap_prior,
)
from benchmarks.toy_audit.api_infogan import (
    CASE_ID,
    NUM_MODES,
    SIGMA_PARALLEL,
    SIGMA_PERP,
    InfoGANAnisotropicFixture,
    InfoGANGenerator,
    InfoGANRecognition,
    anisotropic_ring_geometry,
    build_case,
    sample_infogan_target,
    score_infogan,
)


THRESHOLDS = {
    "modes": (">=", 8.0),
    "hq": (">=", 0.85),
    "mass_tv": ("<=", 0.12),
    "discrete_nmi": (">=", 0.85),
    "q_acc": (">=", 0.90),
    "continuous_r2": (">=", 0.65),
}


def _compute_headroom(values: list[float], op: str, limit: float) -> tuple[float, float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=True).item())
    margin = (mean - limit) if op == ">=" else (limit - mean)
    denom = 3.0 * std
    headroom = float("inf") if denom <= 1e-12 else (margin / denom)
    return mean, std, headroom


def test_level1_oracle_infogan_200_draws():
    """Level 1: 200/200 Oracle draws pass all thresholds with >= 2.0x noise headroom."""
    n_draws = 200
    n_samples = 4096
    centers, tangents, normals = anisotropic_ring_geometry(device="cpu")

    metric_history: dict[str, list[float]] = {k: [] for k in THRESHOLDS}
    pass_count = 0

    for i in range(n_draws):
        rng = torch.Generator().manual_seed(5000 + i)
        samples, mode_ids, u_parallel = sample_infogan_target(
            n_samples, generator=rng, device="cpu", return_codes=True
        )
        # Bayes-optimal recognition log-posteriors across the 8 anisotropic Gaussian modes
        diff = samples.unsqueeze(1) - centers.unsqueeze(0)
        proj_par = (diff * tangents.unsqueeze(0)).sum(dim=-1)
        proj_perp = (diff * normals.unsqueeze(0)).sum(dim=-1)
        q_logits = -0.5 * (
            (proj_par / SIGMA_PARALLEL) ** 2 + (proj_perp / SIGMA_PERP) ** 2
        )

        res = score_infogan(
            samples,
            mode_ids,
            u_parallel,
            q_logits=q_logits,
            target=samples,
        )
        if res["passed"]:
            pass_count += 1
        for k in THRESHOLDS:
            metric_history[k].append(float(res["metrics"][k]))

    assert pass_count == n_draws, f"Expected 200/200 oracle passes, got {pass_count}/{n_draws}"

    for k, (op, limit) in THRESHOLDS.items():
        mean, std, headroom = _compute_headroom(metric_history[k], op, limit)
        assert headroom >= 2.0, (
            f"Metric {k} failed 2.0x noise headroom: mean={mean:.4f}, std={std:.4f}, "
            f"limit={op}{limit}, headroom={headroom:.2f}x"
        )


def test_level2_constructive_capacity_witness_infogan():
    """Level 2: Constructive nn.Module capacity witness passes all thresholds without identity bypass."""
    torch.manual_seed(0)
    G = InfoGANGenerator(z_dim=2, n_disc=NUM_MODES, n_cont=1, out_dim=2, hidden=128)
    Q = InfoGANRecognition(in_dim=2, n_disc=NUM_MODES, n_cont=1, hidden=128)
    prior = MoGParticlePrior(num_particles=256, z_dim=2, sigma=0.05, standardize=False)

    init.deterministic_orthogonal_(G, seed=11)
    init.deterministic_orthogonal_(Q, seed=12)
    init.deterministic_orthogonal_(prior, seed=13)

    centers, tangents, normals = anisotropic_ring_geometry(device="cpu")
    opt = torch.optim.Adam(list(G.parameters()) + list(Q.parameters()), lr=3e-3)

    train_rng = torch.Generator().manual_seed(2024)
    noise_rng = torch.Generator().manual_seed(2025)

    for _ in range(350):
        z, idx = prior.sample(256, generator=train_rng, noise_generator=noise_rng)
        c_idx = (idx % NUM_MODES).long()
        c_disc = F.one_hot(c_idx, num_classes=NUM_MODES).float()
        c_cont = torch.randn(256, 1, generator=train_rng)
        u_perp = torch.tanh(z[:, :1])

        target_x = (
            centers[c_idx]
            + (SIGMA_PARALLEL * c_cont) * tangents[c_idx]
            + (SIGMA_PERP * u_perp) * normals[c_idx]
        )

        pred_x = G(torch.cat([z, c_disc, c_cont], dim=1))
        q_out = Q(pred_x)
        loss_g = F.mse_loss(pred_x, target_x)
        loss_q_disc = F.cross_entropy(q_out[:, :NUM_MODES], c_idx)
        loss_q_cont = F.mse_loss(q_out[:, NUM_MODES : NUM_MODES + 1], c_cont)
        loss = loss_g + 0.1 * (loss_q_disc + loss_q_cont)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    eval_rng = torch.Generator().manual_seed(9001)
    eval_noise_rng = torch.Generator().manual_seed(9002)
    with torch.no_grad():
        z_eval, idx_eval = prior.sample(4096, generator=eval_rng, noise_generator=eval_noise_rng)
        c_idx_eval = (idx_eval % NUM_MODES).long()
        c_disc_eval = F.one_hot(c_idx_eval, num_classes=NUM_MODES).float()
        c_cont_eval = torch.randn(4096, 1, generator=eval_rng)
        samples = G(torch.cat([z_eval, c_disc_eval, c_cont_eval], dim=1))
        q_out_eval = Q(samples)
        target_eval = sample_infogan_target(4096, generator=eval_rng, device="cpu")

    res = score_infogan(samples, c_idx_eval, c_cont_eval, q_logits=q_out_eval, target=target_eval)
    assert res["passed"], f"Constructive witness failed bounds: {res['failed_bounds']} ({res['metrics']})"


def test_level3_destructive_negative_controls_infogan():
    """Level 3: Destructive negative controls deterministically fail each individual threshold."""
    n_samples = 4096
    rng = torch.Generator().manual_seed(31415)
    centers, tangents, normals = anisotropic_ring_geometry(device="cpu")

    base_samples, base_c_idx, base_u_par = sample_infogan_target(
        n_samples, generator=rng, device="cpu", return_codes=True
    )
    base_q_logits = F.one_hot(base_c_idx, num_classes=NUM_MODES).float() * 10.0

    # 1. Mode drop: collapse mode 7 onto mode 0 -> fails modes == 8
    dropped_idx = base_c_idx.clone()
    dropped_idx[dropped_idx == 7] = 0
    u_perp = torch.randn(n_samples, 1, generator=rng)
    dropped_samples = (
        centers[dropped_idx]
        + (SIGMA_PARALLEL * base_u_par.unsqueeze(1)) * tangents[dropped_idx]
        + (SIGMA_PERP * u_perp) * normals[dropped_idx]
    )
    res_mode_drop = score_infogan(
        dropped_samples,
        dropped_idx,
        base_u_par,
        q_logits=F.one_hot(dropped_idx, num_classes=NUM_MODES).float() * 10.0,
        target=base_samples,
    )
    assert not res_mode_drop["passed"]
    assert "modes" in res_mode_drop["failed_bounds"]
    assert res_mode_drop["metrics"]["modes"] == 7

    # 2. Off-manifold radial blowup -> fails hq >= 0.85 while preserving modes, mass_tv, discrete_nmi, continuous_r2
    bad_hq_perp = torch.empty(n_samples, 1).fill_(6.0 * SIGMA_PERP)
    # Keep 60 HQ samples per mode so modes == 8 still holds, isolating the hq failure
    for k in range(NUM_MODES):
        k_indices = (base_c_idx == k).nonzero(as_tuple=False).squeeze(-1)[:60]
        bad_hq_perp[k_indices] = 0.0
    bad_hq_samples = (
        centers[base_c_idx]
        + (SIGMA_PARALLEL * base_u_par.unsqueeze(1)) * tangents[base_c_idx]
        + bad_hq_perp * normals[base_c_idx]
    )
    res_bad_hq = score_infogan(
        bad_hq_samples,
        base_c_idx,
        base_u_par,
        q_logits=base_q_logits,
        target=base_samples,
    )
    assert not res_bad_hq["passed"]
    assert "hq" in res_bad_hq["failed_bounds"]

    # 3. Unbalanced mode weights -> fails mass_tv <= 0.12 while keeping >= 60 samples per mode
    probs = torch.tensor([0.51, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
    unbalanced_idx = torch.multinomial(probs, n_samples, replacement=True, generator=rng)
    unbalanced_samples = (
        centers[unbalanced_idx]
        + (SIGMA_PARALLEL * base_u_par.unsqueeze(1)) * tangents[unbalanced_idx]
        + (SIGMA_PERP * u_perp) * normals[unbalanced_idx]
    )
    res_unbalanced = score_infogan(
        unbalanced_samples,
        unbalanced_idx,
        base_u_par,
        q_logits=F.one_hot(unbalanced_idx, num_classes=NUM_MODES).float() * 10.0,
        target=base_samples,
    )
    assert not res_unbalanced["passed"]
    assert "mass_tv" in res_unbalanced["failed_bounds"]

    # 4. Random code-to-mode permutation -> fails discrete_nmi >= 0.85 and q_acc >= 0.90
    shuffled_c_idx = torch.randint(0, NUM_MODES, (n_samples,), generator=rng)
    res_bad_nmi = score_infogan(
        base_samples,
        shuffled_c_idx,
        base_u_par,
        q_logits=base_q_logits,
        target=base_samples,
    )
    assert not res_bad_nmi["passed"]
    assert "discrete_nmi" in res_bad_nmi["failed_bounds"]
    assert "q_acc" in res_bad_nmi["failed_bounds"]

    # 5. Ignored continuous code (uncorrelated c_cont) -> fails continuous_r2 >= 0.65
    uncorrelated_c_cont = torch.randn(n_samples, generator=rng)
    res_bad_r2 = score_infogan(
        base_samples,
        base_c_idx,
        uncorrelated_c_cont,
        q_logits=base_q_logits,
        target=base_samples,
    )
    assert not res_bad_r2["passed"]
    assert res_bad_r2["failed_bounds"] == ["continuous_r2"]

    # 6. Non-finite NaN / Inf outputs -> fails with nonfinite_output
    nan_samples = base_samples.clone()
    nan_samples[0, 0] = float("nan")
    res_nan = score_infogan(nan_samples, base_c_idx, base_u_par, q_logits=base_q_logits, target=base_samples)
    assert not res_nan["passed"]
    assert "nonfinite_output" in res_nan["failed_bounds"]

    inf_samples = base_samples.clone()
    inf_samples[0, 1] = float("inf")
    res_inf = score_infogan(inf_samples, base_c_idx, base_u_par, q_logits=base_q_logits, target=base_samples)
    assert not res_inf["passed"]
    assert "nonfinite_output" in res_inf["failed_bounds"]


def test_multi_family_execution_infogan():
    """Verify 4 steps of all 8 trainer presets (bispf, bispf_optimized, atlas, e22, bcap, bcap_adam, k3p, ka2) run cleanly and update z."""
    families = ["bispf", "bispf_optimized", "atlas", "e22", "bcap", "bcap_adam", "k3p", "ka2"]

    for family in families:
        if family in ("atlas", "e22"):
            prior = ManipulatedPrior(
                ParticlePrior(num_particles=256, z_dim=2),
                [NormalNoiseManipulator(sigma=0.0065)],
            )
            init.deterministic_orthogonal_(prior, seed=4)
        else:
            prior = None

        fixture = InfoGANAnisotropicFixture(
            recipe_name=family,
            prior=prior,
            device="cpu",
            seed=0,
            max_steps=16,
        )
        base_prior = unwrap_prior(fixture.prior)
        z0 = base_prior.z.detach().clone()

        for step_idx in range(4):
            out = fixture.step()
            assert out["step"] == step_idx + 1
            assert all(math.isfinite(v) for k, v in out.items() if k != "step"), (
                f"Non-finite loss in family {family} at step {step_idx + 1}: {out}"
            )

        z1 = unwrap_prior(fixture.prior).z.detach().clone()
        delta = float((z1 - z0).abs().max().item())
        assert delta > 0.0, f"Prior particles did not update under recipe family {family}"

    # Verify rebind_trainer and checkpoint round-trip
    fx = build_case(CASE_ID, recipe_name="bispf", device="cpu", seed=0, max_steps=16)
    fx.step()
    ckpt = fx.state_dict()
    fx_restored = build_case(CASE_ID, recipe_name="bispf", device="cpu", seed=0, max_steps=16)
    fx_restored.load_state_dict(ckpt)
    assert fx_restored.completed_steps == 1
    assert fx_restored.batch_sequence_sha256 == fx.batch_sequence_sha256

    fx.rebind_trainer(
        get_recipe("atlas"),
        prior=ManipulatedPrior(
            ParticlePrior(num_particles=256, z_dim=2),
            [NormalNoiseManipulator(sigma=0.0065)],
        ),
    )
    out_rebound = fx.step()
    assert out_rebound["step"] == 2
