"""Public-API software verification and tiered toy benchmark suite for BISPF.

Part A: 64-update public-API FormulationContext / GANTrainer verification
        (seed 0, constant LR, no EMA critic copy, exact midpoint continuation).
Part B: Tiered toy benchmarks (8gaussians, 25gaussians, 100gaussians,
        swiss_roll, two_moons, rings, 25gaussians_scale100x, 8gaussians_imbalanced)
        comparing constant-LR BISPF against constant-LR RpGAN+GradCap baseline
        under protocol seed 0 and deterministic orthogonal initialization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.api import FormulationContext
from particlegan import (
    BISPFEngine,
    BISPFLossAdapter,
    GANLoss,
    StratifiedBlueNoisePrior,
    get_recipe,
    init,
)
from particlegan.grad_regularizers import GradientPenalty
from particlegan.vicreg_loss import ParticleRegularizer


def _equal_state(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal_state(x, y) for x, y in zip(a, b))
    return a == b


# ==============================================================================
# Part A: Public-API FormulationContext / GANTrainer Verification (64 updates)
# ==============================================================================


def _make_api_context(enable_sbnp: bool) -> FormulationContext:
    overrides: Dict[str, Any] = dict(
        num_particles=64,
        z_dim=2,
        batch_size=32,
        total_steps=64,
        prior_regularizer="sbnp" if enable_sbnp else "none",
        prior_reg=1.0 if enable_sbnp else 0.0,
    )
    return FormulationContext(
        recipe_preset="bispf",
        recipe_overrides=overrides,
        prior=dict(
            kind="particle_cloud",
            sigma=0,
            standardize=False,
            exception_reason="BISPF SBNP particle cloud software verification",
        ),
        prior_contract="recipe_owned_v1",
        seed=0,
    )


def _make_api_trainer(ctx: FormulationContext):
    g = ctx.construct(
        lambda: nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2)),
        component="generator",
    )
    d = ctx.construct(
        lambda: nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1)),
        component="discriminator",
    )
    return ctx.build_trainer(g, d)


def _api_batch(ctx: FormulationContext) -> torch.Tensor:
    gen = ctx.streams.generator("data", component="bispf_api_demo", purpose="batches")
    angles = torch.randint(0, 8, (32,), generator=gen, dtype=torch.long).float() * (2.0 * math.pi / 8.0)
    centers = torch.stack([math.sqrt(2.0) * torch.cos(angles), math.sqrt(2.0) * torch.sin(angles)], dim=1)
    noise = 0.08 * torch.randn(32, 2, generator=gen)
    return centers + noise


def _draw_scatter_cell(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    size: int,
    points: torch.Tensor,
    lim: float,
    color: Tuple[int, int, int],
    title: str,
    bg_points: torch.Tensor | None = None,
) -> None:
    draw.rectangle([x0, y0, x0 + size, y0 + size], fill=(248, 249, 250), outline=(180, 185, 195))
    draw.text((x0 + 6, y0 + 4), title, fill=(30, 35, 45))
    plot_top = y0 + 22
    plot_h = size - 26
    plot_w = size - 8
    px0 = x0 + 4
    if bg_points is not None:
        pts_bg = bg_points.detach().cpu()
        for i in range(min(pts_bg.shape[0], 600)):
            u = (float(pts_bg[i, 0]) + lim) / (2.0 * lim)
            v = 1.0 - (float(pts_bg[i, 1]) + lim) / (2.0 * lim)
            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                cx = int(px0 + u * plot_w)
                cy = int(plot_top + v * plot_h)
                draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], outline=(190, 190, 190))
    pts = points.detach().cpu()
    for i in range(min(pts.shape[0], 1000)):
        u = (float(pts[i, 0]) + lim) / (2.0 * lim)
        v = 1.0 - (float(pts[i, 1]) + lim) / (2.0 * lim)
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            cx = int(px0 + u * plot_w)
            cy = int(plot_top + v * plot_h)
            draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=color)


def run_part_a(log_file, gif_path: Path) -> Dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    ctrl_ctx = _make_api_context(enable_sbnp=False)
    bispf_ctx = _make_api_context(enable_sbnp=True)
    ctrl_trainer = _make_api_trainer(ctrl_ctx)
    bispf_trainer = _make_api_trainer(bispf_ctx)

    initial_equal = all(
        _equal_state(getattr(ctrl_trainer, k).state_dict(), getattr(bispf_trainer, k).state_dict())
        for k in ("G", "D", "prior")
    )
    no_ema_critic = bispf_trainer.ema_D is None and bispf_trainer.opt_d.ema_critic is None
    initial_locations = bispf_trainer.prior.z.detach().clone()

    target_gen = torch.Generator().manual_seed(0)
    target_angles = torch.randint(0, 8, (512,), generator=target_gen).float() * (2.0 * math.pi / 8.0)
    target_pts = torch.stack(
        [math.sqrt(2.0) * torch.cos(target_angles), math.sqrt(2.0) * torch.sin(target_angles)], dim=1
    ) + 0.08 * torch.randn(512, 2, generator=target_gen)

    frames: List[Image.Image] = []
    events: List[Dict[str, Any]] = []
    before = bispf_trainer.prior.z.detach().clone()
    data_hash = hashlib.sha256()
    saved_midpoint = None
    steps = 64

    for update in range(1, steps + 1):
        batch_a = _api_batch(ctrl_ctx)
        batch_b = _api_batch(bispf_ctx)
        assert torch.equal(batch_a, batch_b)
        data_hash.update(batch_a.numpy().tobytes())
        ctrl_trainer.step(batch_a)
        values = bispf_trainer.step(batch_b)
        displacement = (bispf_trainer.prior.z.detach() - before).norm(dim=1)
        before = bispf_trainer.prior.z.detach().clone()

        if update % 8 == 0:
            rec = bispf_trainer.opt_d.record
            event = dict(
                phase="part_a_public_api",
                step=update,
                finite=all(
                    torch.isfinite(p).all().item()
                    for model in (bispf_trainer.G, bispf_trainer.D, bispf_trainer.prior)
                    for p in model.parameters()
                ),
                d_loss=float(values["loss_d"]),
                g_loss=float(values["loss_g"]),
                grad_pen=float(values["penalty"]),
                prior_regularization=float(values["prior_regularization"]),
                ema_residual=float(rec.ema_residual),
                ema_data_scale=float(rec.ema_data_scale),
                mean_actual_row_displacement=float(displacement.mean()),
                max_actual_row_displacement=float(displacement.max()),
            )
            events.append(event)
            line = json.dumps(event)
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

            outputs = [
                target_pts,
                ctrl_trainer.sample(512),
                bispf_trainer.sample(512),
                bispf_trainer.prior.z.detach(),
            ]
            titles = [
                "Target 8-Gaussians",
                "BISPF (no SBNP)",
                "BISPF + SBNP outputs",
                "SBNP latent cloud",
            ]
            colors = [(43, 92, 143), (217, 95, 2), (27, 158, 119), (117, 112, 179)]
            cell_sz = 200
            img = Image.new("RGB", (4 * cell_sz + 25, cell_sz + 32), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text(
                (8, 4),
                f"Public API GANTrainer (BISPF) update {update}/{steps} | finite={event['finite']} | ema_res={event['ema_residual']:.3f}",
                fill=(20, 20, 20),
            )
            for idx_col, (pts, title, col_rgb) in enumerate(zip(outputs, titles, colors)):
                bg = initial_locations if idx_col == 3 else (target_pts if idx_col in (1, 2) else None)
                _draw_scatter_cell(
                    draw,
                    5 + idx_col * (cell_sz + 5),
                    24,
                    cell_sz,
                    pts,
                    lim=2.5,
                    color=col_rgb,
                    title=title,
                    bg_points=bg,
                )
            frames.append(img)

        if update == steps // 2:
            saved_midpoint = bispf_ctx.state_dict()

    expected_final = bispf_ctx.state_dict()
    restored_ctx = _make_api_context(enable_sbnp=True)
    restored_trainer = _make_api_trainer(restored_ctx)
    restored_ctx.load_state_dict(saved_midpoint)
    for update in range(steps // 2 + 1, steps + 1):
        restored_trainer.step(_api_batch(restored_ctx))
        if update % 8 == 0:
            restored_trainer.sample(512)

    exact_restore = _equal_state(expected_final, restored_ctx.state_dict())
    finite_all = all(e["finite"] for e in events)
    positive_sbnp = all(e["prior_regularization"] > 0 for e in events)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )

    gate = dict(
        initial_tensors_equal=initial_equal,
        no_ema_critic_copy=no_ema_critic,
        finite_all_8_observations=finite_all,
        positive_sbnp_energy_all_8_observations=positive_sbnp,
        exact_checkpoint_continuation=exact_restore,
    )
    return dict(
        status="PASS" if all(gate.values()) else "FAIL",
        gate=gate,
        recipe=bispf_trainer.recipe.to_dict(),
        data_batch_sha256=data_hash.hexdigest(),
        final_observation=events[-1],
    )


# ==============================================================================
# Part B: Tiered Toy Benchmark Suite (Constant LR, No EMA Critic Copy, Seed 0)
# ==============================================================================


class BenchmarkSpec:
    """Specification and ground-truth sampler/scorer for a 2D toy benchmark."""

    def __init__(self, name: str, kind: str, sigma: float, scale: float = 1.0) -> None:
        self.name = name
        self.kind = kind  # "gaussian" or "manifold"
        self.sigma = sigma * scale
        self.scale = scale
        self.centers, self.weights = self._build_centers_and_weights()

    def _build_centers_and_weights(self) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.name in ("8gaussians", "8gaussians_imbalanced"):
            angles = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
            centers = math.sqrt(2.0) * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * self.scale
            if self.name == "8gaussians_imbalanced":
                weights = torch.tensor([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05], dtype=torch.float32)
            else:
                weights = torch.full((8,), 1.0 / 8.0, dtype=torch.float32)
            return centers, weights
        if self.name in ("25gaussians", "25gaussians_scale100x"):
            coords = torch.linspace(-2.0, 2.0, 5, dtype=torch.float32) * self.scale
            gx, gy = torch.meshgrid(coords, coords, indexing="ij")
            centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
            weights = torch.full((25,), 1.0 / 25.0, dtype=torch.float32)
            return centers, weights
        if self.name == "100gaussians":
            coords = torch.linspace(-4.5, 4.5, 10, dtype=torch.float32) * self.scale
            gx, gy = torch.meshgrid(coords, coords, indexing="ij")
            centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
            weights = torch.full((100,), 1.0 / 100.0, dtype=torch.float32)
            return centers, weights
        return None, None

    def sample(self, n: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
        if self.kind == "gaussian":
            assert self.centers is not None and self.weights is not None
            idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
            c = self.centers[idx].to(device)
            noise = torch.randn(n, 2, generator=generator, dtype=torch.float32).to(device) * self.sigma
            return c + noise
        if self.name == "swiss_roll":
            u = torch.rand(n, generator=generator, dtype=torch.float32)
            t = 1.5 * math.pi * (1.0 + 2.0 * u)
            x = t * torch.cos(t) / 7.0
            y = t * torch.sin(t) / 7.0
            pts = torch.stack([x, y], dim=1).to(device)
            noise = torch.randn(n, 2, generator=generator, dtype=torch.float32).to(device) * self.sigma
            return pts + noise
        if self.name == "two_moons":
            n1 = n // 2
            n2 = n - n1
            t1 = torch.rand(n1, generator=generator, dtype=torch.float32) * math.pi
            t2 = torch.rand(n2, generator=generator, dtype=torch.float32) * math.pi
            m1 = torch.stack([torch.cos(t1) - 0.5, torch.sin(t1) - 0.25], dim=1)
            m2 = torch.stack([1.0 - torch.cos(t2) - 0.5, -torch.sin(t2) + 0.25], dim=1)
            pts = torch.cat([m1, m2], dim=0)
            perm = torch.randperm(n, generator=generator)
            pts = pts[perm].to(device)
            noise = torch.randn(n, 2, generator=generator, dtype=torch.float32).to(device) * self.sigma
            return pts + noise
        if self.name == "rings":
            radii = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
            r_idx = torch.randint(0, 4, (n,), generator=generator)
            r = radii[r_idx]
            theta = torch.rand(n, generator=generator, dtype=torch.float32) * (2.0 * math.pi)
            pts = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).to(device)
            noise = torch.randn(n, 2, generator=generator, dtype=torch.float32).to(device) * self.sigma
            return pts + noise
        raise ValueError(f"Unknown benchmark {self.name}")


def _sliced_w1(x: torch.Tensor, y: torch.Tensor, num_projections: int = 64) -> float:
    gen = torch.Generator(device="cpu").manual_seed(0)
    angles = torch.linspace(0.0, math.pi, num_projections, dtype=torch.float32)
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(x.device)
    proj_x = torch.sort(x @ dirs.t(), dim=0)[0]
    proj_y = torch.sort(y @ dirs.t(), dim=0)[0]
    return float((proj_x - proj_y).abs().mean().item())


def evaluate_benchmark(
    spec: BenchmarkSpec,
    x_fake: torch.Tensor,
    x_real_eval: torch.Tensor,
    D: nn.Module,
) -> Dict[str, Any]:
    # Compute critic gradient norm at generated samples
    x_probe = x_fake[:512].detach().requires_grad_(True)
    d_out = D(x_probe).sum()
    grad_x = torch.autograd.grad(d_out, x_probe, create_graph=False)[0]
    grad_norms = grad_x.norm(dim=1).detach()
    mean_grad_norm = float(grad_norms.mean().item())
    median_grad_norm = float(grad_norms.median().item())

    with torch.no_grad():
        scale = spec.scale
        w1 = _sliced_w1(x_fake / scale, x_real_eval / scale)
        threshold_3sigma = 3.0 * spec.sigma

        if spec.kind == "gaussian":
            assert spec.centers is not None and spec.weights is not None
            centers = spec.centers.to(x_fake.device)
            dists = torch.cdist(x_fake, centers, p=2.0)  # (N, M)
            min_dists, nearest_mode = dists.min(dim=1)
            within_3sigma = min_dists <= threshold_3sigma
            hq_ratio = float(within_3sigma.float().mean().item())

            num_modes = centers.shape[0]
            mode_counts = torch.zeros(num_modes, device=x_fake.device)
            valid_modes = nearest_mode[within_3sigma]
            if valid_modes.numel() > 0:
                mode_counts.scatter_add_(
                    0, valid_modes, torch.ones_like(valid_modes, dtype=torch.float32)
                )
            modes_covered = int((mode_counts >= 1.0).sum().item())
            expected_min = max(1.0, 0.25 * x_fake.shape[0] * spec.weights.min().item())
            modes_dense = int((mode_counts >= expected_min).sum().item())
            empirical_probs = mode_counts / max(1.0, float(within_3sigma.sum().item()))
            mass_tv = float(0.5 * (empirical_probs.cpu() - spec.weights).abs().sum().item())

            return dict(
                modes_covered=modes_covered,
                modes_dense=modes_dense,
                total_modes=num_modes,
                hq_3sigma_ratio=round(hq_ratio, 4),
                mass_tv=round(mass_tv, 4),
                sliced_w1=round(w1, 4),
                critic_grad_norm_mean=round(mean_grad_norm, 4),
                critic_grad_norm_median=round(median_grad_norm, 4),
            )
        else:
            # Continuous manifold: distance from fake to dense ground-truth reference cloud
            dists_f2r = torch.cdist(x_fake, x_real_eval, p=2.0)
            min_f2r = dists_f2r.min(dim=1)[0]
            hq_ratio = float((min_f2r <= threshold_3sigma).float().mean().item())

            min_r2f = dists_f2r.min(dim=0)[0]
            recall_3sigma = float((min_r2f <= threshold_3sigma).float().mean().item())

            return dict(
                modes_covered=int(round(recall_3sigma * 100)),
                modes_dense=int(round(recall_3sigma * 100)),
                total_modes=100,
                hq_3sigma_ratio=round(hq_ratio, 4),
                recall_3sigma=round(recall_3sigma, 4),
                sliced_w1=round(w1, 4),
                critic_grad_norm_mean=round(mean_grad_norm, 4),
                critic_grad_norm_median=round(median_grad_norm, 4),
            )


def _make_toy_networks(z_dim: int, hidden: int = 128, seed: int = 0, device: torch.device = torch.device("cpu")):
    G = nn.Sequential(
        nn.Linear(z_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 2),
    ).to(device)
    D = nn.Sequential(
        nn.Linear(2, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    init.deterministic_orthogonal_(G, seed=seed)
    init.deterministic_orthogonal_(D, seed=seed + 1)
    return G, D


def run_benchmark_pair(
    spec: BenchmarkSpec,
    device: torch.device,
    steps: int,
    log_file,
) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    z_dim = 8
    num_particles = 2000
    batch_size = 128

    # Evaluation reference set (4000 points, seed 0)
    eval_gen = torch.Generator(device="cpu").manual_seed(0)
    x_real_eval = spec.sample(4000, eval_gen, device)

    # Initial latent table shared identically between baseline and BISPF
    prior_bispf = StratifiedBlueNoisePrior(
        num_particles=num_particles,
        z_dim=z_dim,
        knn_k=4,
        repulsion_weight=0.05,
        var_weight=1.0,
        cov_weight=1.0,
        jitter_scale=0.10,
        seed=0,
    ).to(device)
    init_particles = prior_bispf.particles.detach().clone()

    # 1. Constant-LR Baseline: Global RpGAN + GradCap(c=1.0) + Uniform randint + Var/Cov prior
    G_base, D_base = _make_toy_networks(z_dim=z_dim, hidden=128, seed=0, device=device)
    z_base = nn.Parameter(init_particles.clone())
    opt_G_base = torch.optim.Adam(G_base.parameters(), lr=2e-4, betas=(0.5, 0.99))
    opt_D_base = torch.optim.Adam(D_base.parameters(), lr=2e-4, betas=(0.5, 0.99))
    opt_Z_base = torch.optim.Adam([z_base], lr=2e-3, betas=(0.5, 0.99))
    rp_loss = GANLoss("relativistic")
    grad_cap = GradientPenalty(coeff=5.0, kappa=1.0, lazy_k=1, anchor_weight=0.0, arm="b_cap")
    vic_reg = ParticleRegularizer(weight=1.0)

    # 2. Constant-LR BISPF: SRpGAN + EBI-TCR + Zero-Mean Gauge + SBNP + RDTN (no EMA critic)
    G_bispf, D_bispf = _make_toy_networks(z_dim=z_dim, hidden=128, seed=0, device=device)
    assert _equal_state(G_base.state_dict(), G_bispf.state_dict())
    assert _equal_state(D_base.state_dict(), D_bispf.state_dict())

    engine = BISPFEngine(
        sinkhorn_eps=0.15,
        sinkhorn_iters=5,
        lambda_floor=0.05,
        lambda_cap=10.0,
        grad_cap_c=1.0,
        ebi_beta=10.0,
        gamma_gauge=0.1,
        sigma_0=0.10,
        ema_decay=0.98,
    ).to(device)
    adapter = BISPFLossAdapter(engine)
    opt_G_bispf, opt_D_bispf, opt_Z_bispf = engine.configure_optimizers(
        G_bispf, D_bispf, prior_bispf, lr_g=2e-4, lr_d=2e-4, lr_z=2e-3, betas=(0.5, 0.99)
    )

    data_gen_base = torch.Generator(device="cpu").manual_seed(100)
    data_gen_bispf = torch.Generator(device="cpu").manual_seed(100)
    idx_gen_base = torch.Generator(device="cpu").manual_seed(200)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        real_d_base = spec.sample(batch_size, data_gen_base, device)
        real_g_base = spec.sample(batch_size, data_gen_base, device)
        real_d_bispf = spec.sample(batch_size, data_gen_bispf, device)
        real_g_bispf = spec.sample(batch_size, data_gen_bispf, device)

        # --- Step Baseline (Constant LR, Global RpGAN + GradCap) ---
        idx_d = torch.randint(0, num_particles, (batch_size,), generator=idx_gen_base).to(device)
        x_r_req = real_d_base.detach().requires_grad_(True)
        x_f_req = G_base(z_base[idx_d].detach()).requires_grad_(True)
        d_r = D_base(x_r_req)
        d_f = D_base(x_f_req)
        loss_d_base = rp_loss.d_loss(d_r, d_f)
        pen_base, _ = grad_cap.penalty(D_base, x_r_req, x_f_req, step)
        opt_D_base.zero_grad(set_to_none=True)
        (loss_d_base + pen_base).backward()
        nn.utils.clip_grad_norm_(D_base.parameters(), max_norm=10.0)
        opt_D_base.step()

        idx_g = torch.randint(0, num_particles, (batch_size,), generator=idx_gen_base).to(device)
        fake_g_base = G_base(z_base[idx_g])
        loss_g_base = rp_loss.g_loss(D_base(fake_g_base), D_base(real_g_base).detach()) + vic_reg(z_base)
        opt_G_base.zero_grad(set_to_none=True)
        opt_Z_base.zero_grad(set_to_none=True)
        loss_g_base.backward()
        nn.utils.clip_grad_norm_(G_base.parameters(), max_norm=10.0)
        opt_G_base.step()
        opt_Z_base.step()

        # --- Step BISPF (Constant LR, SRpGAN + EBI-TCR + SBNP + RDTN) ---
        z_batch_d, _ = prior_bispf.sample_batch(batch_size)
        with torch.no_grad():
            fake_d_bispf = G_bispf(z_batch_d)
        opt_D_bispf.zero_grad(set_to_none=True)
        loss_d_bispf = adapter.d_loss(D_bispf, real_d_bispf, fake_d_bispf)
        loss_d_bispf.backward()
        nn.utils.clip_grad_norm_(D_bispf.parameters(), max_norm=10.0)
        opt_D_bispf.step()

        z_batch_g, _ = prior_bispf.sample_batch(batch_size)
        fake_g_bispf = G_bispf(z_batch_g)
        opt_G_bispf.zero_grad(set_to_none=True)
        opt_Z_bispf.zero_grad(set_to_none=True)
        loss_g_bispf = adapter.g_loss(D_bispf, fake_g_bispf, real_g_bispf, prior=prior_bispf)
        loss_g_bispf.backward()
        nn.utils.clip_grad_norm_(G_bispf.parameters(), max_norm=10.0)
        opt_G_bispf.step()
        opt_Z_bispf.step()

        if step % 1000 == 0 or step == steps:
            with torch.no_grad():
                # Evaluate all 2000 particles x 2 draws = 4000 clean samples
                z_eval_base = z_base.repeat(2, 1)
                x_eval_base = G_base(z_eval_base)
                z_eval_bispf = prior_bispf.particles.repeat(2, 1)
                x_eval_bispf = G_bispf(z_eval_bispf)

            m_base = evaluate_benchmark(spec, x_eval_base, x_real_eval, D_base)
            m_bispf = evaluate_benchmark(spec, x_eval_bispf, x_real_eval, D_bispf)
            m_bispf["thermal_ratio"] = round(float(adapter.last_d_stats["bispf/thermal_ratio"]), 4)
            elapsed = round(time.perf_counter() - t0, 2)
            record = dict(
                phase="part_b_benchmark",
                benchmark=spec.name,
                step=step,
                elapsed_s=elapsed,
                bispf=m_bispf,
                baseline_rp_gradcap=m_base,
            )
            line = json.dumps(record)
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

    return m_bispf, m_base, x_real_eval.detach().cpu(), x_eval_base.detach().cpu(), x_eval_bispf.detach().cpu()


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/bispf")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports/forge/bispf"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "verify_bispf.jsonl"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    with log_path.open("w", encoding="utf-8") as log_file:
        # Part A: Public-API GANTrainer / FormulationContext verification
        part_a = run_part_a(log_file, report_dir / "actual-training.gif")

        # Part B: Tiered Toy Benchmarks
        torch.use_deterministic_algorithms(False)
        specs = [
            BenchmarkSpec("8gaussians", "gaussian", sigma=0.05),
            BenchmarkSpec("25gaussians", "gaussian", sigma=0.05),
            BenchmarkSpec("100gaussians", "gaussian", sigma=0.05),
            BenchmarkSpec("swiss_roll", "manifold", sigma=0.05),
            BenchmarkSpec("two_moons", "manifold", sigma=0.05),
            BenchmarkSpec("rings", "manifold", sigma=0.04),
            BenchmarkSpec("25gaussians_scale100x", "gaussian", sigma=0.05, scale=100.0),
            BenchmarkSpec("8gaussians_imbalanced", "gaussian", sigma=0.05),
        ]

        benchmark_results: Dict[str, Any] = {}
        plot_panels: List[Tuple[str, float, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]] = []

        for spec in specs:
            m_bispf, m_base, pts_real, pts_base, pts_bispf = run_benchmark_pair(
                spec, device=device, steps=steps, log_file=log_file
            )
            passed = (
                m_bispf["modes_covered"] == m_bispf["total_modes"]
                if spec.kind == "gaussian"
                else m_bispf["recall_3sigma"] >= 0.95
            ) and m_bispf["hq_3sigma_ratio"] >= 0.90
            benchmark_results[spec.name] = dict(
                status="PASS" if passed else "FAIL",
                scale=spec.scale,
                sigma=spec.sigma,
                bispf=m_bispf,
                baseline_rp_gradcap=m_base,
            )
            if len(plot_panels) < 6:
                plot_panels.append((spec.name, spec.scale, pts_real, pts_base, pts_bispf, m_base, m_bispf))

        # Render comparison figure with PIL
        cell_sz = 190
        grid_img = Image.new("RGB", (6 * (cell_sz + 6) + 6, 3 * (cell_sz + 6) + 34), (255, 255, 255))
        grid_draw = ImageDraw.Draw(grid_img)
        grid_draw.text(
            (10, 6),
            f"Constant-LR Benchmark Comparison (Seed 0, {steps} steps, No EMA Critic Copy) | Top: Target | Mid: Const-LR RpGAN+GradCap | Bot: Const-LR BISPF",
            fill=(20, 20, 20),
        )
        for col, (name, scale, pts_real, pts_base, pts_bispf, m_base, m_bispf) in enumerate(plot_panels):
            r = pts_real / scale
            b = pts_base / scale
            p = pts_bispf / scale
            lim = 5.5 if "100" in name else 2.6
            x_off = 6 + col * (cell_sz + 6)
            _draw_scatter_cell(
                grid_draw, x_off, 28, cell_sz, r, lim=lim, color=(43, 92, 143), title=f"{name} (Target)"
            )
            _draw_scatter_cell(
                grid_draw,
                x_off,
                28 + (cell_sz + 6),
                cell_sz,
                b,
                lim=lim,
                color=(217, 95, 2),
                title=f"Rp+Cap HQ={m_base['hq_3sigma_ratio']:.0%} W1={m_base['sliced_w1']:.2f}",
                bg_points=r,
            )
            _draw_scatter_cell(
                grid_draw,
                x_off,
                28 + 2 * (cell_sz + 6),
                cell_sz,
                p,
                lim=lim,
                color=(27, 158, 119),
                title=f"BISPF HQ={m_bispf['hq_3sigma_ratio']:.0%} W1={m_bispf['sliced_w1']:.2f}",
                bg_points=r,
            )
        grid_img.save(report_dir / "benchmark-comparison.png")

        receipt = dict(
            schema="pg_bispf_verification_v1",
            seed=0,
            device=str(device),
            steps_benchmark=steps,
            part_a_public_api=part_a,
            part_b_benchmarks=benchmark_results,
            input_source_sha256={
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "particlegan/bispf.py",
                    "particlegan/recipes.py",
                    "particlegan/training.py",
                    "reports/forge/bispf/verify_bispf.py",
                )
            },
        )
        (report_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(part_a_status=part_a["status"], benchmarks={k: v["status"] for k, v in benchmark_results.items()})), flush=True)


if __name__ == "__main__":
    main()
