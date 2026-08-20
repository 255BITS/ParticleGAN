#!/usr/bin/env python
"""
100gaussians.py

100 Gaussians 2D toy problem with a ParticlePrior + regularizer.

This is deliberately nastier than the 25-Gaussian grid:
  - Data: 100-Gaussian mixture on a 10x10 grid in R^2 with small variance.
  - Prior: lib.particle_prior.ParticlePrior (learnable particles in latent space).
  - G: simple MLP mapping z -> x in R^2.
  - D: simple MLP with Fourier input features, x -> scalar score.
  - Loss: R3GAN-style objective — relativistic pairing (RpGAN) logistic loss
    with R1 + R2 zero-centered gradient penalties.

Fast-convergence recipe (found via large-scale sweep; all 100 modes covered
with >=90% of samples within 3 sigma of a center by ~3.5k steps, vs. never
converging with the old hinge/Adam(0.5) defaults):
  - z_dim 4 (overcomplete latent eases transport; 2 is much worse)
  - Fourier features on D's input so it can resolve the sigma=0.03 modes
    from step 1 (a plain MLP D learns low frequencies first and plateaus)
  - RpGAN + R1/R2 (gamma=0.02): the penalty caps D's steepness at the
    samples, which is what stops the sharp Fourier D from stranding modes
  - Adam beta1=0: each particle only gets a real gradient every ~78 steps,
    so momentum drifts the unsampled rows of the particle table
  - EMA (0.995) copies of G and the prior for snapshots/eval: the live
    weights orbit the equilibrium; the EMA copy sits on it
  - delayed cosine LR anneal: full LR for the first 60% of the run (the
    coverage + sharpening phase), then cosine down to a 5% floor. Without
    the anneal the game can destabilize shortly after convergence; annealing
    from step 0 starves the sharpening phase; annealing to exactly 0 also
    fails — a small residual LR is needed.

Visualization:
  - At fixed intervals, we sample the SAME latent particles (fixed_first_n=True)
    and render a scatter plot of:
        * real samples from the 100-Gaussian mixture (fixed across training),
        * fake samples from the current EMA generator.
  - This makes it easy to turn the sequence of PNGs into a video.
"""

import argparse
import copy
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lib.particle_prior import ParticlePrior
from lib.gan_loss import GANLoss
from lib.vicreg_loss import VICRegLikeLoss

# =========================
#  Simple MLP G / D
# =========================

class SimpleMLPGenerator(nn.Module):
    """
    Very small MLP generator: z -> x in R^2.

    Strong enough for the toy problem but still minimal and CPU-friendly.
    """

    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SimpleMLPDiscriminator(nn.Module):
    """
    Simple MLP discriminator: x in R^2 -> scalar score.

    fourier=K appends sin/cos features at frequencies pi * 2^i (i < K) per
    input dimension. MLPs are spectrally biased toward low frequencies, so
    without this D cannot resolve the sigma=0.03 mode structure until very
    late in training and sample sharpness stalls.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        fourier: int = 2,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        dim = in_dim + (2 * fourier * in_dim if fourier > 0 else 0)
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs  # (B, in_dim, K)
            h = torch.cat([h, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        # Return shape (B,) for convenience.
        return self.net(h).squeeze(-1)


# =========================
#  100 Gaussians dataset
# =========================

def sample_100gaussians(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator = None,
    grid_scale: float = 1.0,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Sample from a 100-Gaussian mixture:
      - Centers on a 10x10 grid at coordinates:
            { -4.5, -3.5, ..., 4.5 } * grid_scale
      - Isotropic Gaussian noise with `std`.

    This is intentionally dense and low-variance to stress-test mode coverage.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if generator is None:
        idx_x = torch.randint(0, 10, (batch_size,), device=device)
        idx_y = torch.randint(0, 10, (batch_size,), device=device)
    else:
        idx_x = torch.randint(0, 10, (batch_size,), device=device, generator=generator)
        idx_y = torch.randint(0, 10, (batch_size,), device=device, generator=generator)

    # Map indices 0..9 to coordinates -4.5..4.5
    centers_x = (idx_x - 4.5) * grid_scale
    centers_y = (idx_y - 4.5) * grid_scale

    centers = torch.stack(
        (centers_x, centers_y),
        dim=1,
    ).to(device=device, dtype=torch.float32)

    if generator is None:
        noise = torch.randn(batch_size, 2, device=device) * std
    else:
        noise = torch.randn(batch_size, 2, device=device, generator=generator) * std

    return centers + noise


# =========================
#  Metrics
# =========================

@torch.no_grad()
def mode_coverage(
    generator: nn.Module,
    prior: ParticlePrior,
    device: torch.device,
    n_eval: int = 20000,
    std: float = 0.03,
    min_count: int = 10,
) -> Tuple[int, float]:
    """
    Coverage metrics over the full particle cloud:
      - modes: number of grid centers with >= min_count "high quality" samples
        (within 3 sigma of the center),
      - hq_frac: fraction of samples that are high quality.
    """
    generator.eval()
    idx = torch.randint(0, prior.num_particles, (n_eval,), device=device)
    fake = generator(prior.z[idx])
    coords = torch.arange(10, device=device, dtype=torch.float32) - 4.5
    cx, cy = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([cx.flatten(), cy.flatten()], dim=1)
    dists = torch.cdist(fake, centers)
    mind, nearest = dists.min(dim=1)
    hq = mind <= 3 * std
    counts = torch.bincount(nearest[hq], minlength=100)
    generator.train()
    return int((counts >= min_count).sum().item()), hq.float().mean().item()


# =========================
#  Visualization
# =========================

def save_fake_scatter(
    generator: nn.Module,
    prior: ParticlePrior,
    device: torch.device,
    filename: str,
    real_samples: torch.Tensor,
    n_fake: int = 4096,
    xlim: Tuple[float, float] = (-6.0, 6.0),
    ylim: Tuple[float, float] = (-6.0, 6.0),
) -> None:
    """
    Save a scatter plot comparing:
      - fixed real samples from the 100-Gaussian mixture,
      - fake samples from a fixed subset of particles (fixed_first_n=True).

    This keeps the visual trajectory consistent across training, which is
    ideal for making a video.
    """
    generator.eval()
    prior.eval()

    with torch.no_grad():
        z_fake, _ = prior.sample(n_fake, fixed_first_n=True)
        z_fake = z_fake.to(device)
        fake = generator(z_fake).cpu()

    real = real_samples.cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.2, label="real")
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.8, label="fake")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("100 Gaussians: real vs. model samples")
    fig.tight_layout()

    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    generator.train()
    prior.train()


# =========================
#  Training
# =========================

def train(
    epochs: int = 5,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    z_dim: int = 4,
    num_particles: int = 20_000,
    lr: float = 3e-4,
    d_lr_mult: float = 1.5,
    beta1: float = 0.0,
    lambda_ep: float = 1.0,
    r1_gamma: float = 0.02,
    fourier: int = 2,
    ema_decay: float = 0.995,
    lr_floor: float = 0.05,
    lr_anneal_start: float = 0.6,
    loss_type: str = "logistic",
    gan_mode: str = "rp",
    out_dir: str = "100gaussians_samples",
    log_interval: int = 100,
    snapshot_interval: int = 500,
    seed: int = 1234,
    device_str: str = None,
):
    # Device / seeds
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        train_gen = torch.Generator(device=device)
        viz_gen = torch.Generator(device=device)
    else:
        train_gen = torch.Generator()
        viz_gen = torch.Generator()
    train_gen.manual_seed(seed)
    viz_gen.manual_seed(seed + 1)

    # Models
    prior = ParticlePrior(num_particles=num_particles, z_dim=z_dim).to(device)
    G = SimpleMLPGenerator(z_dim=z_dim).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, fourier=fourier).to(device)

    for m in list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # EMA copies of G + prior for snapshots/eval; the live weights orbit the
    # equilibrium, the averaged ones sit on it.
    ema_G = copy.deepcopy(G)
    ema_prior = copy.deepcopy(prior)
    for p in list(ema_G.parameters()) + list(ema_prior.parameters()):
        p.requires_grad_(False)

    vic_reg = VICRegLikeLoss()
    gan_loss = GANLoss(loss_type=loss_type, mode=gan_mode)

    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=lr,
        betas=(beta1, 0.999),
    )
    opt_prior = torch.optim.Adam(
        prior.parameters(),
        lr=lr * 10.0,  # Particles need higher mobility
        betas=(beta1, 0.999),
    )
    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=lr * d_lr_mult,
        betas=(beta1, 0.999),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fixed real samples for visualization (same throughout training).
    real_viz = sample_100gaussians(
        batch_size=8192,
        device=device,
        generator=viz_gen,
    )

    # Initial snapshot (untrained model).
    save_fake_scatter(
        ema_G,
        ema_prior,
        device,
        str(out_path / f"samples_step_{0:06d}.png"),
        real_samples=real_viz,
    )

    total_steps = epochs * steps_per_epoch
    base_lrs = {
        id(opt): [g["lr"] for g in opt.param_groups]
        for opt in (opt_G, opt_D, opt_prior)
    }

    global_step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            # Full LR until lr_anneal_start, then cosine down to lr_floor.
            anneal_from = lr_anneal_start * total_steps
            if global_step <= anneal_from:
                scale = 1.0
            else:
                frac = (global_step - anneal_from) / max(1.0, total_steps - anneal_from)
                scale = lr_floor + (1.0 - lr_floor) * 0.5 * (
                    1.0 + math.cos(math.pi * frac)
                )
            for opt in (opt_G, opt_D, opt_prior):
                for group, base in zip(opt.param_groups, base_lrs[id(opt)]):
                    group["lr"] = base * scale

            # -------------------------
            # 1) Discriminator step
            # -------------------------
            D.train()
            G.eval()

            x_real = sample_100gaussians(
                batch_size=batch_size,
                device=device,
                generator=train_gen,
            )
            with torch.no_grad():
                z_fake, _ = prior.sample(batch_size)
                x_fake = G(z_fake)

            if r1_gamma > 0:
                x_real = x_real.detach().requires_grad_(True)
                x_fake = x_fake.detach().requires_grad_(True)

            real_logits = D(x_real)
            fake_logits = D(x_fake)

            loss_d = gan_loss.d_loss(real_logits, fake_logits)

            if r1_gamma > 0:
                # R1 + R2 zero-centered gradient penalties (R3GAN). This is
                # what lets the sharp Fourier D coexist with full mode
                # coverage: it caps D's steepness at the samples.
                grad_real = torch.autograd.grad(
                    real_logits.sum(), x_real, create_graph=True
                )[0]
                grad_fake = torch.autograd.grad(
                    fake_logits.sum(), x_fake, create_graph=True
                )[0]
                r1 = grad_real.pow(2).sum(dim=1).mean()
                r2 = grad_fake.pow(2).sum(dim=1).mean()
                loss_d = loss_d + (r1_gamma / 2.0) * (r1 + r2)

            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # -------------------------
            # 2) Generator + prior step
            # -------------------------
            D.eval()
            G.train()

            z_fake, idx = prior.sample(batch_size)
            x_fake = G(z_fake)
            fake_logits = D(x_fake)

            if gan_mode in ("rp", "ra"):
                with torch.no_grad():
                    x_real_g = sample_100gaussians(
                        batch_size=batch_size,
                        device=device,
                        generator=train_gen,
                    )
                real_logits_g = D(x_real_g)
                loss_gan = gan_loss.g_loss(fake_logits, real_logits_g)
            else:
                loss_gan = gan_loss.g_loss(fake_logits)

            with torch.no_grad():
                unique_idx = torch.unique(idx)

            # VICReg-like regularization on the current batch
            ep_z = vic_reg(prior.z[unique_idx])

            loss_g = loss_gan + lambda_ep * ep_z

            opt_G.zero_grad()
            opt_prior.zero_grad()
            loss_g.backward()

            opt_G.step()
            opt_prior.step()

            # EMA update
            with torch.no_grad():
                for pe, p in zip(ema_G.parameters(), G.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                for pe, p in zip(ema_prior.parameters(), prior.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            # -------------------------
            # Logging / snapshots
            # -------------------------
            if global_step % log_interval == 0:
                modes, hq_frac = mode_coverage(ema_G, ema_prior, device)
                print(
                    f"[epoch {epoch:04d} step {global_step:06d}] "
                    f"D: {loss_d.item():.4f} "
                    f"G_gan: {loss_gan.item():.4f} "
                    f"EP(z): {ep_z.item():.4f} "
                    f"modes: {modes}/100 "
                    f"hq: {hq_frac:.3f}"
                )

            if global_step % snapshot_interval == 0 and global_step > 0:
                save_fake_scatter(
                    ema_G,
                    ema_prior,
                    device,
                    str(out_path / f"samples_step_{global_step:06d}.png"),
                    real_samples=real_viz,
                )

            global_step += 1

        # End-of-epoch snapshot + checkpoint
        save_fake_scatter(
            ema_G,
            ema_prior,
            device,
            str(out_path / f"samples_epoch_{epoch:04d}.png"),
            real_samples=real_viz,
        )

    return prior, G, D


def main() -> None:
    parser = argparse.ArgumentParser(
        description="100 Gaussians toy problem with ParticlePrior + EP regularizer.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--num_particles", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_lr_mult", type=float, default=1.5)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--lambda_ep", type=float, default=1.0)
    parser.add_argument("--r1_gamma", type=float, default=0.02)
    parser.add_argument("--fourier", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument(
        "--lr_floor",
        type=float,
        default=0.05,
        help="Cosine LR anneal floor as a fraction of the base LRs.",
    )
    parser.add_argument(
        "--lr_anneal_start",
        type=float,
        default=0.6,
        help="Fraction of the run at full LR before the cosine anneal begins.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="logistic",
        choices=["hinge", "wasserstein", "logistic", "lsgan"],
    )
    parser.add_argument(
        "--gan_mode",
        type=str,
        default="rp",
        choices=["vanilla", "rp", "ra"],
    )
    parser.add_argument("--out_dir", type=str, default="100gaussians_samples")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device string, e.g. 'cpu' or 'cuda:0'. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        num_particles=args.num_particles,
        lr=args.lr,
        d_lr_mult=args.d_lr_mult,
        beta1=args.beta1,
        lambda_ep=args.lambda_ep,
        r1_gamma=args.r1_gamma,
        fourier=args.fourier,
        ema_decay=args.ema_decay,
        lr_floor=args.lr_floor,
        lr_anneal_start=args.lr_anneal_start,
        loss_type=args.loss_type,
        gan_mode=args.gan_mode,
        out_dir=args.out_dir,
        log_interval=args.log_interval,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
