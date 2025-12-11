#!/usr/bin/env python
"""
33.100gaussians.py

100 Gaussians 2D toy problem with a ParticlePrior + Epps–Pulley regularizer.

This is deliberately nastier than the 25-Gaussian grid:
  - Data: 100-Gaussian mixture on a 10x10 grid in R^2 with small variance.
  - Prior: lib.particle_prior.ParticlePrior (learnable particles in latent space).
  - G: simple MLP mapping z -> x in R^2.
  - D: simple MLP mapping x -> scalar score.
  - Loss: lib.gan_loss.GANLoss (default: hinge).
  - Regularizer: lib.lerae.EppsPulley on the active particles.

Visualization:
  - At fixed intervals, we sample the SAME latent particles (fixed_first_n=True)
    and render a scatter plot of:
        * real samples from the 100-Gaussian mixture (fixed across training),
        * fake samples from the current generator.
  - This makes it easy to turn the sequence of PNGs into a video.
"""

import argparse
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
        z_dim: int = 2,
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
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 3,
    ) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return shape (B,) for convenience.
        return self.net(x).squeeze(-1)


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
    epochs: int = 300,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    z_dim: int = 2,
    num_particles: int = 20_000,
    lr: float = 1e-3,
    beta1: float = 0.5,
    lambda_ep: float = 1.0,
    loss_type: str = "hinge",
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
    D = SimpleMLPDiscriminator(in_dim=2).to(device)

    for m in list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    vic_reg = VICRegLikeLoss()
    gan_loss = GANLoss(loss_type=loss_type)

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
        lr=lr,
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
        G,
        prior,
        device,
        str(out_path / f"samples_step_{0:06d}.png"),
        real_samples=real_viz,
    )

    warmup_steps = 5000
    global_step = 0
    step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            step += 1
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
                z_fake = z_fake.to(device)
                x_fake = G(z_fake)

            real_logits = D(x_real)
            fake_logits = D(x_fake)

            loss_d = gan_loss.d_loss(real_logits, fake_logits)

            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # -------------------------
            # 2) Generator + prior step
            # -------------------------
            D.eval()
            G.train()

            z_fake, idx = prior.sample(batch_size)
            z_fake = z_fake.to(device)
            x_fake = G(z_fake)
            fake_logits = D(x_fake)

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

            # -------------------------
            # Logging / snapshots
            # -------------------------
            if global_step % log_interval == 0:
                print(
                    f"[epoch {epoch:04d} step {global_step:06d}] "
                    f"D: {loss_d.item():.4f} "
                    f"G_gan: {loss_gan.item():.4f} "
                    f"EP(z): {ep_z.item():.4f} "
                    f"G_total: {loss_g.item():.4f}"
                )

            if global_step % snapshot_interval == 0 and global_step > 0:
                save_fake_scatter(
                    G,
                    prior,
                    device,
                    str(out_path / f"samples_step_{global_step:06d}.png"),
                    real_samples=real_viz,
                )

            global_step += 1

        # End-of-epoch snapshot + checkpoint
        save_fake_scatter(
            G,
            prior,
            device,
            str(out_path / f"samples_epoch_{epoch:04d}.png"),
            real_samples=real_viz,
        )

    return prior, G, D


def main() -> None:
    parser = argparse.ArgumentParser(
        description="100 Gaussians toy problem with ParticlePrior + EP regularizer.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--num_particles", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda_ep", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="hinge",
        choices=["hinge", "wasserstein", "logistic"],
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
        beta1=args.beta1,
        lambda_ep=args.lambda_ep,
        loss_type=args.loss_type,
        out_dir=args.out_dir,
        log_interval=args.log_interval,
        snapshot_interval=args.snapshot_interval,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
