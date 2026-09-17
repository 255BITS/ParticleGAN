#!/usr/bin/env python
"""
100gaussians.py

100 Gaussians 2D toy problem with a ParticlePrior + regularizer.

This is deliberately nastier than the 25-Gaussian grid:
  - Data: 100-Gaussian mixture on a 10x10 grid in R^2 with small variance.
  - Prior: particlegan.ParticlePrior (learnable particles in latent space).
  - G: simple MLP mapping z -> x in R^2.
  - D: simple MLP with Fourier input features, x -> scalar score.
  - Loss: R3GAN-style objective — relativistic pairing (RpGAN) logistic loss
    with a one-sided cap gradient penalty on reals + fakes.

Fast-convergence recipe (found via large-scale sweep, then a 420-run
regularizer study; all 100 modes covered with >=90% of samples within 3 sigma
of a center by ~5.5k steps and hq 0.986 at 7k, vs. never converging with the
old hinge/Adam(0.5) defaults):
  - z_dim 4 (overcomplete latent eases transport; 2 is much worse)
  - Fourier features on D's input so it can resolve the sigma=0.03 modes
    from step 1 (a plain MLP D learns low frequencies first and plateaus)
  - RpGAN + one-sided cap penalty (`b_cap`, relu(||grad_x D|| - 1)^2 on
    reals and fakes, L2 norm, coeff 1.0): like R1/R2 it caps D's steepness
    at the samples — that is what stops the sharp Fourier D from stranding
    modes, and per the study *any* sample-point penalty damps the game
    equally well — but because it is free below the cap it leaves D usable
    slope, and that buys sharper modes: hq 0.986 with an honest core width
    (per-mode core sigma ratio 0.866) vs. a ~0.94 hq ceiling for R1/R2 on
    this benchmark. See FINDINGS.md for the full study, including the
    provenance caveat that the cap's damping is supplied by the training
    trajectory rather than standing curvature at the endpoint (R1/R2, still
    available via `--reg_arm a_r1r2`, is the safer choice in a game that
    keeps injecting rotation)
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
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Allow `python examples/100gaussians.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from particlegan.particle_prior import (  # noqa: E402
    PRIOR_KINDS, canonical_prior_kind, make_prior,
)
from particlegan import (  # noqa: E402
    GradientPenalty, ParticlePrior, get_recipe, learning_rate_scale,
)

from lib.toy_models import (  # noqa: E402
    SimpleMLPGenerator, SimpleMLPDiscriminator, sample_100gaussians, mode_coverage,
    make_100gaussian_weights,
)
from lib.particle_prior import HopfieldRead  # noqa: E402
from lib.hopfield_metrics import (  # noqa: E402
    evaluate_read, measure_sampling_floor, steps_to_tv,
)

_RECIPE = get_recipe("100gaussians")

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
    read=None,
    fixed_queries=None,
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
        if read is not None:
            z_fake, _ = read.retrieve(fixed_queries)
        else:
            # Small tables cannot provide 4096 distinct visualization rows.
            z_fake, _ = prior.sample(min(n_fake, prior.z.shape[0]), fixed_first_n=True)
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
    epochs: int = _RECIPE.total_steps // 1000,
    steps_per_epoch: int = 1000,
    batch_size: int = _RECIPE.batch_size,
    z_dim: int = _RECIPE.z_dim,
    num_particles: int = _RECIPE.num_particles,
    lr: float = _RECIPE.lr,
    d_lr_mult: float = _RECIPE.d_lr_mult,
    beta1: float = _RECIPE.betas[0],
    lambda_ep: float = _RECIPE.prior_reg,
    reg_arm: str = _RECIPE.reg_arm,
    reg_coeff: float = _RECIPE.reg_coeff,
    fourier: int = 2,
    ema_decay: float = _RECIPE.ema_decay,
    lr_floor: float = _RECIPE.lr_floor,
    lr_anneal_start: float = _RECIPE.lr_anneal_start,
    loss_type: str = _RECIPE.loss_type,
    gan_mode: str = _RECIPE.gan_mode,
    out_dir: str = "100gaussians_samples",
    log_interval: int = 100,
    snapshot_interval: int = 500,
    seed: int = 1234,
    device_str: str = None,
    prior_kind: str = "particles",
    reg_method: str = _RECIPE.reg_method,
    reg_every: int = _RECIPE.reg_every,
    reg_fd_eps: float = 0.05,
    reg_sync_stats: bool = True,
    fused_adam: bool = False,
    return_details: bool = False,
    read: str = "uniform",
    dataset: str = "uniform",
    beta: float = 16.0,
    learn_beta: bool = False,
    study_metrics: bool = False,
    n_eval: int = 100000,
    eval_batch_size: int = 1024,
):
    if read not in ("uniform", "hopfield"):
        raise ValueError("read must be uniform or hopfield")
    if dataset not in ("uniform", "imbalanced"):
        raise ValueError("dataset must be uniform or imbalanced")
    if learn_beta and read != "hopfield":
        raise ValueError("learn_beta requires a Hopfield read")
    if min(log_interval, snapshot_interval, n_eval, eval_batch_size) <= 0:
        raise ValueError("Logging/snapshot intervals and evaluation sizes must be positive")
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
    latent_gen = torch.Generator(device=device).manual_seed(seed + 2)
    penalty_gen = torch.Generator(device=device).manual_seed(seed + 3)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 999)

    # Models
    prior_kind = canonical_prior_kind(prior_kind)
    learnable_prior = prior_kind == "particles"
    if read == "hopfield" and not learnable_prior:
        raise ValueError("Hopfield study requires the learnable particle prior")
    recipe = get_recipe(
        z_dim=z_dim, num_particles=num_particles, batch_size=batch_size,
        total_steps=epochs * steps_per_epoch, lr=lr, d_lr_mult=d_lr_mult,
        betas=(beta1, _RECIPE.betas[1]), loss_type=loss_type, gan_mode=gan_mode,
        reg_arm=reg_arm, reg_coeff=reg_coeff, reg_every=reg_every, reg_method=reg_method,
        prior_reg=lambda_ep, ema_decay=ema_decay, lr_anneal_start=lr_anneal_start,
        lr_floor=lr_floor,
    )
    # The fresh-Gaussian research control retains a fixed visualization table
    # and its historical initialization RNG consumption.
    prior = (make_prior(prior_kind, num_particles=num_particles, z_dim=z_dim)
             if prior_kind == "fresh_gaussian"
             else recipe.make_prior(learnable=learnable_prior)).to(device)
    G = SimpleMLPGenerator(z_dim=z_dim).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, fourier=fourier).to(device)
    read_op = HopfieldRead(prior, beta=beta, learn_beta=learn_beta).to(device) if read == "hopfield" else None

    for m in list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # EMA copies of G + prior for snapshots/eval; the live weights orbit the
    # equilibrium, the averaged ones sit on it.
    ema_G = copy.deepcopy(G)
    ema_prior = copy.deepcopy(prior)
    ema_read = HopfieldRead(ema_prior, beta=beta, learn_beta=False).to(device) if read_op is not None else None
    for p in list(ema_G.parameters()) + list(ema_prior.parameters()):
        p.requires_grad_(False)

    # Keep the raw spread value for diagnostics; apply lambda_ep in the loop.
    vic_reg = recipe.make_prior_regularizer(weight=1.0)
    gan_loss = recipe.make_loss()
    regularizer = recipe.make_gradient_penalty(fd_eps=reg_fd_eps)

    opt_G, opt_D = recipe.make_optimizers(G, D, fused=fused_adam)
    # Keep a separate prior optimizer for the existing update/checkpoint layout.
    opt_prior = (
        torch.optim.Adam((read_op if read_op is not None else prior).parameters(), lr=recipe.lr * recipe.prior_lr_mult,
                         betas=recipe.betas, fused=fused_adam)
        if learnable_prior else None
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_weights = make_100gaussian_weights(dataset, device=device)
    sample_weights = target_weights if dataset == "imbalanced" else None
    history = []
    noise_floor = None
    final_samples = None
    if study_metrics or dataset == "imbalanced":
        torch.save(target_weights.cpu(), out_path / "target_weights.pt")
        np.save(out_path / "target_weights.npy", target_weights.cpu().numpy())
        (out_path / "target_weights.json").write_text(json.dumps(target_weights.cpu().tolist(), indent=2) + "\n")
    if study_metrics:
        floor_gen = torch.Generator(device=device).manual_seed(seed + 1000)
        noise_floor = measure_sampling_floor(target_weights, n_eval=n_eval, repeats=20, generator=floor_gen)
        (out_path / "sampling_floor.json").write_text(json.dumps(noise_floor, indent=2) + "\n")
        (out_path / "metrics.jsonl").write_text("")
        print(f"[study] dataset={dataset} read={read} M={num_particles} seed={seed} "
              f"VICReg={'full table (dense Hopfield gradients)' if read_op is not None else 'unique sampled rows'} "
              f"sampling_floor={noise_floor['mean']:.5f} +/- {noise_floor['sd']:.5f} "
              f"({noise_floor['repeats']} draws of {noise_floor['n_eval']})", flush=True)
    fixed_queries = None
    if read_op is not None:
        query_gen = torch.Generator(device=device).manual_seed(seed + 1001)
        fixed_queries = torch.randn(4096, z_dim, device=device, generator=query_gen)

    # Fixed real samples for visualization (same throughout training).
    real_viz = sample_100gaussians(
        batch_size=8192,
        device=device,
        generator=viz_gen,
        weights=sample_weights,
    )

    # Initial snapshot (untrained model).
    save_fake_scatter(
        ema_G,
        ema_prior,
        device,
        str(out_path / f"samples_step_{0:06d}.png"),
        real_samples=real_viz,
        read=ema_read, fixed_queries=fixed_queries,
    )

    total_steps = epochs * steps_per_epoch
    all_opts = tuple(opt for opt in (opt_G, opt_D, opt_prior) if opt is not None)
    base_lrs = {
        id(opt): [g["lr"] for g in opt.param_groups]
        for opt in all_opts
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def record_evaluation(step):
        nonlocal final_samples
        # A dedicated advancing RNG gives fresh queries at every evaluation
        # without changing the training or visualization RNG streams.
        metrics, final_samples = evaluate_read(
            ema_G, ema_prior, target_weights, read=ema_read, n_eval=n_eval,
            batch_size=eval_batch_size, sample_generator=eval_gen,
        )
        metrics["step"] = step
        history.append(metrics)
        with (out_path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(f"[eval step {step:06d}] modes={metrics['modes']}/100 "
              f"hq={metrics['hq']:.4f} sigma_ratio={metrics['sigma_ratio']} "
              f"TV={metrics['tv']:.5f} KL={metrics['kl']:.5f} "
              f"eff_n={metrics.get('eff_n')} interp_hq={metrics.get('interp_hq')}", flush=True)
        return metrics

    synchronize()
    train_seconds = 0.0
    total_start = block_start = time.perf_counter()
    global_step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            # Full LR until lr_anneal_start, then cosine down to lr_floor.
            # Retain the historical minimum one-update decay duration.
            anneal_from = lr_anneal_start * total_steps
            scale = learning_rate_scale(global_step - anneal_from,
                                        max(1.0, total_steps - anneal_from), 0.0, lr_floor)
            for opt in all_opts:
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
                weights=sample_weights,
            )
            with torch.no_grad():
                z_fake, _ = (read_op(batch_size, generator=latent_gen) if read_op is not None
                             else prior.sample(batch_size, generator=latent_gen))
                x_fake = G(z_fake)

            real_logits = D(x_real)
            fake_logits = D(x_fake)

            loss_d = gan_loss.d_loss(real_logits, fake_logits)

            # Gradient penalty at the samples. This is what lets the sharp
            # Fourier D coexist with full mode coverage: it caps D's
            # steepness where the data is. The regularizer recomputes its own
            # graph internally, so neither batch needs requires_grad here.
            pen, _ = regularizer.penalty(
                D, x_real, x_fake, global_step + 1, generator=penalty_gen,
                collect_stats=reg_sync_stats,
            )
            loss_d = loss_d + pen

            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # -------------------------
            # 2) Generator + prior step
            # -------------------------
            D.eval()
            G.train()

            z_fake, idx = (read_op(batch_size, generator=latent_gen) if read_op is not None
                           else prior.sample(batch_size, generator=latent_gen))
            x_fake = G(z_fake)
            fake_logits = D(x_fake)

            if gan_mode in ("rp", "ra"):
                with torch.no_grad():
                    x_real_g = sample_100gaussians(
                        batch_size=batch_size,
                        device=device,
                        generator=train_gen,
                        weights=sample_weights,
                    )
                real_logits_g = D(x_real_g)
                loss_gan = gan_loss.g_loss(fake_logits, real_logits_g)
            else:
                loss_gan = gan_loss.g_loss(fake_logits)

            ep_z = loss_gan.new_zeros(())
            if learnable_prior:
                if read_op is not None:
                    # Every particle receives read gradients in the dense arm.
                    ep_z = vic_reg(prior.z)
                else:
                    unique_idx = torch.unique(idx)
                    ep_z = vic_reg(prior(unique_idx))
            loss_g = loss_gan + lambda_ep * ep_z

            opt_G.zero_grad()
            if opt_prior is not None:
                opt_prior.zero_grad()
            loss_g.backward()

            opt_G.step()
            if opt_prior is not None:
                opt_prior.step()
                if read_op is not None:
                    read_op.clamp_beta_()

            # EMA update
            with torch.no_grad():
                for pe, p in zip(ema_G.parameters(), G.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                for pe, p in zip(ema_prior.parameters(), prior.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                if ema_read is not None:
                    # Its prior is ema_prior, already updated above exactly once.
                    ema_read.log_beta.mul_(ema_decay).add_(read_op.log_beta, alpha=1 - ema_decay)

            # -------------------------
            # Logging / snapshots
            # -------------------------
            report_step = global_step + 1 if study_metrics else global_step
            log_now = report_step % log_interval == 0
            snapshot_now = report_step % snapshot_interval == 0 and report_step > 0
            maintenance = log_now or snapshot_now
            if maintenance:
                synchronize()
                train_seconds += time.perf_counter() - block_start
            if log_now:
                if study_metrics:
                    measured = record_evaluation(report_step)
                    modes, hq_frac = measured["modes"], measured["hq"]
                else:
                    eval_gen.manual_seed(seed + 999)
                    modes, hq_frac = mode_coverage(
                        ema_G, ema_prior, device, sample_generator=eval_gen,
                        **({"read": ema_read, "n_eval": n_eval} if ema_read is not None else {}),
                    )
                print(
                    f"[epoch {epoch:04d} step {report_step:06d}] "
                    f"D: {loss_d.item():.4f} "
                    f"G_gan: {loss_gan.item():.4f} "
                    f"EP(z): {ep_z.item():.4f} "
                    f"modes: {modes}/100 "
                    f"hq: {hq_frac:.3f}", flush=True,
                )

            if snapshot_now:
                save_fake_scatter(
                    ema_G,
                    ema_prior,
                    device,
                    str(out_path / f"samples_step_{report_step:06d}.png"),
                    real_samples=real_viz,
                    read=ema_read, fixed_queries=fixed_queries,
                )

            if maintenance:
                synchronize()
                block_start = time.perf_counter()
            global_step += 1

        synchronize()
        train_seconds += time.perf_counter() - block_start
        # End-of-epoch snapshot
        save_fake_scatter(
            ema_G,
            ema_prior,
            device,
            str(out_path / f"samples_epoch_{epoch:04d}.png"),
            real_samples=real_viz,
            read=ema_read, fixed_queries=fixed_queries,
        )

        synchronize()
        block_start = time.perf_counter()

    if study_metrics:
        if not history or history[-1]["step"] != global_step:
            record_evaluation(global_step)
        history[-1]["steps_to_tv"] = steps_to_tv(history)
        (out_path / "final_metrics.json").write_text(json.dumps(history[-1], indent=2) + "\n")

    if return_details:
        return {"prior": prior, "G": G, "D": D, "ema_prior": ema_prior, "ema_G": ema_G,
                "read": read_op, "ema_read": ema_read, "history": history,
                "final": history[-1] if history else None, "noise_floor": noise_floor,
                "final_samples": final_samples, "target_weights": target_weights,
                "train_seconds": train_seconds, "total_seconds": time.perf_counter() - total_start}
    return prior, G, D


def main(default_prior="particles", default_out_dir="100gaussians_samples") -> None:
    parser = argparse.ArgumentParser(
        description="100 Gaussians: matched learned-table and Gaussian prior controls.",
    )
    parser.add_argument("--prior", choices=PRIOR_KINDS, default=default_prior)
    parser.add_argument("--read", choices=("uniform", "hopfield"), default="uniform")
    parser.add_argument("--dataset", choices=("uniform", "imbalanced"), default="uniform")
    parser.add_argument("--beta", type=float, default=16.0)
    parser.add_argument("--learn_beta", action="store_true")
    parser.add_argument("--study_metrics", action="store_true",
                        help="Measure TV, width, read health, and empirical noise floor; save metrics.jsonl.")
    parser.add_argument("--n_eval", type=int, default=100000)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=_RECIPE.total_steps // 1000)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=_RECIPE.batch_size)
    parser.add_argument("--z_dim", type=int, default=_RECIPE.z_dim)
    parser.add_argument("--num_particles", type=int, default=_RECIPE.num_particles)
    parser.add_argument("--lr", type=float, default=_RECIPE.lr)
    parser.add_argument("--d_lr_mult", type=float, default=_RECIPE.d_lr_mult)
    parser.add_argument("--beta1", type=float, default=_RECIPE.betas[0])
    parser.add_argument("--lambda_ep", type=float, default=_RECIPE.prior_reg)
    parser.add_argument(
        "--reg_arm",
        type=str,
        default=_RECIPE.reg_arm,
        choices=list(GradientPenalty.ARMS),
        help="Discriminator gradient penalty (particlegan.grad_regularizers). Default "
        "'b_cap' is the one-sided cap that won the regularizer study; "
        "'a_r1r2' is the older zero-centered R1+R2 penalty.",
    )
    parser.add_argument(
        "--reg_coeff",
        type=float,
        default=_RECIPE.reg_coeff,
        help="Gradient penalty strength (0.02 was the tuned value for a_r1r2).",
    )
    parser.add_argument(
        "--r1_gamma",
        type=float,
        default=None,
        help="Deprecated alias: sets --reg_arm a_r1r2 --reg_coeff <value>.",
    )
    parser.add_argument("--reg_method", choices=("autograd", "finite_difference"), default=_RECIPE.reg_method)
    parser.add_argument("--reg_every", type=int, default=_RECIPE.reg_every)
    parser.add_argument("--reg_fd_eps", type=float, default=.05)
    parser.add_argument("--reg_sync_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused_adam", action="store_true")
    parser.add_argument("--fourier", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=_RECIPE.ema_decay)
    parser.add_argument(
        "--lr_floor",
        type=float,
        default=_RECIPE.lr_floor,
        help="Cosine LR anneal floor as a fraction of the base LRs.",
    )
    parser.add_argument(
        "--lr_anneal_start",
        type=float,
        default=_RECIPE.lr_anneal_start,
        help="Fraction of the run at full LR before the cosine anneal begins.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=_RECIPE.loss_type,
        choices=["hinge", "wasserstein", "logistic", "lsgan"],
    )
    parser.add_argument(
        "--gan_mode",
        type=str,
        default=_RECIPE.gan_mode,
        choices=["vanilla", "rp", "ra"],
    )
    parser.add_argument("--out_dir", type=str, default=default_out_dir)
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

    reg_arm, reg_coeff = args.reg_arm, args.reg_coeff
    if args.r1_gamma is not None:
        # r1_gamma <= 0 used to mean "no penalty at all"; keep that meaning.
        reg_arm = "a_r1r2" if args.r1_gamma > 0 else "f_none"
        reg_coeff = args.r1_gamma
        print(
            f"[deprecated] --r1_gamma {args.r1_gamma} -> "
            f"--reg_arm {reg_arm} --reg_coeff {reg_coeff}"
        )

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
        reg_arm=reg_arm,
        reg_coeff=reg_coeff,
        reg_method=args.reg_method, reg_every=args.reg_every, reg_fd_eps=args.reg_fd_eps,
        reg_sync_stats=args.reg_sync_stats, fused_adam=args.fused_adam,
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
        prior_kind=args.prior,
        read=args.read, dataset=args.dataset, beta=args.beta, learn_beta=args.learn_beta,
        study_metrics=args.study_metrics, n_eval=args.n_eval, eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
