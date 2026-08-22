#!/usr/bin/env python
"""
five_modes.py

Five-mode text toy problem: an autoencoder-with-a-critic (BiGAN-style joint D
over (text, latent)) that has to keep five words -- apple / grape / lemon /
melon / berry -- as five separate modes in a 2D latent space, one learnable
particle per word.

The training recipe here is the same one that `examples/100gaussians.py` ships
as its default (the 0.1.2 "study champion"), so the two examples differ only in
the problem they are pointed at:

  - RpGAN objective: relativistic pairing + logistic kernel (lib.gan_loss).
    The pairing term is what the two hand-written BCE terms used to be: G/prior
    push the fake pair up, E pushes the real pair down, now in one paired loss.
  - One-sided cap gradient penalty on D (`b_cap`, relu(||grad D|| - 1)^2 on the
    real and fake pairs, coeff 1.0, lib.grad_regularizers). Replaces the inline
    R1 penalty. Because D here is a *joint* critic D(x, z), the penalty is taken
    on the gradient w.r.t. the whole joint input, which is the BiGAN analogue of
    the 100gaussians recipe's penalty on grad_x D(x).
  - Adam beta1=0 (the particle table is an embedding-like parameter: momentum
    drifts rows that were not sampled), base LR 6e-4, D at 1.5x, particles at
    10x.
  - EMA (0.995) copies of E, G and the prior, used for every dashboard frame:
    the live weights orbit the equilibrium, the averaged copy sits on it.
  - Delayed cosine LR anneal: full LR for the first 60% of the run, then cosine
    down to a 5% floor (annealing to exactly 0 destroys the run).
  - VICReg-like variance/covariance regularization on the particle cloud at
    weight 1.0, which is what keeps the five "stars" from piling up.

What stays specific to this problem: 5 particles (one per word, to encourage a
1-to-1 mapping), z_dim=2 so the latent cloud can be plotted directly, the
character-level one-hot encoding, and the live dashboard / `results/frame_*.png`
frames that the README gif is built from.

See FINDINGS.md and docs/convergence-tips.md for where the recipe comes from.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from collections import deque
from pathlib import Path

# Allow `python examples/five_modes.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.particle_prior import ParticlePrior  # noqa: E402 - after the sys.path shim
from lib.gan_loss import GANLoss  # noqa: E402
from lib.grad_regularizers import GradRegularizer  # noqa: E402
from lib.vicreg_loss import VICRegLikeLoss  # noqa: E402

# ==========================================
# 1. Setup & Data
# ==========================================
WORDS = ['apple', 'grape', 'lemon', 'melon', 'berry'] 
# We set num_particles = len(WORDS) to encourage 1-to-1 mapping for this toy problem
NUM_PARTICLES = len(WORDS) 

CHARS = "abcdefghijklmnopqrstuvwxyz_ "
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_CHAR = {i: c for i, c in enumerate(CHARS)}
SEQ_LEN = 6
X_DIM = len(CHARS) * SEQ_LEN

# Latent stays 2D here: the dashboard plots the particle cloud directly, and
# five modes do not need the overcomplete latent that the 100-mode grid does.
Z_DIM = 2

WORD_COLORS = ['#FF5555', '#50FA7B', '#F1FA8C', '#BD93F9', '#8BE9FD']

def str_to_tensor(text_list):
    batch_indices = []
    for text in text_list:
        text = text.ljust(SEQ_LEN, '_')[:SEQ_LEN]
        indices = [CHAR_IDX.get(c, 26) for c in text]
        batch_indices.append(indices)
    tensor = torch.tensor(batch_indices, dtype=torch.long)
    return F.one_hot(tensor, num_classes=len(CHARS)).permute(0, 2, 1).float()

def tensor_to_str(tensor_logits):
    indices = torch.argmax(tensor_logits, dim=1).cpu().numpy()
    results = []
    for row in indices:
        s = "".join([IDX_CHAR.get(i, '?') for i in row]).replace('_', '').strip()
        results.append(s)
    return results

# ==========================================
# 2. Components
# ==========================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(X_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, Z_DIM) 
        )
    def forward(self, x): 
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, X_DIM)
        )
    def forward(self, z):
        flat = self.net(z)
        return flat.view(-1, len(CHARS), SEQ_LEN)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X_DIM + Z_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) 
        )
    def forward(self, x, z):
        x_flat = x.flatten(1)
        joint = torch.cat([x_flat, z], dim=1)
        return self.net(joint)


class JointInputD(nn.Module):
    """
    Presents the joint critic D(x, z) as a single-input module D([x_flat, z]).

    lib.grad_regularizers penalizes grad_x D(x) for a one-argument D. Feeding it
    the concatenated (text, latent) vector makes it penalize the gradient w.r.t.
    the full joint input, which is the right analogue for a BiGAN-style critic:
    both halves of the pair get their steepness capped.
    """

    def __init__(self, D: nn.Module) -> None:
        super().__init__()
        self.D = D

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        x_flat, z = joint[:, :X_DIM], joint[:, X_DIM:]
        return self.D(x_flat.view(-1, len(CHARS), SEQ_LEN), z)


def join_pair(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack a (text, latent) pair into the flat vector `JointInputD` expects."""
    return torch.cat([x.flatten(1), z], dim=1)

# ==========================================
# 3. Training & Viz
# ==========================================

def train(
    total_steps: int = 20_000,
    batch_size: int = 256,
    lr: float = 6e-4,
    d_lr_mult: float = 1.5,
    beta1: float = 0.0,
    lambda_ep: float = 1.0,
    reg_arm: str = "b_cap",
    reg_coeff: float = 1.0,
    ema_decay: float = 0.995,
    lr_floor: float = 0.05,
    lr_anneal_start: float = 0.6,
    loss_type: str = "logistic",
    gan_mode: str = "rp",
    viz_interval: int = 50,
    frame_interval: int = 200,
    log_interval: int = 500,
    out_dir: str = "results",
    seed: int = 1234,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Models
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)
    D_joint = JointInputD(D)

    for m in list(E.modules()) + list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # NEW: The Particle Prior
    prior = ParticlePrior(num_particles=NUM_PARTICLES, z_dim=Z_DIM).to(device)

    # EMA copies of E / G / prior. Every dashboard frame is read off these:
    # the live weights orbit the equilibrium, the averaged ones sit on it.
    ema_E = copy.deepcopy(E)
    ema_G = copy.deepcopy(G)
    ema_prior = copy.deepcopy(prior)
    for p in list(ema_E.parameters()) + list(ema_G.parameters()) + list(ema_prior.parameters()):
        p.requires_grad_(False)

    vic_loss_fn = VICRegLikeLoss()
    gan_loss = GANLoss(loss_type=loss_type, mode=gan_mode)
    regularizer = GradRegularizer(arm=reg_arm, coeff=reg_coeff)

    # Optimizers
    # The particles get their own optimizer at 10x LR: they are an
    # embedding-like table and want far more mobility than the dense nets.
    opt_GE = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                              lr=lr, betas=(beta1, 0.999))
    opt_prior = torch.optim.Adam(prior.parameters(),
                                 lr=lr * 10.0, betas=(beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(),
                             lr=lr * d_lr_mult, betas=(beta1, 0.999))

    base_lrs = {
        id(opt): [g["lr"] for g in opt.param_groups]
        for opt in (opt_GE, opt_prior, opt_D)
    }

    loss_D_hist = deque(maxlen=200)
    loss_GE_hist = deque(maxlen=200)

    # --- SETUP DASHBOARD ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_latent = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :]) 
    ax_table.axis('off')

    plt.ion()

    for step in range(total_steps + 1):
        # Full LR until lr_anneal_start, then cosine down to lr_floor.
        anneal_from = lr_anneal_start * total_steps
        if step <= anneal_from:
            scale = 1.0
        else:
            frac = (step - anneal_from) / max(1.0, total_steps - anneal_from)
            scale = lr_floor + (1.0 - lr_floor) * 0.5 * (
                1.0 + math.cos(math.pi * frac)
            )
        for opt in (opt_GE, opt_prior, opt_D):
            for group, base in zip(opt.param_groups, base_lrs[id(opt)]):
                group["lr"] = base * scale

        # --- TRAIN D ---
        opt_D.zero_grad()
        real_words = [random.choice(WORDS) for _ in range(batch_size)]
        x_real = str_to_tensor(real_words).to(device)

        z_enc = E(x_real).detach()
        pred_real = D(x_real, z_enc)

        # NEW: Sample from Particle Prior instead of randn
        with torch.no_grad():
            z_prior, _ = prior.sample(batch_size)
            x_gen_soft = F.softmax(G(z_prior), dim=1)
        pred_fake = D(x_gen_soft, z_prior)  # We pass the Particle, not random noise

        loss_d = gan_loss.d_loss(pred_real, pred_fake)

        # Gradient penalty on the joint (text, latent) pairs. Caps how steep D
        # gets where the data is, without forcing it flat the way R1 did.
        pen, _ = regularizer.penalty(
            D_joint,
            join_pair(x_real, z_enc),
            join_pair(x_gen_soft, z_prior),
            step,
        )
        loss_d = loss_d + pen
        loss_d.backward()
        opt_D.step()

        # --- TRAIN GE (and Prior) ---
        opt_GE.zero_grad()
        opt_prior.zero_grad()

        # Encoder side of the pair: E wants the real pair to score *low*.
        z_enc = E(x_real)
        pred_enc = D(x_real, z_enc)

        # Generator side of the pair (sample prior again).
        z_prior, _ = prior.sample(batch_size)
        x_gen_soft = F.softmax(G(z_prior), dim=1)
        pred_gen = D(x_gen_soft, z_prior)

        # RpGAN pairs the two: one relativistic term replaces the two BCEs.
        loss_ge_gan = gan_loss.g_loss(pred_gen, pred_enc)

        # NEW: Regularize the Prior
        # This forces the "stars" to stay apart
        loss_vic = vic_loss_fn(prior.z)

        loss_ge = loss_ge_gan + lambda_ep * loss_vic
        loss_ge.backward()
        opt_GE.step()
        opt_prior.step()

        # EMA update
        with torch.no_grad():
            for ema_m, m in ((ema_E, E), (ema_G, G), (ema_prior, prior)):
                for pe, p in zip(ema_m.parameters(), m.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

        loss_D_hist.append(loss_d.item())
        loss_GE_hist.append(loss_ge.item())

        # --- VISUALIZATION (Every viz_interval steps, off the EMA copies) ---
        if step % viz_interval == 0:
            ax_loss.clear()
            ax_loss.plot(loss_D_hist, label="D", color='#FF5555', alpha=0.8)
            ax_loss.plot(loss_GE_hist, label="G/E/Prior", color='#8BE9FD', alpha=0.8)
            ax_loss.legend(loc='upper right')
            ax_loss.set_title("Losses")

            # 2. Latent Space Map
            ax_latent.clear()
            with torch.no_grad():
                vocab_x = str_to_tensor(WORDS).to(device)
                vocab_z = ema_E(vocab_x).cpu().numpy()
                typo_z = ema_E(str_to_tensor(["aple"]).to(device)).cpu().numpy()
                # Get Prior Positions
                prior_z = ema_prior.z.detach().cpu().numpy()

            # Plot Prior Particles (The "Stars")
            ax_latent.scatter(prior_z[:,0], prior_z[:,1], color='white', marker='*', s=300, label="Prior Particles", edgecolors='black', zorder=1)

            # Plot Encoded Words
            for i, txt in enumerate(WORDS):
                ax_latent.scatter(vocab_z[i,0], vocab_z[i,1], color=WORD_COLORS[i], s=120, edgecolors='white', alpha=0.9, zorder=2)
                ax_latent.text(vocab_z[i,0]+0.05, vocab_z[i,1]+0.05, txt, color=WORD_COLORS[i], fontsize=10, fontweight='bold')
            
            # Plot Typo
            ax_latent.scatter(typo_z[0,0], typo_z[0,1], color='cyan', marker='X', s=150, label="Typo", zorder=3)
            
            ax_latent.set_title(f"Particle Space (Stars=Prior)")
            ax_latent.grid(True, alpha=0.2)

            # 3. Live Reconstruction Table
            ax_table.clear(); ax_table.axis('off')
            with torch.no_grad():
                recon_vocab = tensor_to_str(ema_G(ema_E(vocab_x)))
                recon_typo = tensor_to_str(ema_G(ema_E(str_to_tensor(["aple"]).to(device))))[0]

            table_txt = f"{'ORIGINAL':<12} | {'RECONSTRUCTED':<15} | {'STATUS'}\n"
            table_txt += "-" * 45 + "\n"
            valid_cnt = 0
            for i, word in enumerate(WORDS):
                rec = recon_vocab[i]
                status = "✅" if word == rec else "❌"
                if word == rec: valid_cnt += 1
                table_txt += f"{word:<12} | {rec:<15} | {status}\n"
            
            table_txt += "-" * 45 + "\n"
            status = "✨ MAGICAL ✨" if recon_typo == "apple" else "..."
            table_txt += f"{'aple':<12} | {recon_typo:<15} | {status}\n"

            if step % log_interval == 0:
                print(
                    f"[step {step:06d}] "
                    f"D: {loss_d.item():.4f} "
                    f"G/E: {loss_ge_gan.item():.4f} "
                    f"VIC(z): {loss_vic.item():.4f} "
                    f"acc: {valid_cnt}/{len(WORDS)} "
                    f"typo->{recon_typo!r}",
                    flush=True,
                )

            header = f"Step: {step} | Accuracy: {valid_cnt}/{len(WORDS)}"
            ax_table.text(0.5, 0.9, header, ha='center', fontsize=14, color='white', fontweight='bold')
            ax_table.text(0.5, 0.5, table_txt, ha='center', va='center', fontsize=12, fontfamily='monospace', color='#F8F8F2')

            plt.pause(0.01)
            if step % frame_interval == 0:
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(f"{out_dir}/frame_{step:05d}.png", dpi=100, facecolor='#282a36')

    plt.ioff()
    plt.show()

    return prior, E, G, D

if __name__ == "__main__":
    train()
