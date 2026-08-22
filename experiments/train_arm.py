#!/usr/bin/env python
"""
train_arm.py

Config-driven single run of the 100-Gaussian benchmark for one gradient-penalty
"arm".

This is `examples/100gaussians.py`'s recipe held fixed — RpGAN (relativistic
pairing, logistic), Fourier-2 discriminator, ParticlePrior + VICReg, Adam with
beta1=0, prior LR x10, D LR x1.5, delayed cosine anneal, EMA(0.995) on G *and*
the prior — with exactly one thing swapped out: the inline R1+R2 block becomes a
`lib.grad_regularizers.GradRegularizer`, selected by config. Everything else is
byte-identical so that any difference between runs is attributable to the
penalty.

On top of the example it adds the instrumentation an arm comparison needs:

  * per-eval distributional distance (sliced W1) and gradient-norm field
    statistics at reals / fakes / interpolates,
  * a collapse-episode counter,
  * end-of-run exact OT distances, mixture NLL and mode-balance divergences,
  * oscillation diagnostics (windowed W1 std, and the dominant non-DC frequency
    of the per-step D loss),
  * the dominant eigenvalue modulus of the game's local update map
    (lib.game_jacobian), which is the theory-side prediction of whether the
    dynamics contract or orbit.

Usage:

    python experiments/train_arm.py --config configs/c_eikonal.yaml
    python experiments/train_arm.py --config cfg.yaml --out_dir runs/foo --seed 7
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python experiments/train_arm.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.particle_prior import ParticlePrior
from lib.gan_loss import GANLoss
from lib.vicreg_loss import VICRegLikeLoss
from lib.grad_regularizers import GradRegularizer, grad_norm_stats
from lib.game_jacobian import estimate_update_spectrum
from lib.toy_models import (
    SimpleMLPGenerator,
    SimpleMLPDiscriminator,
    sample_100gaussians,
    mode_coverage,
)
from lib.toy_metrics import (
    sliced_w1,
    exact_w1_w2,
    mixture_nll,
    mode_recall_and_hists,
)


# =========================
#  Config
# =========================

DEFAULTS: Dict = {
    "arm": "a_r1r2",
    "coeff": 0.02,
    "kappa": 1.0,
    "lazy_k": 1,
    "seed": 1234,
    "total_steps": 7000,
    "eval_interval": 100,
    "lr": 3e-4,
    "d_lr_mult": 1.5,
    "lr_mult": 1.0,
    "spectral": True,
    "out_dir": "runs/arm",
}

# Held fixed across every arm: the winning recipe from docs/convergence-tips.md.
BATCH_SIZE = 256
Z_DIM = 4
NUM_PARTICLES = 20_000
FOURIER = 2
BETA1 = 0.0
LAMBDA_EP = 1.0
EMA_DECAY = 0.995
LR_FLOOR = 0.05
LR_ANNEAL_START = 0.6
LOSS_TYPE = "logistic"
GAN_MODE = "rp"

# Instrumentation sizes.
EVAL_N = 4096
GRAD_STATS_N = 1024
FINAL_N = 100_000
SPECTRAL_N = 1024
FFT_WINDOW = 2048
W1_WINDOW = 10


def load_config(path: Optional[str]) -> Dict:
    """Read a YAML config and fill in any missing key from DEFAULTS."""
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(user)
    return cfg


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
    Save a scatter plot of fixed real samples against samples from a fixed
    subset of particles (`fixed_first_n=True`), matching the example's plot.
    """
    generator.eval()
    prior.eval()

    with torch.no_grad():
        z_fake, _ = prior.sample(n_fake, fixed_first_n=True)
        fake = generator(z_fake.to(device)).cpu()

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
#  Oscillation diagnostics
# =========================

def new_collapse_state() -> Dict:
    """Fresh state for `update_collapse`."""
    return {"running_max": 0, "consec_low": 0, "in_episode": False, "events": 0}


def update_collapse(state: Dict, modes: int) -> int:
    """
    Feed one eval's mode count to the collapse detector; returns the event count.

    A collapse *episode* is a run of more than 3 consecutive evals in which
    coverage sits below half of the best coverage seen so far. Episodes are
    counted once each, however long they last, so a run that never recovers
    scores 1 rather than "however many evals were left".
    """
    rmax = state["running_max"]
    if rmax > 0 and modes < 0.5 * rmax:
        state["consec_low"] += 1
        if state["consec_low"] > 3 and not state["in_episode"]:
            state["events"] += 1
            state["in_episode"] = True
    else:
        state["consec_low"] = 0
        state["in_episode"] = False
    state["running_max"] = max(rmax, modes)
    return state["events"]


def dloss_fft(d_losses, window: int = FFT_WINDOW) -> Tuple[float, float]:
    """
    Dominant oscillation in the per-step discriminator loss.

    Takes the last `window` values, removes the mean (so the DC bin carries no
    weight), and returns the largest rFFT power excluding DC together with its
    frequency in cycles/step. A GAN that orbits its equilibrium shows a sharp
    peak here; a contracting one shows broadband noise.

    Returns:
        (peak_power, peak_freq_cycles_per_step); (nan, nan) if there are fewer
        than 4 steps to look at.
    """
    x = np.asarray(d_losses[-window:], dtype=np.float64)
    if x.size < 4 or not np.all(np.isfinite(x)):
        return float("nan"), float("nan")
    x = x - x.mean()
    spec = np.fft.rfft(x)
    power = np.abs(spec) ** 2
    power[0] = 0.0  # DC
    k = int(np.argmax(power))
    return float(power[k]), float(k / x.size)


# =========================
#  Training
# =========================

def train(cfg: Dict, device: torch.device) -> Dict:
    """Run one arm end to end; writes metrics.jsonl, summary.json, final_scatter.png."""
    seed = int(cfg["seed"])
    total_steps = int(cfg["total_steps"])
    eval_interval = int(cfg["eval_interval"])
    lr_mult = float(cfg["lr_mult"])

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    gen_device = device if device.type == "cuda" else None
    train_gen = torch.Generator(device=device) if gen_device else torch.Generator()
    viz_gen = torch.Generator(device=device) if gen_device else torch.Generator()
    eval_gen = torch.Generator(device=device) if gen_device else torch.Generator()
    train_gen.manual_seed(seed)
    viz_gen.manual_seed(seed + 1)

    out_path = Path(cfg["out_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    # -------------------------
    #  Models / optimizers
    # -------------------------
    prior = ParticlePrior(num_particles=NUM_PARTICLES, z_dim=Z_DIM).to(device)
    G = SimpleMLPGenerator(z_dim=Z_DIM).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, fourier=FOURIER).to(device)

    for m in list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    ema_G = copy.deepcopy(G)
    ema_prior = copy.deepcopy(prior)
    for p in list(ema_G.parameters()) + list(ema_prior.parameters()):
        p.requires_grad_(False)

    vic_reg = VICRegLikeLoss()
    gan_loss = GANLoss(loss_type=LOSS_TYPE, mode=GAN_MODE)
    regularizer = GradRegularizer(
        arm=str(cfg["arm"]),
        coeff=float(cfg["coeff"]),
        kappa=float(cfg["kappa"]),
        lazy_k=int(cfg["lazy_k"]),
    )

    base_lr_g = float(cfg["lr"]) * lr_mult
    base_lr_prior = base_lr_g * 10.0
    base_lr_d = base_lr_g * float(cfg["d_lr_mult"])

    opt_G = torch.optim.Adam(G.parameters(), lr=base_lr_g, betas=(BETA1, 0.999))
    opt_prior = torch.optim.Adam(prior.parameters(), lr=base_lr_prior, betas=(BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=base_lr_d, betas=(BETA1, 0.999))
    base_lrs = {
        id(opt): [g["lr"] for g in opt.param_groups]
        for opt in (opt_G, opt_D, opt_prior)
    }

    real_viz = sample_100gaussians(batch_size=8192, device=device, generator=viz_gen)

    # -------------------------
    #  Evaluation
    # -------------------------
    eval_rows = []
    collapse_state = new_collapse_state()

    def run_eval(step: int, d_val: float, g_val: float, pen_val: float) -> Dict:
        """One eval row: sliced W1 + coverage on the EMA model, grad norms on live D."""
        # Re-seeded every time so the same particles and the same reals are used
        # at every eval: sampling noise would otherwise swamp the oscillation
        # signal we are trying to measure.
        eval_gen.manual_seed(seed + 999)

        ema_G.eval()
        with torch.no_grad():
            idx = torch.randint(
                0, ema_prior.num_particles, (EVAL_N,), device=device, generator=eval_gen
            )
            fake = ema_G(ema_prior.z[idx])
            real = sample_100gaussians(
                batch_size=EVAL_N, device=device, generator=eval_gen
            )
        ema_G.train()

        w1 = sliced_w1(fake, real, n_proj=128, seed=seed + 7)
        modes, hq = mode_coverage(ema_G, ema_prior, device)

        with torch.no_grad():
            real_small = sample_100gaussians(
                batch_size=GRAD_STATS_N, device=device, generator=eval_gen
            )
            z_small, _ = prior.sample(GRAD_STATS_N, generator=eval_gen)
            fake_small = G(z_small)
        gstats = grad_norm_stats(D, real_small, fake_small.detach())

        events = update_collapse(collapse_state, modes)

        row = {
            "step": int(step),
            "sliced_w1": float(w1),
            "modes": int(modes),
            "hq": float(hq),
            "med_nr": gstats["med_nr"],
            "med_nf": gstats["med_nf"],
            "med_ni": gstats["med_ni"],
            "q10_nr": gstats["q10_nr"],
            "q90_nr": gstats["q90_nr"],
            "q10_nf": gstats["q10_nf"],
            "q90_nf": gstats["q90_nf"],
            "q10_ni": gstats["q10_ni"],
            "q90_ni": gstats["q90_ni"],
            "d_loss": float(d_val),
            "g_loss": float(g_val),
            "pen": float(pen_val),
            "collapse_events": int(events),
        }
        eval_rows.append(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    # -------------------------
    #  Loop
    # -------------------------
    d_losses = []
    t0 = time.time()
    run_eval(0, float("nan"), float("nan"), 0.0)

    last_d = last_g = float("nan")
    last_pen = 0.0

    for global_step in range(total_steps):
        # Full LR until LR_ANNEAL_START, then cosine down to LR_FLOOR.
        anneal_from = LR_ANNEAL_START * total_steps
        if global_step <= anneal_from:
            scale = 1.0
        else:
            frac = (global_step - anneal_from) / max(1.0, total_steps - anneal_from)
            scale = LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * frac))
        for opt in (opt_G, opt_D, opt_prior):
            for group, base in zip(opt.param_groups, base_lrs[id(opt)]):
                group["lr"] = base * scale

        # ---- 1) Discriminator step ----
        D.train()
        G.eval()

        x_real = sample_100gaussians(
            batch_size=BATCH_SIZE, device=device, generator=train_gen
        )
        with torch.no_grad():
            z_fake, _ = prior.sample(BATCH_SIZE)
            x_fake = G(z_fake)

        # The regularizer recomputes its own graph internally, so neither batch
        # needs requires_grad here.
        loss_d_gan = gan_loss.d_loss(D(x_real), D(x_fake))
        pen, pstats = regularizer.penalty(D, x_real, x_fake, global_step)
        loss_d = loss_d_gan + pen

        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        last_d = float(loss_d_gan.detach())
        last_pen = float(pstats["pen"])
        d_losses.append(last_d)

        # ---- 2) Generator + prior step ----
        D.eval()
        G.train()

        z_fake, idx = prior.sample(BATCH_SIZE)
        fake_logits = D(G(z_fake))

        with torch.no_grad():
            x_real_g = sample_100gaussians(
                batch_size=BATCH_SIZE, device=device, generator=train_gen
            )
        loss_gan = gan_loss.g_loss(fake_logits, D(x_real_g))

        with torch.no_grad():
            unique_idx = torch.unique(idx)
        ep_z = vic_reg(prior.z[unique_idx])
        loss_g = loss_gan + LAMBDA_EP * ep_z

        opt_G.zero_grad()
        opt_prior.zero_grad()
        loss_g.backward()
        opt_G.step()
        opt_prior.step()

        last_g = float(loss_gan.detach())

        # ---- EMA ----
        with torch.no_grad():
            for pe, p in zip(ema_G.parameters(), G.parameters()):
                pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
            for pe, p in zip(ema_prior.parameters(), prior.parameters()):
                pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)

        step_done = global_step + 1
        if step_done % eval_interval == 0 or step_done == total_steps:
            row = run_eval(step_done, last_d, last_g, last_pen)
            print(
                f"[step {step_done:06d}] D {row['d_loss']:.4f} G {row['g_loss']:.4f} "
                f"pen {row['pen']:.4f} w1 {row['sliced_w1']:.4f} "
                f"modes {row['modes']}/100 hq {row['hq']:.3f} "
                f"|gr| {row['med_nr']:.3f} |gf| {row['med_nf']:.3f}",
                flush=True,
            )

    wall_clock = time.time() - t0

    # -------------------------
    #  Final metrics
    # -------------------------
    eval_gen.manual_seed(seed + 4242)
    ema_G.eval()
    with torch.no_grad():
        idx = torch.randint(
            0, ema_prior.num_particles, (FINAL_N,), device=device, generator=eval_gen
        )
        fake_final = ema_G(ema_prior.z[idx])
        real_final = sample_100gaussians(
            batch_size=FINAL_N, device=device, generator=eval_gen
        )
    ema_G.train()

    w1_sliced = sliced_w1(fake_final, real_final, n_proj=512, seed=seed + 11)
    balance = mode_recall_and_hists(fake_final)
    nll = mixture_nll(fake_final)
    w1_exact, w2_exact = exact_w1_w2(fake_final, real_final, max_points=4096, seed=seed)
    modes_final, hq_final = mode_coverage(ema_G, ema_prior, device)

    w1_hist = [r["sliced_w1"] for r in eval_rows[-W1_WINDOW:]]
    w1_windowed_std = float(np.std(np.asarray(w1_hist, dtype=np.float64))) if w1_hist else float("nan")
    fft_peak, fft_freq = dloss_fft(d_losses)

    # The eval row nearest 25% of the run: the gradient-norm field mid-training,
    # before the anneal starts moving it.
    target = 0.25 * total_steps
    mid_row = min(eval_rows, key=lambda r: abs(r["step"] - target))

    # -------------------------
    #  Spectral analysis
    # -------------------------
    spectral: Dict = {}
    if bool(cfg["spectral"]):
        try:
            spec_gen = torch.Generator(device=device) if gen_device else torch.Generator()
            spec_gen.manual_seed(seed + 31337)
            with torch.no_grad():
                x_real_spec = sample_100gaussians(
                    batch_size=SPECTRAL_N, device=device, generator=spec_gen
                )
                idx_spec = torch.randint(
                    0, prior.num_particles, (SPECTRAL_N,), device=device, generator=spec_gen
                )
            spectral = estimate_update_spectrum(
                D,
                G,
                prior,
                gan_loss,
                vic_reg,
                # Every-step clone: the map must include the penalty at every
                # application, not on a lazy schedule.
                GradRegularizer(
                    arm=str(cfg["arm"]),
                    coeff=float(cfg["coeff"]),
                    kappa=float(cfg["kappa"]),
                    lazy_k=1,
                ),
                x_real_spec,
                idx_spec,
                lrs={"D": base_lr_d, "G": base_lr_g, "prior": base_lr_prior},
                krylov_dim=24,
                lambda_ep=LAMBDA_EP,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not kill a run
            spectral = {"error": f"{type(exc).__name__}: {exc}"}

    # -------------------------
    #  Outputs
    # -------------------------
    save_fake_scatter(
        ema_G, ema_prior, device, str(out_path / "final_scatter.png"),
        real_samples=real_viz,
    )

    summary = {
        "config": dict(cfg),
        "final": {
            "w1_exact": float(w1_exact),
            "w2_exact": float(w2_exact),
            "w1_sliced": float(w1_sliced),
            "modes": int(modes_final),
            "hq": float(hq_final),
            "mode_recall": float(balance["mode_recall"]),
            "hist_kl": float(balance["hist_kl"]),
            "hist_js": float(balance["hist_js"]),
            "nll": float(nll),
        },
        "mid": {
            "step": int(mid_row["step"]),
            "med_nr": float(mid_row["med_nr"]),
            "med_nf": float(mid_row["med_nf"]),
            "med_ni": float(mid_row["med_ni"]),
        },
        "collapse_events": int(collapse_state["events"]),
        "spectral": spectral,
        "oscillation": {
            "w1_windowed_std": w1_windowed_std,
            "dloss_fft_peak": fft_peak,
            "dloss_fft_freq": fft_freq,
        },
        "wall_clock_sec": float(wall_clock),
        "steps": int(total_steps),
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[done] {cfg['arm']} coeff={cfg['coeff']} "
        f"w1_exact {w1_exact:.4f} modes {modes_final}/100 hq {hq_final:.3f} "
        f"collapse {collapse_state['events']} "
        f"{total_steps / max(wall_clock, 1e-9):.2f} steps/s",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one gradient-penalty arm on the 100-Gaussian benchmark.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override config out_dir.")
    parser.add_argument("--device", type=str, default=None, help="e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(cfg, device)


if __name__ == "__main__":
    main()
