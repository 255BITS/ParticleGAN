#!/usr/bin/env python
"""
train_sparse.py

Config-driven single run of the sparse conditional mixed-output toy
(reports/sparse-conditional-with-categorical-and-real-output.md) on the
100-Gaussians champion recipe, with a UCD-style (arXiv:2510.00624)
unconditional discriminator as the default conditioning mechanism.

Held fixed unless a config says otherwise: RpGAN logistic, one-sided cap
penalty (b_cap, coeff 1.0) on the joint critic input [x | y], Fourier-2 D,
Adam beta1=0, base LR 6e-4 (prior x10, D x1.5), delayed cosine anneal to a 5%
floor, EMA(0.995) read-out on G and the prior, VICReg on the particle table.

Every key in DEFAULTS is a config key; unknown keys are an error, so configs
stay honest as knobs are added. Every default reproduces the trajectory a
config written before the knob existed trained.

Class pairing: every batch of reals (x, c, s) is paired with fakes generated
*for the same class vector c*, so the relativistic difference D(real) - D(fake)
and the UCD component d(.)[c] are always compared within a class.

Outputs in out_dir: metrics.jsonl (one row per eval), summary.json, log-style
stdout, final_samples.npz, ckpt.pt, and three PNGs: heatmap (mean |x| per
class, real vs fake), confusion (requested class x emitted symbol), pca
(per-class scatter on the top-2 PCs of the real data).

Usage:
    python experiments/train_sparse.py --config configs/sparse/baseline/ucd_l0p1_s1.yaml
    python experiments/train_sparse.py --total_steps 300 --out_dir /tmp/smoke
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
import torch.nn.functional as F
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.particle_prior import ParticlePrior  # noqa: E402
from lib.gan_loss import GANLoss  # noqa: E402
from lib.vicreg_loss import VICRegLikeLoss  # noqa: E402
from lib.grad_regularizers import GradRegularizer  # noqa: E402
from lib.sparse_toy import SparseMixedToy  # noqa: E402
from lib.sparse_models import (  # noqa: E402
    SparseCondGenerator,
    SparseJointDiscriminator,
    JointCritic,
)
from lib.sparse_metrics import evaluate_samples, convergence_bar, particle_class_purity  # noqa: E402


DEFAULTS: Dict = {
    # run
    "seed": 1,
    "total_steps": 5000,
    "eval_interval": 100,
    "out_dir": "results/sparse/runs/adhoc",
    "batch_size": 256,
    "eval_n": 8192,
    "final_n": 20000,
    # data (report §1)
    "d": 24,
    "k": 3,
    "n_modes": 64,
    "n_classes": 8,
    "n_symbols": 8,
    "sigma": 0.05,
    "symbol_map": "identity",     # 'identity' (K=C) or 'split' (K=2C, 50/50 per class)
    "data_seed": 0,               # the problem instance; independent of the training seed
    "n_train": 0,                 # 0 = infinite stream; else a fixed pool of this many reals
    # generator
    "z_dim": 8,
    "num_particles": 2048,
    "prior": "particles",         # 'particles' (learnable) or 'gaussian' (frozen N(0,I) table)
    "prior_partition": "none",    # 'none' (shared table) or 'class' (each class owns a block)
    "prior_lr_mult": 10.0,
    "emb_dim": 8,
    "real_head": "linear",        # linear / gated / topk
    "cat_mode": "gumbel_st",      # gumbel_st / gumbel_soft / st_argmax / soft
    "tau_start": 1.0,
    "tau_end": 1.0,               # linear anneal over the run; equal = constant
    "lambda_sym": 0.0,            # supervised CE anchor on G's symbol logits (0 = pure adversarial)
    "lambda_sp": 0.0,             # gate sparsity penalty mean(sigmoid(g)) (gated/topk heads)
    "gate_start_frac": 0.0,       # gated/topk heads emit ungated values before this fraction of the run
    "lambda_ep": 1.0,             # VICReg on the particle table
    # discriminator
    "d_mode": "ucd",              # scalar / concat / proj / ucd
    "fourier": 2,
    "fourier_ramp_start": 0.0,    # coarse-to-fine: sin/cos features scaled 0 -> 1 between these
    "fourier_ramp_end": 0.0,      # fractions of the run (end <= start = no ramp, full features)
    "ucd_lambda": 0.1,            # lambda_1: CE(d(x), c) on reals + fakes (ucd only)
    "gp_on_y": True,              # penalize grad w.r.t. [x | y] (True) or x only (False)
    # recipe
    "arm": "b_cap",
    "coeff": 1.0,
    "kappa": 1.0,
    "norm": "l2",
    "loss_type": "logistic",
    "lr": 6e-4,
    "d_lr_mult": 1.5,
    "beta1": 0.0,
    "ema_decay": 0.995,
    "lr_floor": 0.05,
    "lr_anneal_start": 0.6,
    "hidden": 128,
    "n_hidden": 3,
}


def load_config(path: Optional[str]) -> Dict:
    cfg = dict(DEFAULTS)
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(user)
    return cfg


# =========================
#  Plots
# =========================

def save_plots(toy: SparseMixedToy, x_f: torch.Tensor, s_f: torch.Tensor, c_f: torch.Tensor,
               x_r: torch.Tensor, c_r: torch.Tensor, out_dir: Path, title: str) -> None:
    C, K, d = toy.n_classes, toy.n_symbols, toy.d
    xf, xr = x_f.cpu().numpy(), x_r.cpu().numpy()
    cf, cr, sf = c_f.cpu().numpy(), c_r.cpu().numpy(), s_f.cpu().numpy()

    # 1) mean |x| heatmap, d x C, real vs fake
    hm_r = np.stack([np.abs(xr[cr == c]).mean(0) for c in range(C)], 1)
    hm_f = np.stack([np.abs(xf[cf == c]).mean(0) if (cf == c).any() else np.zeros(d) for c in range(C)], 1)
    vmax = max(hm_r.max(), hm_f.max(), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, hm, name in zip(axes[:2], (hm_r, hm_f), ("real", "fake")):
        im = ax.imshow(hm, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"mean |x| ({name})")
        ax.set_xlabel("class c")
        ax.set_ylabel("dim")
    diff = np.log10(hm_f + 1e-4) - np.log10(hm_r + 1e-4)
    im2 = axes[2].imshow(diff, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axes[2].set_title("log10 fake/real (smear = warm off-support)")
    axes[2].set_xlabel("class c")
    fig.colorbar(im, ax=axes[:2], shrink=0.8)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(out_dir / "heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) confusion: requested class (rows) x emitted symbol (cols)
    conf = np.zeros((C, K))
    np.add.at(conf, (cf, sf), 1)
    conf = conf / conf.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(conf, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("emitted symbol")
    ax.set_ylabel("requested class")
    ax.set_title("p(symbol | requested class)")
    fig.colorbar(im, ax=ax)
    fig.savefig(out_dir / "confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3) per-class scatter on the real data's top-2 PCs
    mu = xr.mean(0)
    _, _, vt = np.linalg.svd(xr - mu, full_matrices=False)
    pr, pf = (xr - mu) @ vt[:2].T, (xf - mu) @ vt[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, p, cc, name in zip(axes, (pr, pf), (cr, cf), ("real", "fake")):
        ax.scatter(p[:, 0], p[:, 1], c=cc, cmap="tab10", s=3, alpha=0.5, vmin=0, vmax=9)
        ax.set_title(f"{name} (PCA of real, colored by class)")
        ax.set_aspect("equal", "box")
    fig.savefig(out_dir / "pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =========================
#  Training
# =========================

def train(cfg: Dict, device: torch.device) -> Dict:
    seed = int(cfg["seed"])
    total_steps = int(cfg["total_steps"])
    eval_interval = int(cfg["eval_interval"])
    B = int(cfg["batch_size"])

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    mk = (lambda: torch.Generator(device=device)) if device.type == "cuda" else torch.Generator
    train_gen, eval_gen = mk(), mk()
    train_gen.manual_seed(seed)

    out_path = Path(cfg["out_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    # ---- data ----
    toy = SparseMixedToy(
        d=int(cfg["d"]), k=int(cfg["k"]), n_modes=int(cfg["n_modes"]), n_classes=int(cfg["n_classes"]),
        n_symbols=int(cfg["n_symbols"]), sigma=float(cfg["sigma"]), symbol_map=str(cfg["symbol_map"]),
        seed=int(cfg["data_seed"]), device=device,
    )
    print("[data]", toy.describe(), flush=True)
    C, K, d = toy.n_classes, toy.n_symbols, toy.d

    n_train = int(cfg["n_train"])
    pool: Optional[Tuple[torch.Tensor, ...]] = None
    if n_train > 0:
        pool_gen = mk()
        pool_gen.manual_seed(int(cfg["data_seed"]) + 777)
        pool = toy.sample(n_train, generator=pool_gen)[:3]
        print(f"[data] finite pool: n_train={n_train} ({n_train / toy.n_modes:.1f} per mode)", flush=True)

    def sample_real(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pool is None:
            x, c, s, _ = toy.sample(n, generator=train_gen)
            return x, c, s
        idx = torch.randint(0, n_train, (n,), device=device, generator=train_gen)
        return pool[0][idx], pool[1][idx], pool[2][idx]

    # ---- models ----
    prior_kind = str(cfg["prior"])
    if prior_kind not in ("particles", "gaussian"):
        raise ValueError(f"prior must be 'particles' or 'gaussian', got {prior_kind!r}")
    learnable = prior_kind == "particles"
    P = int(cfg["num_particles"])
    partition = str(cfg["prior_partition"])
    if partition not in ("none", "class"):
        raise ValueError(f"prior_partition must be 'none' or 'class', got {partition!r}")
    if partition == "class" and P % C != 0:
        raise ValueError("num_particles must be a multiple of n_classes for prior_partition='class'")
    block = P // C

    prior = ParticlePrior(num_particles=P, z_dim=int(cfg["z_dim"]), learnable=learnable).to(device)
    G = SparseCondGenerator(
        z_dim=int(cfg["z_dim"]), n_classes=C, d=d, n_symbols=K, k=toy.k, hidden=int(cfg["hidden"]),
        n_hidden=int(cfg["n_hidden"]), emb_dim=int(cfg["emb_dim"]), real_head=str(cfg["real_head"]),
        cat_mode=str(cfg["cat_mode"]),
    ).to(device)
    D = SparseJointDiscriminator(
        d=d, n_symbols=K, n_classes=C, hidden=int(cfg["hidden"]), n_hidden=int(cfg["n_hidden"]),
        fourier=int(cfg["fourier"]), d_mode=str(cfg["d_mode"]),
    ).to(device)
    ema_G, ema_prior = copy.deepcopy(G), copy.deepcopy(prior)
    for p in list(ema_G.parameters()) + list(ema_prior.parameters()):
        p.requires_grad_(False)

    def sample_z(pr: ParticlePrior, c: torch.Tensor, gen: Optional[torch.Generator] = None):
        n = c.shape[0]
        if partition == "class":
            local = torch.randint(0, block, (n,), device=device, generator=gen)
            idx = c * block + local
        else:
            idx = torch.randint(0, pr.num_particles, (n,), device=device, generator=gen)
        return pr.z[idx], idx

    # ---- losses / optimizers ----
    gan_loss = GANLoss(loss_type=str(cfg["loss_type"]), mode="rp")
    regularizer = GradRegularizer(arm=str(cfg["arm"]), coeff=float(cfg["coeff"]), kappa=float(cfg["kappa"]), norm=str(cfg["norm"]))
    vic = VICRegLikeLoss()
    lr = float(cfg["lr"])
    beta1 = float(cfg["beta1"])
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr * float(cfg["d_lr_mult"]), betas=(beta1, 0.999))
    opt_P = torch.optim.Adam(prior.parameters(), lr=lr * float(cfg["prior_lr_mult"]), betas=(beta1, 0.999)) if learnable else None
    opts = [o for o in (opt_G, opt_D, opt_P) if o is not None]
    base_lrs = {id(o): [g["lr"] for g in o.param_groups] for o in opts}

    ucd = D.d_mode == "ucd"
    ucd_lambda = float(cfg["ucd_lambda"])
    lambda_sym, lambda_sp, lambda_ep = float(cfg["lambda_sym"]), float(cfg["lambda_sp"]), float(cfg["lambda_ep"])
    gp_on_y = bool(cfg["gp_on_y"])
    tau0, tau1 = float(cfg["tau_start"]), float(cfg["tau_end"])
    ramp_a, ramp_b = float(cfg["fourier_ramp_start"]), float(cfg["fourier_ramp_end"])

    def fourier_scale(frac: float) -> float:
        if ramp_b <= ramp_a:
            return 1.0
        return float(min(1.0, max(0.0, (frac - ramp_a) / (ramp_b - ramp_a))))
    ema_decay = float(cfg["ema_decay"])

    # ---- eval ----
    eval_gen.manual_seed(seed + 999)
    n_eval = int(cfg["eval_n"])
    c_eval = torch.arange(n_eval, device=device) % C
    x_real_eval, c_real_eval, s_real_eval, _ = toy.sample(n_eval, generator=eval_gen)
    rows = []
    first_cross: Dict[str, Optional[int]] = {}

    @torch.no_grad()
    def draw_fakes(Gm: nn.Module, pr: ParticlePrior, c: torch.Tensor, gen: torch.Generator):
        Gm.eval()
        z, idx = sample_z(pr, c, gen)
        out = Gm(z, c)
        Gm.train()
        return out, idx

    def run_eval(step: int, extra: Dict[str, float]) -> Dict:
        eval_gen.manual_seed(seed + 4242 + step)
        out, _ = draw_fakes(ema_G, ema_prior, c_eval, eval_gen)
        row = {"step": int(step)}
        row.update(evaluate_samples(toy, out["x"], out["y"], out["logits"], c_eval, x_real_eval, c_real_eval, seed=seed, with_shape=False))
        if ucd:
            with torch.no_grad():
                y_r = F.one_hot(s_real_eval[:2048], K).float()
                cl_r = D.class_logits(x_real_eval[:2048], y_r)
                cl_f = D.class_logits(out["x"][:2048], out["y"][:2048])
            row["ucd_acc_real"] = float((cl_r.argmax(1) == c_real_eval[:2048]).float().mean())
            row["ucd_acc_fake"] = float((cl_f.argmax(1) == c_eval[:2048]).float().mean())
        row.update(extra)
        bar = convergence_bar(row, toy.n_modes)
        row.update({k: bool(v) for k, v in bar.items()})
        for k, v in bar.items():
            if v and first_cross.get(k) is None:
                first_cross[k] = int(step)
        rows.append(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def fmt(row: Dict) -> str:
        parts = [
            f"[step {row['step']:05d}]",
            f"D {row.get('d_loss', float('nan')):.3f} G {row.get('g_loss', float('nan')):.3f} pen {row.get('pen', 0.0):.3f}",
            f"| modes {row['modes']}/{toy.n_modes} hq {row['hq']:.3f} w1 {row['sliced_w1']:.3f}",
            f"| cond {row['cond_acc']:.3f} sep {row['cond_sep_ratio']:.2f}",
            f"| sym {row['sym_acc_mode']:.3f} joint {row['joint_acc']:.3f} symKL {row['sym_cond_kl']:.2f}",
            f"| sp@1e-2 {row['sparse_prec@0p01']:.3f} sp@1e-3 {row['sparse_prec@0p001']:.3f} rec {row['sparse_rec']:.3f} smear {row['smear']:.3f}",
        ]
        if ucd:
            parts.append(f"| ucd r/f {row['ucd_acc_real']:.2f}/{row['ucd_acc_fake']:.2f}")
        parts.append("| BAR" if row["bar_all"] else "|")
        return " ".join(parts)

    # ---- loop ----
    t0 = time.time()
    print(fmt(run_eval(0, {})), flush=True)
    last = {"d_loss": float("nan"), "g_loss": float("nan"), "pen": 0.0, "ucd_ce": 0.0}

    for step in range(total_steps):
        frac = step / max(1, total_steps)
        tau = tau0 + (tau1 - tau0) * frac
        D.fourier.scale = fourier_scale(frac)
        G.gate_on = ema_G.gate_on = frac >= float(cfg["gate_start_frac"])
        anneal_from = float(cfg["lr_anneal_start"]) * total_steps
        if step <= anneal_from:
            scale = 1.0
        else:
            f_ = (step - anneal_from) / max(1.0, total_steps - anneal_from)
            scale = float(cfg["lr_floor"]) + (1.0 - float(cfg["lr_floor"])) * 0.5 * (1.0 + math.cos(math.pi * f_))
        for o in opts:
            for g, b in zip(o.param_groups, base_lrs[id(o)]):
                g["lr"] = b * scale

        # ---- D step ----
        # G stays in train mode here: the D step must see the same relaxed
        # symbol (gumbel / soft / ST) that the G step will train through.
        D.train()
        G.train()
        x_r, c_r, s_r = sample_real(B)
        y_r = F.one_hot(s_r, K).float()
        with torch.no_grad():
            z, _ = sample_z(prior, c_r)
            fo = G(z, c_r, tau=tau)
            x_f, y_f = fo["x"], fo["y"]
        dr, df = D(x_r, y_r, c_r), D(x_f, y_f, c_r)
        loss_d_gan = gan_loss.d_loss(dr["adv"], df["adv"])
        loss_d = loss_d_gan
        ucd_ce = torch.zeros((), device=device)
        if ucd and ucd_lambda > 0:
            ucd_ce = F.cross_entropy(dr["class_logits"], c_r) + F.cross_entropy(df["class_logits"], c_r)
            loss_d = loss_d + ucd_lambda * ucd_ce
        # Keep joint interpolation locations identical in both ablations; the
        # critic can exclude the y derivative without changing the sampled y.
        critic = JointCritic(D, c_r, grad_on_y=gp_on_y)
        pen, pst = regularizer.penalty(critic, torch.cat([x_r, y_r], 1), torch.cat([x_f, y_f], 1), step)
        loss_d = loss_d + pen
        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        # ---- G (+ prior) step ----
        D.eval(); G.train()
        x_r2, c_r2, s_r2 = sample_real(B)
        y_r2 = F.one_hot(s_r2, K).float()
        z, idx = sample_z(prior, c_r2)
        go = G(z, c_r2, tau=tau)
        adv_f = D(go["x"], go["y"], c_r2)["adv"]
        adv_r = D(x_r2, y_r2, c_r2)["adv"]
        loss_gan = gan_loss.g_loss(adv_f, adv_r)
        loss_g = loss_gan
        if lambda_sym > 0:
            loss_g = loss_g + lambda_sym * F.cross_entropy(go["logits"], toy.symbol_target(c_r2, generator=train_gen))
        if lambda_sp > 0 and "gate_prob" in go:
            loss_g = loss_g + lambda_sp * go["gate_prob"].mean()
        if learnable and lambda_ep > 0:
            with torch.no_grad():
                uniq = torch.unique(idx)
            loss_g = loss_g + lambda_ep * vic(prior.z[uniq])
        opt_G.zero_grad()
        if opt_P is not None:
            opt_P.zero_grad()
        loss_g.backward()
        opt_G.step()
        if opt_P is not None:
            opt_P.step()

        with torch.no_grad():
            for pe, p in zip(ema_G.parameters(), G.parameters()):
                pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
            for pe, p in zip(ema_prior.parameters(), prior.parameters()):
                pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

        last = {"d_loss": float(loss_d_gan.detach()), "g_loss": float(loss_gan.detach()), "pen": float(pst["pen"]),
                "ucd_ce": float(ucd_ce.detach()), "tau": float(tau), "ff_scale": D.fourier.scale}
        done = step + 1
        if done % eval_interval == 0 or done == total_steps:
            print(fmt(run_eval(done, last)), flush=True)

    wall = time.time() - t0

    # ---- final ----
    eval_gen.manual_seed(seed + 31337)
    n_final = int(cfg["final_n"])
    c_fin = torch.arange(n_final, device=device) % C
    x_real_fin, c_real_fin, _, _ = toy.sample(n_final, generator=eval_gen)
    out, final_idx = draw_fakes(ema_G, ema_prior, c_fin, eval_gen)
    final = evaluate_samples(toy, out["x"], out["y"], out["logits"], c_fin, x_real_fin, c_real_fin, seed=seed, with_shape=True)
    if ucd:
        with torch.no_grad():
            cl_r = D.class_logits(x_real_fin[:4096], F.one_hot(toy.symbol_of_mode[toy.assign(x_real_fin[:4096])[0]], K).float())
            cl_f = D.class_logits(out["x"][:4096], out["y"][:4096])
        final["ucd_acc_real"] = float((cl_r.argmax(1) == c_real_fin[:4096]).float().mean())
        final["ucd_acc_fake"] = float((cl_f.argmax(1) == c_fin[:4096]).float().mean())
    final.update({k: bool(v) for k, v in convergence_bar(final, toy.n_modes).items()})

    # Specialization among correctly conditioned draws, using the particles that
    # actually generated out. A shared particle can correctly serve many classes
    # when G has a class embedding; low purity is then expected, not a failure.
    spec = float("nan")
    if learnable and partition == "none":
        with torch.no_grad():
            m_hat, _ = toy.assign(out["x"])
            spec = particle_class_purity(final_idx, c_fin, toy.class_of_mode[m_hat], P, C)
    final["particle_class_purity"] = spec

    save_plots(toy, out["x"], out["logits"].argmax(1), c_fin, x_real_fin, c_real_fin, out_path, title=Path(cfg["out_dir"]).name)
    np.savez_compressed(out_path / "final_samples.npz", x=out["x"].cpu().numpy(), y=out["y"].cpu().numpy(),
                        logits=out["logits"].cpu().numpy(), c=c_fin.cpu().numpy(), particle_idx=final_idx.cpu().numpy())
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
                "ema_G": ema_G.state_dict(), "ema_prior": ema_prior.state_dict()}, out_path / "ckpt.pt")

    # steps-to-bar and a "held" flag: crossed the bar and still on it at the end
    summary = {
        "config": dict(cfg),
        "implementation_versions": {"x_only_gp": 2, "particle_class_purity": 2},
        "final": final,
        "first_cross": first_cross,
        "bar_step": first_cross.get("bar_all"),
        "bar_held": bool(final["bar_all"]),
        "wall_clock_sec": float(wall),
        "steps_per_sec": total_steps / max(wall, 1e-9),
        "steps": total_steps,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[done] {Path(cfg['out_dir']).name}: bar={'PASS' if final['bar_all'] else 'fail'} "
        f"(first@{first_cross.get('bar_all')}) modes {final['modes']}/{toy.n_modes} hq {final['hq']:.3f} "
        f"cond {final['cond_acc']:.3f} sym {final['sym_acc_mode']:.3f} joint {final['joint_acc']:.3f} "
        f"sp@1e-2 {final['sparse_prec@0p01']:.3f} zero {final['exact_zero_frac']:.2f} core {final['core_ratio']:.2f} "
        f"w1 {final['sliced_w1']:.3f} | {summary['steps_per_sec']:.1f} steps/s",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Sparse conditional mixed-output toy: one run.")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml-parsed values)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            raise ValueError(f"unknown key {k}")
        cfg[k] = yaml.safe_load(v)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg, device)


if __name__ == "__main__":
    main()
