#!/usr/bin/env python
"""Fixed-sigma particle routing scout; one seed, mechanism ablations only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particlegan import calibrate_mog_sigma, GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer
from lib.mog_metrics import component_metrics, geometry, sample_metrics
from lib.toy_metrics import sliced_w1
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians

ARMS = ("gan", "route_noise", "route_offset", "route_bounded", "route_zero", "route_grad100",
        "route_balanced", "route_local", "route_local_balanced")
LOCAL_ARMS = ("route_local", "route_local_balanced")
BALANCED_ARMS = ("route_balanced", "route_local_balanced")


def routing_probabilities(distances, temperature, local=False):
    if not local:
        return (-distances / temperature).softmax(1)
    # Prespecified eight-neighbor surrogate. Freeze membership and bandwidth
    # in backward; the forward selection remains the global nearest particle.
    values, neighbors = distances.topk(min(8, distances.shape[1]), largest=False)
    gaps = values - values[:, :1]
    bandwidth = gaps[:, -1:].detach().clamp_min(1e-6)
    weights = (-gaps / bandwidth).softmax(1)
    return torch.zeros_like(distances).scatter(1, neighbors, weights)


def usage_balance(ids, soft):
    """Hard batch frequencies forward, soft aggregate surrogate backward."""
    k = soft.shape[1]
    hard = torch.bincount(ids, minlength=k).to(soft.dtype) / ids.numel()
    mean_soft = soft.mean(0)
    frequencies = hard + (mean_soft - mean_soft.detach())
    return k * (frequencies - 1 / k).square().sum()


class RoutingEncoder(nn.Module):
    """Query a particle, with a separate optional predicted offset.

    Forward uses a single particle. Only the query receives the soft routing
    surrogate; particle gradients use the hard selected row (plus the prior's
    existing global standardization). No averaged-center forward path.
    """

    def __init__(self, width=128):
        super().__init__()
        self.net = SimpleMLPGenerator(2, width, n_hidden=2, out_dim=4)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, x, means, sigma, arm, noise, temperature=.25, return_routing=False):
        scaled = x / math.sqrt(8.25 + .03**2)
        h = self.net(scaled)
        query = scaled + h[:, :2]
        distances = (query[:, None] - means.detach()[None]).square().sum(-1)
        ids = distances.argmin(1)
        soft = routing_probabilities(distances, temperature, arm in LOCAL_ARMS)
        proxy = soft @ means.detach()
        center = means[ids] + (proxy - proxy.detach())
        u = noise if arm == "route_noise" else h[:, 2:]
        if arm == "route_zero":
            u = torch.zeros_like(noise)
        if arm in ("route_bounded", "route_balanced", *LOCAL_ARMS):
            u = 3 * torch.tanh(u / 3)
        if arm == "route_grad100":
            # Exact identity forward; amplify only the offset branch backward.
            u = u.detach() + 100 * (u - u.detach())
        result = (center + sigma * u, ids, u)
        return (*result, soft) if return_routing else result


def json_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def draw_rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def evaluate(g, e, prior, arm, cfg, n, make_plot=None):
    device = prior.z.device
    rng = draw_rng(device, cfg.seed + 10000)
    real = sample_100gaussians(n, device, generator=rng)
    z, ids = prior.sample(n, generator=rng)
    fake = torch.cat([g(v) for v in z.split(4096)])
    metrics, nearest, hq = sample_metrics(fake)
    reference, _, _ = sample_metrics(real)
    metrics["width_ratio"] = metrics["width"] / reference["width"] if metrics["width"] is not None else None
    metrics["reference_hq"] = reference["hq"]
    metrics["reference_width"] = reference["width"]
    metrics["sample_sw1"] = sliced_w1(fake[:8192], real[:8192], seed=cfg.seed + 10001)
    counts = torch.bincount(nearest[hq], minlength=100).float()
    metrics["balance_tv"] = float((counts / counts.sum().clamp_min(1) - .01).abs().sum() / 2)
    comp, _ = component_metrics(ids, nearest, hq, prior.num_particles)
    metrics.update(comp)
    metrics.update(geometry(prior))
    metrics.update(recon_mse=None, recon_same_mode=None, used_particles=None,
                   effective_particles=None, offset_rms=None, zero_offset_mse=None,
                   random_offset_mse=None, offset_sw1=None, conditional_offset_mean_rms=None)
    rec = encoded = None
    if arm != "gan":
        x = real[:8192]
        noise = torch.randn(x.shape, device=device, generator=rng)
        encoded, chosen, u = e(x, prior.means(), prior.sigma, arm, noise, cfg.temperature)
        rec = g(encoded)
        metrics["recon_mse"] = float((rec - x).square().mean())
        mode = lambda v: (v + 4.5).round().clamp(0, 9).long()
        metrics["recon_same_mode"] = float((mode(rec) == mode(x)).all(1).float().mean())
        counts = torch.bincount(chosen, minlength=prior.num_particles).float()
        p = counts / counts.sum()
        metrics["used_particles"] = int((counts > 0).sum())
        metrics["effective_particles"] = float((-(p * p.clamp_min(1e-30).log()).sum()).exp())
        metrics["offset_rms"] = float(u.square().mean().sqrt())
        metrics["offset_abs_gt3"] = float((u.abs() > 3).float().mean())
        metrics["offset_sw1"] = sliced_w1(u, noise, seed=cfg.seed + 10002)
        sums = torch.zeros_like(prior.means()).index_add_(0, chosen, u)
        conditional_means = sums / counts.clamp_min(1)[:, None]
        metrics["conditional_offset_mean_rms"] = float((conditional_means.square().mean(1) * p).sum().sqrt())
        centers = prior.means()[chosen]
        metrics["zero_offset_mse"] = float((g(centers) - x).square().mean())
        metrics["random_offset_mse"] = float((g(centers + prior.sigma * noise) - x).square().mean())
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, values, title in [(axes[0], fake[:12000], "Unconditional samples"),
                                  (axes[1], rec if rec is not None else real[:8192], "Reconstructions" if rec is not None else "Real reference")]:
            v = values.cpu().numpy()
            ax.scatter(v[:, 0], v[:, 1], s=1, alpha=.25)
            ax.set(xlim=(-5.2, 5.2), ylim=(-5.2, 5.2), title=title, aspect="equal")
        means = prior.means().cpu().numpy()
        if encoded is not None:
            v = encoded.cpu().numpy()
            axes[2].scatter(v[:, 0], v[:, 1], s=1, alpha=.15, label="Encoded")
        axes[2].scatter(means[:, 0], means[:, 1], s=8, color="red", label="Particles")
        axes[2].set(title="Latent space", aspect="equal")
        axes[2].legend()
        fig.suptitle(arm)
        fig.tight_layout()
        fig.savefig(make_plot, dpi=150)
        plt.close(fig)
    return metrics


def leaderboard(out):
    rows = [json.loads(p.read_text()) for p in out.glob("*/metrics.json")]
    rows.sort(key=lambda m: (-m["modes"], -m["hq"], m["sample_sw1"]))
    lines = ["# Fixed-sigma particle routing scout", "",
             "Rank: modes covered descending, then high-quality fraction descending, then sample SW1 ascending. Final online weights; no best-checkpoint selection.", "",
             "| Rank | Arm | Steps | Modes /100 | HQ % ↑ | Width /real ≈1 | Balance TV ↓ | SW1 ↓ | Recon MSE ↓ | Used /400 | Offset RMS ≈1 | Train sec |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(v):
        return "—" if v is None else f"{v:.4g}"
    for rank, m in enumerate(rows, 1):
        values = [rank, m["arm"], m["step"], m["modes"], fmt(100*m["hq"]), fmt(m["width_ratio"]),
                  fmt(m["balance_tv"]), fmt(m["sample_sw1"]), fmt(m["recon_mse"]),
                  m["used_particles"] if m["used_particles"] is not None else "—", fmt(m["offset_rms"]), fmt(m["train_seconds"])]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "Width is the existing per-mode core-radius metric divided by a fresh real-data reference; near 1 is desirable. Inspect it alongside HQ and balance, not rank alone.",
              "Reconstruction MSE averages both coordinates. HQ means within 0.09 of a true center; coverage requires ≥10 HQ samples per mode. TV measures imbalance among HQ samples.",
              "One shared seed, distinct mechanisms. This is a scout, not a statistical superiority claim. Sigma is fixed; offsets and means may learn. No KL is used. The two balanced arms add an aggregate particle-usage loss; no arm matches offset distributions.", ""]
    (out / "LEADERBOARD.md").write_text("\n".join(lines))
    json_write(out / "leaderboard.json", rows)


def train(arm, cfg):
    out = cfg.out / arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {out}")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    g = SimpleMLPGenerator(2, cfg.width).to(device)
    d = SimpleMLPDiscriminator(hidden_dim=cfg.width).to(device)
    prior = MoGParticlePrior(num_particles=400, z_dim=2, sigma=0,
                             generator=draw_rng(device, cfg.seed + 1), device=device)
    sigma, d0 = calibrate_mog_sigma(prior.means(), .025)
    prior.set_sigma(sigma)
    prior.d0.copy_(d0)
    prior.sigma_rel = .025
    prior = prior.to(device)
    e = RoutingEncoder(cfg.width).to(device)
    initial_sigma = prior.sigma.detach().clone()
    opt_g = torch.optim.Adam([
        {"params": g.parameters(), "lr": .0006},
        {"params": e.parameters(), "lr": .0006},
        {"params": prior.parameters(), "lr": .006, "betas": (.5, .999)},
    ], betas=(0., .999))
    opt_d = torch.optim.Adam(d.parameters(), lr=.0009, betas=(0., .999))
    adversarial, penalty, spread = GANLoss(), GradientPenalty(lazy_k=4), ParticleRegularizer()
    data_rng = draw_rng(device, cfg.seed + 2)
    prior_rng = draw_rng(device, cfg.seed + 3)
    offset_rng = draw_rng(device, cfg.seed + 4)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}
    metadata.update(arm=arm, sigma=float(prior.sigma), num_particles=400, z_dim=2,
                    lr_g=.0006, lr_e=.0006, lr_d=.0009, lr_prior=.006,
                    gradient_penalty_every=4, sigma_rel=.025, torch=torch.__version__,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    initialization_sha256=hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for m in (g,d,prior,e) for t in m.parameters())).hexdigest())
    json_write(out / "config.json", metadata)
    (out / "source.py").write_text(Path(__file__).read_text())
    train_seconds = 0.
    with (out / "log.txt").open("w", buffering=1) as log, (out / "history.jsonl").open("w", buffering=1) as history:
        def emit(message):
            print(message, flush=True)
            log.write(message + "\n")
        emit(f"START {arm} fixed_sigma={float(prior.sigma):.8g} steps={cfg.steps}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for step in range(1, cfg.steps + 1):
            real = sample_100gaussians(cfg.batch_size, device, generator=data_rng)
            d.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                z, _ = prior.sample(cfg.batch_size, generator=prior_rng)
                fake = g(z)
            dl = adversarial.d_loss(d(real), d(fake)) + penalty(d, real, fake, step)
            dl.backward()
            opt_d.step()
            d.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            z, _ = prior.sample(cfg.batch_size, generator=prior_rng)
            fake = g(z)
            with torch.no_grad():
                real_logits = d(real)
            gl = adversarial.g_loss(d(fake), real_logits)
            reconstruction = gl.new_zeros(())
            balance = gl.new_zeros(())
            if arm != "gan":
                noise = torch.randn(real.shape, generator=offset_rng, device=device)
                if arm in BALANCED_ARMS:
                    encoded, chosen, _, soft = e(real, prior.means(), prior.sigma, arm, noise,
                                                 cfg.temperature, return_routing=True)
                    balance = usage_balance(chosen, soft)
                else:
                    encoded, _, _ = e(real, prior.means(), prior.sigma, arm, noise, cfg.temperature)
                reconstruction = (g(encoded) - real).square().mean()
            loss = gl + cfg.recon_weight * reconstruction + spread(prior.z)
            if arm in BALANCED_ARMS:
                loss = loss + .01 * balance
            loss.backward()
            opt_g.step()
            if step % cfg.log_every == 0 or step == 1:
                emit(f"{arm} step={step}/{cfg.steps} d={float(dl.detach()):.4f} g={float(gl.detach()):.4f} recon={float(reconstruction.detach()):.6f} usage_loss={float(balance.detach()):.4f}")
            if step % cfg.eval_every == 0 or step == cfg.steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                train_seconds += time.perf_counter() - started
                assert torch.equal(initial_sigma, prior.sigma), "sigma changed"
                metrics = evaluate(g, e, prior, arm, cfg,
                                   cfg.eval_samples if step == cfg.steps else min(20000, cfg.eval_samples),
                                   out / "samples.png" if step == cfg.steps else None)
                metrics.update(arm=arm, step=step, train_seconds=train_seconds)
                json_write(out / "metrics.json", metrics)
                history.write(json.dumps(metrics, allow_nan=False) + "\n")
                leaderboard(cfg.out)
                emit(f"EVAL {arm} step={step} modes={metrics['modes']} HQ={metrics['hq']:.3%} width_ratio={metrics['width_ratio']} recon={metrics['recon_mse']} seconds={train_seconds:.1f}")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
        torch.save(dict(g=g.state_dict(), d=d.state_dict(), encoder=e.state_dict(), prior=prior.state_dict(),
                        opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(), step=cfg.steps,
                        data_rng=data_rng.get_state(), prior_rng=prior_rng.get_state(), offset_rng=offset_rng.get_state()), out / "checkpoint.pt")
        emit(f"DONE {arm}: {out / 'metrics.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=24002)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--recon-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=.25)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=100000)
    p.add_argument("--out", type=Path, default=ROOT / "runs/mog_autoencoder/scout")
    cfg = p.parse_args()
    if min(cfg.steps, cfg.width, cfg.batch_size, cfg.log_every, cfg.eval_every, cfg.eval_samples) < 1:
        p.error("counts must be positive")
    if not math.isfinite(cfg.temperature) or cfg.temperature <= 0 or not math.isfinite(cfg.recon_weight) or cfg.recon_weight < 0:
        p.error("temperature must be positive and reconstruction weight nonnegative")
    cfg.out = cfg.out.resolve()
    torch.set_num_threads(2)
    for arm in cfg.arms:
        train(arm, cfg)


if __name__ == "__main__":
    main()
