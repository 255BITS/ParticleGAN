#!/usr/bin/env python
"""100 Gaussians DDGAN + UCD; run without arguments for the selected defaults.

Use --config to run another one-shot / diffusion GAN configuration. CUDA required.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import read_config, recipe_defaults
from particlegan import DDGAN, GradientPenalty, get_recipe, scale_learning_rates, ucd_loss
from particlegan.diffusion import DrawSource
from lib.denoising_toy import (
    GaussianGrid, ToyGenerator, ToyDiscriminator,
    generate, grid_metrics, conditional_probe,
)


DEFAULTS = {
    **recipe_defaults('denoising'),
    'prior': 'learned',
    'sigma_rel': 0.025,
    'standardize': True,
    'prior_betas': None,
    'noise': 'gaussian',
    'seed': 24002,
    'std': 0.03,
    'noise_particles': 1024,
    'hidden': 128,
    'generator_hidden': None,
    'depth': 3,
    'fourier': 2,
    'noise_lr_mult': 1.0,
    'noise_reg': 0.0,
    'reg_fd_eps': 0.05,
    'reg_sync_stats': True,
    'fused_adam': False,
    'drop_xt': False,
    'eval_interval': 1000,
    'eval_samples': 8192,
    'final_samples': 20000,
    'probe_samples': 256,
    'save_checkpoint': True,
    'out_dir': 'results/denoising/ddgan_ucd',
}

DEFAULT_CONFIG = ROOT / "configs" / "denoising" / "default.toml"


def training_recipe(cfg):
    """Resolve legacy experiment fields into the public, caller-owned recipe."""
    return get_recipe(
        model=cfg["model"], z_dim=cfg["z_dim"], num_particles=cfg["num_particles"],
        num_classes=cfg["classes"],
        prior_kind="mog" if cfg["prior"] == "mog" else "particles",
        sigma_rel=cfg.get("sigma_rel", 1 / 40) if cfg["prior"] == "mog" else 0.0,
        standardize=cfg.get("standardize", True), prior_betas=cfg.get("prior_betas"),
        conditioning="conditional" if cfg["d_mode"] == "concat" else cfg["d_mode"],
        ucd_target=cfg["ucd_target"], ucd_weight=cfg["ucd_lambda"],
        alpha_bar=cfg["alpha_bar"], batch_size=cfg["batch_size"], total_steps=cfg["steps"],
        lr=cfg["lr"], d_lr_mult=cfg["d_lr_mult"], prior_lr_mult=cfg["prior_lr_mult"],
        betas=(cfg["beta1"], cfg.get("beta2", .999)), loss_type=cfg["loss_type"], gan_mode=cfg["gan_mode"],
        reg_arm=cfg["reg_arm"], reg_coeff=cfg["reg_coeff"], reg_kappa=cfg["reg_kappa"],
        reg_every=cfg["reg_every"], reg_method=cfg["reg_method"], prior_reg=cfg["prior_reg"],
        ema_decay=cfg["ema"], lr_anneal_start=cfg["lr_anneal_start"], lr_floor=cfg["lr_floor"],
    )


def make_prior(cfg, device):
    """Build latent sources consistently for training and checkpoint probes."""
    if cfg["prior"] == "mog":
        rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 101)
        return training_recipe(cfg).make_prior(device=device, generator=rng)
    return DrawSource(cfg["prior"], cfg["num_particles"], cfg["z_dim"], cfg["seed"] + 101, device)


def validate(cfg):
    GradientPenalty(cfg["reg_arm"], cfg["reg_coeff"], kappa=cfg["reg_kappa"],
                    lazy_k=cfg.get("reg_every", 1), method=cfg.get("reg_method", "autograd"),
                    fd_eps=cfg.get("reg_fd_eps", .05))
    target = cfg.get("ucd_target", "class")
    if target not in ("class", "time_class") or (target == "time_class" and (cfg["model"] != "ddgan" or cfg["d_mode"] != "ucd")):
        raise ValueError("time_class UCD requires DDGAN with a UCD discriminator")
    for key, choices in {"model": ("gan", "ddgan"), "d_mode": ("concat", "ucd", "scalar"),
                         "prior": ("gaussian", "fixed", "learned", "mog", "zero"),
                         "noise": ("gaussian", "fixed", "learned", "zero")}.items():
        if cfg[key] not in choices:
            raise ValueError(f"{key} must be in {choices}")
    for key in ("steps", "batch_size", "hidden", "depth", "z_dim", "num_particles", "noise_particles", "eval_interval", "probe_samples"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["batch_size"] < 2 or cfg["num_particles"] < 2 or cfg["noise_particles"] < 2:
        raise ValueError("variance regularization requires at least two samples/particles")
    width = cfg.get("generator_hidden")
    if width is not None and (type(width) is not int or width < 1):
        raise ValueError("generator_hidden must be a positive integer or None")
    if cfg["classes"] not in (1, 4):
        raise ValueError("classes must be 1 or 4")
    for key in ("eval_samples", "final_samples"):
        if cfg[key] < cfg["classes"] or cfg[key] % cfg["classes"]:
            raise ValueError(f"{key} must be positive and divisible by classes")
    if cfg["model"] == "gan" and (cfg["noise"] != "gaussian" or cfg["drop_xt"]):
        raise ValueError("reverse noise and drop_xt do not apply to one-shot GAN")
    if not (0 <= cfg["ema"] < 1 and 0 <= cfg["lr_anneal_start"] < 1 and 0 <= cfg["lr_floor"] <= 1):
        raise ValueError("invalid EMA/annealing parameters")
    if cfg["lr"] <= 0 or cfg["std"] <= 0:
        raise ValueError("lr and std must be positive")
    training_recipe(cfg)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def render(out, x, c, toy, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = np.array(["#dc6255", "#347bb0", "#43a07c", "#a17ac0"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].scatter(*x[:8000].T, c=colors[c[:8000]], s=2, alpha=.35, linewidths=0)
    means = toy.means.cpu().numpy()
    axes[0].scatter(*means.T, c="black", marker="+", s=10, linewidths=.5)
    axes[0].set(title="Samples colored by requested class", xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect="equal")
    nearest = np.square(x[:, None] - means[None]).sum(-1).argmin(-1)
    count = np.bincount(nearest, minlength=100) / len(x) * 100
    im = axes[1].imshow(count.reshape(10, 10).T, origin="lower", vmin=0, vmax=2, cmap="coolwarm")
    axes[1].set_title("Mode mass / target mass (ideal 1)")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=130)
    plt.close(fig)
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
        for ax, panel in zip(axes[0], panels):
            ax.scatter(*panel["oracle"].T, s=5, alpha=.4, label="oracle")
            ax.scatter(*panel["model"].T, s=5, alpha=.4, label="model")
            ax.set_title(f"Fixed observation, class 0, t={panel['t']}")
            ax.set_aspect("equal")
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(out / "posteriors.png", dpi=130)
        plt.close(fig)


def train(cfg):
    validate(cfg)
    recipe = training_recipe(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a GPU; CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # The grid runner supplies exact source digests; standalone runs record them too.
    from experiments.run_grid import code_provenance
    provenance = code_provenance(str(Path(__file__)), sys.executable)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    write_json(out / "provenance.json", provenance)
    # Preserve the exact code behind dirty/uncommitted working-tree experiments.
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, digest in provenance["sources"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("source changed while capturing experiment provenance")
            archive.writestr(name, data)
    env = {"gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}
    write_json(out / "environment.json", env)
    rngs = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
            for name, offset in (("data", 11), ("time", 12), ("corruption", 13), ("latent", 14), ("noise", 15), ("penalty", 16))}
    toy = GaussianGrid(device, cfg["std"], cfg["classes"])
    # Training creates bounded integer times; retain the original sync-free hot path.
    schedule = DDGAN(cfg["alpha_bar"], validate_args=False).to(device)
    # Independent initialization preserves identical tables across architecture changes.
    prior = make_prior(cfg, device)
    noise = DrawSource(cfg["noise"], cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    g, d = ToyGenerator(cfg).to(device), ToyDiscriminator(cfg).to(device)
    ema_g, ema_prior, ema_noise = copy.deepcopy(g), copy.deepcopy(prior), copy.deepcopy(noise)
    for model in (ema_g, ema_prior, ema_noise):
        model.requires_grad_(False)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d), fused=cfg["fused_adam"])
    if cfg["noise"] == "learned":
        # A research-only source with its own rate; ordinary optimizers stay extensible.
        opt_g.add_param_group({"params": list(noise.parameters()),
                               "lr": recipe.lr * cfg["noise_lr_mult"]})
    bases = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    gan = recipe.make_loss()
    penalty = recipe.make_critic_penalty(opt_d, generator=rngs["penalty"], fd_eps=cfg["reg_fd_eps"],
                                         collect_stats=cfg.get("reg_sync_stats", True))
    spread = recipe.make_prior_regularizer()

    def batch():
        c = torch.randint(cfg["classes"], (cfg["batch_size"],), device=device, generator=rngs["data"])
        x0 = toy.sample(c, rngs["data"])
        if cfg["model"] == "gan":
            return c, x0, None, None
        t = torch.randint(1, schedule.steps + 1, (len(c),), device=device, generator=rngs["time"])
        real, xt = schedule.forward_pair(x0, t, rngs["corruption"])
        return c, real, xt, t

    def fake(c, xt, t):
        z, ids = prior.sample(len(c), rngs["latent"])
        clean = g(z, c, xt, t)
        if cfg["model"] == "gan":
            return clean, ids
        eta, _ = noise.sample(len(c), rngs["noise"])
        return schedule.reverse(clean, xt, t, eta), ids

    @torch.no_grad()
    def evaluate(n):
        er = [torch.Generator(device=device).manual_seed(99000 + k) for k in range(4)]
        c = torch.arange(n, device=device) % cfg["classes"]
        real = toy.sample(c, er[3])
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = generate(ema_g, ema_prior, ema_noise, schedule, c, *er[:3])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("nonfinite generated samples")
        metrics = grid_metrics(x, c, toy, real)
        metrics["sampling_ms_per_1000"] = seconds * 1e6 / n
        return metrics, x, c

    elapsed_train = 0.0
    total_start = time.perf_counter()
    metrics_path = out / "metrics.jsonl"
    with metrics_path.open("w") as log:
        for step in range(1, cfg["steps"] + 1):
            if step == 1 or (step - 1) % cfg["eval_interval"] == 0:
                torch.cuda.synchronize()
                block_start = time.perf_counter()
            scale_learning_rates(step - 1, recipe, (opt_g, opt_d), bases, prior)
            d.requires_grad_(True)
            c, real, xt, t = batch()
            with torch.no_grad():
                xf, _ = fake(c, xt, t)
            dr, cr = d(real, c, xt, t)
            df, cf = d(xf, c, xt, t)
            loss_d = gan.d_loss(dr, df)
            if cfg["d_mode"] == "ucd" and cfg["ucd_lambda"]:
                targets = d.ucd_labels(c, t)
                loss_d = loss_d + ucd_loss(cr, cf, targets, weight=cfg["ucd_lambda"])
            loss_d = loss_d + penalty(d, real, xf, c, xt=xt, t=t)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            d.requires_grad_(False)
            c, real, xt, t = batch()
            xf, ids = fake(c, xt, t)
            df = d(xf, c, xt, t)[0]
            with torch.no_grad():
                dr = d(real, c, xt, t)[0]
            loss_g = gan.g_loss(df, dr)
            if cfg["prior"] in ("learned", "mog") and cfg["prior_reg"]:
                selected = prior.z[ids.unique()]
                if len(selected) > 1:
                    loss_g = loss_g + spread(selected)
            if cfg["noise"] == "learned" and cfg["noise_reg"]:
                # Explicit moment penalty, off in the unconstrained learned-noise arm.
                nt = noise.table - noise.table.mean(0)
                cov = nt.T @ nt / (len(nt) - 1)
                loss_g = loss_g + cfg["noise_reg"] * (noise.table.mean(0).square().mean() +
                                                         (cov - torch.eye(2, device=device)).square().mean())
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            with torch.no_grad():
                for target, source in ((ema_g, g), (ema_prior, prior), (ema_noise, noise)):
                    for pe, p in zip(target.parameters(), source.parameters()):
                        pe.lerp_(p, 1 - cfg["ema"])
            if step % cfg["eval_interval"] == 0 or step == cfg["steps"]:
                torch.cuda.synchronize()
                elapsed_train += time.perf_counter() - block_start
                metrics, _, _ = evaluate(cfg["eval_samples"])
                metrics.update(step=step, train_seconds=elapsed_train, wall_seconds=time.perf_counter() - total_start,
                               d_loss=float(loss_d.detach()), g_loss=float(loss_g.detach()))
                log.write(json.dumps(json_safe(metrics), allow_nan=False) + "\n")
                log.flush()
                print(f"step={step} hq={metrics['joint_hq']:.3f} modes={metrics['modes']} cond={metrics['cond_acc']:.3f} "
                      f"csw1={metrics['conditional_sw1']:.3f} train_s={elapsed_train:.1f}", flush=True)
    final, x, c = evaluate(cfg["final_samples"])
    probe, panels = conditional_probe(ema_g, ema_prior, ema_noise, schedule, toy, cfg["probe_samples"])
    final.update(probe)
    if cfg["noise"] in ("fixed", "learned"):
        final["noise_mean_norm"] = float(ema_noise.table.mean(0).norm())
        final["noise_cov_eigenvalues"] = torch.linalg.eigvalsh(torch.cov(ema_noise.table.T)).cpu().tolist()
        final["noise_table_unique"] = len(torch.unique(ema_noise.table, dim=0))
    else:
        # These are population moments; an unused initialization table is irrelevant.
        final["noise_mean_norm"] = 0.0
        final["noise_cov_eigenvalues"] = [1.0, 1.0] if cfg["noise"] == "gaussian" else [0.0, 0.0]
        final["noise_table_unique"] = None
    final["unique_outputs"] = len(torch.unique(x, dim=0))
    er = torch.Generator(device=device).manual_seed(99003)
    floor_a, floor_b = toy.sample(c, er), toy.sample(c, er)
    floor = grid_metrics(floor_a, c, toy, floor_b)
    np.savez_compressed(out / "final_samples.npz", x=x.cpu().numpy(), c=c.cpu().numpy())
    render(out, x.cpu().numpy(), c.cpu().numpy(), toy, panels)
    if cfg["save_checkpoint"]:
        torch.save({"config": cfg, "G": ema_g.state_dict(), "prior": ema_prior.state_dict(),
                    "noise": ema_noise.state_dict()}, out / "final.pt")
    summary = {"config": cfg, "final": final, "reference_floor": floor,
               "train_seconds": elapsed_train, "samples_per_second": cfg["steps"] * cfg["batch_size"] / elapsed_train,
               "real_draws": 2 * cfg["steps"] * cfg["batch_size"], "total_seconds": time.perf_counter() - total_start,
               "environment": env, "provenance": provenance,
               "parameters": {name: sum(p.numel() for p in model.parameters())
                              for name, model in (("G", g), ("D", d), ("prior", prior), ("noise", noise))}}
    write_json(out / "summary.json", summary)
    print(json.dumps({"final": json_safe(final)}, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="TOML/YAML overrides (default: configs/denoising/default.toml)")
    args = parser.parse_args()
    user = read_config(args.config)
    if not isinstance(user, dict) or (set(user) - set(DEFAULTS)):
        raise ValueError("config must be a mapping with only known keys")
    train({**DEFAULTS, **user})


if __name__ == "__main__":
    main()
