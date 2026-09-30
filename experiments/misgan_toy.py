#!/usr/bin/env python
"""MisGAN (Li, Jiang & Marlin, ICLR 2019) on the 100-Gaussian grid lifted to 8D.

Three generator/critic pairs trained jointly, each with its own recipe-built
particle prior, optimizers and critic penalty, and the recipe's RpGAN loss:

* data   G_x(z_x) -> x;   D_x scores masked data f(x, m) = x*m + fill*(1-m)
* mask   G_m(z_m) -> m = sigmoid(logits / mask_temperature);   D_m scores masks
* imputer G_i([x*m, m, z_i]) -> x_hat = m*x + (1-m)*G_i(.);  D_i scores x_hat
  against complete G_x samples. The particle draw z_i replaces the paper's
  noise omega as an extra input.

Per-generator objectives follow the official code: G_m <- L_m + alpha*L_x,
G_x <- L_x + beta*L_i, G_i <- L_i. They come from ONE scalar,
``L_m + alpha*(L_x + beta*L_i)``: each generator only appears in its own terms,
and Adam is invariant to the constant factor this puts on G_x (alpha) and
G_i (alpha*beta).

Arms (``arm``): oracle (complete data; imputer referenced to true complete
rows), zerofill (plain GAN on f_0(x, m)), misgan, misgan_realmask (fakes masked
with masks resampled from the training pool), misgan_paired (each fake masked
with its paired real row's mask), misgan_gauss (frozen-Gaussian priors for all
three generators, via ``make_prior(learnable=False)``), misgan_hard (G_m masks
binarized with a straight-through gradient), misgan_detach (G_m's masks are
detached in L_x, so G_m trains on L_m alone). G_m is trained in every arm, so
mask metrics exist everywhere.

    python experiments/misgan_toy.py --arm misgan --mechanism mcar_p50
    python experiments/misgan_toy.py --write_grid configs/misgan   # grid configs
    tail -f runs/misgan/mcar_p50__misgan.log
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particlegan import InputNoise, get_recipe, scale_learning_rates  # noqa: E402
from particlegan.training import input_noise_std, output_noise_std  # noqa: E402

from experiments.config import merge_config, read_config  # noqa: E402
from lib.misgan import (DIM, MECHANISMS, Problem, bayes_posterior, generation_metrics,  # noqa: E402
                        imputation_metrics, mask_metrics, masked_sqdist, select_sigma)
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator  # noqa: E402

ARMS = ("oracle", "zerofill", "misgan", "misgan_realmask", "misgan_paired", "misgan_gauss", "misgan_hard",
        "misgan_detach", "aegan_recon", "aegan_ce")
# Arms run on every mechanism; the rest only on mcar_p50.
ALL_MECHANISM_ARMS = ("oracle", "zerofill", "misgan", "misgan_realmask", "misgan_paired", "misgan_detach",
                      "misgan_gauss", "aegan_ce")

DEFAULTS = {
    "arm": "misgan", "mechanism": "mcar_p50", "total_steps": 7000, "batch_size": 2048,
    "eval_every": 500, "alpha": 0.2, "beta": 0.1, "mask_temperature": 0.66, "fill": 0.0,
    "mask_z_dim": 8, "imputer_z_dim": 8, "n_train": 20000, "n_test": 10000,
    "impute_draws": 16, "data_seed": 0, "seed": 1234, "d_fourier": 2, "recipe_overrides": {},
    "recon_weight": 1.0,
    "out_dir": "results/misgan/runs/default", "log_path": "",
}


@torch.no_grad()
def update_ema(average, current, decay):
    for target, source in zip(average.parameters(), current.parameters()):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(average.buffers(), current.buffers()):
        target.copy_(source)


def xavier(*modules):
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


def build_pair(recipe, G, D, prior, noise_gen):
    """One generator/critic pair: recipe optimizers, penalty, noisy critic, EMA copies."""
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
    return SimpleNamespace(
        G=G, D=D, prior=prior, opt_g=opt_g, opt_d=opt_d,
        penalty=recipe.make_critic_penalty(opt_d),
        noisy_d=InputNoise(D, generator=noise_gen),
        base=[[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)],
        ema_g=copy.deepcopy(G).eval().requires_grad_(False),
        ema_prior=copy.deepcopy(prior).eval().requires_grad_(False))


def imputer_net(arm, cfg, recipe):
    """The arm's imputer network and the z_dim of its noise prior (None: no prior).

    G_i([x*m, m, z_i]) -> x for the MisGAN arms; E([x*m, m, z2]) -> z_x for
    aegan_recon; E([x*m, m]) -> logits over G_x's particle table for aegan_ce.
    """
    zi = cfg["imputer_z_dim"]
    if arm == "aegan_ce":
        return SimpleMLPGenerator(z_dim=2 * DIM, out_dim=recipe.num_particles), None
    if arm == "aegan_recon":
        return SimpleMLPGenerator(z_dim=2 * DIM + zi, out_dim=recipe.z_dim), zi
    return SimpleMLPGenerator(z_dim=2 * DIM + zi, out_dim=DIM), zi


def impute(arm, net, prior, gx, prior_x, x, m, generator=None):
    """One imputation per row: m*x + (1-m)*fill, with the arm's fill."""
    inputs = torch.cat([x * m, m], 1)
    if arm == "aegan_ce":
        k = torch.multinomial(torch.softmax(net(inputs), 1), 1, generator=generator).squeeze(1)
        fill = gx(prior_x.z[k])
    else:
        z, _ = prior.sample(len(x), generator=generator)
        fill = net(torch.cat([inputs, z], 1))
        if arm == "aegan_recon":
            fill = gx(fill)
    return m * x + (1 - m) * fill


def train(cfg):
    arm, mech = cfg["arm"], cfg["mechanism"]
    if arm not in ARMS or mech not in MECHANISMS:
        raise ValueError(f"arm must be one of {ARMS}, mechanism one of {MECHANISMS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    name = Path(cfg["out_dir"]).name
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["log_path"] or f"runs/misgan/{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    problem = Problem(mech, cfg["n_train"], cfg["n_test"], cfg["data_seed"], device)
    post, _ = bayes_posterior(problem, problem.x_test, problem.m_test)
    recipe = get_recipe(total_steps=cfg["total_steps"], batch_size=cfg["batch_size"], **cfg["recipe_overrides"])
    learnable = arm != "misgan_gauss"  # the frozen-Gaussian ablation switch
    aegan = arm.startswith("aegan")
    zx, zm = recipe.z_dim, cfg["mask_z_dim"]
    Gx, Gm = SimpleMLPGenerator(z_dim=zx, out_dim=DIM), SimpleMLPGenerator(z_dim=zm, out_dim=DIM)
    net, net_z = imputer_net(arm, cfg, recipe)
    fourier = cfg["d_fourier"]
    Dx, Dm = SimpleMLPDiscriminator(in_dim=DIM, fourier=fourier), SimpleMLPDiscriminator(in_dim=DIM, fourier=0)
    # D_i scores imputations (MisGAN arms); D_z scores E's latents vs prior draws (aegan_recon).
    Dnet = (None if arm == "aegan_ce" else SimpleMLPDiscriminator(in_dim=zx, fourier=0) if aegan
            else SimpleMLPDiscriminator(in_dim=DIM, fourier=fourier))
    xavier(*(module for module in (Gx, Gm, net, Dx, Dm, Dnet) if module is not None))
    noise_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 5)
    data_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    pairs = {}
    roles = [("x", Gx, Dx, zx), ("m", Gm, Dm, zm)]
    if Dnet is not None:
        roles.append(("e" if aegan else "i", net, Dnet, net_z))
    for key, G, D, z_dim in roles:
        prior = recipe.make_prior(z_dim=z_dim, learnable=learnable)
        pairs[key] = build_pair(recipe, G.to(device), D.to(device), prior.to(device), noise_gen)
    px, pm, pi, pe = pairs["x"], pairs["m"], pairs.get("i"), pairs.get("e")
    if arm == "aegan_ce":  # E alone: cross-entropy to the particle posterior, no critic
        net.to(device)
        opt_e = recipe.make_generator_optimizer(net.parameters())
        base_e = [[group["lr"] for group in opt_e.param_groups]]
        ema_e = copy.deepcopy(net).eval().requires_grad_(False)
        sigma_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 7)
        sigma_rows = problem.x_train[:4096], problem.m_train[:4096]
    imp = pi or pe
    gan = recipe.make_loss()
    alpha, beta, temp, fill, B = cfg["alpha"], cfg["beta"], cfg["mask_temperature"], cfg["fill"], cfg["batch_size"]
    weight = cfg["recon_weight"]

    def f(x, m):  # MisGAN masking operator f_tau
        return x * m + fill * (1 - m)

    def gen_masks(G, prior, n, generator=None):
        soft = torch.sigmoid(G(prior.sample(n, generator=generator)[0]) / temp)
        if arm != "misgan_hard":
            return soft
        return (soft > 0.5).float() + soft - soft.detach()  # straight-through

    def masks_for_fakes(m_real, m_gen):
        if arm == "misgan_realmask":
            return problem.m_train[torch.randint(len(problem.m_train), (len(m_real),), device=device,
                                                 generator=data_gen)]
        if arm == "misgan_paired":
            return m_real
        if arm == "misgan_detach" or aegan:
            return m_gen.detach()
        return m_gen  # misgan, misgan_gauss, misgan_hard (oracle/zerofill do not mask fakes)

    def data_views(gx, m_fake, x, m, x_full):
        """(fake, real) inputs of D_x under this arm's view of the data."""
        if arm == "oracle":
            return gx, x_full
        if arm == "zerofill":
            return gx, f(x, m)
        return f(gx, m_fake), f(x, m)

    def encode(x, m, z2):
        return net(torch.cat([x * m, m, z2], 1))

    def ce_sigma():
        with torch.no_grad():
            return select_sigma(px.G(px.prior.z), *sigma_rows, sigma_gen)[0]

    @torch.no_grad()
    def evaluate():
        gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 999)
        n = cfg["n_test"]
        out = generation_metrics(problem, px.ema_g(px.ema_prior.sample(n, generator=gen)[0]))
        out.update(mask_metrics(problem, torch.sigmoid(pm.ema_g(pm.ema_prior.sample(n, generator=gen)[0]) / temp)))
        imp_net, imp_prior = (ema_e, None) if arm == "aegan_ce" else (imp.ema_g, imp.ema_prior)
        draws = torch.stack([impute(arm, imp_net, imp_prior, px.ema_g, px.ema_prior, problem.x_test,
                                    problem.m_test, gen) for _ in range(cfg["impute_draws"])])
        out.update(imputation_metrics(problem, draws, post))
        return out

    def line(step, metrics, seconds):
        keys = ("modes", "hq", "swd", "off", "m_mae", "m_tv", "m_soft", "acc", "acc_lo", "itv", "istd",
                "rmse", "imodes", "ihq") + (("sigma",) if "sigma" in metrics else ())
        body = " ".join(f"{k}={metrics[k]:.4g}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}"
                        for k in keys)
        return f"{name} step={step} {body} t={seconds:.0f}s"

    log(f"# {name} arm={arm} mechanism={mech} steps={recipe.total_steps} batch={B} device={device}")
    history, started = [], time.monotonic()
    sigma = ce_sigma() if arm == "aegan_ce" else None
    zero = torch.zeros((), device=device)
    for step in range(recipe.total_steps):
        for pair in pairs.values():
            scale_learning_rates(step, recipe, (pair.opt_g, pair.opt_d), pair.base, pair.prior)
            pair.noisy_d.std = input_noise_std(recipe, step)
        if arm == "aegan_ce":
            scale_learning_rates(step, recipe, (opt_e,), base_e)
        sigma_out = output_noise_std(recipe, step)

        # 1) Critics: D_m (masks), D_x (masked data), and D_i (imputations vs
        #    G_x) or D_z (encoded latents vs prior draws).
        x, m, x_full = problem.train_batch(B, data_gen)
        with torch.no_grad():
            gx = px.G(px.prior.sample(B)[0])
            gx_noisy = gx + sigma_out * torch.randn(gx.shape, generator=noise_gen, device=device)
            m_gen = gen_masks(pm.G, pm.prior, B)
            if pi is not None:
                imputed = impute(arm, pi.G, pi.prior, None, None, x, m)
                reference = x_full if arm == "oracle" else gx
            if pe is not None:
                z_prior = px.prior.sample(B)[0]
                z_enc = encode(x, m, pe.prior.sample(B)[0])
        fake_x, real_x = data_views(gx_noisy, masks_for_fakes(m, m_gen), x, m, x_full)
        d_loss_m = gan.d_loss(pm.noisy_d(m), pm.noisy_d(m_gen)) + pm.penalty(pm.noisy_d, m, m_gen)
        d_loss_x = (gan.d_loss(px.noisy_d(real_x), px.noisy_d(fake_x))
                    + px.penalty(px.noisy_d, real_x, fake_x))
        d_loss = d_loss_m + d_loss_x  # independent critics
        if pi is not None:
            d_loss = d_loss + (gan.d_loss(pi.noisy_d(reference), pi.noisy_d(imputed))
                               + pi.penalty(pi.noisy_d, reference, imputed))
        if pe is not None:
            d_loss = d_loss + (gan.d_loss(pe.noisy_d(z_prior), pe.noisy_d(z_enc))
                               + pe.penalty(pe.noisy_d, z_prior, z_enc))
        for pair in pairs.values():
            pair.opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        for pair in pairs.values():
            pair.opt_d.step()

        # 2) Generators: L_m + alpha * (L_x + beta*L_i + L_z + w*rec + CE), critics frozen.
        for pair in pairs.values():
            pair.D.requires_grad_(False)
        x, m, x_full = problem.train_batch(B, data_gen)
        gx = px.G(px.prior.sample(B)[0])
        gx_noisy = gx + sigma_out * torch.randn(gx.shape, generator=noise_gen, device=device)
        m_gen = gen_masks(pm.G, pm.prior, B)
        loss_m = gan.g_loss(pm.noisy_d(m_gen), pm.noisy_d(m))
        fake_x, real_x = data_views(gx_noisy, masks_for_fakes(m, m_gen), x, m, x_full)
        loss_x = gan.g_loss(px.noisy_d(fake_x), px.noisy_d(real_x))
        loss_i = loss_z = rec = ce = zero
        if pi is not None:
            imputed = impute(arm, pi.G, pi.prior, None, None, x, m)
            loss_i = gan.g_loss(pi.noisy_d(imputed), pi.noisy_d(x_full if arm == "oracle" else gx))
        if pe is not None:
            z_enc = encode(x, m, pe.prior.sample(B)[0])
            loss_z = gan.g_loss(pe.noisy_d(z_enc), pe.noisy_d(px.prior.sample(B)[0].detach()))
            rec = weight * ((m * (px.G(z_enc) - x)) ** 2).sum(1).mean()
        if arm == "aegan_ce":
            with torch.no_grad():
                target = torch.softmax(-masked_sqdist(x, m, px.G(px.prior.z)) / (2 * sigma * sigma), 1)
            ce = -(target * torch.log_softmax(net(torch.cat([x * m, m], 1)), 1)).sum(1).mean()
        g_loss = loss_m + alpha * (loss_x + beta * loss_i + loss_z + rec + ce)
        for pair in pairs.values():
            pair.opt_g.zero_grad(set_to_none=True)
        if arm == "aegan_ce":
            opt_e.zero_grad(set_to_none=True)
        g_loss.backward()
        for pair in pairs.values():
            pair.opt_g.step()
            pair.D.requires_grad_(True)
            update_ema(pair.ema_g, pair.G, recipe.ema_decay)
            update_ema(pair.ema_prior, pair.prior, recipe.ema_decay)
        if arm == "aegan_ce":
            opt_e.step()
            update_ema(ema_e, net, recipe.ema_decay)

        done = step + 1
        if done % cfg["eval_every"] == 0 or done == recipe.total_steps:
            metrics = evaluate()
            losses = dict(d_loss_x=d_loss_x, loss_x=loss_x, loss_m=loss_m, loss_i=loss_i, loss_z=loss_z,
                          rec=rec, ce=ce)
            metrics.update(step=done, **{k: float(v.detach()) for k, v in losses.items()})
            if arm == "aegan_ce":
                metrics["sigma"] = sigma
                sigma = ce_sigma()  # refit the kernel width on the current G_x
            history.append(metrics)
            log(line(done, metrics, time.monotonic() - started))
            if not all(torch.isfinite(torch.tensor(v, dtype=torch.float64))
                       for v in metrics.values() if isinstance(v, float) and v == v):
                raise RuntimeError("non-finite metrics")
    torch.save({"gx": px.ema_g.state_dict(), "prior_x": px.ema_prior.state_dict(),
                "imputer": (ema_e if arm == "aegan_ce" else imp.ema_g).state_dict(),
                "imputer_prior": None if arm == "aegan_ce" else imp.ema_prior.state_dict()},
               out_dir / "ckpt.pt")
    summary = {"config": cfg, "final": history[-1], "history": history,
               "seconds": round(time.monotonic() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log_file.close()
    return summary


def write_grid(directory, steps):
    """One TOML per (mechanism, arm) plus manifest.json for run_grid.py."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runs = [(mech, arm, arm, steps) for mech in MECHANISMS for arm in ALL_MECHANISM_ARMS]
    runs += [("mcar_p50", "misgan_hard", "misgan_hard", steps)]
    # Follow-ups: does learned-mask G_x converge with a longer budget; is G_m's
    # L_x gradient what hurts G_x at high missingness.
    runs += [("mcar_p50", "misgan", "misgan_long", 3 * steps)]
    paths = []
    runs = [(mech, arm, label, run_steps, "") for mech, arm, label, run_steps in runs]
    # aegan_recon: the masked reconstruction weight w is swept.
    runs += [(mech, "aegan_recon", f"aegan_recon_w{w:g}", steps, f"recon_weight = {w}\n")
             for mech in MECHANISMS for w in (0.1, 1.0)]
    for mech, arm, label, run_steps, extra in runs:
        name = f"{mech}__{label}"
        path = directory / f"{name}.toml"
        path.write_text(f'arm = "{arm}"\nmechanism = "{mech}"\ntotal_steps = {run_steps}\n{extra}'
                        f'out_dir = "results/misgan/runs/{name}"\nlog_path = "runs/misgan/{name}.log"\n')
        paths.append(str(path))
    (directory / "manifest.json").write_text(json.dumps(paths, indent=1) + "\n")
    print(f"wrote {len(paths)} configs to {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--write_grid", help="write grid configs + manifest.json into this directory and exit")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=json.loads if isinstance(value, dict) else type(value))
    args = parser.parse_args()
    if args.write_grid:
        return write_grid(args.write_grid, args.total_steps or DEFAULTS["total_steps"])
    user = read_config(args.config) if args.config else {}
    if unknown := set(user) - set(DEFAULTS):
        parser.error(f"unknown config keys: {sorted(unknown)}")
    user.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    train(merge_config(DEFAULTS, user))


if __name__ == "__main__":
    main()
