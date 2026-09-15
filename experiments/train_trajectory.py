#!/usr/bin/env python
"""Particle DDGAN on conditional futures. No arguments runs default.yaml on CUDA."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.denoising_toy import DiffusionSchedule, DrawSource
from lib.gan_loss import GANLoss
from lib.grad_regularizers import GradRegularizer
from lib.vicreg_loss import VICRegLikeLoss
from lib.trajectory import Routes, TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryCritic, generate, metrics
from lib.trajectory_visuals import render

DEFAULTS = {
    "model": "ddgan", "d_mode": "ucd", "prior": "learned", "noise": "gaussian",
    "geometry_mode": "discrete", "d_architecture": "mlp", "d_temporal_width": 64,
    "seed": 24002, "length": 64, "alpha_bar": [1.0, .9, .5, .05, .0001],
    "z_dim": 32, "num_particles": 20000, "noise_particles": 1024,
    "width": 32, "d_width": 256, "steps": 10000, "batch_size": 128,
    "lr": .0006, "d_lr_mult": 1.5, "prior_lr_mult": 10.0, "noise_lr_mult": 1.0,
    "beta1": 0.0, "prior_reg": 1.0, "ucd_lambda": .02, "ema": .995,
    "reg_coeff": 1.0, "reg_kappa": 1.0, "reg_every": 4,
    "log_interval": 250, "eval_per_context": 512, "save_checkpoint": True,
    "out_dir": "results/trajectory/default",
}


def validate(cfg):
    for key, values in dict(model=("ddgan", "gan"), d_mode=("ucd", "concat"),
                            prior=("learned", "fixed", "gaussian"), noise=("learned", "fixed", "gaussian"),
                            geometry_mode=("discrete", "continuous"), d_architecture=("mlp", "temporal")).items():
        if cfg[key] not in values:
            raise ValueError(f"invalid {key}")
    for key in ("length", "z_dim", "num_particles", "noise_particles", "width", "d_width", "d_temporal_width", "steps", "batch_size", "reg_every", "log_interval", "eval_per_context"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["length"] < 16 or cfg["length"] % 4 or min(cfg["batch_size"], cfg["num_particles"], cfg["noise_particles"]) < 2:
        raise ValueError("length must be a multiple of four >=16; batches/tables >=2")
    if not 0 <= cfg["ema"] < 1 or not 0 <= cfg["beta1"] < 1:
        raise ValueError("invalid EMA/beta1")
    if cfg["d_architecture"] == "temporal" and cfg["d_width"] < 4:
        raise ValueError("temporal D requires d_width >= 4")
    for key in ("lr", "d_lr_mult", "prior_lr_mult", "noise_lr_mult"):
        if cfg[key] <= 0:
            raise ValueError(key)
    for key in ("prior_reg", "ucd_lambda", "reg_coeff", "reg_kappa"):
        if cfg[key] < 0:
            raise ValueError(key)
    if cfg["model"] == "gan" and cfg["noise"] != "gaussian":
        raise ValueError("step noise does not apply to a one-shot GAN")
    DiffusionSchedule(cfg["alpha_bar"])


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def train(cfg):
    validate(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda:0"
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() or (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    from experiments.run_grid import code_provenance
    provenance = code_provenance(__file__, sys.executable)
    write_json(out / "provenance.json", provenance)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, digest in provenance["sources"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(name, data)
    env = dict(gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               torch=torch.__version__, cuda=torch.version.cuda)
    write_json(out / "environment.json", env)
    rngs = [torch.Generator(device=device).manual_seed(cfg["seed"] + i) for i in range(11, 17)]
    toy = Routes(cfg["length"], device, cfg["geometry_mode"])
    schedule = DiffusionSchedule(cfg["alpha_bar"]).to(device)
    prior = DrawSource(cfg["prior"], cfg["num_particles"], cfg["z_dim"], cfg["seed"]+101, device)
    noise = DrawSource(cfg["noise"], cfg["noise_particles"], 2*cfg["length"], cfg["seed"]+102, device)
    g, d = TrajectoryGenerator(cfg).to(device), TrajectoryDiscriminator(cfg).to(device)
    ema_g, ema_prior, ema_noise = copy.deepcopy(g), copy.deepcopy(prior), copy.deepcopy(noise)
    for m in (ema_g, ema_prior, ema_noise):
        m.requires_grad_(False)
    groups = [{"params": list(g.parameters()), "lr": cfg["lr"]}]
    for source, mult in ((prior, "prior_lr_mult"), (noise, "noise_lr_mult")):
        if source.kind == "learned":
            groups.append({"params": list(source.parameters()), "lr": cfg["lr"] * cfg[mult]})
    opt_g = torch.optim.Adam(groups, betas=(cfg["beta1"], .999), fused=True)
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg["lr"]*cfg["d_lr_mult"], betas=(cfg["beta1"], .999), fused=True)
    gan, vic = GANLoss("logistic", "rp"), VICRegLikeLoss()
    reg = GradRegularizer("b_cap", cfg["reg_coeff"], kappa=cfg["reg_kappa"], lazy_k=cfg["reg_every"])

    def batch():
        c, geom, x0 = toy.batch(cfg["batch_size"], rngs[0])
        context = toy.condition(geom)
        if cfg["model"] == "gan":
            return c, context, x0, None, None
        t = torch.randint(1, schedule.steps+1, (len(c),), device=device, generator=rngs[1])
        real, xt = schedule.forward_pair(x0, t, rngs[2])
        return c, context, real, xt, t

    def fake(c, context, xt, t):
        z, ids = prior.sample(len(c), rngs[3])
        clean = g(z, c, context, xt, t)
        if not g.diffusion:
            return clean, ids
        eta = noise.sample(len(c), rngs[4])[0].reshape_as(clean)
        return schedule.reverse(clean, xt, t, eta), ids

    print(f"START device={env['gpu']} visible={env['visible_devices']} model={cfg['model']} prior={cfg['prior']} noise={cfg['noise']} steps={cfg['steps']}", flush=True)
    print(f"D={cfg['d_architecture']} geometry={cfg['geometry_mode']}", flush=True)
    print("Constant LR; shared DDGAN posterior, Rp logistic, joint time/class UCD, exact lazy bcap; endpoint evaluation only", flush=True)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, cfg["steps"]+1):
            d.requires_grad_(True)
            c, context, real, xt, t = batch()
            with torch.no_grad():
                xf, _ = fake(c, context, xt, t)
            dr, cr = d(real, c, context, xt, t)
            df, cf = d(xf, c, context, xt, t)
            ld = gan.d_loss(dr, df)
            if cfg["d_mode"] == "ucd" and cfg["ucd_lambda"]:
                target = d.ucd_labels(c, t)
                ld = ld + cfg["ucd_lambda"] * (F.cross_entropy(cr, target) + F.cross_entropy(cf, target))
            penalty, _ = reg.penalty(TrajectoryCritic(d, c, context, xt, t), real, xf, step, rngs[5], collect_stats=False)
            ld = ld + penalty
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            d.requires_grad_(False)
            c, context, real, xt, t = batch()
            xf, ids = fake(c, context, xt, t)
            df = d(xf, c, context, xt, t)[0]
            with torch.no_grad():
                dr = d(real, c, context, xt, t)[0]
            lg = gan.g_loss(df, dr)
            if prior.kind == "learned" and cfg["prior_reg"]:
                selected = prior.table[ids.unique()]
                if len(selected) > 1:
                    lg = lg + cfg["prior_reg"] * vic(selected)
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            with torch.no_grad():
                for target, source in ((ema_g, g), (ema_prior, prior), (ema_noise, noise)):
                    for pe, p in zip(target.parameters(), source.parameters()):
                        pe.lerp_(p, 1-cfg["ema"])
            if step % cfg["log_interval"] == 0 or step == cfg["steps"]:
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-total_start
                row = dict(step=step, d_loss=float(ld.detach()), g_loss=float(lg.detach()), train_seconds=elapsed,
                           samples_per_second=step*cfg["batch_size"]/elapsed)
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                print(f"step={step}/{cfg['steps']} D={row['d_loss']:.4f} G={row['g_loss']:.4f} samples/s={row['samples_per_second']:.0f} train_s={elapsed:.1f}", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter()-total_start
    final, floors, samples = {}, {}, {}
    with torch.no_grad():
        for split in ("train", "test"):
            er = [torch.Generator(device=device).manual_seed(99000+i) for i in range(5)]
            cc, gg = toy.contexts(split)
            group = torch.arange(len(cc), device=device).repeat_interleave(cfg["eval_per_context"])
            c, geom = cc[group], gg[group]
            context = toy.condition(geom)
            chunks = [generate(ema_g, ema_prior, ema_noise, schedule, c[j:j+256], context[j:j+256], er[:3]) for j in range(0, len(c), 256)]
            x = torch.cat(chunks)
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError("Nonfinite generated trajectories")
            real = toy.sample(c, geom, er[3])[0]
            real2 = toy.sample(c, geom, er[4])[0]
            final[split] = metrics(toy, x, real, c, geom, group)
            floors[split] = metrics(toy, real2, real, c, geom, group)
            samples[split] = dict(x=x.cpu().numpy(), real=real.cpu().numpy(), c=c.cpu().numpy(), geom=geom.cpu().numpy(), group=group.cpu().numpy())
            np.savez_compressed(out / f"{split}_samples.npz", **samples[split])
        # Fix the complete sequence of latent particle IDs across reverse steps.
        # Each column rerolls all Gaussian randomness. This is an intervention,
        # not the ordinary sampler (which draws a fresh particle at every step).
        probes = None
        if prior.kind in ("fixed", "learned"):
            er = [torch.Generator(device=device).manual_seed(88100+i) for i in range(3)]
            c0, geom0 = toy.contexts("test")
            c, geom = c0[:1].expand(96), geom0[:1].expand(96, -1)
            ids = torch.arange(4, device=device).repeat_interleave(24)
            fixed = ids[None].expand(schedule.steps, -1)
            varied_noise = generate(ema_g, ema_prior, ema_noise, schedule, c, toy.condition(geom), er, fixed_ids=fixed)
            varied_particle = generate(ema_g, ema_prior, ema_noise, schedule, c, toy.condition(geom), er, fixed_random=True)
            probes = dict(noise=varied_noise.cpu().numpy(), particles=varied_particle.cpu().numpy(), geom=geom.cpu().numpy())
            np.savez_compressed(out / "particle_probe.npz", **probes)
        render(out, samples, final, probes)
    if cfg["save_checkpoint"]:
        torch.save(dict(config=cfg, G=ema_g.state_dict(), prior=ema_prior.state_dict(), noise=ema_noise.state_dict()), out / "final.pt")
    summary = dict(config=cfg, final=final, reference_floor=floors, train_seconds=train_seconds,
                   samples_per_second=cfg["steps"]*cfg["batch_size"]/train_seconds,
                   real_draws=2*cfg["steps"]*cfg["batch_size"], total_seconds=time.perf_counter()-total_start,
                   environment=env, provenance=provenance,
                   parameters={name: sum(p.numel() for p in m.parameters()) for name, m in (("G", g), ("D", d), ("prior", prior))})
    write_json(out / "summary.json", summary)
    print("COMPLETE "+json.dumps({s: {k: v for k, v in final[s].items() if k != "contexts"} for s in final}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/trajectory/default.yaml"))
    args = parser.parse_args()
    user = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(user, dict) or set(user)-set(DEFAULTS):
        raise ValueError("Config must contain only known keys")
    train({**DEFAULTS, **user})


if __name__ == "__main__":
    main()
