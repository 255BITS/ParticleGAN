#!/usr/bin/env python
"""Particle DDGAN on HumanAct12 motion completion. No arguments runs the baseline on CUDA."""
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.denoising_toy import DiffusionSchedule, DrawSource
from lib.gan_loss import GANLoss
from lib.grad_regularizers import GradRegularizer
from lib.vicreg_loss import VICRegLikeLoss
from lib.trajectory import TrajectoryGenerator, TrajectoryDiscriminator, generate
from lib.motion import MotionData, motion_metrics, baselines
from lib.motion_visuals import render

DEFAULTS = {
    "model": "ddgan", "d_mode": "ucd", "prior": "learned", "noise": "gaussian",
    "d_architecture": "hybrid", "d_temporal_width": 32,
    "channels": 72, "classes": 12, "context_dim": 576, "past_length": 8,
    "data_dir": "data/humanact12/raw/HumanAct12", "expected_clips": 1191,
    "data_sha256": "e8c166350f2704ebdd7e615ff432ac0b1e805c76b2730acacd35633ff2f81903",
    "train_subjects": [1,2,3,4,5,6,7,8,9], "validation_subjects": [10], "test_subjects": [11,12],
    "display_fps": 15, "eval_per_class": 16, "eval_samples": 8,
    "seed": 24002, "length": 16, "alpha_bar": [1.0, .9, .5, .05, .0001],
    "z_dim": 32, "num_particles": 20000, "noise_particles": 1024,
    "width": 32, "d_width": 232, "steps": 10000, "batch_size": 128,
    "lr": .0006, "d_lr_mult": 1.5, "prior_lr_mult": 10.0, "noise_lr_mult": 1.0,
    "beta1": 0.0, "prior_reg": 1.0, "ucd_lambda": .02, "ema": .995,
    "reg_coeff": 1.0, "reg_kappa": 1.0, "reg_every": 4,
    "log_interval": 250, "save_checkpoint": True,
    "out_dir": "results/motion/default",
}


def validate(cfg):
    from experiments.train_trajectory import validate as validate_trajectory, DEFAULTS as TOY
    validate_trajectory({**TOY, **{k: v for k, v in cfg.items() if k in TOY}})
    if cfg['channels'] != 72 or cfg['classes'] != 12 or cfg['context_dim'] != 72*cfg['past_length']:
        raise ValueError('HumanAct12 requires 72 channels, 12 classes, and a complete observed prefix')
    for key in ('past_length', 'eval_per_class', 'eval_samples', 'expected_clips', 'display_fps'):
        if type(cfg[key]) is not int or cfg[key] < (2 if key in ('past_length', 'eval_samples') else 1):
            raise ValueError(key)
    sets = [set(cfg[s+'_subjects']) for s in ('train', 'validation', 'test')]
    if any(not a for a in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Subject splits must be nonempty and disjoint')


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
    data = MotionData(cfg, device)
    if data.manifest["source_fingerprint"] != cfg["data_sha256"]:
        raise ValueError("Dataset checksum differs from config; inspect the source manifest before updating data_sha256")
    write_json(out / "data_manifest.json", data.manifest)
    schedule = DiffusionSchedule(cfg["alpha_bar"]).to(device)
    prior = DrawSource(cfg["prior"], cfg["num_particles"], cfg["z_dim"], cfg["seed"]+101, device)
    noise = DrawSource(cfg["noise"], cfg["noise_particles"], cfg["channels"]*cfg["length"], cfg["seed"]+102, device)
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
        c, context, x0 = data.batch(cfg["batch_size"], rngs[0])
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
    print(f"D={cfg['d_architecture']} prefix={cfg['past_length']} future={cfg['length']} normalization={data.scale:.6f}", flush=True)
    print("Constant LR; shared DDGAN posterior, Rp logistic, joint time/class UCD, exact lazy bcap; endpoint evaluation only", flush=True)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    from lib.conditional_training import run_updates
    train_seconds = run_updates(cfg, out, g, d, prior, noise, ema_g, ema_prior, ema_noise,
                                opt_g, opt_d, gan, vic, reg, batch, fake, rngs, total_start)
    final, references, samples = {}, {}, {}
    with torch.no_grad():
        for split in ('train', 'validation', 'test'):
            c, context, real, windows = data.evaluation(split, cfg['eval_per_class'])
            er = [torch.Generator(device=device).manual_seed(99000+i) for i in range(3)]
            k = cfg['eval_samples']
            cc, ctx = c.repeat_interleave(k), context.repeat_interleave(k, 0)
            x = torch.cat([generate(ema_g, ema_prior, ema_noise, schedule, cc[j:j+128], ctx[j:j+128], er)
                           for j in range(0, len(cc), 128)]).reshape(len(c), k, cfg['channels'], cfg['length'])
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError('Nonfinite generated motion')
            final[split] = motion_metrics(x, real, context, cfg['past_length'], c)
            references[split] = {name: motion_metrics(value[:, None], real, context, cfg['past_length'], c)
                                 for name, value in baselines(real, context, cfg['past_length']).items()}
            samples[split] = dict(x=x.cpu().numpy(), real=real.cpu().numpy(), context=context.cpu().numpy(),
                                  c=c.cpu().numpy())
            np.savez_compressed(out / f'{split}_samples.npz', **samples[split])
            write_json(out / f'{split}_windows.json', windows)
            print(f'EVALUATED {split}: {len(c)} prefixes x {k} futures', flush=True)
        # Does the observed prefix influence completion with all random draws held fixed?
        c, context, real, _ = data.evaluation('test', cfg['eval_per_class'])
        shuffled = context.clone()
        for label in c.unique():
            ids = (c == label).nonzero().flatten()
            shuffled[ids] = context[ids.roll(1)]
        er = [torch.Generator(device=device).manual_seed(77700+i) for i in range(3)]
        original = generate(ema_g, ema_prior, ema_noise, schedule, c, context, er)
        er = [torch.Generator(device=device).manual_seed(77700+i) for i in range(3)]
        changed = generate(ema_g, ema_prior, ema_noise, schedule, c, shuffled, er)
        from lib.motion import joints
        prefix_sensitivity = float((joints(original)-joints(changed)).norm(dim=-1).mean())
    render(out, samples, cfg)
    if cfg["save_checkpoint"]:
        torch.save(dict(config=cfg, G=ema_g.state_dict(), prior=ema_prior.state_dict(), noise=ema_noise.state_dict(), normalization_scale=data.scale, data_fingerprint=data.manifest["fingerprint"]), out / "final.pt")
    summary = dict(config=cfg, final=final, references=references, prefix_sensitivity=prefix_sensitivity, data_fingerprint=data.manifest["fingerprint"], train_seconds=train_seconds,
                   samples_per_second=cfg["steps"]*cfg["batch_size"]/train_seconds,
                   real_draws=2*cfg["steps"]*cfg["batch_size"], total_seconds=time.perf_counter()-total_start,
                   environment=env, provenance=provenance,
                   parameters={name: sum(p.numel() for p in m.parameters()) for name, m in (("G", g), ("D", d), ("prior", prior))})
    write_json(out / "summary.json", summary)
    print("COMPLETE "+json.dumps({s: {k: v for k, v in final[s].items() if k != "actions"} for s in final}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/motion/default.yaml"))
    args = parser.parse_args()
    user = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(user, dict) or set(user)-set(DEFAULTS):
        raise ValueError("Config must contain only known keys")
    train({**DEFAULTS, **user})


if __name__ == "__main__":
    main()
