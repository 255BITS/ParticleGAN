#!/usr/bin/env python
"""Leaderboard artifact run of the transition GAN problem (examples/transition_gan.py).

Training is the shared runner (``benchmarks.toy_runner.ToyRun``) on the problem's
recipe; this module only adds the benchmark's reproducibility artifacts:
provenance and source archive, normalization/prior metadata, per-interval JSON
metrics, the fixed-context EMA evaluation (``evaluate``, pinned by the
leaderboard), encoder inference, the viewer and the checkpoint.

    python -u experiments/train_transition.py --device cpu --out-dir results/transition/run
"""
import argparse
import hashlib
import json
import math
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
from benchmarks.toy_runner import ToyRun
from examples.transition_gan import PROBLEM_DEFAULTS, TransitionGAN, transition_recipe
from experiments.config import read_config
from experiments.run_grid import code_provenance
from lib.transition import encoded_transition, composed_transition, metrics, shuffle_blocks
from lib.transition_visuals import render


DEFAULTS = dict(PROBLEM_DEFAULTS, seed=24002, device="cuda:0", log_interval=250,
                out_dir="results/transition/default", live_log="results/transition/live.log",
                save_checkpoint=True)


def training_recipe(cfg):
    return transition_recipe(cfg)


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unknown or missing keys: {set(cfg) ^ set(DEFAULTS)}")
    for key in ("width", "d_width", "marginal_width", "num_particles", "z_dim", "length", "steps", "batch_size",
                "log_interval", "eval_per_context", "normalization_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if min(cfg["length"], cfg["num_particles"], cfg["batch_size"], cfg["eval_per_context"], cfg["normalization_samples"]) < 2:
        raise ValueError("length and sample counts must be >= 2")
    if cfg["architecture"] not in ("branches", "monolithic"):
        raise ValueError("architecture must be branches or monolithic")
    if cfg["critic_mode"] not in ("joint", "joint_marginals"):
        raise ValueError("critic_mode must be joint or joint_marginals")
    if cfg["d_conditioning"] not in ("ucd", "concat"):
        raise ValueError("d_conditioning must be ucd or concat")
    if not math.isfinite(cfg["g_class_scale"]) or cfg["g_class_scale"] <= 0:
        raise ValueError("g_class_scale must be finite and positive")
    if not math.isfinite(cfg["g_context_scale"]) or cfg["g_context_scale"] <= 0:
        raise ValueError("g_context_scale must be finite and positive")
    if not math.isfinite(cfg["marginal_weight"]) or cfg["marginal_weight"] <= 0:
        raise ValueError("marginal_weight must be finite and positive")
    if cfg["geometry_mode"] not in ("discrete", "continuous"):
        raise ValueError("geometry_mode must be discrete or continuous")
    if type(cfg["seed"]) is not int or type(cfg["save_checkpoint"]) is not bool:
        raise ValueError("invalid seed/save_checkpoint")
    for key in ("out_dir", "live_log", "device"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("encoder", "shared_state_critic"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be bool")
    if type(cfg["encoder_width"]) is not int or cfg["encoder_width"] < 1:
        raise ValueError("encoder_width must be positive integer")
    for key in ("real_encoding_weight", "synthetic_reconstruction_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive finite")
    if cfg["shared_state_critic"] and cfg["critic_mode"] != "joint_marginals":
        raise ValueError("shared state critic requires marginal critics")
    training_recipe(cfg)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


@torch.no_grad()
def evaluate(toy, g, prior, scaler, count, out, critic_mode="joint"):
    final, floors, controls, samples = {}, {}, {}, {}
    # Fixed, distinct contexts include both endpoints, rising and falling motion.
    ticks = sorted(set(round(f*(toy.length-2)) for f in (0, .25, .5, .75, 1)))
    for split in ("train", "test"):
        cc, gg = toy.contexts(split)
        c0, geom0 = cc.repeat_interleave(len(ticks)), gg.repeat_interleave(len(ticks), 0)
        t0 = torch.tensor(ticks, device=toy.device).repeat(len(cc))
        groups = torch.arange(len(c0), device=toy.device).repeat_interleave(count)
        c, geom, tick = c0[groups], geom0[groups], t0[groups]
        context = toy.condition(geom, tick)
        rngs = [torch.Generator(device=toy.device).manual_seed(99000+i) for i in range(4)]
        chunks = []
        for j in range(0, len(c), 256):
            z, _ = prior.sample(len(c[j:j+256]), rngs[0])
            chunks.append(scaler.inverse(g(z, c[j:j+256], context[j:j+256])))
        x = torch.cat(chunks)
        if not torch.isfinite(x).all():
            raise FloatingPointError("nonfinite transitions")
        real = toy.sample(c, geom, tick, rngs[1])
        real2 = toy.sample(c, geom, tick, rngs[2])
        shuffled_real = shuffle_blocks(real2, groups, rngs[3])
        shuffled_fake = shuffle_blocks(x, groups, rngs[3])
        subsets = {split: torch.ones_like(c, dtype=torch.bool)}
        if split == "test":
            subsets.update(interpolation=groups < 6*len(ticks), extrapolation=groups >= 6*len(ticks))
        for name, mask in subsets.items():
            def measure(candidate):
                return metrics(candidate[mask], real[mask], scaler, groups[mask])
            final[name], floors[name] = measure(x), measure(real2)
            controls[name] = dict(shuffled_real=measure(shuffled_real), shuffled_generated=measure(shuffled_fake))
        samples[split] = dict(x=x.cpu().numpy(), real=real.cpu().numpy(),
                              c=c.cpu().numpy(), geom=geom.cpu().numpy(), tick=tick.cpu().numpy(),
                              group=groups.cpu().numpy())
        np.savez_compressed(out / f"{split}_samples.npz", **samples[split])
    render(out, samples, toy.length, g.architecture, critic_mode)
    return final, floors, controls


@torch.no_grad()
def evaluate_encoder(toy, e, g, prior, scaler, out):
    """Paired prediction and synthetic composition on the exact saved benchmark inputs."""
    result = {}
    for split in ("train", "test"):
        saved = np.load(out/f"{split}_samples.npz")
        arrays = {k: torch.as_tensor(saved[k], device=toy.device) for k in saved.files}
        real, fake = scaler(arrays["real"]), scaler(arrays["x"])
        c, groups = arrays["c"], arrays["group"]
        context = toy.condition(arrays["geom"], arrays["tick"])
        predicted, reconstructed, composed, real_ids, synthetic_ids = [], [], [], [], []
        for start in range(0, len(c), 256):
            sl = slice(start, start+256)
            decoded, enc = encoded_transition(e, g, prior, real[sl, :4], c[sl], context[sl])
            syn, _, senc = composed_transition(e, g, prior, fake[sl], c[sl], context[sl])
            predicted.append(scaler.inverse(torch.cat([real[sl, :4], decoded[:, 4:]], 1)))
            reconstructed.append(scaler.inverse(decoded))
            composed.append(scaler.inverse(syn))
            real_ids.append(enc.indices[:, 0]); synthetic_ids.append(senc.indices[:, 0])
        prediction, reconstruction, synthetic = map(torch.cat, (predicted, reconstructed, composed))
        ri, si = torch.cat(real_ids), torch.cat(synthetic_ids)
        masks = {split: torch.ones_like(c, dtype=torch.bool)}
        if split == "test":
            boundary = 6*len(torch.unique(arrays["tick"]))
            masks.update(interpolation=groups < boundary, extrapolation=groups >= boundary)
        def usage(ids):
            counts = torch.bincount(ids, minlength=prior.num_particles).float()
            p = counts[counts > 0]/len(ids)
            return dict(used=int((counts > 0).sum()), effective=float((-p*p.log()).sum().exp()),
                        max_fraction=float(p.max()))
        def paired(mask):
            errors = (prediction[mask, 4:]-arrays["real"][mask, 4:]).norm(dim=1)
            rec = reconstruction[mask]-arrays["real"][mask]
            return dict(next_l2=float(errors.mean()), next_l2_p95=float(errors.quantile(.95)),
                        state_l2=float(rec[:, :2].norm(dim=1).mean()),
                        action_l2=float(rec[:, 2:4].norm(dim=1).mean()),
                        real_routing=usage(ri[mask]), synthetic_routing=usage(si[mask]))
        for name, mask in masks.items():
            result[name] = dict(paired=paired(mask),
                classes={str(i): paired(mask & (c == i)) for i in (0, 1)},
                conditional=metrics(prediction[mask], arrays["real"][mask], scaler, groups[mask]),
                synthetic=metrics(synthetic[mask], arrays["real"][mask], scaler, groups[mask]))
        np.savez_compressed(out/f"{split}_inference.npz", prediction=prediction.cpu().numpy(),
                            reconstruction=reconstruction.cpu().numpy(), synthetic=synthetic.cpu().numpy(),
                            real_ids=ri.cpu().numpy(), synthetic_ids=si.cpu().numpy())
    write_json(out/"inference.json", result)
    return result


def train(cfg):
    validate(cfg)
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    problem = TransitionGAN(device=device, **{k: cfg[k] for k in PROBLEM_DEFAULTS})
    recipe = problem.recipe()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("metrics.jsonl", "summary.json", "log.txt")):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    live = Path(cfg["live_log"])
    live.parent.mkdir(parents=True, exist_ok=True)
    provenance = code_provenance(__file__, sys.executable)
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, digest in provenance["sources"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(name, data)
    env = dict(device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    write_json(out / "environment.json", env)
    run = ToyRun(problem, recipe=recipe, seed=cfg["seed"], device=device)
    toy, scaler = problem.data(device)
    g, prior, e, d = run.nets.generator, run.nets.prior, run.nets.encoder, problem.layout
    write_json(out / "normalization.json", dict(mean=scaler.mean.tolist(), scale=scaler.scale.tolist(),
                                               split="train", seed=91001, count=cfg["normalization_samples"]))
    write_json(out / "prior.json", dict(kind="mog", num_particles=prior.num_particles,
                                       sigma_rel=prior.sigma_rel, sigma=float(prior.sigma),
                                       initial_neighbor_distance=float(prior.d0), standardize=prior.standardize,
                                       regularize="raw centers; full table at <=1024"))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile:
        def log(message):
            message = f"[{out.name}] {message}"
            print(message, flush=True)
            logfile.write(message+"\n")
            livefile.write(message+"\n")

        log(f"START architecture={cfg['architecture']} device={device} steps={cfg['steps']} seed={cfg['seed']} "
            f"critics={list(run.critics)} encoder={e is not None}; shared runner on the recipe")
        sync()
        started = time.perf_counter()
        with (out / "metrics.jsonl").open("w", buffering=1) as metric_log:
            for step in range(1, cfg["steps"]+1):
                losses = run.step()
                if step == 1 or step % cfg["log_interval"] == 0 or step == cfg["steps"]:
                    sync()
                    row = dict(step=step, lr=run.opt_g.param_groups[0]["lr"],
                               train_seconds=time.perf_counter()-started,
                               **{k: float(v) for k, v in losses.items() if k != "step"})
                    metric_log.write(json.dumps(row, allow_nan=False)+"\n")
                    log(json.dumps(row, allow_nan=False))
        sync()
        train_seconds = time.perf_counter()-started
        ema = run.ema_nets
        log("Evaluating EMA: held-out contexts, reference floors and shuffled-branch controls")
        final, floors, controls = evaluate(toy, ema.generator, ema.prior, scaler, cfg["eval_per_context"], out,
                                           cfg["critic_mode"])
        inference = evaluate_encoder(toy, ema.encoder, ema.generator, ema.prior, scaler, out) if e is not None else None
        if cfg["save_checkpoint"]:
            torch.save(dict(config=cfg, recipe=recipe.to_dict(), G=ema.generator.state_dict(),
                            prior=ema.prior.state_dict(), scaler=scaler.state_dict(),
                            E=ema.encoder.state_dict() if e is not None else None, D=d.state_dict()), out / "final.pt")
        summary = dict(inference=inference, config=cfg, recipe=recipe.to_dict(), final=final, reference_floor=floors,
                       controls=controls, train_seconds=train_seconds,
                       total_seconds=time.perf_counter()-started, real_draws=2*cfg["steps"]*cfg["batch_size"],
                       normalization_draws=cfg["normalization_samples"], environment=env, provenance=provenance,
                       verdict=problem.verdict(final["test"]),
                       critic_parameters={name: sum(p.numel() for p in critic.parameters())
                                          for name, critic in d.critics.items()},
                       parameters={name: sum(p.numel() for p in module.parameters())
                                   for name, module in ([("G", g), ("D", d), ("prior", prior)] + ([("E", e)] if e is not None else []))})
        write_json(out / "summary.json", summary)
        log("COMPLETE "+json.dumps({"verdict": summary["verdict"],
                                    **{k: v for k, v in final["test"].items() if k != "contexts"}}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/transition/default.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config(args.config)}
    for key in ("steps", "device", "out_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
