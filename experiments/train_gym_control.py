#!/usr/bin/env python
"""Matched imitation-only and joint three-generator expert control fine-tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from experiments.train_gym_transition import (discriminator_loss, generator_loss,
    parameter_count, sha256, write_json)
from lib.gym_control import build_expert_records, initialize_control, predict_control
from lib.gym_transition import (contact_record, composed_transition, encoded_transition,
    real_reconstruction, synthetic_reconstruction)
from particlegan import K3PCritic, get_recipe, scale_learning_rates

DEFAULTS = dict(arm="joint", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1", imitation_weight=1.,
    checkpoint="results/gym/lunar_lander/adversarial/best.pt",
    episodes="results/gym/lunar_lander/data/episodes.json",
    out_dir="results/gym/lunar_lander_control/joint",
    live_log="results/gym/lunar_lander_control/live.log")
MODULE_KEYS = ("G", "E", "prior", "D", "E_control")


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] not in ("imitation", "joint"):
        raise ValueError("arm must be imitation or joint")
    for key in ("steps", "batch_size", "log_interval"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"Invalid {key}")
    if type(cfg["seed"]) is not int or cfg["imitation_weight"] != 1.:
        raise ValueError("Integer seed and frozen imitation weight 1 required")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def capture_provenance(out, cfg, records, bundle):
    # Capture actual shared implementation used by this run, without modifying it.
    paths = [Path(__file__), ROOT / "lib/gym_control.py", ROOT / "lib/gym_transition.py",
             ROOT / "experiments/train_gym_transition.py", ROOT / "experiments/config.py"]
    paths += sorted((ROOT / "particlegan").glob("*.py"))
    paths += sorted((ROOT / "configs/gym/lunar_lander_control").glob("*.yaml"))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, digest in sources.items():
            value = (ROOT / name).read_bytes()
            if hashlib.sha256(value).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(name, value)
    arrays = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                       sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in records.items()}
    return dict(sources=sources, source_archive_sha256=sha256(out / "source.zip"),
        initial_checkpoint=dict(path=cfg["checkpoint"], sha256=sha256(cfg["checkpoint"]), step=bundle["step"]),
        episodes=dict(path=cfg["episodes"], sha256=sha256(cfg["episodes"])),
        expert_data=dict(split="train", behavior="heuristic", count=len(records["states"]),
            episode_ids=np.unique(records["episode_ids"]).tolist(), arrays=arrays,
            npz_sha256=sha256(out / "expert_records.npz")),
        initial_provenance=bundle["provenance"],
        normalization="Unchanged scaler from initialization; fit originally on old training split")


def train(cfg):
    validate(cfg)
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    live = Path(cfg["live_log"])
    live.parent.mkdir(parents=True, exist_ok=True)
    bundle = initialize_control(cfg["checkpoint"], cfg["arm"], device)
    world = bundle["world_config"]
    expected_episodes = bundle["provenance"].get("dataset", {}).get("episodes.json")
    if expected_episodes is not None and sha256(cfg["episodes"]) != expected_episodes:
        raise ValueError("Expert episode source differs from initialization provenance")
    records = build_expert_records(cfg["episodes"])
    np.savez_compressed(out / "expert_records.npz", **records)
    provenance = capture_provenance(out, cfg, records, bundle)
    physical = torch.as_tensor(np.concatenate([records[k] for k in ("states", "actions", "next_states")], 1),
                               device=device)
    terrain = torch.as_tensor(records["terrain"], device=device)
    previous = torch.as_tensor(records["previous_actions"], device=device)
    scaler = bundle["scaler"]
    normalized = scaler(physical)
    g, e, prior, d, ec = [bundle[k] for k in MODULE_KEYS]
    recipe = get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=world["z_dim"], num_particles=world["num_particles"],
                        total_steps=cfg["steps"], batch_size=cfg["batch_size"])
    if cfg["arm"] == "joint":
        opt_g, opt_d = recipe.make_optimizers(g, d, prior,
            encoder=torch.nn.ModuleList([e, ec]), fused=device.type == "cuda")
    else:
        opt_g = torch.optim.Adam(list(ec.parameters()) + list(g.branches[1].parameters()),
            lr=recipe.lr, betas=recipe.betas, fused=device.type == "cuda")
        opt_d = None
    optimizers = [opt_g] + ([] if opt_d is None else [opt_d])
    base_rates = [[p["lr"] for p in opt.param_groups] for opt in optimizers]
    ema = {**bundle}
    for key in ("G", "E", "prior", "E_control"):
        ema[key] = copy.deepcopy(bundle[key]).eval().requires_grad_(False)
    reg = K3PCritic(recipe, d, opt_d) if opt_d is not None else None
    gan, spread = recipe.make_loss(), recipe.make_prior_regularizer()
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(data=11, d_data=21, latent=12, contact=31, d_latent=22, d_contact=32).items()}
    reg_rngs = {role: torch.Generator(device=device).manual_seed(cfg["seed"] + 40 + i)
                for i, role in enumerate(d.roles())}
    weights = dict(continuous_weight=world["continuous_weight"], contact_weight=world["contact_weight"])
    parameter_counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    trainable_counts = {key: sum(p.numel() for p in bundle[key].parameters() if p.requires_grad) for key in MODULE_KEYS}
    inference_count = parameter_count(ec) + parameter_count(g.branches[1]) + parameter_count(prior)
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    write_json(out / "normalization.json", {k: v.cpu().tolist() for k,v in scaler.state_dict().items()})
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({s for s in cfg["checkpoints"] if s <= cfg["steps"]} | {cfg["steps"]})
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def batch(name):
        return torch.randint(len(physical), (cfg["batch_size"],), device=device, generator=rng[name])
    def save(step):
        saved = dict(format="gym_control_v1", config=cfg, world_config=world, recipe=recipe.to_dict(),
            scaler=scaler.state_dict(), step=step, provenance=provenance,
            validation=dict(status="Awaiting independent control rollout evaluation"))
        saved.update({key: ema[key].state_dict() for key in MODULE_KEYS})
        torch.save(saved, out / f"checkpoint_{step}.pt")
    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[{out.name}] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")
        log(f"START arm={cfg['arm']} steps={cfg['steps']} expert_records={len(physical)} "
            f"episodes={len(np.unique(records['episode_ids']))} device={device}")
        log("G1 -> st; G2 -> at; G3 -> st+1; E_control(st, previous at) -> z -> G2; E_pair(st, current at) -> z")
        log(f"Trainable parameters={trainable_counts}; inference={inference_count}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg["steps"] + 1):
            lr_scale, _ = scale_learning_rates(step - 1, recipe, optimizers, base_rates, prior)
            ld = lg = lp = le = physical.new_zeros(())
            if opt_d is not None:
                d.requires_grad_(True)
                ids = batch("d_data")
                real_d, ctx_d = normalized[ids], terrain[ids]
                with torch.no_grad():
                    fake_d = contact_record(g(prior.sample(len(ids), rng["d_latent"])[0], ctx_d), rng=rng["d_contact"])
                    composed_d = composed_transition(e, g, prior, fake_d, ctx_d, rng=rng["d_contact"])[0]
                    half = len(ids) // 2
                    fake_d = torch.cat([fake_d[:half], composed_d[half:]])
                ld, _ = discriminator_loss(d, real_d, fake_d, ctx_d, gan, reg, step, reg_rngs)
                if not torch.isfinite(ld):
                    raise FloatingPointError(f"Nonfinite discriminator loss at step {step}")
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                reg.step()  # spike guard, Adam step, K3P anchor/LR record
                d.requires_grad_(False)
            ids = batch("data")
            real, ctx = normalized[ids], terrain[ids]
            action, _ = predict_control(bundle, physical[ids, :8], previous[ids], ctx)
            li = torch.nn.functional.mse_loss(scaler.action(action), real[:, 8:10])
            if opt_d is not None:
                z, _ = prior.sample(len(ids), rng["latent"])
                fake = contact_record(g(z, ctx), rng=rng["contact"], straight_through=True)
                composed, decoded_fake, _ = composed_transition(e, g, prior, fake, ctx,
                    rng=rng["contact"], straight_through=True)
                decoded_real, _ = encoded_transition(e, g, prior, real[:, :10], ctx)
                lr, _ = real_reconstruction(decoded_real, real, **weights)
                ls, _ = synthetic_reconstruction(decoded_fake, fake, **weights)
                le = world["real_encoding_weight"] * lr + world["synthetic_reconstruction_weight"] * ls
                lp = spread(prior.z)
                original_loss, _ = generator_loss(d, real, fake, ctx, gan, world["marginal_weight"])
                composed_loss, _ = generator_loss(d, real, composed, ctx, gan, world["marginal_weight"])
                lg = (original_loss + composed_loss) / 2
            loss = cfg["imitation_weight"] * li + le + lg + lp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite generator/control loss at step {step}")
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "E_control"):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), imitation_loss=float(li.detach()),
                    d_loss=float(ld.detach()), g_loss=float(lg.detach()), prior_loss=float(lp.detach()),
                    reconstruction_loss=float(le.detach()), lr_scale=lr_scale,
                    elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} imitation={row['imitation_loss']:.5f} "
                    f"D={row['d_loss']:.5f} G={row['g_loss']:.5f} reconstruction={row['reconstruction_loss']:.5f} "
                    f"elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment
                if any(not torch.isfinite(p).all() for key in MODULE_KEYS for p in bundle[key].parameters()):
                    raise FloatingPointError(f"Nonfinite parameters at checkpoint {step}")
                save(step)
                log(f"CHECKPOINT step={step}; selection deferred to control validation")
                sync()
                segment = time.perf_counter()
        shutil.copyfile(out / f"checkpoint_{cfg['steps']}.pt", out / "final.pt")
        summary = dict(config=cfg, recipe=recipe.to_dict(), provenance=provenance,
            parameters=parameter_counts, full_parameters=sum(parameter_counts.values()),
            trainable_parameters=trainable_counts, total_trainable_parameters=sum(trainable_counts.values()),
            inference_parameters=inference_count, unique_training_records=len(physical),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"] if opt_d is not None else 0,
            real_draws=cfg["steps"] * cfg["batch_size"] * (2 if opt_d is not None else 1),
            simulator_calls=0, train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={p.name: sha256(p) for p in sorted(out.glob("*.pt"))},
            selection="Deferred: highest validation landing fraction, tie-break mean return",
            synthetic_semantics="Joint only: detached sampled target, live E_pair synthetic input; unchanged from initialization trainer",
            control_input="Expert previous command in finite shuffled records; learner previous command during rollout",
            initialization="Same frozen checkpoint; E_control copied from paired E; original scaler retained")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f}; awaiting rollout selection")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=("imitation", "joint"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ("arm", "steps", "device", "out_dir", "checkpoint"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
