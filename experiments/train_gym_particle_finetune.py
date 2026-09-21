#!/usr/bin/env python
"""Fine-tune Lunar Lander control with ParticleGAN losses instead of paired L2."""
import argparse
import copy
import hashlib
import json
import os
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
from experiments.train_gym_transition import parameter_count, sha256, training_recipe, write_json
from lib.gym_control import build_expert_records
from lib.gym_particle_finetune import (FAKE_PATHS, MODULE_KEYS, REMOVED_L2,
    diagnostic_l2, initialize_particle_finetune, particle_game, require_classic_particle_gan,
    transition_batch)
from particlegan import learning_rate_scale

DEFAULTS = dict(arm="particle", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1", marginal_weight=1.,
    imitation_weight=0., real_encoding_weight=0., synthetic_reconstruction_weight=0.,
    checkpoint="results/gym/lunar_lander/adversarial/best.pt",
    episodes="results/gym/lunar_lander/data/episodes.json",
    out_dir="results/gym/lunar_lander_particle_finetune/particle",
    live_log="results/gym/lunar_lander_particle_finetune/live.log")
L2_KEYS = ("imitation_weight", "real_encoding_weight", "synthetic_reconstruction_weight")


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] != "particle":
        raise ValueError("arm must be particle")
    for key in ("steps", "batch_size", "log_interval"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"Invalid {key}")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key in L2_KEYS:
        if cfg[key] != 0:
            raise ValueError(f"{key} is removed; Arm A does not keep an auxiliary L2 term")
    if cfg["marginal_weight"] != 1.:
        raise ValueError("marginal_weight stays 1")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    for key in ("device", "checkpoint", "episodes", "out_dir", "live_log"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def capture_provenance(out, cfg, records, bundle):
    paths = [Path(__file__), ROOT / "lib/gym_particle_finetune.py", ROOT / "lib/gym_control.py",
             ROOT / "lib/gym_previous_gan.py", ROOT / "lib/gym_transition.py",
             ROOT / "experiments/train_gym_transition.py", ROOT / "experiments/config.py"]
    paths += sorted((ROOT / "particlegan").glob("*.py"))
    paths += sorted((ROOT / "configs/gym/lunar_lander_particle_finetune").glob("*.yaml"))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, digest in sources.items():
            value = (ROOT / name).read_bytes()
            if hashlib.sha256(value).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(name, value)
    arrays = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                       sha256=hashlib.sha256(value.tobytes()).hexdigest())
              for key, value in records.items()}
    return dict(sources=sources, source_archive_sha256=sha256(out / "source.zip"),
        initial_checkpoint=dict(path=cfg["checkpoint"], sha256=sha256(cfg["checkpoint"]), step=bundle["step"]),
        episodes=dict(path=cfg["episodes"], sha256=sha256(cfg["episodes"])),
        expert_data=dict(split="train", behavior="heuristic", count=len(records["states"]),
            episode_ids=np.unique(records["episode_ids"]).tolist(), arrays=arrays,
            npz_sha256=sha256(out / "expert_records.npz")),
        initial_provenance=bundle["provenance"],
        normalization="Unchanged scaler from initialization; fit originally on old training split",
        removed_l2=list(REMOVED_L2), l2_aux_weight=0., fake_paths=list(FAKE_PATHS),
        critics=["joint", "action", "shared state for current and next"],
        gan="Rp logistic GANLoss",
        gradient_penalty="sample-point b_cap, autograd L2, coeff 1, kappa 1, every step",
        control_input="Expert previous command in shuffled records; learner previous command at playback")


def link_live_log(out, live):
    """Point out_dir/live.log at the shared flushed log so either path can be tailed."""
    live.parent.mkdir(parents=True, exist_ok=True)
    live.touch(exist_ok=True)
    alias = out / "live.log"
    if alias.resolve() != live.resolve():
        alias.symlink_to(os.path.relpath(live, out))


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
    link_live_log(out, live)
    bundle = initialize_particle_finetune(cfg["checkpoint"], device)
    world = bundle["world_config"]
    expected_episodes = bundle["provenance"].get("dataset", {}).get("episodes.json")
    if expected_episodes is not None and sha256(cfg["episodes"]) != expected_episodes:
        raise ValueError("Expert episode source differs from initialization provenance")
    records = build_expert_records(cfg["episodes"])
    np.savez_compressed(out / "expert_records.npz", **records)
    provenance = capture_provenance(out, cfg, records, bundle)
    columns = [torch.as_tensor(records[key], device=device) for key in
               ("states", "previous_actions", "actions", "next_states", "terrain")]
    g, e, prior, d, ec = [bundle[key] for key in MODULE_KEYS]
    recipe = training_recipe({**world, "steps": cfg["steps"], "batch_size": cfg["batch_size"]})
    gan, reg, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    require_classic_particle_gan(gan, reg)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, encoder=torch.nn.ModuleList([e, ec]),
                                          fused=device.type == "cuda")
    optimizers = (opt_g, opt_d)
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    ema = {**bundle}
    for key in ("G", "E", "prior", "E_control"):
        ema[key] = copy.deepcopy(bundle[key]).eval().requires_grad_(False)
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(data=11, d_data=21, latent=12, contact=31, d_latent=22, d_contact=32).items()}
    reg_rngs = {role: torch.Generator(device=device).manual_seed(cfg["seed"] + 40 + i)
                for i, role in enumerate(d.roles())}
    parameter_counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    trainable_counts = {key: sum(p.numel() for p in bundle[key].parameters() if p.requires_grad)
                        for key in MODULE_KEYS}
    inference_count = parameter_count(ec) + parameter_count(g.branches[1]) + parameter_count(prior)
    groups = dict(g_encoder=dict(lr=recipe.lr, betas=list(recipe.betas), modules=["G", "E", "E_control"]),
        prior=dict(lr=recipe.lr * recipe.prior_lr_mult,
                   betas=list(recipe.prior_betas if recipe.prior_betas is not None else recipe.betas),
                   modules=["prior"]),
        discriminator=dict(lr=recipe.lr * recipe.d_lr_mult, betas=list(recipe.betas),
                           modules=["D.joint", "D.action", "D.state"]))
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    write_json(out / "optimizers.json", groups)
    write_json(out / "normalization.json", {k: v.cpu().tolist() for k, v in bundle["scaler"].state_dict().items()})
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({step for step in cfg["checkpoints"] if step <= cfg["steps"]} | {cfg["steps"]})

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def rows(name):
        ids = torch.randint(len(columns[0]), (cfg["batch_size"],), device=device, generator=rng[name])
        return [column[ids] for column in columns]

    def save(step):
        saved = dict(format="gym_particle_finetune_v1", config=cfg, world_config=world,
            recipe=recipe.to_dict(), scaler=bundle["scaler"].state_dict(), step=step,
            provenance=provenance, validation=dict(status="Awaiting independent control rollout evaluation"))
        saved.update({key: ema[key].state_dict() for key in MODULE_KEYS})
        torch.save(saved, out / f"checkpoint_{step}.pt")

    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[{out.name}] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")

        log(f"START arm=particle steps={cfg['steps']} expert_records={len(columns[0])} "
            f"episodes={len(np.unique(records['episode_ids']))} device={device}")
        log("G1 -> st; G2 -> at; G3 -> st+1; E_control(st, previous at) -> z -> G; "
            "E_pair(st, current at) -> z -> G; prior -> z -> G")
        log("REMOVED L2: imitation MSE; real reconstruction MSE/BCE; synthetic reconstruction MSE/BCE. "
            "AUX L2 weight=0.")
        log("KEPT: Rp logistic GANLoss; sample-point b_cap on joint, action, and shared state D; "
            "MoG prior variance/covariance regularizer. No slider critic.")
        log(f"LR groups G+E+E_control={groups['g_encoder']['lr']} prior={groups['prior']['lr']} "
            f"D={groups['discriminator']['lr']} b_cap coeff={reg.coeff} kappa={reg.kappa}")
        log(f"Trainable parameters={trainable_counts}; inference={inference_count}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg["steps"] + 1):
            lr_scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            for opt, rates in zip(optimizers, base_rates):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * lr_scale
            d.requires_grad_(True)
            d_rows = rows("d_data")
            with torch.no_grad():
                real_d, fakes_d, _ = transition_batch(bundle, *d_rows, rng["d_latent"], rng["d_contact"])
            ld, d_terms = particle_game(d, real_d, fakes_d, d_rows[-1], gan, reg=reg, step=step, rngs=reg_rngs)
            if not torch.isfinite(ld):
                raise FloatingPointError(f"Nonfinite discriminator loss at step {step}")
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            d.requires_grad_(False)
            g_rows = rows("data")
            real, fakes, decoded_control = transition_batch(
                bundle, *g_rows, rng["latent"], rng["contact"], straight_through=True)
            lg, g_terms = particle_game(d, real, fakes, g_rows[-1], gan, marginal_weight=cfg["marginal_weight"])
            lp = spread(prior.z)
            diag = diagnostic_l2(decoded_control, real)
            loss = lg + lp
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite generator loss at step {step}")
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "E_control"):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), d_loss=float(ld.detach()),
                    g_loss=float(lg.detach()), prior_loss=float(lp.detach()), l2_aux_weight=0.,
                    lr_scale=lr_scale, elapsed_seconds=time.perf_counter() - started,
                    **{f"d_{key}": float(value.detach()) for key, value in d_terms.items()},
                    **{f"g_{key}": float(value.detach()) for key, value in g_terms.items()},
                    **{key: float(value.detach()) for key, value in diag.items()})
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} D={row['d_loss']:.5f} "
                    f"G={row['g_loss']:.5f} prior={row['prior_loss']:.5f} "
                    f"diag_action_mse={row['action_mse']:.5f} l2_aux=0 elapsed_s={row['elapsed_seconds']:.1f}")
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
        summary = dict(config=cfg, recipe=recipe.to_dict(), optimizers=groups, provenance=provenance,
            parameters=parameter_counts, trainable_parameters=trainable_counts,
            total_trainable_parameters=sum(trainable_counts.values()),
            inference_parameters=inference_count, unique_training_records=len(columns[0]),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            real_draws=2 * cfg["steps"] * cfg["batch_size"], simulator_calls=0,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={path.name: sha256(path) for path in sorted(out.glob("*.pt"))},
            removed_l2=list(REMOVED_L2), l2_aux_weight=0., fake_paths=list(FAKE_PATHS),
            selection="Deferred: no landing evaluation has been run for this arm",
            initialization="Same frozen adversarial checkpoint; E_control copied from paired E; scaler retained")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f}; awaiting rollout selection")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ("steps", "device", "out_dir", "checkpoint"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
