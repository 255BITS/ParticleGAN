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
from lib.gym_particle_finetune import (EDIT_CAP_EVERY, MODULE_KEYS, REMOVED_L2,
    build_edit_critic, configure_control_scope, controller_objective,
    discriminator_objective, edit_cap, initialize_particle_finetune, normalized_g2_action,
    require_live_adversary)
from particlegan import learning_rate_scale

DEFAULTS = dict(arm="particle", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1", marginal_weight=1.,
    imitation_weight=0., real_encoding_weight=0., synthetic_reconstruction_weight=0.,
    adv_weight=1., train_scope="control", error_tokens=8, error_width=48, error_heads=4,
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
    require_live_adversary(cfg["adv_weight"])
    if cfg["train_scope"] != "control":
        raise ValueError("train_scope stays control: E_control and G2, not the world")
    for key in ("error_tokens", "error_width", "error_heads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["error_width"] % cfg["error_heads"] != 0:
        raise ValueError("error_width must divide across error_heads")
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
        removed_l2=list(REMOVED_L2), l2_aux_weight=0., adv_weight=1.,
        controller_objective="Rp logistic on the edit-normalized G2 residual; no action MSE",
        gradient_penalty=f"sample-point b_cap on the edit critic, autograd L2, coeff 1, kappa 1, every {EDIT_CAP_EVERY} steps",
        train_scope="E_control and G2; G1, G3, E_pair, prior, and transition D frozen",
        control_input="Expert previous command in shuffled records; learner previous command at playback")


def _batched_actions(bundle, columns, chunk=1024):
    """Normalized G2 actions at initialization, in chunks so the norm fit stays small."""
    pieces = []
    for start in range(0, len(columns[0]), chunk):
        stop = start + chunk
        pieces.append(normalized_g2_action(bundle, columns[0][start:stop], columns[1][start:stop],
                                           columns[4][start:stop]))
    return torch.cat(pieces, 0)


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
    configure_control_scope(bundle)
    g, ec = bundle["G"], bundle["E_control"]
    recipe = training_recipe({**world, "steps": cfg["steps"], "batch_size": cfg["batch_size"]})
    with torch.no_grad():
        neutrals = _batched_actions(bundle, columns)
        targets = bundle["scaler"].action(columns[2])
    critic = build_edit_critic(targets, neutrals, cfg).to(device)
    reg = edit_cap()
    adam_kwargs = dict(fused=True) if device.type == "cuda" else {}
    opt_g = torch.optim.Adam([p for p in list(ec.parameters()) + list(g.branches[1].parameters())
                              if p.requires_grad], lr=recipe.lr, betas=recipe.betas, **adam_kwargs)
    opt_r = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad],
                             lr=recipe.lr * recipe.d_lr_mult, betas=recipe.betas, **adam_kwargs)
    optimizers = (opt_g, opt_r)
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    ema = {**bundle}
    for key in ("G", "E", "prior", "E_control"):
        ema[key] = copy.deepcopy(bundle[key]).eval().requires_grad_(False)
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(data=11, d_data=21, edit_d=41, edit_g=42).items()}
    parameter_counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    parameter_counts["R"] = parameter_count(critic)
    trainable_counts = {key: sum(p.numel() for p in bundle[key].parameters() if p.requires_grad)
                        for key in MODULE_KEYS}
    trainable_counts["R"] = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    inference_count = parameter_count(ec) + parameter_count(g.branches[1]) + parameter_count(bundle["prior"])
    groups = dict(controller=dict(lr=recipe.lr, betas=list(recipe.betas), modules=["E_control", "G2"]),
        edit_critic=dict(lr=recipe.lr * recipe.d_lr_mult, betas=list(recipe.betas), modules=["R"]),
        frozen=["G1", "G3", "E", "prior", "D"])
    objective = dict(loss_type="logistic", gan_mode="rp", reg_arm="b_cap", reg_method="autograd",
        reg_every=EDIT_CAP_EVERY, adv_weight=1., train_scope="control", l2_aux_weight=0.,
        critic="gmix_t8_w48_l1", normalization=critic.normalization,
        initialization_recipe=recipe.to_dict())
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", objective)
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
            recipe=objective, scaler=bundle["scaler"].state_dict(), step=step,
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
            f"episodes={len(np.unique(records['episode_ids']))} device={device} adv_weight=1")
        log("PLAYBACK E_control(st, previous at) -> z -> G2. TRAIN E_control and G2 only. "
            "FROZEN G1, G3, E_pair, prior, transition D.")
        log("REMOVED L2: imitation MSE; real reconstruction MSE/BCE; synthetic reconstruction MSE/BCE. "
            "AUX L2 weight=0.")
        log("CONTROLLER STEP: Rp logistic on noise versus noise plus the edit-normalized G2 residual. "
            "sample-point b_cap on that critic every 4 updates. adv_weight=1. Not supervised_only.")
        log(f"LR controller={groups['controller']['lr']} edit_critic={groups['edit_critic']['lr']} "
            f"b_cap coeff={reg.coeff} kappa={reg.kappa} lazy_k={reg.lazy_k}")
        log(f"Trainable parameters={trainable_counts}; inference={inference_count}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        b_cap_applications = 0
        for step in range(1, cfg["steps"] + 1):
            lr_scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            for opt, rates in zip(optimizers, base_rates):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * lr_scale
            d_rows = rows("d_data")
            with torch.no_grad():
                predicted_d = normalized_g2_action(bundle, d_rows[0], d_rows[1], d_rows[4])
            target_d = bundle["scaler"].action(d_rows[2])
            ld, d_terms = discriminator_objective(critic, predicted_d, target_d, step, rng["edit_d"], reg,
                                                  cfg["steps"])
            if not torch.isfinite(ld):
                raise FloatingPointError(f"Nonfinite discriminator loss at step {step}")
            opt_r.zero_grad(set_to_none=True)
            ld.backward()
            opt_r.step()
            if d_terms["b_cap_applied"]:
                b_cap_applications += 1
            g_rows = rows("data")
            predicted = normalized_g2_action(bundle, g_rows[0], g_rows[1], g_rows[4])
            target = bundle["scaler"].action(g_rows[2])
            lg, g_terms = controller_objective(critic, predicted, target, step, rng["edit_g"], cfg["steps"],
                                               cfg["adv_weight"])
            if not torch.isfinite(lg):
                raise FloatingPointError(f"Nonfinite controller loss at step {step}")
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "E_control"):
                    for target_p, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target_p.lerp_(source, 1 - recipe.ema_decay)
                action_mse = torch.nn.functional.mse_loss(predicted.detach(), target.detach())
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(lg.detach()), d_loss=float(ld.detach()),
                    g_loss=float(lg.detach()), prior_loss=0., l2_aux_weight=0., adv_weight=1.,
                    b_cap_applied=d_terms["b_cap_applied"], lr_scale=lr_scale,
                    elapsed_seconds=time.perf_counter() - started, action_mse=float(action_mse),
                    **{key: float(value) for key, value in {**d_terms, **g_terms}.items()
                       if key not in ("b_cap_applied", "adv_weight")})
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} D={row['d_loss']:.5f} "
                    f"G={row['g_loss']:.5f} adv_weight=1 b_cap_applied={int(row['b_cap_applied'])} "
                    f"diag_action_mse={row['action_mse']:.5f} l2_aux=0 elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment
                if any(not torch.isfinite(p).all() for key in ("E_control",) for p in bundle[key].parameters()):
                    raise FloatingPointError(f"Nonfinite parameters at checkpoint {step}")
                if any(not torch.isfinite(p).all() for p in bundle["G"].branches[1].parameters()):
                    raise FloatingPointError(f"Nonfinite G2 at checkpoint {step}")
                save(step)
                log(f"CHECKPOINT step={step}; landing selection is not part of this train step")
                sync()
                segment = time.perf_counter()
        shutil.copyfile(out / f"checkpoint_{cfg['steps']}.pt", out / "final.pt")
        summary = dict(config=cfg, recipe=objective, optimizers=groups, provenance=provenance,
            parameters=parameter_counts, trainable_parameters=trainable_counts,
            total_trainable_parameters=sum(trainable_counts.values()),
            inference_parameters=inference_count, unique_training_records=len(columns[0]),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            real_draws=2 * cfg["steps"] * cfg["batch_size"], simulator_calls=0,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={path.name: sha256(path) for path in sorted(out.glob("*.pt"))},
            removed_l2=list(REMOVED_L2), l2_aux_weight=0., adv_weight=1.,
            b_cap_applications=b_cap_applications,
            selection="Deferred: no landing evaluation has been run for this arm",
            initialization="Same frozen adversarial checkpoint; E_control copied from paired E; scaler retained. "
                           "Controller step is paired-error RpGAN plus sample-point b_cap.")
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
