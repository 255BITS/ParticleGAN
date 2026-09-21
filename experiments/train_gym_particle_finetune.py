#!/usr/bin/env python
"""Fine-tune Lunar Lander G2 with the model-glue paired continuation.

RpGAN and sample-point b_cap stay configured and do not update the controller.
Only the action head trains, with a reduced paired anchor and a kinematic
functional match. This does not score landings.
"""
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
    glue_control_loss, initialize_particle_finetune, require_classic_particle_gan)
from lib.model_glue_control import (ADV_WEIGHT, ANCHOR_WEIGHT, EMA_DECAY, FUNCTIONAL_WEIGHT,
    HEAD_LR, MAX_GRAD_NORM, TRAINABLE_PARTS, BETAS)
from particlegan import learning_rate_scale

DEFAULTS = dict(arm="particle", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1", marginal_weight=1.,
    imitation_weight=0., real_encoding_weight=0., synthetic_reconstruction_weight=0.,
    action_anchor_weight=ANCHOR_WEIGHT, functional_weight=FUNCTIONAL_WEIGHT,
    adv_weight=ADV_WEIGHT, head_lr=HEAD_LR, ema_decay=EMA_DECAY,
    trainable_parts=TRAINABLE_PARTS,
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
    locked = dict(action_anchor_weight=ANCHOR_WEIGHT, functional_weight=FUNCTIONAL_WEIGHT,
                  adv_weight=ADV_WEIGHT, head_lr=HEAD_LR, ema_decay=EMA_DECAY,
                  trainable_parts=TRAINABLE_PARTS)
    for key, value in locked.items():
        if cfg[key] != value:
            raise ValueError(f"{key} must stay {value} for the model-glue continuation")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    for key in ("device", "checkpoint", "episodes", "out_dir", "live_log"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def capture_provenance(out, cfg, records, bundle):
    paths = [Path(__file__), ROOT / "lib/gym_particle_finetune.py", ROOT / "lib/model_glue_control.py",
             ROOT / "lib/gym_control.py", ROOT / "lib/gym_previous_gan.py", ROOT / "lib/gym_transition.py",
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
        gan="Rp logistic GANLoss configured, adv_weight 0, not applied",
        gradient_penalty="sample-point b_cap configured, supervised_only, not applied",
        continuation="model-glue head: G2 only, action anchor 0.1, functional kinematic match, EMA 0.98",
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
    g, _, prior, _, ec = [bundle[key] for key in MODULE_KEYS]
    recipe = training_recipe({**world, "steps": cfg["steps"], "batch_size": cfg["batch_size"]})
    gan, reg = recipe.make_loss(), recipe.make_gradient_penalty()
    require_classic_particle_gan(gan, reg)
    head = g.branches[1]
    opt = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"], betas=BETAS, fused=device.type == "cuda")
    ema = {**bundle}
    for key in ("G", "E", "prior", "E_control"):
        ema[key] = copy.deepcopy(bundle[key]).eval().requires_grad_(False)
    data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    parameter_counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    trainable_counts = {key: sum(p.numel() for p in bundle[key].parameters() if p.requires_grad)
                        for key in MODULE_KEYS}
    if set(trainable_counts) != set(MODULE_KEYS) or trainable_counts["G"] != parameter_count(head):
        raise RuntimeError(f"Only G2 should train, got {trainable_counts}")
    inference_count = parameter_count(ec) + parameter_count(head) + parameter_count(prior)
    groups = dict(action_head=dict(lr=cfg["head_lr"], betas=list(BETAS), modules=["G2"],
                                   max_grad_norm=MAX_GRAD_NORM, ema_decay=cfg["ema_decay"]),
                  frozen=dict(lr=0., modules=["G1", "G3", "E", "E_control", "prior", "D"]),
                  configured_inactive=dict(adv_weight=cfg["adv_weight"], gan=recipe.gan_mode,
                                           reg_arm=recipe.reg_arm, reg_coeff=reg.coeff, reg_kappa=reg.kappa,
                                           note="RpGAN and sample-point b_cap are not applied"))
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

    def rows():
        ids = torch.randint(len(columns[0]), (cfg["batch_size"],), device=device, generator=data_rng)
        return [column[ids] for column in columns]

    def proxy_action_mse():
        view = {**bundle, **{key: ema[key] for key in ("G", "E", "prior", "E_control")}}
        n = min(256, len(columns[0]))
        with torch.no_grad():
            _, _, diag = glue_control_loss(view, *[column[:n] for column in columns])
        return float(diag["action_mse"])

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
        log("PLAYBACK E_control(st, previous at) -> z -> G2. Trainable=G2. "
            "Frozen=G1, G3, E_pair, E_control, prior, D.")
        log("REMOVED L2: imitation MSE; real reconstruction MSE/BCE; synthetic reconstruction MSE/BCE. "
            "AUX L2 weight=0.")
        log("MODEL-GLUE continuation: action_anchor="
            f"{cfg['action_anchor_weight']} functional={cfg['functional_weight']} "
            f"adv_weight={cfg['adv_weight']} head_lr={cfg['head_lr']} ema={cfg['ema_decay']} "
            "supervised_only=true. No slider critic.")
        log("CONFIGURED inactive: Rp logistic GANLoss; sample-point b_cap "
            f"coeff={reg.coeff} kappa={reg.kappa}; MoG prior regularizer. Not applied.")
        log(f"Trainable parameters={trainable_counts}; inference={inference_count}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        best = None
        for step in range(1, cfg["steps"] + 1):
            lr_scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            opt.param_groups[0]["lr"] = cfg["head_lr"] * lr_scale
            batch = rows()
            loss, terms, diag = glue_control_loss(bundle, *batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite control loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_GRAD_NORM)
            opt.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "E_control"):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - cfg["ema_decay"])
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), l2_aux_weight=0.,
                    adv_weight=cfg["adv_weight"], action_anchor=float(terms["action_anchor"].detach()),
                    functional=float(terms["functional"].detach()),
                    lr_scale=lr_scale, elapsed_seconds=time.perf_counter() - started,
                    **{key: float(value.detach()) for key, value in diag.items()})
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} "
                    f"anchor={row['action_anchor']:.5f} functional={row['functional']:.5f} "
                    f"diag_action_mse={row['action_mse']:.5f} l2_aux=0 adv_weight=0 "
                    f"elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment
                if any(not torch.isfinite(p).all() for key in MODULE_KEYS for p in bundle[key].parameters()):
                    raise FloatingPointError(f"Nonfinite parameters at checkpoint {step}")
                save(step)
                score = proxy_action_mse()
                if best is None or score < best["proxy_action_mse"]:
                    best = dict(step=step, proxy_action_mse=score)
                log(f"CHECKPOINT step={step} proxy_action_mse={score:.5f} "
                    "selection=min proxy action MSE on the record prefix; not a landing score")
                sync()
                segment = time.perf_counter()
        shutil.copyfile(out / f"checkpoint_{cfg['steps']}.pt", out / "final.pt")
        shutil.copyfile(out / f"checkpoint_{best['step']}.pt", out / "selected.pt")
        summary = dict(config=cfg, recipe=recipe.to_dict(), optimizers=groups, provenance=provenance,
            parameters=parameter_counts, trainable_parameters=trainable_counts,
            total_trainable_parameters=sum(trainable_counts.values()),
            inference_parameters=inference_count, unique_training_records=len(columns[0]),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=0,
            real_draws=cfg["steps"] * cfg["batch_size"], simulator_calls=0,
            adversarial_updates=0, supervised_only=True,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={path.name: sha256(path) for path in sorted(out.glob("*.pt"))},
            removed_l2=list(REMOVED_L2), l2_aux_weight=0., adv_weight=cfg["adv_weight"],
            action_anchor_weight=cfg["action_anchor_weight"], functional_weight=cfg["functional_weight"],
            fake_paths=list(FAKE_PATHS),
            selected_step=best["step"], proxy_action_mse=best["proxy_action_mse"],
            selection="Min EMA proxy action MSE on the first 256 training records. Not a landing rate.",
            initialization="Adversarial checkpoint; E_control copied from paired E; only G2 trains")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f} selected_step={best['step']} "
            f"proxy_action_mse={best['proxy_action_mse']:.5f}; landings not run")
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
