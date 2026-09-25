#!/usr/bin/env python
"""Fine-tune Lunar Lander control by replacing action MSE with the slider error game."""
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
from experiments.train_gym_transition import parameter_count, sha256, write_json
from lib.gym_control import build_expert_records
from lib.gym_slider_finetune import (MODULE_KEYS, action_decoded, assert_finetune_scope,
    build_error_critic, generator_objective, initialize_finetune)
from lib.gym_slider_gan import error_loss
from particlegan import get_recipe, scale_learning_rates

DEFAULTS = dict(arm="slider_finetune", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1", paired_error_weight=1.,
    error_tokens=8, error_width=48, error_heads=4,
    checkpoint="results/gym/lunar_lander/adversarial/best.pt",
    episodes="results/gym/lunar_lander/data/episodes.json",
    out_dir="results/gym/lunar_lander_slider_finetune/action_error",
    live_log="results/gym/lunar_lander_slider_finetune/live.log")


def validate(cfg):
    if set(cfg) != set(DEFAULTS) or cfg["arm"] != "slider_finetune":
        raise ValueError("Unexpected slider fine-tune configuration")
    for key in ("steps", "batch_size", "log_interval", "error_tokens", "error_width", "error_heads"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["error_width"] % cfg["error_heads"]:
        raise ValueError("error_width must be divisible by error_heads")
    if type(cfg["seed"]) is not int or cfg["paired_error_weight"] != 1.:
        raise ValueError("Integer seed and frozen paired-error weight 1 required")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def source_paths():
    names = ["experiments/train_gym_slider_finetune.py", "lib/gym_slider_finetune.py", "lib/gym_slider_gan.py",
             "lib/gym_control.py", "lib/gym_transition.py", "experiments/train_gym_transition.py",
             "experiments/config.py"]
    return ([ROOT / name for name in names] + sorted((ROOT / "particlegan").glob("*.py"))
            + sorted((ROOT / "configs/gym/lunar_lander_slider_finetune").glob("*.yaml"))
            + sorted(path for path in (ROOT / "lib/vendor/concept_slider_core").glob("*") if path.is_file()))


def train(cfg):
    validate(cfg)
    device = torch.device(cfg["device"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, live = Path(cfg["out_dir"]), Path(cfg["live_log"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    live.parent.mkdir(parents=True, exist_ok=True)
    bundle = initialize_finetune(cfg, device)
    world = bundle["world_config"]
    expected = bundle["provenance"].get("dataset", {}).get("episodes.json")
    if expected is not None and sha256(cfg["episodes"]) != expected:
        raise ValueError("Expert episode source differs from initialization provenance")
    records = build_expert_records(cfg["episodes"])
    np.savez_compressed(out / "expert_records.npz", **records)
    physical = torch.as_tensor(np.concatenate([records[k] for k in ("states", "actions", "next_states")], 1),
                               device=device)
    terrain = torch.as_tensor(records["terrain"], device=device)
    previous = torch.as_tensor(records["previous_actions"], device=device)
    scaler = bundle["scaler"]
    normalized = scaler(physical)
    bundle["R"] = build_error_critic(cfg, normalized[:, 8:10], device)
    assert_finetune_scope(bundle)
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, digest in sources.items():
            value = (ROOT / name).read_bytes()
            if hashlib.sha256(value).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(name, value)
    arrays = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                       sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in records.items()}
    provenance = dict(sources=sources, source_archive_sha256=sha256(out / "source.zip"),
        initial_checkpoint=dict(path=cfg["checkpoint"], sha256=sha256(cfg["checkpoint"]), step=bundle["step"]),
        episodes=dict(path=cfg["episodes"], sha256=sha256(cfg["episodes"])),
        expert_data=dict(split="train", behavior="heuristic", count=len(records["states"]),
            episode_ids=np.unique(records["episode_ids"]).tolist(), arrays=arrays,
            npz_sha256=sha256(out / "expert_records.npz")),
        initial_provenance=bundle["provenance"],
        normalization="Unchanged scaler from the world-model checkpoint; error scale fit on its normalized training actions",
        removed_losses=["standardized action MSE"],
        absent_losses=["joint GAN", "marginal GAN", "transition-sample gradient penalty",
                       "state MSE", "contact BCE", "prior regularizer", "synthetic cycle"],
        objective="Paired-error critic on the two normalized action coordinates; G/E remove that error",
        error_cap="recipe critic penalty (recipe.make_critic_penalty) on R noise coordinates only; not applied to transitions",
        minibatch="Generator draws use seed+11, the imitation fine-tune stream; critic draws use seed+21",
        control_input="Expert previous command in shuffled records; learner previous command at rollout")
    recipe = get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=world["z_dim"], num_particles=world["num_particles"],
                        total_steps=cfg["steps"], batch_size=cfg["batch_size"])
    fused = dict(fused=True) if device.type == "cuda" else {}
    opt_g = recipe.make_generator_optimizer(
        list(bundle["E_control"].parameters()) + list(bundle["G"].branches[1].parameters()), **fused)
    opt_r = recipe.make_critic_optimizer(bundle["R"], ema_critic=copy.deepcopy(bundle["R"]), **fused)
    optimizers = (opt_g, opt_r)
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    penalty = recipe.make_critic_penalty(opt_r)
    ema = {key: copy.deepcopy(bundle[key]).eval().requires_grad_(False) for key in ("G", "E", "prior", "E_control")}
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(data=11, d_data=21, error_noise=151, d_error_noise=161).items()}
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", {**recipe.to_dict(),
        "penalty_scope": "Recipe optimizers, EMA, and LR schedule. The recipe critic penalty acts on R's noise coordinates. Transition D is not optimized."})
    write_json(out / "error_normalization.json", dict(scope="action", coordinates=2,
        target_mean=bundle["R"].target_mean.cpu().tolist(), scale=bundle["R"].target_std.cpu().tolist(),
        edit_rms=float(bundle["R"].edit_rms), noise_start=bundle["R"].sigma(1), noise_hold=1.,
        noise_floor=0.03, noise_horizon=cfg["steps"], normalization=bundle["R"].normalization,
        cap=dict(arm=recipe.reg_arm, every=recipe.reg_every)))
    write_json(out / "normalization.json", {key: value.cpu().tolist() for key, value in scaler.state_dict().items()})
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({step for step in cfg["checkpoints"] if step <= cfg["steps"]} | {cfg["steps"]})
    counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    trainable = {key: sum(p.numel() for p in bundle[key].parameters() if p.requires_grad) for key in MODULE_KEYS}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def batch(name):
        return torch.randint(len(physical), (cfg["batch_size"],), device=device, generator=rng[name])

    def save(step):
        saved = dict(format="gym_slider_finetune_v1", config=cfg, world_config=world, recipe=recipe.to_dict(),
            scaler=scaler.state_dict(), step=step, provenance=provenance,
            validation=dict(status="Awaiting independent control rollout evaluation"))
        saved.update({key: ema[key].state_dict() for key in ("G", "E", "prior", "E_control")})
        saved["D"] = bundle["D"].state_dict()
        saved["R"] = bundle["R"].state_dict()
        torch.save(saved, out / f"checkpoint_{step}.pt")

    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[{out.name}] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")
        log(f"START arm=slider_finetune steps={cfg['steps']} expert_records={len(physical)} "
            f"episodes={len(np.unique(records['episode_ids']))} device={device}")
        log("E_control(st, previous at, terrain) -> z -> G2 -> at; R sees noise vs noise+action error; "
            "action MSE logged only; G1/G3/E_pair/prior/D frozen")
        log(f"Trainable parameters={trainable}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg["steps"] + 1):
            lr_scale, _ = scale_learning_rates(step - 1, recipe, optimizers, base_rates)
            critic_ids = batch("d_data")
            bundle["R"].requires_grad_(True)
            with torch.no_grad():
                decoded_r = action_decoded(bundle, physical[critic_ids, :8], previous[critic_ids],
                                           terrain[critic_ids], normalized[critic_ids])
            critic_loss, critic_terms = error_loss(bundle["R"], decoded_r, normalized[critic_ids], step,
                                                   rng["d_error_noise"], penalty=penalty)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError(f"Nonfinite error-critic loss at step {step}")
            opt_r.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_r.step()
            bundle["R"].requires_grad_(False)
            ids = batch("data")
            decoded = action_decoded(bundle, physical[ids, :8], previous[ids], terrain[ids], normalized[ids])
            loss, terms = generator_objective(bundle["R"], decoded, normalized[ids], step, rng["error_noise"],
                                             cfg["paired_error_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite paired-error generator loss at step {step}")
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ema:
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), action_mse=float(terms["action_mse"]),
                    error_g=float(terms["error_g_adversarial"].detach()), error_d=float(critic_loss.detach()),
                    error_d_adversarial=float(critic_terms["error_d_adversarial"].detach()),
                    error_cap=float(critic_terms["error_cap"].detach()), error_sigma=bundle["R"].sigma(step),
                    lr_scale=lr_scale, elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} error_G={row['error_g']:.5f} "
                    f"error_D={row['error_d']:.5f} cap={row['error_cap']:.5f} "
                    f"action_mse={row['action_mse']:.5f} sigma={row['error_sigma']:.3f} "
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
        summary = dict(config=cfg, recipe=recipe.to_dict(), error_cap=dict(arm=recipe.reg_arm, every=recipe.reg_every), provenance=provenance,
            parameters=counts, trainable_parameters=trainable,
            total_trainable_parameters=sum(trainable.values()),
            inference_parameters=parameter_count(bundle["E_control"]) + parameter_count(bundle["G"].branches[1])
            + parameter_count(bundle["prior"]),
            unique_training_records=len(physical),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            error_critic_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=0, simulator_calls=0,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={path.name: sha256(path) for path in sorted(out.glob("*.pt"))},
            removed_losses=provenance["removed_losses"], absent_losses=provenance["absent_losses"],
            selection="Deferred: no landing evaluation in this recipe check",
            initialization="World-model checkpoint; E_control copied from paired E; fresh action-error critic")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f}; action MSE was diagnostic only")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--live-log")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ("steps", "device", "out_dir", "checkpoint", "live_log"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
