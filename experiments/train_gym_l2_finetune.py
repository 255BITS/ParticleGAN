#!/usr/bin/env python
"""Fine-tune the Lunar Lander controller with standardized expert action MSE.

Continues the validation-selected three-generator world model. E_control starts
as a copy of the paired encoder. Only E_control and G2 train. G1, G3, the
paired encoder, the MoG prior, and the discriminators stay frozen. The step
loss is standardized action MSE and nothing else.
"""
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
from experiments.train_gym_transition import sha256, write_json
from lib.gym_control import build_expert_records, initialize_control, predict_control
from lib.gym_l2_finetune import LOCKED_SUPERVISION, assert_l2_supervision, l2_objective
from particlegan import get_recipe, learning_rate_scale

DEFAULTS = dict(arm="l2", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24002, device="cuda:1",
    checkpoint="results/gym/lunar_lander/adversarial/best.pt",
    episodes="results/gym/lunar_lander/data/episodes.json",
    out_dir="results/gym/lunar_lander_finetune/l2",
    live_log="results/gym/lunar_lander_finetune/live.log")
MODULE_KEYS = ("G", "E", "prior", "D", "E_control")


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] != "l2":
        raise ValueError("arm must be l2")
    for key in ("steps", "batch_size", "log_interval"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "batch_size" else 1):
            raise ValueError(f"Invalid {key}")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")
    for key in ("checkpoint", "episodes", "out_dir", "live_log"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")


def capture_provenance(out, cfg, records, bundle):
    paths = [Path(__file__), ROOT / "lib/gym_l2_finetune.py", ROOT / "lib/gym_control.py",
             ROOT / "lib/gym_transition.py", ROOT / "experiments/train_gym_transition.py",
             ROOT / "experiments/config.py"]
    paths += sorted((ROOT / "particlegan").glob("*.py"))
    paths += sorted((ROOT / "configs/gym/lunar_lander_finetune").glob("*.yaml"))
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
        normalization="Unchanged scaler from the world-model checkpoint",
        supervision=LOCKED_SUPERVISION,
        initialization="Adversarial world-model checkpoint; E_control copied from paired E; G1/G3/E/prior/D frozen")


def train(cfg):
    validate(cfg)
    checkpoint, episodes = Path(cfg["checkpoint"]), Path(cfg["episodes"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"L2 finetune requires the pretrained world-model checkpoint at {checkpoint}. "
            "Reports reference results/gym/lunar_lander/adversarial/best.pt; that file is not vendored.")
    if not episodes.is_file():
        raise FileNotFoundError(
            f"L2 finetune requires expert episodes at {episodes}. "
            "Reports reference results/gym/lunar_lander/data/episodes.json; that file is not vendored.")
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
    bundle = initialize_control(cfg["checkpoint"], "imitation", device)
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
    g, ec = bundle["G"], bundle["E_control"]
    recipe = get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=world["z_dim"], num_particles=world["num_particles"],
                        total_steps=cfg["steps"], batch_size=cfg["batch_size"])
    opt = torch.optim.Adam(list(ec.parameters()) + list(g.branches[1].parameters()),
                           lr=recipe.lr, betas=recipe.betas, fused=device.type == "cuda")
    base_rates = [group["lr"] for group in opt.param_groups]
    ema = {**bundle}
    for key in ("G", "E", "prior", "E_control"):
        ema[key] = copy.deepcopy(bundle[key]).eval().requires_grad_(False)
    data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    trainable = ("E_control", "G2")
    frozen = ("G1", "G3", "E", "prior", "D")
    saved_config = {**cfg, "supervision": LOCKED_SUPERVISION}
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    write_json(out / "normalization.json", {k: v.cpu().tolist() for k, v in scaler.state_dict().items()})
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({s for s in cfg["checkpoints"] if s <= cfg["steps"]} | {cfg["steps"]})

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def save(step):
        saved = dict(format="gym_control_v1", config=saved_config, world_config=world, recipe=recipe.to_dict(),
            scaler=scaler.state_dict(), step=step, provenance=provenance,
            validation=dict(status="Awaiting independent control rollout evaluation"),
            supervision=LOCKED_SUPERVISION)
        saved.update({key: ema[key].state_dict() for key in MODULE_KEYS})
        torch.save(saved, out / f"checkpoint_{step}.pt")

    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[l2] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")

        log(f"START arm=l2 steps={cfg['steps']} expert_records={len(physical)} "
            f"episodes={len(np.unique(records['episode_ids']))} device={device}")
        log("FINETUNE from adversarial world model; E_control(st, previous at) -> z -> G2")
        log("OBJECTIVE standardized_action_mse only; adversarial_updates=0; contact_bce=0; "
            f"trainable={trainable}; frozen={frozen}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg["steps"] + 1):
            lr_scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            for group, rate in zip(opt.param_groups, base_rates):
                group["lr"] = rate * lr_scale
            ids = torch.randint(len(physical), (cfg["batch_size"],), device=device, generator=data_rng)
            real, ctx = normalized[ids], terrain[ids]
            action, _ = predict_control(bundle, physical[ids, :8], previous[ids], ctx)
            loss, terms = l2_objective(scaler, action, real[:, 8:10])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite action MSE at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "E_control"):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                mse = float(terms["standardized_action_mse"].detach())
                row = dict(step=step, loss=float(loss.detach()), standardized_action_mse=mse,
                    adversarial_updates=0, lr_scale=lr_scale, elapsed_seconds=time.perf_counter() - started)
                if row["loss"] != mse:
                    raise RuntimeError("Logged loss diverged from standardized action MSE")
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} action_mse={mse:.5f} adversarial_updates=0 "
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
        summary = dict(config=saved_config, recipe=recipe.to_dict(), provenance=provenance,
            supervision=LOCKED_SUPERVISION, loss_terms=["standardized_action_mse"],
            adversarial_updates=0, discriminator_optimizer_record_draws=0,
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            unique_training_records=len(physical), simulator_calls=0,
            trainable=list(trainable), frozen=list(frozen),
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={p.name: sha256(p) for p in sorted(out.glob("*.pt"))},
            selection="Deferred: highest validation landing fraction, tie-break mean return",
            initialization="Same frozen adversarial checkpoint as imitation; E_control copied from paired E",
            difference_from_scratch="Direct next-state L2 and the previous-action GAN start from fresh weights. This continues pretrained G2 and a copy of E.",
            difference_from_joint="Joint adds reconstruction, prior updates, and discriminator losses. This loss is standardized action MSE only.")
        assert_l2_supervision(saved_config, summary)
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f} adversarial_updates=0; awaiting rollout selection")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/gym/lunar_lander_finetune/l2.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes")
    parser.add_argument("--live-log")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config(args.config)}
    for key in ("steps", "device", "out_dir", "checkpoint", "episodes", "live_log"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
