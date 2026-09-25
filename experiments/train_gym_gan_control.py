#!/usr/bin/env python
"""GAN throughout: masked joint critic versus joint plus marginal critics."""
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
from lib.gym_sparse_action import build_sparse_records, fit_sparse_scaler, sparse_task_losses
from lib.gym_state_control import training_recipe
from lib.gym_gan_control import (MODULE_KEYS, build_gan_models, initial_hashes, real_views,
    fake_views, discriminator_loss, generator_loss)
from particlegan import scale_learning_rates

DEFAULTS = dict(arm="joint", steps=2500, batch_size=256, checkpoints=[250, 1000, 2500],
    log_interval=250, seed=24003, device="cuda:1", z_dim=32, num_particles=1024,
    width=128, encoder_width=128, context_dim=11, d_width=256, marginal_width=128, marginal_weight=1.,
    adversarial_weight=1., lambda_state=1., lambda_next=1.,
    continuous_weight=1., contact_weight=1.,
    labeled_episode_count=5,
    episodes="results/gym/lunar_lander/data/episodes.json",
    out_dir="results/gym/lunar_lander_gan_control/joint",
    live_log="results/gym/lunar_lander_gan_control/live.log")


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["labeled_episode_count"] != 5:
        raise ValueError("Frozen first comparison requires five labeled episodes")
    if cfg["arm"] not in ("joint", "marginals"):
        raise ValueError("arm must be joint or marginals")
    for key in ("steps", "batch_size", "log_interval", "z_dim", "num_particles", "width", "encoder_width", "d_width", "marginal_width"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key in ("batch_size", "num_particles") else 1):
            raise ValueError(f"Invalid {key}")
    if type(cfg["seed"]) is not int or cfg["context_dim"] != 11:
        raise ValueError("Integer seed and terrain11 required")
    for key in ("lambda_state", "lambda_next", "continuous_weight", "contact_weight", "marginal_weight", "adversarial_weight"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["adversarial_weight"] <= 0:
        raise ValueError("GAN loss must be active throughout training")
    if not isinstance(cfg["checkpoints"], list) or any(type(s) is not int or s <= 0 for s in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def capture_provenance(out, cfg, records, selection, bundle):
    paths = [Path(__file__), ROOT / "lib/gym_gan_control.py", ROOT / "lib/gym_sparse_action.py", ROOT / "lib/gym_state_control.py",
             ROOT / "lib/gym_transition.py", ROOT / "experiments/train_gym_transition.py",
             ROOT / "experiments/config.py"]
    paths += sorted((ROOT / "particlegan").glob("*.py"))
    paths += sorted((ROOT / "configs/gym/lunar_lander_gan_control").glob("*.yaml"))
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
        episodes=dict(path=cfg["episodes"], sha256=sha256(cfg["episodes"])),
        expert_data=dict(split="train", behavior="heuristic", count=len(records["states"]),
            episode_ids=np.unique(records["episode_ids"]).tolist(), arrays=arrays,
            npz_sha256=sha256(out / "sparse_records.npz")),
        labeled_data=dict(count=len(records["labeled_actions"]), episode_ids=selection["labeled_episode_ids"]),
        selection=selection, initial_parameters=initial_hashes(bundle), gan_training=True,
        adversarial_semantics="Every update: prior and encoded-state fake paths averaged; labeled complete and all-record action-hidden joint views averaged; mask fixed context and structural input mask; no synthetic cycle",
        normalization="Shared state scaler fit all available training states/successors; action scaler fit labeled actions only",
        initialization="All E/G1/G2/G3/prior parameters freshly initialized; no pretrained model weights",
        loss_reductions="action: mean squared standardized error over labeled batch x 2; each state: mean squared standardized error over batch x 6 + mean BCE logits over batch x 2; sum three weighted heads + default MoG prior regularizer once + adversarial_weight*(joint + marginal_weight*mean three marginals); D=sumroles; each role averages its views and fake paths")


def train(cfg):
    validate(cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg["device"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    live = Path(cfg["live_log"])
    live.parent.mkdir(parents=True, exist_ok=True)
    records, selection = build_sparse_records(cfg["episodes"], cfg["labeled_episode_count"])
    scaler = fit_sparse_scaler(records)
    bundle = build_gan_models(cfg, scaler, device)
    np.savez_compressed(out / "sparse_records.npz", **records)
    write_json(out / "label_selection.json", selection)
    provenance = capture_provenance(out, cfg, records, selection, bundle)
    values = {k: torch.as_tensor(records[k], device=device) for k in ("states", "next_states", "terrain")}
    labeled_indices = torch.as_tensor(records["labeled_indices"], device=device)
    labeled_actions = torch.as_tensor(records["labeled_actions"], device=device)
    recipe = training_recipe(cfg)
    optimizer, optimizer_d = recipe.make_optimizers(bundle["G"], bundle["D"], bundle["prior"],
        encoder=bundle["E"], ema_critic=copy.deepcopy(bundle["D"]), fused=device.type == "cuda")
    optimizers = (optimizer, optimizer_d)
    base_rates = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    prior_regularizer = recipe.make_prior_regularizer()
    gan = recipe.make_loss()
    ema = {**bundle, **{key: copy.deepcopy(bundle[key]).eval().requires_grad_(False) for key in ("G", "E", "prior")}}
    rng = {name: torch.Generator(device=device).manual_seed(cfg["seed"] + offset)
           for name, offset in dict(labeled=11, auxiliary=21, d_labeled=31, d_auxiliary=41,
                                   latent=51, contact=61, d_latent=71, d_contact=81).items()}
    reg_rngs = {role: torch.Generator(device=device).manual_seed(cfg["seed"] + 100 + i)
                for i, role in enumerate(("joint", "action", "state", "next_state"))}
    critic_penalties = {role: recipe.make_critic_penalty(optimizer_d, generator=generator)
                        for role, generator in reg_rngs.items()}
    draws_digest = {name: hashlib.sha256() for name in ("labeled", "auxiliary", "d_labeled", "d_auxiliary")}
    def batch(prefix=""):
        label_name, all_name = prefix + "labeled", prefix + "auxiliary"
        labeled_ids = torch.randint(len(labeled_actions), (cfg["batch_size"],), device=device, generator=rng[label_name])
        ids = torch.randint(len(records["states"]), (cfg["batch_size"],), device=device, generator=rng[all_name])
        draws_digest[label_name].update(labeled_ids.cpu().numpy().tobytes())
        draws_digest[all_name].update(ids.cpu().numpy().tobytes())
        source_ids = labeled_indices[labeled_ids]
        return dict(labeled_states=values["states"][source_ids], labeled_actions=labeled_actions[labeled_ids],
                    labeled_next_states=values["next_states"][source_ids], labeled_terrain=values["terrain"][source_ids],
                    **{k: v[ids] for k, v in values.items()})
    counts = {key: parameter_count(bundle[key]) for key in MODULE_KEYS}
    inference_count = counts["E"] + counts["prior"] + parameter_count(bundle["G"].branches[1])
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
        checkpoint = dict(format="gym_gan_control_v1", gan_training=True, gan_steps=step, config=cfg, recipe=recipe.to_dict(),
            scaler=scaler.state_dict(), step=step, provenance=provenance,
            data_draw_sha256={k: v.hexdigest() for k,v in draws_digest.items()},
            validation=dict(status="Awaiting independent control rollout evaluation"))
        checkpoint.update({key: ema[key].state_dict() for key in MODULE_KEYS})
        torch.save(checkpoint, out / f"checkpoint_{step}.pt")
    with (out / "log.txt").open("w", buffering=1) as logfile, live.open("a", buffering=1) as livefile, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics:
        def log(message):
            text = f"[{out.name}] {message}"
            print(text, flush=True)
            logfile.write(text + "\n")
            livefile.write(text + "\n")
        log(f"START arm={cfg['arm']} steps={cfg['steps']} expert_records={len(records['states'])} labeled={len(labeled_actions)} device={device}")
        log("st -> E -> z; G1 -> st; G2 -> at; G3 -> st+1; terrain enters E and Gs; no action/history input")
        log(f"GAN active from step1; critics={bundle['D'].roles()}; trainable={counts}; action inference={inference_count}")
        started = time.perf_counter()
        sync()
        segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg["steps"] + 1):
            lr_scale, _ = scale_learning_rates(step - 1, recipe, optimizers, base_rates, bundle["prior"])
            # No generator graph or stale gradients during discriminator optimization.
            optimizer.zero_grad(set_to_none=True)
            bundle["D"].requires_grad_(True)
            d_batch = batch("d_")
            d_views = real_views(bundle, d_batch)
            with torch.no_grad():
                d_fakes = fake_views(bundle, d_views, rng["d_latent"], rng["d_contact"], straight_through=False)
            d_loss, d_terms = discriminator_loss(bundle["D"], d_views, d_fakes, gan, critic_penalties)
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Nonfinite discriminator loss at step {step}")
            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_d.step()
            optimizer_d.zero_grad(set_to_none=True)
            bundle["D"].requires_grad_(False)
            g_batch = batch()
            views = real_views(bundle, g_batch)
            fakes = fake_views(bundle, views, rng["latent"], rng["contact"], straight_through=True)
            adversarial, gan_terms = generator_loss(bundle["D"], views, fakes, gan, cfg["marginal_weight"])
            task, terms = sparse_task_losses(bundle, **{k: v for k, v in g_batch.items() if k != "labeled_next_states"})
            prior_loss = prior_regularizer(bundle["prior"].z)
            loss = task + prior_loss + cfg["adversarial_weight"] * adversarial
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite generator loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for key in ("G", "E", "prior"):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), prior_loss=float(prior_loss.detach()),
                    d_loss=float(d_loss.detach()), g_adversarial_loss=float(adversarial.detach()),
                    **{f"d_{k}": float(v.detach()) for k,v in d_terms.items()},
                    **{f"g_{k}": float(v.detach()) for k,v in gan_terms.items()},
                    **{k: float(v.detach()) for k,v in terms.items()}, lr_scale=lr_scale,
                    elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} action={row['action_loss']:.5f} "
                    f"state={row['state_loss']:.5f} next={row['next_loss']:.5f} prior={row['prior_loss']:.5f} "
                    f"D={row['d_loss']:.5f} GAN={row['g_adversarial_loss']:.5f} elapsed_s={row['elapsed_seconds']:.1f}")
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
            gan_training=True, gan_steps=cfg["steps"], discriminator_parameters=counts["D"],
            parameters=counts, full_parameters=sum(counts.values()), trainable_parameters=counts,
            total_trainable_parameters=sum(counts.values()), inference_parameters=inference_count,
            unique_training_records=len(records["states"]),
            unique_labeled_records=len(labeled_actions), unique_labeled_episodes=len(selection["labeled_episode_ids"]),
            labeled_record_draws=cfg["steps"] * cfg["batch_size"], auxiliary_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_labeled_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_auxiliary_record_draws=cfg["steps"] * cfg["batch_size"],
            real_draws=4 * cfg["steps"] * cfg["batch_size"],
            generator_optimizer_record_draws=2 * cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=2 * cfg["steps"] * cfg["batch_size"], data_draw_sha256={k: v.hexdigest() for k,v in draws_digest.items()}, simulator_calls=0,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            checkpoints={p.name: sha256(p) for p in sorted(out.glob("*.pt"))},
            selection="Deferred: highest validation landing fraction, tie-break mean return",
            control_input="Current observed state plus terrain; no current/previous action or successor input",
            initialization=provenance["initialization"],
            auxiliary_gradient_scope="G1/G3 plus E/prior",
            adversarial_fake_paths=["prior", "encoded"], observation_views=["labeled complete", "all pairs action hidden"])
        write_json(out / "summary.json", summary)
        log(f"COMPLETE train_seconds={optimization_seconds:.1f}; awaiting rollout selection")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=("joint", "marginals"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ("arm", "steps", "device", "out_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
