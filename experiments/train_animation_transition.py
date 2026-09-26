#!/usr/bin/env python
"""Sprite animation world model: z -> (G1 st, G2 st+1, G3 gt), E(st) -> z dream loop.

Arms (one config schema, leaderboard rows differ only in these switches):
  gan         joint + marginal critics, E anchored on st/st+1/gt, live (undetached) composition
  direct      supervised st -> (st+1, gt) baseline, same updates and frame decoder
  persistence no training; st+1 = st with exact frames (evaluation only)
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

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from lib.sprite_animation import load_episodes, make_dataset, render, STATE_DIM
from lib.animation_transition import (AnimationGenerator, AnimationEncoder, AnimationCritics,
    DirectPredictor, StateScaler, composed, encoded_step, join, split)
from lib.animation_evaluation import (evaluate_split, prior_validity, encoder_health,
    gan_predictor, persistence_predictor)
from particlegan import get_recipe, learning_rate_scale


DEFAULTS = dict(name="base", arm="gan", joint_d=True, anchor="full", detach_synthetic=False, routing_temperature=.25,
    width=256, encoder_width=256, channels=64,
    d_width=256, marginal_width=128, d_channels=32, z_dim=32, num_particles=1024,
    real_encoding_weight=1., synthetic_reconstruction_weight=1., marginal_weight=1.,
    lr=1e-3, seed=24002, data_seed=24002, device="cuda:1", steps=20000, batch_size=256,
    log_interval=250, checkpoints=[2500, 5000, 10000, 15000, 20000],
    data_dir="results/animation/bouncing_sprite/data",
    out_dir="results/animation/bouncing_sprite/base",
    live_log="results/animation/bouncing_sprite/live.log")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_count(module):
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def training_recipe(cfg):
    return get_recipe(prior_kind="mog", sigma_rel=0.025, z_dim=cfg["z_dim"],
                      num_particles=cfg["num_particles"], total_steps=cfg["steps"],
                      batch_size=cfg["batch_size"], lr=cfg["lr"])


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unknown or missing configuration keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] not in ("gan", "direct", "persistence"):
        raise ValueError("arm must be gan, direct or persistence")
    if cfg["anchor"] not in ("full", "g2_only"):
        raise ValueError("anchor must be full or g2_only")


def ensure_data(cfg):
    data = Path(cfg["data_dir"])
    if not (data / "metadata.json").exists():
        make_dataset(data, seed=cfg["data_seed"])
    return {name: sha256(data / f"{name}.npz") for name in ("train", "validation", "test", "ood_test")}


def build_models(cfg, device):
    recipe = training_recipe(cfg)
    torch.manual_seed(cfg["seed"])
    g = AnimationGenerator(cfg["z_dim"], cfg["width"], cfg["channels"]).to(device)
    prior = recipe.make_prior(device=device,
        generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 101))
    torch.manual_seed(cfg["seed"] + 102)
    e = AnimationEncoder(cfg["z_dim"], cfg["encoder_width"], cfg["routing_temperature"]).to(device)
    torch.manual_seed(cfg["seed"] + 100)
    d = AnimationCritics(cfg["d_width"], cfg["marginal_width"], cfg["d_channels"],
                         joint=cfg["joint_d"]).to(device)
    direct = None
    if cfg["arm"] == "direct":
        torch.manual_seed(cfg["seed"] + 103)
        direct = DirectPredictor(cfg["width"], cfg["channels"]).to(device)
        g = e = prior = d = None
    return dict(G=g, E=e, prior=prior, D=d, direct=direct)


def discriminator_loss(d, real, fake, gan, reg, step, rngs):
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        xr, xf = d.inputs(role, real), d.inputs(role, fake)
        penalty, _ = reg.penalty(critic, xr, xf, step, rngs[role], collect_stats=False)
        terms[role] = gan.d_loss(critic(xr), critic(xf)) + penalty
    return sum(terms.values()), terms


def generator_loss(d, real, fake, gan, marginal_weight):
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        with torch.no_grad():
            dr = critic(d.inputs(role, real))
        terms[role] = gan.g_loss(critic(d.inputs(role, fake)), dr)
    marginals = [v for k, v in terms.items() if k != "joint"]
    return terms.get("joint", 0.) + marginal_weight * sum(marginals) / len(marginals), terms


def record_mse(pred, target, parts=("state", "next_state", "image")):
    """Average of per-part MSE so the 1,024 pixels do not outweigh 6 state coordinates."""
    p, t = split(pred), split(target)
    index = dict(state=0, next_state=1, image=2)
    return sum((p[index[k]] - t[index[k]]).square().mean() for k in parts) / len(parts)


def predictor_for(cfg, models, scaler):
    if cfg["arm"] == "persistence":
        return persistence_predictor(scaler)
    if cfg["arm"] == "direct":
        return models["direct"]
    return gan_predictor(models["E"], models["G"], models["prior"])


def evaluate_all(cfg, models, scaler, splits, final=True):
    predict = predictor_for(cfg, models, scaler)
    names = ("test", "ood_test") if final else ("validation",)
    out = {name: evaluate_split(predict, scaler, splits[name]) for name in names}
    if final and cfg["arm"] == "gan":
        out["prior"] = prior_validity(models["G"], models["prior"], scaler)
        out["encoder"] = encoder_health(models["E"], models["prior"], scaler,
                                        splits["test"].reshape(-1, STATE_DIM))
    return out


def source_provenance(out):
    files = [Path(__file__), ROOT / "lib/sprite_animation.py", ROOT / "lib/animation_transition.py",
             ROOT / "lib/animation_evaluation.py"] + sorted((ROOT / "particlegan").glob("*.py"))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.writestr(path, (ROOT / path).read_bytes())
    return dict(sources=sources, source_archive_sha256=sha256(out / "source.zip"))


def train(cfg):
    validate(cfg)
    device = torch.device(cfg["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Use a fresh empty output directory: {out}")
    live = Path(cfg["live_log"])
    live.parent.mkdir(parents=True, exist_ok=True)
    dataset = ensure_data(cfg)
    splits = {k: load_episodes(cfg["data_dir"], k, device) for k in dataset}
    episodes = splits["train"]
    scaler = StateScaler.fit(episodes.reshape(-1, STATE_DIM)).to(device)
    models = build_models(cfg, device)
    g, e, prior, d, direct = (models[k] for k in ("G", "E", "prior", "D", "direct"))
    provenance = dict(**source_provenance(out), dataset=dataset)
    write_json(out / "provenance.json", provenance)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    parameters = {k: 0 if cfg["arm"] == "persistence" else parameter_count(v) for k, v in models.items()}
    started = time.perf_counter()
    log_file = (out / "log.txt").open("w", buffering=1)
    live_file = live.open("a", buffering=1)

    def log(message):
        text = f"[{cfg['name']}] {message}"
        print(text, flush=True)
        log_file.write(text + "\n")
        live_file.write(text + "\n")

    log(f"START arm={cfg['arm']} joint_d={cfg['joint_d']} anchor={cfg['anchor']} "
        f"detach={cfg['detach_synthetic']} steps={cfg['steps']} "
        f"parameters={parameters}")
    history, best, best_step, train_seconds = [], float("inf"), None, 0.
    if cfg["arm"] == "persistence":
        result = evaluate_all(cfg, models, scaler, splits)
        return finish(cfg, out, log, dict(parameters=parameters, train_seconds=0., best_step=None,
                                          provenance=provenance, evaluation=result, history=[]))

    recipe = training_recipe(cfg)
    trainable = [m for m in (g, e, prior, direct) if m is not None]
    ema = {k: copy.deepcopy(v).eval().requires_grad_(False) if k in ("G", "E", "prior", "direct")
           and v is not None else v for k, v in models.items()}
    if d is not None:
        opt_g, opt_d = recipe.make_optimizers(g, d, prior, encoder=e, fused=device.type == "cuda")
        gan, reg, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
        reg_rngs = {r: torch.Generator(device=device).manual_seed(cfg["seed"] + 40 + i)
                    for i, r in enumerate(d.roles())}
    else:
        opt_g = torch.optim.Adam(direct.parameters(), lr=recipe.lr, betas=recipe.betas)
        opt_d = None
    optimizers = [o for o in (opt_g, opt_d) if o is not None]
    base_rates = [[group["lr"] for group in o.param_groups] for o in optimizers]
    data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    d_data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 21)
    latent_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 12)
    d_latent_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 22)
    def batch(rng):
        """Independent real (st, st+1, render(st)) records; training never unrolls a trajectory."""
        n, length = episodes.shape[:2]
        ids = torch.randint(n, (cfg["batch_size"],), device=device, generator=rng)
        t = torch.randint(length - 1, (cfg["batch_size"],), device=device, generator=rng)
        state, successor = episodes[ids, t], episodes[ids, t + 1]
        return join(scaler(state), scaler(successor), render(state))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    segment = time.perf_counter()
    checkpoints = sorted({x for x in cfg["checkpoints"] if x <= cfg["steps"]} | {cfg["steps"]})
    with (out / "metrics.jsonl").open("w", buffering=1) as metrics_file:
        for step in range(1, cfg["steps"] + 1):
            scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            for o, rates in zip(optimizers, base_rates):
                for group, rate in zip(o.param_groups, rates):
                    group["lr"] = rate * scale
            ld, d_terms, terms = torch.zeros((), device=device), {}, {}
            if d is not None:
                d.requires_grad_(True)
                real_d = batch(d_data_rng)
                with torch.no_grad():
                    fakes = [g(prior.sample(len(real_d), d_latent_rng)[0])]
                    fakes.append(composed(e, g, prior, fakes[0])[0])
                    half = len(real_d) // 2
                    fake_d = torch.cat([fakes[0][:half], fakes[1][half:]])
                ld, d_terms = discriminator_loss(d, real_d, fake_d, gan, reg, step, reg_rngs)
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                opt_d.step()
                d.requires_grad_(False)
            real = batch(data_rng)
            if direct is not None:
                loss = record_mse(direct(split(real)[0]), real, ("next_state", "image"))
                terms["direct"] = loss
            else:
                decoded_real = encoded_step(e, g, prior, split(real)[0])[0]
                parts = ("state", "next_state", "image") if cfg["anchor"] == "full" else ("next_state",)
                terms["real_encoding"] = record_mse(decoded_real, real, parts)
                fake = g(prior.sample(len(real), latent_rng)[0])
                comp, decoded_fake, _ = composed(e, g, prior, fake, cfg["detach_synthetic"])
                terms["synthetic"] = record_mse(decoded_fake, fake.detach())
                terms["prior"] = spread(prior.z)
                adversarial = [generator_loss(d, real, fake, gan, cfg["marginal_weight"]),
                               generator_loss(d, real, comp, gan, cfg["marginal_weight"])]
                terms["adversarial"] = sum(a[0] for a in adversarial) / len(adversarial)
                loss = (cfg["real_encoding_weight"] * terms["real_encoding"]
                        + cfg["synthetic_reconstruction_weight"] * terms["synthetic"]
                        + terms["prior"] + terms["adversarial"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ("G", "E", "prior", "direct"):
                    if models[key] is not None:
                        for target, source in zip(ema[key].parameters(), models[key].parameters()):
                            target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg["log_interval"] == 0:
                row = dict(step=step, loss=loss.item(), d_loss=ld.item(),
                           elapsed_seconds=time.perf_counter() - started,
                           terms={k_: float(v.detach()) if torch.is_tensor(v) else float(v) for k_, v in terms.items()},
                           d_terms={k_: float(v) for k_, v in d_terms.items()})
                metrics_file.write(json.dumps(row) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.4f} D={row['d_loss']:.4f} "
                    + " ".join(f"{k_}={v:.4f}" for k_, v in row["terms"].items())
                    + f" elapsed_s={row['elapsed_seconds']:.0f}")
            if step in checkpoints:
                sync()
                train_seconds += time.perf_counter() - segment
                validation = evaluate_all(cfg, ema, scaler, splits, final=False)["validation"]
                history.append(dict(step=step, **validation))
                torch.save({k_: None if v is None else v.state_dict() for k_, v in ema.items()
                            if k_ != "D"} | dict(scaler=scaler.state_dict(), config=cfg, step=step),
                           out / f"checkpoint_{step}.pt")
                if validation["dream_score"] < best:
                    best, best_step = validation["dream_score"], step
                    shutil.copyfile(out / f"checkpoint_{step}.pt", out / "best.pt")
                log("VALIDATION " + json.dumps(history[-1]))
                sync()
                segment = time.perf_counter()
    saved = torch.load(out / "best.pt", map_location=device)
    for key in ("G", "E", "prior", "direct"):
        if ema[key] is not None:
            ema[key].load_state_dict(saved[key])
    result = evaluate_all(cfg, ema, scaler, splits)
    write_dream_gif(cfg, ema, scaler, splits, out)
    return finish(cfg, out, log, dict(parameters=parameters, train_seconds=train_seconds,
        best_step=best_step, selection="minimum validation dream_score", provenance=provenance,
        evaluation=result, history=history))


def write_dream_gif(cfg, models, scaler, splits, out, n=50):
    """Top row: dream frames g1..gn; bottom row: simulator frames; test then OOD episode."""
    from PIL import Image
    from lib.animation_evaluation import dream
    predict = predictor_for(cfg, models, scaler)
    columns = []
    for name in ("test", "ood_test"):
        truth = splits[name][:1, :n + 1]
        _, frames = dream(predict, scaler, truth[:, 0], n)
        dreamed = frames[0].view(n, 32, 32)
        real = render(truth[0, :n])[:, 0]
        columns.append(torch.cat([dreamed, real], 1))
    video = ((torch.cat(columns, 2).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    images = [Image.fromarray(f).resize((f.shape[1] * 4, f.shape[0] * 4), Image.NEAREST) for f in video]
    images[0].save(out / "dream.gif", save_all=True, append_images=images[1:], duration=50, loop=0)


def finish(cfg, out, log, summary):
    summary = dict(config=cfg, total_seconds=summary.get("train_seconds", 0.), **summary)
    write_json(out / "summary.json", summary)
    ev = summary["evaluation"]
    log(f"COMPLETE test dream_score={ev['test']['dream_score']:.5f} "
        f"ood dream_score={ev['ood_test']['dream_score']:.5f} best_step={summary['best_step']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/animation/bouncing_sprite/base.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config(args.config)}
    for key in ("steps", "device", "out_dir", "data_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.steps is not None:
        cfg["checkpoints"] = [s for s in cfg["checkpoints"] if s <= args.steps]
        cfg["log_interval"] = min(cfg["log_interval"], args.steps)
    train(cfg)


if __name__ == "__main__":
    main()
