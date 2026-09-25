#!/usr/bin/env python
"""Finite-data Lunar Lander: direct baseline and three-generator MoG world models."""
import argparse
import copy
import hashlib
import json
import math
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
from lib.gym_transition import (GymTransitionScaler, GymTransitionGenerator,
    GymTransitionEncoder, GymTransitionCritics, DirectPredictor, contact_record,
    encoded_transition, composed_transition, real_reconstruction,
    synthetic_reconstruction, state_reconstruction)
from particlegan import K3PCritic, get_recipe, learning_rate_scale


DEFAULTS = dict(arm="adversarial", width=128, encoder_width=128, d_width=256,
    marginal_width=128, z_dim=32, num_particles=1024, context_dim=11,
    continuous_weight=1., contact_weight=1., real_encoding_weight=1.,
    synthetic_reconstruction_weight=1., marginal_weight=1., seed=24002,
    device="cuda:1", steps=10000, batch_size=256, log_interval=250,
    checkpoints=[1000, 2500, 5000, 10000],
    data_dir="results/gym/lunar_lander/data",
    out_dir="results/gym/lunar_lander/adversarial",
    live_log="results/gym/lunar_lander/live.log")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_recipe(cfg):
    return get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=cfg["z_dim"], num_particles=cfg["num_particles"],
                      total_steps=cfg["steps"], batch_size=cfg["batch_size"])


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unknown or missing configuration keys: {set(cfg) ^ set(DEFAULTS)}")
    if cfg["arm"] not in ("direct", "reconstruction", "adversarial"):
        raise ValueError("arm must be direct, reconstruction, or adversarial")
    for key in ("width", "encoder_width", "d_width", "marginal_width", "z_dim",
                "num_particles", "context_dim", "steps", "batch_size", "log_interval"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["num_particles"] < 2 or cfg["batch_size"] < 2:
        raise ValueError("num_particles and batch_size must be at least two")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(cfg["checkpoints"], list) or any(type(x) is not int or x < 1 for x in cfg["checkpoints"]):
        raise ValueError("checkpoints must be positive integer steps")
    for key in ("continuous_weight", "contact_weight", "real_encoding_weight",
                "synthetic_reconstruction_weight", "marginal_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("device", "data_dir", "out_dir", "live_log"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    training_recipe(cfg)


def load_split(data_dir, split, device="cpu", context_dim=11):
    """Load independent transitions; no history, identity, or future metadata enters a model."""
    path = Path(data_dir) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in
                  ("states", "actions", "next_states", "terrain")}
    n = len(arrays["states"])
    for key, width in (("states", 8), ("actions", 2), ("next_states", 8), ("terrain", context_dim)):
        if arrays[key].shape != (n, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {split}/{key} shape or nonfinite values")
    if n < 2:
        raise ValueError("At least two transitions per split are required")
    for key in ("states", "next_states"):
        if not np.isin(arrays[key][:, 6:8], [0, 1]).all():
            raise ValueError("Observed contacts must be binary")
    if (np.abs(arrays["actions"]) > 1).any():
        raise ValueError("Actions must lie in [-1, 1]")
    triples = np.concatenate([arrays[k] for k in ("states", "actions", "next_states")], axis=1)
    return (torch.as_tensor(triples, device=device, dtype=torch.float32),
            torch.as_tensor(arrays["terrain"], device=device, dtype=torch.float32))


def parameter_count(module):
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def build_models(cfg, scaler, device):
    """Initialize matching G/E/prior graphs before selecting the comparison arm."""
    recipe = training_recipe(cfg)
    torch.manual_seed(cfg["seed"])
    g = GymTransitionGenerator(z_dim=cfg["z_dim"], width=cfg["width"],
        context_dim=cfg["context_dim"], scaler=scaler).to(device)
    prior = recipe.make_prior(device=device,
        generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 101))
    torch.manual_seed(cfg["seed"] + 102)
    e = GymTransitionEncoder(z_dim=cfg["z_dim"], width=cfg["encoder_width"],
                             context_dim=cfg["context_dim"]).to(device)
    inference_target = parameter_count(e) + parameter_count(g.branches[2]) + parameter_count(prior)
    d = direct = None
    if cfg["arm"] == "direct":
        torch.manual_seed(cfg["seed"] + 103)
        direct = DirectPredictor(context_dim=cfg["context_dim"],
                                 target_parameters=inference_target).to(device)
        g = e = prior = None
    elif cfg["arm"] == "adversarial":
        torch.manual_seed(cfg["seed"] + 100)
        d = GymTransitionCritics(width=cfg["d_width"], marginal_width=cfg["marginal_width"],
                                context_dim=cfg["context_dim"]).to(device)
    return dict(G=g, E=e, prior=prior, D=d, direct=direct, scaler=scaler,
                config=cfg, device=torch.device(device), inference_target=inference_target)


@torch.no_grad()
def predict(bundle, states, actions, terrain, batch_size=1024):
    """Physical next observations; last two fields are contact probabilities."""
    device, scaler = bundle["device"], bundle["scaler"]
    states, actions, terrain = [torch.as_tensor(x, device=device, dtype=torch.float32)
                               for x in (states, actions, terrain)]
    if states.ndim != 2 or states.shape[1] != 8 or actions.shape != (len(states), 2):
        raise ValueError("Expected states [N,8], actions [N,2]")
    if terrain.shape != (len(states), bundle["config"]["context_dim"]):
        raise ValueError("Terrain must have one context row per observation")
    if not len(states):
        return states.clone()
    chunks = []
    for start in range(0, len(states), batch_size):
        sl = slice(start, start + batch_size)
        inputs = torch.cat([scaler.state(states[sl]), scaler.action(actions[sl])], 1)
        if bundle["direct"] is not None:
            decoded = bundle["direct"](inputs, terrain[sl])
        else:
            decoded = encoded_transition(bundle["E"], bundle["G"], bundle["prior"],
                                         inputs, terrain[sl])[0][:, 10:]
        decoded = torch.cat([decoded[:, :6], decoded[:, 6:].sigmoid()], 1)
        chunks.append(scaler.inverse_state(decoded))
    return torch.cat(chunks)


def load_checkpoint(path, device="cpu"):
    """Load an EMA inference checkpoint. Checkpoints do not contain optimizer resume state."""
    saved = torch.load(path, map_location=device, weights_only=False)
    scaler = GymTransitionScaler(**saved["scaler"]).to(device)
    bundle = build_models(saved["config"], scaler, device)
    for key in ("G", "E", "prior", "D", "direct"):
        if bundle[key] is not None:
            bundle[key].load_state_dict(saved[key])
            bundle[key].eval().requires_grad_(False)
    bundle.update(step=saved["step"], validation=saved["validation"], provenance=saved["provenance"])
    return bundle


def discriminator_loss(d, real, fake, terrain, gan, reg, step, rngs):
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        xr, context = d.inputs(role, real, terrain)
        xf, _ = d.inputs(role, fake, terrain)
        dr, df = critic(xr, context)[0], critic(xf, context)[0]
        penalty, _ = reg.penalty(
            lambda x: critic(x, context)[0], xr, xf, step, generator=rngs[role],
            ema_critic=reg.ema_critic(lambda m, x: m.critic_for(role)(x, context)[0]))
        terms[role] = gan.d_loss(dr, df) + penalty
    return sum(terms.values()), terms


def generator_loss(d, real, fake, terrain, gan, marginal_weight):
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        xr, context = d.inputs(role, real, terrain)
        xf, _ = d.inputs(role, fake, terrain)
        with torch.no_grad():
            dr = critic(xr, context)[0]
        terms[role] = gan.g_loss(critic(xf, context)[0], dr)
    return terms["joint"] + marginal_weight * sum(v for k, v in terms.items() if k != "joint") / 3, terms


@torch.no_grad()
def validation_metrics(bundle, real, terrain):
    pred = predict(bundle, real[:, :8], real[:, 8:10], terrain)
    scale = bundle["scaler"].state_scale
    errors = ((pred[:, :6] - real[:, 10:16]) / scale).square().mean(1)
    probs, target = pred[:, 6:].clamp(1e-7, 1 - 1e-7), real[:, 16:18]
    return dict(next_standardized_mse=float(errors.mean()),
                next_standardized_mse_p95=float(errors.quantile(.95)),
                contact_brier=float((probs-target).square().mean()),
                contact_bce=float(torch.nn.functional.binary_cross_entropy(probs, target)))


def source_provenance(out, data_dir):
    files = [Path(__file__), ROOT / "lib/gym_transition.py", ROOT / "lib/gym_data.py",
             ROOT / "examples/gym_world_model.py", ROOT / "experiments/collect_gym_transition.py"]
    files += sorted((ROOT / "particlegan").glob("*.py"))
    files += sorted((ROOT / "configs/gym/lunar_lander").glob("*.yaml"))
    files = [p for p in files if p.exists()]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path, digest in sources.items():
            data = (ROOT / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Source changed during capture")
            archive.writestr(path, data)
    dataset = {name: sha256(Path(data_dir) / name) for name in
               ("train.npz", "validation.npz", "test.npz", "metadata.json", "episodes.json")
               if (Path(data_dir) / name).exists()}
    return dict(sources=sources, dataset=dataset, source_archive_sha256=sha256(out / "source.zip"))


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
    train_real, train_terrain = load_split(cfg["data_dir"], "train", device, cfg["context_dim"])
    validation_real, validation_terrain = load_split(cfg["data_dir"], "validation", device, cfg["context_dim"])
    scaler = GymTransitionScaler.fit(train_real).to(device)
    real_normalized = scaler(train_real)
    models = build_models(cfg, scaler, device)
    recipe = training_recipe(cfg)
    g, e, prior, d, direct = [models[k] for k in ("G", "E", "prior", "D", "direct")]
    ema = {**models}
    for key in ("G", "E", "prior", "direct"):
        if models[key] is not None:
            ema[key] = copy.deepcopy(models[key]).eval().requires_grad_(False)
    if d is not None:
        opt_g, opt_d = recipe.make_optimizers(g, d, prior, encoder=e, fused=device.type == "cuda")
    else:
        params = list(direct.parameters()) if direct is not None else list(g.parameters()) + list(e.parameters())
        groups = [dict(params=params, lr=recipe.lr)]
        if prior is not None:
            groups.append(dict(params=list(prior.parameters()), lr=recipe.lr * recipe.prior_lr_mult,
                               betas=recipe.prior_betas or recipe.betas))
        opt_g = torch.optim.Adam(groups, lr=recipe.lr, betas=recipe.betas, fused=device.type == "cuda")
        opt_d = None
    optimizers = [opt_g] + ([opt_d] if opt_d is not None else [])
    base_rates = [[group["lr"] for group in opt.param_groups] for opt in optimizers]
    # One K3P bundle per critic optimizer (none when D is absent).
    reg = K3PCritic(recipe, d, opt_d) if opt_d is not None else None
    gan, spread = recipe.make_loss(), recipe.make_prior_regularizer()
    data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    d_data_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 21)
    latent_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 12)
    contact_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 31)
    d_latent_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 22)
    d_contact_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 32)
    reg_rngs = {role: torch.Generator(device=device).manual_seed(cfg["seed"] + 40 + i)
                for i, role in enumerate(d.roles())} if d is not None else {}
    weights = dict(continuous_weight=cfg["continuous_weight"], contact_weight=cfg["contact_weight"])
    provenance = source_provenance(out, cfg["data_dir"])
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    write_json(out / "normalization.json", dict(split="train", count=len(train_real),
        **{k: v.detach().cpu().tolist() for k, v in scaler.state_dict().items()}))
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    if prior is not None:
        write_json(out / "prior.json", dict(kind="mog", components=prior.num_particles,
            sigma_rel=prior.sigma_rel, sigma=float(prior.sigma),
            initial_neighbor_distance=float(prior.d0), regularize="full raw center table"))
    parameters = {key: parameter_count(models[key]) for key in ("G", "E", "prior", "D", "direct")}
    inference_parameters = parameter_count(direct) if direct is not None else models["inference_target"]
    checkpoints = sorted({x for x in cfg["checkpoints"] if x <= cfg["steps"]} | {cfg["steps"]})
    history, best, best_step = [], float("inf"), None
    optimization_seconds = 0.

    def batch(rng):
        ids = torch.randint(len(real_normalized), (cfg["batch_size"],), device=device, generator=rng)
        return real_normalized[ids], train_terrain[ids]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def save(path, step, validation):
        saved = dict(config=cfg, recipe=recipe.to_dict(), scaler=scaler.state_dict(),
                     step=step, validation=validation, provenance=provenance)
        for key in ("G", "E", "prior", "D", "direct"):
            saved[key] = None if ema[key] is None else ema[key].state_dict()
        torch.save(saved, path)

    with (out / "log.txt").open("w", buffering=1) as log_file, live.open("a", buffering=1) as live_file, \
         (out / "metrics.jsonl").open("w", buffering=1) as metrics_file:
        def log(message):
            text = f"[{out.name}] {message}"
            print(text, flush=True)
            log_file.write(text + "\n")
            live_file.write(text + "\n")

        log(f"START arm={cfg['arm']} steps={cfg['steps']} finite_train={len(train_real)} device={device}")
        log("G1 -> st; G2 -> at; G3 -> st+1; E(st,at,terrain) -> z_hat -> G3; shared MoG1024 latent" if g is not None
            else "Direct supervised comparison: (st,at,terrain) -> st+1")
        log("State contacts use BCE; fake adversarial contacts use Bernoulli bits with straight-through G gradients")
        log(f"Parameters={parameters}; inference={inference_parameters}; target={models['inference_target']}")
        started = time.perf_counter()
        sync()
        segment_started = time.perf_counter()
        for step in range(1, cfg["steps"] + 1):
            lr_scale = learning_rate_scale(step - 1, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor)
            for opt, rates in zip(optimizers, base_rates):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * lr_scale
            ld = train_real.new_zeros(())
            d_terms = {}
            if d is not None:
                d.requires_grad_(True)
                real_d, context_d = batch(d_data_rng)
                with torch.no_grad():
                    fake_d = contact_record(g(prior.sample(len(real_d), d_latent_rng)[0], context_d), rng=d_contact_rng)
                    composed_d = composed_transition(e, g, prior, fake_d, context_d, rng=d_contact_rng)[0]
                    half = len(real_d) // 2
                    fake_d = torch.cat([fake_d[:half], composed_d[half:]])
                ld, d_terms = discriminator_loss(d, real_d, fake_d, context_d, gan, reg, step, reg_rngs)
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                reg.step()  # spike guard, Adam step, K3P anchor/LR record
                d.requires_grad_(False)
            real, context = batch(data_rng)
            lg = lp = real.new_zeros(())
            g_terms, reconstruction_terms = {}, {}
            if direct is not None:
                decoded = direct(real[:, :10], context)
                le, reconstruction_terms = state_reconstruction(decoded, real[:, 10:], **weights)
            else:
                z, ids = prior.sample(len(real), latent_rng)
                fake = contact_record(g(z, context), rng=contact_rng, straight_through=True)
                composed, decoded_fake, _ = composed_transition(e, g, prior, fake, context,
                    rng=contact_rng, straight_through=True)
                decoded_real, _ = encoded_transition(e, g, prior, real[:, :10], context)
                lr, real_terms = real_reconstruction(decoded_real, real, **weights)
                ls, synthetic_terms = synthetic_reconstruction(decoded_fake, fake, **weights)
                le = cfg["real_encoding_weight"] * lr + cfg["synthetic_reconstruction_weight"] * ls
                reconstruction_terms = {**{f"real_{k}": v for k, v in real_terms.items()},
                                        **{f"synthetic_{k}": v for k, v in synthetic_terms.items()}}
                lp = spread(prior.z)
                if d is not None:
                    original_loss, g_terms = generator_loss(d, real, fake, context, gan, cfg["marginal_weight"])
                    composed_loss, _ = generator_loss(d, real, composed, context, gan, cfg["marginal_weight"])
                    lg = (original_loss + composed_loss) / 2
            loss = le + lg + lp
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
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), d_loss=float(ld.detach()),
                    g_loss=float(lg.detach()), prior_loss=float(lp.detach()),
                    reconstruction_loss=float(le.detach()), lr_scale=lr_scale,
                    elapsed_seconds=time.perf_counter() - started,
                    reconstruction_terms={k: float(v.detach()) for k, v in reconstruction_terms.items()},
                    d_terms={k: float(v.detach()) for k, v in d_terms.items()},
                    g_terms={k: float(v.detach()) for k, v in g_terms.items()})
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} D={row['d_loss']:.5f} "
                    f"G={row['g_loss']:.5f} reconstruction={row['reconstruction_loss']:.5f} "
                    f"elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment_started
                validation = validation_metrics(ema, validation_real, validation_terrain)
                history.append(dict(step=step, **validation))
                save(out / f"checkpoint_{step}.pt", step, validation)
                if validation["next_standardized_mse"] < best:
                    best, best_step = validation["next_standardized_mse"], step
                    shutil.copyfile(out / f"checkpoint_{step}.pt", out / "best.pt")
                log("VALIDATION " + json.dumps(history[-1], allow_nan=False))
                sync()
                segment_started = time.perf_counter()
        shutil.copyfile(out / f"checkpoint_{cfg['steps']}.pt", out / "final.pt")
        summary = dict(config=cfg, recipe=recipe.to_dict(), provenance=provenance,
            parameters=parameters, full_parameters=sum(parameters.values()),
            inference_parameters=inference_parameters, inference_target=models["inference_target"],
            direct_width=direct.width if direct is not None else None,
            unique_training_triples=len(train_real), normalization_triples=len(train_real),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"] if d is not None else 0,
            real_draws=cfg["steps"] * cfg["batch_size"] * (2 if d is not None else 1),
            validation_record_draws=len(validation_real) * len(checkpoints),
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            validation_history=history, best_step=best_step, best_validation_mse=best,
            final_validation=history[-1], selection="minimum validation six-coordinate standardized MSE",
            checkpoints={p.name: sha256(p) for p in sorted(out.glob("*.pt"))},
            contact_training="BCE logits; Bernoulli hard fake records with ST G gradient; detached synthetic contact bits",
            inference="deterministic E->G3 continuous point prediction and contact probabilities; privileged terrain context")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE best_step={best_step} validation_mse={best:.7f} train_seconds={optimization_seconds:.1f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/gym/lunar_lander/adversarial.yaml"))
    parser.add_argument("--arm", choices=("direct", "reconstruction", "adversarial"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **read_config(args.config)}
    for key in ("arm", "steps", "device", "out_dir", "data_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == "__main__":
    main()
