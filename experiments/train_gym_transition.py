#!/usr/bin/env python
"""Finite-data Lunar Lander: direct baseline and three-generator MoG world models.

``GymWorldModel`` declares the problem (data, networks, critic views,
reconstruction losses, validation metrics, verdict); the shared
``benchmarks.toy_runner.ToyRun`` trains it from the recipe. ``train`` keeps
only this experiment's checkpoint protocol (EMA checkpoints at fixed steps,
best-by-validation selection, provenance) that downstream control experiments
read through ``load_checkpoint``.
"""
import argparse
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
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from lib.gym_transition import (GymTransitionScaler, GymTransitionGenerator,
    GymTransitionEncoder, GymTransitionCritics, DirectPredictor, contact_record,
    encoded_transition, composed_transition, real_reconstruction,
    synthetic_reconstruction, state_reconstruction)
from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, ToyRun, View


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
    prior = init.deterministic_orthogonal_(recipe.make_prior(device=device,
        generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 101)))
    torch.manual_seed(cfg["seed"] + 102)
    e = GymTransitionEncoder(z_dim=cfg["z_dim"], width=cfg["encoder_width"],
                             context_dim=cfg["context_dim"]).to(device)
    # Deterministic G/E initialization for every arm (D is initialized below).
    init.deterministic_orthogonal_(g, seed=0)
    init.deterministic_orthogonal_(e, seed=2)
    inference_target = parameter_count(e) + parameter_count(g.branches[2]) + parameter_count(prior)
    d = direct = None
    if cfg["arm"] == "direct":
        torch.manual_seed(cfg["seed"] + 103)
        direct = DirectPredictor(context_dim=cfg["context_dim"],
                                 target_parameters=inference_target).to(device)
        init.deterministic_orthogonal_(direct, seed=0)
        g = e = prior = None
    elif cfg["arm"] == "adversarial":
        torch.manual_seed(cfg["seed"] + 100)
        d = GymTransitionCritics(width=cfg["d_width"], marginal_width=cfg["marginal_width"],
                                context_dim=cfg["context_dim"]).to(device)
        init.deterministic_orthogonal_(d, seed=1)
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


def discriminator_loss(d, real, fake, terrain, gan, penalties):
    """Rp + recipe penalty per critic role; ``penalties`` maps role -> critic penalty
    (or is one penalty for all roles)."""
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        xr, context = d.inputs(role, real, terrain)
        xf, _ = d.inputs(role, fake, terrain)
        dr, df = critic(xr, context)[0], critic(xf, context)[0]
        penalty = (penalties[role] if isinstance(penalties, dict) else penalties)(critic, xr, xf, context)
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
             ROOT / "examples/gym_world_model.py", ROOT / "experiments/collect_gym_transition.py",
             ROOT / "benchmarks/toy_runner.py"]
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


class _Logits(nn.Module):
    """Score adapter: a transition critic returns ``(logits, logits[:, None])``."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, x, context):
        return self.critic(x, context)[0]


class GymWorldModel(ToyProblem):
    """Finite-data Lunar Lander transitions: the problem only.

    Roles: G (G1 -> st, G2 -> at, G3 -> st+1) and E (st, at, terrain) -> z_hat on
    a shared MoG prior, plus three critics scoring four views (joint, state,
    action, next_state). ``arm`` picks direct (supervised predictor, no prior),
    reconstruction (G + E + prior, no critic) or adversarial. The shared runner
    builds every optimizer, the loss, penalties, noise and EMA from the recipe.
    Verdict: PASS when validation next-state MSE beats persistence (st+1 = st).
    """

    name = "gym_world_model"

    def __init__(self, cfg, train_real, train_terrain, validation_real, validation_terrain, scaler):
        self.cfg, self.scaler, self.device = cfg, scaler, train_real.device
        self.train_x, self.train_terrain = scaler(train_real), train_terrain
        self.validation_real, self.validation_terrain = validation_real, validation_terrain
        persistence = (validation_real[:, :6] - validation_real[:, 10:16]) / scaler.state_scale
        self.persistence_mse = float(persistence.square().mean())
        self.weights = dict(continuous_weight=cfg["continuous_weight"], contact_weight=cfg["contact_weight"])
        self.models = None

    def recipe(self):
        return training_recipe(self.cfg)

    def networks(self, recipe, seed):
        self.models = m = build_models(self.cfg, self.scaler, self.device)
        if m["direct"] is not None:
            return Networks(generator=m["direct"], critics={}, prior=None)
        critics = {} if m["D"] is None else nn.ModuleDict({k: _Logits(v) for k, v in m["D"].critics.items()})
        return Networks(generator=m["G"], critics=critics, prior=m["prior"], encoder=m["E"])

    def real(self, n, stream):
        ids = torch.randint(len(self.train_x), (n,), device=self.device, generator=stream)
        return Sample(self.train_x[ids], condition=(self.train_terrain[ids],))

    def fake(self, nets, n, stream, real):
        """Paired with a real batch's terrain; condition carries the clean parts for ``losses``."""
        if real is None:
            raise ValueError("gym_world_model samples are conditioned on a real batch's terrain")
        context = real.condition[0]
        if self.cfg["arm"] == "direct":
            prediction = nets.generator(real.x[:, :10], context)
            return Sample(prediction, condition=(context, prediction))
        prior = nets.priors[0]
        z, ids = prior.sample(n, stream)
        record = contact_record(nets.generator(z, context), rng=stream, straight_through=True)
        composed, decoded, _ = composed_transition(nets.encoder, nets.generator, prior, record, context,
                                                   rng=stream, straight_through=True)
        return Sample(torch.cat([record, composed]), condition=(context, record, decoded), indices=ids)

    def views(self, nets, real, fake):
        """Sampled and composed records each score every role; marginals share ``marginal_weight``.

        The runner uses one weight per view for both the D and the G step, so
        the critic side now carries the generator weights: joint 0.5 x 2 = 1,
        each marginal role (marginal_weight / 3) x 2. The pre-runner
        ``discriminator_loss`` weighted every role 1 on one half-sampled /
        half-composed batch, so ``loss_d`` does not compare with pre-runner logs.
        """
        if self.models["D"] is None:
            return []
        n, context, marginal = len(real.x), real.condition[0], self.cfg["marginal_weight"]
        views = []
        for part in (fake.x[:n], fake.x[n:]):
            for role in GymTransitionCritics.roles():
                xr, c = self.models["D"].inputs(role, real.x, context)
                xf, _ = self.models["D"].inputs(role, part, context)
                views.append(View("state" if role == "next_state" else role, xr, xf, (c,),
                                  .5 if role == "joint" else .5 * marginal / 3))
        return views

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        context = real.condition[0]
        if self.cfg["arm"] == "direct":
            return {"reconstruction": state_reconstruction(fake.condition[1], real.x[:, 10:], **self.weights)[0]}
        _, record, decoded = fake.condition
        decoded_real, _ = encoded_transition(nets.encoder, nets.generator, nets.priors[0], real.x[:, :10], context)
        return {"real_reconstruction": self.cfg["real_encoding_weight"]
                * real_reconstruction(decoded_real, real.x, **self.weights)[0],
                "synthetic_reconstruction": self.cfg["synthetic_reconstruction_weight"]
                * synthetic_reconstruction(decoded, record, **self.weights)[0]}

    def bundle(self, nets):
        direct = self.cfg["arm"] == "direct"
        return dict(G=None if direct else nets.generator, E=nets.encoder, direct=nets.generator if direct else None,
                    prior=None if direct else nets.priors[0], scaler=self.scaler, config=self.cfg,
                    device=self.device)

    def metrics(self, model):
        metrics = validation_metrics(self.bundle(model.nets), self.validation_real, self.validation_terrain)
        return dict(metrics, persistence_mse=self.persistence_mse)

    def verdict(self, metrics):
        return "PASS" if metrics["next_standardized_mse"] < metrics["persistence_mse"] else "FAIL"


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
    problem = GymWorldModel(cfg, train_real, train_terrain, validation_real, validation_terrain, scaler)
    toy = ToyRun(problem, seed=cfg["seed"], device=device)
    models, recipe, adversarial = problem.models, toy.recipe, bool(toy.critics)
    # Checkpoint schema read by load_checkpoint: EMA generator side, live critics.
    ema = {key: problem.bundle(toy.ema_nets)[key] for key in ("G", "E", "prior", "direct")}
    ema["D"] = models["D"]
    provenance = source_provenance(out, cfg["data_dir"])
    write_json(out / "provenance.json", provenance)
    write_json(out / "recipe.json", recipe.to_dict())
    write_json(out / "normalization.json", dict(split="train", count=len(train_real),
        **{k: v.detach().cpu().tolist() for k, v in scaler.state_dict().items()}))
    write_json(out / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    prior = models["prior"]
    if prior is not None:
        write_json(out / "prior.json", dict(kind="mog", components=prior.num_particles,
            sigma_rel=prior.sigma_rel, sigma=float(prior.sigma),
            initial_neighbor_distance=float(prior.d0), regularize="recipe prior_reg via the shared runner"))
    parameters = {key: parameter_count(models[key]) for key in ("G", "E", "prior", "D", "direct")}
    inference_parameters = parameters["direct"] if models["direct"] is not None else models["inference_target"]
    checkpoints = sorted({x for x in cfg["checkpoints"] if x <= cfg["steps"]} | {cfg["steps"]})
    history, best, best_step = [], float("inf"), None
    optimization_seconds = 0.

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

        log(f"START arm={cfg['arm']} steps={cfg['steps']} finite_train={len(train_real)} device={device} "
            f"runner=benchmarks.toy_runner persistence_mse={problem.persistence_mse:.6f}")
        log(f"Parameters={parameters}; inference={inference_parameters}; target={models['inference_target']}")
        started = time.perf_counter()
        sync()
        segment_started = time.perf_counter()
        # FLAG: this loop duplicates benchmarks.toy_runner.run() (step + non-finite
        # check) because run()'s observer(step, measure) gets neither the ToyRun
        # (toy.ema_nets / critics, needed to write load_checkpoint-schema .pt files)
        # nor the step's losses (needed for metrics.jsonl), and run() always adds
        # its own observation schedule. Missing shared hook: an observer that
        # receives (toy, step, losses), with run()'s built-in observations optional.
        for _ in range(cfg["steps"]):
            terms ={k: float(v) for k, v in toy.step().items() if k != "step"}
            step = toy.completed_steps
            if not all(math.isfinite(v) for v in terms.values()):
                raise FloatingPointError(f"Nonfinite training loss at step {step}: {terms}")
            if step == 1 or step % cfg["log_interval"] == 0 or step in checkpoints:
                sync()
                group = toy.opt_g.param_groups[0]
                row = dict(step=step, loss=terms["loss_g"], d_loss=terms.get("loss_d", 0.),
                    g_loss=terms["loss_gan"],
                    reconstruction_loss=sum(v for k, v in terms.items() if "reconstruction" in k),
                    lr_scale=group["lr"] / group["base_lr"], elapsed_seconds=time.perf_counter() - started,
                    terms=terms)
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"step={step}/{cfg['steps']} loss={row['loss']:.5f} D={row['d_loss']:.5f} "
                    f"G={row['g_loss']:.5f} reconstruction={row['reconstruction_loss']:.5f} "
                    f"elapsed_s={row['elapsed_seconds']:.1f}")
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment_started
                validation = toy.measure(ema=True)
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
            direct_width=models["direct"].width if models["direct"] is not None else None,
            unique_training_triples=len(train_real), normalization_triples=len(train_real),
            generator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"],
            discriminator_optimizer_record_draws=cfg["steps"] * cfg["batch_size"] if adversarial else 0,
            real_draws=cfg["steps"] * cfg["batch_size"] * (2 if adversarial else 1),
            validation_record_draws=len(validation_real) * len(checkpoints),
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            validation_history=history, best_step=best_step, best_validation_mse=best,
            final_validation=history[-1], verdict=history[-1]["verdict"],
            persistence_mse=problem.persistence_mse,
            selection="minimum validation six-coordinate standardized MSE",
            checkpoints={p.name: sha256(p) for p in sorted(out.glob("*.pt"))},
            runner="benchmarks.toy_runner.ToyRun: recipe-built optimizers, loss, penalties, noise, EMA",
            contact_training="BCE logits; Bernoulli hard fake records with ST G gradient; detached synthetic contact bits",
            inference="deterministic E->G3 continuous point prediction and contact probabilities; privileged terrain context")
        write_json(out / "summary.json", summary)
        log(f"COMPLETE verdict={summary['verdict']} best_step={best_step} validation_mse={best:.7f} "
            f"final_mse={history[-1]['next_standardized_mse']:.7f} persistence_mse={problem.persistence_mse:.7f} "
            f"train_seconds={optimization_seconds:.1f}")
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
