"""Small procedural image GANs: problem declarations on the shared toy runner.

No downloaded data, labels, templates or evaluation metrics enter either
network. The uniformly sampled learned particle prior is finite; evaluating
every particle measures its exact output distribution.

``ImageTask(spec)`` declares only the problem (template data, conv G/D,
metrics, verdict). Optimizers and their LR schedule, the loss, the critic
penalty, the prior regularizer, noise and EMA come from a recipe through
``benchmarks.toy_runner``. ``run_episode(spec)`` trains under
``spec_recipe(spec)``, the legacy recipe carrying the task card's declared
formulation (LR, betas, penalty, prior weight, EMA)::

    python -u -m benchmarks.transfer_suite.image_tasks --output /tmp/images --log runs/toy-refactor/transfer_image.log
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.toy100.device import host_device, rng_fork_devices
from benchmarks.toy_runner import Networks, ToyProblem, ToyRun, run
from benchmarks.locked_shared.observation import sustained
from particlegan import get_recipe, init


def _spec(name, pattern, *, tier="ranking", reason, architecture="transpose", width=12,
          z_dim=8, steps=600, split="development", family="image_conv_transpose"):
    counts = {"stripes2": 2, "bars4": 4, "blobs4": 4, "intensity2": 2, "bars8": 8}
    return dict(name=name, family=family, split=split, tier=tier, importance_reason=reason,
                limitations="Finite 32-particle prior; 8x8 grayscale templates; no natural-image or stochastic-texture fidelity claim.",
                pattern=pattern, architecture=architecture, width=width, z_dim=z_dim,
                steps=steps, modes=counts[pattern], batch_size=32, particles=32,
                lr_g=.0017, lr_d=.0017, adam_betas=[0., .99], noise_std=.01,
                gradient_penalty="b_cap", penalty_coeff=3., kappa=1.25,
                prior_weight=.05, ema_decay=.99,
                thresholds=dict(hq_min=.9, modes=counts[pattern],
                                quality_rmse=.06 if pattern == "intensity2" else .10,
                                min_mode_fraction=.5 / counts[pattern], observations=24,
                                minimum_stable_checks=5))


TASKS = [
    _spec("img_stripes2", "stripes2", reason="Healthy orientation transfer: two distinct stripe orientations with an adequately sized convolutional GAN."),
    _spec("img_bars4", "bars4", reason="Healthy location transfer: four horizontal/vertical bar positions test spatial coverage."),
    _spec("img_blobs4", "blobs4", reason="Healthy location transfer: four small corner patches test localized quality and coverage."),
    _spec("img_intensity2", "intensity2", reason="Healthy photometric transfer: two patch intensities require intensity fidelity as well as support coverage."),
    _spec("img_bars8", "bars8", tier="diagnostic", reason="Denser support stress: eight bar positions may exceed the short budget; failure cannot disqualify a controller."),
    _spec("img_tiny_generator", "bars4", tier="diagnostic", width=2, z_dim=1, steps=480,
          reason="Undercapacity stress: width2 and a one-dimensional latent may constrain representation and optimization; non-blocking."),
    _spec("img_mean_discriminator", "blobs4", tier="diagnostic", architecture="mean_discriminator", steps=480,
          reason="Low-information architecture stress: D sees only image mean; equal-mass patch positions are indistinguishable, so failure is non-blocking."),
    _spec("img_uniform_generator", "stripes2", tier="diagnostic", architecture="uniform_generator", steps=480,
          reason="Known representation failure: G can only output spatially uniform images, so stripe quality is impossible; diagnostic only."),
]
RESERVED = _spec("img_residual_bars4", "bars4", family="image_conv_residual_upsample",
                 split="reserved", architecture="residual_upsample", width=16, steps=600,
                 reason="Reserved architecture transfer: nearest-neighbor upsampling with residual convolutions; never evaluated during development.")


def templates(spec):
    """No labels are supplied during training; these templates only define data."""
    kind = spec["pattern"]
    images = torch.zeros(spec["modes"], 1, 8, 8)
    if kind == "stripes2":
        images[0, 0, 3:5, :] = 1.
        images[1, 0, :, 3:5] = 1.
    elif kind in ("bars4", "bars8"):
        positions = [1, 5] if kind == "bars4" else [0, 2, 4, 6]
        for index, position in enumerate(positions):
            images[index, 0, :, position:position + 2] = 1.
            images[index + len(positions), 0, position:position + 2, :] = 1.
    elif kind == "blobs4":
        for index, (row, col) in enumerate(((1, 1), (1, 5), (5, 1), (5, 5))):
            images[index, 0, row:row + 2, col:col + 2] = 1.
    elif kind == "intensity2":
        images[0, 0, 2:6, 2:6] = .35
        images[1, 0, 2:6, 2:6] = .85
    else:
        raise ValueError(f"unknown pattern {kind}")
    return images


class Generator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        width = spec["width"]
        self.architecture = spec["architecture"]
        self.input = nn.Linear(spec["z_dim"], width * 4)
        if self.architecture == "residual_upsample":
            self.first = nn.Conv2d(width, width, 3, padding=1)
            self.second = nn.Conv2d(width, width, 3, padding=1)
        else:
            self.first = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
            self.second = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.output = nn.Conv2d(width, 1, 3, padding=1)
        self.width = width

    def forward(self, z):
        value = F.leaky_relu(self.input(z).reshape(-1, self.width, 2, 2), .2)
        if self.architecture == "residual_upsample":
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.first(value), .2)
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.second(value), .2)
        else:
            value = F.leaky_relu(self.first(value), .2)
            value = F.leaky_relu(self.second(value), .2)
        value = self.output(value).sigmoid()
        if self.architecture == "uniform_generator":
            value = value.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        return value


class Discriminator(nn.Module):
    def __init__(self, spec):
        super().__init__()
        # The deliberately tiny-generator diagnostic keeps a healthy D.
        width = max(12, spec["width"])
        self.mean_only = spec["architecture"] == "mean_discriminator"
        self.network = nn.Sequential(nn.Conv2d(1, width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                     nn.Flatten(), nn.Linear(8 * width, 1))

    def forward(self, images):
        if self.mean_only:
            images = images.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        return self.network(images).flatten()


def image_metrics(images, centers, thresholds):
    if images.ndim != 4 or images.shape[1:] != (1, 8, 8) or not torch.isfinite(images).all():
        raise ValueError("finite N×1×8×8 images required")
    rmses = (images[:, None] - centers[None]).square().mean((2, 3, 4)).sqrt()
    nearest_rmse, assignment = rmses.min(1)
    quality = nearest_rmse <= thresholds["quality_rmse"]
    counts = torch.bincount(assignment[quality], minlength=len(centers))
    fractions = counts.double() / len(images)
    all_counts = torch.bincount(assignment, minlength=len(centers)).double() / len(images)
    return dict(modes=int((fractions >= thresholds["min_mode_fraction"]).sum()),
                hq=float(quality.double().mean()), mean_rmse=float(nearest_rmse.mean()),
                quality_mode_fractions=fractions.tolist(),
                mode_fractions=all_counts.tolist(),
                distribution_tv=float((all_counts - 1. / len(centers)).abs().sum() / 2))


def evaluation_steps(spec):
    count = spec["thresholds"]["observations"]
    if spec["steps"] < count:
        raise ValueError("budget must permit all 24 distinct evaluations")
    return [math.ceil(index * spec["steps"] / count) for index in range(1, count + 1)]


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    names = [Path(__file__), root / "benchmarks/toy_runner.py", root / "benchmarks/gan_v3.py",
             *sorted((root / "benchmarks/legacy").glob("*.py")),
             root / "benchmarks/locked_shared/observation.py", *sorted((root / "particlegan").glob("*.py"))]
    return dict(version="transfer-images-v2", seed=0, device=str(host_device()), torch_threads=1,
                python=platform.python_version(), torch=torch.__version__,
                torch_git_revision=torch.version.git_version, machine=platform.machine(),
                torch_build=torch.__config__.show(),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in names})


@torch.no_grad()
def measure(generator, prior, centers, thresholds):
    """Enumerate the exact finite prior: no evaluation RNG, no sampled coverage."""
    with torch.random.fork_rng(devices=rng_fork_devices()):
        images = generator(prior.z)
        return image_metrics(images, centers.to(images.device), thresholds)


def passes(metrics, thresholds):
    return metrics["modes"] >= thresholds["modes"] and metrics["hq"] >= thresholds["hq_min"]


class ImageTask(ToyProblem):
    """One procedural 8x8 task: noisy templates, conv G + particle prior, conv D.

    ``spec`` supplies the data (pattern, pixel noise), the architecture (card,
    width, ``prior_learnable``), the budget and the gates. Its formulation
    fields are read only by ``spec_recipe``.
    """

    def __init__(self, spec):
        self.spec, self.name = spec, spec["name"]
        self.centers = templates(spec)

    def recipe(self):
        spec = self.spec
        return get_recipe(z_dim=spec["z_dim"], num_particles=spec["particles"],
                          batch_size=spec["batch_size"], total_steps=spec["steps"])

    def networks(self, recipe, seed):
        spec = self.spec
        if (recipe.z_dim, recipe.num_particles) != (spec["z_dim"], spec["particles"]):
            raise ValueError("recipe latent shape differs from the image task")
        generator = init.deterministic_orthogonal_(Generator(spec), seed=seed)
        critic = init.deterministic_orthogonal_(Discriminator(spec), seed=seed + 1)
        # The R2 table is declared on the learnable Parameter; a fixed prior is a
        # buffer init never touches, so it copies that same table (both arms of
        # the fixed-vs-learnable card start from identical particles).
        table = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        prior = table
        if not spec.get("prior_learnable", True):
            prior = recipe.make_prior(learnable=False)
            prior.z.copy_(table.z.detach())
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        centers = self.centers.to(stream.device)
        images = centers[torch.randint(len(centers), (n,), generator=stream, device=stream.device)]
        noise = torch.randn(images.shape, generator=stream, device=images.device)
        return (images + self.spec["noise_std"] * noise).clamp(0., 1.)

    def metrics(self, model):
        return measure(model.nets.generator, model.nets.prior, self.centers, self.spec["thresholds"])

    def verdict(self, metrics):
        return "PASS" if passes(metrics, self.spec["thresholds"]) else "FAIL"


class SupervisedWitness(ImageTask):
    """Representability control, not a GAN: no critic; particle i regresses
    onto template ``i % modes`` by MSE through the recipe generator optimizer."""

    def networks(self, recipe, seed):
        nets = super().networks(recipe, seed)
        nets.critics = {}
        return nets

    def losses(self, role, nets, real, fake):
        z = nets.prior.z
        target = self.centers.to(z.device)[torch.arange(len(z), device=z.device) % len(self.centers)]
        return {"supervised_mse": F.mse_loss(nets.generator(z), target)}


def spec_recipe(spec):
    """The legacy (GAN v3) recipe carrying the task card's declared formulation."""
    from benchmarks.gan_v3 import gan_v3_recipe
    return gan_v3_recipe(
        z_dim=spec["z_dim"], num_particles=spec["particles"], batch_size=spec["batch_size"],
        total_steps=spec["steps"], lr=spec["lr_g"], d_lr_mult=spec["lr_d"] / spec["lr_g"],
        prior_lr_mult=spec.get("prior_lr_multiplier", 1.), betas=tuple(spec["adam_betas"]),
        loss_type=spec.get("loss_type", "logistic"), gan_mode=spec.get("gan_mode", "rp"),
        reg_arm=spec["gradient_penalty"], reg_coeff=spec["penalty_coeff"], reg_kappa=spec["kappa"],
        prior_reg=spec["prior_weight"], ema_decay=spec["ema_decay"])


def receipts(spec, recipe):
    """What the runner builds from ``recipe`` for this task: optimizer groups and shapes."""
    probe = ToyRun(ImageTask(spec), recipe=recipe)
    optimizers = [(probe.opt_g, None), *((opt, "critic") for opt in probe.opt_d.values())]
    groups = [dict(optimizer=type(opt).__name__, role=role or group["role"], lr=group["lr"],
                   betas=list(group["betas"]), parameters=sum(p.numel() for p in group["params"]))
              for opt, role in optimizers for group in opt.param_groups]
    nets = probe.nets
    with torch.no_grad(), torch.random.fork_rng(devices=rng_fork_devices()):
        fake = nets.generator(nets.prior.z[:2])
    shapes = dict(real_batch=[recipe.batch_size, 1, 8, 8], latent_batch=[recipe.batch_size, recipe.z_dim],
                  generator_output=list(fake.shape), prior=list(nets.prior.z.shape),
                  generator_parameters=sum(p.numel() for p in nets.generator.parameters()),
                  discriminator_parameters=sum(p.numel() for c in probe.critics.values() for p in c.parameters()))
    return groups, shapes


def _training_streams(toy):
    return [stream.get_state() for stream in (toy.data_stream, toy.latent_stream, toy.noise_stream)]


def train(spec, recipe=None, *, problem=None, max_steps=None, log=None, witness=None):
    """Train one task on the shared runner under ``recipe`` (default: the problem's recipe).

    Returns the frozen result shape: final live/EMA metrics, observations at
    ``evaluation_steps`` (live metrics plus an ``ema`` row), losses at the
    runner's observations, and the sustained-convergence summary.
    ``eval_streams_preserved`` is observed: every evaluation left the run's
    data, latent and noise streams byte-identical. ``witness(step, toy)``, if
    given, sees the run after every update (read-only receipts).
    """
    problem = ImageTask(spec) if problem is None else problem
    recipe = problem.recipe() if recipe is None else recipe
    if (recipe.total_steps, recipe.batch_size) != (spec["steps"], spec["batch_size"]):
        raise ValueError("recipe budget or batch differs from the image task")
    steps = spec["steps"] if max_steps is None else min(max_steps, spec["steps"])
    expected = evaluation_steps(spec)
    wanted, observations, losses, preserved = set(expected), [], [], []
    started = time.perf_counter()

    def observe(step, measure):
        toy = getattr(measure, "__self__", None)
        if not isinstance(toy, ToyRun):
            raise TypeError("toy_runner.run must pass the run's bound measure")
        if witness is not None:
            witness(step, toy)
        if step in wanted:
            before = _training_streams(toy)
            live, ema = measure(), measure(ema=True)
            preserved.append(all(torch.equal(a, b) for a, b in zip(before, _training_streams(toy))))
            for row in (live, ema):
                row.pop("verdict")
            observations.append(dict(step=step, seconds=time.perf_counter() - started, **live, ema=ema))

    def record(row):
        if "loss_g" in row:
            losses.append({k: v for k, v in row.items() if k == "step" or k.startswith("loss")
                           or k == "supervised_mse"})
        if log is not None:
            log(row)
    out = run(problem, recipe=recipe, steps=steps, observe_every=max(1, spec["steps"] // 24),
              log=record, observer=observe)
    strip = lambda row: {k: v for k, v in row.items() if k != "verdict"}
    result = dict(route="benchmarks.toy_runner", recipe=recipe.to_dict(), live=strip(out["live"]),
                  ema=strip(out["ema"]), observations=observations, losses=losses, hold=out["hold"],
                  eval_streams_preserved=bool(preserved) and all(preserved),
                  update_counts=dict(g=steps, d=0 if isinstance(problem, SupervisedWitness) else steps),
                  seconds=time.perf_counter() - started)
    if len(observations) == len(expected):
        result["convergence"] = sustained(observations, [("modes", ">=", spec["thresholds"]["modes"]),
                                                        ("hq", ">=", spec["thresholds"]["hq_min"])],
                                          expected_steps=expected,
                                          minimum=spec["thresholds"]["minimum_stable_checks"])
    return result


def run_episode(spec, policy=None, *, ablation="none", fixed=True, recipe=None, log=None):
    """One CPU seed0 episode under ``spec_recipe(spec)`` (or ``recipe``); errors are recorded.

    ``policy``/``fixed``/``ablation`` keep the suite's call shape only: the
    recipe-built optimizers own the LR schedule, so no learned or feedback
    LR controller can drive this host.
    """
    if not fixed or ablation != "none":
        raise ValueError("image tasks have no LR controller: the recipe-built optimizers own the schedule")
    started = time.perf_counter()
    result = dict(spec=deepcopy(spec), policy=deepcopy(policy), protocol=fingerprint(),
                  created_at=datetime.now(timezone.utc).isoformat())
    try:
        torch.set_num_threads(1)
        recipe = spec_recipe(spec) if recipe is None else recipe
        # The groups the recipe-built optimizers hold (there is no controller to audit).
        result["applied"], result["shapes"] = receipts(spec, recipe)
        result.update(train(spec, recipe, log=log))
        json.dumps(result, allow_nan=False)
    except Exception:
        result.update(error=traceback.format_exc(), live={}, ema={}, convergence={})
    result["seconds"] = time.perf_counter() - started
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", choices=[spec["name"] for spec in TASKS])
    parser.add_argument("--log", type=Path, default=Path("runs/toy-refactor/transfer_image.log"),
                        help="one JSON line per observation (tail -f)")
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("")
    selected = [spec for spec in TASKS if not args.tasks or spec["name"] in args.tasks]
    declaration = dict(tasks=TASKS, reserved=RESERVED, protocol=fingerprint())
    declaration_path = args.output / "declaration.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_text()) != declaration:
            raise RuntimeError("source/spec declaration differs; use a new output directory")
    else:
        declaration_path.write_text(json.dumps(declaration, indent=2))

    def log(row):
        with args.log.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False, default=float) + "\n")
    for spec in selected:
        path = args.output / f"{spec['name']}.json"
        if path.exists():
            raise FileExistsError(path)
        print(f"START {spec['name']} steps={spec['steps']}", flush=True)
        result = run_episode(spec, log=log)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(dict(event="DONE", task=spec["name"], live=result["live"],
                              convergence=result["convergence"], seconds=result["seconds"],
                              error=result.get("error"))), flush=True)


if __name__ == "__main__":
    main()
