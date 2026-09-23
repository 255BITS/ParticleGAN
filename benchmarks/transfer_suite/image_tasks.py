"""Small procedural image GANs for controller development, not default gates.

No downloaded data, labels, templates or evaluation metrics enter either
network or the LR controller. The uniformly sampled learned particle prior is
finite; evaluating every particle measures its exact output distribution.
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

from particlegan import GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer
from benchmarks.locked_shared.observation import sustained
from benchmarks.smart_descent.controller import GradientFeedback
from benchmarks.smart_descent.evaluate import FixedControl


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
    names = [Path(__file__), root / "benchmarks/smart_descent/controller.py",
             root / "benchmarks/smart_descent/evaluate.py", root / "benchmarks/learned_lr_evaluation.py",
             root / "benchmarks/locked_shared/observation.py", *sorted((root / "particlegan").glob("*.py"))]
    return dict(version="transfer-images-v1", seed=0, device="cpu", torch_threads=1,
                python=platform.python_version(), torch=torch.__version__,
                torch_git_revision=torch.version.git_version, machine=platform.machine(),
                torch_build=torch.__config__.show(),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in names})


@torch.no_grad()
def measure(generator, prior, centers, thresholds):
    # Enumerate the exact finite prior: no evaluation RNG, no sampled coverage.
    # fork_rng additionally protects training if measurement later gains RNG.
    with torch.random.fork_rng(devices=[]):
        return image_metrics(generator(prior.z), centers, thresholds)


def run_episode(spec, policy, *, ablation="none", fixed=False):
    """Run one CPU seed0 episode; all controller actions follow backward.

    Callers must explicitly authorize reserved tasks; this function supports
    their later execution, while the calibration CLI refuses them.
    """
    started = time.perf_counter()
    observations, losses = [], []
    controller = None
    callback_seconds = 0.
    result = dict(spec=deepcopy(spec), policy=deepcopy(policy), ablation=ablation, fixed=fixed,
                  protocol=fingerprint(), created_at=datetime.now(timezone.utc).isoformat(),
                  live={}, ema={}, observations=observations, losses=losses,
                  convergence={}, actions=[])
    try:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        centers = templates(spec)
        g, d = Generator(spec), Discriminator(spec)
        prior = ParticlePrior(spec["particles"], spec["z_dim"])
        ema_g, ema_prior = deepcopy(g), deepcopy(prior)
        gan = GANLoss("logistic", "rp")
        penalty = GradientPenalty(arm=spec["gradient_penalty"], coeff=spec["penalty_coeff"], kappa=spec["kappa"])
        spread = ParticleRegularizer(weight=spec["prior_weight"])
        opt_g = torch.optim.Adam([*g.parameters(), *prior.parameters()], lr=spec["lr_g"], betas=spec["adam_betas"])
        opt_d = torch.optim.Adam(d.parameters(), lr=spec["lr_d"], betas=spec["adam_betas"])
        controller = (FixedControl if fixed else GradientFeedback)(policy, spec["steps"], ablation=ablation)
        expected_steps = evaluation_steps(spec)
        def update(optimizer, index, role):
            nonlocal callback_seconds
            tick = time.perf_counter()
            controller.step(optimizer, index, role=role)
            callback_seconds += time.perf_counter() - tick
            optimizer.step()
        for index in range(spec["steps"]):
            real = centers[torch.randint(len(centers), (spec["batch_size"],))]
            real = (real + spec["noise_std"] * torch.randn_like(real)).clamp(0., 1.)
            fake = g(prior.sample(spec["batch_size"])[0]).detach()
            opt_d.zero_grad(set_to_none=True)
            d_adv = gan.d_loss(d(real), d(fake))
            d_reg = penalty(d, real, fake, step=index + 1)
            d_loss = d_adv + controller.regularization_scale("d") * d_reg
            d_loss.backward()
            update(opt_d, index, "d")
            d.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake = g(prior.sample(spec["batch_size"])[0])
            g_adv = gan.g_loss(d(fake), d(real).detach())
            g_reg = spread(prior.z)
            g_loss = g_adv + controller.regularization_scale("g") * g_reg
            g_loss.backward()
            update(opt_g, index, "g")
            d.requires_grad_(True)
            if not bool(torch.isfinite(d_loss) & torch.isfinite(g_loss)):
                raise FloatingPointError("nonfinite training loss")
            with torch.no_grad():
                for live, average in ((g, ema_g), (prior, ema_prior)):
                    for parameter, averaged in zip(live.parameters(), average.parameters()):
                        averaged.lerp_(parameter, 1. - spec["ema_decay"])
            if index + 1 in expected_steps:
                live = measure(g, prior, centers, spec["thresholds"])
                ema = measure(ema_g, ema_prior, centers, spec["thresholds"])
                observations.append(dict(step=index + 1, seconds=time.perf_counter() - started, **live, ema=ema))
                losses.append(dict(step=index + 1, d=float(d_loss.detach()), g=float(g_loss.detach()),
                                   d_penalty=float(d_reg.detach()), prior=float(g_reg.detach())))
        result.update(live=live, ema=ema)
        result["convergence"] = sustained(observations, [("modes", ">=", spec["thresholds"]["modes"]),
                                                        ("hq", ">=", spec["thresholds"]["hq_min"])],
                                           expected_steps=expected_steps,
                                           minimum=spec["thresholds"]["minimum_stable_checks"])
        json.dumps(result, allow_nan=False)
    except Exception:
        result["error"] = traceback.format_exc()
        result["live"], result["ema"], result["convergence"] = {}, {}, {}
    result.update(seconds=time.perf_counter() - started, controller_seconds=callback_seconds,
                  actions=controller.trace if controller is not None else [])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", choices=[spec["name"] for spec in TASKS])
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [spec for spec in TASKS if not args.tasks or spec["name"] in args.tasks]
    policy = dict(weights=torch.zeros(2, 2, 5).tolist(), schedule=args.schedule)
    declaration = dict(tasks=TASKS, reserved=RESERVED, protocol=fingerprint())
    declaration_path = args.output / "declaration.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_text()) != declaration:
            raise RuntimeError("source/spec declaration differs; use a new output directory")
    else:
        declaration_path.write_text(json.dumps(declaration, indent=2))
    for spec in selected:
        path = args.output / f"{args.schedule}-{spec['name']}.json"
        if path.exists():
            raise FileExistsError(path)
        print(f"START {args.schedule} {spec['name']} steps={spec['steps']}", flush=True)
        result = run_episode(spec, policy, fixed=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(dict(task=spec["name"], schedule=args.schedule, live=result["live"],
                              convergence=result["convergence"], seconds=result["seconds"], error=result.get("error"))), flush=True)


if __name__ == "__main__":
    main()
