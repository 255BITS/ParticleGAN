"""Predeclared vector-distribution transfer tasks; reserved data is never calibrated."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import torch

from benchmarks.toy100.device import host_device, rng_fork_devices
from particlegan import GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer
from lib.toy_metrics import sliced_w1
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator
from benchmarks.locked_shared.observation import sustained
from benchmarks.smart_descent.controller import FEATURES, GradientFeedback
from benchmarks.smart_descent.evaluate import FixedControl

OBSERVATIONS = 24
EVAL_SAMPLES = 4096
SEPARATED_BOUNDS = [["sw1_normalized", "<=", .18], ["mass_tv", "<=", .15],
                    ["hq", ">=", .85], ["component_covariance_error", "<=", .85],
                    ["component_min_eigen_ratio", ">=", .15]]
DISTRIBUTION_BOUNDS = [["sw1_normalized", "<=", .18], ["mean_error", "<=", .15],
                       ["covariance_error", "<=", .45]]
DEFAULTS = dict(hidden=64, layers=2, fourier=2, z_dim=4, particles=256, batch=128,
                steps=1200, lr=.001, d_lr_mult=1.5, prior_lr_mult=10., prior_reg=.05,
                betas=[0., .99], ema_decay=.995, reg_arm="b_cap", reg_coeff=3.,
                reg_kappa=1.25, d_every=1, g_every=1)


def _isotropic(widths):
    return [[[s*s, 0.], [0., s*s]] for s in widths]


def _task(name, family, reason, *, means, widths=None, covariance=None, masses=None,
          identifiable=True, tier="ranking", limitations="Finite-particle approximation; one initialization only.", **options):
    count = len(means)
    return deepcopy(DEFAULTS) | dict(
        name=name, family=family, split="development", tier=tier,
        importance_reason=reason, limitations=limitations, kind="gaussian_mixture",
        means=means, covariances=covariance or _isotropic(widths),
        masses=masses or [1/count]*count, identifiable=identifiable,
        thresholds=deepcopy(SEPARATED_BOUNDS if identifiable else DISTRIBUTION_BOUNDS)) | options


_CORNERS = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
TASKS = [
    _task("vector_two_broad", "separated_broad", "Basic learnable multimodal distribution and within-mode spread.",
          means=[[-1., 0.], [1., 0.]], widths=[.25, .25]),
    _task("vector_unequal_mass", "unequal_mass", "Checks target occupancy including the rare 2% component, not uniformity.",
          means=_CORNERS, widths=[.18]*4, masses=[.55, .30, .13, .02],
          thresholds=deepcopy(SEPARATED_BOUNDS)+[["min_mass_ratio", ">=", .25]]),
    _task("vector_unequal_width", "unequal_width", "Checks component-specific scales without imposing one shared Gaussian width.",
          means=_CORNERS, widths=[.07, .12, .20, .30]),
    _task("vector_anisotropic", "anisotropic", "Checks covariance shape: a narrow axis cannot be rescued by a wide one.",
          means=[[-2., -1.], [0., 1.5], [2., -1.]],
          covariance=[[[.09, .018], [.018, .0081]], [[.0081, -.018], [-.018, .09]], [[.04, .03], [.03, .04]]]),
    _task("vector_overlap", "overlapping", "Scores the observable distribution when latent components are not identifiable.",
          means=[[-.35, 0.], [.35, 0.]], widths=[.55, .55], identifiable=False,
          limitations="Component labels and mode recall are deliberately not scored for overlapping densities."),
    _task("vector_narrow", "narrow_resolution", "Deliberately narrow components test critic resolution; nonblocking diagnostic.",
          means=_CORNERS, widths=[.025]*4, steps=1800, tier="diagnostic",
          limitations="Fixed two-band Fourier critic is deliberately mismatched to sigma=.025; failure does not veto a controller."),
    _task("vector_scale_drift", "changing_scale", "Tests adaptation to changing input units before a stationary final window.",
          means=[[-1., 0.], [1., 0.]], widths=[.18, .18], steps=1600, tier="diagnostic",
          scale_start=.6, scale_end=1.6, scale_ramp_end=.6,
          limitations="Nonstationary diagnostic; each checkpoint uses the current target, which stops changing at 60% of budget."),
    dict(**deepcopy(DEFAULTS)) | dict(name="vector_spiral", family="curved_continuous", split="development", tier="ranking",
         importance_reason="Checks continuous curved mass rather than a finite list of target mode centers.",
         limitations="Finite particles approximate a noisy spiral; sliced distances do not prove identical density.",
         kind="spiral", turns=1.5, radius_min=.3, radius_max=2., noise=.08, steps=1600,
         thresholds=deepcopy(DISTRIBUTION_BOUNDS)),
]
RESERVED = [dict(**deepcopy(DEFAULTS)) | dict(
    name="reserved_annulus", family="annulus", split="reserved", tier="ranking",
    importance_reason="Unseen rotationally symmetric continuous support and radial mass law.",
    limitations="Reserved family; no samples or evaluations are produced during calibration.",
    kind="annulus", radius_min=.8, radius_max=2., steps=1600,
    thresholds=deepcopy(DISTRIBUTION_BOUNDS))]


def resolve(spec, *, allow_reserved=False):
    out = deepcopy(DEFAULTS) | deepcopy(spec)
    if out.get("split") == "reserved" and not allow_reserved:
        raise ValueError("reserved tasks cannot be evaluated during development")
    if out.get("kind") not in ("gaussian_mixture", "spiral", "annulus"):
        raise ValueError(f"unsupported vector data kind: {out.get('kind')}")
    if out.get("requires_dynamic_target_scoring"):
        raise ValueError("this dynamic target family has no implemented scorer")
    for key in ("hidden", "layers", "z_dim", "particles", "batch", "steps", "d_every", "g_every"):
        if type(out[key]) is not int or out[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if out["steps"] < OBSERVATIONS:
        raise ValueError("at least 24 steps are required for 24 distinct observations")
    for key in ("lr", "d_lr_mult", "prior_lr_mult"):
        if not math.isfinite(out[key]) or out[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if out["kind"] == "gaussian_mixture":
        means = torch.as_tensor(out["means"], dtype=torch.float64)
        cov = torch.as_tensor(out["covariances"], dtype=torch.float64)
        masses = torch.as_tensor(out["masses"], dtype=torch.float64)
        if means.ndim != 2 or means.shape[1] != 2 or cov.shape != (len(means), 2, 2) or masses.shape != (len(means),):
            raise ValueError("mixture shapes must be [K,2], [K,2,2], [K]")
        if not all(torch.isfinite(x).all() for x in (means, cov, masses)) or not (masses > 0).all() or not torch.isclose(masses.sum(), masses.new_tensor(1.)):
            raise ValueError("mixture values must be finite with positive normalized masses")
        if not torch.allclose(cov, cov.transpose(-1, -2)) or not (torch.linalg.eigvalsh(cov) > 0).all():
            raise ValueError("covariances must be symmetric positive definite")
        if type(out.get("identifiable")) is not bool:
            raise ValueError("mixture spec must explicitly declare identifiable")
        forbidden = {"mass_tv", "hq", "component_covariance_error", "min_mass_ratio", "component_min_eigen_ratio"}
        if not out["identifiable"] and any(key in forbidden for key, _, _ in out["thresholds"]):
            raise ValueError("overlapping mixtures cannot require component recovery metrics")
    return out


def target_scale(spec, completed_steps):
    start, end = spec.get("scale_start", 1.), spec.get("scale_end", 1.)
    progress = min(1., completed_steps / max(1., spec.get("scale_ramp_end", .6)*spec["steps"]))
    return start+(end-start)*progress


def sample_target(spec, count, rng, completed_steps):
    """Only explicit generators are used; target draws cannot touch training RNGs."""
    kind = spec["kind"]
    if kind == "gaussian_mixture":
        means = torch.tensor(spec["means"], dtype=torch.float32)
        cov = torch.tensor(spec["covariances"], dtype=torch.float32)
        index = torch.multinomial(torch.tensor(spec["masses"]), count, replacement=True, generator=rng)
        noise = torch.randn(count, 2, generator=rng)
        points = means[index] + torch.bmm(torch.linalg.cholesky(cov)[index], noise.unsqueeze(2)).squeeze(2)
    elif kind == "spiral":
        u = torch.rand(count, generator=rng)
        angle = u*(2*math.pi*spec["turns"])
        radius = spec["radius_min"]+(spec["radius_max"]-spec["radius_min"])*u
        points = radius[:, None]*torch.stack([angle.cos(), angle.sin()], 1)
        points += spec["noise"]*torch.randn(count, 2, generator=rng)
    elif kind == "annulus":
        angle = 2*math.pi*torch.rand(count, generator=rng)
        radius = (spec["radius_min"]**2 + (spec["radius_max"]**2-spec["radius_min"]**2)*torch.rand(count, generator=rng)).sqrt()
        points = radius[:, None]*torch.stack([angle.cos(), angle.sin()], 1)
    else:
        raise ValueError(f"unsupported vector data kind: {kind}")
    return points*target_scale(spec, completed_steps)


@torch.no_grad()
def score_samples(fake, spec, completed_steps):
    if not torch.isfinite(fake).all():
        return {key: None for key, _, _ in spec["thresholds"]}
    real = sample_target(spec, len(fake), torch.Generator().manual_seed(991), completed_steps)
    centered = real-real.mean(0)
    scale = centered.square().sum(1).mean().sqrt().clamp_min(1e-8)
    rcov = centered.T@centered/len(real)
    fcenter = fake-fake.mean(0)
    fcov = fcenter.T@fcenter/len(fake)
    result = dict(sw1_normalized=sliced_w1(fake, real, 32, seed=992)/float(scale),
                  mean_error=float((fake.mean(0)-real.mean(0)).norm()/scale),
                  covariance_error=float((fcov-rcov).norm()/rcov.norm().clamp_min(1e-8)),
                  target_scale=float(scale), sample_count=len(fake))
    if spec["kind"] == "gaussian_mixture" and spec["identifiable"]:
        units = target_scale(spec, completed_steps)
        means = torch.tensor(spec["means"])*units
        cov = torch.tensor(spec["covariances"])*units**2
        target = torch.tensor(spec["masses"])
        assignment = torch.cdist(fake, means).argmin(1)
        counts = torch.bincount(assignment, minlength=len(means))
        mass = counts/len(fake)
        delta = fake-means[assignment]
        mahal = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(cov)[assignment], delta)
        errors, eigen_ratios = [], []
        for k in range(len(means)):
            points = fake[assignment == k]
            if len(points) < 10:
                errors.append(1.)
                eigen_ratios.append(0.)
            else:
                x = points-points.mean(0)
                empirical = x.T@x/len(points)
                errors.append(float((empirical-cov[k]).norm()/cov[k].norm()))
                inverse = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
                eigen_ratios.append(float(torch.linalg.eigvalsh(inverse@empirical@inverse.T).min()))
        result.update(mass_tv=float((mass-target).abs().sum()/2),
                      min_mass_ratio=float((mass/target).min()), hq=float((mahal <= 9).float().mean()),
                      component_covariance_error=sum(errors)/len(errors), component_covariance_errors=errors,
                      component_min_eigen_ratio=min(eigen_ratios),
                      component_mass=mass.tolist(), target_mass=target.tolist(), component_counts=counts.tolist())
    return result


def passes(metrics, thresholds):
    return all(isinstance(metrics.get(key), (int, float)) and not isinstance(metrics[key], bool)
               and math.isfinite(metrics[key]) and (metrics[key] >= bound if op == ">=" else metrics[key] <= bound)
               for key, op, bound in thresholds)


def run_episode(spec, policy, *, ablation="none", fixed=False, allow_reserved=False):
    """Serial CPU episode; only current gradients enter feedback, never toy metrics.

    d_every/g_every vary update ratios on the shared outer-step clock. Evaluation
    uses fixed independent draws. Reserved and unsupported families fail closed.
    """
    started = time.perf_counter()
    result = dict(live={}, ema={}, observations=[], convergence={}, actions=[], controller_seconds=0.)
    controller = None
    try:
        cfg = resolve(spec, allow_reserved=allow_reserved)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        controller = (FixedControl if fixed else GradientFeedback)(policy, cfg["steps"], ablation=ablation)
        data_rng, latent_rng = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
        penalty_rng = torch.Generator().manual_seed(2)
        prior = ParticlePrior(cfg["particles"], cfg["z_dim"], init_std=.5, generator=torch.Generator().manual_seed(0))
        generator = SimpleMLPGenerator(cfg["z_dim"], cfg["hidden"], cfg["layers"], 2)
        critic = SimpleMLPDiscriminator(2, cfg.get("d_hidden", cfg["hidden"]), cfg.get("d_layers", cfg["layers"]), cfg["fourier"])
        ema_g, ema_prior = deepcopy(generator), deepcopy(prior)
        opt_g = torch.optim.Adam([{"params": generator.parameters(), "lr": cfg["lr"]},
                                  {"params": prior.parameters(), "lr": cfg["lr"]*cfg["prior_lr_mult"]}], betas=tuple(cfg["betas"]))
        opt_d = torch.optim.Adam(critic.parameters(), lr=cfg["lr"]*cfg["d_lr_mult"], betas=tuple(cfg["betas"]))
        gan = GANLoss(cfg.get("loss_type", "logistic"), cfg.get("gan_mode", "rp"))
        penalty = GradientPenalty(cfg["reg_arm"], coeff=cfg["reg_coeff"], kappa=cfg["reg_kappa"])
        spread = ParticleRegularizer(weight=cfg["prior_reg"])
        expected = {math.ceil(i*cfg["steps"]/OBSERVATIONS) for i in range(1, OBSERVATIONS+1)}
        updates = {"g": 0, "d": 0}

        def update(optimizer, step, role):
            tick = time.perf_counter()
            controller.step(optimizer, step, role=role)
            result["controller_seconds"] += time.perf_counter()-tick
            optimizer.step()
            updates[role] += 1

        @torch.no_grad()
        def measure(model, latent, completed):
            samples = model(latent.sample(EVAL_SAMPLES, generator=torch.Generator().manual_seed(990))[0])
            return score_samples(samples, cfg, completed)

        for step in range(cfg["steps"]):
            completed = step+1
            real = sample_target(cfg, cfg["batch"], data_rng, completed)
            if step % cfg["d_every"] == 0:
                fake = generator(prior.sample(cfg["batch"], generator=latent_rng)[0]).detach()
                d_loss = gan.d_loss(critic(real), critic(fake))
                d_loss += penalty(critic, real, fake, step=completed, generator=penalty_rng)*controller.regularization_scale("d")
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                update(opt_d, step, "d")
            if step % cfg["g_every"] == 0:
                critic.requires_grad_(False)
                try:
                    latent = prior.sample(cfg["batch"], generator=latent_rng)[0]
                    fake_logits = critic(generator(latent))
                    real_g = sample_target(cfg, cfg["batch"], data_rng, completed)
                    real_logits = critic(real_g) if gan.mode in ("rp", "ra") else None
                    g_loss = gan.g_loss(fake_logits, real_logits)
                    g_loss += spread(prior.z)*controller.regularization_scale("g")
                    opt_g.zero_grad(set_to_none=True)
                    g_loss.backward()
                    update(opt_g, step, "g")
                finally:
                    critic.requires_grad_(True)
                with torch.no_grad():
                    for target, source in ((ema_g, generator), (ema_prior, prior)):
                        for averaged, current in zip(target.parameters(), source.parameters()):
                            averaged.mul_(cfg["ema_decay"]).add_(current, alpha=1-cfg["ema_decay"])
            if completed in expected:
                with torch.random.fork_rng(devices=rng_fork_devices()):
                    live, ema = measure(generator, prior, completed), measure(ema_g, ema_prior, completed)
                result["observations"].append(dict(**live, ema=ema, step=completed, seconds=time.perf_counter()-started))
        result.update(live=live, ema=ema, update_counts=updates,
                      convergence=sustained(result["observations"], cfg["thresholds"], expected_steps=expected),
                      status="PASS" if passes(live, cfg["thresholds"]) else "FAIL")
        json.dumps(result, allow_nan=False)
    except Exception:
        result["error"] = traceback.format_exc()
        result["status"] = "ERROR"
    result.update(seconds=time.perf_counter()-started, actions=[] if controller is None else controller.trace)
    return result


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = [*root.joinpath("particlegan").rglob("*.py"),
             root/"lib/toy_models.py", root/"lib/toy_metrics.py",
             root/"benchmarks/locked_shared/observation.py",
             root/"benchmarks/learned_lr_evaluation.py",
             *root.joinpath("benchmarks/smart_descent").glob("*.py"), Path(__file__)]
    return dict(version="transfer-vectors-v2", seed=0, device=str(host_device()), threads=1,
                python=platform.python_version(), torch=str(torch.__version__), torch_git_revision=torch.version.git_version,
                torch_build=torch.__config__.show(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                evaluation_samples=EVAL_SAMPLES, observations=OBSERVATIONS,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def fixed_policy(schedule="cosine"):
    return dict(version=2, features=list(FEATURES), weights=[[[0.]*5 for _ in range(2)] for _ in range(2)], interval=5, schedule=schedule)
