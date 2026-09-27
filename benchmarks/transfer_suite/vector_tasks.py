"""Predeclared vector-distribution transfer tasks; reserved data is never calibrated.

Problem only: the 2-D targets (mixtures, spiral, annulus), the host MLPs, the
sliced/mixture metrics and the verdict (``VectorTask``). Training runs on the
shared ``benchmarks.toy_runner``; optimizers and their LR schedule, loss,
critic penalty, noise and EMA come from the recipe::

    python -m benchmarks.transfer_suite.vector_tasks --task vector_two_broad \
        --log runs/toy-refactor/vector_two_broad.log
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import torch

from benchmarks.gan_v3 import gan_v3_recipe
from benchmarks.locked_shared.observation import checkpoint, recording, sustained
from benchmarks.toy100.device import host_device
from benchmarks.toy_runner import Networks, ToyProblem, run
from lib.toy_metrics import sliced_w1
from lib.toy_models import SimpleMLPGenerator, SimpleMLPDiscriminator
from particlegan import get_recipe, init
from particlegan.training import output_noise_std

# Nothing here reads this name; compare_defaults.optimizer_defaults (a core
# file) still patches it for the frozen LR-control research.
FixedControl = None

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


class VectorTask(ToyProblem):
    """One predeclared 2-D target on the host SimpleMLP generator and critic.

    ``card`` is a declared research-discriminator card (``vector_discriminator``);
    None keeps the host SimpleMLP critic.
    """

    def __init__(self, spec, card=None, *, allow_reserved=False):
        self.spec = resolve(spec, allow_reserved=allow_reserved)
        if self.spec.get("scale_start", 1.) != 1. or self.spec.get("scale_end", 1.) != 1.:
            raise ValueError("a ramped target scale is not expressible on the shared runner "
                             "(it has a one-shot shift, not a per-step target)")
        if self.spec["d_every"] != 1 or self.spec["g_every"] != 1:
            raise ValueError("the shared runner performs one critic and one generator update per step")
        self.card, self.name = card, self.spec["name"]

    def recipe(self):
        s = self.spec
        return get_recipe(z_dim=s["z_dim"], num_particles=s["particles"], batch_size=s["batch"],
                          total_steps=s["steps"])

    def networks(self, recipe, seed):
        s = self.spec
        generator = SimpleMLPGenerator(s["z_dim"], s["hidden"], s["layers"], 2)
        generator = init.deterministic_orthogonal_(generator, seed=seed)
        critic = init.deterministic_orthogonal_(vector_discriminator(s, self.card), seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        return sample_target(self.spec, n, stream, 0)

    def metrics(self, model):
        return score_samples(model.sample(EVAL_SAMPLES).x, self.spec, self.spec["steps"])

    def verdict(self, metrics):
        return "PASS" if passes(metrics, self.spec["thresholds"]) else "FAIL"


def vector_discriminator(spec, card=None):
    """The host critic, or the critic of a declared research card."""
    from particlegan import BatchDistanceDiscriminator
    from .shared_critic_research import constructor as smooth_constructor
    if card is None:
        return SimpleMLPDiscriminator(2, spec.get("d_hidden", spec["hidden"]),
                                      spec.get("d_layers", spec["layers"]), spec["fourier"])
    if card["implementation"] == "shared_batch_feature_v1":
        if (card["feature"], card["placement"], card["trunk_normalization"],
                card["name"]) != ("distance", "head", "center", "batchfeat_center6_distance_head"):
            raise ValueError("only the promoted public batch-distance card is allowed")
        return BatchDistanceDiscriminator(in_dim=2, hidden_dim=card["width"],
            n_hidden=card["layers"], scales=tuple(card["kernel_scales"]),
            beta=card["softplus_beta"], eps=card["eps"])
    if card["implementation"] == "shared_critic_v1":
        return smooth_constructor(card)(2, card["hidden"], card["layers"], card["fourier"])
    raise ValueError("unknown declared discriminator")


# Spec fields that declare the frozen transfer host's recipe (not its problem).
HOST_RECIPE_FIELDS = ("lr", "d_lr_mult", "prior_lr_mult", "betas", "prior_reg", "ema_decay",
                      "loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_kappa")


def spec_recipe(spec, schedule="cosine"):
    """The frozen transfer host's declared GAN v3 recipe at the task's shape.

    A fixed ``cosine`` card is that recipe's own schedule; ``constant`` is the
    same recipe with ``lr_floor=1``. The recipe-built optimizers apply it.
    """
    if schedule not in ("cosine", "constant"):
        raise ValueError(f"unknown fixed schedule: {schedule}")
    cfg = resolve(spec, allow_reserved=True)
    fields = {key: cfg[key] for key in HOST_RECIPE_FIELDS if key in cfg}
    fields["betas"] = tuple(fields["betas"])
    if schedule == "constant":
        fields["lr_floor"] = 1.
    return gan_v3_recipe(z_dim=cfg["z_dim"], num_particles=cfg["particles"], batch_size=cfg["batch"],
                         total_steps=cfg["steps"], **fields)


def _groups(toy):
    """``(role, optimizer, group)`` for the g, prior and d groups of a 1G + prior + 1D run."""
    (opt_d,) = toy.opt_d.values()
    if len(toy.opt_g.param_groups) != 2 or len(opt_d.param_groups) != 1:
        raise RuntimeError("expected generator, prior and critic optimizer groups")
    return (("g", toy.opt_g, toy.opt_g.param_groups[0]), ("prior", toy.opt_g, toy.opt_g.param_groups[1]),
            ("d", opt_d, opt_d.param_groups[0]))


def train(problem, recipe=None, *, log_path=None):
    """Run ``problem`` on the shared runner; the frozen 24-checkpoint result.

    Observations (live, EMA under ``ema``) go through the shared
    ``observation.Recorder``. ``actions`` and ``applied`` are receipts read
    back from the recipe-built optimizers; nothing here sets a rate.
    """
    started = time.perf_counter()
    recipe = problem.recipe() if recipe is None else recipe
    actions, owner = [], {}

    def observe(step, measure):
        toy = owner.setdefault("toy", measure.__self__)  # the ToyRun whose measure this is
        rates = {role: (group["lr"], group["base_lr"]) for role, _, group in _groups(toy)}
        network, prior = (rates[role][0] / rates[role][1] for role in ("g", "prior"))
        action = dict(step=step, multiplier=network)
        if toy.recipe.network_lr_horizon_cap is not None:
            action.update(network_multiplier=network, prior_multiplier=prior,
                          network_lr_horizon_cap=toy.recipe.network_lr_horizon_cap)
            if toy.recipe.network_lr_floor is not None:
                action["network_lr_floor"] = float(toy.recipe.network_lr_floor)
        (noisy,) = toy.noisy.values()
        actions.append(action | dict(lr_g=rates["g"][0], lr_prior=rates["prior"][0], lr_d=rates["d"][0],
                                     input_sigma=noisy.std,
                                     output_sigma=output_noise_std(toy.recipe, step - 1)))
        checkpoint(step, lambda: {**measure(), "ema": measure(ema=True)})

    with recording(recipe.total_steps) as recorder:
        outcome = run(problem, recipe=recipe, observer=observe, log_path=log_path)
    toy = owner["toy"]
    applied = [dict(role=role, lr=group["base_lr"], betas=list(group["betas"]),
                    parameters=sum(p.numel() for p in group["params"]),
                    optimizer="Adam" if isinstance(opt, torch.optim.Adam) else type(opt).__name__)
               for role, opt, group in _groups(toy)]
    nets = toy.nets
    shapes = dict(real_batch=[recipe.batch_size, 2], latent_batch=[recipe.batch_size, recipe.z_dim],
                  prior=list(nets.prior.z.shape),
                  generator_parameters=sum(p.numel() for p in nets.generator.parameters()),
                  discriminator_parameters=sum(p.numel() for p in nets.critics.parameters()))
    strip = lambda row: {k: v for k, v in row.items() if k != "verdict"}
    observations = recorder.curve
    result = dict(live=strip(outcome["live"]), ema=strip(outcome["ema"]), observations=observations,
                  actions=actions, applied=applied, shapes=shapes,
                  update_counts=dict(g=toy.opt_g.completed_steps,
                                     d=next(iter(toy.opt_d.values())).completed_steps),
                  convergence=sustained(observations, problem.spec["thresholds"], expected_steps=recorder.steps),
                  status=outcome["verdict"], hold=outcome["hold"], recipe=outcome["recipe"],
                  seconds=time.perf_counter() - started)
    json.dumps(result, allow_nan=False)
    return result


def run_episode(spec, policy, *, ablation="none", fixed=False, allow_reserved=False):
    """One episode on the shared runner under the spec's declared host recipe.

    Only fixed schedule cards run. The recipe-built optimizers own the LR, so
    an adaptive (feedback) controller or its ablations cannot act here; they
    return an ERROR result rather than silently running the fixed schedule.
    """
    started = time.perf_counter()
    try:
        if not fixed:
            raise NotImplementedError("adaptive LR controllers are not expressible: "
                                      "recipe-built optimizers own the learning rate")
        if ablation != "none":
            raise ValueError("ablations apply to adaptive controllers only")
        torch.set_num_threads(1)
        problem = VectorTask(spec, allow_reserved=allow_reserved)
        result = train(problem, spec_recipe(spec, (policy or {}).get("schedule", "cosine")))
    except Exception:
        result = dict(live={}, ema={}, observations=[], convergence={}, actions=[],
                      error=traceback.format_exc(), status="ERROR")
    result.update(seconds=time.perf_counter() - started, controller_seconds=0.)
    return result


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = [*root.joinpath("particlegan").rglob("*.py"),
             root/"lib/toy_models.py", root/"lib/toy_metrics.py",
             root/"benchmarks/locked_shared/observation.py",
             root/"benchmarks/learned_lr_evaluation.py",
             root/"benchmarks/toy_runner.py", Path(__file__)]
    return dict(version="transfer-vectors-v3", seed=0, device=str(host_device()), threads=1,
                python=platform.python_version(), torch=str(torch.__version__), torch_git_revision=torch.version.git_version,
                torch_build=torch.__config__.show(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                evaluation_samples=EVAL_SAMPLES, observations=OBSERVATIONS,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def fixed_policy(schedule="cosine"):
    """The suite's fixed-schedule card (smart_descent card format; training reads only ``schedule``)."""
    from benchmarks.smart_descent.controller import FEATURES
    return dict(version=2, features=list(FEATURES), weights=[[[0.]*5 for _ in range(2)] for _ in range(2)], interval=5, schedule=schedule)


def main(argv=None):
    """``python -m benchmarks.transfer_suite.vector_tasks --task NAME [toy_runner.main options]``."""
    import argparse
    from benchmarks.toy_runner import main as runner_main
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default=TASKS[0]["name"], choices=[t["name"] for t in TASKS])
    args, rest = parser.parse_known_args(argv)
    return runner_main(VectorTask(next(t for t in TASKS if t["name"] == args.task)), rest)


if __name__ == "__main__":
    raise SystemExit(main())
