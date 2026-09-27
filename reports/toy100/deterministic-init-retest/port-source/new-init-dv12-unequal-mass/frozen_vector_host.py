"""Exact frozen DV12 vector model/data/scorer definitions; no legacy learner."""
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional
import math
import torch
from torch import nn
MIN_STABLE_CHECKS = 5
DEFAULTS = dict(hidden=64, layers=2, fourier=2, z_dim=4, particles=256, batch=128,
                steps=1200, lr=.001, d_lr_mult=1.5, prior_lr_mult=10., prior_reg=.05,
                betas=[0., .99], ema_decay=.995, reg_arm="b_cap", reg_coeff=3.,
                reg_kappa=1.25, d_every=1, g_every=1)
OBSERVATIONS = 24
EVAL_SAMPLES = 4096

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
    real = sample_target(spec, len(fake), torch.Generator(device=torch.get_default_device()).manual_seed(991), completed_steps)
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


class SimpleMLPGenerator(nn.Module):
    """
    Very small MLP generator: z -> x in R^2.

    Strong enough for the toy problem but still minimal and CPU-friendly.
    """

    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def sliced_w1(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 128,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Sliced Wasserstein-1 distance between two point clouds.

    Projects both clouds onto `n_proj` random unit directions, computes the 1D
    W1 distance per direction as the mean absolute difference of the sorted
    projections, and averages over directions. Pure torch, so it runs on the
    GPU and never leaves the device until the final scalar.

    The two clouds must have the same number of points (the 1D W1 shortcut
    "sort both, take mean |difference|" is only valid for equal-size uniform
    empirical measures).

    Args:
        x, y: (N, 2) samples.
        n_proj: number of random projection directions.
        seed: if given, projections are drawn from a fresh generator with this
            seed, making the estimate reproducible. Ignored if `generator` is
            passed.
        generator: explicit torch.Generator (must live on `x`'s device).

    Returns:
        The sliced W1 estimate as a python float.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"sliced_w1 needs equal sample counts, got {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    device = x.device
    dim = x.shape[1]

    if generator is None and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    with torch.no_grad():
        if generator is None:
            proj = torch.randn(dim, n_proj, device=device, dtype=x.dtype)
        else:
            proj = torch.randn(
                dim, n_proj, device=device, dtype=x.dtype, generator=generator
            )
        proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-12)

        px, _ = (x @ proj).sort(dim=0)  # (N, n_proj)
        py, _ = (y.to(x.dtype) @ proj).sort(dim=0)
        return float((px - py).abs().mean())


def sustained(curve, requirements, *, expected_steps, minimum=MIN_STABLE_CHECKS):
    """Find a passing suffix, never count a transient pass as convergence.

    Only recorded observations are certified. First/confirmation times include
    setup and measurement overhead, and are not inferred for historical rows.
    """
    def passes(point):
        for key, op, bound in requirements:
            value = point.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
            if not (value >= bound if op == ">=" else value <= bound):
                return False
        return True
    if minimum < 2:
        raise ValueError("minimum must be at least two observations")
    if any(b["step"] <= a["step"] for a, b in zip(curve, curve[1:])):
        raise ValueError("observations must have unique increasing steps")
    passing = [passes(p) for p in curve]
    complete = [p["step"] for p in curve] == sorted(expected_steps)
    first = next((p for p, ok in zip(curve, passing) if ok), None)
    start = len(curve)
    while start and passing[start - 1]:
        start -= 1
    suffix = curve[start:]
    stable = complete and len(suffix) >= minimum
    return {"complete": complete, "observations": len(curve), "passing_observations": sum(passing),
            "minimum_stable_checks": minimum, "passing_suffix": len(suffix),
            "first_pass_step": first["step"] if first else None,
            "stable_from_step": suffix[0]["step"] if stable else None,
            "stable_from_seconds": suffix[0].get("seconds") if stable else None,
            "confirmed_step": suffix[minimum - 1]["step"] if stable else None,
            "confirmed_seconds": suffix[minimum - 1].get("seconds") if stable else None}


def score_metrics(values, requirements):
    cells = []
    for name, op, threshold in requirements:
        value = values.get(name)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ok = numeric and (value >= threshold if op == ">=" else value <= threshold)
        cells.append({"metric": name, "value": value if numeric else None, "op": op, "threshold": threshold,
                      "status": "PASS" if ok else "FAIL" if numeric else "MISSING",
                      "margin": (value - threshold if op == ">=" else threshold - value) if numeric else None})
    return cells


def requirements(spec):
    """Image hosts additionally declare measurement settings in their native schema."""
    thresholds = spec["thresholds"]
    if isinstance(thresholds, dict):
        return [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]
    return thresholds


def test_verdict(spec, result):
    """Recompute sustained success from complete live curves, never trust a stamp."""
    if result is None:
        return dict(status="MISSING", attempted=False, passed=False, confirmation_fraction=2., shortfall=2.)
    if result.get("error"):
        return dict(status="ERROR", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    observations = result.get("observations", result.get("curve", []))
    steps = sorted({math.ceil(i * spec["steps"] / 24) for i in range(1, 25)})
    try:
        convergence = sustained(observations, requirements(spec), expected_steps=steps)
    except (KeyError, TypeError, ValueError):
        return dict(status="INVALID", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    cells = baseline.score_metrics(result.get("live", {}), requirements(spec))
    passed = (convergence["confirmed_step"] is not None
              and all(c["status"] == "PASS" for c in cells))
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [2. if c["margin"] is None else min(2., max(0., -c["margin"]) / (abs(c["threshold"]) or 1.))
                for c in cells]
    return dict(status=status, attempted=True, passed=passed, metrics=cells,
                convergence=convergence,
                shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.,
                confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.)


baseline = SimpleNamespace(score_metrics=score_metrics)
