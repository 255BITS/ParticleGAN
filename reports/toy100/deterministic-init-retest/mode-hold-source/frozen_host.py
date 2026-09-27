"""Exact extracted host definitions. No legacy learner or candidate imports."""
from __future__ import annotations
import hashlib
import json
import math
from types import SimpleNamespace
import torch
import torch.nn as nn

class SimpleMLPGenerator(nn.Module):
    """Small MLP generator: z -> x in R^2."""

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


class SimpleMLPDiscriminator(nn.Module):
    """MLP critic: x in R^2 -> scalar score.

    ``fourier=K`` appends sin/cos features at frequencies ``pi * 2^i``
    (i < K) per input dimension.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        fourier: int = 2,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        dim = in_dim + (2 * fourier * in_dim if fourier > 0 else 0)
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs
            h = torch.cat([h, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        return self.net(h).squeeze(-1)


N_MODES = 8


RADIUS = 3.0


SIGMA = 0.07


Z_DIM = 4


HIDDEN = 96


N_HIDDEN = 3


FOURIER = 3


BATCH = 128


EVAL_N = 4096


def ring_means(n_modes: int = N_MODES, radius: float = RADIUS) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, int(n_modes) + 1)[:-1]
    return torch.stack((angles.cos(), angles.sin()), dim=1) * float(radius)


def sample_ring(means: torch.Tensor, n: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, means.shape[0], (int(n),), generator=generator)
    noise = torch.randn(int(n), means.shape[1], generator=generator)
    return means[idx] + float(sigma) * noise


def diversity(samples: torch.Tensor, means: torch.Tensor, sigma: float = SIGMA, *, detailed=False) -> dict:
    """Mode hold on the ring. HQ = within 3 sigma of a center, balls disjoint."""
    dist = torch.cdist(samples, means)
    nearest, which = dist.min(dim=1)
    hq = nearest <= 3.0 * float(sigma)
    counts = torch.bincount(which[hq], minlength=means.shape[0]).to(dtype=torch.float32)
    modes = int((counts > 0).sum())
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    positive = probs[probs > 0]
    entropy = -(positive * positive.log()).sum()
    effective = float(entropy.exp()) if modes else 0.0
    n_modes = int(means.shape[0])
    result = {
        "modes": modes,
        "n_modes": n_modes,
        "hq": float(hq.float().mean()),
        "cover": modes / n_modes,
        "effective_modes": effective,
    }
    if detailed:
        result.update(
            sample_count=len(samples),
            hq_counts=counts.to(torch.int64).tolist(),
            nearest_counts=torch.bincount(which, minlength=n_modes).tolist(),
            closest_distance_per_mode=dist.min(dim=0).values.tolist(),
            hq_radius=3.0 * float(sigma),
            missing_modes=(counts == 0).nonzero().flatten().tolist(),
        )
        if len(samples) <= 128:
            result.update(points=samples.tolist(), nearest_mode=which.tolist(),
                          nearest_distance=nearest.tolist(), is_hq=hq.tolist())
    return result


MIN_STABLE_CHECKS = 5


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


baseline = SimpleNamespace(score_metrics=score_metrics)


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


def digest(value):
    """Portable content hash, including tensor dtype/shape but not device."""
    out = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            out.update(json.dumps(["tensor", str(cpu.dtype), list(cpu.shape)]).encode())
            out.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            out.update(b"dict[")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
            out.update(b"]")
        elif isinstance(item, (list, tuple)):
            out.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            out.update(b"]")
        else:
            out.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())

    visit(value)
    return out.hexdigest()


def rates(trainer):
    return {f"{role}_{i}": float(group["lr"])
            for optimizer, roles in zip((trainer.opt_g, trainer.opt_d), trainer.roles)
            for i, (group, role) in enumerate(zip(optimizer.param_groups, roles))}

