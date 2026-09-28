"""Exact selected recovery-host definitions; no benchmark/candidate imports."""
import hashlib
import json
import math
import torch
from torch import nn

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


HIDDEN = 96


N_HIDDEN = 3


FOURIER = 3


EVAL_N = 4096


def ring_means(n_modes: int = N_MODES, radius: float = RADIUS) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, int(n_modes) + 1)[:-1]
    return torch.stack((angles.cos(), angles.sin()), dim=1) * float(radius)


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


def state_receipt(trainer, stream, means):
    state = trainer.state_dict()
    return {"step": trainer.completed_steps,
            "trainer_sha256": digest(state),
            "models_sha256": {name: digest(values) for name, values in state["models"].items()},
            "optimizers_sha256": [digest(values) for values in state["optimizers"]],
            "streams_sha256": {name: digest(values) for name, values in state["streams"].items()},
            "cpu_rng_sha256": digest(state["cpu_rng"]),
            "cuda_rng_sha256": digest(state["cuda_rng"]),
            "real_stream_sha256": digest(stream.get_state()), "means_sha256": digest(means)}


def good(p):
    return p['modes'] == 8 and p['hq'] >= .9


def segment(points, start, end):
    xs = [p for p in points if start < p['step'] <= end]
    first = next((p['step'] for p in xs if good(p)), None)
    after = [p for p in xs if first is not None and p['step'] >= first]
    suffix = []
    for p in reversed(xs):
        if not good(p):
            break
        suffix.append(p)
    failures = [p['step'] for p in after if not good(p)]
    departures = [p['step'] for i, p in enumerate(after) if i and good(after[i-1]) and not good(p)]
    return dict(start=start, end=end, first_arrival=first,
                delay=None if first is None else first-start, passing_since_arrival=sum(map(good, after)),
                checks_since_arrival=len(after), every_failing_observation_since_arrival=failures,
                departures=departures, stable_suffix_start=suffix[-1]['step'] if suffix else None,
                stable_suffix_checks=len(suffix),
                minimum_hq_since_arrival=min((p['hq'] for p in after), default=None),
                minimum_modes_since_arrival=min((p['modes'] for p in after), default=None),
                minimum_hq_entire_segment=min((p['hq'] for p in xs), default=None),
                final=xs[-1] if xs else None)


