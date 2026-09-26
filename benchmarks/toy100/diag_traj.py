"""Read-only per-step trace for the ring mode-hold loop.

Enabled only when ``K3P_DIAG_TRAJ`` is a file path. Every value is taken from
tensors the step already computed, or from a no-grad forward of the inner
generator on the particle table (no noise generator). Nothing here is written
back into parameters, gradients, or RNG streams.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

_STASH: dict = {}
_FILE = None


def enabled() -> bool:
    return bool(os.environ.get("K3P_DIAG_TRAJ"))


def note(**kwargs) -> None:
    if enabled():
        _STASH.update(kwargs)


def begin_step() -> None:
    _STASH.clear()


def _r(value, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def _sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def assign(samples: torch.Tensor, means: torch.Tensor, sigma: float) -> dict:
    """Nearest-mode counts. HQ matches ``mode_hold.diversity`` (3 sigma)."""
    dist = torch.cdist(samples.detach(), means.detach())
    nearest, which = dist.min(dim=1)
    n = int(means.shape[0])
    hq = nearest <= 3.0 * float(sigma)
    hq_counts = torch.bincount(which[hq], minlength=n).tolist()
    nearest_counts = torch.bincount(which, minlength=n).tolist()
    dist_sum = torch.zeros(n, dtype=torch.float64)
    dist_sum.scatter_add_(0, which.cpu(), nearest.detach().cpu().to(torch.float64))
    counts = torch.tensor(nearest_counts, dtype=torch.float64).clamp_min(1)
    mean_dist = dist_sum / counts
    empty = torch.tensor(nearest_counts) == 0
    mean_dist = mean_dist.masked_fill(empty, float("nan"))
    return {
        "hq_counts": [int(c) for c in hq_counts],
        "nearest_counts": [int(c) for c in nearest_counts],
        "modes": int(sum(c > 0 for c in hq_counts)),
        "hq": _r(hq.float().mean()),
        "mean_dist": [None if empty[k] else _r(mean_dist[k]) for k in range(n)],
        "which": which,
        "nearest": nearest,
    }


def _by_mode(values: torch.Tensor, which: torch.Tensor, n: int) -> list:
    flat = values.detach().flatten().to(torch.float64).cpu()
    which = which.cpu()
    sums = torch.zeros(n, dtype=torch.float64)
    sums.scatter_add_(0, which, flat)
    counts = torch.bincount(which, minlength=n)
    out = []
    for k in range(n):
        out.append(None if int(counts[k]) == 0 else _r(sums[k] / counts[k]))
    return out


def _rms(params) -> float | None:
    total = None
    n = 0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        square = grad.square().sum()
        total = square if total is None else total + square
        n += grad.numel()
    if n == 0 or total is None:
        return None
    return _r((total / n).sqrt())


def _amax(params) -> float | None:
    peak = None
    for param in params:
        if param.grad is None:
            continue
        value = param.grad.detach().abs().max()
        peak = value if peak is None else torch.maximum(peak, value)
    return None if peak is None else _r(peak)


def observe_critic(role: str, real, fake, real_logits, fake_logits, means, sigma) -> None:
    """Log mode mass and relativistic logits for one D or G pairing."""
    if not enabled():
        return
    with torch.no_grad():
        fake_m = assign(fake, means, sigma)
        real_m = assign(real, means, sigma)
        real_l = real_logits.detach().flatten()
        fake_l = fake_logits.detach().flatten()
        margin = real_l - fake_l
        n = int(means.shape[0])
        radius = fake.detach().norm(dim=1)
        note(**{
            f"{role}_fake_modes": fake_m["modes"],
            f"{role}_fake_hq": fake_m["hq"],
            f"{role}_fake_hq_counts": fake_m["hq_counts"],
            f"{role}_fake_nearest_counts": fake_m["nearest_counts"],
            f"{role}_fake_mean_dist": fake_m["mean_dist"],
            f"{role}_real_counts": real_m["nearest_counts"],
            f"{role}_logit_real": _r(real_l.mean()),
            f"{role}_logit_fake": _r(fake_l.mean()),
            f"{role}_logit_real_abs_p99": _r(real_l.abs().quantile(0.99)),
            f"{role}_logit_fake_abs_p99": _r(fake_l.abs().quantile(0.99)),
            f"{role}_paired_margin": _r(margin.mean()),
            f"{role}_paired_margin_std": _r(margin.std(unbiased=False)),
            f"{role}_fake_wins": _r((fake_l > real_l).float().mean()),
            f"{role}_logit_real_by_mode": _by_mode(real_l, real_m["which"], n),
            f"{role}_logit_fake_by_mode": _by_mode(fake_l, fake_m["which"], n),
            f"{role}_radius_mean": _r(radius.mean()),
            f"{role}_radius_std": _r(radius.std(unbiased=False)),
            f"{role}_spread": _r(fake.detach().std(unbiased=False)),
        })


def grad_report(role: str, optimizer, prior) -> None:
    """Gradient RMS after backward and before the optimizer step."""
    if not enabled():
        return
    prior_ids = {id(param) for param in prior.parameters()}
    network, table = [], []
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                (table if id(param) in prior_ids else network).append(param)
        payload = {
            f"{role}_grad_rms": _rms(network),
            f"{role}_grad_max": _amax(network),
        }
        if table:
            payload[f"{role}_prior_grad_rms"] = _rms(table)
            payload[f"{role}_prior_grad_max"] = _amax(table)
            row_norms = []
            for param in table:
                if param.grad is None:
                    continue
                row_norms.extend(_r(v) for v in param.grad.detach().norm(dim=-1).tolist())
            if len(row_norms) <= 64:
                payload[f"{role}_prior_row_grad"] = row_norms
        note(**payload)


def inner_generator(generator: nn.Module) -> nn.Module:
    model = getattr(generator, "model", None)
    return model if isinstance(model, nn.Module) else generator


def particle_support(generator, prior, means, sigma) -> dict:
    """Noise-free images of every particle. Does not touch a noise generator."""
    with torch.no_grad():
        points = inner_generator(generator)(prior.z).detach()
        assigned = assign(points, means, sigma)
        n = int(points.shape[0])
        payload = {
            "support_modes": assigned["modes"],
            "support_hq": assigned["hq"],
            "support_hq_counts": assigned["hq_counts"],
            "support_nearest_counts": assigned["nearest_counts"],
            "support_mean_dist": assigned["mean_dist"],
            "support_missing": [k for k, count in enumerate(assigned["hq_counts"]) if count == 0],
        }
        if n <= 64:
            which = assigned["which"].tolist()
            nearest = assigned["nearest"].tolist()
            payload["particle_mode"] = [int(k) for k in which]
            payload["particle_dist"] = [_r(v) for v in nearest]
            payload["particle_xy"] = [[_r(x), _r(y)] for x, y in points.tolist()]
        return payload


def init_fingerprint(generator, critic, prior) -> dict:
    weights = [param.detach() for param in generator.parameters()]
    critic_weights = [param.detach() for param in critic.parameters()]
    g0 = weights[0] if weights else prior.z
    d0 = critic_weights[0] if critic_weights else prior.z
    return {
        "n_particles": int(prior.z.shape[0]),
        "z_dim": int(prior.z.shape[1]),
        "init_z_sha": _sha(prior.z),
        "init_g0_sha": _sha(g0),
        "init_d0_sha": _sha(d0),
    }


def _group_lrs(optimizer) -> list:
    return [_r(group["lr"], 8) for group in optimizer.param_groups]


def flush_step(step: int, extra: dict | None = None) -> None:
    if not enabled():
        return
    row = {"step": int(step)}
    if extra:
        row.update(extra)
    row.update(_STASH)
    _STASH.clear()
    if row.get("d_grad_rms") and row.get("g_grad_rms"):
        row["grad_ratio_d_over_g"] = _r(row["d_grad_rms"] / row["g_grad_rms"])
    global _FILE
    if _FILE is None:
        path = Path(os.environ["K3P_DIAG_TRAJ"])
        path.parent.mkdir(parents=True, exist_ok=True)
        _FILE = path.open("a", buffering=1)
    _FILE.write(json.dumps(row, separators=(",", ":")) + "\n")
    if step == 0 or step % 50 == 0:
        brief = {
            "diag": step,
            "support_modes": row.get("support_modes"),
            "support_hq": row.get("support_hq"),
            "missing": row.get("support_missing"),
            "s": row.get("s"),
            "prox": row.get("prox"),
            "phase": row.get("phase"),
            "margin": row.get("d_paired_margin"),
            "d_g": row.get("grad_ratio_d_over_g"),
            "a2": row.get("a2_scoped"),
            "ema_pull": row.get("ema_pull_rms"),
        }
        print(json.dumps(brief), flush=True)
