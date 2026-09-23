"""Numerical host extracted from HyperGAN/conceptmod commit 5571213.

Training/data/evaluation logic retained; config-identity refusals removed.
See ../SOURCE.md and ../LICENSE. Candidate settings are supplied by baseline.py.
"""

from __future__ import annotations


import json


from typing import Callable


from ..observation import checkpoint, schedule_optimizer

import torch


from torch import nn


from particlegan import GANLoss, ParticlePrior, ParticleRegularizer


from particlegan.grad_regularizers import GradRegularizer


from ..trajectory import (
    PROTOCOL as TRAJECTORY_PROTOCOL,
    PAIRINGS,
    PASS_IDENTITY_MSE,
    identity_mse,
    pairing_index,
    trajectories,
)


SLOW_IMPACT_MAX = 0.10


LAND_TOL = 0.25


SUCCESS_MIN = 1.0


RESIDUAL_WEIGHT = 1.0


LOG_EVERY = 100


def _xy(arc: torch.Tensor) -> torch.Tensor:
    frames = int(PROTOCOL["frames"])
    if arc.shape[-1] != frames * 2:
        raise ValueError(f"expected arc dim {frames * 2}, got {arc.shape[-1]}")
    return arc.reshape(arc.shape[0], frames, 2)


def endpoints(arc: torch.Tensor) -> torch.Tensor:
    """Last (x, y) of each packed arc."""
    return _xy(arc)[:, -1]


def impact(arc: torch.Tensor) -> torch.Tensor:
    """Chord of the last step. This is the touchdown speed proxy."""
    xy = _xy(arc)
    return (xy[:, -1] - xy[:, -2]).norm(dim=-1)


def both_land_mask(slow: torch.Tensor, fast: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows whose slow teacher and paired fast teacher both succeed here.

    The slow teacher must make a gentle touchdown. The paired fast teacher
    must finish on this seed's pad. Only the same-seed fast arc does that.
    """
    if index.shape != (slow.shape[0],):
        raise ValueError("pairing index must be one entry per identity")
    slow_lands = impact(slow) <= SLOW_IMPACT_MAX
    own_pad = endpoints(fast)
    paired_pad = own_pad[index]
    fast_lands = (paired_pad - own_pad).norm(dim=-1) <= LAND_TOL
    return slow_lands & fast_lands


def landing_stats(pred: torch.Tensor, fast: torch.Tensor) -> dict:
    """Own-pad success and wrong-pad crashes for a predicted fast arc."""
    pad = endpoints(fast)
    end = endpoints(pred)
    dist = (end - pad).norm(dim=-1)
    landed = dist <= LAND_TOL
    nearest = torch.cdist(end, pad).argmin(dim=1)
    own = torch.arange(end.shape[0])
    wrong_pad = nearest != own
    return {
        "success_rate": float(landed.float().mean()),
        "wrong_pad_rate": float(wrong_pad.float().mean()),
        "endpoint_l2": float(dist.mean()),
    }


def passed(mse: float, success_rate: float, wrong_pad_rate: float) -> bool:
    """Identity, every seed on its own pad, and no wrong-pad touchdown."""
    return (
        mse <= PASS_IDENTITY_MSE
        and success_rate >= SUCCESS_MIN
        and wrong_pad_rate == 0.0
    )


class ResidualHead(nn.Module):
    """``slow + head(slow, z)``. A zero head leaves the slow arc untouched."""

    def __init__(self, slow_dim: int, z_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, slow_dim),
        )

    def delta(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return slow + self.delta(slow, z)


class _Critic(nn.Module):
    """Scores a slow/fast pair. This toy owns the critic."""

    def __init__(self, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, fast), dim=-1)).squeeze(-1)


class _FastView(nn.Module):
    """Gradient penalty sees the fast arc; the slow arc stays conditioning."""

    def __init__(self, critic: _Critic) -> None:
        super().__init__()
        self.critic = critic

    def forward(self, fast: torch.Tensor) -> torch.Tensor:
        return self.critic(self.slow, fast).unsqueeze(-1)


def _cover(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Demo cover: each true fast arc must sit near some generated arc."""
    return torch.cdist(real, fake).min(dim=1).values.square().mean()


def _emit(record: dict, echo: bool, log: Callable[[dict], None] | None) -> None:
    if log is not None:
        log(record)
    if echo:
        print(json.dumps(record), flush=True)


def train(*, pairing: str = "shared", echo: bool = False,
          log: Callable[[dict], None] | None = None) -> dict:
    """Train the residual head; score the resulting predictions."""
    torch.set_num_threads(1)
    torch.manual_seed(PROTOCOL["seed"])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    mask = both_land_mask(slow, fast, index)
    hidden = PROTOCOL["critic_hidden"]
    head = ResidualHead(slow.shape[1], PROTOCOL["z_dim"], hidden)
    critic = _Critic(slow.shape[1] + fast.shape[1], hidden)
    view = _FastView(critic)
    prior = ParticlePrior(
        PROTOCOL["n_particles"], PROTOCOL["z_dim"], init_std=0.1,
        generator=torch.Generator().manual_seed(PROTOCOL["seed"]),
    )
    gan = GANLoss(PROTOCOL["loss_type"], PROTOCOL["gan_mode"])
    regularizer = GradRegularizer(
        PROTOCOL["reg_arm"], PROTOCOL["reg_coeff"], kappa=PROTOCOL["reg_kappa"],
        norm=PROTOCOL["reg_norm"], lazy_k=PROTOCOL["reg_lazy"],
        target_anneal=PROTOCOL["target_anneal"],
    )
    spread = ParticleRegularizer(weight=PROTOCOL["vicreg_weight"])
    opt_g = torch.optim.Adam(
        list(head.parameters()) + list(prior.parameters()),
        lr=PROTOCOL["lr"], betas=(PROTOCOL["beta1"], PROTOCOL["beta2"]),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=PROTOCOL["lr"], betas=(PROTOCOL["beta1"], PROTOCOL["beta2"]),
    )
    _emit({
        "event": "config",
        "family": "residual_student",
        "pairing": pairing,
        "both_land_rows": int(mask.sum()),
        "fm_weight": PROTOCOL["fm_weight"],
        "cover_weight": PROTOCOL["cover_weight"],
        "reg_arm": regularizer.arm,
        "reg_coeff": regularizer.coeff,
        "reg_kappa": regularizer.kappa,
        "reg_norm": regularizer.norm,
        "n_particles": PROTOCOL["n_particles"],
        "particle_l2": PROTOCOL["particle_l2"],
        "vicreg_weight": PROTOCOL["vicreg_weight"],
        "residual_weight": RESIDUAL_WEIGHT,
        "land_tol": LAND_TOL,
        "slow_impact_max": SLOW_IMPACT_MAX,
        "steps": PROTOCOL["steps"],
        "seed": PROTOCOL["seed"],
        "pass_identity_mse": PASS_IDENTITY_MSE,
        "success_min": SUCCESS_MIN,
    }, echo, log)

    steps = PROTOCOL["steps"]
    both = int(mask.sum())
    for step in range(1, steps + 1):
        fake = head(slow, prior.z)
        opt_d.zero_grad(set_to_none=True)
        d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))
        view.slow = slow.detach()
        d_loss = d_loss + regularizer(view, paired, fake.detach(), step=step)
        d_loss.backward()
        schedule_optimizer(opt_d, step - 1)
        opt_d.step()

        flags = [p.requires_grad for p in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            fake = head(slow, prior.z)
            g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
            # Cover matches the true fast cloud (set coverage). It does not
            # retarget identity. The residual term does, and only on both-land
            # rows. fm_weight is 0: no feature-matching term is added.
            g_loss = g_loss + PROTOCOL["cover_weight"] * _cover(fake, fast)
            g_loss = g_loss + PROTOCOL["particle_l2"] * prior.z.square().mean()
            g_loss = g_loss + spread(prior.z)
            if both:
                residual = (fake[mask] - fast[mask]).pow(2).mean()
            else:
                residual = fake.new_zeros(())
            g_loss = g_loss + RESIDUAL_WEIGHT * residual
            g_loss.backward()
            schedule_optimizer(opt_g, step - 1)
            opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)
        def observe_student():
            with torch.no_grad():
                pred = head(slow, prior.z)
                return {"identity_mse": identity_mse(pred, fast), **landing_stats(pred, fast)}
        checkpoint(step, observe_student)

        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            with torch.no_grad():
                pred = head(slow, prior.z)
                mse = identity_mse(pred, fast)
                stats = landing_stats(pred, fast)
            _emit({
                "event": "step",
                "step": step,
                "pairing": pairing,
                "d_loss": float(d_loss.detach()),
                "g_loss": float(g_loss.detach()),
                "residual_mse": float(residual.detach()),
                "identity_mse": mse,
                "success_rate": stats["success_rate"],
                "wrong_pad_rate": stats["wrong_pad_rate"],
            }, echo, log)

    with torch.no_grad():
        pred = head(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
        stats = landing_stats(pred, fast)
    ok = passed(mse, stats["success_rate"], stats["wrong_pad_rate"])
    result = {
        "event": "done",
        "family": "residual_student",
        "pairing": pairing,
        "both_land_rows": both,
        "identity_mse": mse,
        "paired_target_mse": paired_mse,
        "success_rate": stats["success_rate"],
        "wrong_pad_rate": stats["wrong_pad_rate"],
        "endpoint_l2": stats["endpoint_l2"],
        "pass": ok,
        "pass_identity_mse": PASS_IDENTITY_MSE,
        "success_min": SUCCESS_MIN,
        "land_tol": LAND_TOL,
        "slow_impact_max": SLOW_IMPACT_MAX,
        "residual_weight": RESIDUAL_WEIGHT,
        "fm_weight": PROTOCOL["fm_weight"],
        "cover_weight": PROTOCOL["cover_weight"],
        "reg_arm": regularizer.arm,
        "reg_coeff": regularizer.coeff,
        "reg_kappa": regularizer.kappa,
        "reg_norm": regularizer.norm,
        "n_particles": prior.num_particles,
        "particle_l2": PROTOCOL["particle_l2"],
        "vicreg_weight": PROTOCOL["vicreg_weight"],
        "steps": steps,
        "seed": PROTOCOL["seed"],
    }
    _emit(result, echo, log)
    return result


PROTOCOL = dict(TRAJECTORY_PROTOCOL, loss_type="logistic", gan_mode="rp", reg_arm="b_cap", reg_coeff=1.0, reg_kappa=1.0, reg_norm="l2", reg_lazy=1, target_anneal="none", fm_weight=0.0)
