"""Numerical host extracted from HyperGAN/conceptmod commit 5571213.

Training/data/evaluation logic retained; config-identity refusals removed.
See ../SOURCE.md and ../LICENSE. Candidate settings are supplied by baseline.py.
"""

from __future__ import annotations


from dataclasses import dataclass, replace


from ..observation import checkpoint, schedule_optimizer

import torch


import torch.nn.functional as F


from torch import nn


from particlegan import GANLoss, GradientPenalty


UNUSED_HOLD_MIN = 0.85


CONCEPT_MOVE_MIN = 0.85


DIM = 2


N_SLOTS = 2


UNUSED = 0


CONCEPT = 1


N_ROWS = 8


LR = 5e-3


BETAS = (0.0, 0.99)


CRITIC_HIDDEN = 64


STEPS = 200


HOLD_WEIGHT = 1.0


DEMO_COVER = 1.5


NEU = torch.tensor([[1.0, 0.0], [0.0, 0.0]])


CONCEPT_DIR = torch.tensor([0.0, 1.0])


@dataclass(frozen=True)
class UnusedHoldRecipe:
    """locked_shared card plus the unused-token hold switch.

    ``steps`` and ``seed`` are budget. ``hold_weight=0`` and
    ``pairing='stranger'`` are named drifts, not silent aliases.
    """

    name: str = "locked_shared"
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    pairing: str = "matched"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = DEMO_COVER
    n_particles: int = 12
    particle_l2: float = 0.02
    hold_weight: float = HOLD_WEIGHT
    steps: int = STEPS
    seed: int = 0

    def replace(self, **overrides) -> "UnusedHoldRecipe":
        return replace(self, **overrides)


def hold_pairs(pairing: str) -> list[tuple[int, int]]:
    """Unused-token partners. Concept index is never the prediction slot.

    ``matched`` pins unused → encode(neu) at the unused index.
    ``stranger`` pins unused → the concept slot (the wrong neu partner).
    """
    if pairing == "matched":
        return [(UNUSED, UNUSED)]
    if pairing == "stranger":
        return [(UNUSED, CONCEPT)]
    raise ValueError(f"pairing must be 'matched' or 'stranger', got {pairing!r}")


def unused_hold_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """Masked MSE of unused positions onto the partner embed.

    ``pred`` / ``tgt`` are ``(T, D)``. Empty pairs contribute 0 — there is
    nothing to pin, which is the Anima fail-closed empty alignment.
    """
    if not pairs:
        return pred.reshape(-1)[:1].sum() * 0.0
    pred_idx = [i for i, _ in pairs]
    tgt_idx = [j for _, j in pairs]
    return F.mse_loss(pred[pred_idx], tgt[tgt_idx])


class SharedSlotStudent(nn.Module):
    """One residual added to every slot, plus a per-slot correction.

    Concept loss does not see the unused slot. The shared vector still
    moves it, unless the hold loss trains the unused correction to cancel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(DIM))
        self.slot = nn.Parameter(torch.zeros(N_SLOTS, DIM))
        self.register_buffer("neu", NEU.clone())

    def embeds(self, scale: float) -> torch.Tensor:
        # Scale 0 is the adapter-off identity (Anima UNI scale 0).
        return self.neu + float(scale) * (self.shared + self.slot)


class SlotCritic(nn.Module):
    """Two-layer LeakyReLU MLP on the concept slot. Hidden 64, locked card."""

    def __init__(self, hidden: int = CRITIC_HIDDEN) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(DIM, hidden), nn.LeakyReLU(0.2))
        self.out = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(z)).squeeze(-1)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return self.hidden(z)


def _batch(vector: torch.Tensor, rows: int = N_ROWS) -> torch.Tensor:
    return vector.detach().unsqueeze(0).expand(rows, -1)


def score_student(student: SharedSlotStudent) -> dict[str, float]:
    """Unused hold and concept move at scale +1. Scale 0 is identity."""
    embeds = student.embeds(1.0).detach()
    unused = embeds[UNUSED]
    concept = embeds[CONCEPT]
    pin = student.neu[UNUSED]
    origin = student.neu[CONCEPT]
    dist = float((unused - pin).norm())
    target_norm = float(CONCEPT_DIR.norm())
    hold = 1.0 - min(1.0, dist / target_norm)
    delta = concept - origin
    delta_norm = float(delta.norm())
    if delta_norm <= 1e-8:
        cos = 0.0
    else:
        cos = float(F.cosine_similarity(delta.unsqueeze(0), CONCEPT_DIR.unsqueeze(0)))
    mag = max(0.0, 1.0 - abs(delta_norm / target_norm - 1.0))
    move = max(0.0, cos) * mag
    scale0 = student.embeds(0.0).detach()
    return {
        "unused_hold": hold,
        "concept_move": move,
        "unused_dist": dist,
        "concept_cos": cos,
        "concept_delta_norm": delta_norm,
        "scale0_err": float((scale0 - student.neu).norm()),
    }


def _log(name: str, step: int, metrics: dict[str, float], verdict: str | None = None) -> None:
    tail = f" verdict={verdict}" if verdict else ""
    print(
        f"unused-token arm={name} step={step} "
        f"hold={metrics['unused_hold']:.4f} concept={metrics['concept_move']:.4f} "
        f"unused_dist={metrics['unused_dist']:.4f}{tail}",
        flush=True,
    )


def _make_regularizer(recipe: UnusedHoldRecipe) -> GradientPenalty:
    return GradientPenalty(
        arm=recipe.reg_arm,
        coeff=recipe.reg_coeff,
        kappa=recipe.reg_kappa,
        norm=recipe.reg_norm,
        lazy_k=recipe.reg_lazy,
        target_anneal=recipe.target_anneal,
    )


def train(recipe: UnusedHoldRecipe, regularizer: GradientPenalty | None = None,
          *, noise_policy=None) -> dict:
    """Fit one arm. Prints a tailable line at the checkpoints."""
    torch.manual_seed(int(recipe.seed))
    student = SharedSlotStudent()
    critic = SlotCritic()
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input
        critic = wrap_input(critic, noise_policy)
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    reg = regularizer if regularizer is not None else _make_regularizer(recipe)
    if noise_policy is not None:
        noise_policy.register_generator_base(student)
    g_parameters = list(student.parameters()) + (
        noise_policy.scale_parameters() if noise_policy is not None else []
    )
    opt_g = torch.optim.Adam(g_parameters, lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    real = _batch(CONCEPT_DIR)
    pairs = hold_pairs(recipe.pairing)
    bcap_applied = 0
    for step in range(int(recipe.steps)):
        if noise_policy is not None:
            noise_policy.set_step(step)
        fake = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1).detach()
        if noise_policy is not None:
            fake = noise_policy.output(fake, generator_step=False)
        opt_d.zero_grad(set_to_none=True)
        penalty, stats = reg.penalty(critic, real, fake, step=step + 1)
        if stats.get("applied"):
            bcap_applied += 1
        d_loss = gan.d_loss(critic(real), critic(fake)) + penalty
        d_loss.backward()
        schedule_optimizer(opt_d, step)
        opt_d.step()

        critic.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        fake_g = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1)
        if noise_policy is not None:
            fake_g = noise_policy.output(fake_g, generator_step=True)
        g_loss = gan.g_loss(critic(fake_g), critic(real).detach())
        # Demo cover and the n=12 cloud are recorded on the card and are
        # not added here. The train pin is the unused-token hold.
        if float(recipe.fm_weight) != 0.0:
            real_feat = critic.features(real).detach().mean(0)
            fake_feat = critic.features(fake_g).mean(0)
            g_loss = g_loss + float(recipe.fm_weight) * (real_feat - fake_feat).pow(2).mean()
        loss = g_loss
        if float(recipe.hold_weight) != 0.0:
            embeds = student.embeds(1.0)
            loss = loss + float(recipe.hold_weight) * unused_hold_loss(embeds, student.neu, pairs)
        loss.backward()
        critic.requires_grad_(True)
        schedule_optimizer(opt_g, step)
        opt_g.step()
        checkpoint(step + 1, lambda: score_student(student))

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(recipe.steps):
            _log(recipe.name, step + 1, score_student(student))

    metrics = score_student(student)
    return metrics
