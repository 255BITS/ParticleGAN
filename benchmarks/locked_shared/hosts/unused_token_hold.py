"""Unused-token hold toy: an adapter moves the concept slot, not the unused one.

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the two-slot token-embedding student, the slot critic, the
unused-token hold loss, the hold/move metrics and the verdict. Everything else
(optimizers and their LR schedule, loss, critic penalty, noise, EMA,
observation logging) comes from the shipped recipe through
``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.unused_token_hold --log runs/toy-refactor/unused_token_hold.log
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, ToyProblem, main, run
from ..observation import checkpoint

UNUSED_HOLD_MIN = 0.85
CONCEPT_MOVE_MIN = 0.85
DIM = 2
N_SLOTS = 2
UNUSED = 0
CONCEPT = 1
N_ROWS = 8
CRITIC_HIDDEN = 64
STEPS = 200
HOLD_WEIGHT = 1.0
NEU = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
CONCEPT_DIR = torch.tensor([0.0, 1.0])


def hold_pairs(pairing: str) -> list[tuple[int, int]]:
    """Unused-token partners. Concept index is never the prediction slot.

    ``matched`` pins unused -> encode(neu) at the unused index.
    ``stranger`` pins unused -> the concept slot (the wrong neu partner).
    """
    if pairing == "matched":
        return [(UNUSED, UNUSED)]
    if pairing == "stranger":
        return [(UNUSED, CONCEPT)]
    raise ValueError(f"pairing must be 'matched' or 'stranger', got {pairing!r}")


def unused_hold_loss(pred: torch.Tensor, tgt: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Masked MSE of unused positions onto the partner embed.

    ``pred`` / ``tgt`` are ``(T, D)``. Empty pairs contribute 0 -- there is
    nothing to pin, which is the Anima fail-closed empty alignment.
    """
    if not pairs:
        return pred.reshape(-1)[:1].sum() * 0.0
    return F.mse_loss(pred[[i for i, _ in pairs]], tgt[[j for _, j in pairs]])


class SharedSlotStudent(nn.Module):
    """One residual added to every slot, plus a per-slot correction.

    Concept loss does not see the unused slot. The shared vector still
    moves it, unless the hold loss trains the unused correction to cancel.
    Both start at zero: the adapter-off identity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(DIM))
        self.slot = nn.Parameter(torch.zeros(N_SLOTS, DIM))
        self.register_buffer("neu", NEU.clone())

    def embeds(self, scale: float) -> torch.Tensor:
        # Scale 0 is the adapter-off identity (Anima UNI scale 0).
        return self.neu + float(scale) * (self.shared + self.slot)


# The zero residuals are the identity by construction, not a draw.
init.register(SharedSlotStudent, {"shared": init.KEEP, "slot": init.KEEP})


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


def score_student(student: SharedSlotStudent) -> dict[str, float]:
    """Unused hold and concept move at scale +1. Scale 0 is identity."""
    embeds = student.embeds(1.0).detach()
    unused, concept = embeds[UNUSED], embeds[CONCEPT]
    pin, origin = student.neu[UNUSED], student.neu[CONCEPT]
    dist = float((unused - pin).norm())
    target_norm = float(CONCEPT_DIR.norm())
    hold = 1.0 - min(1.0, dist / target_norm)
    delta = concept - origin
    delta_norm = float(delta.norm())
    cos = 0.0 if delta_norm <= 1e-8 else float(F.cosine_similarity(delta.unsqueeze(0), CONCEPT_DIR.unsqueeze(0)))
    mag = max(0.0, 1.0 - abs(delta_norm / target_norm - 1.0))
    scale0 = student.embeds(0.0).detach()
    return {
        "unused_hold": hold,
        "concept_move": max(0.0, cos) * mag,
        "unused_dist": dist,
        "concept_cos": cos,
        "concept_delta_norm": delta_norm,
        "scale0_err": float((scale0 - student.neu).norm()),
    }


def verdict(metrics: dict) -> str:
    """PASS: the unused slot holds and the concept slot moves onto the target."""
    ok = metrics["unused_hold"] >= UNUSED_HOLD_MIN and metrics["concept_move"] >= CONCEPT_MOVE_MIN
    return "PASS" if ok else "FAIL"


class UnusedTokenHold(ToyProblem):
    """Student-only generator side (no prior): the concept slot is the fake
    sample and the critic sees ``CONCEPT_DIR`` as real. ``hold_weight=0`` and
    ``pairing='stranger'`` are the named drift arms; ``fm_weight`` adds the
    host's critic feature-matching generator term."""

    name = "unused_token_hold"

    def __init__(self, *, hold_weight: float = HOLD_WEIGHT, pairing: str = "matched", fm_weight: float = 0.0):
        self.hold_weight, self.fm_weight = float(hold_weight), float(fm_weight)
        self.pairs = hold_pairs(pairing)

    def recipe(self):
        return get_recipe(batch_size=N_ROWS, total_steps=STEPS)

    def networks(self, recipe, seed):
        student = init.deterministic_orthogonal_(SharedSlotStudent(), seed=seed)
        critic = init.deterministic_orthogonal_(SlotCritic(), seed=seed + 1)
        return Networks(generator=student, critics=critic, prior=None)

    def real(self, n, stream):
        return CONCEPT_DIR.unsqueeze(0).expand(n, -1)

    def fake(self, nets, n, stream, real):
        return nets.generator.embeds(1.0)[CONCEPT].unsqueeze(0).expand(n, -1)

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        terms = {}
        if self.hold_weight != 0:
            student = nets.generator
            terms["unused_hold"] = self.hold_weight * unused_hold_loss(student.embeds(1.0), student.neu, self.pairs)
        if self.fm_weight != 0:
            critic = nets.critics
            gap = critic.features(real.x).detach().mean(0) - critic.features(fake.x).mean(0)
            terms["feature_match"] = self.fm_weight * gap.pow(2).mean()
        return terms

    def metrics(self, model):
        return score_student(model.nets.generator)

    def verdict(self, metrics):
        return verdict(metrics)


def train_unused_token_hold(problem: UnusedTokenHold | None = None, *, seed: int = 0, diagnostics: bool = False,
                            log=None, steps: int | None = None, recipe=None) -> dict:
    """Train on the shared runner; the EMA row with its verdict plus ``live``
    (and ``live_curve``/``hold`` when ``diagnostics`` is set).

    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    problem = UnusedTokenHold() if problem is None else problem
    result = run(problem, recipe=recipe, seed=seed, steps=steps, observe_every=50 if diagnostics else None,
                 log=log, observer=checkpoint)
    final = {**result["ema"], "step": result["steps"], "seed": seed, "live": result["live"]}
    if diagnostics:
        final["live_curve"] = result["curve"]
        final["hold"] = result["hold"]
    return final


if __name__ == "__main__":
    raise SystemExit(main(UnusedTokenHold()))
