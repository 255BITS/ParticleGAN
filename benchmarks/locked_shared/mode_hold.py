"""Ring mode-hold toy: can a 12-particle prior hold all 8 modes of a ring?

Problem only, from HyperGAN/conceptmod at 5571213 (see SOURCE.md and LICENSE):
the 8-mode ring data, the host MLPs, the mode/HQ metrics and the verdict.
Everything else (optimizers and their LR schedule, loss, critic penalty,
prior, noise, EMA, observation logging) comes from the shipped recipe through
``benchmarks.toy_runner``. This module is the reference example of that
runner::

    python -m benchmarks.locked_shared.mode_hold --log runs/toy-refactor/mode_hold.log
"""

from __future__ import annotations

import math

import torch

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, main, run
from .mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from .observation import checkpoint

N_MODES = 8
RADIUS = 3.0
SIGMA = 0.07
Z_DIM = 4
N_PARTICLES = 12
HIDDEN = 96
N_HIDDEN = 3
FOURIER = 3
BATCH = 128
STEPS = 1200
EVAL_N = 4096
PASS_MODES = 7
PASS_HQ = 0.90
COLLAPSE_MODES = 2


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


def verdict(row: dict) -> str:
    """PASS holds the ring. FAIL is mode collapse. Anything between is inconclusive."""
    if row["modes"] >= PASS_MODES and row["hq"] >= PASS_HQ:
        return "PASS"
    if row["modes"] <= COLLAPSE_MODES:
        return "FAIL"
    return "INCONCLUSIVE"


class ModeHold(ToyProblem):
    """The 8-mode ring. ``fm_weight`` adds the host's feature-mean generator term
    (the "FM-on drift" arm); ``detailed`` adds per-mode diagnostics and the
    deterministic support of the particle table to the metrics."""

    name = "mode_hold"

    def __init__(self, *, fm_weight: float = 0.0, detailed: bool = False):
        self.fm_weight, self.detailed = float(fm_weight), detailed
        self.means = ring_means()

    def recipe(self):
        return get_recipe(z_dim=Z_DIM, num_particles=N_PARTICLES, batch_size=BATCH, total_steps=STEPS)

    def networks(self, recipe, seed):
        generator = init.deterministic_orthogonal_(SimpleMLPGenerator(Z_DIM, HIDDEN, N_HIDDEN, 2), seed=seed)
        # Host critic shape from the 100-Gaussians toy; Fourier width is the sharp-D stress.
        critic = init.deterministic_orthogonal_(SimpleMLPDiscriminator(2, HIDDEN, N_HIDDEN, FOURIER), seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        return sample_ring(self.means, n, SIGMA, stream)

    def losses(self, role, nets, real, fake):
        if role != "generator" or self.fm_weight == 0:
            return {}
        return {"feature_mean": self.fm_weight * (fake.x.mean(0) - real.x.mean(0)).pow(2).sum()}

    def metrics(self, model):
        row = diversity(model.sample(EVAL_N).x, self.means, detailed=self.detailed)
        if self.detailed:
            row["support"] = diversity(model.nets.generator(model.nets.prior.z), self.means, detailed=True)
        return row

    def verdict(self, metrics):
        return verdict(metrics)

    def shift(self):
        """Rotate the ring in place by half a mode spacing."""
        angle = math.pi / N_MODES
        rotation = torch.tensor([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        self.means.copy_(self.means @ rotation)


def train_mode_hold(problem: ModeHold | None = None, *, seed: int = 0, diagnostics: bool = False,
                    log=None, steps: int | None = None, recipe=None) -> dict:
    """Train the ring on the shared runner; the EMA row with its verdict, plus
    ``live``/``live_curve`` when ``diagnostics`` is set.

    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    problem = ModeHold(detailed=diagnostics) if problem is None else problem
    result = run(problem, recipe=recipe, seed=seed, steps=steps, observe_every=50 if diagnostics else None,
                 log=log, observer=checkpoint)
    final = {**result["ema"], "step": result["steps"], "seed": seed}
    if diagnostics:
        final["live"] = result["live"]
        final["live_curve"] = result["curve"]
        final["hold"] = result["hold"]
    return final


if __name__ == "__main__":
    raise SystemExit(main(ModeHold()))
