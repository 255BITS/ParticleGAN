"""AE+GAN hold toy: does an adversarial AE keep reconstruction and hold both anchors?

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the two-anchor data, the host MLPs (encoder, decoder = G, critic),
the host's reconstruction / particle-L2 / anchor-cover terms, the
reconstruction and hold metrics and the verdict. Everything else (optimizers
and their LR schedule, loss, critic penalty, MoG prior, noise, EMA,
observation logging) comes from the shipped ``ae_gan`` recipe through
``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.ae_gan_hold --log runs/toy-refactor/ae_gan_hold.log
"""

from __future__ import annotations

import torch
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, ToyProblem, main, run
from ..observation import checkpoint

DEMO_COVER = 1.5
PARTICLE_L2 = 0.02
N_PARTICLES = 12
STEPS = 250
BATCH = 64
EVAL_N = 1024
DATA_STD = 0.05
ANCHORS = ((-1.5, 0.0), (1.5, 0.0))
RECON_MAX = 0.05
HOLD_MAX = 0.35


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[1](self.net[0](x))


def _anchors(like: torch.Tensor | None = None) -> torch.Tensor:
    anchors = torch.tensor(ANCHORS, dtype=torch.float32)
    return anchors if like is None else anchors.to(like)


def sample_data(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    choice = torch.randint(0, len(ANCHORS), (n,), generator=generator)
    return _anchors()[choice] + DATA_STD * torch.randn(n, 2, generator=generator)


def _hold_distance(fake: torch.Tensor) -> float:
    """Mean over anchors of the closest generated sample (unconditional hold)."""
    return float(torch.cdist(_anchors(fake), fake).min(dim=1).values.mean())


def verdict(row: dict) -> str:
    return "PASS" if row["recon_mse"] <= RECON_MAX and row["hold"] <= HOLD_MAX else "FAIL"


class AEGanHold(ToyProblem):
    """Two anchors, AE (encoder E, decoder as G) + MLP critic on a MoG prior.

    ``particle_l2`` (L2 on the prior means), ``cover_weight`` (anchor cover on
    generated samples) and ``fm_weight`` (critic feature matching) are the
    host's own generator-side terms; reconstruction uses the recipe's
    ``encode`` and ``reconstruction_weight``.
    """

    name = "ae_gan_hold"

    def __init__(self, *, particle_l2: float = PARTICLE_L2, cover_weight: float = DEMO_COVER,
                 fm_weight: float = 0.0):
        self.particle_l2, self.cover_weight, self.fm_weight = float(particle_l2), float(cover_weight), float(fm_weight)

    def recipe(self):
        return get_recipe("ae_gan", z_dim=2, num_particles=N_PARTICLES, batch_size=BATCH, total_steps=STEPS)

    def networks(self, recipe, seed):
        self._recipe = recipe  # routes encoder queries (recipe.encode) and weights reconstruction
        encoder = init.deterministic_orthogonal_(MLP(2, 4), seed=seed)
        decoder = init.deterministic_orthogonal_(MLP(2, 2), seed=seed + 1)
        critic = init.deterministic_orthogonal_(MLP(2, 1), seed=seed + 2)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        return Networks(generator=decoder, critics=critic, prior=prior, encoder=encoder)

    def real(self, n, stream):
        return sample_data(n, stream)

    def _reconstruct(self, nets, x):
        query, offset = nets.encoder(x).chunk(2, dim=1)
        encoded = self._recipe.encode(query, nets.prior, offset=offset)
        return encoded, nets.generator(encoded.codes[:, 0])

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        encoded, reconstructed = self._reconstruct(nets, real.x)
        terms = {
            "reconstruction": self._recipe.reconstruction_weight
            * encoded.reconstruction_loss(reconstructed[:, None], real.x),
            "particle_l2": self.particle_l2 * nets.prior.z.square().mean(),
            "cover": self.cover_weight * torch.cdist(_anchors(fake.x), fake.x).min(dim=1).values.mean(),
        }
        if self.fm_weight > 0:
            critic = nets.critics
            gap = critic.features(real.x).detach().mean(0) - critic.features(fake.x).mean(0)
            terms["feature_matching"] = self.fm_weight * gap.square().mean()
        return terms

    def metrics(self, model):
        data = self.real(EVAL_N, model.stream)
        _, recon = self._reconstruct(model.nets, data)
        return {"recon_mse": float((recon - data).square().mean()),
                "hold": _hold_distance(model.sample(EVAL_N).x)}

    def verdict(self, metrics):
        return verdict(metrics)


def train_ae_gan_hold(problem: AEGanHold | None = None, *, seed: int = 0, steps: int | None = None,
                      recipe=None, log=None) -> dict:
    """Train on the shared runner; the EMA row (top level) plus ``live``, ``curve`` and ``hold_summary``.

    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    result = run(AEGanHold() if problem is None else problem, recipe=recipe, seed=seed, steps=steps,
                 log=log, observer=checkpoint)
    return {**result["ema"], "live": result["live"], "curve": result["curve"],
            "hold_summary": result["hold"], "steps": result["steps"], "seed": seed}


if __name__ == "__main__":
    raise SystemExit(main(AEGanHold()))
