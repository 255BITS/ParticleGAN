#!/usr/bin/env python
"""100 Gaussians: can the shipped recipe cover a dense 10x10 grid of sharp modes?

Problem only. The data is a 100-Gaussian mixture on a 10x10 grid in R^2
(sigma 0.03); G is a small MLP z -> x; D is an MLP with Fourier input
features (so it can resolve sigma=0.03 modes from step 1). Everything else --
the recipe-built optimizers (which own the LR schedule), the loss, the
critic penalty, the prior, critic input / generator output noise, EMA and
observation logging -- comes from ``particlegan.get_recipe()`` through the
shared ``benchmarks.toy_runner``::

    python examples/100gaussians.py --log runs/toy-refactor/100gaussians.log
    python examples/100gaussians.py --prior frozen_gaussian   # matched controls

``--prior`` picks the latent control (``particles`` is the recipe's learned
table; ``mog``, ``frozen_gaussian``, ``fresh_gaussian``); every other option is
the runner's shared CLI (``--steps``, ``--seed``, ``--device``, ``--log``,
``--shift-step``). The log is one JSON line per observation (``tail -f``).

PASS: all 100 modes hold at least 10 high-quality samples (within 3 sigma of
the center) and at least 90% of 20,000 samples are high quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Allow `python examples/100gaussians.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from particlegan import get_recipe, init  # noqa: E402
from particlegan.particle_prior import PRIOR_KINDS, canonical_prior_kind, make_prior  # noqa: E402
from benchmarks.toy_runner import Networks, ToyProblem, main  # noqa: E402
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians  # noqa: E402

STD = 0.03
N_EVAL = 20_000
MIN_COUNT = 10
PASS_MODES = 100
PASS_HQ = 0.90
_CENTERS = torch.cartesian_prod(torch.arange(10.0) - 4.5, torch.arange(10.0) - 4.5)


def coverage(x: torch.Tensor) -> dict:
    """Modes with >= MIN_COUNT high-quality samples, and the HQ fraction."""
    distance, nearest = torch.cdist(x, _CENTERS.to(x)).min(dim=1)
    hq = distance <= 3 * STD
    counts = torch.bincount(nearest[hq], minlength=100)
    return {"modes": int((counts >= MIN_COUNT).sum()), "hq": float(hq.float().mean())}


def zero_biases(module: nn.Module) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return module


class Gaussians100(ToyProblem):
    """The 100-Gaussian grid under one latent control.

    ``prior_kind``: ``particles`` (the recipe's learned table), ``mog``
    (learned MoG means, with ``sigma_rel`` / ``standardize``), ``frozen_gaussian``
    (a fixed random table) or ``fresh_gaussian`` (fresh N(0, I) draws).
    ``distribution`` adds the grid distribution diagnostics (TV, SW1, per-mode
    width/covariance ratios) to every measurement.
    """

    def __init__(self, prior_kind: str = "particles", *, fourier: int = 2, sigma_rel: float = 0.0,
                 standardize: bool = False, distribution: bool = False):
        self.prior_kind = canonical_prior_kind(prior_kind)
        self.fourier, self.sigma_rel, self.standardize = fourier, sigma_rel, standardize
        self.distribution = distribution
        self.name = "100gaussians" if self.prior_kind == "particles" else f"100gaussians_{self.prior_kind}"

    def recipe(self):
        return get_recipe()  # the task shape (z_dim 2, 20k particles, batch 2048, 7k steps) is the default

    def networks(self, recipe, seed):
        # The prior is drawn first: the frozen / fresh-Gaussian controls keep
        # that random draw; learned tables are replaced by R2 points below.
        kind = self.prior_kind
        if kind == "mog":
            prior = recipe.make_prior(prior_kind="mog", sigma_rel=self.sigma_rel, standardize=self.standardize)
        elif kind == "fresh_gaussian":
            prior = make_prior(kind, num_particles=recipe.num_particles, z_dim=recipe.z_dim)
        else:
            prior = recipe.make_prior(learnable=kind == "particles")
        if kind in ("particles", "mog"):
            init.deterministic_orthogonal_(prior)
        # Orthogonal weights with the examples' seeds (G=0, D=1; the run seed selects
        # the data, latent and noise streams). Biases start at zero:
        # deterministic_orthogonal_ keeps zero vectors as set on purpose.
        generator = init.deterministic_orthogonal_(zero_biases(SimpleMLPGenerator(z_dim=recipe.z_dim)), seed=0)
        critic = init.deterministic_orthogonal_(zero_biases(SimpleMLPDiscriminator(in_dim=2, fourier=self.fourier)), seed=1)
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        return sample_100gaussians(n, stream.device, generator=stream)

    def metrics(self, model):
        x = model.sample(N_EVAL).x
        row = coverage(x)
        if self.distribution:
            from lib.denoising_toy import GaussianGrid, grid_metrics
            from lib.toy_metrics import per_mode_moments
            toy = GaussianGrid(device=x.device, std=STD, classes=1)
            labels = torch.zeros(len(x), dtype=torch.long, device=x.device)
            real = toy.sample(labels, model.stream)
            row["distribution"] = {**grid_metrics(x, labels, toy, real), **per_mode_moments(x, min_count=20, std=STD)}
        return row

    def verdict(self, metrics):
        return "PASS" if metrics["modes"] >= PASS_MODES and metrics["hq"] >= PASS_HQ else "FAIL"


def cli(default_prior: str = "particles", argv=None) -> int:
    """``--prior`` selects the problem; everything else is the shared runner CLI."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prior", choices=PRIOR_KINDS, default=default_prior)
    args, rest = parser.parse_known_args(argv)
    return main(Gaussians100(args.prior), rest)


if __name__ == "__main__":
    raise SystemExit(cli())
