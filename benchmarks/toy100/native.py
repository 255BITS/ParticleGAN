"""toy100 declared as a problem on the shared runner (``benchmarks.toy_runner``).

Problem only: the unlabelled sampler (``problems.sample_real``), the networks,
and the frozen metrics/verdict (``metrics.evaluate_samples`` / ``metrics.passes``).
Optimizers and their LR schedule, loss, critic penalty, prior group, critic
input noise, generator output noise, EMA and the training loop come from the
shipped recipe through ``benchmarks.toy_runner``. The gate evidence files are
written by the observer in ``benchmarks.toy100.evidence``::

    # gate protocol, all three problems; gate/accuracy/render grade it unchanged
    python -u -m benchmarks.toy100 run --output runs/toy-refactor/toy100_native
    # one problem through the shared runner CLI, one JSON line per observation
    python -m benchmarks.toy100.native grid100 --log runs/toy-refactor/toy100_grid100.log
"""
from __future__ import annotations

import math
import sys

import torch
from torch import nn

from lib.toy_models import SimpleMLPDiscriminator
from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, ToyProblem, main as runner_main

from .metrics import EVAL_N, evaluate_samples, passes
from .problems import PROBLEM_NAMES, sample_real

STEPS = 7000
D_HIDDEN = 128
N_HIDDEN = 3
FOURIER = 3
# The affine_square_v1 table was uniform on [-5, 5]^2; particlegan.init draws
# particle tables as R2 points through a normal quantile, so match its std.
PRIOR_STD = 5.0 / math.sqrt(3.0)


class IdentityAffine(nn.Linear):
    """The affine_square_v1 generator: ``nn.Linear(d, d)`` constructed as the identity."""

    def __init__(self, dim: int = 2):
        super().__init__(dim, dim)
        with torch.no_grad():
            self.weight.copy_(torch.eye(dim))
            self.bias.zero_()


init.register(IdentityAffine, {"weight": init.KEEP, "bias": init.KEEP})


class Toy100(ToyProblem):
    """One of grid100 / rotated100 / staggered100: 100 equal-weight modes, sigma 0.03."""

    def __init__(self, problem: str = "grid100", *, steps: int = STEPS):
        if problem not in PROBLEM_NAMES:
            raise ValueError(f"unknown toy100 problem {problem!r}")
        self.problem, self.steps, self.name = problem, int(steps), f"toy100_{problem}"

    def recipe(self):
        return get_recipe(z_dim=2, num_particles=20_000, batch_size=2048, total_steps=self.steps)

    def networks(self, recipe, seed):
        generator = init.deterministic_orthogonal_(IdentityAffine(2), seed=seed)
        critic = init.deterministic_orthogonal_(
            SimpleMLPDiscriminator(in_dim=2, hidden_dim=D_HIDDEN, n_hidden=N_HIDDEN, fourier=FOURIER),
            seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(init_std=PRIOR_STD), seed=seed)
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        return sample_real(self.problem, n, device=stream.device, generator=stream)

    def metrics(self, model):
        return dict(evaluate_samples(model.sample(EVAL_N).x, self.problem))

    def verdict(self, metrics):
        return "PASS" if passes(self.problem, metrics) else "FAIL"


if __name__ == "__main__":
    argv = sys.argv[1:]
    name = argv.pop(0) if argv and argv[0] in PROBLEM_NAMES else "grid100"
    raise SystemExit(runner_main(Toy100(name), argv))
