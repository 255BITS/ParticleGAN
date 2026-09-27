"""Paired 2D transport without output MSE: problem definition only.

Data maps, the routed particle student, the paired-error construction and the
MSE-based paired-accuracy metrics come from HyperGAN/particle-sliders 8e2ea7e
(see SOURCE.md and LICENSE). Everything else -- optimizers and their LR
schedule, loss, critic penalty, particle-table regularization, critic input
and generator output noise, EMA, logging -- comes from the shipped recipe
through ``benchmarks.toy_runner``. Output MSE is evaluation-only.

The critic scores paired errors: ``real = n`` and
``fake = (n + T(G(x))) - T(y)`` with the same ``n ~ N(0, sigma(step)^2)``
inside a pair, where ``T`` standardizes by the training targets.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem

TASKS = ("affine2", "swirl2")
CLOUDS = ("movable", "fixed")
SPLIT_SEEDS = dict(train=922071, validation=922193, test=922827)
SIZES = dict(train=1024, validation=1024, test=4096)
BATCH, STEPS = 64, 6000
PARTICLES, PARTICLE_DIM, WIDTH, ROUTER_WIDTH = 128, 4, 48, 16
TARGET_STD_FLOOR = 1e-4
# Paired-error data noise: geometric anneal over a fixed 8,000-update horizon,
# held at ``NOISE_HOLD_RATIO`` x the target scale when that is below the start.
NOISE_START, NOISE_FLOOR, NOISE_HORIZON, NOISE_HOLD_RATIO = 1.0, 0.03, 8000, 1.3
PASS_NMSE, PASS_P95 = 0.01, 0.2


def targets(x, task):
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected [batch, 2] source coordinates")
    if task == "affine2":
        return x @ x.new_tensor([[.8, -.6], [.6, .8]]) + x.new_tensor([.2, -.3])
    if task == "swirl2":
        # Radius-dependent rotation, preserving each point's radius.
        u = x / math.sqrt(3)
        angle = 1.7 * u.square().sum(1)
        c, s = angle.cos(), angle.sin()
        return torch.stack((c * x[:, 0] - s * x[:, 1],
                            s * x[:, 0] + c * x[:, 1]), dim=1)
    raise ValueError(task)


def data(task, split):
    rng = torch.Generator().manual_seed(SPLIT_SEEDS[split])
    x = (2 * torch.rand(SIZES[split], 2, generator=rng) - 1) * math.sqrt(3)
    return x, targets(x, task)


def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


class RoutedMLP(nn.Module):
    """Residual edit ``MLP(concat(x, softmax(q(x) P^T / sqrt(d)) P))`` over a particle table P."""

    def __init__(self):
        super().__init__()
        self.router = mlp(2, PARTICLE_DIM, ROUTER_WIDTH)
        self.net = mlp(2 + PARTICLE_DIM, 2, WIDTH)

    def forward(self, x, particles):
        weights = (self.router(x) @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        return self.net(torch.cat((x, weights @ particles), dim=-1))


def predict(nets, x):
    """The student: an ordinary forward pass ``x + bridge(x, P)``."""
    return x + nets.generator(x, nets.prior.z)


def noise_std(step, target_scale):
    sigma = NOISE_START * (NOISE_FLOOR / NOISE_START) ** min(step / NOISE_HORIZON, 1.)
    hold = target_scale * NOISE_HOLD_RATIO
    # Source rule: a hold above the start does not raise noise.
    return max(sigma, hold) if NOISE_START > hold else sigma


@torch.no_grad()
def metrics(prediction, target):
    error = prediction - target
    distance = error.norm(dim=1)
    return dict(nmse=float(error.square().mean() / target.var(0, unbiased=False).mean()),
                rmse=float(error.square().mean().sqrt()),
                p95_distance=float(torch.quantile(distance, .95)),
                within_0p1=float((distance < .1).float().mean()))


class PairedError2D(ToyProblem):
    """One map (``task``) with a movable or fixed (control) particle cloud."""

    def __init__(self, task="swirl2", cloud="movable"):
        if task not in TASKS or cloud not in CLOUDS:
            raise ValueError((task, cloud))
        self.task, self.cloud = task, cloud
        self.name = f"paired_error_2d_{task}_{cloud}"
        self.x, self.y = data(task, "train")
        self.validation, self.test = data(task, "validation"), data(task, "test")
        self.mean = self.y.mean(0)
        self.std = self.y.std(0).clamp_min(TARGET_STD_FLOOR)
        self.target_scale = float(self.std.square().mean().sqrt())
        self.y_normalized = self.normalize(self.y)
        self.draws = 0  # real() calls: one for the critic and one for the generator per update

    def normalize(self, v):
        return (v - self.mean) / self.std

    def recipe(self):
        return get_recipe(z_dim=PARTICLE_DIM, num_particles=PARTICLES, batch_size=BATCH, total_steps=STEPS)

    def networks(self, recipe, seed):
        generator = init.deterministic_orthogonal_(RoutedMLP(), seed=seed)
        critic = init.deterministic_orthogonal_(mlp(2, 1, WIDTH), seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        # The fixed control keeps the same initial table and routing; only updates are disabled.
        prior.z.requires_grad_(self.cloud == "movable")
        return Networks(generator=generator, critics=critic, prior=prior)

    def real(self, n, stream):
        sigma = noise_std(self.draws // 2 + 1, self.target_scale)
        self.draws += 1
        rows = torch.randint(len(self.x), (n,), generator=stream)
        return Sample(torch.randn(n, 2, generator=stream) * sigma, indices=rows)

    def fake(self, nets, n, stream, real):
        if real is None:  # noiseless paired errors on training rows
            rows, noise = torch.randint(len(self.x), (n,), generator=stream), 0.
        else:
            rows, noise = real.indices, real.x
        return noise + self.normalize(predict(nets, self.x[rows])) - self.y_normalized[rows]

    def metrics(self, model):
        (vx, vy), (tx, ty) = self.validation, self.test
        row = metrics(predict(model.nets, vx), vy)
        test = metrics(predict(model.nets, tx), ty)
        row.update(test_nmse=test["nmse"], test_p95_distance=test["p95_distance"])
        return row

    def verdict(self, metrics):
        return "PASS" if metrics["nmse"] <= PASS_NMSE and metrics["p95_distance"] <= PASS_P95 else "FAIL"
