"""Standalone paired transport using ParticleGAN's public numerical primitives.

Architecture/normalization/RNG contract extracted from HyperGAN/particle-sliders
8e2ea7e and the model-glue comparison at c89476d. See SOURCE.md and LICENSE.
There are no imports from either application. Output MSE is evaluation-only.
"""

from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn

from benchmarks.legacy.gan_loss import GANLoss
from benchmarks.legacy.grad_regularizers import GradRegularizer
from particlegan import learning_rate_scale
from particlegan.vicreg_loss import VICRegLikeLoss


ARMS = {
    "baseline": dict(kappa=1., coefficient=1., rate=1., cosine=False, vic=1.),
    "cap-cosine": dict(kappa=1.25, coefficient=3., rate=.85, cosine=True, vic=1.),
    "cap-cosine-vic005": dict(kappa=1.25, coefficient=3., rate=.85, cosine=True, vic=.05),
}
PROTOCOL = dict(seed=0, steps=6000, record_every=500, train_size=1024,
                validation_size=1024, test_size=4096,
                split_seeds=dict(train=922071, validation=922193, test=922827),
                batch=64, particles=128, particle_dim=4, width=48, router_width=16,
                generator_lr=.0006, critic_lr=.0009, particle_lr=.006,
                betas=[0., .999], cap_every=4, ema=.995,
                noise_start=1., noise_floor=.03, noise_horizon=8000,
                noise_hold_ratio=1.3, target_std_floor=1e-4,
                cosine_start=.6, cosine_floor=.05,
                validation_nmse_bound=.01, validation_p95_bound=.2, stable_tail=3)
TASKS = ("affine2", "swirl2")
CLOUDS = ("movable", "fixed")


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


def data(task, split, protocol=PROTOCOL):
    rng = torch.Generator().manual_seed(protocol["split_seeds"][split])
    x = (2 * torch.rand(protocol[f"{split}_size"], 2, generator=rng) - 1) * math.sqrt(3)
    return x, targets(x, task)


def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


class RoutedMLP(nn.Module):
    def __init__(self, protocol=PROTOCOL):
        super().__init__()
        self.router = mlp(2, protocol["particle_dim"], protocol["router_width"])
        self.net = mlp(2 + protocol["particle_dim"], 2, protocol["width"])

    def forward(self, x, particles):
        q = self.router(x)
        weights = (q @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        code = weights @ particles
        return self.net(torch.cat((x, code), dim=-1))


class Student(nn.Module):
    def __init__(self, cloud="movable", protocol=PROTOCOL):
        super().__init__()
        if cloud not in CLOUDS:
            raise ValueError(cloud)
        # Keep names and initialization order for exact source-state comparison.
        self.particles = nn.Parameter(torch.randn(protocol["particles"], protocol["particle_dim"]),
                                      requires_grad=cloud == "movable")
        self.bridge = RoutedMLP(protocol)

    def forward(self, x):
        return x + self.bridge(x, self.particles)


class ErrorCritic(nn.Module):
    def __init__(self, training_targets, protocol=PROTOCOL):
        super().__init__()
        if len(training_targets) < 2 or not torch.isfinite(training_targets).all():
            raise ValueError("Need finite training targets for normalization")
        scale = training_targets.std(0).clamp_min(protocol["target_std_floor"])
        self.register_buffer("target_mean", training_targets.mean(0))
        self.register_buffer("target_std", scale)
        self.register_buffer("input_scale", scale.square().mean().sqrt())
        self.register_buffer("edit_rms", self.input_scale.detach().clone())
        self.net = mlp(2, 1, protocol["width"])

    def normalize(self, x):
        return (x - self.target_mean) / self.target_std

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Sampler:
    def __init__(self, count, protocol=PROTOCOL):
        self.count, self.batch = count, protocol["batch"]
        self.rngs = {k: torch.Generator().manual_seed(protocol["seed"] + offset)
                     for k, offset in [("data", 10), ("noise", 30), ("vic", 40)]}

    def draw(self, sigma):
        ids = torch.randint(self.count, (self.batch,), generator=self.rngs["data"])
        noise = torch.randn(self.batch, 2, generator=self.rngs["noise"]) * sigma
        return ids, noise

    def vic_rows(self, count):
        return torch.randperm(count, generator=self.rngs["vic"])[:64]

    def state_dict(self):
        return dict(count=self.count, batch_size=self.batch,
                    rngs={k: r.get_state() for k, r in self.rngs.items()})

    def load_state_dict(self, state):
        if state["count"] != self.count or state["batch_size"] != self.batch:
            raise ValueError("Sampler changed")
        for k, v in state["rngs"].items():
            self.rngs[k].set_state(v)


def noise_std(step, critic, protocol=PROTOCOL):
    start = protocol["noise_start"]
    sigma = start * (protocol["noise_floor"] / start) ** min(step / protocol["noise_horizon"], 1.)
    hold = float(critic.edit_rms) * protocol["noise_hold_ratio"]
    # Preserve the source rule: a hold above the start does not raise noise.
    return max(sigma, hold) if start > hold else sigma


class Game:
    def __init__(self, training_targets, arm="baseline", cloud="movable", protocol=PROTOCOL):
        self.protocol = copy.deepcopy(protocol)
        self.arm_name, self.cloud, self.arm = arm, cloud, ARMS[arm]
        torch.manual_seed(protocol["seed"])
        self.model = Student(cloud, protocol)
        torch.manual_seed(protocol["seed"] + 10000)
        self.critic = ErrorCritic(training_targets, protocol)
        self.g = torch.optim.Adam([
            dict(params=list(self.model.bridge.parameters()), lr=protocol["generator_lr"], role="generator"),
            dict(params=[self.model.particles], lr=protocol["particle_lr"], role="particles"),
        ], betas=tuple(protocol["betas"]))
        self.d = torch.optim.Adam(self.critic.parameters(), lr=protocol["critic_lr"],
                                  betas=tuple(protocol["betas"]))
        for opt in (self.g, self.d):
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"] * self.arm["rate"]
                group["lr"] = group["initial_lr"]
        self.sampler = Sampler(len(training_targets), protocol)
        self.ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.gan = GANLoss(loss_type="logistic", mode="rp")
        self.cap = GradRegularizer(arm="b_cap", coeff=self.arm["coefficient"],
            kappa=self.arm["kappa"], norm="l2", lazy_k=protocol["cap_every"], method="autograd")
        self.vic = VICRegLikeLoss(target_std=1., eps=1e-4)

    def update(self, x, target, step):
        p = self.protocol
        scale = learning_rate_scale(step - 1, p["steps"], p["cosine_start"], p["cosine_floor"]) \
            if self.arm["cosine"] else 1.
        for opt in (self.g, self.d):
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * scale
        y = self.critic.normalize(target).detach()
        sigma = noise_std(step, self.critic, p)
        self.critic.requires_grad_(True)
        rows, real = self.sampler.draw(sigma)
        real = real.to(y)
        with torch.no_grad():
            # Keep the historical floating-point association: (n + pred) - target.
            fake = real + self.critic.normalize(self.model(x[rows])) - y[rows]
        self.d.zero_grad(set_to_none=True)
        d_adv = self.gan.d_loss(self.critic(real), self.critic(fake))
        cap, _ = self.cap.penalty(self.critic, real, fake, step, collect_stats=False)
        d_loss = d_adv + cap
        if not torch.isfinite(d_loss):
            raise FloatingPointError("Nonfinite discriminator loss")
        d_loss.backward()
        self.d.step()
        self.d.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        rows, real = self.sampler.draw(sigma)
        real = real.to(y)
        with torch.no_grad():
            real_score = self.critic(real)
        self.g.zero_grad(set_to_none=True)
        fake = real + self.critic.normalize(self.model(x[rows])) - y[rows]
        # Public GANLoss takes fake first; the old slider helper takes real first.
        g_adv = self.gan.g_loss(self.critic(fake), real_score)
        indices = self.sampler.vic_rows(len(self.model.particles))
        vic = self.vic(self.model.particles[indices])
        g_loss = g_adv + self.arm["vic"] * vic
        if not torch.isfinite(g_loss):
            raise FloatingPointError("Nonfinite generator loss")
        g_loss.backward()
        self.g.step()
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                self.ema[k].lerp_(v, 1. - p["ema"])
        return {k: float(v.detach()) for k, v in dict(d_adv=d_adv, cap=cap, g_adv=g_adv, vic=vic).items()}

    def state_dict(self):
        return dict(model=self.model.state_dict(), critic=self.critic.state_dict(),
                    g=self.g.state_dict(), d=self.d.state_dict(), ema=self.ema,
                    sampler=self.sampler.state_dict(), torch_rng=torch.get_rng_state())

    def load_state_dict(self, state):
        for key in ("model", "critic", "g", "d", "sampler"):
            getattr(self, key).load_state_dict(state[key])
        self.ema = {k: v.clone() for k, v in state["ema"].items()}
        torch.set_rng_state(state["torch_rng"])


def state_hash(model):
    digest = hashlib.sha256()
    for k, v in model.state_dict().items():
        digest.update(k.encode())
        digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def metrics(prediction, target):
    error = prediction - target
    distance = error.norm(dim=1)
    return dict(nmse=float(error.square().mean() / target.var(0, unbiased=False).mean()),
                rmse=float(error.square().mean().sqrt()),
                p95_distance=float(torch.quantile(distance, .95)),
                within_0p1=float((distance < .1).float().mean()))
