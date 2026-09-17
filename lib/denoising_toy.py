"""Analytic Gaussian-grid denoising and small conditional adversarial models."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import DDGAN, ucd_labels, ucd_scores
from particlegan.diffusion import DrawSource

# Historical imports remain valid for existing analysis scripts/checkpoints.
DiffusionSchedule = DDGAN


class GaussianGrid:
    def __init__(self, device="cpu", std=0.03, classes=4):
        if classes not in (1, 4):
            raise ValueError("classes must be 1 or 4")
        ij = torch.cartesian_prod(torch.arange(10, device=device), torch.arange(10, device=device))
        self.means = ij.float() - 4.5
        self.labels = (2 * (ij[:, 0] % 2) + ij[:, 1] % 2) if classes == 4 else torch.zeros(100, dtype=torch.long, device=device)
        self.by_class = torch.stack([torch.where(self.labels == c)[0] for c in range(classes)])
        self.std, self.classes = float(std), classes

    def sample(self, c, rng):
        j = torch.randint(self.by_class.shape[1], (len(c),), device=c.device, generator=rng)
        ids = self.by_class[c, j]
        x = self.means[ids] + self.std * torch.randn((len(c), 2), device=c.device, generator=rng)
        return x

    def posterior(self, xt, c, abar):
        """Return weights, means, scalar variance of exact q(x0 | xt,c)."""
        a = torch.as_tensor(abar, device=xt.device, dtype=xt.dtype).reshape(-1, 1, 1)
        mu = self.means[self.by_class[c]]
        v = a * self.std**2 + 1 - a
        delta = xt[:, None, :] - a.sqrt() * mu
        weights = (-delta.square().sum(-1) / (2 * v.squeeze(-1))).softmax(-1)
        means = mu + a.sqrt() * self.std**2 / v * delta
        var = self.std**2 * (1 - a) / v
        return weights, means, var

    def oracle_clean(self, xt, c, abar, rng):
        weights, means, var = self.posterior(xt, c, abar)
        k = torch.multinomial(weights, 1, generator=rng).squeeze(1)
        picked = means[torch.arange(len(xt), device=xt.device), k]
        return picked + var.reshape(-1, 1).sqrt() * torch.randn(xt.shape, device=xt.device, generator=rng)


def mlp(in_dim, width, depth, out_dim):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, width), nn.LeakyReLU(0.2)])
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class ToyGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.diffusion = cfg["model"] == "ddgan"
        self.classes = cfg["classes"]
        self.steps = len(cfg["alpha_bar"]) - 1
        inp = cfg["z_dim"] + self.classes + (2 + self.steps if self.diffusion else 0)
        width = cfg.get("generator_hidden")
        self.net = mlp(inp, cfg["hidden"] if width is None else width, cfg["depth"], 2)
        init_weights(self)

    def forward(self, z, c, xt=None, t=None):
        pieces = [z, F.one_hot(c, self.classes).to(z)]
        if self.diffusion:
            pieces.extend([xt, F.one_hot(t - 1, self.steps).to(z)])
        return self.net(torch.cat(pieces, -1))


class ToyDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.diffusion = cfg["model"] == "ddgan"
        self.mode, self.classes = cfg["d_mode"], cfg["classes"]
        self.ucd_target = cfg.get("ucd_target", "class")
        self.joint_ucd = self.ucd_target == "time_class"
        if self.ucd_target not in ("class", "time_class") or (self.joint_ucd and (not self.diffusion or self.mode != "ucd")):
            raise ValueError("time_class UCD requires DDGAN with a UCD discriminator")
        self.drop_xt = cfg["drop_xt"]
        self.steps = len(cfg["alpha_bar"]) - 1
        self.register_buffer("freqs", math.pi * 2 ** torch.arange(cfg["fourier"], dtype=torch.float32))
        inp = 2 + 4 * cfg["fourier"]
        if self.diffusion:
            inp += (0 if self.joint_ucd else self.steps) + (0 if self.drop_xt else 2)
        if self.mode == "concat":
            inp += self.classes
        heads = self.classes * (self.steps if self.joint_ucd else 1)
        self.net = mlp(inp, cfg["hidden"], cfg["depth"], heads if self.mode == "ucd" else 1)
        init_weights(self)

    def ucd_labels(self, c, t):
        """Shared head index for adversarial selection and UCD classification."""
        # Trainers generate valid labels/times; avoid a GPU range-check sync per forward.
        return ucd_labels(c, t, num_classes=self.classes, target=self.ucd_target, validate_args=False)

    def forward(self, x, c, xt=None, t=None):
        xf = x[:, :, None] * self.freqs
        pieces = [x, xf.sin().flatten(1), xf.cos().flatten(1)]
        if self.diffusion:
            if not self.drop_xt:
                pieces.append(xt)
            if not self.joint_ucd:
                pieces.append(F.one_hot(t - 1, self.steps).to(x))
        if self.mode == "concat":
            pieces.append(F.one_hot(c, self.classes).to(x))
        logits = self.net(torch.cat(pieces, -1))
        score = ucd_scores(logits, c, t, num_classes=self.classes, target=self.ucd_target,
                           num_steps=self.steps, validate_args=False) if self.mode == "ucd" else logits.squeeze(1)
        return score, logits


class FixedConditionCritic(nn.Module):
    """Penalty differentiates candidate only; noisy input/time/class stay fixed."""
    def __init__(self, d, c, xt, t):
        super().__init__()
        self.d, self.c, self.xt, self.t = d, c, xt, t

    def forward(self, x):
        return self.d(x, self.c, self.xt, self.t)[0]


@torch.no_grad()
def generate(g, prior, noise, schedule, c, rng_z, rng_noise, rng_start):
    if not g.diffusion:
        return g(prior.sample(len(c), rng_z)[0], c)
    xt = torch.randn((len(c), 2), device=c.device, generator=rng_start)
    for step in range(schedule.steps, 0, -1):
        t = torch.full_like(c, step)
        x0 = g(prior.sample(len(c), rng_z)[0], c, xt, t)
        xt = schedule.reverse(x0, xt, t, noise.sample(len(c), rng_noise)[0])
    return xt


@torch.no_grad()
def grid_metrics(x, c, toy, real, seed=31415):
    from lib.toy_metrics import sliced_w1, per_mode_core_ratio, per_mode_covariance_ratios

    distance, nearest = torch.cdist(x, toy.means).min(1)
    correct = toy.labels[nearest] == c
    hq = distance < 3 * toy.std
    counts = torch.bincount(nearest, minlength=100).float()
    hqcounts = torch.bincount(nearest[hq & correct], minlength=100)
    expected = torch.ones_like(counts) / 100
    p = counts / len(x)
    conditional_tv = []
    for cls in range(toy.classes):
        ids = nearest[c == cls]
        cp = torch.bincount(ids, minlength=100).float() / len(ids)
        target = (toy.labels == cls).float() / toy.by_class.shape[1]
        conditional_tv.append((cp - target).abs().sum() / 2)
    out = {"hq": float(hq.float().mean()), "cond_acc": float(correct.float().mean()),
           "joint_hq": float((hq & correct).float().mean()),
           "modes": int((hqcounts >= max(5, 0.05 * len(x) / 100)).sum()),
           "mode_tv": float((p - expected).abs().sum() / 2),
           "conditional_mode_tv": float(torch.stack(conditional_tv).mean()),
           "tail_10sigma": float((distance > 10 * toy.std).float().mean()),
           "sw1": sliced_w1(x, real, 64, seed=seed),
           "conditional_sw1": sum(sliced_w1(x[c == cls], real[c == cls], 64, seed=seed)
                                  for cls in range(toy.classes)) / toy.classes}
    out.update(per_mode_core_ratio(x, min_count=20, std=toy.std))
    out.update(per_mode_covariance_ratios(x, min_count=20, data_std=toy.std))
    return out


@torch.no_grad()
def conditional_probe(g, prior, noise, schedule, toy, n=256, seed=8128):
    """Oracle and model transitions at fixed observations; primary metric is SW1."""
    from lib.toy_metrics import sliced_w1
    if not g.diffusion:
        return {}, None
    device = toy.means.device
    rng = torch.Generator(device=device).manual_seed(seed)
    probes = torch.tensor([[-.5, -.5], [.5, .5], [0., 0.], [3., -2.]], device=device)
    results, panels = [], []
    for step in range(1, schedule.steps + 1):
        for cls in range(toy.classes):
            for u in probes:
                xt = (schedule.ab[step].sqrt() * u).expand(n, -1)
                c = torch.full((n,), cls, device=device, dtype=torch.long)
                t = torch.full_like(c, step)
                ref0 = toy.oracle_clean(xt, c, schedule.ab[step], rng)
                ref = schedule.reverse(ref0, xt, t, torch.randn((n, 2), device=device, generator=rng))
                pred0 = g(prior.sample(n, rng)[0], c, xt, t)
                pred = schedule.reverse(pred0, xt, t, noise.sample(n, rng)[0])
                ref02 = toy.oracle_clean(xt, c, schedule.ab[step], rng)
                ref2 = schedule.reverse(ref02, xt, t, torch.randn((n, 2), device=device, generator=rng))
                results.append((step, sliced_w1(pred, ref, 32, seed=seed), sliced_w1(ref2, ref, 32, seed=seed)))
                if cls == 0 and bool(torch.equal(u, probes[1])):
                    panels.append({"t": step, "xt": xt[0].cpu().numpy(),
                                   "model": pred.cpu().numpy(), "oracle": ref.cpu().numpy()})
    out = {"posterior_sw1": sum(r[1] for r in results) / len(results),
           "posterior_floor": sum(r[2] for r in results) / len(results)}
    for step in range(1, schedule.steps + 1):
        rows = [r for r in results if r[0] == step]
        out[f"posterior_sw1_t{step}"] = sum(r[1] for r in rows) / len(rows)
    return out, panels
