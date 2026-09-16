"""Conditional two-route sequences, temporal models, and geometric diagnostics.

Diffusion time indexes corruption of an entire [2, length] future. Sequence
time indexes positions along that future. Route IDs are never model inputs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import ucd_labels


class Routes:
    classes = 2
    # start height, obstacle height, obstacle radius. Test geometries are absent
    # from training; test classes are the same two preference labels.
    train_geometry = [(-.2, -.12, .22), (-.2, .12, .32),
                      (.2, -.12, .32), (.2, .12, .22)]
    test_geometry = [(0., 0., .27), (-.1, .06, .25),
                     (.1, -.06, .29), (.3, .18, .35)]
    upper_probability = (.8, .3)

    def __init__(self, length=64, device="cpu", geometry_mode="discrete"):
        if geometry_mode not in ("discrete", "continuous"):
            raise ValueError(geometry_mode)
        self.geometry_mode = geometry_mode
        self.length, self.device = length, device
        self.s = torch.linspace(0, 1, length, device=device)
        s = self.s
        self.basis = torch.zeros(3, 2, length, device=device)
        self.basis[0, 0] = .06 * torch.sin(math.pi * s)
        self.basis[1, 1] = .06 * torch.sin(math.pi * s).square()
        self.basis[2, 1] = .035 * torch.sin(2 * math.pi * s) * torch.sin(math.pi * s)
        self.pinv = torch.linalg.pinv(self.basis.flatten(1).T)

    def contexts(self, split):
        if split not in ("train", "test"):
            raise ValueError(split)
        geometries = self.train_geometry if split == "train" else self.test_geometry
        geom = torch.tensor(geometries, device=self.device).repeat_interleave(2, 0)
        c = torch.arange(2, device=self.device).repeat(len(geometries))
        return c, geom

    def condition(self, geom):
        past = geom.new_zeros(len(geom), 2, 8)
        past[:, 0] = torch.linspace(-1.35, -1., 8, device=geom.device)
        past[:, 1] = geom[:, 0, None]
        return torch.cat([past.flatten(1), geom[:, 1:]], 1)

    def templates(self, geom):
        s = self.s
        bump = torch.sin(math.pi * s).square()
        base_y = geom[:, 0, None] * (1 - s) + geom[:, 1, None] * bump
        out = geom.new_empty(len(geom), 2, 2, self.length)
        out[:, :, 0] = -1 + 2 * s
        # index 0 = lower, 1 = upper
        signs = geom.new_tensor([-1, 1])
        out[:, :, 1] = base_y[:, None] + signs[None, :, None] * (geom[:, 2, None, None] + .30) * bump
        return out

    def sample(self, c, geom, rng):
        p = geom.new_tensor(self.upper_probability)[c]
        route = (torch.rand(len(c), device=geom.device, generator=rng) < p).long()
        coeff = 2 * torch.rand(len(c), 3, device=geom.device, generator=rng) - 1
        x = self.templates(geom)[torch.arange(len(c), device=geom.device), route]
        return x + torch.einsum("nk,kct->nct", coeff, self.basis), route

    def batch(self, n, rng):
        if self.geometry_mode == "continuous":
            # Same coordinate bounds as the original four geometries. Fixed
            # evaluation contexts stay unchanged; scene 4 remains outside.
            c = torch.randint(2, (n,), device=self.device, generator=rng)
            geom = torch.rand(n, 3, device=self.device, generator=rng)
            geom = geom * geom.new_tensor([.4, .24, .10]) + geom.new_tensor([-.2, -.12, .22])
            return c, geom, self.sample(c, geom, rng)[0]
        c, geom = self.contexts("train")
        i = torch.randint(len(c), (n,), device=geom.device, generator=rng)
        x, _ = self.sample(c[i], geom[i], rng)
        return c[i], geom[i], x

    def diagnose(self, x, geom):
        delta = x[:, None] - self.templates(geom)
        coeff = delta.flatten(2) @ self.pinv.T
        residual = delta - torch.einsum("nrk,kct->nrct", coeff, self.basis)
        # Choose nearest point in the bounded analytic support, rather than
        # labeling every arbitrary output a successful route.
        clipped = torch.einsum("nrk,kct->nrct", coeff.clamp(-1, 1), self.basis)
        distance = (delta - clipped).square().mean((2, 3)).sqrt()
        route = distance.argmin(1)
        idx = torch.arange(len(x), device=x.device)
        tube = distance[idx, route]
        coeff = coeff[idx, route]
        # Continuous line-segment/circle collision, including between frames.
        center = torch.stack([torch.zeros_like(geom[:, 1]), geom[:, 1]], 1)
        a = x[:, :, :-1] - center[:, :, None]
        v = x[:, :, 1:] - x[:, :, :-1]
        u = (-(a * v).sum(1) / v.square().sum(1).clamp_min(1e-12)).clamp(0, 1)
        clearance = (a + v * u[:, None]).norm(dim=1).min(1).values - geom[:, 2]
        start = torch.stack([-torch.ones_like(geom[:, 0]), geom[:, 0]], 1)
        end = x.new_tensor([1., 0.])
        boundary = torch.maximum((x[:, :, 0] - start).norm(dim=1), (x[:, :, -1] - end).norm(dim=1))
        valid = (clearance > 0) & (boundary < .1) & (tube < .05)
        return dict(route=route, coeff=coeff, tube=tube, clearance=clearance,
                    boundary=boundary, valid=valid,
                    residual=residual[idx, route].square().mean((1, 2)).sqrt())


class Block(nn.Module):
    def __init__(self, inp, out, cond):
        super().__init__()
        self.conv1 = nn.Conv1d(inp, out, 3, padding=1)
        self.conv2 = nn.Conv1d(out, out, 3, padding=1)
        self.mod = nn.Linear(cond, out)
        self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, h):
        y = F.leaky_relu(self.conv1(x) + self.mod(h)[:, :, None], .2)
        return F.leaky_relu((self.conv2(y) + self.skip(x)) / math.sqrt(2), .2)


class TrajectoryGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.diffusion = cfg["model"] == "ddgan"
        self.length, self.steps = cfg["length"], len(cfg["alpha_bar"]) - 1
        w = cfg["width"]
        self.embed = nn.Sequential(nn.Linear(cfg["z_dim"] + 18 + 2 + self.steps, 2*w), nn.LeakyReLU(.2), nn.Linear(2*w, 2*w))
        self.enc0 = Block(3, w, 2*w)
        self.enc1 = Block(w, 2*w, 2*w)
        self.mid = Block(2*w, 2*w, 2*w)
        self.dec1 = Block(4*w, w, 2*w)
        self.dec0 = Block(2*w, w, 2*w)
        self.out = nn.Conv1d(w, 2, 3, padding=1)
        self.register_buffer("position", torch.linspace(-1, 1, self.length)[None, None])

    def forward(self, z, c, context, xt=None, t=None):
        time = F.one_hot(t - 1, self.steps).to(z) if self.diffusion else z.new_zeros(len(z), self.steps)
        h = self.embed(torch.cat([z, context, F.one_hot(c, 2).to(z), time], 1))
        if xt is None:
            xt = z.new_zeros(len(z), 2, self.length)
        a = self.enc0(torch.cat([xt, self.position.expand(len(z), -1, -1)], 1), h)
        b = self.enc1(F.avg_pool1d(a, 2), h)
        m = self.mid(F.avg_pool1d(b, 2), h)
        y = self.dec1(torch.cat([F.interpolate(m, size=b.shape[-1], mode="nearest"), b], 1), h)
        y = self.dec0(torch.cat([F.interpolate(y, size=a.shape[-1], mode="nearest"), a], 1), h)
        return self.out(y)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.diffusion, self.mode = cfg["model"] == "ddgan", cfg["d_mode"]
        self.steps = len(cfg["alpha_bar"]) - 1
        self.heads = 2 * (self.steps if self.diffusion else 1)
        extra = (2 + self.steps) if self.mode == "concat" else 0
        inp = 2 * cfg["length"] * (2 if self.diffusion else 1) + 18 + extra
        w = cfg["d_width"]
        self.architecture = cfg.get("d_architecture", "mlp")
        if self.architecture in ("temporal", "hybrid"):
            tw = cfg.get("d_temporal_width", 64)
            channels = 2 * (2 if self.diffusion else 1) + 1
            self.stages = nn.ModuleList([
                nn.Sequential(nn.Conv1d(channels, tw, 5, padding=2), nn.LeakyReLU(.2)),
                nn.Sequential(nn.Conv1d(tw, 2*tw, 5, stride=2, padding=2), nn.LeakyReLU(.2)),
                nn.Sequential(nn.Conv1d(2*tw, 2*tw, 5, stride=2, padding=2), nn.LeakyReLU(.2)),
            ])
            self.register_buffer("position", torch.linspace(-1, 1, cfg["length"])[None, None])
            if self.architecture == "hybrid":
                # Preserve a direct full-path view alongside local features.
                # c/t still enter only through UCD head selection (or concat).
                self.global_net = nn.Sequential(nn.Linear(inp-extra, w), nn.LeakyReLU(.2),
                                                nn.Linear(w, w), nn.LeakyReLU(.2))
                fused, hidden = 20*tw + w + extra, w//2
            else:
                fused, hidden = 20*tw + 18 + extra, w//4
            self.net = nn.Sequential(nn.Linear(fused, hidden), nn.LeakyReLU(.2),
                                     nn.Linear(hidden, self.heads if self.mode == "ucd" else 1))
        elif self.architecture == "mlp":
            self.net = nn.Sequential(nn.Linear(inp, w), nn.LeakyReLU(.2),
                                 nn.Linear(w, w), nn.LeakyReLU(.2),
                                 nn.Linear(w, w), nn.LeakyReLU(.2),
                                 nn.Linear(w, self.heads if self.mode == "ucd" else 1))
        else:
            raise ValueError(self.architecture)

    def ucd_labels(self, c, t):
        return ucd_labels(c, t, num_classes=2, target="time_class" if self.diffusion else "class", validate_args=False)

    def forward(self, x, c, context, xt=None, t=None):
        if self.architecture in ("temporal", "hybrid"):
            inputs = [x, self.position.expand(len(x), -1, -1)]
            if self.diffusion:
                inputs.append(xt)
            h = torch.cat(inputs, 1)
            pieces = []
            for stage in self.stages:
                h = stage(h)
                pieces.append(F.adaptive_avg_pool1d(h, 4).flatten(1))
            if self.architecture == "hybrid":
                global_inputs = [x.flatten(1), context]
                if self.diffusion:
                    global_inputs.append(xt.flatten(1))
                pieces.append(self.global_net(torch.cat(global_inputs, 1)))
            else:
                pieces.append(context)
        else:
            pieces = [x.flatten(1), context]
            if self.diffusion:
                pieces.append(xt.flatten(1))
        if self.mode == "concat":
            pieces.extend([F.one_hot(c, 2).to(x), F.one_hot(t - 1, self.steps).to(x) if self.diffusion else x.new_zeros(len(x), self.steps)])
        logits = self.net(torch.cat(pieces, 1))
        score = logits.gather(1, self.ucd_labels(c, t)[:, None]).squeeze(1) if self.mode == "ucd" else logits.squeeze(1)
        return score, logits


class TrajectoryCritic(nn.Module):
    def __init__(self, d, c, context, xt, t):
        super().__init__()
        self.d, self.c, self.context, self.xt, self.t = d, c, context, xt, t

    def forward(self, x):
        return self.d(x, self.c, self.context, self.xt, self.t)[0]


@torch.no_grad()
def generate(g, prior, noise, schedule, c, context, rngs, fixed_ids=None, fixed_random=False):
    n = len(c)
    def latent(step):
        return prior.table[fixed_ids[step]] if fixed_ids is not None else prior.sample(n, rngs[0])[0]
    def random_start():
        x = torch.randn((1 if fixed_random else n, 2, g.length), device=c.device, generator=rngs[2])
        return x.expand(n, -1, -1)
    if not g.diffusion:
        return g(latent(0), c, context)
    xt = random_start()
    for step in range(schedule.steps, 0, -1):
        t = torch.full_like(c, step)
        clean = g(latent(step-1), c, context, xt, t)
        eta = noise.sample(1 if fixed_random else n, rngs[1])[0].reshape(-1, 2, g.length).expand(n, -1, -1)
        xt = schedule.reverse(clean, xt, t, eta)
    return xt


@torch.no_grad()
def metrics(toy, x, real, c, geom, groups):
    from lib.toy_metrics import sliced_w1
    dx, dr = toy.diagnose(x, geom), toy.diagnose(real, geom)
    rows = []
    for k in groups.unique().tolist():
        mask = groups == k
        p = toy.upper_probability[int(c[mask][0])]
        valid = dx["valid"][mask]
        route = dx["route"][mask]
        mass = torch.stack([((route == r) & valid).float().mean() for r in (0, 1)])
        coverage = int((mass > .05 * x.new_tensor([1-p, p])).sum())
        ratios = []
        for r in (0, 1):
            a = mask & (dx["route"] == r) & dx["valid"]
            b = mask & (dr["route"] == r)
            ratios.append(float(dx["coeff"][a].var(0).mean() / dr["coeff"][b].var(0).mean()) if int(a.sum()) >= 10 and int(b.sum()) >= 10 else None)
        rows.append(dict(context=k, upper_target=p, upper_observed=float(route.float().mean()),
                         route_tv=abs(float(route.float().mean())-p), valid_route_mass=mass.tolist(),
                         valid=float(valid.float().mean()), routes_covered=coverage,
                         coefficient_variance_ratio=ratios,
                         sw1=sliced_w1(x[mask].flatten(1), real[mask].flatten(1), 64, seed=31415)))
    return dict(valid=float(dx["valid"].float().mean()), collision=float((dx["clearance"] <= 0).float().mean()),
                boundary_error=float(dx["boundary"].mean()), support_rmse=float(dx["tube"].mean()),
                residual_rmse=float(dx["residual"].mean()),
                route_tv=sum(r["route_tv"] for r in rows)/len(rows),
                conditional_sw1=sum(r["sw1"] for r in rows)/len(rows),
                routes_covered=sum(r["routes_covered"] for r in rows), routes_total=2*len(rows), contexts=rows)
