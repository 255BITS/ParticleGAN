"""Single analytic route transitions and shared-latent joint generators.

Action means displacement. Only two positions are evaluated per observation;
no network receives a trajectory, hidden route ID, or analytic coefficients.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from lib.trajectory import Routes
from lib.toy_metrics import sliced_w1
from particlegan import ucd_scores
from particlegan.autoencoder import particle_ae


class Transitions:
    train_geometry = Routes.train_geometry
    test_geometry = Routes.test_geometry
    upper_probability = Routes.upper_probability

    def __init__(self, length=64, device="cpu", geometry_mode="discrete"):
        if length < 2 or geometry_mode not in ("discrete", "continuous"):
            raise ValueError("length >= 2 and discrete/continuous geometry required")
        self.length, self.device, self.geometry_mode = length, device, geometry_mode
        self.times = torch.linspace(0, 1, length, device=device)

    def contexts(self, split):
        if split not in ("train", "test"):
            raise ValueError(split)
        geometries = self.train_geometry if split == "train" else self.test_geometry
        geom = torch.tensor(geometries, device=self.device).repeat_interleave(2, 0)
        c = torch.arange(2, device=self.device).repeat(len(geometries))
        return c, geom

    def condition(self, geom, tick):
        return torch.cat([geom, self.times[tick, None]], 1)

    @staticmethod
    def position(geom, time, route, coeff):
        bump = torch.sin(math.pi * time).square()
        x = -1 + 2*time + coeff[:, 0] * .06 * torch.sin(math.pi*time)
        y = (geom[:, 0]*(1-time) + geom[:, 1]*bump
             + (2*route-1)*(geom[:, 2]+.30)*bump + coeff[:, 1]*.06*bump
             + coeff[:, 2]*.035*torch.sin(2*math.pi*time)*torch.sin(math.pi*time))
        return torch.stack([x, y], 1)

    def sample(self, c, geom, tick, rng):
        p = geom.new_tensor(self.upper_probability)[c]
        route = (torch.rand(len(c), device=self.device, generator=rng) < p).long()
        coeff = 2*torch.rand(len(c), 3, device=self.device, generator=rng)-1
        state = self.position(geom, self.times[tick], route, coeff)
        next_state = self.position(geom, self.times[tick+1], route, coeff)
        return torch.cat([state, next_state-state, next_state], 1)

    def batch(self, n, rng):
        if self.geometry_mode == "discrete":
            cc, gg = self.contexts("train")
            ids = torch.randint(len(cc), (n,), device=self.device, generator=rng)
            c, geom = cc[ids], gg[ids]
        else:
            c = torch.randint(2, (n,), device=self.device, generator=rng)
            geom = torch.rand(n, 3, device=self.device, generator=rng)
            geom = geom*geom.new_tensor([.4, .24, .10])+geom.new_tensor([-.2, -.12, .22])
        tick = torch.randint(self.length-1, (n,), device=self.device, generator=rng)
        return c, geom, tick, self.sample(c, geom, tick, rng)


class TransitionScaler(nn.Module):
    """Frozen training statistics for each coordinate in the three blocks."""
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    @classmethod
    def fit(cls, toy, count=32768, seed=91001):
        rng = torch.Generator(device=toy.device).manual_seed(seed)
        real = toy.batch(count, rng)[-1]
        return cls(real.mean(0), real.std(0, unbiased=False).clamp_min(1e-6))

    def forward(self, x):
        return (x-self.mean)/self.scale

    def inverse(self, x):
        return x*self.scale+self.mean


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.LeakyReLU(.2),
                         nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, out))


class TransitionGenerator(nn.Module):
    def __init__(self, z_dim=32, architecture="branches", width=128, class_scale=1.0, context_scale=1.0):
        super().__init__()
        if architecture not in ("branches", "monolithic"):
            raise ValueError(architecture)
        if not math.isfinite(class_scale) or class_scale <= 0:
            raise ValueError("class_scale must be finite and positive")
        if not math.isfinite(context_scale) or context_scale <= 0:
            raise ValueError("context_scale must be finite and positive")
        self.architecture = architecture
        self.class_scale = float(class_scale)
        self.context_scale = float(context_scale)
        # Three separate MLPs, not three heads attached to a common trunk.
        self.branches = nn.ModuleList([mlp(z_dim+6, width, 2) for _ in range(3)]) \
            if architecture == "branches" else nn.ModuleList([mlp(z_dim+6, width, 6)])

    def forward(self, z, c, context):
        shared = torch.cat([z, self.context_scale*context, self.class_scale*F.one_hot(c, 2).to(z)], 1)
        return torch.cat([branch(shared) for branch in self.branches], 1)


class TransitionEncoder(nn.Module):
    """E(st, at, observed context) -> one deterministic particle code; no s_next input."""
    def __init__(self, z_dim=32, width=128, class_scale=8., context_scale=1.):
        super().__init__()
        self.class_scale, self.context_scale = class_scale, context_scale
        self.features = mlp(10, width, width)
        self.query = nn.Linear(width, z_dim)
        self.offset = nn.Linear(width, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, state_action, c, context, prior):
        if state_action.ndim != 2 or state_action.shape[1] != 4:
            raise ValueError("E requires exactly state/action coordinates, never next state")
        h = self.features(torch.cat([state_action, self.context_scale*context,
                                    self.class_scale*F.one_hot(c, 2).to(state_action)], 1))
        query = F.layer_norm(self.query(h), (self.query.out_features,))
        return particle_ae(query, self.offset(h), prior, temperature=.25,
                           distance_reduction="sum", offset_bound=3.)


def encoded_transition(e, g, prior, state_action, c, context):
    encoding = e(state_action, c, context, prior)
    return g(encoding.codes[:, 0], c, context), encoding


def composed_transition(e, g, prior, fake, c, context):
    decoded, encoding = encoded_transition(e, g, prior, fake[:, :4], c, context)
    return torch.cat([fake[:, :4], decoded[:, 4:]], 1), decoded, encoding


class TransitionDiscriminator(nn.Module):
    def __init__(self, width=256, input_dim=6, conditioning="ucd"):
        super().__init__()
        if conditioning not in ("ucd", "concat"):
            raise ValueError(conditioning)
        self.conditioning = conditioning
        # UCD selects a class head; concat feeds the class into a scalar critic.
        extra = 2 if conditioning == "concat" else 0
        heads = 1 if conditioning == "concat" else 2
        self.net = nn.Sequential(nn.Linear(input_dim+4+extra, width), nn.LeakyReLU(.2),
                                 nn.Linear(width, width), nn.LeakyReLU(.2),
                                 nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, heads))

    def forward(self, x, c, context):
        inputs = [x, context]
        if self.conditioning == "concat":
            inputs.append(F.one_hot(c, 2).to(x))
        logits = self.net(torch.cat(inputs, 1))
        if self.conditioning == "concat":
            return logits.squeeze(1), logits
        return ucd_scores(logits, c, num_classes=2, validate_args=False), logits


class TransitionCritics(nn.Module):
    """One joint D, optionally plus independent conditional marginal Ds.

    Marginals receive only their own two coordinates and observed context.
    Keeping joint D first preserves its initialization when adding critics.
    """
    def __init__(self, width=256, mode="joint", marginal_width=128, conditioning="ucd",
                 shared_state=False, scaler=None, length=64):
        super().__init__()
        if mode not in ("joint", "joint_marginals"):
            raise ValueError(mode)
        self.mode = mode
        self.shared_state = shared_state
        self.dt = 1/(length-1)
        if shared_state:
            if mode != "joint_marginals" or scaler is None:
                raise ValueError("shared state critic requires marginals and a scaler")
            # Equal mixture of the two training-state populations, in physical units.
            mean = (scaler.mean[:2]+scaler.mean[4:])/2
            variance = ((scaler.scale[:2].square()+scaler.scale[4:].square())/2
                        + (scaler.mean[:2]-scaler.mean[4:]).square()/4)
            self.register_buffer("state_mean", mean.clone())
            self.register_buffer("state_scale", variance.sqrt())
            self.register_buffer("triple_mean", scaler.mean.clone())
            self.register_buffer("triple_scale", scaler.scale.clone())
        self.critics = nn.ModuleDict({"joint": TransitionDiscriminator(width, conditioning=conditioning)})
        if mode == "joint_marginals":
            for name in ("state", "action", "next_state"):
                if name != "next_state" or not shared_state:
                    self.critics[name] = TransitionDiscriminator(marginal_width, input_dim=2, conditioning=conditioning)

    def roles(self):
        return ("joint",) if self.mode == "joint" else ("joint", "state", "action", "next_state")

    def critic_for(self, role):
        return self.critics["state" if self.shared_state and role == "next_state" else role]

    def inputs(self, role, x, context):
        observation = self.observation(role, x)
        if self.shared_state and role in ("state", "next_state"):
            start = 0 if role == "state" else 4
            physical = observation*self.triple_scale[start:start+2]+self.triple_mean[start:start+2]
            observation = (physical-self.state_mean)/self.state_scale
            if role == "next_state":
                context = context.clone()
                context[:, -1] += self.dt
        return observation, context

    @staticmethod
    def observation(name, x):
        if name == "joint":
            return x
        start = {"state": 0, "action": 2, "next_state": 4}[name]
        return x[:, start:start+2]


def residual(x):
    return (x[:, 4:]-x[:, :2]-x[:, 2:4]).norm(dim=1)


def shuffle_blocks(x, groups, rng):
    """Preserve each conditional marginal exactly; destroy its pairing."""
    shuffled = x.clone()
    for group in groups.unique():
        ids = torch.where(groups == group)[0]
        for start in (0, 2, 4):
            perm = torch.randperm(len(ids), device=x.device, generator=rng)
            shuffled[ids, start:start+2] = x[ids[perm], start:start+2]
    return shuffled


@torch.no_grad()
def metrics(x, real, scaler, groups):
    xn, rn = scaler(x), scaler(real)
    rows = []
    for group in groups.unique().tolist():
        mask = groups == group
        a, b = xn[mask], rn[mask]
        distance = torch.cdist(a, b)
        reference_distance = torch.cdist(b, b)
        reference_distance.fill_diagonal_(float("inf"))
        # A reference-derived local radius; floor comparison supplies calibration.
        radius = reference_distance.min(1).values.quantile(.95).clamp_min(1e-6)
        row = dict(context=group, joint_sw1=sliced_w1(a, b, 128, seed=31415),
                   coverage=float((distance.min(0).values <= radius).float().mean()),
                   precision=float((distance.min(1).values <= radius).float().mean()),
                   spread_ratio=float(a.var(0, unbiased=False).mean() /
                                      b.var(0, unbiased=False).mean().clamp_min(1e-12)))
        for start, name in ((0, "state"), (2, "action"), (4, "next_state")):
            row[name+"_sw1"] = sliced_w1(a[:, start:start+2], b[:, start:start+2], 128, seed=31415)
        rows.append(row)
    out = {key: sum(row[key] for row in rows)/len(rows) for key in rows[0] if key != "context"}
    r = residual(x)
    out.update(consistency_mean=float(r.mean()), consistency_p50=float(r.quantile(.5)),
               consistency_p95=float(r.quantile(.95)), contexts=rows)
    return out
