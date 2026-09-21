"""Three independent latent generators for numerical Lunar Lander transitions.

Records are [state(8), action(2), next_state(8)]. Network state coordinates
0:6 use one training-only scaler for both roles, and coordinates 6:8 are
contact logits on decoder outputs, or bits/probabilities on observed records.
Terrain is explicit privileged scene context, never a trajectory or episode ID.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from particlegan.autoencoder import particle_ae


STATE_DIM = 8
ACTION_DIM = 2
RECORD_DIM = 18
CONTEXT_DIM = 11


def _check(x, dimension, name):
    if x.ndim != 2 or x.shape[1] != dimension:
        raise ValueError(f"{name} requires [batch, {dimension}] coordinates")


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.LeakyReLU(.2),
                         nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, out))


class GymTransitionScaler(nn.Module):
    """Frozen shared continuous-state statistics; contacts stay binary."""
    def __init__(self, state_mean, state_scale, action_mean, action_scale):
        super().__init__()
        for name, value, size in (("state_mean", state_mean, 6), ("state_scale", state_scale, 6),
                                  ("action_mean", action_mean, 2), ("action_scale", action_scale, 2)):
            value = torch.as_tensor(value).detach().clone()
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid {name}")
            if name.endswith("scale") and (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, value)

    @classmethod
    def fit(cls, training_triples):
        x = torch.as_tensor(training_triples, dtype=torch.float32)
        _check(x, RECORD_DIM, "training triples")
        if not len(x) or not torch.isfinite(x).all():
            raise ValueError("training triples must be finite and nonempty")
        states = torch.cat([x[:, :6], x[:, 10:16]], 0)
        return cls(states.mean(0), states.std(0, unbiased=False).clamp_min(1e-3),
                   x[:, 8:10].mean(0), x[:, 8:10].std(0, unbiased=False).clamp_min(1e-3))

    def state(self, x):
        _check(x, STATE_DIM, "state")
        return torch.cat([(x[:, :6]-self.state_mean)/self.state_scale, x[:, 6:8]], 1)

    def inverse_state(self, x):
        _check(x, STATE_DIM, "state")
        return torch.cat([x[:, :6]*self.state_scale+self.state_mean, x[:, 6:8]], 1)

    def action(self, x):
        _check(x, ACTION_DIM, "action")
        return (x-self.action_mean)/self.action_scale

    def inverse_action(self, x):
        _check(x, ACTION_DIM, "action")
        return x*self.action_scale+self.action_mean

    def forward(self, x):
        _check(x, RECORD_DIM, "transition")
        return torch.cat([self.state(x[:, :8]), self.action(x[:, 8:10]), self.state(x[:, 10:])], 1)

    def inverse(self, x):
        _check(x, RECORD_DIM, "transition")
        return torch.cat([self.inverse_state(x[:, :8]), self.inverse_action(x[:, 8:10]),
                          self.inverse_state(x[:, 10:])], 1)


class GymTransitionGenerator(nn.Module):
    """G1 -> st, G2 -> at, G3 -> st+1, using the same latent draw."""
    def __init__(self, scaler, z_dim=32, width=128, context_dim=CONTEXT_DIM):
        super().__init__()
        self.z_dim, self.context_dim = z_dim, context_dim
        self.branches = nn.ModuleList([mlp(z_dim+context_dim, width, out) for out in (8, 2, 8)])
        self.register_buffer("action_mean", scaler.action_mean.detach().clone())
        self.register_buffer("action_scale", scaler.action_scale.detach().clone())

    def forward(self, z, terrain):
        _check(z, self.z_dim, "latent")
        _check(terrain, self.context_dim, "terrain")
        shared = torch.cat([z, terrain], 1)
        state, action, successor = [branch(shared) for branch in self.branches]
        action = (action.tanh()-self.action_mean)/self.action_scale
        return torch.cat([state, action, successor], 1)


class GymTransitionEncoder(nn.Module):
    """E(st, at, terrain) -> z_hat; no successor input or conditional noise."""
    def __init__(self, z_dim=32, width=128, context_dim=CONTEXT_DIM):
        super().__init__()
        self.context_dim = context_dim
        self.features = mlp(10+context_dim, width, width)
        self.query = nn.Linear(width, z_dim)
        self.offset = nn.Linear(width, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, state_action, terrain, prior):
        _check(state_action, 10, "E state/action input (never next state)")
        _check(terrain, self.context_dim, "terrain")
        features = self.features(torch.cat([state_action, terrain], 1))
        query = F.layer_norm(self.query(features), (self.query.out_features,))
        return particle_ae(query, self.offset(features), prior, temperature=.25,
                           distance_reduction="sum", offset_bound=3.)


def contact_state(output, rng=None, straight_through=False, mode="sample"):
    """Convert next/state logits to probabilities or binary observations.

    Sampled Bernoulli values have a biased straight-through sigmoid derivative
    only when requested. Recursive evaluation uses mode='threshold' (p >= .5).
    """
    _check(output, STATE_DIM, "decoded state")
    probability = output[:, 6:8].sigmoid()
    if mode == "probability":
        contact = probability
    elif mode in ("sample", "threshold"):
        if mode == "sample":
            contact = (torch.rand(probability.shape, device=probability.device,
                                  dtype=probability.dtype, generator=rng) < probability).to(probability)
        else:
            contact = (probability >= .5).to(probability)
        if straight_through:
            contact = contact + (probability-probability.detach())
    else:
        raise ValueError("contact mode must be sample, probability, or threshold")
    return torch.cat([output[:, :6], contact], 1)


def contact_record(output, rng=None, straight_through=False, mode="sample"):
    _check(output, RECORD_DIM, "decoded transition")
    return torch.cat([contact_state(output[:, :8], rng, straight_through, mode), output[:, 8:10],
                      contact_state(output[:, 10:], rng, straight_through, mode)], 1)


def encoded_transition(e, g, prior, state_action, terrain):
    encoding = e(state_action, terrain, prior)
    return g(encoding.codes[:, 0], terrain), encoding


def composed_transition(e, g, prior, fake, terrain, rng=None, straight_through=False, mode="sample"):
    """Preserve the sampled G1/G2 inputs and decode the composed successor."""
    _check(fake, RECORD_DIM, "sampled generated record")
    decoded, encoding = encoded_transition(e, g, prior, fake[:, :10], terrain)
    successor = contact_state(decoded[:, 10:], rng, straight_through, mode)
    return torch.cat([fake[:, :10], successor], 1), decoded, encoding


def state_reconstruction(prediction, target, continuous_weight=1., contact_weight=1.):
    _check(prediction, STATE_DIM, "state prediction")
    _check(target, STATE_DIM, "state target")
    if any(not math.isfinite(w) or w < 0 for w in (continuous_weight, contact_weight)):
        raise ValueError("reconstruction weights must be finite and nonnegative")
    continuous = F.mse_loss(prediction[:, :6], target[:, :6])
    contact = F.binary_cross_entropy_with_logits(prediction[:, 6:8], target[:, 6:8])
    return continuous_weight*continuous+contact_weight*contact, dict(continuous=continuous, contact=contact)


def real_reconstruction(decoded, real, continuous_weight=1., contact_weight=1.):
    """Average three roles; do not weight the state roles by their dimensions."""
    _check(decoded, RECORD_DIM, "decoded transition")
    _check(real, RECORD_DIM, "real transition")
    state, st = state_reconstruction(decoded[:, :8], real[:, :8], continuous_weight, contact_weight)
    successor, sn = state_reconstruction(decoded[:, 10:], real[:, 10:], continuous_weight, contact_weight)
    action = F.mse_loss(decoded[:, 8:10], real[:, 8:10])
    return (state+action+successor)/3, dict(state=state, action=action, next_state=successor,
        state_continuous=st["continuous"], state_contact=st["contact"],
        next_state_continuous=sn["continuous"], next_state_contact=sn["contact"])


def synthetic_reconstruction(decoded, fake, continuous_weight=1., contact_weight=1.):
    """Detached sampled targets, with the caller's live E input path preserved."""
    _check(decoded, RECORD_DIM, "decoded transition")
    _check(fake, RECORD_DIM, "sampled generated transition")
    target = fake.detach()
    state, terms = state_reconstruction(decoded[:, :8], target[:, :8], continuous_weight, contact_weight)
    action = F.mse_loss(decoded[:, 8:10], target[:, 8:10])
    return (state+action)/2, dict(state=state, action=action,
                                state_continuous=terms["continuous"], state_contact=terms["contact"])


class GymTransitionDiscriminator(nn.Module):
    def __init__(self, input_dim, context_dim, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim+context_dim, width), nn.LeakyReLU(.2),
                                 nn.Linear(width, width), nn.LeakyReLU(.2),
                                 nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, observation, context):
        logits = self.net(torch.cat([observation, context], 1))
        return logits.squeeze(1), logits


class GymTransitionCritics(nn.Module):
    """Four scoring roles, three networks; common state D receives role 0/1."""
    def __init__(self, width=256, marginal_width=128, context_dim=CONTEXT_DIM):
        super().__init__()
        self.critics = nn.ModuleDict({
            "joint": GymTransitionDiscriminator(18, context_dim, width),
            "state": GymTransitionDiscriminator(8, context_dim+1, marginal_width),
            "action": GymTransitionDiscriminator(2, context_dim, marginal_width)})

    @staticmethod
    def roles():
        return ("joint", "state", "action", "next_state")

    def critic_for(self, role):
        return self.critics["state" if role == "next_state" else role]

    @staticmethod
    def observation(role, x):
        if role == "joint":
            return x
        return x[:, {"state": slice(0, 8), "action": slice(8, 10), "next_state": slice(10, 18)}[role]]

    def inputs(self, role, x, terrain):
        if role in ("state", "next_state"):
            terrain = torch.cat([terrain, terrain.new_full((len(terrain), 1), float(role == "next_state"))], 1)
        return self.observation(role, x), terrain


class DirectPredictor(nn.Module):
    """Supervised comparison only, sized near E + G3 + prior before training."""
    def __init__(self, context_dim=CONTEXT_DIM, width=None, target_parameters=None):
        super().__init__()
        inp, out = 10+context_dim, STATE_DIM
        if width is None:
            if target_parameters is None:
                # Default three-generator graph: z32, width128, MoG1024.
                target_parameters = (inp*128+128 + 2*(128*128+128) + 2*(128*32+32)
                                     + (32+context_dim)*128+128 + 128*128+128+128*8+8 + 1024*32)
            # Parameters for the two-hidden-layer MLP: w² + w*(inp+out+2) + out.
            root = (-(inp+out+2)+math.sqrt((inp+out+2)**2+4*(target_parameters-out)))/2
            candidates = {max(1, math.floor(root)), max(1, math.ceil(root))}
            width = min(candidates, key=lambda w: abs(w*w+w*(inp+out+2)+out-target_parameters))
        self.width, self.context_dim = width, context_dim
        self.net = mlp(inp, width, out)

    def forward(self, state_action, terrain):
        _check(state_action, 10, "direct state/action input")
        _check(terrain, self.context_dim, "terrain")
        return self.net(torch.cat([state_action, terrain], 1))
