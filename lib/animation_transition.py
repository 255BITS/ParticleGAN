"""Three generators for a sprite animation: z -> (G1 st, G2 st+1, G3 frame of st).

Records are flat [st(6), st+1(6), frame(1024)] rows in scaled state units and
[-1, 1] pixels, so gradient penalties see one tensor per critic. E(st) -> z
routes through the shared MoG prior; the dream loop is
    st -> E -> z -> (G3 -> gt, G2 -> st+1) -> E -> ...
"""
import torch
from torch import nn
from torch.nn import functional as F

from particlegan.autoencoder import particle_ae
from lib.sprite_animation import IMAGE_SIZE, STATE_DIM


PIXELS = IMAGE_SIZE * IMAGE_SIZE
RECORD_DIM = 2 * STATE_DIM + PIXELS
OFFSET_BOUND = 3.


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.LeakyReLU(.2),
                         nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, out))


def split(record):
    return record[:, :STATE_DIM], record[:, STATE_DIM:2 * STATE_DIM], record[:, 2 * STATE_DIM:]


def join(state, successor, frame):
    return torch.cat([state, successor, frame.reshape(len(frame), -1)], 1)


class StateScaler(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).clone())

    @classmethod
    def fit(cls, states):
        return cls(states.mean(0), states.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, x):
        return (x - self.mean) / self.scale

    def inverse(self, x):
        return x * self.scale + self.mean


class FrameDecoder(nn.Module):
    def __init__(self, inp, channels=64):
        super().__init__()
        self.c = channels
        self.fc = nn.Linear(inp, 2 * channels * 16)
        self.net = nn.Sequential(nn.LeakyReLU(.2),
            nn.ConvTranspose2d(2 * channels, channels, 4, 2, 1), nn.LeakyReLU(.2),     # 8
            nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1), nn.LeakyReLU(.2),    # 16
            nn.ConvTranspose2d(channels // 2, 1, 4, 2, 1), nn.Tanh())                  # 32

    def forward(self, h):
        return self.net(self.fc(h).view(len(h), 2 * self.c, 4, 4)).flatten(1)


class FrameFeatures(nn.Module):
    def __init__(self, channels=32, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 4, 2, 1), nn.LeakyReLU(.2),                  # 16
            nn.Conv2d(channels, 2 * channels, 4, 2, 1), nn.LeakyReLU(.2),       # 8
            nn.Conv2d(2 * channels, 4 * channels, 4, 2, 1), nn.LeakyReLU(.2),   # 4
            nn.Flatten(), nn.Linear(4 * channels * 16, out), nn.LeakyReLU(.2))

    def forward(self, frame):
        return self.net(frame.view(len(frame), 1, IMAGE_SIZE, IMAGE_SIZE))


class AnimationGenerator(nn.Module):
    """Shared trunk; G1/G2 state heads and a G3 deconvolution frame head."""
    def __init__(self, z_dim=32, width=256, channels=64):
        super().__init__()
        self.z_dim = z_dim
        self.trunk = nn.Sequential(nn.Linear(z_dim, width), nn.LeakyReLU(.2),
                                   nn.Linear(width, width), nn.LeakyReLU(.2))
        self.g1, self.g2 = mlp(width, width, STATE_DIM), mlp(width, width, STATE_DIM)
        self.g3 = FrameDecoder(width, channels)

    def forward(self, z):
        h = self.trunk(z)
        return join(self.g1(h), self.g2(h), self.g3(h))


class AnimationEncoder(nn.Module):
    """E(st) -> particle code; returns the raw offset for bound-saturation health."""
    def __init__(self, z_dim=32, width=256, temperature=.25):
        super().__init__()
        self.temperature = temperature
        self.features = mlp(STATE_DIM, width, width)
        self.query, self.offset = nn.Linear(width, z_dim), nn.Linear(width, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, state, prior):
        features = self.features(state)
        query = F.layer_norm(self.query(features), (self.query.out_features,))
        offset = self.offset(features)
        return particle_ae(query, offset, prior, temperature=self.temperature, distance_reduction="sum",
                           offset_bound=OFFSET_BOUND), offset


def encoded_step(e, g, prior, state):
    """One dream step from scaled st: decoded record, encoding and raw offset."""
    encoding, offset = e(state, prior)
    return g(encoding.codes[:, 0]), encoding, offset


def composed(e, g, prior, fake, detach_input=False):
    """G(E(G1(z))) keeping the sampled st; detach_input stops E pushing G1 outward."""
    state = split(fake)[0]
    decoded, encoding, offset = encoded_step(e, g, prior, state.detach() if detach_input else state)
    _, successor, frame = split(decoded)
    return join(state, successor, frame), decoded, encoding


def rollout(e, g, prior, state, n):
    """Dream n steps from scaled st: returns scaled states [B, n+1, 6] and frames [B, n, 1024]."""
    states, frames = [state], []
    for _ in range(n):
        decoded = encoded_step(e, g, prior, states[-1])[0]
        _, successor, frame = split(decoded)
        states.append(successor)
        frames.append(frame)
    return torch.stack(states, 1), torch.stack(frames, 1)


class _Head(nn.Module):
    def __init__(self, inp, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, width), nn.LeakyReLU(.2),
                                 nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class StateCritic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = _Head(STATE_DIM + 1, width)

    def forward(self, x):  # [state, role flag]
        return self.head(x)


class FrameCritic(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.features, self.head = FrameFeatures(channels, 256), nn.Linear(256, 1)

    def forward(self, frame):
        return self.head(self.features(frame)).squeeze(1)


class JointCritic(nn.Module):
    def __init__(self, width=256, channels=32):
        super().__init__()
        self.features = FrameFeatures(channels, 256)
        self.head = _Head(256 + 2 * STATE_DIM, width)

    def forward(self, record):
        state, successor, frame = split(record)
        return self.head(torch.cat([self.features(frame), state, successor], 1))


class AnimationCritics(nn.Module):
    """Roles joint/state/next_state/image; one state critic scores both state roles."""
    def __init__(self, width=256, marginal_width=128, channels=32, joint=True):
        super().__init__()
        critics = dict(state=StateCritic(marginal_width), image=FrameCritic(channels))
        if joint:
            critics["joint"] = JointCritic(width, channels)
        self.critics = nn.ModuleDict(critics)

    def roles(self):
        return (("joint",) if "joint" in self.critics else ()) + ("state", "next_state", "image")

    def critic_for(self, role):
        return self.critics["state" if role == "next_state" else role]

    @staticmethod
    def inputs(role, record):
        state, successor, frame = split(record)
        if role == "joint":
            return record
        if role == "image":
            return frame
        x = state if role == "state" else successor
        return torch.cat([x, x.new_full((len(x), 1), float(role == "next_state"))], 1)


class DirectPredictor(nn.Module):
    """Supervised baseline st -> (st+1, gt) with the same frame decoder shape."""
    def __init__(self, width=256, channels=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(STATE_DIM, width), nn.LeakyReLU(.2),
                                   nn.Linear(width, width), nn.LeakyReLU(.2))
        self.successor, self.frame = mlp(width, width, STATE_DIM), FrameDecoder(width, channels)

    def forward(self, state):
        h = self.trunk(state)
        return join(state, self.successor(h), self.frame(h))
