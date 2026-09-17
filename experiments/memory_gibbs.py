"""Conditional opposing inference/generation joints at one physical time.

The ephemeral latent h represents the current candidate, not the persistent
particle z or D-owned history M. Inner refinements never advance M or time.
"""
from dataclasses import dataclass

import torch
from torch import nn

from experiments.memory_path import mlp
from experiments.memory_recent import LocalReader, clock_features

VERSION = 1


class GibbsReader(LocalReader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gibbs_steps = cfg.gibbs_steps
        self.gibbs_latent_dim = cfg.gibbs_latent_dim
        # Isolated initialization preserves the original reader's initialization
        # and all existing data/particle RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(56)
            self.latent_seed = nn.Linear(cfg.recipe.get('z_dim', 4), cfg.gibbs_latent_dim)
            self.inference = mlp(cfg.memory_dim+2+2*self.clock_bands,
                                 cfg.gibbs_latent_dim, cfg.gibbs_width)
            self.latent_projection = nn.Linear(cfg.gibbs_latent_dim, cfg.memory_dim, bias=False)
            with torch.no_grad():
                self.latent_projection.weight.mul_(.1)

    def initial_latent(self, z):
        return self.latent_seed(z).tanh()

    def infer(self, memory, point, *, time_index=None):
        inputs = [memory.flatten(1), point]
        if self.clock_bands:
            inputs.append(clock_features(time_index, point, self.clock_bands,
                                         self.clock_frequency, self.clock_rate))
        return self.inference(torch.cat(inputs, -1)).tanh()

    def latent_memory(self, memory, latent):
        return memory+self.latent_projection(latent).reshape_as(memory)

    def decode(self, z, memory, latent, *, time_index=None):
        return super().forward(z, self.latent_memory(memory, latent), time_index=time_index)

    def producer_latent(self, z, memory, *, time_index=None, steps=None):
        steps = self.gibbs_steps if steps is None else steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('gibbs_steps counts total decoder calls and must be positive')
        if steps == 1:
            return self.initial_latent(z)
        # Only the last inference/decoder transition has a parameter graph.
        # M/z are not globally detached: the final transition must preserve the
        # existing feedback and pair-loss derivatives through the frozen writer.
        with torch.no_grad():
            latent = self.initial_latent(z)
            for index in range(steps-1):
                point, _ = self.decode(z, memory, latent, time_index=time_index)
                if index < steps-2:
                    latent = self.infer(memory, point, time_index=time_index)
        return self.infer(memory, point.detach(), time_index=time_index)

    def joint(self, z, memory, *, time_index=None, steps=None):
        latent = self.producer_latent(z, memory, time_index=time_index, steps=steps)
        point, _ = self.decode(z, memory, latent, time_index=time_index)
        return point, latent

    def forward(self, z, memory, hidden=None, *, time_index=None):
        if hidden is not None:
            raise ValueError('GibbsReader has no persistent G state')
        point, _ = self.joint(z, memory, time_index=time_index)
        return point, None

    def readable_memory(self, z, memory, *, time_index=None):
        latent = self.producer_latent(z, memory, time_index=time_index)
        return super().readable_memory(z, self.latent_memory(memory, latent), time_index=time_index)


class GibbsCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.condition = cfg.gibbs_condition
        self.clock_bands = cfg.clock_bands if self.condition else 0
        self.clock_frequency, self.clock_rate = cfg.clock_frequency, cfg.clock_rate
        self.candidate_dim = cfg.gibbs_latent_dim+2
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(57)
            self.head = mlp(self.candidate_dim+int(self.condition)*cfg.memory_dim+2*self.clock_bands,
                            1, cfg.gibbs_width)

    def candidate(self, latent, point):
        return torch.cat((latent, point), -1)

    def score_candidate(self, candidate, anchor, time_index=None):
        inputs = [candidate]
        if self.condition:
            inputs.append(anchor.flatten(1))
            if self.clock_bands:
                inputs.append(clock_features(time_index, candidate, self.clock_bands,
                                             self.clock_frequency, self.clock_rate))
        return self.head(torch.cat(inputs, -1)).squeeze(-1)


class GibbsView(nn.Module):
    def __init__(self, critic, anchor, times=None):
        super().__init__()
        self.critic, self.anchor, self.times = critic, anchor, times

    def forward(self, candidate):
        return self.critic.score_candidate(candidate, self.anchor, self.times)


@dataclass
class GibbsBatch:
    anchor: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor
    positions: torch.Tensor
    times: torch.Tensor


def examples(cfg, generator, critic, joint_critic, observed, positions, z, times, target, mode):
    """Opposite-direction joints, with detached real-prefix conditioning.

    Real E(M,x_real) stays connected in generator mode. Fake uses the latent
    which actually produced x_fake, never the latent inferred after that point.
    """
    assert mode in ('critic', 'generator')
    from experiments.memory_handoff_scout import selected_memories
    eligible = positions.flatten() >= 4
    if not eligible.any():
        return None
    with torch.no_grad():
        anchor = selected_memories(critic.writer, observed, positions, cfg.max_prefix)[eligible]
    clock, particle = times[eligible], z[eligible]
    actual = target[eligible].detach()
    with torch.set_grad_enabled(mode == 'generator'):
        inferred = generator.infer(anchor, actual, time_index=clock)
        proposed, producer = generator.joint(particle, anchor, time_index=clock)
        real = joint_critic.candidate(inferred, actual)
        fake = joint_critic.candidate(producer, proposed)
    return GibbsBatch(anchor, real, fake, positions.flatten()[eligible], clock)


def critic_objective(gan, penalty, critic, batch, step):
    view = GibbsView(critic, batch.anchor, batch.times)
    real, fake = view(batch.real), view(batch.fake)
    adv = gan.d_loss(real, fake)
    reg = penalty(view, batch.real, batch.fake, step=step)
    stats = {'gibbs_rank': (real > fake).float().mean().detach(),
             'gibbs_margin': (real-fake).mean().detach()}
    return adv, reg, stats


def generator_objective(gan, critic, batch):
    view = GibbsView(critic, batch.anchor, batch.times)
    # Inference is cooperative with generation: both opposing joints connected.
    return gan.g_loss(view(batch.fake), view(batch.real))


def ramp(cfg, step):
    return min(1., step/max(1, cfg.gibbs_ramp_steps))


def metadata(cfg):
    return {
        'enabled': bool(cfg.gibbs_latent_dim), 'version': VERSION,
        'joint_enabled': bool(cfg.gibbs_weight), 'weight': cfg.gibbs_weight,
        'latent_dimension': cfg.gibbs_latent_dim, 'decoder_calls_per_generator': cfg.gibbs_steps,
        'condition': 'detached clean real-prefix memory and same clock; no particle' if cfg.gibbs_condition else 'none',
        'real': '(E(M,x_real,t), x_real), both inference and generation cooperate against K',
        'fake': '(h_producer, P(z,M,h_producer,t)); h is before final decoding',
        'latent': 'ephemeral tanh state; initialized deterministically from the same fixed particle each physical step',
        'refinement': 'steps counts total decoder calls; identical M/z/clock throughout; no writer calls',
        'gradient': 'warmup inner iterations detached; final E->decoder connected; steps=1 has seed->decoder only',
        'seed_projection': 'learned for steps=1; fixed initialized projection for steps>1 because warmup is detached; particle still learns via final decoder',
        'real_gradient': 'inference connected in G phase; no detach of real joint score',
        'writer': 'existing main D objectives only; detached conditioning for joint objective',
        'penalty': 'public default exact B-cap in concatenated (h,x) coordinates, cached M and clock',
        'runtime': 'same refinement as training; no extra persistent memory or fresh noise',
        'scope': 'conditional deterministic-particle adaptation, not the stochastic unconditional GibbsNet chain',
        'additional_generated_writes': 0,
    }
