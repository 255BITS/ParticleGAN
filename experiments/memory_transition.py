"""Local adversarial successor matching; one write from a real-prefix anchor.

The auxiliary critic is training-only. The runtime remains G(z,M,t), W(M,x).
Each phase constructs an independent branch: generated states are never carried
between updates or fed into another training branch.
"""
from dataclasses import dataclass

import torch
from torch import nn

from experiments.memory_path import mlp
from experiments.autonomous_memory import frozen

VERSION = 1


def enabled(cfg):
    return bool(cfg.transition_weight or cfg.transition_writer_weight or cfg.transition_critic_only)


class TransitionCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.condition = cfg.transition_condition
        self.include_x = cfg.transition_include_x
        self.space = cfg.transition_space
        self.mismatch_weight = cfg.transition_mismatch_weight
        feature_dim = 2 if self.space == 'read' else cfg.memory_dim
        self.candidate_dim = feature_dim + 2*int(self.include_x)
        # Enabling an unused auxiliary critic cannot perturb the baseline RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(55)
            self.head = mlp(self.candidate_dim + cfg.memory_dim*int(self.condition),
                            1, cfg.transition_width)

    def candidate(self, point, feature):
        """Pack a successor memory or successor read, according to self.space."""
        flat = feature.flatten(1)
        return torch.cat((point, flat), -1) if self.include_x else flat

    def score_candidate(self, candidate, anchor):
        inputs = torch.cat((candidate, anchor.flatten(1)), -1) if self.condition else candidate
        return self.head(inputs).squeeze(-1)


class TransitionView(nn.Module):
    def __init__(self, critic, anchor):
        super().__init__()
        self.critic, self.anchor = critic, anchor

    def forward(self, candidate):
        return self.critic.score_candidate(candidate, self.anchor)


@dataclass
class TransitionBatch:
    anchor: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor
    positions: torch.Tensor
    times: torch.Tensor
    episodes: torch.Tensor | None = None


def examples(cfg, generator, critic, transition, observed, positions, z, times, target, mode):
    """Return aligned local branches with explicit parameter-gradient ownership.

    All anchors and real successors are detached. In writer mode G proposals
    are also detached, leaving only the fake W.write connected. Generator mode
    preserves G -> frozen W -> K input derivatives; callers freeze W and K.
    Critic mode constructs the whole batch without a G/W graph.
    """
    assert mode in ('critic', 'writer', 'generator')
    # Lazy import avoids a trainer/helper import cycle.
    from experiments.memory_handoff_scout import selected_memories, local_point
    eligible = positions.flatten() >= 4
    if not eligible.any():
        return None
    latent, clock = z[eligible], times[eligible]
    with torch.no_grad():
        anchor = selected_memories(critic.writer, observed, positions, cfg.max_prefix)[eligible]
        actual = target[eligible].detach()
        real_successor = critic.writer.write(anchor, actual)
        real_feature = real_successor
        if transition.space == 'read':
            real_feature, _ = local_point(generator, latent.detach(), real_successor, clock+1)
        real = transition.candidate(actual, real_feature).detach()
    with torch.set_grad_enabled(mode == 'generator'):
        proposed, _ = local_point(generator, latent, anchor, clock)
    with torch.set_grad_enabled(mode != 'critic'):
        successor = critic.writer.write(anchor, proposed)
        fake_feature = successor
        if transition.space == 'read':
            # The current G is a frozen differentiable measurement instrument.
            # No reader-parameter or direct successor-particle gradient; the
            # derivative reaches only the first proposal or the fake write.
            with frozen(generator):
                fake_feature, _ = local_point(generator, latent.detach(), successor, clock+1)
        fake = transition.candidate(proposed, fake_feature)
    episodes = torch.arange(len(positions), device=positions.device).repeat_interleave(positions.shape[1])[eligible]
    return TransitionBatch(anchor, real, fake, positions.flatten()[eligible], clock, episodes)


def generator_objective(gan, transition, batch):
    view = TransitionView(transition, batch.anchor)
    return gan.g_loss(view(batch.fake), view(batch.real))


def other_episode_donors(episodes):
    """First eligible entry from the next episode, with circular wraparound.

    Entries retain row-major order after filtering. Selection reads episode IDs
    only, never points, memories, targets, or future observations.
    """
    if episodes is None or len(episodes) < 2:
        return None
    unique, inverse, counts = torch.unique_consecutive(episodes, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        return None
    return counts.cumsum(0)[inverse] % len(episodes)


def critic_objective(gan, penalty, transition, batch, step):
    view = TransitionView(transition, batch.anchor)
    real_score, fake_score = view(batch.real), view(batch.fake)
    adv = gan.d_loss(real_score, fake_score)
    reg = penalty(view, batch.real, batch.fake, step=step)
    stats = {
        'transition_rank': (real_score > fake_score).float().mean().detach(),
        'transition_margin': (real_score-fake_score).mean().detach(),
    }
    if transition.mismatch_weight:
        donors = other_episode_donors(batch.episodes)
        if donors is not None:
            negative = batch.real[donors].detach()
            negative_score = view(negative)
            mismatch = gan.d_loss(real_score, negative_score)
            mismatch_reg = penalty(view, batch.real, negative, step=step)
            weight = transition.mismatch_weight
            adv = (adv+weight*mismatch)/(1+weight)
            reg = (reg+weight*mismatch_reg)/(1+weight)
            stats.update(transition_mismatch=mismatch.detach(),
                         transition_mismatch_rank=(real_score > negative_score).float().mean().detach())
    return adv, reg, stats


def ramp(cfg, step):
    return min(1., step/max(1, cfg.transition_ramp_steps))


def metadata(cfg):
    return {
        'enabled': enabled(cfg), 'version': VERSION,
        'g_weight': cfg.transition_weight, 'writer_weight': cfg.transition_writer_weight,
        'ramp_steps': cfg.transition_ramp_steps, 'critic_only': cfg.transition_critic_only,
        'condition': cfg.transition_condition, 'include_x': cfg.transition_include_x,
        'space': cfg.transition_space, 'mismatch_weight': cfg.transition_mismatch_weight,
        'mismatch': 'K-only other-episode real candidates; row-major next episode, no target-based selection; normalized by 1+weight',
        'anchor': 'detached clean real-prefix memory, positions >= 4; no jitter or replacement',
        'real': 'observed next point and detached real successor from current writer',
        'fake': 'same particle and clock; one generated point and one writer update',
        'critic_gradients': 'K only; real/fake/anchor all detached',
        'g_gradients': 'G and learned particles, through frozen writer and K; anchor/real detached',
        'writer_gradients': 'fake write only; real successor, anchor and generated point detached; K frozen',
        'optimizer': 'separate K Adam at public recipe D learning rate/betas; writer uses existing D Adam',
        'weighting': 'additive G/writer losses; existing objectives unchanged; K loss unweighted',
        'penalty': 'public recipe exact B-cap in joint candidate coordinates, cached detached anchor',
        'candidate_dimension': (2 if cfg.transition_space == 'read' else cfg.memory_dim)+2*int(cfg.transition_include_x),
        'read_space': 'current G at same detached particle and t+1, parameters frozen but input derivative retained; real read detached; only first proposal receives G gradients',
        'generator_calls_per_phase': 3 if cfg.transition_space == 'read' else 1,
        'read_space_target': 'current G after real write, not an actual future observation; adversarial read-consistency',
        'runtime': 'auxiliary critic unused; no persistent generated training states',
        'sequential_generated_writes': int(enabled(cfg)),
    }
