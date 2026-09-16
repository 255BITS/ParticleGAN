"""Local history discrimination, prefix recovery and direct adversarial future queries.

No generated trajectories, geometric labels, or regression training targets.
Import the trainer lazily to keep its config/build entry point backward compatible.
"""
from dataclasses import replace

import torch
from torch import nn

from experiments.memory_recent import LocalReader


class QueryReader(LocalReader):
    """Shared point reader with an explicit relative query offset; runtime offset is zero."""
    def __init__(self, cfg):
        self.query_bands = cfg.future_query_bands
        reader_cfg = replace(cfg, recipe={**cfg.recipe,
            'z_dim': cfg.recipe.get('z_dim', 4)+2*self.query_bands})
        super().__init__(reader_cfg)

    def query_particle(self, z, offset=0):
        offset = torch.as_tensor(offset, device=z.device, dtype=z.dtype).expand(len(z))
        frequency = 2.**torch.arange(self.query_bands, device=z.device, dtype=z.dtype)/16
        angle = offset[:, None]*frequency
        # Exactly zero extra features in normal runtime, including after training.
        return torch.cat((z, angle.sin(), angle.cos()-1), -1)

    def forward(self, z, memory, hidden=None, *, time_index=None, query_offset=0):
        return super().forward(self.query_particle(z, query_offset), memory, hidden,
                               time_index=time_index)

    def readable_memory(self, z, memory, *, time_index=None):
        return super().readable_memory(self.query_particle(z), memory, time_index=time_index)


def observation_disturbance(cfg, observed, rng):
    if not cfg.recovery_noise:
        return None
    noise = torch.randn(observed.shape, device=observed.device, generator=rng)
    chosen = torch.rand((len(observed), 1, 1), device=observed.device,
                        generator=rng) < cfg.recovery_probability
    return noise*chosen*cfg.recovery_noise


def mismatch_indices(observed, positions, kind):
    """Other-episode donor selected by last observation only, never future targets.

    Nearby endpoints make a harder history test than arbitrary shuffled points.
    These are mismatched examples, not guaranteed impossible events under noise.
    """
    rows = torch.arange(len(observed), device=observed.device)[:, None].expand_as(positions).flatten()
    pos = positions.flatten()
    valid = torch.nonzero(pos >= 4, as_tuple=True)[0]
    if len(valid) < 2 or rows[valid].unique().numel() < 2:
        return valid[:0], valid[:0]
    with torch.no_grad():
        last = observed[rows[valid], pos[valid]-1]
        if kind == 'nearest':
            distance = torch.cdist(last, last).square()
        else:
            # Deterministic circular donor order; no extra random stream or seed sweep.
            order = torch.arange(len(valid), device=observed.device)
            distance = ((order[None, :]-order[:, None]-1) % len(valid)).float()
        distance.masked_fill_(rows[valid, None] == rows[valid][None, :], float('inf'))
        donor = distance.argmin(1)
    return valid, valid[donor]


def mismatch_objective(cfg, critic, observed, real, positions, times, gan, penalty, step):
    from experiments import memory_handoff_scout as h
    valid, donor = mismatch_indices(observed, positions, cfg.mismatch_kind)
    if not len(valid):
        zero = next(critic.parameters()).sum()*0
        return zero, zero, {'mismatch_rank_accuracy': zero.detach()}
    memory = h.selected_memories(critic.writer, observed, positions, cfg.max_prefix)[valid]
    rows = torch.arange(len(real), device=real.device)[:, None]
    targets = real[rows, positions].flatten(0, 1)
    actual, wrong = targets[valid], targets[donor]
    view = h.CandidateView(critic, memory, times[valid])
    real_score, wrong_score = view(actual), view(wrong)
    loss = gan.d_loss(real_score, wrong_score)
    reg = penalty(view, actual, wrong, step=step)
    return loss, reg, {'d_mismatch': loss.detach(),
        'mismatch_rank_accuracy': (real_score > wrong_score).float().mean().detach()}


class FutureView(nn.Module):
    def __init__(self, critic, memory, times):
        super().__init__()
        self.critic, self.memory, self.times = critic, memory, times

    def forward(self, points):
        return self.critic.score_future(points, self.memory, self.times)


def future_examples(cfg, generator, critic, observed, real, positions, z, times,
                    proposal_grad=False):
    from experiments import memory_handoff_scout as h
    starts = positions.clamp_max(real.shape[1]-1-cfg.future_offsets[-1])
    memory = h.selected_memories(critic.writer, observed, starts, cfg.max_prefix)
    clocks = times+(starts-positions).flatten()
    with torch.set_grad_enabled(proposal_grad):
        # Independent direct reads of identical state and particle, no writes.
        fake = torch.stack([generator(z, memory, time_index=clocks+k, query_offset=k)[0]
                            for k in cfg.future_offsets], 1)
    rows = torch.arange(len(real), device=real.device)[:, None]
    actual = torch.stack([real[rows, starts+k].flatten(0, 1) for k in cfg.future_offsets], 1)
    return FutureView(critic, memory, clocks), actual, fake


def metadata(cfg):
    return {
        'mismatch': {'weight': cfg.mismatch_weight, 'kind': cfg.mismatch_kind,
            'head': 'same point head used by G', 'min_prefix': 4,
            'donor': 'different episode; nearest last observation or circular shuffle; no future lookup',
            'loss': 'D-only paired GAN ranking; D loss/penalty normalized by 1+weight',
            'caveat': 'mismatched does not guarantee impossible under observation noise'},
        'recovery': {'noise': cfg.recovery_noise, 'probability': cfg.recovery_probability,
            'scope': 'independent Gaussian disturbances on all prefix observations; same draw both phases',
            'judge': 'unperturbed observed prefix, shared for real/fake',
            'branch': 'existing local pair replaced; two G outputs, one generated write',
            'cost': 'one additional real-prefix encoding per phase when enabled'},
        'future': {'weight': cfg.future_weight, 'offsets': cfg.future_offsets,
            'query_bands': cfg.future_query_bands,
            'scope': 'direct independent reads from one causal prefix with explicit offset and fixed z',
            'judge': 'joint future points with clean prefix memory; separate D head',
            'loss': 'convex GAN and default exact B-cap mixture; no regression loss',
            'runtime': 'query offset zero; no future access, no extra state or generated writes',
            'cost': 'one additional prefix encoding and len(offsets) G calls per phase'},
    }
