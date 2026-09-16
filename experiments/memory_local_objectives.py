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


def mismatch_indices(observed, positions, kind, *, max_position=None):
    """Other-episode donor selected by last observation only, never future targets.

    Nearby endpoints make a harder history test than arbitrary shuffled points.
    These are mismatched examples, not guaranteed impossible events under noise.
    """
    rows = torch.arange(len(observed), device=observed.device)[:, None].expand_as(positions).flatten()
    pos = positions.flatten()
    eligible = pos >= 4
    if max_position is not None:
        eligible = eligible & (pos <= max_position)
    valid = torch.nonzero(eligible, as_tuple=True)[0]
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


def mismatch_contexts(cfg, generator, critic, observed, positions, times, z, valid, step):
    """One independent replacement write, strictly before the ranked target.

    The proposal is detached; D's writer remains connected unless this objective's
    writer-gradient control is disabled. Mixed contexts average losses, not states.
    No new random stream or continuation targets enter the memory construction.
    """
    from experiments import memory_handoff_scout as h
    if cfg.mismatch_context == 'clean':
        clean = h.selected_memories(critic.writer, observed, positions, cfg.max_prefix)[valid]
        contexts = [(1., clean)]
    else:
        previous_positions = (positions-1).clamp_min(0)
        snapshots = h.selected_memories(critic.writer, observed,
            torch.cat((positions, previous_positions), 1), cfg.max_prefix)
        snapshots = snapshots.reshape(len(observed), 2*positions.shape[1], cfg.memory_dim//4, 4)
        clean = snapshots[:, :positions.shape[1]].flatten(0, 1)[valid]
        previous = snapshots[:, positions.shape[1]:].flatten(0, 1)[valid]
        with torch.no_grad():
            proposed, _ = h.local_point(generator, z[valid], previous, times[valid]-1)
        rows = torch.arange(len(observed), device=observed.device)[:, None]
        actual = observed[rows, previous_positions].flatten(0, 1)[valid]
        strength = cfg.mismatch_write_strength*min(1., step/max(1, cfg.mismatch_ramp_steps))
        replacement = ((1-strength)*actual+strength*proposed).detach()
        explored = critic.writer.write(previous, replacement)
        contexts = ([(.5, clean), (.5, explored)] if cfg.mismatch_context == 'mixed'
                    else [(1., explored)])
    return [(weight, memory if cfg.mismatch_writer_grad else memory.detach())
            for weight, memory in contexts]


def mismatch_objective(cfg, critic, observed, real, positions, times, gan, penalty, step,
                       *, generator=None, z=None):
    from experiments import memory_handoff_scout as h
    valid, donor = mismatch_indices(observed, positions, cfg.mismatch_kind)
    if not len(valid):
        zero = next(critic.parameters()).sum()*0
        return zero, zero, {'mismatch_rank_accuracy': zero.detach()}
    contexts = mismatch_contexts(cfg, generator, critic, observed, positions, times, z, valid, step)
    rows = torch.arange(len(real), device=real.device)[:, None]
    targets = real[rows, positions].flatten(0, 1)
    actual, wrong = targets[valid], targets[donor]
    loss, reg, accuracy = 0., 0., 0.
    for weight, memory in contexts:
        view = h.CandidateView(critic, memory, times[valid])
        real_score, wrong_score = view(actual), view(wrong)
        loss = loss+weight*gan.d_loss(real_score, wrong_score)
        reg = reg+weight*penalty(view, actual, wrong, step=step)
        accuracy = accuracy+weight*(real_score > wrong_score).float().mean()
    return loss, reg, {'d_mismatch': loss.detach(),
        'mismatch_rank_accuracy': accuracy.detach()}


class FutureView(nn.Module):
    def __init__(self, critic, memory, times):
        super().__init__()
        self.critic, self.memory, self.times = critic, memory, times

    def forward(self, points):
        return self.critic.score_future(points, self.memory, self.times)


class FutureRankView(nn.Module):
    """The ordinary G-facing candidate head, queried at an explicit future horizon."""
    def __init__(self, critic, memory, times, horizon):
        super().__init__()
        self.critic, self.memory, self.times, self.horizon = critic, memory, times, horizon

    def forward(self, candidate):
        return self.critic.score_candidate(candidate, self.memory, self.times, horizon=self.horizon)


def future_rank_contexts(cfg, generator, critic, observed, positions, times, z, valid, step):
    # Reuse the audited causal replacement mechanics; the original mismatch
    # objective and its gradient policy remain independent and unchanged.
    context_cfg = replace(cfg, mismatch_weight=1., mismatch_context=cfg.future_rank_context,
        mismatch_write_strength=cfg.future_rank_strength,
        mismatch_ramp_steps=cfg.future_rank_ramp_steps, mismatch_writer_grad=True)
    contexts = mismatch_contexts(context_cfg, generator, critic, observed, positions, times, z, valid, step)
    if cfg.future_rank_context != 'clean' and not cfg.future_rank_explored_grad:
        weight, explored = contexts[-1]
        contexts[-1] = weight, explored.detach()
    return contexts


def future_rank_objective(cfg, generator, critic, observed, real, positions, times, z,
                          gan, penalty, step):
    """Recognize original-episode futures after at most one detached proposal write.

    Offsets are independent candidate queries, not generated future trajectories.
    Both donor and anchor pools exclude targets whose furthest query is unavailable.
    """
    valid, donor = mismatch_indices(observed, positions, cfg.mismatch_kind,
        max_position=real.shape[1]-1-cfg.future_rank_offsets[-1])
    if not len(valid):
        zero = next(critic.parameters()).sum()*0
        return zero, zero, {'future_rank_accuracy': zero.detach()}
    contexts = future_rank_contexts(cfg, generator, critic, observed, positions, times, z, valid, step)
    rows = torch.arange(len(real), device=real.device)[:, None].expand_as(positions).flatten()
    pos = positions.flatten()
    loss, reg, accuracy = 0., 0., 0.
    for horizon in cfg.future_rank_offsets:
        actual = real[rows[valid], pos[valid]+horizon]
        wrong = real[rows[donor], pos[donor]+horizon]
        for context_weight, memory in contexts:
            weight = context_weight/len(cfg.future_rank_offsets)
            view = FutureRankView(critic, memory, times[valid], horizon)
            real_score, wrong_score = view(actual), view(wrong)
            loss = loss+weight*gan.d_loss(real_score, wrong_score)
            reg = reg+weight*penalty(view, actual, wrong, step=step)
            accuracy = accuracy+weight*(real_score > wrong_score).float().mean()
    return loss, reg, {'d_future_rank': loss.detach(), 'future_rank_accuracy': accuracy.detach()}


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
        'future_rank': {'weight': cfg.future_rank_weight, 'offsets': cfg.future_rank_offsets,
            'bands': cfg.future_rank_bands, 'context': cfg.future_rank_context,
            'write_strength': cfg.future_rank_strength, 'ramp_steps': cfg.future_rank_ramp_steps,
            'explored_writer_grad': cfg.future_rank_explored_grad,
            'context_weights': [.5, .5] if cfg.future_rank_context == 'mixed' else [1.],
            'head': 'shared point head; zero-initialized bias-free horizon projection before first activation',
            'horizon_features': 'sin/cos-minus-one, dyadic radians per step divided by 16; exactly zero at runtime horizon 0',
            'targets': 'original episode observations at anchor+offset; different-episode donor at donor_anchor+same offset',
            'eligibility': 'filter both anchors and donors: anchor>=4 and anchor+max(offsets)<train_length; no clamping',
            'write': 'replace last prefix observation; one detached G proposal with fixed z and anchor clock minus 1',
            'writer': 'clean context connected; only explored context detached when explored_writer_grad is false',
            'loss': 'D-only paired GAN ranking; average contexts and horizons, normalize existing D loss and default B-cap by 1+weight',
            'penalty': 'candidate point coordinates only; same cached memory and horizon',
            'cost': 'one additional real-prefix encoding; nonclean adds one D-phase G call and one independent write',
            'runtime': 'unchanged G, fixed particle, horizon zero; no expert or extra state'},
        'mismatch': {'weight': cfg.mismatch_weight, 'kind': cfg.mismatch_kind,
            'context': cfg.mismatch_context, 'write_strength': cfg.mismatch_write_strength,
            'ramp_steps': cfg.mismatch_ramp_steps, 'writer_grad': cfg.mismatch_writer_grad,
            'context_weights': [.5, .5] if cfg.mismatch_context == 'mixed' else [1.],
            'write': 'replace last prefix observation; one detached G proposal, same z, clock target-1',
            'writer': 'connected prefix and replacement write' if cfg.mismatch_writer_grad else 'detached for this objective only',
            'cost': 'additional real-prefix encoding; explored/mixed add one D-phase G call and one independent write',
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
