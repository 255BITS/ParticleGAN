"""Local memory GAN scouts with optional single-write feedback; no full training rollouts."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.autonomous_memory import frozen
from experiments.memory_path import circles, mlp
from experiments import memory_scout as base
from experiments.memory_recent import RecentWriter, SlowFastWriter, LocalReader, clock_features
from experiments.memory_core_scout import evaluate
from particlegan import get_recipe, learning_rate_scale


@dataclass
class Config(base.Config):
    name: str = "handoff_dense4"
    writer: str = "gru"
    samples_per_episode: int = 4
    max_prefix: int = 63
    zero_fraction: float = .125
    context_noise: float = 0.
    memory_noise: float = 0.
    point_head: str = "concat"
    predict_weight: float = 0.
    temporal_weight: float = 0.
    feedback_probability: float = 0.
    feedback_strength: float = 1.
    feedback_min_prefix: int = 1
    feedback_ramp_steps: int = 0
    feedback_backprop: bool = False
    feedback_judge_memory: str = 'shared'
    feedback_shared_weight: float = .5
    feedback_strength_distribution: str = 'fixed'
    feedback_full_probability: float = .25
    local_pair_weight: float = 0.
    adversarial_only: bool = False
    recent_points: int = 0
    residual_output: bool = False
    output_bound: float = 0.
    clock_bands: int = 0
    clock_frequency: float = .03125
    clock_rate: float = 1.
    clock_to_d: bool = False
    clock_origin_max: int = 0
    slow_dim: int = 0
    slow_rate: float = .1
    g_memory_adapter: str = 'none'
    adapter_width: int = 64
    adapter_bottleneck: int = 16
    repair_weight: float = 0.
    repair_noise: float = .05
    repair_target: str = 'raw'
    stability_g_weight: float = 0.
    stability_d_weight: float = 0.
    stability_noise: float = .03
    stability_max_gain: float = 1.1
    dynamics_min_prefix: int = 4
    eval_prefixes: tuple = (8, 32)

    def __post_init__(self):
        super().__post_init__()
        self.eval_prefixes = tuple(self.eval_prefixes)
        assert self.writer == "gru" and not self.g_private and not self.g_film and self.read_memory
        assert self.critic == "flat" and not self.differences and self.geometry == "none"
        assert not self.frozen_writer and self.freeze_writer_at is None and self.writer_lr_mult == 1
        assert self.point_head in ("concat", "interaction")
        assert self.samples_per_episode >= 1 and 1 <= self.max_prefix < self.train_length
        assert 0 < self.zero_fraction < 1 and min(self.context_noise, self.memory_noise) >= 0
        assert tuple(self.eval_prefixes) == (8, 32), "keep common evaluation panel"
        assert min(self.predict_weight, self.temporal_weight) >= 0
        assert 0 <= self.feedback_probability <= 1 and 0 <= self.feedback_strength <= 1
        assert 1 <= self.feedback_min_prefix <= self.max_prefix and self.feedback_ramp_steps >= 0
        assert self.recent_points in (0, 2, 4) and self.memory_dim > 2*self.recent_points
        assert not self.residual_output or self.recent_points > 0
        assert self.output_bound >= 0
        assert not self.feedback_backprop or self.feedback_probability > 0
        assert 0 <= self.clock_bands <= 16 and self.clock_frequency > 0 and self.clock_rate >= 0
        assert self.clock_origin_max >= 0
        assert self.clock_bands or (not self.clock_to_d and self.clock_origin_max == 0)
        assert 0 <= self.slow_dim <= self.memory_dim-2*self.recent_points and 0 < self.slow_rate <= 1
        assert self.g_memory_adapter in ('none', 'residual', 'proposal')
        assert min(self.adapter_width, self.adapter_bottleneck) > 0
        assert min(self.repair_weight, self.stability_g_weight, self.stability_d_weight) >= 0
        assert self.repair_noise > 0 and self.stability_noise > 0 and self.stability_max_gain > 0
        assert self.repair_target in ('raw', 'translated')
        assert not self.repair_weight or self.g_memory_adapter != 'none'
        assert not self.repair_weight or self.g_memory_adapter != 'proposal'
        assert self.feedback_judge_memory in ('shared', 'clean', 'mixed')
        assert self.feedback_judge_memory == 'shared' or self.feedback_probability > 0
        assert 0 < self.feedback_shared_weight < 1
        assert self.feedback_strength_distribution in ('fixed', 'uniform', 'mild_full')
        assert self.feedback_strength_distribution == 'fixed' or self.feedback_probability > 0
        assert 0 <= self.feedback_full_probability <= 1
        assert 0 <= self.local_pair_weight < 1
        assert not self.adversarial_only or not any((self.predict_weight, self.temporal_weight,
            self.repair_weight, self.stability_g_weight, self.stability_d_weight))
        assert self.dynamics_min_prefix >= 1
        assert not (self.repair_weight or self.stability_g_weight or self.stability_d_weight) or self.dynamics_min_prefix <= self.max_prefix


class Critic(nn.Module):
    """D owns the writer, point head, and optional two-point transition head."""
    def __init__(self, cfg):
        super().__init__()
        self.clock_bands = cfg.clock_bands if cfg.clock_to_d else 0
        self.clock_frequency, self.clock_rate = cfg.clock_frequency, cfg.clock_rate
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            self.writer = (RecentWriter(cfg.memory_dim, cfg.recent_points, cfg.slow_dim, cfg.slow_rate)
                           if cfg.recent_points else
                           SlowFastWriter(cfg.memory_dim, cfg.slow_dim, cfg.slow_rate) if cfg.slow_dim
                           else base.Writer(cfg.writer, cfg.memory_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(46)
            self.handoff_head = mlp(cfg.memory_dim+2+2*self.clock_bands, 1, cfg.d_width)
        if cfg.local_pair_weight:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(51)
                self.pair_head = mlp(cfg.memory_dim+4+2*self.clock_bands, 1, cfg.d_width)
        self.interaction = cfg.point_head == "interaction"
        if self.interaction:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(47)
                self.point_embedding = nn.Linear(2, cfg.memory_dim)
                self.memory_embedding = nn.Linear(cfg.memory_dim, cfg.memory_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(48)
            if cfg.predict_weight:
                self.predict_head = mlp(cfg.memory_dim, 2, cfg.d_width)
            torch.manual_seed(49)
            if cfg.temporal_weight:
                self.temporal_head = mlp(cfg.memory_dim+2, 1, cfg.d_width)

    def score_candidate(self, candidate, memory, time_index=None):
        flat = memory.flatten(1)
        inputs = [candidate, flat]
        if self.clock_bands:
            inputs.append(clock_features(time_index, candidate, self.clock_bands,
                                         self.clock_frequency, self.clock_rate))
        score = self.handoff_head(torch.cat(inputs, -1)).squeeze(-1)
        if self.interaction:
            score = score+(self.point_embedding(candidate)*self.memory_embedding(flat)).sum(-1)/(flat.shape[-1]**.5)
        return score

    def score_pair(self, pair, memory, time_index=None):
        inputs = [pair.flatten(1), memory.flatten(1)]
        if self.clock_bands:
            inputs.append(clock_features(time_index, pair[:, 0], self.clock_bands,
                                         self.clock_frequency, self.clock_rate))
        return self.pair_head(torch.cat(inputs, -1)).squeeze(-1)


class CandidateView(nn.Module):
    def __init__(self, critic, memory, time_index=None):
        super().__init__()
        self.critic, self.memory = critic, memory
        self.time_index = time_index

    def forward(self, candidate):
        # Real, fake, and penalty all use the exact same cached memory tensor.
        return self.critic.score_candidate(candidate, self.memory, self.time_index)


class PairView(CandidateView):
    def forward(self, pair):
        return self.critic.score_pair(pair, self.memory, self.time_index)


def candidate_views(cfg, critic, memory, judging_memory, times):
    if cfg.feedback_judge_memory == 'mixed':
        return [(1-cfg.feedback_shared_weight, CandidateView(critic, judging_memory, times)),
                (cfg.feedback_shared_weight, CandidateView(critic, memory, times))]
    return [(1., CandidateView(critic, judging_memory, times))]


def sample_feedback_strength(cfg, positions, rng):
    """Draw once per update; reuse the same strengths after D's update."""
    if cfg.feedback_strength_distribution == 'fixed':
        return None
    uniform = torch.rand(positions.shape, device=positions.device, generator=rng)
    if cfg.feedback_strength_distribution == 'uniform':
        return cfg.feedback_strength*uniform.flatten()[:, None]
    return torch.where(uniform < cfg.feedback_full_probability, 1.,
                       cfg.feedback_strength).flatten()[:, None]


def local_pair_examples(cfg, generator, critic, observed, real, positions, z, times,
                        proposal_grad=False):
    """Independent local branch: real prefix -> G -> one write -> G.

    Start one point before the target, or at zero for empty-prefix targets.
    Zero targets judge the pair real[0:2], providing a local cold-start task.
    No explored point-loss history feeds this branch; no third point is generated.
    """
    starts = (positions-1).clamp_min(0)
    memory = selected_memories(critic.writer, observed, starts, cfg.max_prefix)
    latent, clock = z, times-(positions.flatten() > 0).to(times.dtype)
    with torch.set_grad_enabled(proposal_grad):
        first, _ = local_point(generator, latent, memory, clock)
        updated = critic.writer.write(memory, first)
        second, _ = local_point(generator, latent, updated, clock+1)
        fake = torch.stack((first, second), 1)
    rows = torch.arange(len(real), device=real.device)[:, None]
    actual = torch.stack((real[rows, starts], real[rows, starts+1]), 2).flatten(0, 1)
    return PairView(critic, memory, clock), actual, fake


def local_point(generator, z, memory, time_index):
    args = {'time_index': time_index} if getattr(generator, 'clock_bands', 0) else {}
    return generator(z, memory, **args)


def generator_loss(gan, view, target, fake):
    # With a generated context, real scores depend on G as well. Retain both
    # branches so common memory-only offsets cancel in the paired API loss.
    return gan.g_loss(view(fake), view(target))


def selected_memories(writer, observed, positions, max_prefix):
    """Encode real history once, then gather strictly-before-target snapshots."""
    memory = writer.initial(observed)
    history = [memory]
    for point in observed[:, :max_prefix].unbind(1):
        memory = writer.write(memory, point)
        history.append(memory)
    history = torch.stack(history, 1)
    rows = torch.arange(len(observed), device=observed.device)[:, None]
    selected = history[rows, positions]
    return selected.flatten(0, 1)


def batch_inputs(cfg, real, rngs):
    batch, samples = len(real), cfg.samples_per_episode
    positions = torch.randint(1, cfg.max_prefix+1, (batch, samples), device=real.device, generator=rngs['context'])
    positions[torch.rand(batch, samples, device=real.device, generator=rngs['context']) < cfg.zero_fraction] = 0
    # Draw augmentations even at zero strength, keeping these scouts' streams aligned.
    observed = real+cfg.context_noise*torch.randn(real.shape, device=real.device, generator=rngs['input_noise'])
    jitter = torch.randn(batch*samples, cfg.memory_dim//4, 4, device=real.device, generator=rngs['state_noise'])*cfg.memory_noise
    # Preserve the true zero state in zero-prefix examples.
    jitter = jitter*(positions.flatten()!=0)[:, None, None]
    target = real[torch.arange(batch, device=real.device)[:, None], positions].flatten(0, 1)
    return positions, observed, jitter, target


def training_memory(cfg, generator, critic, observed, positions, z, jitter, feedback_mask, step,
                    proposal_grad=False, clock_origin=None, return_reference=False, feedback_strength=None):
    """Replace at most the last observation before each target by a G point.

    Both returned states retain writer gradients. The caller chooses which state
    D uses for scoring; G's fake is detached during D training. Neither candidate
    score writes. return_reference preserves the unmodified real-history state.
    """
    if not cfg.feedback_probability:
        memory = selected_memories(critic.writer, observed, positions, cfg.max_prefix)+jitter
        return (memory, memory) if return_reference else memory
    previous_positions = (positions-1).clamp_min(0)
    snapshots = selected_memories(critic.writer, observed,
        torch.cat((positions, previous_positions), 1), cfg.max_prefix)
    snapshots = snapshots.reshape(len(observed), 2*cfg.samples_per_episode, cfg.memory_dim//4, 4)
    ordinary = snapshots[:, :cfg.samples_per_episode].flatten(0, 1)
    previous = snapshots[:, cfg.samples_per_episode:].flatten(0, 1)
    times = previous_positions if clock_origin is None else previous_positions+clock_origin
    with torch.set_grad_enabled(proposal_grad):
        proposed, _ = local_point(generator, z, previous, times.flatten())
    rows = torch.arange(len(observed), device=observed.device)[:, None]
    actual = observed[rows, previous_positions].flatten(0, 1)
    strength = (cfg.feedback_strength if feedback_strength is None else feedback_strength)*min(1., step/max(1, cfg.feedback_ramp_steps))
    replacement = (1-strength)*actual+strength*proposed
    if not proposal_grad:
        replacement = replacement.detach()
    updated = critic.writer.write(previous, replacement)
    eligible = feedback_mask.flatten() & (positions.flatten() >= cfg.feedback_min_prefix)
    memory = torch.where(eligible[:, None, None], updated, ordinary)+jitter
    return (memory, ordinary+jitter) if return_reference else memory


def training_contexts(cfg, generator, critic, observed, positions, z, jitter, feedback_mask,
                      step, proposal_grad=False, clock_origin=None, feedback_strength=None):
    """G may explore a generated write while D judges against real history.

    The clean reference is strictly before the target and is shared by both D
    candidate scores and B-cap. No clean reference is supplied to G's final read.
    """
    memory, reference = training_memory(cfg, generator, critic, observed, positions, z,
        jitter, feedback_mask, step, proposal_grad, clock_origin, return_reference=True,
        feedback_strength=feedback_strength)
    return memory, reference if cfg.feedback_judge_memory in ('clean', 'mixed') else memory


def local_auxiliary(cfg, critic, memory, real, positions, target):
    """D-only local tasks. Wrong-time points never enter the GAN negative class.

    Exclude short/empty prefixes: motion is not identifiable from those contexts.
    Targets are supervision only; the shared memory remains strictly causal.
    """
    valid = positions.flatten() >= 3
    zero = memory.sum()*0
    prediction, temporal = zero, zero
    accuracy = zero.detach()
    if valid.any():
        flat = memory[valid].flatten(1)
        correct = target[valid]
        if cfg.predict_weight:
            prediction = F.mse_loss(critic.predict_head(flat), correct)
        if cfg.temporal_weight:
            rows = torch.arange(len(real), device=real.device)[:, None]
            # Previous and three-steps-later points are on the same real orbit.
            # Near the episode end, use three steps earlier instead; never label
            # the actual target as a negative by clamping to the final point.
            before = real[rows, (positions-1).clamp_min(0)].flatten(0, 1)[valid]
            other_positions = torch.where(positions+3 < real.shape[1], positions+3, positions-3)
            after = real[rows, other_positions.clamp_min(0)].flatten(0, 1)[valid]
            pos = critic.temporal_head(torch.cat((correct, flat), -1)).squeeze(-1)
            neg = torch.cat([critic.temporal_head(torch.cat((point, flat), -1)).squeeze(-1)
                             for point in (before, after)])
            temporal = (F.softplus(-pos).mean()+F.softplus(neg).mean())/2
            accuracy = ((pos > 0).float().mean()+(neg < 0).float().mean()).detach()/2
    return cfg.predict_weight*prediction+cfg.temporal_weight*temporal, {
        'prediction_mse': prediction.detach(), 'temporal_bce': temporal.detach(),
        'temporal_accuracy': accuracy}


def memory_repair(cfg, generator, memory, positions, noise):
    """G-only read repair. Clean branch is detached; D state is never rewritten."""
    valid = positions.flatten() >= cfg.dynamics_min_prefix
    if not valid.any():
        return memory.new_zeros(())
    clean = memory.detach()[valid]
    with torch.no_grad():
        target = clean if cfg.repair_target == 'raw' else generator.translate_memory(clean)
    repaired = generator.translate_memory(clean+cfg.repair_noise*noise[valid])
    return F.mse_loss(repaired, target)


def local_stability(cfg, generator, critic, z, memory, positions, times, noise):
    """Two parallel one-step feedback branches; no generated continuation.

    Penalize mean-square finite-perturbation gain above a configured bound.
    Prefix states and particles are detached. The caller freezes the opposite
    module, retaining gradients through its operations when needed.
    """
    valid = positions.flatten() >= cfg.dynamics_min_prefix
    zero = memory.new_zeros(())
    if not valid.any():
        return zero, {'gain_rms': zero, 'active_fraction': zero}
    clean, latent, clock = memory.detach()[valid], z.detach()[valid], times[valid]
    perturbation = noise[valid]
    perturbation = cfg.stability_noise*perturbation/perturbation.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    altered = clean+perturbation
    point, _ = local_point(generator, latent, clean, clock)
    altered_point, _ = local_point(generator, latent, altered, clock)
    next_clean = critic.writer.write(clean, point)
    next_altered = critic.writer.write(altered, altered_point)
    input_mse = perturbation.square().mean((1, 2))
    output_mse = (next_altered-next_clean).square().mean((1, 2))
    gain_squared = output_mse/input_mse.clamp_min(1e-12)
    loss = (gain_squared-cfg.stability_max_gain**2).relu().mean()
    return loss, {'gain_rms': gain_squared.mean().sqrt().detach(),
                  'active_fraction': (gain_squared > cfg.stability_max_gain**2).float().mean().detach()}


def build(cfg, device):
    torch.manual_seed(42)
    generator = LocalReader(cfg).to(device)
    torch.manual_seed(44)
    critic = Critic(cfg).to(device)
    recipe = get_recipe(total_steps=cfg.schedule_steps, batch_size=cfg.batch_size,
                        num_particles=512, **cfg.recipe)
    torch.manual_seed(45)
    prior = recipe.make_prior().to(device)
    return generator, critic, prior, recipe


def train(cfg, out, device, log):
    generator, critic, prior, recipe = build(cfg, device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rates = [[group['lr'] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    rngs = {'data': torch.Generator(device=device).manual_seed(31415),
            'latent': torch.Generator(device=device).manual_seed(27182),
            'context': torch.Generator(device=device).manual_seed(16180),
            'input_noise': torch.Generator(device=device).manual_seed(16181),
            'state_noise': torch.Generator(device=device).manual_seed(16182),
            'feedback': torch.Generator(device=device).manual_seed(16183),
            'clock': torch.Generator(device=device).manual_seed(16184),
            'stability': torch.Generator(device=device).manual_seed(16185),
            'repair': torch.Generator(device=device).manual_seed(16186),
            'feedback_strength': torch.Generator(device=device).manual_seed(16187)}
    start = 0
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device, weights_only=False)
        assert not cfg.feedback_backprop or saved.get('feedback_gradient_version') == 2, \
            'Cannot resume legacy feedback-backprop checkpoints with detached real-context scores'
        old = asdict(Config(**saved['config']))
        mutable = {'name', 'steps', 'resume', 'eval_steps', 'eval_batch', 'log_every'}
        assert all(old[k] == v for k, v in asdict(cfg).items() if k not in mutable), 'resume changes training'
        for name, module in [('generator', generator), ('critic', critic), ('prior', prior)]:
            module.load_state_dict(saved[name])
        opt_g.load_state_dict(saved['opt_g'])
        opt_d.load_state_dict(saved['opt_d'])
        for name, rng in rngs.items():
            if name in saved['rngs']:
                rng.set_state(saved['rngs'][name].cpu())
            else:
                assert ((name == 'feedback' and not cfg.feedback_probability)
                        or (name == 'clock' and not cfg.clock_bands)
                        or (name == 'stability' and not (cfg.stability_g_weight or cfg.stability_d_weight))
                        or (name == 'repair' and not cfg.repair_weight)
                        or (name == 'feedback_strength' and cfg.feedback_strength_distribution == 'fixed'))
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
        start = saved['step']
        assert cfg.steps > start
    resolved = {**asdict(cfg), 'resolved_recipe': recipe.to_dict(), 'device': device,
                'cold_weight': 0., 'warm_weight': 0., 'gradient_clipping': None, 'ema': False,
                'training_objective': 'local adversarial handoff with optional feedback, D auxiliaries, G read repair, and local feedback stability',
                'auxiliary_min_prefix': 3,
                'memory_layout': {'learned_gru': cfg.memory_dim-2*cfg.recent_points,
                                  'slow_gru_coordinates': cfg.slow_dim,
                                  'recent_observation_coordinates': 2*cfg.recent_points},
                'state_diagnostic_note': 'whole-memory norms/saturation include raw coordinates when recent_points > 0',
                'clock': {'features': 'sin/cos dyadic frequencies in radians per step; no raw time',
                          'frequencies': [cfg.clock_frequency*2**i for i in range(cfg.clock_bands)],
                          'training': 'target index plus optional independent per-episode origin; shared by all sampled positions',
                          'evaluation': 'cold starts at t=0; real-prefix continuation starts at t=prefix_length',
                          'state': 'external integer counter only; no learned private G state'},
                'writer_training': 'D only', 'particle_policy': 'one fixed z per real episode, shared across sampled positions',
                'training_generator_unroll': 2 if cfg.feedback_probability else 1,
                'generator_calls_d_phase': 1+int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_d_weight)),
                'generator_calls_g_phase': 1+int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_g_weight)),
                'reader_calls_per_generator': 2 if cfg.g_memory_adapter == 'proposal' else 1,
                'reader_calls_d_phase': (1+int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_d_weight)))*(2 if cfg.g_memory_adapter == 'proposal' else 1),
                'reader_calls_g_phase': (1+int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_g_weight)))*(2 if cfg.g_memory_adapter == 'proposal' else 1),
                'scoring_memory': cfg.feedback_judge_memory,
                'proposal_adapter': {'input': 'raw D memory and unrefined point from the same particle and clock',
                                     'clock': 'enters through the point reader; no extra adapter clock input',
                                     'training': 'GAN gradients only; both reader passes connected',
                                     'write': 'only final point; repaired memory is never stored'},
                'local_dynamics': {'branches': 'two parallel one-step D.write(G(M)) evaluations; same z/time',
                                   'memory_and_particles': 'detached anchors, prefix >= dynamics_min_prefix',
                                   'gradient_ownership': 'D phase freezes G; G phase freezes D; no writer gradient from G loss',
                                   'scope': 'sampled local perturbations, not worst-case or global stability guarantee'},
                'g_memory_repair': {'persistent_state': False, 'write_back': False,
                                    'auxiliary_active': bool(cfg.repair_weight),
                                    'target': 'stop-gradient raw or translated current clean D memory' if cfg.repair_weight else None,
                                    'ownership': 'G adapter only; memory inputs detached' if cfg.repair_weight else 'GAN gradients only'},
                'fake_writes_in_training': int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_d_weight or cfg.stability_g_weight)),
                'fake_writes_d_phase': int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_d_weight)),
                'fake_writes_g_phase': int(bool(cfg.feedback_probability))+2*int(bool(cfg.stability_g_weight)),
                'fake_write_counting': 'maximum writes per phase, including parallel local stability branches; not sequential unroll depth',
                'feedback_gradient': ('two G evaluations connected through frozen D writer in G phase; detached proposal in D phase'
                                      if cfg.feedback_backprop else 'generated replacement detached'),
                'feedback_writer_d_gradient': 'real-history judging branch only' if cfg.feedback_judge_memory == 'clean' else 'shared judging branch, including eligible replacement writes',
                'feedback_gradient_version': 2,
                'feedback_target': 'original next real point as recovery target',
                'memory_timing': 'strictly before target; G optionally reads replaced last observation; D uses configured shared or clean memory, identical for both scores',
                'penalty_domain': 'candidate coordinates only; same cached memory; exact default B-cap',
                'context_rebuilds_per_update': 2,
                'point_examples_per_update': cfg.batch_size*cfg.samples_per_episode,
                'parameters': {'g': sum(p.numel() for p in generator.parameters()),
                               'd': sum(p.numel() for p in critic.parameters())}}
    resolved['local_pair'] = {
        'weight': cfg.local_pair_weight,
        'context': 'real prefix before both candidates; no future observations',
        'fake': 'two consecutive G outputs with one intervening D write; same particle',
        'branch': 'independent of point-loss exploration; never chained to that branch',
        'penalty': 'default exact B-cap in four candidate coordinates, cached prefix memory',
        'writer_gradients': 'D trains real prefix through pair score; G differentiates frozen generated write',
        'empty_prefix_targets': 'pair starts from M=0, compares generated first two points to real[0:2]',
    }
    resolved['judging_view_weights'] = ([1-cfg.feedback_shared_weight, cfg.feedback_shared_weight]
                                       if cfg.feedback_judge_memory == 'mixed' else [1.])
    if cfg.feedback_judge_memory == 'mixed':
        resolved['feedback_writer_d_gradient'] = 'weighted clean and explored judging histories'
    if cfg.local_pair_weight:
        for phase in ('d', 'g'):
            resolved[f'generator_calls_{phase}_phase'] += 2
            resolved[f'reader_calls_{phase}_phase'] += 2*resolved['reader_calls_per_generator']
            resolved[f'fake_writes_{phase}_phase'] += 1
        resolved['training_generator_unroll'] = 2
        resolved['fake_writes_in_training'] += 1
        resolved['penalty_domain'] = 'point coordinates and independent local pair coordinates; weighted default exact B-cap'
    resolved['max_sequential_generated_writes'] = int(bool(cfg.feedback_probability or cfg.local_pair_weight
        or cfg.stability_g_weight or cfg.stability_d_weight))
    (out/'config.json').write_text(json.dumps(resolved, indent=2))
    log(event='start', start_step=start, config=resolved)
    started = time.monotonic()
    for step in range(start+1, cfg.steps+1):
        scale = learning_rate_scale(step-1, cfg.schedule_steps, recipe.lr_anneal_start, recipe.lr_floor)
        for opt, rates_ in zip((opt_g, opt_d), rates):
            for group, rate in zip(opt.param_groups, rates_):
                group['lr'] = rate*scale
        real, _ = circles(cfg.batch_size, cfg.train_length, rngs['data'], device, noise=cfg.noise)
        z, indices = prior.sample(cfg.batch_size, generator=rngs['latent'])
        z = z[:, None].expand(-1, cfg.samples_per_episode, -1).flatten(0, 1)
        positions, observed, jitter, target = batch_inputs(cfg, real, rngs)
        clock_origin = torch.randint(cfg.clock_origin_max+1, (cfg.batch_size, 1),
                                     device=device, generator=rngs['clock'])
        times = (positions+clock_origin).flatten()
        feedback_mask = torch.rand(positions.shape, device=device, generator=rngs['feedback']) < cfg.feedback_probability
        feedback_strength = sample_feedback_strength(cfg, positions, rngs['feedback_strength'])
        stability_noise = (torch.randn(jitter.shape, device=device, generator=rngs['stability'])
                           if cfg.stability_g_weight or cfg.stability_d_weight else None)
        repair_noise = (torch.randn(jitter.shape, device=device, generator=rngs['repair'])
                        if cfg.repair_weight else None)
        opt_d.zero_grad(set_to_none=True)
        memory, judging_memory = training_contexts(cfg, generator, critic, observed, positions, z, jitter, feedback_mask, step,
                                                   clock_origin=clock_origin, feedback_strength=feedback_strength)
        views = candidate_views(cfg, critic, memory, judging_memory, times)
        with torch.no_grad():
            fake, _ = local_point(generator, z, memory, times)
        d_adv = sum(weight*gan.d_loss(view(target), view(fake)) for weight, view in views)
        d_reg = sum(weight*penalty(view, target, fake, step=step) for weight, view in views)
        d_pair = memory.new_zeros(())
        if cfg.local_pair_weight:
            pair = local_pair_examples(cfg, generator, critic, observed, real, positions, z, times)
            if pair is not None:
                pair_view, pair_real, pair_fake = pair
                d_pair = gan.d_loss(pair_view(pair_real), pair_view(pair_fake))
                pair_reg = penalty(pair_view, pair_real, pair_fake, step=step)
                d_adv = (1-cfg.local_pair_weight)*d_adv+cfg.local_pair_weight*d_pair
                d_reg = (1-cfg.local_pair_weight)*d_reg+cfg.local_pair_weight*pair_reg
        d_aux, diagnostics = local_auxiliary(cfg, critic, memory, real, positions, target)
        d_stability = memory.new_zeros(())
        if cfg.stability_d_weight:
            with frozen(generator):
                d_stability, stats = local_stability(cfg, generator, critic, z, memory, positions, times, stability_noise)
            diagnostics.update({f'd_stability_{key}': value for key, value in stats.items()})
        (d_adv+d_reg+d_aux+cfg.stability_d_weight*d_stability).backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        with frozen(critic):
            opt_g.zero_grad(set_to_none=True)
            # Rebuild with updated D weights; reuse the same observations/jitter.
            memory, judging_memory = training_contexts(cfg, generator, critic, observed, positions, z, jitter, feedback_mask, step,
                proposal_grad=cfg.feedback_backprop, clock_origin=clock_origin, feedback_strength=feedback_strength)
            views = candidate_views(cfg, critic, memory, judging_memory, times)
            fake, _ = local_point(generator, z, memory, times)
            g_adv = sum(weight*generator_loss(gan, view, target, fake) for weight, view in views)
            g_pair = memory.new_zeros(())
            if cfg.local_pair_weight:
                pair = local_pair_examples(cfg, generator, critic, observed, real, positions, z, times,
                                           proposal_grad=True)
                if pair is not None:
                    pair_view, pair_real, pair_fake = pair
                    g_pair = generator_loss(gan, pair_view, pair_real, pair_fake)
                    g_adv = (1-cfg.local_pair_weight)*g_adv+cfg.local_pair_weight*g_pair
            repair = memory_repair(cfg, generator, memory, positions, repair_noise) if cfg.repair_weight else memory.new_zeros(())
            g_stability = memory.new_zeros(())
            if cfg.stability_g_weight:
                g_stability, stats = local_stability(cfg, generator, critic, z, memory, positions, times, stability_noise)
                diagnostics.update({f'g_stability_{key}': value for key, value in stats.items()})
            (g_adv+spread(prior(indices.unique()))+cfg.repair_weight*repair+cfg.stability_g_weight*g_stability).backward()
            opt_g.step()
        if step == start+1 or step % cfg.log_every == 0 or step == cfg.steps:
            losses = {'d': d_adv.item(), 'g': g_adv.item(), 'penalty': d_reg.item()}
            losses.update(repair_mse=repair.item(), d_stability_loss=d_stability.item(), g_stability_loss=g_stability.item())
            if cfg.local_pair_weight:
                losses.update(d_pair=d_pair.item(), g_pair=g_pair.item())
            if feedback_strength is not None:
                eligible = (feedback_mask & (positions >= cfg.feedback_min_prefix)).flatten()
                strengths = feedback_strength.flatten()[eligible]*min(1., step/max(1, cfg.feedback_ramp_steps))
                losses['feedback_strength_mean'] = strengths.mean().item() if len(strengths) else 0.
            losses.update({key: value.item() for key, value in diagnostics.items()})
            losses['feedback_fraction'] = (feedback_mask & (positions >= cfg.feedback_min_prefix)).float().mean().item()
            if cfg.feedback_probability:
                losses['feedback_judge_gap_rms'] = (memory.detach()-judging_memory.detach()).square().mean().sqrt().item()
            if not all(np.isfinite(v) for v in losses.values()):
                raise RuntimeError(f'Nonfinite losses: {losses}')
            log(event='train', step=step, seconds=round(time.monotonic()-started, 2), **losses)
    train_seconds = time.monotonic()-started
    torch.save({'generator': generator.state_dict(), 'critic': critic.state_dict(),
                'writer': critic.writer.state_dict(), 'prior': prior.state_dict(),
                'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(),
                'rngs': {key: rng.get_state() for key, rng in rngs.items()},
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                'step': cfg.steps, 'config': asdict(cfg), 'feedback_gradient_version': 2}, out/'model.pt')
    generator.eval(), critic.eval(), prior.eval()
    metrics, paths = evaluate(generator, critic, prior, cfg, device)
    np.savez_compressed(out/'trajectories.npz', **paths)
    result = {'name': cfg.name, 'steps': cfg.steps, 'seconds': time.monotonic()-started,
              'train_seconds': train_seconds, 'seconds_per_update': train_seconds/(cfg.steps-start),
              'metrics': metrics, 'config': resolved}
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    log(event='complete', steps=cfg.steps, seconds=round(result['seconds'], 2), metrics=metrics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    provenance = {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'torch': torch.__version__, 'argv': sys.argv, 'sources': {}}
    for name in ('memory_handoff_scout.py', 'memory_recent.py', 'memory_core_scout.py', 'memory_scout.py', 'autonomous_memory.py', 'memory_path.py'):
        source = Path(__file__).with_name(name).read_bytes()
        (args.out/name).write_bytes(source)
        provenance['sources'][name] = hashlib.sha256(source).hexdigest()
    (args.out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.out/'input.json').write_bytes(args.config.read_bytes())
    with (args.out/'experiment.log').open('w', buffering=1) as stream:
        def log(**row):
            line = json.dumps({'name': cfg.name, **row}, allow_nan=False)
            stream.write(line+'\n')
            print(line, flush=True)
        train(cfg, args.out, args.device, log)


if __name__ == '__main__':
    main()
