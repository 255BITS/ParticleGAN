"""Observation-updated G state; reads and proposal refinement never advance it."""
from dataclasses import replace

import torch
from torch import nn

from experiments.memory_recent import LocalReader


class RecurrentReader(LocalReader):
    def __init__(self, cfg):
        super().__init__(replace(cfg, recipe={**cfg.recipe,
            'z_dim': cfg.recipe.get('z_dim', 4)+cfg.g_state_dim}))
        self.g_state_dim = cfg.g_state_dim
        self.g_state_reads_d = cfg.g_state_reads_d
        self.g_use_d_memory = cfg.g_use_d_memory
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(52)
            self.state_cell = nn.GRUCell(2+cfg.memory_dim*int(cfg.g_state_reads_d), cfg.g_state_dim)

    def initial_state(self, reference):
        return reference.new_zeros(len(reference), self.g_state_dim)

    def write_state(self, state, point, memory):
        inputs = torch.cat((point, memory.flatten(1)), -1) if self.g_state_reads_d else point
        return self.state_cell(inputs, state)

    def generated_state(self, state, point, memory, features=None, *, actual=None, strength=1.):
        observation = point if actual is None else (1-strength)*actual+strength*point
        return self.write_state(state, observation, memory)

    def forward(self, z, memory, hidden=None, *, time_index=None):
        if hidden is None:
            raise ValueError('Recurrent reader requires explicit observation history state')
        if not self.g_use_d_memory:
            memory = torch.zeros_like(memory)
        point, _ = super().forward(torch.cat((z, hidden), -1), memory, time_index=time_index)
        return point, hidden

    def readable_memory(self, z, memory, *, time_index=None, hidden=None):
        if hidden is None:
            raise ValueError('Recurrent reader requires explicit observation history state')
        if not self.g_use_d_memory:
            memory = torch.zeros_like(memory)
        return super().readable_memory(torch.cat((z, hidden), -1), memory, time_index=time_index)


class InternalStateReader(RecurrentReader):
    """Separate G state, advanced from final decoder features or a matched control.

    Real observations enter through a learned encoder into the shared GRU input
    space. Generated transitions use those encoded outputs (embedded), internal
    decoder features (intent), or their equal mixture (hybrid). Reads are pure;
    only generated_state advances state, once per final emitted point.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.state_update = cfg.g_state_update
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(55)
            self.observation_encoder = nn.Sequential(nn.Linear(2, cfg.g_width), nn.SiLU())
            self.state_cell = nn.GRUCell(cfg.g_width, cfg.g_state_dim)

    def write_state(self, state, point, memory):
        return self.state_cell(self.observation_encoder(point), state)

    def _read(self, z, memory, hidden, raw_memory):
        # Config validation excludes FiLM/private state/residual outputs. Expose
        # the features of the FINAL decoder read, before its 2D output bottleneck.
        inputs = torch.cat((z, memory.flatten(1)), -1) if self.memory_concat else z
        features = self.net[:-1](inputs)
        point = self.net[-1](features)
        if self.output_bound:
            point = self.output_bound*torch.tanh(point/self.output_bound)
        return point, features

    def forward(self, z, memory, hidden=None, *, time_index=None):
        if hidden is None:
            raise ValueError('Internal state reader requires explicit G state')
        # Complete separation: neither observation encoder, decoder nor state
        # transition can access D memory, including through proposal repair.
        return LocalReader.forward(self, torch.cat((z, hidden), -1),
                                   torch.zeros_like(memory), time_index=time_index)

    def generated_state(self, state, point, memory, features=None, *, actual=None, strength=1.):
        if self.state_update == 'embedded':
            inputs = self.observation_encoder(point)
        else:
            if features is None:
                raise ValueError('Internal transition requires final decoder features')
            inputs = features
            if self.state_update == 'hybrid':
                inputs = .5*(inputs+self.observation_encoder(point))
        if actual is not None:
            # Bounded local handoff in the common GRU input space. At strength
            # zero this is exactly teacher forcing; at one exactly runtime.
            inputs = (1-strength)*self.observation_encoder(actual)+strength*inputs
        return self.state_cell(inputs, state)


def selected_states(generator, writer, observed, positions, max_prefix, state_grad):
    """Real-prefix BPTT for S, D-owned graph for M; snapshots precede targets.

    S's prefix inputs are observations and optionally pre-write M. Detaching M
    here prevents G's history encoder from training D's writer in the D phase.
    G phase freezes D externally. No generated prefix is constructed.
    """
    memory = writer.initial(observed)
    state = generator.initial_state(observed)
    memories, states = [memory], [state]
    for point in observed[:, :max_prefix].unbind(1):
        with torch.set_grad_enabled(state_grad):
            state = generator.write_state(state, point, memory.detach())
        memory = writer.write(memory, point)
        memories.append(memory)
        states.append(state)
    rows = torch.arange(len(observed), device=observed.device)[:, None]
    return (torch.stack(memories, 1)[rows, positions].flatten(0, 1),
            torch.stack(states, 1)[rows, positions].flatten(0, 1))


def training_contexts(cfg, generator, critic, observed, positions, z, jitter,
                      feedback_mask, step, proposal_grad=False, clock_origin=None,
                      feedback_strength=None, state_grad=False):
    previous_positions = (positions-1).clamp_min(0)
    indices = torch.cat((positions, previous_positions), 1) if cfg.feedback_probability else positions
    memory, state = selected_states(generator, critic.writer, observed, indices, cfg.max_prefix, state_grad)
    if not cfg.feedback_probability:
        return memory+jitter, memory+jitter, state
    count = cfg.samples_per_episode
    memory = memory.reshape(len(observed), 2*count, cfg.memory_dim//4, 4)
    state = state.reshape(len(observed), 2*count, cfg.g_state_dim)
    ordinary, previous = memory[:, :count].flatten(0, 1), memory[:, count:].flatten(0, 1)
    ordinary_s, previous_s = state[:, :count].flatten(0, 1), state[:, count:].flatten(0, 1)
    times = previous_positions if clock_origin is None else previous_positions+clock_origin
    with torch.set_grad_enabled(proposal_grad):
        proposed, features = generator(z, previous, previous_s, time_index=times.flatten())
    rows = torch.arange(len(observed), device=observed.device)[:, None]
    actual = observed[rows, previous_positions].flatten(0, 1)
    strength = (cfg.feedback_strength if feedback_strength is None else feedback_strength)*min(1., step/max(1, cfg.feedback_ramp_steps))
    replacement = (1-strength)*actual+strength*proposed
    if not proposal_grad:
        replacement = replacement.detach()
    updated = critic.writer.write(previous, replacement)
    with torch.set_grad_enabled(state_grad):
        updated_s = generator.generated_state(previous_s, proposed, previous, features,
                                              actual=actual, strength=strength)
    eligible = feedback_mask.flatten() & (positions.flatten() >= cfg.feedback_min_prefix)
    memory = torch.where(eligible[:, None, None], updated, ordinary)+jitter
    state = torch.where(eligible[:, None], updated_s, ordinary_s)
    judging = ordinary+jitter if cfg.feedback_judge_memory in ('clean', 'mixed') else memory
    return memory, judging, state


def continuation(generator, writer, z, prefix, steps, intervention=None, states=False):
    """Expert-free suffix; interventions alter reads, never stored D memory."""
    memory, state = writer.initial(z), generator.initial_state(z)
    for point in prefix.unbind(1):
        state = generator.write_state(state, point, memory)
        memory = writer.write(memory, point)
    path, history = [], []
    for t in range(steps):
        read = torch.zeros_like(memory) if intervention == 'zero' else memory.roll(1, 0) if intervention == 'shuffle' else memory
        read_s = torch.zeros_like(state) if intervention == 'g_zero' else state.roll(1, 0) if intervention == 'g_shuffle' else state
        point, features = generator(z, read, read_s, time_index=prefix.shape[1]+t)
        state = generator.generated_state(read_s, point, read, features)
        memory = writer.write(memory, point)
        path.append(point)
        if states:
            history.append(memory.flatten(1))
    return torch.stack(path, 1), torch.stack(history, 1) if states else memory
