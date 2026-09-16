"""D-owned GRU state plus explicit recent observations in one memory tensor."""
from dataclasses import replace
import torch
from torch import nn
from experiments import memory_scout as base


class SlowFastWriter(base.Writer):
    """One D-owned GRU; selected coordinates receive damped updates.

    Both groups read the complete previous state. The rate is per observation,
    not a training learning-rate multiplier, and adds no learned parameters.
    """
    def __init__(self, memory_dim, slow_dim, slow_rate):
        super().__init__('gru', memory_dim)
        self.slow_dim, self.slow_rate = slow_dim, slow_rate

    def write(self, memory, point):
        proposed = super().write(memory, point).flatten(1)
        old = memory.flatten(1)
        slow = old[:, :self.slow_dim]+self.slow_rate*(proposed[:, :self.slow_dim]-old[:, :self.slow_dim])
        return torch.cat((slow, proposed[:, self.slow_dim:]), -1).reshape_as(memory)


class RecentWriter(nn.Module):
    def __init__(self, memory_dim, recent_points, slow_dim=0, slow_rate=1.):
        super().__init__()
        self.memory_dim = memory_dim
        self.recent_points = recent_points
        self.learned_dim = memory_dim-2*recent_points
        assert self.learned_dim > 0 and self.learned_dim % 4 == 0
        self.core = (SlowFastWriter(self.learned_dim, slow_dim, slow_rate) if slow_dim
                     else base.Writer('gru', self.learned_dim))

    def initial(self, batch_like):
        return batch_like.new_zeros(len(batch_like), self.memory_dim//4, 4)

    def write(self, memory, point):
        flat = memory.flatten(1)
        learned = flat[:, :self.learned_dim].reshape(-1, self.learned_dim//4, 4)
        learned = self.core.write(learned, point).flatten(1)
        recent = torch.cat((point, flat[:, self.learned_dim:-2]), -1)
        return torch.cat((learned, recent), -1).reshape(-1, self.memory_dim//4, 4)


def clock_features(time_index, reference, bands, frequency, rate=1.):
    """Bounded external time coordinate, in radians per observation step."""
    if time_index is None:
        raise ValueError('Clock-enabled readers require an explicit time_index')
    times = torch.as_tensor(time_index, device=reference.device, dtype=reference.dtype)
    times = times.expand(len(reference)).reshape(-1, 1)
    frequencies = frequency*2.**torch.arange(bands, device=reference.device, dtype=reference.dtype)
    angles = times*rate*frequencies
    return torch.cat((angles.sin(), angles.cos()), -1)


class LocalReader(base.Reader):
    def __init__(self, cfg):
        self.clock_bands = getattr(cfg, 'clock_bands', 0)
        self.clock_frequency = getattr(cfg, 'clock_frequency', .03125)
        self.clock_rate = getattr(cfg, 'clock_rate', 1.)
        reader_cfg = cfg
        if self.clock_bands:
            reader_cfg = replace(cfg, recipe={**cfg.recipe,
                'z_dim': cfg.recipe.get('z_dim', 4)+2*self.clock_bands})
        super().__init__(reader_cfg)
        self.recent_start = cfg.memory_dim-2*cfg.recent_points
        self.residual_output = cfg.residual_output
        self.output_bound = cfg.output_bound
        self.memory_adapter = None
        self.proposal_adapter = getattr(cfg, 'g_memory_adapter', 'none') == 'proposal'
        if getattr(cfg, 'g_memory_adapter', 'none') in ('residual', 'proposal'):
            # Identity initialization keeps the initial GAN identical to its
            # control. Adapter initialization does not consume the main RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(50)
                self.memory_adapter = nn.Sequential(
                    nn.Linear(cfg.memory_dim+2*int(self.proposal_adapter), cfg.adapter_width), nn.SiLU(),
                    nn.Linear(cfg.adapter_width, cfg.adapter_bottleneck), nn.SiLU(),
                    nn.Linear(cfg.adapter_bottleneck, cfg.memory_dim))
                nn.init.zeros_(self.memory_adapter[-1].weight)
                nn.init.zeros_(self.memory_adapter[-1].bias)

    def translate_memory(self, memory, proposal=None):
        if self.memory_adapter is None:
            return memory
        flat = memory.flatten(1)
        inputs = flat
        if self.proposal_adapter:
            if proposal is None:
                raise ValueError('Proposal adapter requires the unrefined point; use readable_memory for diagnostics')
            inputs = torch.cat((flat, proposal), -1)
        return (flat+self.memory_adapter(inputs)).reshape_as(memory)

    def _with_clock(self, z, time_index):
        if self.clock_bands:
            clock = clock_features(time_index, z, self.clock_bands, self.clock_frequency, self.clock_rate)
            z = torch.cat((z, clock), -1)
        return z

    def _read(self, z, memory, hidden, raw_memory):
        point, hidden = super().forward(z, memory, hidden)
        if self.output_bound:
            # Smooth output parameterization; no gradient clipping. For residual
            # output this bounds each increment, including the first from M=0.
            point = self.output_bound*torch.tanh(point/self.output_bound)
        if self.residual_output:
            point = point+raw_memory.flatten(1)[:, self.recent_start:self.recent_start+2]
        return point, hidden

    def _readable(self, z, memory, hidden=None):
        proposal = None
        if self.proposal_adapter and self.memory_adapter is not None:
            proposal, _ = self._read(z, memory, hidden, memory)
        return self.translate_memory(memory, proposal)

    def readable_memory(self, z, memory, *, time_index=None):
        """Actual translated read, including proposal conditioning when enabled."""
        return self._readable(self._with_clock(z, time_index), memory)

    def forward(self, z, memory, hidden=None, *, time_index=None):
        z = self._with_clock(z, time_index)
        # Both passes use the same raw D state, particle and clock. No writeback.
        return self._read(z, self._readable(z, memory, hidden), hidden, memory)
