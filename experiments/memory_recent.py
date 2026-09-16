"""D-owned GRU state plus explicit recent observations in one memory tensor."""
from dataclasses import replace
import torch
from torch import nn
from experiments import memory_scout as base


class RecentWriter(nn.Module):
    def __init__(self, memory_dim, recent_points):
        super().__init__()
        self.memory_dim = memory_dim
        self.recent_points = recent_points
        self.learned_dim = memory_dim-2*recent_points
        assert self.learned_dim > 0 and self.learned_dim % 4 == 0
        self.core = base.Writer('gru', self.learned_dim)

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

    def forward(self, z, memory, hidden=None, *, time_index=None):
        if self.clock_bands:
            clock = clock_features(time_index, z, self.clock_bands, self.clock_frequency, self.clock_rate)
            z = torch.cat((z, clock), -1)
        point, hidden = super().forward(z, memory, hidden)
        if self.output_bound:
            # Smooth output parameterization; no gradient clipping. For residual
            # output this bounds each increment, including the first from M=0.
            point = self.output_bound*torch.tanh(point/self.output_bound)
        if self.residual_output:
            point = point+memory.flatten(1)[:, self.recent_start:self.recent_start+2]
        return point, hidden
