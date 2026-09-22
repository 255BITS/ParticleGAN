"""Optional vector discriminator from the behavioral formulation study."""
import math

import torch
from torch import nn


class _SmoothFourierMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden, fourier, beta):
        super().__init__()
        self.register_buffer('freqs', torch.pi * (2. ** torch.arange(fourier, dtype=torch.float32)))
        width = in_dim + 2 * fourier * in_dim
        layers = []
        for _ in range(n_hidden):
            layers.extend((nn.Linear(width, hidden_dim), nn.Softplus(beta=beta)))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        phase = (x.unsqueeze(-1) * self.freqs).flatten(1)
        features = torch.cat([x, phase.sin(), phase.cos()], dim=1)
        return self.net(features).squeeze(-1)


class LinearSkipDiscriminator(nn.Module):
    """Smooth Fourier MLP plus an initially zero, learned raw-input linear score.

    Defaults reproduce the 2D rare-mixture winner (96x2, Softplus beta5, two
    Fourier bands, 10,467 parameters). Inputs are flat [batch, in_dim] vectors.
    This is an optional reference architecture: callers can supply any critic
    to GANTrainer. Architecture support across the suite uses different critics.
    """

    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=2, fourier=2, beta=5.):
        super().__init__()
        for name, value in (('in_dim', in_dim), ('hidden_dim', hidden_dim), ('n_hidden', n_hidden)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(fourier) is not int or fourier < 0:
            raise ValueError('fourier must be a nonnegative integer')
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError('beta must be finite and positive')
        self.main = _SmoothFourierMLP(in_dim, hidden_dim, n_hidden, fourier, beta)
        self.skip = nn.Linear(in_dim, 1, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        return self.main(x).reshape(-1) + self.skip(x).squeeze(-1)
