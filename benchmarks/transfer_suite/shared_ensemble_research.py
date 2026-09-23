"""Independent smooth discriminator branches for the fixed shared-c6 study.

Every transform is pointwise and fixed before training. Coordinate, Fourier,
branch and score scales are declared architecture parameters, not optimizer
changes or estimates from the target distribution.
"""
from copy import deepcopy
import math

import torch
from torch import nn


def _card(name, *, scales, width=64, layers=2, activations='softplus',
          score_scale=None, frequencies=(), raw_skip=False):
    count = len(scales)
    if isinstance(activations, str):
        activations = [activations] * count
    return dict(name=name, implementation='shared_ensemble_v1',
                input_scales=list(scales), width=width, layers=layers,
                activations=list(activations), softplus_beta=5.,
                frequency_multipliers=list(frequencies),
                score_scale=count**-.5 if score_scale is None else score_scale,
                raw_linear_skip=raw_skip, normalization='none')


# Entire initial screen is declared here before evaluating either target case.
ARCHITECTURES = [
    _card('ensemble2_equal_softplus64_l2', scales=[1., 1.]),
    _card('ensemble2_multiscale_softplus64_l2', scales=[.5, 2.]),
    _card('ensemble3_multiscale_softplus64_l2', scales=[.5, 1., 2.]),
    _card('ensemble3_multiscale_silu64_l2', scales=[.5, 1., 2.], activations='silu'),
    _card('ensemble2_multiscale_silu64_l3', scales=[.5, 2.], layers=3, activations='silu'),
    _card('ensemble3_broad_softplus48_l3', scales=[.75, 1.5, 3.], width=48, layers=3),
    _card('ensemble4_broad_silu48_l2', scales=[.5, 1., 2., 4.], width=48, activations='silu'),
    _card('ensemble2_highscale_softplus96_l2', scales=[1., 3.], width=96),
    _card('ensemble2_mixed64_l2', scales=[1., 2.], activations=['softplus', 'silu']),
    _card('ensemble2_multiscale_halfscore64_l2', scales=[.5, 2.], score_scale=.5),
    _card('ensemble2_multiscale_fullscore64_l2', scales=[.5, 2.], score_scale=1.),
    _card('ensemble2_multiscale_spectrum64_l2', scales=[.5, 2.], frequencies=[.5, 1.5]),
]


class _Branch(nn.Module):
    def __init__(self, input_dim, card, activation):
        super().__init__()
        width = card['width']
        self.layers = nn.ModuleList([nn.Linear(input_dim, width)] +
                                    [nn.Linear(width, width) for _ in range(card['layers']-1)])
        self.head = nn.Linear(width, 1)
        if activation == 'softplus':
            self.activation = nn.Softplus(beta=card['softplus_beta'])
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError('unknown branch activation')

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.head(x).squeeze(-1)


class SharedEnsembleCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden=2, fourier=0, *, architecture):
        super().__init__()
        self.card = deepcopy(architecture)
        card = self.card
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], len(card['frequency_multipliers'])):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_ensemble_v1' or card['normalization'] != 'none':
            raise ValueError('unsupported architecture declaration')
        if not card['input_scales'] or len(card['input_scales']) != len(card['activations']):
            raise ValueError('one fixed scale and activation required per branch')
        if not all(math.isfinite(scale) and scale > 0 for scale in card['input_scales']):
            raise ValueError('invalid fixed input scale')
        if not math.isfinite(card['score_scale']) or card['score_scale'] <= 0:
            raise ValueError('invalid fixed score scale')
        frequencies = torch.tensor(card['frequency_multipliers'], dtype=torch.float32) * math.pi
        self.register_buffer('frequencies', frequencies)
        input_dim = 2 + 4 * len(frequencies)
        self.branches = nn.ModuleList(_Branch(input_dim, card, activation)
                                      for activation in card['activations'])
        if card['raw_linear_skip']:
            self.skip = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(self.skip.weight)

    def encode(self, x):
        if not len(self.frequencies):
            return x
        phase = (x.unsqueeze(-1) * self.frequencies).flatten(1)
        return torch.cat((x, phase.sin(), phase.cos()), 1)

    def forward(self, x):
        score = sum(branch(self.encode(scale*x)) for branch, scale in
                    zip(self.branches, self.card['input_scales'])) * self.card['score_scale']
        if self.card['raw_linear_skip']:
            score = score + self.skip(x).squeeze(-1)
        return score


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=len(card['frequency_multipliers']),
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedEnsembleCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
