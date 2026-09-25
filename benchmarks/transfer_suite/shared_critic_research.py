"""Declared pointwise critics for the fixed shared-cap6 architecture study.

Features are generic functions of each input, never estimated from target data.
Fixed feature/output/residual scales are architecture choices, recorded in cards.
"""
from copy import deepcopy
import math

import torch
from torch import nn


def _card(name, **kwargs):
    return dict(name=name, implementation='shared_critic_v1', hidden=96, layers=3,
                activation='softplus', beta=5., features='raw', fourier=0,
                frequency_scale=1., quadratic_scale=.25, residual=False,
                residual_scale=2**-.5, output_scale=1., raw_linear_skip=False,
                branch_hidden=32, branch_scale=.25, normalization='none') | kwargs


ARCHITECTURES = [
    _card('raw_softplus96_l3'),
    _card('raw_silu128_l3', hidden=128, activation='silu'),
    _card('quadratic_softplus96_l2', layers=2, features='quadratic'),
    _card('quadratic_tanh96_l3', features='quadratic', activation='tanh'),
    _card('residual_raw_softplus96_l3', residual=True),
    _card('residual_lowfreq_softplus96_l3', features='axis', fourier=1,
          frequency_scale=.25, residual=True),
    _card('halfscore_fourier_skip96_l2', layers=2, features='axis', fourier=2,
          output_scale=.5, raw_linear_skip=True),
    _card('additive_raw_fourier64_l2', hidden=64, layers=2, features='additive',
          fourier=2, raw_linear_skip=True),
]


def activation(card):
    if card['activation'] == 'softplus':
        return nn.Softplus(beta=card['beta'])
    if card['activation'] == 'silu':
        return nn.SiLU()
    if card['activation'] == 'tanh':
        return nn.Tanh()
    raise ValueError('unknown activation')


class _Stack(nn.Module):
    def __init__(self, input_dim, width, card):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, width)] +
                                    [nn.Linear(width, width) for _ in range(card['layers']-1)])
        self.activation = activation(card)
        self.head = nn.Linear(width, 1)
        self.residual = card['residual']
        self.residual_scale = card['residual_scale']

    def forward(self, x):
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:]:
            value = self.activation(layer(x))
            x = self.residual_scale*(x+value) if self.residual else value
        return self.head(x).squeeze(-1)


class SharedResearchCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = deepcopy(architecture)
        card = self.card
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['hidden'], card['layers'], card['fourier']):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_critic_v1' or card['normalization'] != 'none':
            raise ValueError('unsupported architecture declaration')
        self.register_buffer('frequencies', math.pi*card['frequency_scale']*2.**torch.arange(fourier))
        kind = card['features']
        input_dim = {'raw': 2, 'quadratic': 5, 'axis': 2+4*fourier, 'additive': 2}[kind]
        self.main = _Stack(input_dim, hidden_dim, card)
        if kind == 'additive':
            self.branch = _Stack(4*fourier, card['branch_hidden'], card)
        if card['raw_linear_skip']:
            self.skip = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(self.skip.weight)

    def periodic(self, x):
        phase = (x.unsqueeze(-1)*self.frequencies).flatten(1)
        return torch.cat((phase.sin(), phase.cos()), 1)

    def encode(self, x):
        if self.card['features'] == 'axis':
            return torch.cat((x, self.periodic(x)), 1)
        if self.card['features'] == 'quadratic':
            products = torch.stack((x[:, 0]**2, x[:, 0]*x[:, 1], x[:, 1]**2), 1)
            return torch.cat((x, self.card['quadratic_scale']*products), 1)
        return x

    def forward(self, x):
        score = self.card['output_scale']*self.main(self.encode(x))
        if self.card['features'] == 'additive':
            score = score+self.card['branch_scale']*self.branch(self.periodic(x))
        if self.card['raw_linear_skip']:
            score = score+self.skip(x).squeeze(-1)
        return score


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['hidden'], d_layers=card['layers'],
                fourier=card['fourier'], research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedResearchCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
