"""Smooth scalar score readouts over the unchanged LN Softplus beta4 trunk.

These pointwise D architectures share the original G, data and shared-c6 recipe.
No target features, update modifiers, feedback, or extra losses are used.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def _card(name, kind, scale):
    return dict(name=name, implementation='shared_geometry_readout_v1',
                width=96, layers=3, fourier=0, normalization='layernorm_all',
                layernorm_eps=1e-5, softplus_beta=4., readout=kind,
                readout_scale=scale)


# Frozen final focused screen after coordinate and residual-pathway failures.
# Tanh score saturation at scales 1/2/4 was already tested in the independent
# gradient study, so only slower-decaying readout maps are retained here.
ARCHITECTURES = [
    _card('geometry_readout_asinh2', 'asinh', 2.),
    _card('geometry_readout_rational2', 'rational', 2.),
]


class SharedGeometryReadoutCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], 0):
            raise ValueError('host discriminator dimensions differ from readout card')
        if card['implementation'] != 'shared_geometry_readout_v1' or card['normalization'] != 'layernorm_all':
            raise ValueError('unsupported readout declaration')
        self.layers = nn.ModuleList()
        self.normalizers = nn.ModuleList()
        for index in range(card['layers']):
            self.layers.append(nn.Linear(2 if index == 0 else card['width'], card['width']))
            self.normalizers.append(nn.LayerNorm(card['width'], eps=card['layernorm_eps']))
        self.head = nn.Linear(card['width'], 1)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.normalizers):
            x = F.softplus(norm(layer(x)), beta=self.card['softplus_beta'])
        score = self.head(x).squeeze(-1)
        scale = self.card['readout_scale']
        kind = self.card['readout']
        if kind == 'asinh':
            return scale*torch.asinh(score/scale)
        if kind == 'rational':
            return score/torch.sqrt(1.+(score/scale).square())
        raise ValueError('unknown score readout')


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedGeometryReadoutCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
