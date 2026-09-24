"""Pointwise nonperiodic geometry critics for the frozen shared-c6 recipe.

The feature maps are fixed before training, use no examples or evaluation data,
and have an independent local RNG where randomness is required. Every trainable
parameter belongs to the ordinary discriminator optimizer and cap objective.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def _card(name, *, features='raw', activation='softplus', gated=False,
          quadratic_head=False, ridge_sharpness=2., ridge_count=48):
    return dict(name=name, implementation='shared_geometry_v1', width=96,
                layers=3, fourier=0, normalization='layernorm_all',
                layernorm_eps=1e-5, features=features, activation=activation,
                softplus_beta=4., gated=gated, quadratic_head=quadratic_head,
                ridge_sharpness=ridge_sharpness, ridge_count=ridge_count,
                ridge_seed=0, ridge_offset_std=1.5, ridge_window_halfwidth=.4,
                tanh_coordinate_scales=[.5, 2., 4.],
                asinh_coordinate_scales=[2., 4.],
                quadratic_input_scale=.25, cubic_input_scale=.1,
                quadratic_head_scale=.25)


# Frozen before the rare-mass screen. All cards use the same recipe and host.
ARCHITECTURES = [
    _card('geometry_poly2_lnsp96', features='poly2'),
    _card('geometry_poly3_lnsp96', features='poly3'),
    _card('geometry_tanhcoords_lnsp96', features='tanhcoords'),
    _card('geometry_asinhcoords_lnsp96', features='asinhcoords'),
    _card('geometry_ridgehinge2_lnsp96', features='ridge_hinge', ridge_sharpness=2.),
    _card('geometry_ridgehinge8_lnsp96', features='ridge_hinge', ridge_sharpness=8.),
    _card('geometry_ridgetanh2_lnsp96', features='ridge_tanh', ridge_sharpness=2.),
    _card('geometry_ridgewindow4_lnsp96', features='ridge_window', ridge_sharpness=4.),
    _card('geometry_raw_lngelu96', activation='gelu'),
    _card('geometry_raw_lnmish96', activation='mish'),
    _card('geometry_raw_lntanh96', activation='tanh'),
    _card('geometry_raw_lngate96', gated=True),
    _card('geometry_poly2_lngate96', features='poly2', gated=True),
    _card('geometry_raw_lnquadhead96', quadratic_head=True),
    _card('geometry_poly2_lnquadhead96', features='poly2', quadratic_head=True),
]


class SharedGeometryCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], 0):
            raise ValueError('host discriminator dimensions differ from geometry card')
        if card['implementation'] != 'shared_geometry_v1' or card['normalization'] != 'layernorm_all':
            raise ValueError('unsupported geometry discriminator declaration')
        feature_width = {'raw': 2, 'poly2': 5, 'poly3': 9, 'tanhcoords': 8,
                         'asinhcoords': 6, 'ridge_hinge': 2 + card['ridge_count'],
                         'ridge_tanh': 2 + card['ridge_count'],
                         'ridge_window': 2 + card['ridge_count']}[card['features']]
        if card['features'].startswith('ridge_'):
            stream = torch.Generator().manual_seed(card['ridge_seed'])
            directions = torch.randn(card['ridge_count'], 2, generator=stream)
            directions = F.normalize(directions, dim=1)
            offsets = card['ridge_offset_std'] * torch.randn(card['ridge_count'], generator=stream)
            self.register_buffer('ridge_directions', directions)
            self.register_buffer('ridge_offsets', offsets)
        self.layers = nn.ModuleList()
        self.normalizers = nn.ModuleList()
        self.gates = nn.ModuleList() if card['gated'] else None
        for index in range(card['layers']):
            input_width = feature_width if index == 0 else card['width']
            self.layers.append(nn.Linear(input_width, card['width']))
            self.normalizers.append(nn.LayerNorm(card['width'], eps=card['layernorm_eps']))
            if self.gates is not None:
                self.gates.append(nn.Linear(input_width, card['width']))
        self.head = nn.Linear(card['width'], 1)
        if card['quadratic_head']:
            self.square_head = nn.Linear(card['width'], 1, bias=False)
            nn.init.zeros_(self.square_head.weight)

    def encode(self, x):
        feature = self.card['features']
        if feature == 'raw':
            return x
        if feature in ('poly2', 'poly3'):
            a, b = x.unbind(-1)
            quad = torch.stack((a.square(), a*b, b.square()), dim=-1)
            parts = [x, self.card['quadratic_input_scale'] * quad]
            if feature == 'poly3':
                cubic = torch.stack((a**3, a.square()*b, a*b.square(), b**3), dim=-1)
                parts.append(self.card['cubic_input_scale'] * cubic)
            return torch.cat(parts, dim=-1)
        if feature == 'tanhcoords':
            return torch.cat([x] + [(scale*x).tanh()
                                    for scale in self.card['tanh_coordinate_scales']], dim=-1)
        if feature == 'asinhcoords':
            return torch.cat([x] + [torch.asinh(scale*x)/scale
                                    for scale in self.card['asinh_coordinate_scales']], dim=-1)
        ridge = x @ self.ridge_directions.T + self.ridge_offsets
        sharpness = self.card['ridge_sharpness']
        if feature == 'ridge_hinge':
            basis = F.softplus(sharpness*ridge) / sharpness
        elif feature == 'ridge_tanh':
            basis = (sharpness*ridge).tanh()
        elif feature == 'ridge_window':
            halfwidth = self.card['ridge_window_halfwidth']
            basis = torch.sigmoid(sharpness*(ridge+halfwidth))*torch.sigmoid(sharpness*(halfwidth-ridge))
        else:
            raise ValueError('unknown feature map')
        return torch.cat((x, basis), dim=-1)

    def activate(self, x):
        kind = self.card['activation']
        if kind == 'softplus':
            return F.softplus(x, beta=self.card['softplus_beta'])
        if kind == 'gelu':
            return F.gelu(x)
        if kind == 'mish':
            return F.mish(x)
        if kind == 'tanh':
            return x.tanh()
        raise ValueError('unknown activation')

    def forward(self, x):
        h = self.encode(x)
        for index, (layer, norm) in enumerate(zip(self.layers, self.normalizers)):
            value = self.activate(norm(layer(h)))
            if self.gates is not None:
                value = 2.*value*torch.sigmoid(self.gates[index](h))
            h = value
        score = self.head(h).squeeze(-1)
        if self.card['quadratic_head']:
            score = score + self.card['quadratic_head_scale']*self.square_head(h.square()).squeeze(-1)
        return score


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedGeometryCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
