"""Initially-zero nonperiodic feature pathways on the LN Softplus beta4 critic.

The original critic's weights, output and global RNG state after construction
are preserved. Extra feature parameters use the same D optimizer and objective.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.toy100.device import rng_fork_devices


def _card(name, *, features, site, activation_basis='none'):
    return dict(name=name, implementation='shared_geometry_residual_v1',
                width=96, layers=3, fourier=0, normalization='layernorm_all',
                layernorm_eps=1e-5, softplus_beta=4., features=features, site=site,
                activation_basis=activation_basis, feature_scale=.25,
                quadratic_input_scale=.25, ridge_count=48, ridge_sharpness=2.,
                ridge_seed=0, ridge_offset_std=1.5)


# Bounded follow-up to the first full rare-mass geometry screen, frozen before fitting.
ARCHITECTURES = [
    _card('geometry_resid_poly2_first', features='poly2', site='first'),
    _card('geometry_resid_hinge_first', features='ridge_hinge', site='first'),
    _card('geometry_resid_hinge_all', features='ridge_hinge', site='all'),
    _card('geometry_resid_raw_later', features='raw', site='later'),
    _card('geometry_resid_poly2_later', features='poly2', site='later'),
    _card('geometry_resid_hinge_later', features='ridge_hinge', site='later'),
    _card('geometry_resid_poly2_head', features='poly2', site='head'),
    _card('geometry_resid_hinge_head', features='ridge_hinge', site='head'),
    _card('geometry_resid_tanh_activation', features='raw', site='activation',
          activation_basis='tanh2'),
]


class SharedGeometryResidualCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], 0):
            raise ValueError('host discriminator dimensions differ from geometry card')
        if card['implementation'] != 'shared_geometry_residual_v1' or card['normalization'] != 'layernorm_all':
            raise ValueError('unsupported geometry residual declaration')
        # Exact construction order of the established LN-all beta4 lead.
        self.layers = nn.ModuleList()
        self.normalizers = nn.ModuleList()
        for index in range(card['layers']):
            self.layers.append(nn.Linear(2 if index == 0 else card['width'], card['width']))
            self.normalizers.append(nn.LayerNorm(card['width'], eps=card['layernorm_eps']))
        self.head = nn.Linear(card['width'], 1)
        width = {'raw': 2, 'poly2': 5, 'ridge_hinge': 2+card['ridge_count']}[card['features']]
        if card['features'] == 'ridge_hinge':
            stream = torch.Generator().manual_seed(card['ridge_seed'])
            directions = torch.randn(card['ridge_count'], 2, generator=stream)
            self.register_buffer('ridge_directions', F.normalize(directions, dim=1))
            self.register_buffer('ridge_offsets', card['ridge_offset_std'] *
                                 torch.randn(card['ridge_count'], generator=stream))
        self.injections = nn.ModuleDict()
        sites = {'first': [0], 'later': [1, 2], 'all': [0, 1, 2],
                 'head': [], 'activation': []}[card['site']]
        # nn.Linear's temporary default initialization consumes global random
        # numbers even when immediately zeroed; isolate that consumption.
        with torch.random.fork_rng(devices=rng_fork_devices()):
            for index in sites:
                self.injections[str(index)] = nn.Linear(width, card['width'], bias=False)
                nn.init.zeros_(self.injections[str(index)].weight)
            if card['site'] == 'head':
                self.feature_head = nn.Linear(width, 1, bias=False)
                nn.init.zeros_(self.feature_head.weight)
        if card['site'] == 'activation':
            if card['activation_basis'] != 'tanh2':
                raise ValueError('unknown activation correction')
            self.activation_coefficients = nn.ParameterList(
                nn.Parameter(torch.zeros(card['width'])) for _ in range(card['layers']))

    def encode(self, x):
        if self.card['features'] == 'raw':
            return x
        if self.card['features'] == 'poly2':
            a, b = x.unbind(-1)
            return torch.cat((x, self.card['quadratic_input_scale']*
                              torch.stack((a.square(), a*b, b.square()), dim=-1)), dim=-1)
        ridge = x@self.ridge_directions.T+self.ridge_offsets
        sharpness = self.card['ridge_sharpness']
        return torch.cat((x, F.softplus(sharpness*ridge)/sharpness), dim=-1)

    def forward(self, x):
        features = self.encode(x)
        h = x
        for index, (layer, norm) in enumerate(zip(self.layers, self.normalizers)):
            value = layer(h)
            key = str(index)
            if key in self.injections:
                value = value + self.card['feature_scale']*self.injections[key](features)
            normalized = norm(value)
            h = F.softplus(normalized, beta=self.card['softplus_beta'])
            if self.card['site'] == 'activation':
                h = h + self.activation_coefficients[index]*torch.tanh(2.*normalized)
        score = self.head(h).squeeze(-1)
        if self.card['site'] == 'head':
            score = score + self.card['feature_scale']*self.feature_head(features).squeeze(-1)
        return score


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedGeometryResidualCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
