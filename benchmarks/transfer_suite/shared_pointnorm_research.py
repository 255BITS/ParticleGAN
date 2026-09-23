"""Pointwise smooth discriminator normalization for the shared-c6 recipe.

LayerNorm and RMSNorm normalize features within each example. WeightNorm
parametrizes each layer's weights. None use batch statistics or target data.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def _card(name, *, width, activation, normalization, scope, beta=5., raw_skip=False):
    card = dict(name=name, implementation='shared_pointnorm_v1', width=width,
                layers=3, fourier=0, activation=activation, softplus_beta=beta,
                normalization=normalization, normalization_scope=scope,
                layernorm_eps=1e-5, rmsnorm_eps=1e-5, output_scale=1.)
    if raw_skip:
        card['raw_linear_skip'] = True
    return card


# This bounded follow-up is frozen before either task runs.
ARCHITECTURES = [
    _card('pointnorm_layer_first_softplus96_l3', width=96, activation='softplus', normalization='layernorm', scope='first'),
    _card('pointnorm_layer_all_softplus96_l3', width=96, activation='softplus', normalization='layernorm', scope='all'),
    _card('pointnorm_layer_first_silu128_l3', width=128, activation='silu', normalization='layernorm', scope='first'),
    _card('pointnorm_layer_all_silu128_l3', width=128, activation='silu', normalization='layernorm', scope='all'),
    _card('pointnorm_rms_all_silu128_l3', width=128, activation='silu', normalization='rmsnorm', scope='all'),
    _card('pointnorm_weight_all_silu128_l3', width=128, activation='silu', normalization='weightnorm', scope='all'),
    # Frozen rare-component follow-up after the six-card pointwise norm screen.
    _card('pointnorm_layer_all_softplus64_l3', width=64, activation='softplus', normalization='layernorm', scope='all'),
    _card('pointnorm_layer_all_softplus128_l3', width=128, activation='softplus', normalization='layernorm', scope='all'),
    _card('pointnorm_layer_all_softplus160_l3', width=160, activation='softplus', normalization='layernorm', scope='all'),
    _card('pointnorm_layer_all_softplus96_beta2_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=2.),
    _card('pointnorm_layer_all_softplus96_beta10_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=10.),
    _card('pointnorm_layer_all_softplus96_skip_l3', width=96, activation='softplus', normalization='layernorm', scope='all', raw_skip=True),
    # Frozen interpolation after a related raw-Softplus width PASS.
    _card('pointnorm_layer_all_softplus96_beta3_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=3.),
    _card('pointnorm_layer_all_softplus96_beta4_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=4.),
    _card('pointnorm_layer_all_softplus96_beta6_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=6.),
    _card('pointnorm_layer_all_softplus96_beta8_l3', width=96, activation='softplus', normalization='layernorm', scope='all', beta=8.),
]


class SharedPointnormCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = deepcopy(architecture)
        card = self.card
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], card['fourier']):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_pointnorm_v1':
            raise ValueError('unsupported architecture declaration')
        self.layers = nn.ModuleList()
        self.normalizers = nn.ModuleList()
        for i in range(card['layers']):
            linear = nn.Linear(2 if i == 0 else card['width'], card['width'])
            if card['normalization'] == 'weightnorm':
                linear = weight_norm(linear)
            self.layers.append(linear)
            selected = card['normalization_scope'] == 'all' or i == 0
            if not selected or card['normalization'] == 'weightnorm':
                normalizer = nn.Identity()
            elif card['normalization'] == 'layernorm':
                normalizer = nn.LayerNorm(card['width'], eps=card['layernorm_eps'])
            elif card['normalization'] == 'rmsnorm':
                normalizer = nn.RMSNorm(card['width'], eps=card['rmsnorm_eps'])
            else:
                raise ValueError('unsupported normalization')
            self.normalizers.append(normalizer)
        self.head = nn.Linear(card['width'], 1)
        if card['normalization'] == 'weightnorm':
            self.head = weight_norm(self.head)
        if card.get('raw_linear_skip', False):
            self.skip = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(self.skip.weight)
        if card['activation'] == 'softplus':
            self.activation = nn.Softplus(beta=card['softplus_beta'])
        elif card['activation'] == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError('unsupported activation')

    def forward(self, x):
        raw = x
        for linear, normalizer in zip(self.layers, self.normalizers):
            x = self.activation(normalizer(linear(x)))
        score = self.card['output_scale'] * self.head(x).squeeze(-1)
        if self.card.get('raw_linear_skip', False):
            score = score + self.skip(raw).squeeze(-1)
        return score


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedPointnormCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
