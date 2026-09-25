"""Fixed pointwise critic-gradient regularity cards for shared-c6.

Only discriminator internals change. Spectral parametrizations constrain layer
operator norm; bounded score heads constrain the score and its input gradient.
No data statistics, labels, metrics or target parameters enter the critic.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from benchmarks.toy100.device import rng_fork_devices


def _card(name, *, norm='none', spectral='none', score_bound=None):
    return dict(name=name, implementation='shared_rare_gradient_v1', width=96,
                layers=3, fourier=0, activation='softplus', softplus_beta=4.,
                pointwise_normalization=norm, spectral_scope=spectral,
                spectral_power_iterations=1, score_bound=score_bound)


# Frozen 12-card screen before any candidate episode.
ARCHITECTURES = [
    _card('rare_sn_raw_first', spectral='first'),
    _card('rare_sn_raw_hidden', spectral='hidden'),
    _card('rare_sn_raw_head', spectral='head'),
    _card('rare_sn_raw_all', spectral='all'),
    _card('rare_sn_layer_first', norm='layer', spectral='first'),
    _card('rare_sn_layer_hidden', norm='layer', spectral='hidden'),
    _card('rare_sn_layer_head', norm='layer', spectral='head'),
    _card('rare_sn_layer_all', norm='layer', spectral='all'),
    _card('rare_bound_layer_1', norm='layer', score_bound=1.),
    _card('rare_bound_layer_2', norm='layer', score_bound=2.),
    _card('rare_bound_layer_4', norm='layer', score_bound=4.),
    _card('rare_bound_raw_2', score_bound=2.),
]


class SharedRareGradientCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], card['fourier']):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_rare_gradient_v1' or card['activation'] != 'softplus':
            raise ValueError('unsupported architecture card')
        if card['pointwise_normalization'] not in ('none', 'layer'):
            raise ValueError('unsupported pointwise normalization')
        if card['spectral_scope'] not in ('none', 'first', 'hidden', 'head', 'all'):
            raise ValueError('unsupported spectral scope')
        self.layers = nn.ModuleList(nn.Linear(2 if i == 0 else hidden_dim, hidden_dim)
                                    for i in range(n_hidden))
        self.normalizers = nn.ModuleList(
            nn.Identity() if card['pointwise_normalization'] == 'none' else nn.LayerNorm(hidden_dim, eps=1e-5)
            for _ in range(n_hidden))
        self.head = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus(beta=card['softplus_beta'])
        # Draw power-iteration vectors without shifting the host's random stream.
        with torch.random.fork_rng(devices=rng_fork_devices()):
            for i in range(n_hidden):
                if card['spectral_scope'] == 'all' or (i == 0 and card['spectral_scope'] == 'first') or (
                        i > 0 and card['spectral_scope'] == 'hidden'):
                    self.layers[i] = spectral_norm(self.layers[i], n_power_iterations=card['spectral_power_iterations'])
            if card['spectral_scope'] in ('head', 'all'):
                self.head = spectral_norm(self.head, n_power_iterations=card['spectral_power_iterations'])

    def forward(self, x):
        for linear, normalizer in zip(self.layers, self.normalizers):
            x = self.activation(normalizer(linear(x)))
        score = self.head(x).squeeze(-1)
        bound = self.card['score_bound']
        return score if bound is None else bound * torch.tanh(score / bound)


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedRareGradientCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
