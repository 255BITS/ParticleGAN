"""Pointwise normalization structure trials for the unchanged shared_c6 recipe.

All operations act on one discriminator example. The cards contain no data or
task statistics, labels, batch state, or evaluation feedback.
"""
from copy import deepcopy

import torch
from torch import nn


def _card(name, *, norm='layer', placement='pre', affine=True, scope='all',
          raw_blend=0., blend_scope='all', residual=0., input_injection=0.,
          mix_rms=0., power=1., groups=1, width=96, beta=4.):
    return dict(name='normstruct_'+name, implementation='shared_norm_structure_v1',
                width=width, layers=3, fourier=0, activation='softplus',
                softplus_beta=beta, normalization=norm, placement=placement,
                elementwise_affine=affine, normalization_scope=scope,
                raw_blend=raw_blend, blend_scope=blend_scope, residual=residual,
                input_injection=input_injection, mix_rms=mix_rms,
                variance_power=power, groups=groups, normalization_eps=1e-5)


# Frozen together before the first training episode. Each card is a distinct D.
ARCHITECTURES = [
    _card('ln_pre_fixed', affine=False),
    _card('ln_post_affine', placement='post'),
    _card('ln_post_fixed', placement='post', affine=False),
    _card('ln_first_last', scope='first_last'),
    _card('ln_last', scope='last'),
    _card('rms_pre_affine', norm='rms'),
    _card('rms_pre_fixed', norm='rms', affine=False),
    _card('rms_post_affine', norm='rms', placement='post'),
    _card('center_pre_affine', norm='center'),
    _card('center_pre_fixed', norm='center', affine=False),
    _card('ln_raw_blend025', raw_blend=.25),
    _card('ln_raw_blend050', raw_blend=.5),
    _card('ln_post_raw_blend025', placement='post', raw_blend=.25),
    _card('ln_residual025', residual=.25),
    _card('ln_input_injection025', input_injection=.25),
    _card('ln_rms_mix050', mix_rms=.5),
    # Frozen focused refinement after all 16 screen cards completed.
    _card('power025_fixed', norm='power', affine=False, power=.25),
    _card('power050_fixed', norm='power', affine=False, power=.5),
    _card('power075_fixed', norm='power', affine=False, power=.75),
    _card('power025_affine', norm='power', power=.25),
    _card('power050_affine', norm='power', power=.5),
    _card('group2_affine', norm='group', groups=2),
    _card('group4_affine', norm='group', groups=4),
    _card('group2_fixed', norm='group', groups=2, affine=False),
    _card('ln_blend050_first', raw_blend=.5, blend_scope='first'),
    _card('ln_blend050_last', raw_blend=.5, blend_scope='last'),
    _card('ln_blend050_first_last', raw_blend=.5, blend_scope='first_last'),
    _card('center_fixed_blend025', norm='center', affine=False, raw_blend=.25),
    # Frozen six-card follow-up to the stable center-only screen result.
    _card('center_fixed96_beta3', norm='center', affine=False, beta=3.),
    _card('center_fixed96_beta5', norm='center', affine=False, beta=5.),
    _card('center_fixed96_beta6', norm='center', affine=False, beta=6.),
    _card('center_fixed96_beta8', norm='center', affine=False, beta=8.),
    _card('center_fixed128_beta4', norm='center', affine=False, width=128),
    _card('center_fixed128_beta8', norm='center', affine=False, width=128, beta=8.),
]


class CenterNorm(nn.Module):
    """Remove per-example feature mean without rescaling feature variance."""
    def __init__(self, width, affine):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width)) if affine else None
        self.offset = nn.Parameter(torch.zeros(width)) if affine else None

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return x if self.scale is None else x * self.scale + self.offset


class PowerNorm(nn.Module):
    """Center each example, then scale by feature standard deviation to a fixed power."""
    def __init__(self, width, affine, power, eps):
        super().__init__()
        self.power, self.eps = power, eps
        self.scale = nn.Parameter(torch.ones(width)) if affine else None
        self.offset = nn.Parameter(torch.zeros(width)) if affine else None

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x * (x.square().mean(dim=-1, keepdim=True) + self.eps).pow(-self.power/2)
        return x if self.scale is None else x * self.scale + self.offset


def normalizer(card):
    norm, width, affine = card['normalization'], card['width'], card['elementwise_affine']
    if norm == 'layer':
        return nn.LayerNorm(width, eps=card['normalization_eps'], elementwise_affine=affine)
    if norm == 'rms':
        return nn.RMSNorm(width, eps=card['normalization_eps'], elementwise_affine=affine)
    if norm == 'center':
        return CenterNorm(width, affine)
    if norm == 'power':
        return PowerNorm(width, affine, card['variance_power'], card['normalization_eps'])
    if norm == 'group':
        return nn.GroupNorm(card['groups'], width, eps=card['normalization_eps'], affine=affine)
    raise ValueError('unsupported normalization')


class SharedNormStructureCritic(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], card['fourier']):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_norm_structure_v1':
            raise ValueError('unsupported architecture declaration')
        self.layers = nn.ModuleList(nn.Linear(2 if i == 0 else card['width'], card['width'])
                                    for i in range(card['layers']))
        selected = lambda i: card['normalization_scope'] == 'all' or (
            card['normalization_scope'] == 'first_last' and i in (0, card['layers']-1)) or (
            card['normalization_scope'] == 'last' and i == card['layers']-1)
        self.normalizers = nn.ModuleList(normalizer(card) if selected(i) else nn.Identity()
                                         for i in range(card['layers']))
        self.rms_normalizers = (nn.ModuleList(nn.RMSNorm(card['width'], eps=card['normalization_eps'],
                                                       elementwise_affine=card['elementwise_affine'])
                                             for _ in range(card['layers'])) if card['mix_rms'] else None)
        self.injectors = (nn.ModuleList(nn.Linear(2, card['width'], bias=False)
                                        for _ in range(card['layers']-1)) if card['input_injection'] else None)
        self.activation = nn.Softplus(beta=card['softplus_beta'])
        self.head = nn.Linear(card['width'], 1)

    def forward(self, x):
        raw_input = x
        card = self.card
        for i, (linear, normalizer) in enumerate(zip(self.layers, self.normalizers)):
            prior = x
            pre = linear(x)
            if card['placement'] == 'pre':
                normalized = normalizer(pre)
                if self.rms_normalizers is not None:
                    normalized = ((1-card['mix_rms'])*normalized
                                  + card['mix_rms']*self.rms_normalizers[i](pre))
                x = self.activation(normalized)
            elif card['placement'] == 'post':
                x = normalizer(self.activation(pre))
            else:
                raise ValueError('unsupported placement')
            blend = card['blend_scope'] == 'all' or (card['blend_scope'] == 'first' and i == 0) or (
                card['blend_scope'] == 'last' and i == card['layers']-1) or (
                card['blend_scope'] == 'first_last' and i in (0, card['layers']-1))
            if card['raw_blend'] and blend:
                x = x + card['raw_blend']*self.activation(pre)
            if card['residual'] and i > 0:
                x = x + card['residual']*prior
            if self.injectors is not None and i > 0:
                x = x + card['input_injection']*self.injectors[i-1](raw_input)
        return self.head(x).squeeze(-1)


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return SharedNormStructureCritic(in_dim, hidden_dim, n_hidden, fourier,
                                         architecture=card)
    return create
