"""Generic permutation-equivariant minibatch discriminator feature cards.

Real and fake calls each derive features solely from their current input batch.
All pairwise kernels are smooth in their inputs. No branch detaches a feature.
"""
from copy import deepcopy

import torch
from torch import nn


SCALES = [.1, .25, .5, 1.]


def _card(name, *, trunk='center', beta=6., feature='std_scalar', placement='head'):
    return dict(name='batchfeat_'+name, implementation='shared_batch_feature_v1',
                width=96, layers=3, fourier=0, trunk_normalization=trunk,
                softplus_beta=beta, feature=feature, placement=placement,
                kernel_scales=SCALES, eps=1e-5, batch_dependence=True,
                batch_statistics='current discriminator input only')


# Frozen eight-card rare-only screen, before any training episode.
ARCHITECTURES = [
    _card('center6_std_scalar', feature='std_scalar'),
    _card('center6_std_vector', feature='std_vector'),
    _card('layer4_std_scalar', trunk='layer', beta=4., feature='std_scalar'),
    _card('center6_density_head', feature='density'),
    _card('center6_distance_head', feature='distance'),
    _card('center6_density_input', feature='density', placement='input'),
    _card('layer4_density_head', trunk='layer', beta=4., feature='density'),
    _card('center6_density_std_head', feature='density_std'),
]


class CenterNorm(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=-1, keepdim=True)


class BatchFeatureCritic(nn.Module):
    """Pointwise smooth trunk with differentiable current-batch features."""
    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['width'], card['layers'], card['fourier']):
            raise ValueError('host discriminator dimensions do not match declared architecture')
        if card['implementation'] != 'shared_batch_feature_v1' or not card['batch_dependence']:
            raise ValueError('unsupported batch-feature declaration')
        feature, placement = card['feature'], card['placement']
        if feature not in ('std_scalar', 'std_vector', 'density', 'distance', 'density_std'):
            raise ValueError('unsupported batch feature')
        if placement not in ('head', 'input') or (placement == 'input' and feature != 'density'):
            raise ValueError('unsupported feature placement')
        self.register_buffer('scales', torch.tensor(card['kernel_scales']))
        head_extra = 0 if placement == 'input' else (
            card['width'] if feature == 'std_vector' else 5 if feature == 'density_std' else
            4 if feature in ('density', 'distance') else 1)
        first_dim = 2 + (4 if placement == 'input' else 0)
        self.layers = nn.ModuleList(nn.Linear(first_dim if i == 0 else card['width'], card['width'])
                                    for i in range(card['layers']))
        self.normalizers = nn.ModuleList(
            CenterNorm() if card['trunk_normalization'] == 'center' else
            nn.LayerNorm(card['width'], eps=card['eps']) for _ in range(card['layers']))
        self.activation = nn.Softplus(beta=card['softplus_beta'])
        self.head = nn.Linear(card['width']+head_extra, 1)

    def pairwise_features(self, x):
        # Squared distances have smooth first and second derivatives, including
        # at coincident samples; self-pairs are removed with a fixed mask.
        delta = x[:, None, :] - x[None, :, :]
        d2 = delta.square().sum(-1)
        scales2 = self.scales.square()
        kernels = torch.exp(-d2[..., None] / (2*scales2))
        n = len(x)
        offdiag = (1-torch.eye(n, device=x.device, dtype=x.dtype))[..., None]
        kernels = kernels*offdiag
        if self.card['feature'] in ('density', 'density_std'):
            return kernels.sum(dim=1)/max(n-1, 1)
        # Soft local mean-square neighbor distance, scaled per kernel width.
        weighted = (kernels*d2[..., None]).sum(dim=1)
        return weighted/(kernels.sum(dim=1)+self.card['eps'])/scales2

    def std_features(self, h):
        variance = (h-h.mean(dim=0, keepdim=True)).square().mean(dim=0)
        std = (variance+self.card['eps']).sqrt()
        if self.card['feature'] == 'std_vector':
            return std.expand(len(h), -1)
        return std.mean().expand(len(h), 1)

    def forward(self, x):
        card = self.card
        batch_feature = (self.pairwise_features(x) if card['feature'] in
                         ('density', 'distance', 'density_std') else None)
        if card['placement'] == 'input':
            x = torch.cat((x, batch_feature), dim=-1)
        for linear, normalizer in zip(self.layers, self.normalizers):
            x = self.activation(normalizer(linear(x)))
        if card['placement'] == 'head':
            if card['feature'] == 'density_std':
                feature = torch.cat((batch_feature, self.std_features(x)), dim=-1)
            elif batch_feature is not None:
                feature = batch_feature
            else:
                feature = self.std_features(x)
            x = torch.cat((x, feature), dim=-1)
        return self.head(x).squeeze(-1)


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['width'],
                d_layers=card['layers'], fourier=card['fourier'],
                research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=2):
        return BatchFeatureCritic(in_dim, hidden_dim, n_hidden, fourier,
                                  architecture=card)
    return create
