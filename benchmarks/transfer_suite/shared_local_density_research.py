"""Generic pointwise local-curvature critics under the unchanged shared recipe.

No target data, labels, centers or moments enter construction. Feature banks use
one fixed local seed0 initialization; no seed is a search parameter.
"""
from copy import deepcopy
import torch
from torch import nn


def _card(name, family, **options):
    return dict(name=name, implementation='shared_local_density_v1', family=family,
                hidden=64, layers=2, fourier=0, normalization='none', count=128,
                center_scale=2., feature_seed=0, widths=[.25, .5, 1., 2.],
                train_centers=False, train_widths=False, kernel='gaussian',
                head='linear', activation='silu', beta=5.) | options


ARCHITECTURES = [
    _card('local_rbf64_direct', 'radial', count=64, hidden=64, layers=1),
    _card('local_rbf128_direct', 'radial', hidden=128, layers=1),
    _card('local_rbf256_direct', 'radial', count=256, hidden=256, layers=1),
    _card('local_rbf128_adaptive', 'radial', hidden=128, layers=1,
          train_centers=True, train_widths=True),
    _card('local_cauchy128_direct', 'radial', hidden=128, layers=1, kernel='cauchy'),
    _card('local_rbf128_softplus64', 'radial', head='mlp', activation='softplus'),
    _card('local_quad16_width1', 'quadratic_experts', count=16, hidden=16, layers=1, widths=[1.]),
    _card('local_quad32_width1', 'quadratic_experts', count=32, hidden=32, layers=1, widths=[1.]),
    _card('local_quad32_width05', 'quadratic_experts', count=32, hidden=32, layers=1, widths=[.5]),
    _card('local_quad32_adaptive', 'quadratic_experts', count=32, hidden=32, layers=1,
          widths=[1.], train_centers=True, train_widths=True),
    _card('local_product_silu64_l2', 'product'),
    _card('local_product_silu96_l2', 'product', hidden=96),
    _card('local_product_silu128_l2', 'product', hidden=128),
    _card('local_squared_silu64_l3', 'square', layers=3),
    _card('local_squared_silu96_l3', 'square', hidden=96, layers=3),
    _card('local_product_softplus96_l2', 'product', hidden=96, activation='softplus'),
]


def activation(card):
    return nn.SiLU() if card['activation'] == 'silu' else nn.Softplus(beta=card['beta'])


class LocalDensityCritic(nn.Module):
    """Signed radial basis, gated local quadratics, or smooth product features.

Radial heads use ordinary linear initialization. Quadratic expert coefficients
start at zero, alongside an ordinary raw linear head. Positive trainable widths
are exponentials of log-width, with no clipping, penalty or custom gradient.
All learned parameters receive the exact same discriminator Adam settings.
"""
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden=2, fourier=0, *, architecture):
        super().__init__()
        self.card = card = deepcopy(architecture)
        if (in_dim, hidden_dim, n_hidden, fourier) != (2, card['hidden'], card['layers'], 0):
            raise ValueError('host dimensions differ from declared local-density architecture')
        if card['implementation'] != 'shared_local_density_v1' or card['normalization'] != 'none':
            raise ValueError('unsupported local-density card')
        family = card['family']
        if family in ('radial', 'quadratic_experts'):
            centers = card['center_scale']*torch.randn(card['count'], 2,
                       generator=torch.Generator().manual_seed(card['feature_seed']))
            widths = torch.tensor([card['widths'][i % len(card['widths'])] for i in range(card['count'])])
            for name, value, learn in [('centers', centers, card['train_centers']),
                                       ('log_width', widths.log(), card['train_widths'])]:
                if learn:
                    self.register_parameter(name, nn.Parameter(value))
                else:
                    self.register_buffer(name, value)
        if family == 'radial':
            if card['head'] == 'linear':
                self.head = nn.Linear(card['count']+2, 1)
            else:
                self.head = nn.Sequential(nn.Linear(card['count']+2, hidden_dim), activation(card),
                                          nn.Linear(hidden_dim, hidden_dim), activation(card), nn.Linear(hidden_dim, 1))
        elif family == 'quadratic_experts':
            self.coefficients = nn.Parameter(torch.zeros(card['count'], 6))
            self.raw = nn.Linear(2, 1)
        elif family in ('product', 'square'):
            dims = [2] + [hidden_dim]*n_hidden
            self.left = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
            if family == 'product':
                self.right = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
            self.activation = activation(card)
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError('unsupported local-density family')

    def forward(self, x):
        family = self.card['family']
        if family in ('radial', 'quadratic_experts'):
            delta = (x[:, None, :]-self.centers[None, :, :])/self.log_width.exp()[None, :, None]
            distance = delta.square().sum(-1)
            if family == 'radial':
                values = (-.5*distance).exp() if self.card['kernel'] == 'gaussian' else (1+distance).reciprocal()
                return self.head(torch.cat((x, values), dim=1)).squeeze(-1)
            weights = (-.5*distance).softmax(dim=1)
            dx, dy = delta.unbind(-1)
            features = torch.stack((torch.ones_like(dx), dx, dy, dx.square(), dx*dy, dy.square()), -1)
            local = (features*self.coefficients[None]).sum(-1)
            return (weights*local).sum(1)+self.raw(x).squeeze(-1)
        for i, left in enumerate(self.left):
            value = self.activation(left(x))
            x = value.square() if family == 'square' else value*self.activation(self.right[i](x))
        return self.head(x).squeeze(-1)


def variant(card):
    return dict(name=card['name'], overrides=dict(d_hidden=card['hidden'], d_layers=card['layers'],
                fourier=0, research_discriminator=deepcopy(card)))


def constructor(card):
    card = deepcopy(card)
    def create(in_dim=2, hidden_dim=64, n_hidden=2, fourier=0):
        return LocalDensityCritic(in_dim, hidden_dim, n_hidden, fourier, architecture=card)
    return create
