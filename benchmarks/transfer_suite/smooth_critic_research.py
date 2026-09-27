"""Research-only smooth critic architectures; no data-derived features or losses."""
from copy import deepcopy
import math
import torch
from torch import nn

ARCHITECTURES = [
    dict(name='axis_silu', features='axis', activation='silu'),
    dict(name='axis_softplus1', features='axis', activation='softplus', beta=1.),
    dict(name='axis_softplus5', features='axis', activation='softplus', beta=5.),
    dict(name='axis_tanh', features='axis', activation='tanh'),
    # LeakyReLU(.2) on the first hidden layer only (piecewise-linear kinks near the input), SiLU after.
    dict(name='axis_leaky1_silu', features='axis', activation='silu', first_activation='leaky_relu'),
    # Adds a bias-free raw-coordinate linear score, initialized to zero, beside the MLP output.
    dict(name='axis_silu_raw_skip', features='axis', activation='silu', raw_skip=True),
    dict(name='oriented4_silu', features='oriented', activation='silu', radial_bands=[1.,1.,2.,2.], orientation_seed=0),
    dict(name='oriented8_silu', features='oriented', activation='silu', radial_bands=[.5,.5,1.,1.,2.,2.,4.,4.], orientation_seed=0),
    dict(name='oriented8_softplus1', features='oriented', activation='softplus', beta=1., radial_bands=[.5,.5,1.,1.,2.,2.,4.,4.], orientation_seed=0),
    dict(name='oriented8_tanh', features='oriented', activation='tanh', radial_bands=[.5,.5,1.,1.,2.,2.,4.,4.], orientation_seed=0),
]


class SmoothFourierCritic(nn.Module):
    """Same MLP widths/depth; replace activation and optionally Fourier projection.

    Axis features exactly match the existing toy discriminator. Oriented features
    use fixed origin-centered planes and predeclared radial frequencies in radians.
    Their local RNG does not consume training/initialization RNG state. All input
    coordinates are retained; no target examples or component centers are accepted.
    """
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden=2, fourier=2, *, architecture):
        super().__init__()
        self.architecture = deepcopy(architecture)
        self.features = architecture['features']
        if self.features == 'axis':
            self.register_buffer('freqs', torch.pi * (2.**torch.arange(fourier,dtype=torch.float32)))
            dim = in_dim + 2*fourier*in_dim
        elif self.features == 'oriented':
            if in_dim != 2:
                raise ValueError('this research projection is declared for 2-D inputs')
            radial = torch.tensor(architecture['radial_bands'], dtype=torch.float32)*torch.pi
            angles = 2*torch.pi*torch.rand(len(radial),generator=torch.Generator().manual_seed(architecture['orientation_seed']))
            self.register_buffer('projection',torch.stack([angles.cos(),angles.sin()],1)*radial[:,None])
            dim = in_dim + 2*len(radial)
        else:
            raise ValueError('unknown research Fourier features')
        layers=[]
        for i in range(n_hidden):
            layers.append(nn.Linear(dim,hidden_dim))
            kind=architecture.get('first_activation',architecture['activation']) if i==0 else architecture['activation']
            if kind=='leaky_relu':layers.append(nn.LeakyReLU(.2))
            elif kind=='silu':layers.append(nn.SiLU())
            elif kind=='softplus':layers.append(nn.Softplus(beta=architecture['beta']))
            elif kind=='tanh':layers.append(nn.Tanh())
            else:raise ValueError('unknown smooth activation')
            dim=hidden_dim
        layers.append(nn.Linear(dim,1))
        self.net=nn.Sequential(*layers)
        # Created after the MLP, so the MLP's initialization is unchanged by the option.
        self.skip=None
        if architecture.get('raw_skip'):
            self.skip=nn.Linear(in_dim,1,bias=False)
            nn.init.zeros_(self.skip.weight)

    def encode(self,x):
        phase=(x.unsqueeze(-1)*self.freqs).flatten(1) if self.features=='axis' else x@self.projection.T
        return torch.cat([x,phase.sin(),phase.cos()],dim=1)

    def forward(self,x):
        out=self.net(self.encode(x)).squeeze(-1)
        return out if self.skip is None else out+self.skip(x).squeeze(-1)


def constructor(architecture):
    card=deepcopy(architecture)
    def create(in_dim=2,hidden_dim=64,n_hidden=2,fourier=2):
        return SmoothFourierCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=card)
    return create
