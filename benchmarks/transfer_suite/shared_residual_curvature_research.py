"""Initially-zero local quadratic corrections to an unchanged raw smooth MLP."""
from copy import deepcopy
import torch
from torch import nn
from .shared_critic_research import ARCHITECTURES as BASES, SharedResearchCritic


def _card(base, count, width):
    return dict(name=f'curvature_{base}_q{count}_w{str(width).replace(".", "p")}',
                implementation='shared_residual_curvature_v1', base=base,
                hidden=128 if base=='raw_silu128_l3' else 96, layers=3, fourier=0,
                experts=count, width=width, center_scale=2., feature_seed=0,
                coefficients_initialization='zero', train_centers=False, train_widths=False,
                normalization='none', basis=['1','dx','dy','dx*dx','dx*dy','dy*dy'])


ARCHITECTURES = [_card(base, count, width)
                 for base in ['raw_silu128_l3','raw_softplus96_l3']
                 for count,width in [(32,1.),(64,1.),(32,.5)]]


class ResidualCurvatureCritic(nn.Module):
    """Same initial MLP score plus a generic pointwise local quadratic branch.

    The MLP is constructed first. Local buffers use a separate fixed generator,
    and branch coefficients are zeros, preserving both its weights and the global
    RNG state after initialization. The usual cap applies to the summed score;
    no branch loss, optimizer group, gradient scaling or target information exists.
    """
    def __init__(self,in_dim=2,hidden_dim=128,n_hidden=3,fourier=0,*,architecture):
        super().__init__()
        self.card=card=deepcopy(architecture)
        if (in_dim,hidden_dim,n_hidden,fourier)!=(2,card['hidden'],3,0):
            raise ValueError('host dimensions differ from residual-curvature card')
        base=next(b for b in BASES if b['name']==card['base'])
        self.main=SharedResearchCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=base)
        self.register_buffer('centers',card['center_scale']*torch.randn(card['experts'],2,
                             generator=torch.Generator().manual_seed(card['feature_seed'])))
        self.coefficients=nn.Parameter(torch.zeros(card['experts'],6))

    def forward(self,x):
        delta=(x[:,None,:]-self.centers[None,:,:])/self.card['width']
        weights=(-.5*delta.square().sum(-1)).softmax(1)
        dx,dy=delta.unbind(-1)
        basis=torch.stack((torch.ones_like(dx),dx,dy,dx.square(),dx*dy,dy.square()),-1)
        local=(basis*self.coefficients[None,:,:]).sum(-1)
        return self.main(x)+(weights*local).sum(1)


def variant(card):
    return dict(name=card['name'],overrides=dict(d_hidden=card['hidden'],d_layers=3,
                fourier=0,research_discriminator=deepcopy(card)))


def constructor(card):
    card=deepcopy(card)
    def create(in_dim=2,hidden_dim=128,n_hidden=3,fourier=0):
        return ResidualCurvatureCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=card)
    return create
