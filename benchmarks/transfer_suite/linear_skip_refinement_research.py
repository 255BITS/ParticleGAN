"""Final bounded width/Softplus refinements of a pointwise raw-linear bypass."""
from copy import deepcopy
from torch import nn
from .smooth_critic_research import SmoothFourierCritic

ARCHITECTURES=[
 dict(name='linear_skip_d64_beta6',hidden=64,beta=6.),
 dict(name='linear_skip_d64_beta10',hidden=64,beta=10.),
 dict(name='linear_skip_d96_beta5',hidden=96,beta=5.),
 dict(name='linear_skip_d96_beta6',hidden=96,beta=6.),
]
for card in ARCHITECTURES:
 card.update(features='axis_fourier',activation='softplus',layers=2,fourier_bands=2,skip='raw_linear',
             skip_bias=False,skip_output_initialization='zero',normalization='none')


class LinearSkipCritic(nn.Module):
 """Same axis-Fourier critic plus a learned, initially zero raw linear score."""
 def __init__(self,in_dim=2,hidden_dim=64,n_hidden=2,fourier=2,*,architecture):
  super().__init__();self.architecture=deepcopy(architecture)
  if in_dim!=2 or hidden_dim!=architecture['hidden'] or n_hidden!=2 or fourier!=2:
   raise ValueError('host dimensions must match the declared linear-skip card')
  base=dict(name='base',features='axis',activation='softplus',beta=architecture['beta'])
  self.main=SmoothFourierCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=base)
  self.skip=nn.Linear(in_dim,1,bias=False);nn.init.zeros_(self.skip.weight)

 def forward(self,x):
  return self.main(x).reshape(-1)+self.skip(x).squeeze(-1)


def constructor(architecture):
 card=deepcopy(architecture)
 def create(in_dim=2,hidden_dim=64,n_hidden=2,fourier=2):
  return LinearSkipCritic(in_dim,hidden_dim,n_hidden,fourier,architecture=card)
 return create
