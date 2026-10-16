"""Deterministic conditional output-shape objectives using consumed data only."""
from functools import lru_cache
import math
import torch

@lru_cache(maxsize=16)
def normal_panel(dimension):
    engine = torch.quasirandom.SobolEngine(dimension, scramble=False)
    u = engine.draw(33, dtype=torch.float64)[1:]
    z = torch.erfinv(2*u-1)*math.sqrt(2)
    return torch.cat((z,-z))

def directions(dimension, like):
    if dimension == 1:return like.new_ones(1,1)
    if dimension != 2:raise ValueError('Conditional shape currently supports D1/D2')
    theta=torch.arange(8,device=like.device,dtype=like.dtype)*math.pi/8
    return torch.stack((theta.cos(),theta.sin()),dim=1)

def select(x):
    if len(x)<=256:return x
    index=torch.linspace(0,len(x)-1,256,device=x.device).round().long()
    return x[index]

def quantiles(x):
    x=x.sort(dim=0).values
    pos=((torch.arange(128,device=x.device,dtype=x.dtype)+.5)/128)*(len(x)-1)
    low=pos.floor().long();high=pos.ceil().long();weight=pos-low
    return x[low]*(1-weight[:,None])+x[high]*weight[:,None]

def discrepancy(x,y,kind):
    if kind=='rbf':
        target_distance=torch.cdist(y,y).square()
        positive=target_distance[target_distance>0]
        bandwidth=(positive.median() if len(positive) else y.new_tensor(0)).clamp_min(torch.finfo(y.dtype).eps).detach()
        kernel=lambda a,b:torch.exp(-torch.cdist(a,b).square()/(2*bandwidth))
        return kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()
    basis=directions(x.shape[1],x);a=x@basis.T;b=y@basis.T
    if kind=='sliced':return (quantiles(a)-quantiles(b)).square().mean()
    if kind=='fourth':
        ma=a.pow(4).mean(0);mb=b.pow(4).mean(0)
        return ((ma-mb)/(1+mb)).square().mean()
    raise ValueError('Unknown conditional shape objective')

def loss(generator,prior,reference,assignment,bank,kind):
    centers=prior.means();nodes=normal_panel(centers.shape[1]).to(centers)
    outputs=generator((centers[:,None]+prior.sigma*nodes[None]).reshape(-1,centers.shape[1]))
    outputs=outputs.reshape(len(centers),len(nodes),-1)
    real=torch.cat(list(bank)).detach()
    labels=torch.cdist(real,reference['means']).argmin(1)
    values=[]
    for c in range(len(reference['means'])):
        x=select(outputs[assignment==c].reshape(-1,outputs.shape[-1]))
        y=select(real[labels==c])
        if len(x)<1 or len(y)<1:raise ValueError('Empty conditional shape panel')
        eigenvalues,vectors=torch.linalg.eigh(reference['covariance'][c])
        inverse_root=(vectors*eigenvalues.rsqrt()[None,:])@vectors.T
        x=(x-x.mean(0))@inverse_root;y=(y-y.mean(0))@inverse_root
        values.append(discrepancy(x,y,kind))
    weights=reference['allocations'].to(centers)/len(centers)
    value=(weights*torch.stack(values)).sum()
    return value,outputs.numel()//outputs.shape[-1],dict(conditional_shape=float(value.detach()),shape_panel_points=len(nodes),shape_kind=kind)
