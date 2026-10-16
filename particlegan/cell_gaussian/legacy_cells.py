"""Consumed-real data cells and local MoG output moments; no oracle inputs."""
from collections import deque
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from .parent_control import log_mass_deviance_loss


@torch.no_grad()
def fit_data_cells(real_bank,rows):
    n,d=real_bank.shape
    k=min(n-1,max(d+1,math.ceil(math.log(n))))
    distance=torch.cdist(real_bank,real_bank)
    distance.fill_diagonal_(float('inf'))
    neighbors=distance.topk(k,largest=False).indices.cpu().numpy()
    rr=np.repeat(np.arange(n),k)
    graph=coo_matrix((np.ones(n*k),(rr,neighbors.reshape(-1))),shape=(n,n)).tocsr()
    number,lab=connected_components(graph,directed=False)
    labels=torch.tensor(lab,device=real_bank.device,dtype=torch.long)
    counts=torch.bincount(labels,minlength=number)
    assert int(counts.min())>=k+1
    means=real_bank.new_zeros(number,d).index_add_(0,labels,real_bank)/counts[:,None]
    centered=real_bank-means[labels]
    products=centered[:,:,None]*centered[:,None,:]
    covariance=real_bank.new_zeros(number,d,d).index_add_(0,labels,products)/counts[:,None,None]
    mu=covariance.diagonal(dim1=-2,dim2=-1).mean(-1)
    alpha=covariance.square().mean((-2,-1))
    denominator=(counts+1)*(alpha-mu.square()/d)
    shrink=torch.where(denominator>0,(alpha+mu.square())/denominator,torch.ones_like(mu)).clamp(0,1)
    unbiased=covariance*counts[:,None,None]/(counts-1)[:,None,None]
    eye=torch.eye(d,device=real_bank.device,dtype=real_bank.dtype)
    cov=(1-shrink[:,None,None])*unbiased+shrink[:,None,None]*unbiased.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
    scale=(real_bank-real_bank.mean(0)).square().mean().clamp_min(torch.finfo(real_bank.dtype).eps)
    jitter=torch.finfo(real_bank.dtype).eps*scale
    cov=cov+jitter*eye
    sign,logdet=torch.linalg.slogdet(cov);assert bool((sign>0).all())
    expected=counts.double().cpu().numpy()*rows/n
    allocations=np.floor(expected).astype(np.int64)
    rest=rows-int(allocations.sum())
    order=np.lexsort((np.arange(number),-(expected-allocations)))
    allocations[order[:rest]]+=1
    slots=torch.tensor(np.repeat(np.arange(number),allocations),device=real_bank.device,dtype=torch.long)
    assert len(slots)==rows and np.all(allocations>0)
    return dict(means=means,covariance=cov,inverse=torch.linalg.inv(cov),logdet=logdet,slots=slots,
        jitter=jitter,counts=counts,allocations=torch.tensor(allocations,device=real_bank.device),
        shrink=shrink,neighbor_k=k,reference_rows=n)


def cubature_moments(generator,prior):
    centers=prior.means();rows,zdim=centers.shape
    assert type(prior).__name__=='MoGParticlePrior' and not prior.standardize
    assert float(prior.sigma)>0
    eye=torch.eye(zdim,device=centers.device,dtype=centers.dtype)
    nodes=torch.cat([eye,-eye])*math.sqrt(zdim)
    latent=centers[:,None,:]+prior.sigma*nodes[None]
    outputs=generator(latent.reshape(-1,zdim)).reshape(rows,2*zdim,-1)
    mean=outputs.mean(1);delta=outputs-mean[:,None]
    cov=torch.einsum('mni,mnj->mij',delta,delta)/(2*zdim)
    return mean,cov,rows*2*zdim


def moment_cost(mean,cov,reference):
    d=mean.shape[-1]
    eye=torch.eye(d,device=mean.device,dtype=mean.dtype)
    cov=cov+reference['jitter']*eye
    sign,logdet=torch.linalg.slogdet(cov)
    assert bool((sign.detach()>0).all())
    delta=mean[:,None]-reference['means'][None]
    placement=torch.einsum('mki,kij,mkj->mk',delta,reference['inverse'],delta)/d
    trace=torch.einsum('kij,mji->mk',reference['inverse'],cov)
    shape=(trace-logdet[:,None]+reference['logdet'][None]-d)/d
    return placement+shape,placement,shape


def assigned_loss(mean,cov,reference):
    cost,placement,shape=moment_cost(mean,cov,reference)
    expanded=cost[:,reference['slots']]
    rr,cc=linear_sum_assignment(expanded.detach().double().cpu().numpy())
    assert np.array_equal(rr,np.arange(len(mean)))
    index=torch.tensor(cc,device=mean.device,dtype=torch.long)
    assignment=reference['slots'][index]
    rows=torch.arange(len(mean),device=mean.device)
    loss=cost[rows,assignment].mean()
    parts=dict(placement=placement[rows,assignment].mean(),shape=shape[rows,assignment].mean())
    return loss,parts,assignment


class ConsumedCellGaussianKL:
    def __init__(self):
        self.bank=deque(maxlen=8);self.steps=0;self.reference=None;self.fit_step=0
        self.forward_rows=0;self.reference_row_usages=0;self.last={};self.fit_history=[]
    def __call__(self,fake,real,trainer):
        self.bank.append(real.detach().clone());self.steps+=1
        if len(self.bank)<8:
            value,parts=log_mass_deviance_loss(fake,real)
            self.last=dict(stage='EXACT_PARENT_WARMUP',step=self.steps,reference_rows=len(self.bank)*len(real),extra_forward_rows=0)
            return value,parts
        if self.reference is None or self.steps%8==0:
            bank=torch.cat(list(self.bank))
            self.reference=fit_data_cells(bank,trainer.prior.num_particles)
            self.fit_step=self.steps;self.reference_row_usages+=len(bank)
            self.fit_history.append(dict(step=self.steps,reference_rows=len(bank),neighbor_k=self.reference['neighbor_k'],
                cells=len(self.reference['means']),data_counts=self.reference['counts'].cpu().tolist(),
                allocations=self.reference['allocations'].cpu().tolist(),
                OAS_shrink=self.reference['shrink'].cpu().tolist()))
        mean,cov,extra=cubature_moments(trainer.G,trainer.prior)
        value,parts,assignment=assigned_loss(mean,cov,self.reference)
        self.forward_rows+=extra
        self.last=dict(stage='GAUSSIAN_CELL_KL',step=self.steps,fit_step=self.fit_step,
            reference_rows=self.reference['reference_rows'],cells=len(self.reference['means']),extra_forward_rows=extra,
            predicted_means=mean.detach().cpu().clone(),predicted_covariances=cov.detach().cpu().clone(),
            assigned_data_cells=assignment.cpu().clone())
        return value,{k:v.detach() for k,v in parts.items()}
    def state_dict(self):
        return dict(bank=[v.detach().clone() for v in self.bank],steps=self.steps,
            reference=self.reference,fit_step=self.fit_step,forward_rows=self.forward_rows,
            reference_row_usages=self.reference_row_usages,fit_history=self.fit_history)
    def load_state_dict(self,state):
        self.bank=deque((v.detach().clone() for v in state['bank']),maxlen=8)
        for key in ['steps','reference','fit_step','forward_rows','reference_row_usages','fit_history']:setattr(self,key,state[key])
