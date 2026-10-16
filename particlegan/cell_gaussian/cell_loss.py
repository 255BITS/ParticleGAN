"""Stable blind cells and cumulative training-only moments; no oracle inputs."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .legacy_cells import fit_data_cells,cubature_moments,moment_cost,assigned_loss
from .parent_control import log_mass_deviance_loss

@torch.no_grad()
def bootstrap_labels(real):
    n,d=real.shape;k=min(n-1,max(d+1,math.ceil(math.log(n))))
    distance=torch.cdist(real,real);distance.fill_diagonal_(float('inf'))
    neighbors=distance.topk(k,largest=False).indices.cpu().numpy()
    rr=np.repeat(np.arange(n),k)
    graph=coo_matrix((np.ones(n*k),(rr,neighbors.reshape(-1))),shape=(n,n)).tocsr()
    number,labels=connected_components(graph,directed=False)
    return torch.tensor(labels,device=real.device,dtype=torch.long),number,k

class StreamingCells:
    def __init__(self):self.counts=None;self.mean=None;self.M2=None;self.neighbor_k=None
    @torch.no_grad()
    def initialize(self,real):
        labels,number,self.neighbor_k=bootstrap_labels(real)
        self.counts=torch.zeros(number,device=real.device,dtype=torch.long)
        self.mean=torch.zeros(number,real.shape[1],device=real.device,dtype=torch.float64)
        self.M2=torch.zeros(number,real.shape[1],real.shape[1],device=real.device,dtype=torch.float64)
        self.add(real,labels)
        return labels
    @torch.no_grad()
    def add(self,real,labels=None):
        values=real.double();number=len(self.mean)
        if labels is None:labels=torch.cdist(values,self.mean).argmin(1)
        bn=torch.bincount(labels,minlength=number)
        sums=torch.zeros_like(self.mean).index_add_(0,labels,values)
        bm=sums/bn.clamp_min(1)[:,None]
        centered=values-bm[labels]
        bM2=torch.zeros_like(self.M2).index_add_(0,labels,centered[:,:,None]*centered[:,None,:])
        total=self.counts+bn;delta=bm-self.mean
        ratio=bn.double()/total.clamp_min(1)
        correction=self.counts.double()*bn.double()/total.clamp_min(1)
        self.M2=self.M2+bM2+correction[:,None,None]*delta[:,:,None]*delta[:,None,:]
        self.mean=self.mean+ratio[:,None]*delta
        self.counts=total
        return labels
    @torch.no_grad()
    def reference(self,rows,dtype):
        counts=self.counts;n=int(counts.sum());d=self.mean.shape[1]
        emp=self.M2/counts[:,None,None]
        mu=emp.diagonal(dim1=-2,dim2=-1).mean(-1);alpha=emp.square().mean((-2,-1))
        denom=(counts+1)*(alpha-mu.square()/d)
        shrink=torch.where(denom>0,(alpha+mu.square())/denom,torch.ones_like(mu)).clamp(0,1)
        unbiased=self.M2/(counts-1)[:,None,None]
        eye=torch.eye(d,device=self.mean.device,dtype=torch.float64)
        cov=(1-shrink[:,None,None])*unbiased+shrink[:,None,None]*unbiased.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
        global_mean=(self.mean*counts[:,None]).sum(0)/n
        global_M2=(self.M2+counts[:,None,None]*(self.mean-global_mean)[:,:,None]*(self.mean-global_mean)[:,None,:]).sum(0)
        scale=(global_M2.diagonal().sum()/(n*d)).clamp_min(torch.finfo(dtype).eps)
        jitter=(torch.finfo(dtype).eps*scale).to(dtype)
        cov=cov.to(dtype)+jitter*eye.to(dtype)
        sign,logdet=torch.linalg.slogdet(cov);assert bool((sign>0).all())
        expected=counts.double().cpu().numpy()*rows/n;allocations=np.floor(expected).astype(np.int64)
        rest=rows-int(allocations.sum());order=np.lexsort((np.arange(len(counts)),-(expected-allocations)))
        allocations[order[:rest]]+=1
        slots=torch.tensor(np.repeat(np.arange(len(counts)),allocations),device=self.mean.device,dtype=torch.long)
        assert len(slots)==rows and np.all(allocations>0)
        return dict(means=self.mean.to(dtype),covariance=cov,inverse=torch.linalg.inv(cov),logdet=logdet,slots=slots,
            jitter=jitter,counts=counts.clone(),allocations=torch.tensor(allocations,device=self.mean.device),
            shrink=shrink.to(dtype),neighbor_k=self.neighbor_k,reference_rows=n)
    def state_dict(self):return deepcopy(dict(counts=self.counts,mean=self.mean,M2=self.M2,neighbor_k=self.neighbor_k))
    def load_state_dict(self,state):
        for k,v in deepcopy(state).items():setattr(self,k,v)

class ConsumedCellGaussianKL:
    """Compatibility name for the unchanged adapter; candidate identity is streaming v1."""
    def __init__(self):
        self.bank=deque(maxlen=8);self.steps=0;self.reference=None;self.fit_step=0
        self.forward_rows=0;self.reference_row_usages=0;self.last={};self.fit_history=[]
        self.statistics=StreamingCells()
    def __call__(self,fake,real,trainer):
        self.bank.append(real.detach().clone());self.steps+=1
        if self.steps<8:
            value,parts=log_mass_deviance_loss(fake,real)
            self.last=dict(stage='EXACT_PARENT_WARMUP',step=self.steps,reference_rows=len(self.bank)*len(real),extra_forward_rows=0)
            return value,parts
        if self.reference is None:
            bank=torch.cat(list(self.bank));labels=self.statistics.initialize(bank)
            self.reference=fit_data_cells(bank,trainer.prior.num_particles)
            assert torch.equal(self.statistics.counts,self.reference['counts'])
            assert torch.allclose(self.statistics.mean.to(real.dtype),self.reference['means'],atol=1e-6,rtol=1e-6)
            self.reference_row_usages=len(bank)
        else:
            self.statistics.add(real.detach());self.reference_row_usages+=len(real)
        if self.steps==8 or self.steps%8==0:
            if self.steps!=8:self.reference=self.statistics.reference(trainer.prior.num_particles,real.dtype)
            self.fit_step=self.steps
            self.fit_history.append(dict(step=self.steps,reference_rows=self.reference['reference_rows'],neighbor_k=self.reference['neighbor_k'],
                cells=len(self.reference['means']),data_counts=self.reference['counts'].cpu().tolist(),
                allocations=self.reference['allocations'].cpu().tolist(),OAS_shrink=self.reference['shrink'].cpu().tolist()))
        mean,cov,extra=cubature_moments(trainer.G,trainer.prior)
        value,parts,assignment=self.assigned_loss(mean,cov,self.reference);self.forward_rows+=extra
        self.last=dict(stage='STREAMING_GAUSSIAN_CELL_KL',step=self.steps,fit_step=self.fit_step,
            reference_rows=self.reference['reference_rows'],cells=len(self.reference['means']),extra_forward_rows=extra,
            predicted_means=mean.detach().cpu().clone(),predicted_covariances=cov.detach().cpu().clone(),
            assigned_data_cells=assignment.cpu().clone())
        return value,{k:v.detach() for k,v in parts.items()}
    def state_dict(self):
        return deepcopy(dict(bank=list(self.bank),steps=self.steps,reference=self.reference,fit_step=self.fit_step,
            forward_rows=self.forward_rows,reference_row_usages=self.reference_row_usages,fit_history=self.fit_history,
            statistics=self.statistics.state_dict()))
    def load_state_dict(self,state):
        state=deepcopy(state);self.bank=deque(state['bank'],maxlen=8)
        for key in ['steps','reference','fit_step','forward_rows','reference_row_usages','fit_history']:setattr(self,key,state[key])
        self.statistics.load_state_dict(state['statistics'])
