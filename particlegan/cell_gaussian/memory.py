"""Data-only finite-memory references; never consumes evaluator signals or RNG."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
import torch
from scipy.stats import chi2
from .cell_loss import StreamingCells

def empty_like(value):
    result=StreamingCells()
    result.counts=torch.zeros_like(value.counts)
    result.mean=value.mean.clone()
    result.M2=torch.zeros_like(value.M2)
    result.neighbor_k=value.neighbor_k
    return result

def summarize(bank,identity):
    values=torch.cat(list(bank))
    result=empty_like(identity)
    labels=torch.cdist(values.double(),identity.mean).argmin(1)
    result.add(values,labels)
    if not bool((result.counts>1).all()):
        raise ValueError('Finite-memory cell has insufficient observed training rows')
    return result

def merge(a,b):
    result=empty_like(b)
    total=a.counts+b.counts
    delta=b.mean-a.mean
    result.mean=a.mean+delta*(b.counts.double()/total.clamp_min(1))[:,None]
    result.M2=a.M2+b.M2+(a.counts.double()*b.counts.double()/total.clamp_min(1))[:,None,None]*delta[:,:,None]*delta[:,None,:]
    result.counts=total
    return result

def minimum_change_allocation(expected, previous):
    """Exact positive capacities with at most one-kernel rounding error.

    Minimize L1 movement from previous capacities among floor/ceiling choices.
    Equal movement costs retain the original largest-remainder tie ordering.
    The caller first exercises the unchanged parent empty-quota rejection.
    """
    expected=np.asarray(expected,dtype=np.float64)
    previous=np.asarray(previous,dtype=np.int64)
    if expected.ndim!=1 or previous.shape!=expected.shape or not np.isfinite(expected).all():
        raise ValueError('Invalid quota inputs')
    total=int(previous.sum());floor=np.floor(expected).astype(np.int64)
    ceil=np.ceil(expected).astype(np.int64)
    if np.any(previous<=0) or abs(expected.sum()-total)>1e-6*max(1,total):
        raise ValueError('Invalid previous quota total')
    allocation=floor.copy()
    mandatory=np.flatnonzero(floor==0)
    if np.any(ceil[mandatory]<1):raise ValueError('No positive floor/ceiling quota')
    allocation[mandatory]=1
    remaining=total-int(allocation.sum())
    eligible=np.flatnonzero((ceil>floor)&(floor>0))
    if remaining<0 or remaining>len(eligible):raise ValueError('Infeasible exact quota')
    movement_cost=np.abs(floor[eligible]+1-previous[eligible])-np.abs(floor[eligible]-previous[eligible])
    order=np.lexsort((eligible,-(expected[eligible]-floor[eligible]),movement_cost))
    allocation[eligible[order[:remaining]]]+=1
    if (allocation.sum()!=total or np.any(allocation<=0) or
            np.any(allocation<floor) or np.any(allocation>ceil)):
        raise ValueError('Invalid minimum-change quota')
    return allocation

class ReferenceMemory:
    POLICIES=('rolling8','block8','exponential8','evidence_reset')
    def __init__(self,policy,quota_policy='largest_remainder'):
        if policy not in self.POLICIES:raise ValueError('Unknown training reference memory')
        if quota_policy not in ('largest_remainder','minimum_change') or (quota_policy=='minimum_change' and policy!='exponential8'):
            raise ValueError('Unsupported quota policy')
        self.quota_policy=quota_policy;self.previous_allocations=None
        self.policy=policy;self.bank=deque(maxlen=8);self.statistics=StreamingCells()
        self.history=None;self.w2=None;self.steps=0;self.seen_rows=0;self.checks=0;self.events=[]

    @torch.no_grad()
    def consume(self,real):
        outgoing=self.bank[0] if len(self.bank)==8 else None
        self.bank.append(real.detach().clone());self.steps+=1;self.seen_rows+=len(real)
        if self.steps<8:return False
        if self.steps==8:
            self.statistics.initialize(torch.cat(list(self.bank)))
            self.w2=self.statistics.counts.double().clone()
            self.history=empty_like(self.statistics)
            return True
        if self.policy=='rolling8':
            self.statistics=summarize(self.bank,self.statistics);return True
        if self.policy=='block8':
            if self.steps%8==0:self.statistics=summarize(self.bank,self.statistics);return True
            return False
        if self.policy=='exponential8':
            labels=torch.cdist(real.double(),self.statistics.mean).argmin(1)
            bn=torch.bincount(labels,minlength=len(self.statistics.counts)).double()
            self.statistics.counts=self.statistics.counts.double()*(7/8)
            self.statistics.M2=self.statistics.M2*(7/8)
            self.w2=self.w2*(7/8)**2+bn
            self.statistics.add(real,labels)
            return True
        # Disjoint historical and recent panels; only observed real rows enter.
        self.history.add(outgoing)
        if self.steps%8:return False
        recent=summarize(self.bank,self.statistics)
        self.checks+=1
        if bool((self.history.counts>1).all()):
            k,d=recent.mean.shape
            variance=self.history.M2/(self.history.counts-1)[:,None,None]/self.history.counts[:,None,None]
            variance=variance+recent.M2/(recent.counts-1)[:,None,None]/recent.counts[:,None,None]
            scale=(torch.cat(list(self.bank)).double().var(0,unbiased=False).mean()).clamp_min(torch.finfo(torch.float64).eps)
            variance=variance+torch.finfo(torch.float64).eps*scale*torch.eye(d,device=real.device,dtype=torch.float64)
            delta=recent.mean-self.history.mean
            statistic=torch.einsum('ki,kij,kj->k',delta,torch.linalg.inv(variance),delta)
            alpha=.05/(k*self.checks*(self.checks+1))
            threshold=float(chi2.isf(alpha,d))
            if bool((statistic>threshold).any()):
                self.events.append(dict(step=self.steps,mean_change_statistic=statistic.cpu().tolist(),threshold=threshold,alpha=alpha,reference_only_reset=True))
                self.history=empty_like(recent)
        self.statistics=merge(self.history,recent)
        return True

    @torch.no_grad()
    def reference(self,rows,dtype):
        if self.policy!='exponential8':return self.statistics.reference(rows,dtype)
        stats=self.statistics;counts=stats.counts.double();d=stats.mean.shape[1]
        effective=counts.square()/self.w2
        emp=stats.M2/counts[:,None,None]
        denom=counts-self.w2/counts
        if not bool((denom>0).all()):raise ValueError('Invalid effective training reference size')
        unbiased=stats.M2/denom[:,None,None]
        mu=emp.diagonal(dim1=-2,dim2=-1).mean(-1);alpha=emp.square().mean((-2,-1))
        denominator=(effective+1)*(alpha-mu.square()/d)
        shrink=torch.where(denominator>0,(alpha+mu.square())/denominator,torch.ones_like(mu)).clamp(0,1)
        eye=torch.eye(d,device=counts.device,dtype=torch.float64)
        cov=(1-shrink[:,None,None])*unbiased+shrink[:,None,None]*unbiased.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
        total=counts.sum();mean=(stats.mean*counts[:,None]).sum(0)/total
        pooled=(stats.M2+counts[:,None,None]*(stats.mean-mean)[:,:,None]*(stats.mean-mean)[:,None,:]).sum(0)
        jitter=(torch.finfo(dtype).eps*(pooled.diagonal().sum()/(total*d)).clamp_min(torch.finfo(dtype).eps)).to(dtype)
        cov=cov.to(dtype)+jitter*eye.to(dtype)
        sign,logdet=torch.linalg.slogdet(cov)
        if not bool((sign>0).all()):raise ValueError('Invalid weighted covariance')
        expected=(counts/counts.sum()*rows).cpu().numpy();alloc=np.floor(expected).astype(np.int64)
        order=np.lexsort((np.arange(len(counts)),-(expected-alloc)));alloc[order[:rows-int(alloc.sum())]]+=1
        if not np.all(alloc>0):raise ValueError('Weighted reference has an empty integral quota')
        if self.quota_policy=='minimum_change' and self.previous_allocations is not None:
            alloc=minimum_change_allocation(expected,self.previous_allocations.cpu().numpy())
        self.previous_allocations=torch.tensor(alloc,device=counts.device,dtype=torch.long)
        slots=torch.tensor(np.repeat(np.arange(len(counts)),alloc),device=counts.device,dtype=torch.long)
        return dict(means=stats.mean.to(dtype),covariance=cov,inverse=torch.linalg.inv(cov),logdet=logdet,slots=slots,jitter=jitter,
                    counts=counts.clone(),allocations=torch.tensor(alloc,device=counts.device),shrink=shrink.to(dtype),neighbor_k=stats.neighbor_k,
                    reference_rows=float(total),effective_rows=effective.cpu().tolist())

    def state_dict(self):
        return deepcopy(dict(policy=self.policy,quota_policy=self.quota_policy,previous_allocations=self.previous_allocations,bank=list(self.bank),statistics=self.statistics.state_dict(),history=None if self.history is None else self.history.state_dict(),w2=self.w2,steps=self.steps,seen_rows=self.seen_rows,checks=self.checks,events=self.events))

    def load_state_dict(self,state):
        if state['policy']!=self.policy:raise ValueError('Reference policy checkpoint mismatch')
        if state.get('quota_policy')!=self.quota_policy:raise ValueError('Quota policy checkpoint mismatch')
        previous=state.get('previous_allocations')
        if previous is not None and (not isinstance(previous,torch.Tensor) or previous.ndim!=1 or previous.dtype!=torch.long or not bool((previous>0).all())):
            raise ValueError('Invalid previous allocations')
        state=deepcopy(state);self.bank=deque(state['bank'],maxlen=8);self.statistics.load_state_dict(state['statistics'])
        self.previous_allocations=state['previous_allocations']
        self.history=None
        if state['history'] is not None:self.history=StreamingCells();self.history.load_state_dict(state['history'])
        for name in ['w2','steps','seen_rows','checks','events']:setattr(self,name,state[name])
