"""Uncalibrated,training-gradient-only attenuation; no evaluator or RNG."""
from copy import deepcopy
import math
import torch

class GradientSignal:
    def __init__(self,roles):self.roles=tuple(roles);self.packets={}
    @torch.no_grad()
    def scale(self,groups):
        parameters={r:[p for g in groups if g['role']==r for p in g['params']] for r in self.roles}
        scales={}
        for role,params in parameters.items():
            if not params:continue
            gradients=[torch.zeros_like(p,dtype=torch.float64) if p.grad is None else p.grad.double() for p in params]
            if any(not bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('Nonfinite gradient-signal input')
            if role not in self.packets:
                self.packets[role]=dict(means=[torch.zeros_like(g) for g in gradients],second=gradients[0].new_zeros(()),weight=0.,weight2=0.,count=0,zero_steps=0,last={})
            state=self.packets[role];beta=7/8;alpha=1-beta
            state['means']=[m*beta+g*alpha for m,g in zip(state['means'],gradients)]
            state['second']=state['second']*beta+alpha*sum(g.square().sum() for g in gradients)
            state['weight']=state['weight']*beta+alpha;state['weight2']=state['weight2']*beta*beta+alpha*alpha;state['count']+=1
            effective=state['weight']**2/state['weight2'];mean2=sum((m/state['weight']).square().sum() for m in state['means'])
            empirical=(state['second']/state['weight']-mean2).clamp_min(0)
            variance=empirical/(1-1/effective) if effective>1+1e-12 else empirical.new_zeros(())
            signal=(mean2-variance/effective).clamp_min(0);rho=float((signal/(signal+variance+1e-16)).clamp(0,1))
            if not math.isfinite(rho):raise ValueError('Nonfinite gradient-signal attenuation')
            if rho==0:state['zero_steps']+=1
            state['last']=dict(rho=rho,effective_gradient_count=effective,squared_signal=float(signal),gradient_variance=float(variance))
            scales[role]=rho
        return scales
    def state_dict(self):return deepcopy(dict(roles=list(self.roles),packets=self.packets))
    def validate(self,value,groups):
        if not isinstance(value,dict) or set(value)!={'roles','packets'} or value['roles']!=list(self.roles):raise ValueError('Gradient-signal checkpoint roles mismatch')
        expected={r:[p for g in groups if g['role']==r for p in g['params']] for r in self.roles}
        if not isinstance(value['packets'],dict) or not set(value['packets'])<=set(expected):raise ValueError('Unknown gradient-signal role')
        for role,state in value['packets'].items():
            if set(state)!={'means','second','weight','weight2','count','zero_steps','last'}:raise ValueError('Malformed gradient-signal packet')
            if type(state['count']) is not int or state['count']<1 or type(state['zero_steps']) is not int or not 0<=state['zero_steps']<=state['count']:raise ValueError('Bad gradient-signal counters')
            if not 0<state['weight']<=1 or not 0<state['weight2']<=1 or not isinstance(state['means'],list) or len(state['means'])!=len(expected[role]):raise ValueError('Bad gradient-signal weights/shape')
            tensors=state['means']+[state['second']]
            if any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float64 or not bool(torch.isfinite(t).all()) for t in tensors):raise ValueError('Bad gradient-signal tensors')
            if state['second'].numel()!=1 or bool(state['second']<0) or any(m.shape!=p.shape for m,p in zip(state['means'],expected[role])):raise ValueError('Bad gradient-signal second moment')
            if not isinstance(state['last'],dict) or any(type(v) not in (int,float) or not math.isfinite(v) for v in state['last'].values()) or not 0<=state['last'].get('rho',-1)<=1:raise ValueError('Bad gradient-signal receipt')
    def load(self,value,groups):
        self.validate(value,groups);self.packets=deepcopy(value['packets'])
        for role,state in self.packets.items():
            params=[p for g in groups if g['role']==role for p in g['params']]
            state['means']=[m.to(device=p.device,dtype=torch.float64) for m,p in zip(state['means'],params)]
            state['second']=state['second'].to(device=params[0].device,dtype=torch.float64)
