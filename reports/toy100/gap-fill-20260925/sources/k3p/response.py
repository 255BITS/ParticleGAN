"""Amplify direct sample-particle Adam response for aligned relative gradients.
No target statistics, task identity, evaluations, extra forwards, or extra steps.
"""
import torch
from particlegan import ParticlePrior
prior_ids=set()
_original_prior_init=ParticlePrior.__init__
def prior_init(self,*args,**kwargs):
    _original_prior_init(self,*args,**kwargs)
    prior_ids.update(id(p) for p in self.parameters())
ParticlePrior.__init__=prior_init
previous={}
receipt={'formula':'gain=1+relu(cos(center(g_t),center(g_previous)))', 'scope':'direct sample particles only; registered ParticlePrior parameters excluded','particle_betas':[0.,.9], 'calls':0,'rows':[],'history_devices':[]}

def gain_from(current, old):
    if old is None:return 1.,0.
    cosine=float(torch.nn.functional.cosine_similarity(current,old,dim=0,eps=1e-12))
    return 1.+max(0.,min(1.,cosine)),cosine

def begin(opt):
    saved=[]
    for index,group in enumerate(opt.param_groups):
        particles=[p for p in group['params'] if group.get('_comparison_prior',False) and id(p) not in prior_ids]
        if not particles:continue
        assert len(particles)==len(group['params'])
        residuals=[(p.grad.detach()-p.grad.detach().mean(dim=0,keepdim=True)).flatten() for p in particles if p.grad is not None]
        if not residuals:continue
        current=torch.cat(residuals)
        key=(id(opt),index)
        gain,cosine=gain_from(current,previous.get(key))
        previous[key]=current.clone()
        assert previous[key].device == current.device
        saved.append((group,group['lr'],group['betas']))
        group['betas']=(0.,.9)
        group['lr']*=gain
        receipt['calls']+=1
        receipt['rows'].append(dict(call=receipt['calls'],group=index,gain=gain,relative_gradient_cosine=cosine,scheduled_lr=saved[-1][1],effective_lr=group['lr']))
    receipt['history_devices']=sorted({str(t.device) for t in previous.values()})
    return saved

def end(saved):
    for group,lr,betas in saved:
        group['lr']=lr
        group['betas']=betas
