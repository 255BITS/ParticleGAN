"""Opt-in finite output balancing of generator and learned-prior roles."""
from copy import deepcopy
import math

import torch
from torch.func import functional_call


class RoleMotionBalance:
    """Cap network-only RMS travel by the actual prior-only RMS travel.

    The reference uses exactly the consumed latent rows and kernel jitters.
    The prior proposal is retained; the generator ray is reduced by halving
    until its measured finite motion meets the cap. No target information,
    extra sampling, optimizer calls or persistent control state is introduced.
    Counters are checkpointed. A zero prior response restores G exactly.
    """
    def __init__(self):
        self.summary=dict(updates=0,limited=0,rejected=0,probe_calls=0,
                          max_network_prior_ratio=0.,minimum_scale=1.)
        for key in ('scale','network_proposed','prior','network_accepted','joint',
                    'cross','network_center','prior_center','network_deformation',
                    'prior_deformation','joint_deformation','work'):
            self.summary[key+'_sum']=0.

    @staticmethod
    def rms(x):
        return float(x.flatten(1).square().sum(1).mean().sqrt())

    @torch.no_grad()
    def step(self,optimizer,generator,prior,latent,indices):
        before={k:p.detach().clone() for k,p in generator.named_parameters()}
        old_latent=latent.detach().clone()
        centers=prior.z[indices].detach().clone()
        optimizer.step()
        after={k:p.detach().clone() for k,p in generator.named_parameters()}
        deltas={k:after[k]-p for k,p in before.items()}
        new_centers=prior.z[indices].detach()
        new_latent=old_latent+(new_centers-centers)
        def old(z):return functional_call(generator,before,(z,))
        y0=old(old_latent);yp=old(new_latent)
        yg=generator(old_latent)
        budget=self.rms(yp-y0);proposed=self.rms(yg-y0)
        if not math.isfinite(budget) or not math.isfinite(proposed):
            raise ValueError('role motion requires finite output proposals')
        scale=1.;calls=3
        # Scale1 is a true no-op: never recompose an unchanged floating tensor.
        if proposed>budget:
            scale=min(1.,budget/proposed) if proposed else 1.
            for _ in range(24):
                if scale==0:break
                for name,p in generator.named_parameters():
                    p.copy_(before[name]+scale*deltas[name])
                yg=generator(old_latent);calls+=1
                value=self.rms(yg-y0)
                if math.isfinite(value) and value<=budget:break
                scale*=.5
            else:scale=0.
            if scale==0:
                for name,p in generator.named_parameters():p.copy_(before[name])
                yg=y0
        accepted=self.rms(yg-y0)
        if accepted>budget:
            raise RuntimeError('role-motion finite network bound failed')
        yj=generator(new_latent)
        c0=old(centers);cg=generator(centers);cp=old(new_centers);cj=generator(new_centers)
        calls+=5
        network=yg-y0;prior_motion=yp-y0;joint=yj-y0
        network_center=cg-c0;prior_center=cp-c0
        values=dict(scale=scale,network_proposed=proposed,prior=budget,
            network_accepted=accepted,joint=self.rms(joint),
            cross=self.rms(joint-network-prior_motion),
            network_center=self.rms(network_center),prior_center=self.rms(prior_center),
            network_deformation=self.rms(network-network_center),
            prior_deformation=self.rms(prior_motion-prior_center),
            joint_deformation=self.rms(joint-(cj-c0)),
            work=-sum(float((p.grad*(p-before[k])).sum())
                      for k,p in generator.named_parameters() if p.grad is not None))
        row=self.summary;row['updates']+=1;row['limited']+=int(scale<1)
        row['rejected']+=int(scale==0);row['probe_calls']+=calls
        row['max_network_prior_ratio']=max(row['max_network_prior_ratio'],accepted/budget if budget else 0.)
        row['minimum_scale']=min(row['minimum_scale'],scale)
        for key,value in values.items():row[key+'_sum']+=value

    def state_dict(self):
        return dict(schema=1,rule='finite_network_rms_le_prior_rms',summary=deepcopy(self.summary))

    def validate_state_dict(self,state,steps):
        if (not isinstance(state,dict) or set(state)!={'schema','rule','summary'}
                or state['schema']!=1 or state['rule']!='finite_network_rms_le_prior_rms'):
            raise ValueError('invalid role-motion checkpoint identity')
        row=state['summary']
        if (not isinstance(row,dict) or row.keys()!=self.summary.keys()
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in row.values())
                or row['updates']!=steps or not 0<=row['rejected']<=row['limited']<=steps
                or not 0<=row['max_network_prior_ratio']<=1
                or not 0<=row['minimum_scale']<=1
                or any(type(row[k]) is not int for k in ('updates','limited','rejected','probe_calls'))
                or row['probe_calls']<8*steps):
            raise ValueError('invalid role-motion checkpoint counters')

    def load_state_dict(self,state):
        self.summary=deepcopy(state['summary'])
