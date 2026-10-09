"""Bounded finite acceptance of actual optimizer proposals, with replayed noise.

Armijo(1966) sufficient decrease is tested at 1, 1/2, ..., 1/1024 with c=.1.
The same scalar batch objective and the actual joint G/prior displacement are
used. This is not a convergence guarantee for a changing adversarial game.
Optimizer history advances once per proposal, including rejected proposals;
parameters take the accepted fractional displacement (zero on rejection).
"""
from copy import deepcopy
import math
import torch


class FiniteStep:
    def __init__(self):
        self.stats = dict(proposals=0, accepted=0, shrunk=0, rejected=0,
                          downhill=0, downhill_loss_increases=0,
                          full_loss_increases=0, backtracks=0,
                          scale_sum=0., minimum_scale=1., armijo_violations=0)
        self.last = None

    @staticmethod
    def capture(modules):
        """Capture objective-side buffers/RNG before its original forward."""
        return dict(buffers=[(b,b.detach().clone()) for m in modules for b in m.buffers()],
                    cpu=torch.get_rng_state().clone(),
                    cuda={p.device.index:torch.cuda.get_rng_state(p.device).clone()
                          for m in modules for p in m.parameters() if p.device.type=='cuda'})

    @staticmethod
    def restore(snapshot):
        with torch.no_grad():
            for buffer,value in snapshot['buffers']: buffer.copy_(value)
        torch.set_rng_state(snapshot['cpu'])
        for device,state in snapshot['cuda'].items(): torch.cuda.set_rng_state(state,device)

    def apply(self, optimizer, before_loss, objective, replay, modules):
        """Take one optimizer proposal; replay only forward objective checks."""
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        original=[p.detach().clone() for p in parameters]
        gradients=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        history=deepcopy(optimizer.state_dict())
        after_forward=self.capture(modules)
        before=float(before_loss.detach())
        try:
            optimizer.step()
            proposed=[p.detach().clone() for p in parameters]
            delta=[b-a for a,b in zip(original,proposed)]
            slope=sum(float((g.double()*change.double()).sum()) for g,change in zip(gradients,delta))
            accepted=False
            scale=1.
            full_loss=None
            value=None
            backtracks=0
            with torch.no_grad():
                for exponent in range(11):
                    scale=2.**(-exponent)
                    if exponent:
                        for p,a,change in zip(parameters,original,delta):p.copy_(a+scale*change)
                    self.restore(replay)
                    value=float(objective())
                    self.restore(after_forward)
                    if exponent==0:full_loss=value
                    if math.isfinite(value) and ((slope<0 and value<=before+.1*scale*slope and value<before)
                                                or (slope==0 and all(torch.count_nonzero(v)==0 for v in delta)
                                                    and value==before)):
                        accepted=True
                        break
                    backtracks+=int(exponent<10)
                if not accepted:
                    scale=0.
                    for p,a in zip(parameters,original):p.copy_(a)
            self.stats['proposals']+=1
            self.stats['accepted']+=int(accepted)
            self.stats['shrunk']+=int(accepted and scale<1)
            self.stats['rejected']+=int(not accepted)
            self.stats['downhill']+=int(slope<0)
            self.stats['full_loss_increases']+=int(full_loss>before)
            self.stats['downhill_loss_increases']+=int(slope<0 and full_loss>before)
            self.stats['backtracks']+=backtracks
            self.stats['scale_sum']+=scale
            self.stats['minimum_scale']=min(self.stats['minimum_scale'],scale)
            self.last=dict(before_loss=before,full_loss=full_loss,checked_loss=value,
                           slope=slope,scale=scale,accepted=accepted,backtracks=backtracks)
            return self.last
        except Exception:
            with torch.no_grad():
                for p,a in zip(parameters,original):p.copy_(a)
            optimizer.load_state_dict(history)
            raise
        finally:
            self.restore(after_forward)

    def state_dict(self):
        return deepcopy(dict(schema=1,rule='all-proposals-armijo-c0.1-halves10',stats=self.stats,last=self.last))

    @staticmethod
    def validate(state):
        expected=FiniteStep().state_dict()
        if (not isinstance(state,dict) or state.keys()!=expected.keys() or state['schema']!=1
                or state['rule']!=expected['rule'] or state['stats'].keys()!=expected['stats'].keys()):
            raise ValueError('invalid finite-step checkpoint')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in state['stats'].values()):
            raise ValueError('invalid finite-step statistics')
        if state['stats']['accepted']+state['stats']['rejected']!=state['stats']['proposals']:
            raise ValueError('inconsistent finite-step counters')
        if state['last'] is not None and (not isinstance(state['last'],dict)
                or state['last'].keys()!=dict(before_loss=0,full_loss=0,checked_loss=0,slope=0,scale=0,accepted=0,backtracks=0).keys()):
            raise ValueError('invalid finite-step last proposal')

    def load_state_dict(self,state):
        self.validate(state)
        self.stats=deepcopy(state['stats']);self.last=deepcopy(state['last'])
