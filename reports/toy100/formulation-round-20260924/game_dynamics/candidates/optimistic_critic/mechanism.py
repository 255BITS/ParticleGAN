"""One-sided optimistic Adam: correct D memory, retain original G/prior Adam."""
from contextlib import contextmanager
from unittest.mock import patch
import inspect
import torch


class Correction:
    def __init__(self):
        self.calls=0;self.corrected_tensors=0;self.roles={'d':0,'g':0}
        self.optimizers=[];self.models={};self.generators={};self.trainers=[]

    @contextmanager
    def context(self):
        from benchmarks import learned_lr_evaluation as bridge
        from particlegan.training import GANTrainer
        original_role = bridge.optimizer_role
        original_init = GANTrainer.__init__
        def role(opt, local_variables):
            value = original_role(opt, local_variables)
            opt._game_role = value
            return value
        def init(trainer, *args, **kwargs):
            original_init(trainer, *args, **kwargs)
            trainer.opt_d._game_role = 'd'
            trainer.opt_g._game_role = 'g'
            self.trainers.append(trainer)
        with patch.object(bridge, 'optimizer_role', role), patch.object(GANTrainer, '__init__', init):
            yield

    def capture(self,opt):
        if opt in self.optimizers:return
        self.optimizers.append(opt)
        frame=inspect.currentframe()
        try:
            while frame is not None:
                local=frame.f_locals
                if (any(local.get(name) is opt for name in ('opt_d','opt_g','opt_p','opt'))
                        and any(isinstance(value,torch.nn.Module) for value in local.values())):
                    for name,value in local.items():
                        if isinstance(value,torch.nn.Module):self.models[name]=value
                        elif isinstance(value,torch.Generator):self.generators[name]=value
                    policy=local.get('noise_policy')
                    if policy is not None:
                        for name in ('input_stream','output_stream'):
                            value=getattr(policy,name,None)
                            if value is not None:self.generators['noise_'+name]=value
                    break
                frame=frame.f_back
        finally:del frame

    def step(self,opt,original_step,*args,**kwargs):
        self.capture(opt)
        role=opt._game_role;self.roles[role]+=1
        if role=='g':
            result=original_step(opt,*args,**kwargs)
        else:
            entries=[(p,p.detach().clone(),float(group['lr'])) for group in opt.param_groups
                     for p in group['params'] if p.grad is not None]
            result=original_step(opt,*args,**kwargs)
            with torch.no_grad():
                for p,before,lr in entries:
                    assert lr>0 and p.device.type=='cuda'
                    direction=(p-before)/lr
                    previous=opt.state[p].get('game_previous_direction')
                    if previous is not None:
                        p.add_(direction-previous,alpha=lr);self.corrected_tensors+=1
                    opt.state[p]['game_previous_direction']=direction
        self.calls+=1
        return result

    def snapshot(self,path,audit):
        torch.save(dict(models={k:v.state_dict() for k,v in self.models.items()},
            optimizers=[dict(role=o._game_role,parameters=[p.detach().clone() for g in o.param_groups for p in g['params']],state=o.state_dict()) for o in self.optimizers],
            trainers=[t.state_dict() for t in self.trainers],
            named_generators={k:v.get_state() for k,v in self.generators.items()},
            audited_generators=[dict(seed=g.initial_seed(),device=str(g.device),state=g.get_state()) for g in audit.generators.values()],
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),correction=self.receipt()),path)

    def receipt(self):
        clocks=[]
        for opt in self.optimizers:
            values=[int(s['step'].item()) for s in opt.state.values() if 'step' in s]
            clocks.append(dict(role=opt._game_role,minimum=min(values),maximum=max(values)))
        return dict(mechanism='optimistic_critic',calls=self.calls,roles=self.roles,
                    corrected_tensors=self.corrected_tensors,adam_state_clocks=clocks,
                    extra_forward_evaluations=0,extra_backward_evaluations=0,
                    extra_optimizer_updates=0,memory_device='cuda')
