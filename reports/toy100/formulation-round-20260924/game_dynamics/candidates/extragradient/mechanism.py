"""Simultaneous stochastic ExtraAdam, replaying the same oracle at prediction."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def copies(opt):
    return [(p,p.detach().clone()) for group in opt.param_groups for p in group['params']]


def restore(values):
    with torch.no_grad():
        for p,value in values:
            p.copy_(value)


class CaptureGenerators(TorchDispatchMode):
    def __init__(self, owner):
        super().__init__(); self.owner=owner
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if torch.Tag.nondeterministic_seeded in func.tags:
            self.owner.replayed_random_calls+=1
            generator=kwargs.get('generator')
            if generator is not None and id(generator) not in self.owner.generators:
                self.owner.generators[id(generator)]=(generator,generator.get_state())
        return func(*args,**kwargs)


class Correction:
    def __init__(self):
        self.phase=None; self.calls=0; self.outer_steps=0
        self.roles={'predictor':{'d':0,'g':0},'corrector':{'d':0,'g':0},'ordinary':{'d':0,'g':0}}
        self.pending=[]; self.bases={}; self.generators={}
        self.replayed_random_calls=0; self.rng_replays=0

    @contextmanager
    def context(self):
        from benchmarks import learned_lr_evaluation as bridge
        from particlegan.training import GANTrainer
        original_role=bridge.optimizer_role
        original_init=GANTrainer.__init__
        original_trainer_step=GANTrainer.step
        def role(opt, local_variables):
            value=original_role(opt,local_variables);opt._game_role=value;return value
        def init(trainer,*args,**kwargs):
            original_init(trainer,*args,**kwargs)
            trainer.opt_d._game_role='d';trainer.opt_g._game_role='g'
        def trainer_step(trainer,*args,**kwargs):
            completed=trainer.completed_steps
            ema=[(p,p.detach().clone()) for model in (trainer.ema_G,trainer.ema_prior)
                 for p in list(model.parameters())+list(model.buffers())]
            for phase in self.passes():
                result=original_trainer_step(trainer,*args,**kwargs)
                if phase=='predictor':
                    trainer.completed_steps=completed
                    restore(ema)
            return result
        root=Path(__file__).resolve().parent
        # The modified function source is snapshotted and hashed before launch.
        for item in json.loads((root/'transforms.json').read_text()):
            module=importlib.import_module(item['module'])
            source=root/'sources'/item['source']
            module.__dict__['_game_correction']=self
            exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
        with ExitStack() as stack:
            stack.enter_context(patch.object(bridge,'optimizer_role',role))
            stack.enter_context(patch.object(GANTrainer,'__init__',init))
            stack.enter_context(patch.object(GANTrainer,'step',trainer_step))
            yield
        assert self.phase is None and not self.pending

    def passes(self):
        assert self.phase is None and not self.pending
        self.bases={};self.generators={}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        self.phase='predictor'
        with CaptureGenerators(self):
            yield self.phase
        assert not self.pending
        # Keep predicted parameters, discard predictor moments and oracle advancement.
        for opt,base in self.bases.items():
            opt.state.clear()
            opt.state.update(base['state'])
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        for generator,state in self.generators.values():
            generator.set_state(state)
            assert torch.equal(generator.get_state(),state)
        self.rng_replays+=1
        self.phase='corrector'
        yield self.phase
        assert not self.pending
        self.phase=None;self.outer_steps+=1
        self.bases={};self.generators={}

    def step(self,opt,original_step,*args,**kwargs):
        role=opt._game_role
        phase=self.phase or 'ordinary'
        self.calls+=1;self.roles[phase][role]+=1
        if phase=='ordinary':
            return original_step(opt,*args,**kwargs)
        if phase=='predictor':
            assert opt not in self.bases
            self.bases[opt]={'params':copies(opt),'state':{p:deepcopy(s) for p,s in opt.state.items()}}
            result=original_step(opt,*args,**kwargs)
            if role=='d':
                assert not self.pending
                self.pending=copies(opt)
                restore(self.bases[opt]['params'])  # G sees the original D, a simultaneous predictor.
            else:
                restore(self.pending);self.pending=[]
        else:
            assert opt in self.bases
            predicted=copies(opt) if role=='d' else None
            restore(self.bases[opt]['params'])
            result=original_step(opt,*args,**kwargs)
            if role=='d':
                assert not self.pending
                self.pending=copies(opt)
                restore(predicted)  # G differentiates at the same joint prediction as D.
            else:
                restore(self.pending);self.pending=[]
        for state in opt.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in state: assert state[key].device.type=='cuda'
        return result

    def receipt(self):
        return dict(mechanism='extragradient',calls=self.calls,roles=self.roles,
                    outer_steps=self.outer_steps,rng_replays=self.rng_replays,
                    extra_random_evaluations=self.replayed_random_calls,
                    extra_forward_backward_blocks=self.outer_steps,
                    extra_optimizer_preview_calls=sum(self.roles['predictor'].values()),
                    accepted_optimizer_updates=sum(self.roles['corrector'].values())+sum(self.roles['ordinary'].values()),
                    model_gradient_work_multiplier=2,pending=len(self.pending),memory_device='cuda')
