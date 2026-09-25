"""Predict the updated opponent for G; retain one accepted D and G update."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


class Correction:
    def __init__(self):
        self.calls = 0
        self.predictions = 0
        self.restores = 0
        self.pending = []
        self.roles = {'d': 0, 'g': 0}

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
        with patch.object(bridge, 'optimizer_role', role), patch.object(GANTrainer, '__init__', init):
            yield
        assert not self.pending, 'unconsumed critic prediction'

    def step(self, opt, original_step, *args, **kwargs):
        role = opt._game_role
        self.roles[role] += 1
        if role == 'd':
            assert not self.pending, 'D must be followed by its G response'
            before = [(p, p.detach().clone()) for g in opt.param_groups
                      for p in g['params'] if p.grad is not None]
            result = original_step(opt, *args, **kwargs)
            with torch.no_grad():
                for p, old in before:
                    accepted = p.detach().clone()
                    assert p.device.type == 'cuda'
                    p.add_(accepted - old)
                    self.pending.append((p, accepted))
            self.predictions += 1
        else:
            # The host already differentiated G at predicted D. Restore accepted D.
            with torch.no_grad():
                for p, accepted in self.pending:
                    p.copy_(accepted)
            self.restores += bool(self.pending)
            self.pending = []
            result = original_step(opt, *args, **kwargs)
        self.calls += 1
        return result

    def receipt(self):
        return dict(mechanism='predictive_critic',calls=self.calls,roles=self.roles,
                    predictions=self.predictions,restores=self.restores,pending=len(self.pending),
                    extra_forward_evaluations=0,extra_backward_evaluations=0,
                    extra_optimizer_updates=0,virtual_critic_displacements=self.predictions,
                    memory_device='cuda')
