"""Synchronize both Adam players to slow weights every five G responses."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


class Correction:
    def __init__(self):
        self.calls = 0
        self.roles = {'d': 0, 'g': 0}
        self.optimizers = []
        self.synchronizations = 0

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

    def step(self, opt, original_step, *args, **kwargs):
        role = opt._game_role
        if opt not in self.optimizers:
            self.optimizers.append(opt)
        fresh = [(p, p.detach().clone()) for g in opt.param_groups for p in g['params']
                 if p.grad is not None and 'game_slow_parameter' not in opt.state.get(p, {})]
        result = original_step(opt, *args, **kwargs)
        for p, initial in fresh:
            assert p.device.type == 'cuda'
            opt.state[p]['game_slow_parameter'] = initial
        self.roles[role] += 1
        if role == 'g' and self.roles['g'] % 5 == 0:
            with torch.no_grad():
                for player in self.optimizers:
                    for group in player.param_groups:
                        for p in group['params']:
                            slow = player.state.get(p, {}).get('game_slow_parameter')
                            if slow is not None:
                                slow.lerp_(p, 0.5)
                                p.copy_(slow)
            self.synchronizations += 1
        self.calls += 1
        return result

    def receipt(self):
        return dict(mechanism='joint_lookahead',calls=self.calls,roles=self.roles,
                    synchronizations=self.synchronizations,period=5,slow_fraction=0.5,
                    extra_forward_evaluations=0,extra_backward_evaluations=0,
                    extra_optimizer_updates=0,memory_device='cuda')
