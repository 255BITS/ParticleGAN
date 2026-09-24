"""Exact bounded replay from a snapshot taken after the first noise clock.

The saved state is a pre-gradient, pre-D outer boundary AFTER set_step.
Restore all state before the first replayed body and skip exactly that one
set_step call. Later noise clocks run normally. This is separate from the
post-update resumer; no inverse or deletion of cumulative noise history is
used. Completion stops at the last checkpoint, before its optional extra
diagnostics/final evaluation. The original host budget remains unchanged.
"""

import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.pr84_critic_refinement_capture import _clone, _sha, snapshot
from reports.toy100.pr84_critic_refinement_resume import (
    ModeHoldResume, NOISE_HORIZON, RATES, _moment_steps, clean_support, resumed_source,
)


class LocalContinuationComplete(Exception):
    """The declared final update completed, including EMA and live check."""


@torch.no_grad()
def restore_after_set_step(local, saved, completed_steps):
    if (saved['snapshot_scope']['version'] != 1
            or saved['snapshot_scope']['host_loop_step'] != completed_steps
            or saved['noise']['step_calls'] != completed_steps + 1
            or saved['noise_policy']['_step_calls'] != completed_steps + 1):
        raise ValueError('expected a complete pre-gradient snapshot after set_step')
    policy = local['noise_policy']
    policy_state = saved['noise_policy']
    keys = set(vars(policy)) - {'input_stream', 'output_stream'}
    if keys != set(policy_state) or policy.output_scale is not None or policy_state['output_scale'] is not None:
        raise ValueError('fixed-scale NoisePolicy schema changed')
    for key in ('total_steps', 'input_std', 'input_anneal_end', 'output_std', 'output_noise_warmup',
                'output_noise_rng', 'output_noise_learnable', 'seed', 'generator_base_parameters'):
        if vars(policy)[key] != policy_state[key]:
            raise ValueError(f'replayed noise configuration changed: {key}')
    if policy.total_steps != NOISE_HORIZON:
        raise ValueError('replay must keep the original1200 noise horizon')
    if saved['rng']['data'] is None or saved['ema_g'] is None or saved['ema_z'] is None:
        raise ValueError('mode_hold replay lacks data stream or EMA state')
    for name in ('generator', 'critic', 'prior'):
        local[name].load_state_dict(saved[name])
    for role in ('d', 'g'):
        optimizer = local['opt_' + role]
        optimizer.load_state_dict(deepcopy(saved['optimizer_' + role]))
        if any(step != completed_steps for step in _moment_steps(optimizer)):
            raise ValueError('replay Adam steps do not equal completed outer updates')
        if tuple(group['lr'] for group in optimizer.param_groups) != RATES[role]:
            raise ValueError('replay rates differ from the declared constant recipe')
    if len(local['ema_g']) != len(saved['ema_g']):
        raise ValueError('EMA parameter structure changed')
    for target, value in zip(local['ema_g'], saved['ema_g']):
        target.copy_(value)
    local['ema_z'].copy_(saved['ema_z'])
    vars(policy).update(_clone(policy_state))
    torch.set_rng_state(saved['rng']['torch'])
    local['stream'].set_state(saved['rng']['data'])
    policy.input_stream.set_state(saved['rng']['input'])
    if (policy.output_stream is None) != (saved['rng']['output'] is None):
        raise ValueError('output-noise stream presence changed')
    if policy.output_stream is not None:
        policy.output_stream.set_state(saved['rng']['output'])
    if _sha(snapshot(local)) != _sha(saved):
        raise RuntimeError('after-set-step full model/Adam/EMA/noise/RNG restore was not exact')


def after_set_step_source(source, completed_steps, target_steps):
    tree, _ = resumed_source(source, completed_steps, target_steps)
    original = deepcopy(tree)
    loop = next(node for node in tree.body[0].body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'step')
    matches = [node for node in loop.body if isinstance(node, ast.If)
               and len(node.body) == 1
               and ast.unparse(node.body[0]) == 'noise_policy.set_step(step)']
    if len(matches) != 1 or ast.unparse(matches[0].test) != 'noise_policy is not None':
        raise RuntimeError('frozen first noise-clock statement changed')
    noise_clock = matches[0]
    noise_clock.test = ast.BoolOp(op=ast.And(), values=[noise_clock.test,
        ast.parse(f'step != {completed_steps}', mode='eval').body])
    ast.fix_missing_locations(tree)
    inverse = deepcopy(tree)
    inverse_loop = next(node for node in inverse.body[0].body if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == 'step')
    inverse_clock = next(node for node in inverse_loop.body if isinstance(node, ast.If)
                         and len(node.body) == 1
                         and ast.unparse(node.body[0]) == 'noise_policy.set_step(step)')
    inverse_clock.test = deepcopy(noise_clock.test.values[0])
    if ast.dump(inverse, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('after-clock transformation changed more than the first set_step guard')
    return tree, ast.unparse(tree) + '\n'


class AfterSetStepResume(ModeHoldResume):
    def __init__(self, *args, host_steps, **kwargs):
        super().__init__(*args, fail_fast=False, **kwargs)
        self.host_steps = host_steps

    def before_step(self, step, local):
        if not self.restored:
            self.local = local
            if step != self.completed_steps or local['recipe'].steps != self.host_steps:
                raise RuntimeError('replay loop or unchanged host budget differs')
            restore_after_set_step(local, self.saved, self.completed_steps)
            self.restored = True
            self.before_clock = snapshot(local)  # First boundary is explicitly AFTER the clock.
            self._before_support = clean_support(local)
        else:
            super().before_step(step, local)

    def checkpoint(self, step, measure):
        super().checkpoint(step, measure)
        if step == self.target_steps:
            raise LocalContinuationComplete()

    def receipt(self):
        value = super().receipt()
        value.update(restored_before_set_step=False, restored_after_set_step=self.restored,
                     first_set_step_skipped=True, host_budget_unchanged=self.host_steps,
                     final_stage='after final requested checkpoint; before extra diagnostics/final evaluation',
                     quality_is_diagnostic=True, cold_acquisition_pass_claim=False,
                     shared_gate_eligible=False)
        return value


@contextmanager
def resume_after_set_step(recorder, generated_source, saved, *, completed_steps,
                          target_steps, host_steps=1200, log=None):
    tree, source = after_set_step_source(generated_source, completed_steps, target_steps)
    state = AfterSetStepResume(recorder, saved, completed_steps, target_steps,
                               host_steps=host_steps, log=log)
    state.source, state.source_sha256 = source, hashlib.sha256(source.encode()).hexdigest()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(mode_hold.__dict__, {'_refinement_resume': state}))
        namespace = {}
        exec(compile(tree, '<refinement-after-set-step-replay>', 'exec'), mode_hold.__dict__, namespace)
        stack.enter_context(patch.object(mode_hold, 'train_mode_hold', namespace['train_mode_hold']))
        stack.enter_context(patch.object(mode_hold, 'checkpoint', state.checkpoint))
        stack.enter_context(patch.object(recorder, 'step', state.optimizer_step))
        try:
            yield state
        finally:
            import sys
            if sys.exc_info()[0] in (None, LocalContinuationComplete):
                state.finalize()
