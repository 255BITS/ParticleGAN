"""Exact mode_hold continuation from a complete post-update snapshot.

Only the outer loop bounds and one pre-set_step restore/observer call are
inserted into the source-bound host. The noise horizon stays 1200. Dense live
checks are passive; optional fail-fast termination saves the before-clock and
after-checkpoint states without changing any accepted update. No perturbation
or training driver is implemented here.
"""

import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.pr84_critic_refinement_capture import _clone, _rng, _sha, snapshot


NOISE_HORIZON = 1200
RATES = {'d': (.00425,), 'g': (.00425, .0085)}


class FirstHoldFailure(RuntimeError):
    """A completed, measured update failed the unchanged hold threshold."""


def _moment_steps(optimizer):
    return [int(optimizer.state[p]['step']) for group in optimizer.param_groups
            for p in group['params']]


@torch.no_grad()
def restore_snapshot(local, saved, *, completed_steps):
    """Restore before the next set_step; preserve all saved noise histories.

    The caller must source-bind that this is an after-complete-update state,
    including current EMA, rather than an intermediate pre-EMA stage capture.
    """
    if (saved['snapshot_scope']['version'] != 1
            or saved['noise']['step_calls'] != completed_steps
            or saved['snapshot_scope']['host_loop_step'] not in (completed_steps - 1, completed_steps)
            or saved['noise_policy']['_step_calls'] != completed_steps):
        raise ValueError('snapshot is not the declared mode_hold post-update boundary')
    policy = local['noise_policy']
    configured = vars(policy)
    policy_state = saved['noise_policy']
    keys = set(configured) - {'input_stream', 'output_stream'}
    if keys != set(policy_state) or policy.output_scale is not None or policy_state['output_scale'] is not None:
        raise ValueError('fixed-scale NoisePolicy schema changed')
    for key in ('total_steps', 'input_std', 'input_anneal_end', 'output_std', 'output_noise_warmup',
                'output_noise_rng', 'output_noise_learnable', 'seed', 'generator_base_parameters'):
        if configured[key] != policy_state[key]:
            raise ValueError(f'resumed noise configuration changed: {key}')
    if policy.total_steps != NOISE_HORIZON:
        raise ValueError('resumption must keep the original1200 noise horizon')
    if saved['rng']['data'] is None or saved['ema_g'] is None or saved['ema_z'] is None:
        raise ValueError('mode_hold snapshot lacks data stream or EMA state')
    for name in ('generator', 'critic', 'prior'):
        local[name].load_state_dict(saved[name])
    for role in ('d', 'g'):
        opt = local['opt_' + role]
        opt.load_state_dict(deepcopy(saved['optimizer_' + role]))
        if _moment_steps(opt) != [completed_steps] * sum(len(g['params']) for g in opt.param_groups):
            raise ValueError('snapshot contains inconsistent Adam moment steps')
        if tuple(group['lr'] for group in opt.param_groups) != RATES[role]:
            raise ValueError('snapshot does not have declared constant G/D/prior rates')
    if len(local['ema_g']) != len(saved['ema_g']):
        raise ValueError('EMA parameter structure changed')
    for target, value in zip(local['ema_g'], saved['ema_g']):
        target.copy_(value)
    local['ema_z'].copy_(saved['ema_z'])
    configured.update(_clone(policy_state))
    torch.set_rng_state(saved['rng']['torch'])
    local['stream'].set_state(saved['rng']['data'])
    policy.input_stream.set_state(saved['rng']['input'])
    if (policy.output_stream is None) != (saved['rng']['output'] is None):
        raise ValueError('output-noise stream presence changed')
    if policy.output_stream is not None:
        policy.output_stream.set_state(saved['rng']['output'])
    actual = snapshot(local)
    # The new Python loop is already positioned at the next integer step;
    # this bookkeeping field is the sole expected snapshot metadata change.
    actual['snapshot_scope']['host_loop_step'] = saved['snapshot_scope']['host_loop_step']
    if _sha(actual) != _sha(saved):
        raise RuntimeError('complete model/optimizer/EMA/RNG/noise restoration was not exact')


def resumed_source(source, completed_steps, target_steps):
    if (type(completed_steps) is not int or type(target_steps) is not int
            or completed_steps < 1 or target_steps <= completed_steps):
        raise ValueError('continuation needs positive completed steps and a larger target')
    original = ast.parse(source)
    tree = deepcopy(original)
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == 'step']
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(recipe.steps)':
        raise RuntimeError('frozen mode_hold outer loop changed')
    loop = loops[0]
    loop.iter = ast.parse(f'range({completed_steps}, {target_steps})', mode='eval').body
    hook = ast.parse('_refinement_resume.before_step(step, locals())').body[0]
    loop.body.insert(0, hook)
    ast.fix_missing_locations(tree)
    inverse = deepcopy(tree)
    restored = next(node for node in inverse.body[0].body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'step')
    if ast.dump(restored.body.pop(0), include_attributes=False) != ast.dump(hook, include_attributes=False):
        raise RuntimeError('unrecognized pre-step insertion')
    restored.iter = deepcopy(next(node for node in original.body[0].body if isinstance(node, ast.For)
                                 and isinstance(node.target, ast.Name) and node.target.id == 'step').iter)
    if ast.dump(inverse, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('continuation changed more than loop bounds and pre-step observer')
    return tree, ast.unparse(tree) + '\n'


@torch.no_grad()
def clean_support(local):
    """Measure clean functional movement without a noise draw or buffer write."""
    before = _sha(_rng(local))
    generator = getattr(local['generator'], 'model', local['generator'])
    buffers = [(buffer, buffer.detach().clone()) for model in (generator, local['prior'])
               for buffer in model.buffers()]
    points = generator(local['prior'].z).detach().clone()
    if _sha(_rng(local)) != before or any(not torch.equal(a, b) for a, b in buffers):
        raise RuntimeError('clean movement observer changed RNG or module buffers')
    if not torch.isfinite(points).all():
        raise FloatingPointError('nonfinite clean generated support')
    return points


@torch.no_grad()
def gradient_metric_receipt(recorder):
    """Raw used G field and post-base Adam denominator, split by ownership."""
    result = {}
    offset = 0
    for group in recorder.optimizers[1].param_groups:
        role = 'prior' if group.get('_comparison_prior') else 'g'
        gradients = recorder.gg0[offset:offset+len(group['params'])]
        offset += len(group['params'])
        denominator = []
        for parameter in group['params']:
            state = recorder.optimizers[1].state[parameter]
            value = (state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
            denominator.append(value.detach().double().flatten())
        denom = torch.cat(denominator)
        raw = torch.cat([value.double().flatten() for value in gradients])
        metric = group['lr']/denom
        result[role] = dict(raw_gradient_l2=float(raw.norm()), raw_gradient_rms=float(raw.square().mean().sqrt()),
                            denominator_min=float(denom.min()), denominator_median=float(denom.median()),
                            denominator_max=float(denom.max()), metric_min=float(metric.min()),
                            metric_median=float(metric.median()), metric_max=float(metric.max()),
                            saved_metric_gradient_norm=float((metric*raw.square()).sum().sqrt()))
    return result


class ModeHoldResume:
    def __init__(self, recorder, saved, completed_steps, target_steps, *, fail_fast, log):
        self.recorder, self.saved = recorder, saved
        self.completed_steps, self.target_steps = completed_steps, target_steps
        self.fail_fast, self.log = fail_fast, log
        self.restored = False
        self.points, self.rates = [], {'d': [], 'g': []}
        self.movement = []
        self.actual_adam_steps = {'d': 0, 'g': 0}
        self.failure = self.before_clock = self.final_state = None
        self.local = None
        self._ordinary_step = recorder.step
        self._ordinary_checkpoint = mode_hold.checkpoint

    def before_step(self, step, local):
        self.local = local
        if not self.restored:
            if step != self.completed_steps or local['recipe'].steps != self.target_steps:
                raise RuntimeError('resume loop or declared final host step changed')
            restore_snapshot(local, self.saved, completed_steps=self.completed_steps)
            self.restored = True
        if local['noise_policy']._step_calls != step:
            raise RuntimeError('restoration or resumption occurred after set_step')
        self.before_clock = snapshot(local)
        self._before_support = clean_support(local)

    def optimizer_step(self, optimizer, ordinary_step, closure=None):
        role = 'd' if optimizer is self.recorder.optimizers[0] else 'g'
        values = tuple(group['lr'] for group in optimizer.param_groups)
        if values != RATES[role]:
            raise RuntimeError(f'{role} resumed nominal rates changed')
        self.rates[role].append(values)
        before = _moment_steps(optimizer)
        result = self._ordinary_step(optimizer, ordinary_step, closure)
        after = _moment_steps(optimizer)
        increment = 1 if (role == 'd' and self.recorder.phase == 0) or (
            role == 'g' and self.recorder.phase == 1) else 0
        if any(b - a != increment for a, b in zip(before, after)):
            raise RuntimeError('Adam moments changed outside their one declared phase')
        self.actual_adam_steps[role] += increment
        return result

    def checkpoint(self, step, measure):
        before = _sha(_rng(self.local))
        with torch.random.fork_rng(devices=[]):
            point = dict(step=step, **measure())
        if _sha(_rng(self.local)) != before:
            raise RuntimeError('dense hold evaluation advanced training random streams')
        if self.local['noise_policy']._step_calls != step:
            raise RuntimeError('noise clock did not advance once for the resumed update')
        self.points.append(point)
        self._ordinary_checkpoint(step, lambda: {key: value for key, value in point.items() if key != 'step'})
        displacement = clean_support(self.local) - self._before_support
        self.movement.append(dict(step=step,
                                  clean_output_rms=float(displacement.square().sum(-1).mean().sqrt()),
                                  clean_output_max=float(displacement.norm(dim=-1).max()),
                                  g_factor=self.recorder.records[-1]['g']['factor'],
                                  gradient_metric=gradient_metric_receipt(self.recorder)))
        passing = point['modes'] == 8 and point['hq'] >= .9
        if self.log is not None:
            self.log(dict(event='RESUME_CHECK', **point, passed=passing))
        if not passing and self.failure is None:
            self.failure = dict(step=step, point=point,
                                before_set_step=self.before_clock,
                                after_checkpoint=snapshot(self.local))
        if not passing and self.fail_fast:
            raise FirstHoldFailure(f'fixed-target hold failed at update {step}')

    def finalize(self):
        if not self.restored:
            raise RuntimeError('continuation never restored its snapshot')
        self.final_state = snapshot(self.local)
        updates = len(self.points)
        for role, optimizer in zip(('d', 'g'), self.recorder.optimizers):
            if len(self.rates[role]) != 3 * updates or self.actual_adam_steps[role] != updates:
                raise RuntimeError('incomplete resumed optimizer/update accounting')
            if any(step != self.completed_steps + updates for step in _moment_steps(optimizer)):
                raise RuntimeError('resumed moment count differs from absolute update count')

    def receipt(self):
        return dict(completed_before_resume=self.completed_steps, target_steps=self.target_steps,
                    updates=len(self.points), checkpoints=self.points,
                    restored_snapshot_sha256=_sha(self.saved), restored_before_set_step=self.restored,
                    final_snapshot_sha256=None if self.final_state is None else _sha(self.final_state),
                    source_sha256=self.source_sha256, noise_horizon=NOISE_HORIZON,
                    actual_adam_updates=self.actual_adam_steps,
                    optimizer_callbacks={role: len(values) for role, values in self.rates.items()},
                    accepted_movement=self.movement,
                    movement_observer='clean G(prior.z), same RNG and immutable buffers verified; diagnostic only',
                    nominal_rates=RATES, first_failure_step=None if self.failure is None else self.failure['step'],
                    pass_all=bool(self.points) and self.failure is None,
                    completed=self.completed_steps + len(self.points) == self.target_steps,
                    evaluation_history='preserved in full, including any earlier final-evaluation counters')


@contextmanager
def resume_mode_hold(recorder, generated_source, saved, *, completed_steps, target_steps,
                     fail_fast=True, log=None):
    """Run the existing host inside this context; catch FirstHoldFailure outside.

    Set its ModeHoldRecipe.steps to target_steps. Instantiate NoisePolicy with
    its original1200 horizon and original settings. The caller retains the
    same constant-recipe optimizer/schedule context used by cold acquisition.
    Source/gate checks on the input artifact remain the caller's responsibility.
    """
    tree, source = resumed_source(generated_source, completed_steps, target_steps)
    state = ModeHoldResume(recorder, saved, completed_steps, target_steps,
                           fail_fast=fail_fast, log=log)
    state.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    state.source = source
    with ExitStack() as stack:
        stack.enter_context(patch.dict(mode_hold.__dict__, {'_refinement_resume': state}))
        namespace = {}
        exec(compile(tree, '<exact-refinement-resume>', 'exec'), mode_hold.__dict__, namespace)
        stack.enter_context(patch.object(mode_hold, 'train_mode_hold', namespace['train_mode_hold']))
        stack.enter_context(patch.object(mode_hold, 'checkpoint', state.checkpoint))
        stack.enter_context(patch.object(recorder, 'step', state.optimizer_step))
        try:
            yield state
        finally:
            # Validation on a declared hold failure remains meaningful; an
            # unrelated partial-update exception must retain its original error.
            import sys
            if sys.exc_info()[0] in (None, FirstHoldFailure):
                state.finalize()
