"""Passive selected-update capture for the warm or cold refinement adapter.

No training driver or update/controller change lives here. Wrap an existing
recorder, select update numbers in advance, and serialize ``saved_states``
afterward. Captures preserve the prior CaptureRecorder._state keys and add
the full fixed-scale noise-policy state; trajectory has None for its absent
EMA and data stream. All four stages precede the current outer update's
EMA/checkpoint. Only pre_step is a complete outer-loop restart point;
intermediate states are materialization/field diagnostics.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.toy100.warm_equilibrium_probe import _feed_hash


STAGES = ('pre_step', 'post_accepted_d', 'post_refined_d', 'post_bounded_g')


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return deepcopy(value)


def _rng(local):
    policy = local['noise_policy']
    stream = local.get('stream')
    return dict(torch=torch.get_rng_state().clone(),
                data=None if stream is None else stream.get_state().clone(),
                input=policy.input_stream.get_state().clone(),
                output=None if policy.output_stream is None else policy.output_stream.get_state().clone())


def _state(local, opt_d, opt_g):
    """Same tensor/state keys as the archived stationary CaptureRecorder."""
    policy = local['noise_policy']
    return _clone(dict(generator=local['generator'].state_dict(),
                       critic=local['critic'].state_dict(), prior=local['prior'].state_dict(),
                       optimizer_d=opt_d.state_dict(), optimizer_g=opt_g.state_dict(),
                       ema_g=local.get('ema_g'), ema_z=local.get('ema_z'), rng=_rng(local),
                       noise=dict(step_calls=policy._step_calls,
                                  input_sigma=policy.input_sigma, output_sigma=policy.output_sigma)))


def snapshot(local):
    """Copy frozen-host learning state, including cumulative noise receipts.

    ``rng`` owns all generator states; ``noise_policy`` holds every other
    attribute of the fixed-scale NoisePolicy, including counters, traces and
    current context flags. The host/optimizer recipe and adapter's intra-step
    scratch history live in the source-bound declaration, not this snapshot.
    Save after an outer update or at pre_step to restart learning; mid-update
    states do not by themselves restart the recorder's three-phase program.
    """
    policy = local['noise_policy']
    if policy.output_scale is not None:
        raise ValueError('full refinement snapshot supports fixed output scale only')
    value = _state(local, local['opt_d'], local['opt_g'])
    value['noise_policy'] = _clone({key: item for key, item in vars(policy).items()
                                    if key not in ('input_stream', 'output_stream')})
    value['snapshot_scope'] = dict(
        version=1, fixed_output_scale=True,
        noise_policy='all attributes except generator objects; their states are in rng',
        host_loop_step=local['step'],
        adapter_scratch='not included; restart only at outer-loop boundary',
        external_host_configuration='required from source-bound declaration')
    return value


def _sha(value):
    digest = hashlib.sha256()
    _feed_hash(digest, value)
    return digest.hexdigest()


class RefinementCapture:
    def __init__(self, recorder, *, task, steps):
        if task not in ('mode_hold', 'trajectory'):
            raise ValueError('unknown refinement capture host')
        if any(type(step) is not int or step < 1 for step in steps):
            raise ValueError('capture steps must be positive update numbers')
        self.recorder, self.task, self.steps = recorder, task, frozenset(steps)
        self.saved_states, self.hashes = {}, {}
        self.current_step = None
        self._selected = False
        self._original_phases = recorder.phases
        self._original_arm = recorder._arm_smoothed_critic
        self.rng_checks = 0

    @torch.no_grad()
    def _capture(self, name, local, opt_d, opt_g):
        if name in self.saved_states[self.current_step]:
            raise RuntimeError('duplicate selected refinement stage')
        before = _rng(local)
        value = snapshot(local)
        if _sha(before) != _sha(_rng(local)):
            raise RuntimeError('passive tensor snapshot changed a training RNG stream')
        self.saved_states[self.current_step][name] = value
        self.hashes[self.current_step][name] = _sha(value)
        self.rng_checks += 1

    def phases(self, step, opt_d, opt_g, local):
        self.current_step = step + (self.task == 'mode_hold')
        self._selected = self.current_step in self.steps
        if self._selected:
            if (not self.recorder.enabled or step < self.recorder.start_step
                    or not self.recorder.refinement):
                raise ValueError('selected capture update must have refinement active')
            if self.current_step in self.saved_states:
                raise RuntimeError('selected update repeated')
            self.saved_states[self.current_step], self.hashes[self.current_step] = {}, {}
            self._capture('pre_step', local, opt_d, opt_g)
        try:
            for phase in self._original_phases(step, opt_d, opt_g, local):
                yield phase
                if self._selected and phase == 2:
                    self._capture('post_bounded_g', local, opt_d, opt_g)
            if self._selected and tuple(self.saved_states[self.current_step]) != STAGES:
                raise RuntimeError('refinement stage ordering changed')
        finally:
            self._selected = False

    def arm(self):
        observe = self._selected and self.recorder.phase == 1
        if observe:
            self._capture('post_accepted_d', self.recorder._local, *self.recorder.optimizers)
        result = self._original_arm()
        if observe:
            self._capture('post_refined_d', self.recorder._local, *self.recorder.optimizers)
        return result

    def receipt(self):
        return dict(task=self.task, requested_steps=sorted(self.steps),
                    captured_steps=sorted(self.saved_states), stage_order=list(STAGES),
                    stage_scope='all before current-update EMA/checkpoint; only pre_step restarts an outer loop',
                    operations='detached tensor copies and hashing only; no forwards or quality reads',
                    rng_checks=self.rng_checks, state_sha256=self.hashes,
                    observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@contextmanager
def capture_refinement(recorder, *, task='mode_hold', steps=()):
    """Wrap a live recorder; save ``capture.saved_states`` after the run."""
    observer = RefinementCapture(recorder, task=task, steps=tuple(steps))
    with patch.object(recorder, 'phases', observer.phases), \
         patch.object(recorder, '_arm_smoothed_critic', observer.arm):
        yield observer
