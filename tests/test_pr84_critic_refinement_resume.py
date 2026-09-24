"""Complete noise-clock/Adam/EMA continuation, including prefix eval history."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from reports.toy100.pr84_critic_refinement_capture import _sha, snapshot
from reports.toy100.pr84_critic_refinement_cold import pr84_critic_refinement_cold
from reports.toy100.pr84_critic_refinement_resume import FirstHoldFailure, resume_mode_hold
from reports.toy100.pr84_smoothed_parity import _without_runtime_timing


class StopAtBoundary(Exception):
    pass


def run(target, *, saved=None, split=None, isolated=False, fail_fast=False):
    from benchmarks import learned_lr_evaluation as bridge
    from benchmarks.locked_shared import mode_hold
    from benchmarks.smart_descent import evaluate
    from benchmarks.toy100.continuous_probe import _noise_policy, prepared_config
    from benchmarks.transfer_suite import vector_tasks
    from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/toy100/constraints_simple_regularization.json').read_text())
    if isolated:
        config['output_noise_rng'] = 'isolated'
    recipe, noise, _ = declared_recipe(prepared_config(config, 'constant'))
    policy = _noise_policy(noise, 1200)
    points = []
    state = boundary = continuation = None
    with ExitStack() as stack:
        stack.enter_context(optimizer_defaults(recipe, []))
        control = evaluate.FixedControl(vector_tasks.fixed_policy('cosine'), 1200)
        stack.enter_context(bridge.control_host_schedules(control))
        rec, source = stack.enter_context(pr84_critic_refinement_cold())
        def observe(step, measure):
            nonlocal boundary
            points.append(dict(step=step, **measure()))
            if step == split:
                boundary = snapshot(rec._local)
                raise StopAtBoundary()
        stack.enter_context(patch.object(mode_hold, 'checkpoint', observe))
        if saved is not None:
            continuation = stack.enter_context(resume_mode_hold(
                rec, source, saved, completed_steps=saved['noise']['step_calls'],
                target_steps=target, fail_fast=fail_fast))
        common = candidate(recipe)
        try:
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=target, particle_l2=0.,
                                          vicreg_weight=recipe.prior_reg),
                gan_factory=common.make_loss, cap_factory=common.make_penalty,
                noise_policy=policy, diagnostics=True)
        except StopAtBoundary:
            assert split is not None and saved is None
        except FirstHoldFailure:
            assert fail_fast and continuation.failure is not None
        state = boundary if boundary is not None else snapshot(rec._local)
    return dict(state=state, points=points, recorder=rec, continuation=continuation)


def suffix_records(recorder, offset):
    records = deepcopy(recorder.records)
    for row in records:
        row['outer_step'] += offset
        row['critic_refinement']['outer_step'] += offset
    return _without_runtime_timing(records)


@pytest.mark.parametrize('isolated', [False, True])
def test_checkpoint_split_resumes_bitwise_full_state_and_complete_noise_history(isolated):
    full = run(3, isolated=isolated)
    prefix = run(3, split=1, isolated=isolated)
    resumed = run(3, saved=prefix['state'], isolated=isolated)
    assert _sha(resumed['state']) == _sha(full['state'])
    assert resumed['points'] == full['points'][1:]
    assert suffix_records(resumed['recorder'], 1) == _without_runtime_timing(full['recorder'].records[1:])
    receipt = resumed['continuation'].receipt()
    assert receipt['restored_before_set_step'] and receipt['completed']
    assert receipt['actual_adam_updates'] == dict(d=2, g=2)
    assert receipt['optimizer_callbacks'] == dict(d=6, g=6)
    assert receipt['noise_horizon'] == 1200
    assert resumed['state']['noise_policy']['_step_calls'] == 3
    assert len(resumed['state']['noise_policy']['_effective_step_trace']) == 3
    assert all(row['clean_output_rms'] > 0 and row['clean_output_max'] >= row['clean_output_rms']
               for row in receipt['accepted_movement'])
    assert all(set(row['gradient_metric']) == {'g', 'prior'} for row in receipt['accepted_movement'])


def _without_eval_history(state):
    value = deepcopy(state)
    policy = value['noise_policy']
    policy['_counts'] = {key: item for key, item in policy['_counts'].items() if '_eval_' not in key}
    policy['_output_eval_state_pairs'] = []
    return value


def test_after_final_evaluation_state_retains_extra_eval_history_but_exact_learning_state():
    full = run(3, isolated=True)
    prefix = run(1, isolated=True)
    resumed = run(3, saved=prefix['state'], isolated=True)
    assert _sha(resumed['state']) != _sha(full['state'])
    assert _sha(_without_eval_history(resumed['state'])) == _sha(_without_eval_history(full['state']))
    assert resumed['points'] == full['points'][1:]
    assert suffix_records(resumed['recorder'], 1) == _without_runtime_timing(full['recorder'].records[1:])
    before = prefix['state']['noise_policy']['_counts']
    after = resumed['state']['noise_policy']['_counts']
    assert all(after[key] >= value for key, value in before.items())
    assert after['output_eval_calls'] > full['state']['noise_policy']['_counts']['output_eval_calls']


def test_failure_capture_stops_after_completed_update_with_constant_rates_and_no_extra_moments():
    prefix = run(3, split=1)
    resumed = run(3, saved=prefix['state'], fail_fast=True)
    continuation = resumed['continuation']
    assert continuation.receipt()['first_failure_step'] == 2  # Initial cold model is not acquired.
    assert not continuation.receipt()['completed']
    assert continuation.receipt()['actual_adam_updates'] == dict(d=1, g=1)
    failure = continuation.failure
    assert failure['before_set_step']['noise_policy']['_step_calls'] == 1
    assert failure['after_checkpoint']['noise_policy']['_step_calls'] == 2
    assert _sha(continuation.final_state) == _sha(failure['after_checkpoint'])


def test_wrong_noise_horizon_is_rejected_before_resumed_training():
    prefix = run(2, split=1)
    bad = deepcopy(prefix['state'])
    bad['noise_policy']['total_steps'] = 2400
    with pytest.raises(ValueError, match='noise configuration changed: total_steps'):
        run(2, saved=bad)
