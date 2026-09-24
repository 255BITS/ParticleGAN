"""Exact complete-state replay from the actual after-set_step capture stage."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from reports.toy100.pr84_critic_refinement_after_step import (
    LocalContinuationComplete, resume_after_set_step,
)
from reports.toy100.pr84_critic_refinement_capture import _sha, capture_refinement, snapshot
from reports.toy100.pr84_critic_refinement_finite import pr84_critic_refinement_finite
from reports.toy100.pr84_smoothed_parity import _without_runtime_timing


def _run(*, saved=None, isolated=False):
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
    config['output_noise_warmup'] = 0
    if isolated:
        config['output_noise_rng'] = 'isolated'
    recipe, noise, _ = declared_recipe(prepared_config(config, 'constant'))
    policy = _noise_policy(noise, 1200)
    points = []
    continuation = None
    with ExitStack() as stack:
        stack.enter_context(optimizer_defaults(recipe, []))
        control = evaluate.FixedControl(vector_tasks.fixed_policy('cosine'), 1200)
        stack.enter_context(bridge.control_host_schedules(control))
        rec, source = stack.enter_context(pr84_critic_refinement_finite())
        capture = stack.enter_context(capture_refinement(rec, steps=(2,)))
        def observe(step, measure):
            points.append(dict(step=step, **measure()))
            if step == 3 and saved is None:
                raise LocalContinuationComplete()
        stack.enter_context(patch.object(mode_hold, 'checkpoint', observe))
        if saved is not None:
            continuation = stack.enter_context(resume_after_set_step(rec, source, saved,
                completed_steps=1, target_steps=3, host_steps=3))
        common = candidate(recipe)
        try:
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3, particle_l2=0.,
                                          vicreg_weight=recipe.prior_reg),
                gan_factory=common.make_loss, cap_factory=common.make_penalty,
                noise_policy=policy, diagnostics=True)
        except LocalContinuationComplete:
            state = snapshot(rec._local)
    return dict(state=state, points=points, recorder=rec, capture=capture, continuation=continuation)


@pytest.mark.parametrize('isolated', [False, True])
def test_after_set_step_split_exact_all_models_moments_ema_rng_noise_histories(isolated):
    full = _run(isolated=isolated)
    saved = full['capture'].saved_states[2]['pre_step']
    assert saved['noise_policy']['_step_calls'] == 2
    assert len(saved['noise_policy']['_effective_step_trace']) == 2
    resumed = _run(saved=saved, isolated=isolated)
    assert _sha(resumed['state']) == _sha(full['state'])
    assert resumed['points'] == full['points'][1:]
    assert _sha(resumed['capture'].saved_states[2]['pre_step']) == _sha(saved)
    records = deepcopy(resumed['recorder'].records)
    for row in records:
        row['outer_step'] += 1
        row['critic_refinement']['outer_step'] += 1
    assert _without_runtime_timing(records) == _without_runtime_timing(full['recorder'].records[1:])
    receipt = resumed['continuation'].receipt()
    assert receipt['restored_after_set_step'] and not receipt['restored_before_set_step']
    assert receipt['first_set_step_skipped'] and receipt['completed']
    assert receipt['actual_adam_updates'] == dict(d=2, g=2)
    assert receipt['optimizer_callbacks'] == dict(d=6, g=6)
    assert resumed['state']['noise_policy']['_step_calls'] == 3
    assert len(resumed['state']['noise_policy']['_effective_step_trace']) == 3
    assert not receipt['cold_acquisition_pass_claim']


def test_post_update_stage_is_not_accepted_as_after_set_step():
    full = _run()
    with pytest.raises(ValueError, match='pre-gradient snapshot after set_step'):
        _run(saved=full['state'])
