"""Selected full-state snapshots must not change any training state or metric."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from reports.toy100.pr84_critic_refinement import pr84_critic_refinement
from reports.toy100.pr84_critic_refinement_cold import pr84_critic_refinement_cold
from reports.toy100.pr84_critic_refinement_capture import STAGES, _sha, capture_refinement, snapshot


def _run(kind, observed):
    from benchmarks.locked_shared import mode_hold, trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_parity import _state_sha, _without_runtime_timing
    torch.set_num_threads(1)
    task = 'trajectory' if kind == 'trajectory' else 'mode_hold'
    policy = NoisePolicy(.029, 0. if kind == 'warm' else .5, .1,
                         400 if task == 'trajectory' else 1200, output_noise_rng='isolated')
    context = pr84_critic_refinement() if kind == 'warm' else pr84_critic_refinement_cold(task=task)
    with context as (rec, _):
        with (capture_refinement(rec, task=task, steps=[1, 2]) if observed else nullcontext()) as observer:
            if task == 'trajectory':
                with patch.dict(trajectory.PROTOCOL, {'steps': 2}):
                    result = trajectory.train(noise_policy=policy, diagnostics=True)
            else:
                result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
                                                   noise_policy=policy, diagnostics=True)
        state = _state_sha(rec)
    return (result, state, policy.receipt(), _without_runtime_timing(rec.records),
            _sha(snapshot(rec._local))), observer


@pytest.mark.parametrize('kind', ['warm', 'cold', 'trajectory'])
def test_observer_preserves_full_state_noise_metrics_and_records(kind):
    plain, _ = _run(kind, False)
    observed, observer = _run(kind, True)
    assert observed == plain
    assert observer.receipt()['captured_steps'] == [1, 2]
    assert observer.rng_checks == 8
    for number, stages in observer.saved_states.items():
        assert tuple(stages) == STAGES
        before, accepted, refined, after = (stages[name] for name in STAGES)
        assert _sha(accepted['generator']) == _sha(refined['generator']) == _sha(before['generator'])
        assert _sha(accepted['prior']) == _sha(refined['prior']) == _sha(before['prior'])
        assert _sha(accepted['optimizer_d']) == _sha(refined['optimizer_d']) == _sha(after['optimizer_d'])
        assert _sha(accepted['optimizer_g']) == _sha(refined['optimizer_g']) == _sha(before['optimizer_g'])
        assert _sha(refined['critic']) == _sha(after['critic'])
        assert _sha(accepted['critic']) != _sha(refined['critic'])
        assert _sha(accepted['rng']) == _sha(refined['rng'])
        assert all(int(state['step']) == number for state in after['optimizer_d']['state'].values())
        assert all(int(state['step']) == number for state in after['optimizer_g']['state'].values())
        # Captures end before the host's current-update EMA materialization.
        assert all(_sha(row['ema_g']) == _sha(before['ema_g'])
                   and _sha(row['ema_z']) == _sha(before['ema_z']) for row in stages.values())
        if kind == 'trajectory':
            assert before['ema_g'] is before['ema_z'] is before['rng']['data'] is None
        else:
            assert isinstance(before['ema_g'], list) and before['rng']['data'] is not None


def test_final_snapshot_is_weights_only_loadable_and_preserves_complete_noise_history():
    from io import BytesIO
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    _, observer = _run('trajectory', True)
    local = observer.recorder._local
    original = local['noise_policy']
    value = snapshot(local)
    buffer = BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert _sha(loaded) == _sha(value)
    restored = NoisePolicy(.029, .5, .1, 400, output_noise_rng='isolated')
    restored.__dict__.update(loaded['noise_policy'])
    restored.input_stream.set_state(loaded['rng']['input'])
    restored.output_stream.set_state(loaded['rng']['output'])
    assert restored.receipt() == original.receipt()
    # A later set_step and fresh input/output draw must also agree. This
    # exercises cumulative counters and stream states beyond static fields.
    generated = torch.ones(12, 16)
    restored.set_step(2)
    original.set_step(2)
    assert torch.equal(restored.output(generated), original.output(generated))
    assert torch.equal(restored.input(generated), original.input(generated))
    assert restored.receipt() == original.receipt()
