"""Keep shared-default search free of per-example recipe overrides."""
from pathlib import Path

from benchmarks.locked_shared.recorded_recipes import GAN_V1, GAN_V2
import pytest

from particlegan import get_recipe
from benchmarks.transfer_suite.compare_defaults import read
from benchmarks.transfer_suite.shared_default_search import episode, prepare


@pytest.mark.parametrize('declaration', [
    dict(candidates=[dict(name='bad', overrides={}, task_overrides={'two_pole': {'lr': .1}})]),
    dict(candidates=[dict(name='bad', overrides={'batch_size': 1})]),
    dict(candidates=[dict(name='bad', overrides={})], tasks=['two_pole', 'two_pole']),
])
def test_rejects_task_overrides_or_changed_test_resources(declaration):
    with pytest.raises(ValueError):
        prepare(declaration)


def test_shared_recipe_is_resolved_once_for_all_nineteen_jobs():
    jobs, recipes = prepare(dict(candidates=[dict(name='equal_lr', overrides=dict(lr=.00425, d_lr_mult=1., prior_lr_mult=1.))]))
    assert len(jobs) == 19 and len(recipes) == 1
    assert recipes[0][1].lr == .00425


def test_new_runner_reproduces_existing_default_control():
    jobs, _ = prepare(dict(candidates=[dict(name='control', overrides={})], tasks=['two_pole']))
    result = episode(jobs[0], GAN_V2.replace(name='control'))['result']
    expected = read(Path(__file__).with_name('fixtures')/'shared_default_two_pole.json')['result']
    clean = lambda curve: [{k: v for k, v in p.items() if k != 'seconds'} for p in curve]
    # Archived float32 training metrics need not be bit-identical across CPU
    # kernels. CI observed a 3.6e-7 absolute grad_med delta on the same inputs.
    # Allow only small numerical drift; schedule steps and actions stay exact.
    assert result['live'] == pytest.approx(expected['live'], rel=1e-5, abs=1e-6)
    observations, recorded = clean(result['observations']), clean(expected['observations'])
    assert len(observations) == len(recorded)
    for actual, reference in zip(observations, recorded):
        assert actual['step'] == reference['step']
        assert actual == pytest.approx(reference, rel=1e-5, abs=1e-6)
    assert result['actions'] == expected['actions']
