"""Keep shared-default search free of per-example recipe overrides."""

import pytest

from benchmarks.transfer_suite.shared_default_search import prepare


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
