from copy import deepcopy

import pytest

from benchmarks.transfer_suite import image_solvability as study
from benchmarks.transfer_suite import image_tasks as host


def test_cards_are_recipe_or_task_changes():
    spec = study.resolve(study.HEALTHY[0], dict(changes=dict(prior_lr_multiplier=10., gan_mode='vanilla')))
    recipe = host.spec_recipe(spec)
    assert (recipe.prior_lr_mult, recipe.gan_mode, recipe.loss_type) == (10., 'vanilla', 'logistic')
    groups, _ = host.receipts(spec, recipe)
    assert [g['lr'] for g in groups] == [spec['lr_g'], 10. * spec['lr_g'], spec['lr_d']]


def test_fixed_prior_remains_fixed_and_gates_cannot_change():
    spec = study.resolve(study.HEALTHY[0], dict(changes=dict(prior_learnable=False)))
    task = host.ImageTask(spec)
    nets = task.networks(task.recipe(), 0)
    assert not list(nets.prior.parameters())
    assert not nets.prior.z.requires_grad
    assert [g['role'] for g in host.receipts(spec, host.spec_recipe(spec))[0]] == ['network', 'critic']
    with pytest.raises(ValueError, match='gates'):
        study.resolve(study.HEALTHY[0], dict(changes=dict(thresholds={})))


def test_research_baseline_is_exact_host_parity():
    spec=deepcopy(study.HEALTHY[0]); spec.update(steps=24,batch_size=4,particles=8)
    reference=host.run_episode(spec,study.policy())
    result=study.episode(spec,study.CARDS[0])
    assert 'error' not in reference and 'error' not in result
    assert reference['live']==result['live']
    assert reference['ema']==result['ema']
    assert reference['losses']==result['losses']
    for left,right in zip(reference['observations'],result['observations']):
        assert {k:v for k,v in left.items() if k!='seconds'}=={k:v for k,v in right.items() if k!='seconds'}
