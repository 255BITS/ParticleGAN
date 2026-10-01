from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite import image_solvability as study
from benchmarks.transfer_suite import image_tasks as host


def test_configuration_restores_globals_and_scales_only_prior_lr():
    original_adam, original_prior, original_loss = torch.optim.Adam, host.ParticlePrior, host.GANLoss
    spec=study.resolve(study.HEALTHY[0],dict(changes=dict(prior_lr_multiplier=10.,gan_mode='vanilla')))
    with study.configuration(spec):
        prior=host.ParticlePrior(4,2)
        network=torch.nn.Linear(2,1)
        optimizer=torch.optim.Adam([*network.parameters(),*prior.parameters()],lr=.001)
        assert [g['lr'] for g in optimizer.param_groups]==[.001,.01]
        assert optimizer.param_groups[1]['params'][0] is prior.z
        assert host.GANLoss('logistic','rp').mode=='vanilla'
    assert (torch.optim.Adam,host.ParticlePrior,host.GANLoss)==(original_adam,original_prior,original_loss)


def test_fixed_prior_remains_fixed_and_gates_cannot_change():
    spec=study.resolve(study.HEALTHY[0],dict(changes=dict(prior_learnable=False)))
    with study.configuration(spec):
        prior=host.ParticlePrior(4,2)
        assert not list(prior.parameters())
        assert not prior.z.requires_grad
    with pytest.raises(ValueError,match='gates'):
        study.resolve(study.HEALTHY[0],dict(changes=dict(thresholds={})))


def test_research_baseline_is_exact_host_parity():
    spec=deepcopy(study.HEALTHY[0]); spec.update(steps=24,batch_size=4,particles=8)
    reference=host.run_episode(spec,study.policy(),fixed=True)
    result=study.episode(spec,study.CARDS[0])
    assert 'error' not in reference and 'error' not in result
    assert reference['live']==result['live']
    assert reference['ema']==result['ema']
    assert reference['losses']==result['losses']
    for left,right in zip(reference['observations'],result['observations']):
        assert {k:v for k,v in left.items() if k!='seconds'}=={k:v for k,v in right.items() if k!='seconds'}
