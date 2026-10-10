"""Architecture invariants and real optimizer receipts, not scored toy runs."""
from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite.compare_defaults import effective_spec, plan
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.shared_critic_research import ARCHITECTURES, constructor, variant
from benchmarks.transfer_suite.shared_discriminator_search import episode, prepare, recipe
from benchmarks.transfer_suite.shared_variants import architecture_spec


@pytest.mark.parametrize('card', ARCHITECTURES, ids=lambda c: c['name'])
def test_pointwise_scores_and_cap_parameter_derivatives(card):
    torch.manual_seed(0)
    model = constructor(card)(2, card['hidden'], card['layers'], card['fourier'])
    x = torch.tensor([[.3, -.7], [1.8, 1.2]], requires_grad=True)
    score = model(x)
    assert score.shape == (2,)
    assert torch.allclose(score, torch.cat([model(point[None]) for point in x]), atol=1e-7)
    derivative = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    # Exercise the second-order parameter path used by the gradient cap.
    (derivative.square().sum()+score.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_variants_change_only_discriminator_and_one_recipe_is_used():
    jobs = [j for j in plan() if j['spec']['runner'] == 'vector']
    for job in jobs:
        base = axes(effective_spec(job['spec'], recipe()), 'vector')
        for card in ARCHITECTURES:
            actual = axes(effective_spec(architecture_spec(job['spec'], variant(card)), recipe()), 'vector')
            assert actual['architecture']['generator'] == base['architecture']['generator']
            for key in ('formulation', 'training', 'resources', 'target'):
                assert actual[key] == base[key]
    assert recipe().lr == .00425 and recipe().reg_coeff == 6


def test_rejects_recipe_or_resource_mutations():
    with pytest.raises(ValueError):
        prepare(dict(purpose='bad', architectures=[ARCHITECTURES[0]['name']],
                     tasks=['vector_unequal_mass'], overrides={'lr': .001}))
    original = next(j['spec'] for j in plan() if j['spec']['name'] == 'vector_unequal_mass')
    with pytest.raises(ValueError):
        architecture_spec(original, dict(name='bad', overrides={'steps': 2400}))


def test_small_unit_host_records_actual_unchanged_adam_groups():
    torch.set_num_threads(1)
    job = deepcopy(next(j for j in plan() if j['spec']['name'] == 'vector_unequal_mass'))
    # Shortened solely for API verification; never included in experiment artifacts.
    job['spec']['steps'] = 24
    payload = episode(job, ARCHITECTURES[0])
    assert 'error' not in payload['result']
    receipts = {x['role']: x for x in payload['applied']}
    assert set(receipts) == {'g', 'd', 'prior'}
    assert {k: v['lr'] for k, v in receipts.items()} == {'g': .00425, 'd': .00425, 'prior': .0085}
    assert all(x['betas'] == [0., .99] for x in receipts.values())
    assert len(payload['result']['observations']) == 24
    assert payload['original_spec'] == job['spec']
