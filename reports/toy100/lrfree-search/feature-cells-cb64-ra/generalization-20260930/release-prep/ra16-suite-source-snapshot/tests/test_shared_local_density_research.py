"""Pointwise cap gradients and immutable recipe/resource tests for local D cards."""
from copy import deepcopy
import pytest
import torch

from benchmarks.transfer_suite.compare_defaults import effective_spec, plan
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.shared_discriminator_search import recipe
from benchmarks.transfer_suite.shared_local_density_research import ARCHITECTURES, constructor, variant
from benchmarks.transfer_suite.shared_local_density_search import episode
from benchmarks.transfer_suite.shared_variants import architecture_spec


@pytest.mark.parametrize('card', ARCHITECTURES, ids=lambda c: c['name'])
def test_pointwise_cap_path_and_finite_parameter_gradients(card):
    torch.manual_seed(0)
    model = constructor(card)(2, card['hidden'], card['layers'], 0)
    x = torch.tensor([[.3, -.7], [1.8, 1.2]], requires_grad=True)
    score = model(x)
    assert torch.allclose(score, torch.cat([model(p[None]) for p in x]), atol=1e-7)
    gradient = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    (gradient.square().sum()+score.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_cards_change_only_discriminator():
    for job in plan():
        if job['spec']['runner'] != 'vector':
            continue
        before = axes(effective_spec(job['spec'], recipe()), 'vector')
        for card in ARCHITECTURES:
            after = axes(effective_spec(architecture_spec(job['spec'], variant(card)), recipe()), 'vector')
            assert after['architecture']['generator'] == before['architecture']['generator']
            for key in ('training', 'formulation', 'resources', 'target'):
                assert after[key] == before[key]


def test_actual_parameter_groups_keep_shared_cap6_recipe():
    torch.set_num_threads(1)
    job = deepcopy(next(j for j in plan() if j['spec']['name'] == 'vector_unequal_mass'))
    job['spec']['steps'] = 24  # Unit check only; never a scored experiment.
    payload = episode(job, ARCHITECTURES[3])
    assert 'error' not in payload['result']
    receipts = {g['role']: (g['lr'], g['betas']) for g in payload['applied']}
    assert receipts == {'d': (.00425, [0., .99]), 'g': (.00425, [0., .99]), 'prior': (.0085, [0., .99])}
    assert len(payload['result']['observations']) == 24
    assert payload['recipe']['reg_coeff'] == 6 and payload['recipe']['prior_reg'] == .05
