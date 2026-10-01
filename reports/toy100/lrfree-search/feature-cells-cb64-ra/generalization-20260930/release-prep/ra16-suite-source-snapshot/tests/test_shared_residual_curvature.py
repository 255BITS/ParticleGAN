"""Verify exact base initialization and the cap path for residual-curvature cards."""
from copy import deepcopy
import pytest
import torch
from benchmarks.transfer_suite.compare_defaults import effective_spec,plan
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.shared_critic_research import SharedResearchCritic,ARCHITECTURES as BASES
from benchmarks.transfer_suite.shared_residual_curvature_research import ARCHITECTURES,constructor,variant
from benchmarks.transfer_suite.shared_residual_curvature_search import episode
from benchmarks.transfer_suite.shared_discriminator_search import recipe
from benchmarks.transfer_suite.shared_variants import architecture_spec

@pytest.mark.parametrize('card',ARCHITECTURES,ids=lambda c:c['name'])
def test_exact_base_initial_score_rng_and_cap_path(card):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model=constructor(card)(2,card['hidden'],3,0)
    actual_rng=torch.get_rng_state()
    torch.manual_seed(0)
    base_card=next(c for c in BASES if c['name']==card['base'])
    base=SharedResearchCritic(2,card['hidden'],3,0,architecture=base_card)
    assert torch.equal(torch.get_rng_state(),actual_rng)
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in model.main.state_dict().items())
    x=torch.tensor([[.3,-.7],[1.8,1.2]],requires_grad=True)
    assert torch.equal(model(x),base(x))
    gradient=torch.autograd.grad(model(x).sum(),x,create_graph=True)[0]
    derivative=torch.autograd.grad(gradient.square().sum(),model.coefficients)[0]
    assert torch.isfinite(derivative).all() and derivative.abs().sum()>0
    assert torch.allclose(model(x),torch.cat([model(p[None]) for p in x]),atol=1e-7)


def test_discriminator_only_overrides_and_actual_receipts():
    torch.set_num_threads(1)
    job=deepcopy(next(j for j in plan() if j['spec']['name']=='vector_unequal_mass'))
    before=axes(effective_spec(job['spec'],recipe()),'vector')
    for card in ARCHITECTURES:
        after=axes(effective_spec(architecture_spec(job['spec'],variant(card)),recipe()),'vector')
        assert before['architecture']['generator']==after['architecture']['generator']
        assert all(before[k]==after[k] for k in ('training','formulation','resources','target'))
    job['spec']['steps']=24 # Unit API check only, never scored evidence.
    result=episode(job,ARCHITECTURES[0])
    assert 'error' not in result['result']
    assert {a['role']:(a['lr'],a['betas']) for a in result['applied']}=={'g':(.00425,[0.,.99]),'d':(.00425,[0.,.99]),'prior':(.0085,[0.,.99])}
    assert len(result['result']['observations'])==24
