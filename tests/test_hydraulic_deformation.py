"""Public API software checks; these tiny updates are not scientific evidence."""
from pathlib import Path
import json
import pytest
import torch

from particlegan.hydraulic import HydraulicDeformationTravel
from experiments.forge.api import task_formulation_context
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from experiments.forge.state import state_digest


def build(candidate='hydraulic-secant-deformation-v2'):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/forge/ideas'/f'{candidate}.json').read_text())
    task=json.loads((root/'configs/forge/tasks/gaussian1d_smoke.json').read_text())
    context=task_formulation_context(c,task,device='cpu',root=root)
    g,d=build_vector_models(context,resolve_vector_spec(task))
    return context,context.build_trainer(g,d,max_steps=4)


def test_secant_formula_and_network_only_gradient():
    g=torch.nn.Linear(1,1,bias=True)
    with torch.no_grad(): g.weight.fill_(2);g.bias.fill_(5)
    centers=torch.nn.Parameter(torch.tensor([[0.],[1.]]))
    latent=centers+torch.tensor([[.2],[-.2]])
    loss=HydraulicDeformationTravel(1,1).regularizer(g,latent,centers,torch.tensor([[0.],[2.]]),torch.tensor(3.))
    assert float(loss.detach())==pytest.approx(.48)
    loss.backward()
    assert centers.grad is None
    assert float(g.weight.grad)==pytest.approx(.48)
    assert float(g.bias.grad)==pytest.approx(0)


def test_zero_spacing_completes_without_partial_failure():
    _,trainer=build()
    before=[p.detach().clone() for group in trainer.opt_g.param_groups for p in group['params']]
    result=trainer.step(torch.ones(128,1),generator_real=torch.ones(128,1))
    assert result['step']==1 and trainer.hydraulic.summary['zero_spacing']==1
    assert trainer.hydraulic.summary['accepted_rms_sum']==0
    for p,old in zip((p for group in trainer.opt_g.param_groups for p in group['params']),before):
        assert torch.equal(p,old)
    assert trainer.hydraulic.summary['updates']==1
    trainer.hydraulic.validate_state_dict(trainer.hydraulic.state_dict(),1)
    assert torch.isfinite(result['loss_g'])


def test_nonfinite_and_callable_refuse_before_any_mutation():
    _,trainer=build();before=state_digest(trainer.state_dict())
    for data,generator_real in [(torch.full((128,1),float('nan')),None),(torch.ones(128,1),lambda:torch.ones(128,1))]:
        with pytest.raises(ValueError):trainer.step(data,generator_real=generator_real)
        assert state_digest(trainer.state_dict())==before


def test_duplicate_spacing_uses_distinct_samples():
    limiter=HydraulicDeformationTravel(1,1)
    assert limiter.radius(torch.tensor([[0.],[0.],[1.],[1.]]))==1
    assert limiter.radius(torch.ones(1,1))==0


def test_checkpoint_and_consumed_rng_match_control():
    torch.set_num_threads(1)
    context,trainer=build();real=torch.linspace(1,3,128).reshape(-1,1)
    trainer.step(real,generator_real=real);saved=context.state_dict()
    trainer.step(real,generator_real=real);expected=state_digest(context.state_dict())
    context.load_state_dict(saved);trainer.step(real,generator_real=real)
    assert state_digest(context.state_dict())==expected
    _,control=build('hydraulic-output-travel-v1')
    control.step(real,generator_real=real);control.step(real,generator_real=real)
    for name in trainer._STREAMS:
        assert torch.equal(getattr(trainer,name).get_state(),getattr(control,name).get_state())
    assert trainer.hydraulic.summary['max_accepted_radius_ratio']<=1
    packet=trainer.state_dict();packet['hydraulic']['deformation_weight']=2
    before=state_digest(trainer.state_dict())
    with pytest.raises(ValueError,match='hydraulic'):trainer.load_state_dict(packet)
    assert state_digest(trainer.state_dict())==before
