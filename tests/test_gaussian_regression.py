"""Composition/protocol checks; tiny public calls are software, not quality arms."""
from pathlib import Path
import hashlib
import torch
import pytest
from particlegan import GANTrainer,get_recipe,training
from experiments.forge.planning import resolve_idea
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('role,mode',[('local','none'),('direction','direction_blend'),('finite','strict_progress')])
def test_global_configuration_five_original_tasks_and_owned_streams(role,mode,monkeypatch):
    request=resolve_idea(ROOT,f'gaussian_regression-round5-{role}-v1',study=f'gaussian_regression-round5-{role}-study-v1')
    assert request['study_review']['status']=='READY'
    assert set(request['tasks'])=={'gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_two_broad','vector_unequal_width'}
    assert sum(j['budget_seconds'] for j in request['jobs'])==6120
    calls=[];original=training.constraint_geometry_backward
    def capture(loss,opt,protected,**kw):
        calls.append((loss.detach(),[p.detach() for p in protected],kw.get('protected_evaluator')))
        return original(loss,opt,protected,**kw)
    monkeypatch.setattr(training,'constraint_geometry_backward',capture)
    def make(which):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            recipe=get_recipe('bcap',**{**request['candidate']['recipe_overrides'],'constraint_geometry_mode':which},num_particles=12,z_dim=2,batch_size=6,total_steps=2,standardize=False,input_noise_std=0.,output_noise_std=0.)
            return GANTrainer(recipe,torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    baseline,trainer=make('none'),make(mode)
    real=torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
    baseline.step(real);calls.clear();result=trainer.step(real)
    assert len(calls)==1 and len(calls[0][1])==1
    assert calls[0][1][0].item()==result['loss_gan'].item()
    assert calls[0][0].item()==pytest.approx((result['loss_gan']+result['kinetic_transport']+result['kinetic_transport_local']).item())
    assert (calls[0][2] is not None)==(mode=='strict_progress')
    for key in baseline.D.state_dict():assert torch.equal(baseline.D.state_dict()[key],trainer.D.state_dict()[key])
    for key in baseline.state_dict()['streams']:assert torch.equal(baseline.state_dict()['streams'][key],trainer.state_dict()['streams'][key])
    restored=make(mode);restored.load_state_dict(trainer.state_dict())
    a,b=trainer.step(real),restored.step(real)
    for key in a:assert torch.equal(a[key],b[key]) if isinstance(a[key],torch.Tensor) else a[key]==b[key]
    for key in trainer.G.state_dict():assert torch.equal(trainer.G.state_dict()[key],restored.G.state_dict()[key])
