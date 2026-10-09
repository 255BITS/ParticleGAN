"""Software checks for finite constraints/replay, not scientific qualification."""
from pathlib import Path
import json
import pytest
import torch
from particlegan.hydraulic_shape import HydraulicShapeTravel
from experiments.forge.api import task_formulation_context, CapabilityError
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from experiments.forge.state import state_digest
from experiments.forge.planning import load_idea

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE='hydraulic-local-shape-v3'
WINNER='bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36'


def build(candidate=CANDIDATE,device='cpu'):
    card=load_idea(ROOT,candidate)
    task=json.loads((ROOT/'configs/forge/tasks/gaussian1d_smoke.json').read_text())
    ctx=task_formulation_context(card,task,device=device,root=ROOT)
    g,d=build_vector_models(ctx,resolve_vector_spec(task))
    return ctx,ctx.build_trainer(g,d,max_steps=4)


def test_graph_capacity_and_rigid_motion_invariance():
    real=torch.tensor([[-1.1,0.],[-1.,0.],[-.9,0.],[.9,0.],[1.,0.],[1.1,0.]])
    limiter=HydraulicShapeTravel(1.)
    radius,labels,capacity=limiter.prepare(real)
    assert radius==pytest.approx(.1)
    assert labels.tolist()==[0,0,0,1,1,1]
    assert capacity.tolist()==pytest.approx([.01,.01])
    moved=real[:,[1,0]]+torch.tensor([2.,-3.])
    r2,l2,c2=limiter.prepare(moved)
    assert torch.equal(labels,l2) and torch.allclose(capacity,c2,atol=2e-8)
    assert r2==pytest.approx(radius,abs=3e-7)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_excess_progress_and_mean_constraint(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    weight=torch.nn.Parameter(torch.tensor([[2.]],device=device))
    bias=torch.nn.Parameter(torch.tensor([[.07]],device=device))
    opt=torch.optim.SGD([weight,bias],lr=.1)
    weight.grad=torch.tensor([[-10.]],device=device);bias.grad=torch.tensor([[-.3]],device=device)
    real=torch.tensor([[-1.1],[-1.],[-.9],[.9],[1.],[1.1]],device=device)
    epsilon=torch.tensor([[-.1],[.1]],device=device)
    def shape_probe():return weight*epsilon+bias,-weight*epsilon+bias
    def probe():
        output=weight*epsilon+bias
        return output,output
    limiter=HydraulicShapeTravel(1.);frame=limiter.prepare(real)
    limiter.step(opt,real,probe,shape_probe=shape_probe,prepared=frame)
    row=limiter.summary
    assert row['shape_active']==1 and row['shape_corrections']>0
    assert row['accepted_excess_sum']<row['old_excess_sum']
    assert row['max_shape_bound_violation']==0
    assert row['max_accepted_radius_ratio']<=1
    assert row['max_mean_linear_residual']<=1e-8
    assert row['rejected']==0
    limiter.validate_state_dict(limiter.state_dict(),1)


def test_initial_growth_within_data_capacity_preserves_full_proposal():
    weight=torch.nn.Parameter(torch.tensor([[.1]]));bias=torch.nn.Parameter(torch.zeros(1,1))
    opt=torch.optim.SGD([weight,bias],lr=.1)
    weight.grad=torch.tensor([[-1.]]);bias.grad=torch.tensor([[-.1]])
    real=torch.tensor([[-1.1],[-1.],[-.9],[.9],[1.],[1.1]])
    epsilon=torch.tensor([[-.1],[.1]])
    def shape_probe():return weight*epsilon+bias,-weight*epsilon+bias
    def probe():return (weight*epsilon+bias,)*2
    expected_weight=weight.detach()-.1*weight.grad
    expected_bias=bias.detach()-.1*bias.grad
    limiter=HydraulicShapeTravel(1.)
    limiter.step(opt,real,probe,shape_probe=shape_probe,prepared=limiter.prepare(real))
    assert torch.equal(weight,expected_weight) and torch.equal(bias,expected_bias)
    assert limiter.summary['shape_active']==0 and limiter.summary['scale_sum']==1


def test_public_checkpoint_replay_stream_parity_and_settings_validation():
    torch.set_num_threads(1)
    context,trainer=build();_,control=build(WINNER)
    real=torch.linspace(1,3,128).reshape(-1,1);rng=torch.get_rng_state().clone()
    trainer.step(real,generator_real=real);saved=context.state_dict()
    trainer.step(real,generator_real=real);expected=state_digest(context.state_dict())
    context.load_state_dict(saved);trainer.step(real,generator_real=real)
    assert state_digest(context.state_dict())==expected
    assert torch.equal(rng,torch.get_rng_state())
    for _ in range(2):control.step(real,generator_real=real)
    for name in trainer._STREAMS:
        assert torch.equal(getattr(trainer,name).get_state(),getattr(control,name).get_state())
    trainer.hydraulic.validate_state_dict(trainer.hydraulic.state_dict(),2)
    packet=trainer.state_dict();packet['hydraulic']['shape_settings']['ray_trials']=8
    before=state_digest(trainer.state_dict())
    with pytest.raises(ValueError,match='hydraulic'):trainer.load_state_dict(packet)
    assert state_digest(trainer.state_dict())==before


def test_zero_spacing_and_preflight_refusals():
    _,trainer=build();before=[p.detach().clone() for group in trainer.opt_g.param_groups for p in group['params']]
    result=trainer.step(torch.ones(128,1),generator_real=torch.ones(128,1))
    assert result['step']==1 and trainer.hydraulic.summary['zero_spacing']==1
    assert all(torch.equal(p,old) for p,old in zip((p for group in trainer.opt_g.param_groups for p in group['params']),before))
    trainer.hydraulic.validate_state_dict(trainer.hydraulic.state_dict(),1)
    before=state_digest(trainer.state_dict())
    for batch,other in [(torch.full((128,1),float('nan')),None),(torch.ones(128,1),lambda:torch.ones(128,1))]:
        with pytest.raises(ValueError):trainer.step(batch,generator_real=other)
        assert state_digest(trainer.state_dict())==before
    with pytest.raises(ValueError,match='coordinates'):HydraulicShapeTravel(1.).prepare(torch.ones(4,3))


def test_two_pole_predecessor_and_candidate_are_genuine_blockers():
    task=json.loads((ROOT/'configs/forge/tasks/two_pole.json').read_text())
    for candidate in [CANDIDATE,'hydraulic-secant-deformation-v2']:
        card=json.loads((ROOT/'configs/forge/ideas'/f'{candidate}.json').read_text())
        with pytest.raises(CapabilityError):task_formulation_context(card,task,root=ROOT)
