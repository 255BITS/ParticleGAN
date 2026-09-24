"""Algebra and real-host accounting for the projected skew mechanism."""
import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.projected_skew_scratch import ProjectedSkewRecorder,projected_skew


def scalar_game(matrix,initial=(2.,1.)):
    d=torch.nn.Parameter(torch.tensor([initial[0]],dtype=torch.float64))
    g=torch.nn.Parameter(torch.tensor([initial[1]],dtype=torch.float64))
    od=torch.optim.Adam([d],lr=.1,betas=(0.,.9),eps=1e-12)
    og=torch.optim.Adam([g],lr=.1,betas=(0.,.9),eps=1e-12)
    recorder=ProjectedSkewRecorder()
    draws=[]
    for _ in recorder.phases(0,od,og,{}):
        draws.append(torch.rand(3))
        field=matrix@torch.cat((d.detach(),g.detach()))
        d.grad=field[:1].clone();recorder.step(od,torch.optim.Adam.step)
        g.grad=field[1:].clone();recorder.step(og,torch.optim.Adam.step)
    assert all(torch.equal(draws[0],draw) for draw in draws)
    assert od.state[d]['step']==og.state[g]['step']==1
    for row in recorder.receipt()['optimizers']:
        assert row['calls']==1+len(recorder.queries)
    return torch.cat((d.detach(),g.detach())),recorder


def test_bilinear_matches_exact_implicit_response():
    field=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    final,recorder=scalar_game(field)
    base=torch.tensor([2.,1.],dtype=torch.float64)
    p=torch.diag(.1/((field@base).abs()+1e-12))
    expected=base+torch.linalg.solve(torch.eye(2,dtype=torch.float64)+p@field,-p@field@base)
    assert torch.allclose(final,expected,atol=1e-12,rtol=0)
    assert recorder.skew_steps[0]['projected_linear_relative_residual']<1e-12
    assert recorder.skew_steps[0]['actual_metric_norm_ratio']<=1


def test_symmetric_potential_game_preserves_gradient_step():
    field=torch.tensor([[2.,1.],[1.,3.]],dtype=torch.float64)
    final,recorder=scalar_game(field)
    assert torch.allclose(final,torch.tensor([1.9,.9],dtype=torch.float64),atol=1e-12,rtol=0)
    assert abs(recorder.skew_steps[0]['skew_coefficient'])<1e-12


def test_own_hessian_is_excluded_from_skew_response():
    field=torch.tensor([[2.,-1.],[3.,4.]],dtype=torch.float64)
    final,recorder=scalar_game(field)
    base=torch.tensor([2.,1.],dtype=torch.float64);gradient=field@base
    root=torch.diag((.1/(gradient.abs()+1e-12)).sqrt())
    weighted=root@field@root;skew=(weighted-weighted.T)/2
    expected=base+root@torch.linalg.solve(torch.eye(2,dtype=torch.float64)+skew,-root@gradient)
    assert torch.allclose(final,expected,atol=1e-12,rtol=0)


def test_zero_field_sits_without_queries():
    final,recorder=scalar_game(torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64),initial=(0.,0.))
    assert torch.count_nonzero(final)==0
    assert recorder.queries==[]
    assert recorder.skew_steps[0]['zero_field']


def test_real_host_inactive_identity_and_active_rng_accounting():
    torch.set_num_threads(1)
    def run():
        policy=NoisePolicy(.029,.5,.1,1200,output_noise_rng='isolated')
        result=mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1),noise_policy=policy,diagnostics=True)
        return result,policy.receipt(),torch.get_rng_state().clone(),policy.input_stream.get_state().clone(),policy.output_stream.get_state().clone()
    original=run()
    with projected_skew(start_step=1000) as (inactive,_):wrapped=run()
    assert original[:2]==wrapped[:2]
    assert all(torch.equal(a,b) for a,b in zip(original[2:],wrapped[2:]))
    assert inactive.outer_steps==0
    with projected_skew() as (active,_):observed=run()
    assert all(torch.equal(a,b) for a,b in zip(original[2:],observed[2:]))
    assert original[1]['step_calls']==observed[1]['step_calls']==1
    assert active.outer_steps==1
    assert active.rng_replay_verified==len(active.queries)
    for row in active.receipt()['optimizers']:
        assert row['calls']==1+len(active.queries)
        assert all(step==1 for group in row['groups'] for step in group['moment_steps'])
