"""Analytic order, accepted descent, RNG and moment audits."""
import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_curvature_scratch import alternating_curvature
from reports.toy100.alternating_linesearch_scratch import DLineSearchRecorder,alternating_linesearch


def test_d_backtracking_precedes_g_gradient_and_keeps_one_moment_update():
    d=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    g=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    od=torch.optim.Adam([d],lr=10.,betas=(0.,.9),eps=1e-12)
    og=torch.optim.Adam([g],lr=1.,betas=(0.,.9),eps=1e-12)
    losses={};recorder=DLineSearchRecorder(loss_reader=lambda role:losses[role])
    draws=[]
    for _ in recorder.phases(0,od,og,{}):
        draws.append(torch.rand(4))
        losses['d']=float(2*d.detach().square()-g.detach()*d.detach())
        d.grad=4*d.detach()-g.detach();recorder.step(od,torch.optim.Adam.step)
        losses['g']=float(d.detach()*g.detach()+1.5*g.detach().square())
        g.grad=d.detach()+3*g.detach();recorder.step(og,torch.optim.Adam.step)
    assert d.item()==pytest.approx(-.5)
    assert g.item()==pytest.approx(1.-.25/(3/2.5))
    assert recorder.records[0]['d']['alpha']==.25
    assert recorder.records[0]['d']['backtracks']==2
    assert all(torch.equal(draws[0],x) for x in draws)
    assert od.state[d]['step']==og.state[g]['step']==1
    assert recorder.rng_replay_verified==4
    assert recorder.rows[od]['calls']==recorder.rows[og]['calls']==5
    for trial in recorder.records[0]['d_trials']:
        assert trial['accepted']==(trial['loss']<=trial['armijo_threshold']+trial['rounding_tolerance'])


def test_zero_field_can_rest():
    d=torch.nn.Parameter(torch.tensor([0.]));g=torch.nn.Parameter(torch.tensor([0.]))
    od=torch.optim.Adam([d],lr=.1,betas=(0.,.9));og=torch.optim.Adam([g],lr=.1,betas=(0.,.9))
    recorder=DLineSearchRecorder(loss_reader=lambda role:0.)
    for _ in recorder.phases(0,od,og,{}):
        d.grad=torch.zeros_like(d);recorder.step(od,torch.optim.Adam.step)
        g.grad=torch.zeros_like(g);recorder.step(og,torch.optim.Adam.step)
    assert d.item()==g.item()==0
    assert recorder.records[0]['d']['alpha']==1
    assert od.state[d]['step']==og.state[g]['step']==1


def test_nonsmooth_exhaustion_rejects_d_at_exact_base_and_still_updates_g():
    d=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    g=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    od=torch.optim.Adam([d],lr=.1,betas=(0.,.9));og=torch.optim.Adam([g],lr=.1,betas=(0.,.9))
    def loss(role):
        return float(.5*d.detach().square())+(10. if d.item()!=1 else 0.)
    recorder=DLineSearchRecorder(loss_reader=loss,max_retries=2)
    for _ in recorder.phases(0,od,og,{}):
        d.grad=d.detach().clone();recorder.step(od,torch.optim.Adam.step)
        g.grad=g.detach().clone();recorder.step(og,torch.optim.Adam.step)
    row=recorder.records[0]['d']
    assert row['alpha']==0 and row['rejected_proposal'] and row['zero_step_identity_verified']
    assert d.item()==1 and g.item()<1
    assert od.state[d]['step']==og.state[g]['step']==1


class CapturePolicy(NoisePolicy):
    def register_generator_optimizer(self,opt_g,opt_d):
        self.optimizers=(opt_d,opt_g)
        return super().register_generator_optimizer(opt_g,opt_d)


def host(context=None):
    torch.set_num_threads(1)
    policy=CapturePolicy(.029,.5,.1,1200,output_noise_rng='isolated')
    if context is None:
        result=mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3),noise_policy=policy,diagnostics=True)
        recorder=None
    else:
        with context as (recorder,_):
            result=mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3),noise_policy=policy,diagnostics=True)
    state=[]
    for opt in policy.optimizers:
        for group in opt.param_groups:
            for p in group['params']:
                state.extend([p.detach().clone(),*[v.detach().clone() for v in opt.state[p].values() if isinstance(v,torch.Tensor)]])
    return result,state,[torch.get_rng_state().clone(),policy.input_stream.get_state().clone(),policy.output_stream.get_state().clone()],recorder


def test_disabled_search_and_g_bound_preserve_alternating_adam_exactly():
    plain=host();wrapped=host(alternating_linesearch(search_enabled=False,curvature_bound=1e9))
    assert plain[0]==wrapped[0]
    assert all(torch.equal(a,b) for a,b in zip(plain[1]+plain[2],wrapped[1]+wrapped[2]))
    assert wrapped[3].rng_replay_verified==6


def test_replacing_only_d_guard_keeps_pr82_g_rule_exact():
    original=host(alternating_curvature(curvature_bound=.25,advantage_gate=None))
    observed=host(alternating_linesearch(search_enabled=False))
    assert original[0]==observed[0]
    assert all(torch.equal(a,b) for a,b in zip(original[1]+original[2],observed[1]+observed[2]))
    assert [x['factor'] for x in original[3].records]==[x['g']['factor'] for x in observed[3].records]


def test_active_host_preserves_one_batch_rng_and_moment_progression():
    ordinary=host();active=host(alternating_linesearch())
    assert all(torch.equal(a,b) for a,b in zip(ordinary[2],active[2]))
    recorder=active[3]
    assert recorder.outer_steps==3
    assert recorder.rng_replay_verified==sum(len(x['d_trials'])+1 for x in recorder.records)
    for opt in recorder.receipt()['optimizers']:
        assert opt['calls']==3+recorder.rng_replay_verified
        assert all(step==3 for group in opt['moment_steps'] for step in group)
