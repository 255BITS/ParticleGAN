"""Directional G acceptance and same-batch accounting for the second arm."""
import pytest
import torch

from reports.toy100.alternating_two_player_linesearch import TwoPlayerLineSearchRecorder,alternating_two_player_linesearch
from test_alternating_linesearch_scratch import host


def run_scalar(sign=1.,curvature=4.):
    d=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64))
    g=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    od=torch.optim.Adam([d],lr=1.,betas=(0.,.9),eps=1e-12)
    og=torch.optim.Adam([g],lr=1.,betas=(0.,.9),eps=1e-12)
    losses={};recorder=TwoPlayerLineSearchRecorder(loss_reader=lambda role:losses[role])
    draws=[]
    for _ in recorder.phases(0,od,og,{}):
        draws.append(torch.rand(3))
        losses['d']=float(.5*d.detach().square());d.grad=d.detach().clone()
        recorder.step(od,torch.optim.Adam.step)
        losses['g']=float(.5*sign*curvature*g.detach().square());g.grad=sign*curvature*g.detach().clone()
        recorder.step(og,torch.optim.Adam.step)
    assert all(torch.equal(draws[0],x) for x in draws)
    assert od.state[d]['step']==og.state[g]['step']==1
    return d,g,recorder


def test_scalar_positive_curvature_matches_declared_margin():
    d,g,recorder=run_scalar()
    assert d.item()==0 and g.item()==pytest.approx(.75)
    row=recorder.records[0]
    assert row['g']['factor']==.25
    assert row['g']['accepted_effective_norm_curvature']==pytest.approx(.25)
    assert row['g']['backtracks']==2
    assert recorder.rng_replay_verified==4
    for opt in recorder.receipt()['optimizers']:assert opt['calls']==5


def test_negative_curvature_keeps_useful_full_proposal():
    _,g,recorder=run_scalar(sign=-1.,curvature=1.)
    assert g.item()==pytest.approx(2.)
    assert recorder.records[0]['g']['factor']==1
    assert recorder.records[0]['g']['actual_improvement']>recorder.records[0]['g']['predicted_improvement']


def test_real_host_replays_one_batch_and_advances_moments_once():
    ordinary=host();observed=host(alternating_two_player_linesearch())
    assert all(torch.equal(a,b) for a,b in zip(ordinary[2],observed[2]))
    recorder=observed[3]
    expected_replays=sum(len(row['d_trials'])+len(row['g_trials']) for row in recorder.records)
    assert recorder.rng_replay_verified==expected_replays
    for row in recorder.receipt()['optimizers']:
        assert row['calls']==3+expected_replays
        assert all(step==3 for group in row['moment_steps'] for step in group)
