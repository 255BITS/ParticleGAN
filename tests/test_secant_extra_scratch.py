import pytest
import torch

from reports.toy100.secant_extra_scratch import SecantExtraRecorder


def test_secant_backtracks_bilinear_once_per_moment_update():
    x=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    y=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    ox=torch.optim.Adam([x],lr=2.,betas=(0.,.9),eps=1e-12)
    oy=torch.optim.Adam([y],lr=2.,betas=(0.,.9),eps=1e-12)
    recorder=SecantExtraRecorder(c=.5)
    rng_before=torch.get_rng_state()
    for phase in recorder.phases(0,oy,ox,{}):
        torch.rand(3)
        y.grad=-x.detach().clone();recorder.step(oy,torch.optim.Adam.step)
        x.grad=y.detach().clone();recorder.step(ox,torch.optim.Adam.step)
    assert x.item()==pytest.approx(.375)
    assert y.item()==pytest.approx(2.25)
    assert recorder.last_scale==.25
    assert recorder.rng_replay_verified==3
    assert ox.state[x]['step']==1 and oy.state[y]['step']==1
    end=torch.get_rng_state();torch.set_rng_state(rng_before);torch.rand(3)
    assert torch.equal(end,torch.get_rng_state())
    assert [row['accepted'] for row in recorder.trials]==[False,False,True]


def test_secant_zero_field_and_rejection_budget():
    x=torch.nn.Parameter(torch.tensor([0.]));y=torch.nn.Parameter(torch.tensor([0.]))
    ox=torch.optim.Adam([x],lr=.1,betas=(0.,.99));oy=torch.optim.Adam([y],lr=.1,betas=(0.,.99))
    recorder=SecantExtraRecorder()
    for phase in recorder.phases(0,oy,ox,{}):
        y.grad=-x.detach().clone();recorder.step(oy,torch.optim.Adam.step)
        x.grad=y.detach().clone();recorder.step(ox,torch.optim.Adam.step)
    assert x.item()==0 and y.item()==0
    assert recorder.last_scale==1 and recorder.trials[0]['secant_ratio']==0


def test_secant_rejects_nonfinite_game_gradient():
    x=torch.nn.Parameter(torch.tensor([1.]));y=torch.nn.Parameter(torch.tensor([2.]))
    ox=torch.optim.Adam([x],lr=.1,betas=(0.,.99));oy=torch.optim.Adam([y],lr=.1,betas=(0.,.99))
    recorder=SecantExtraRecorder()
    with pytest.raises(FloatingPointError):
        for phase in recorder.phases(0,oy,ox,{}):
            y.grad=torch.full_like(y,float('nan'));recorder.step(oy,torch.optim.Adam.step)
