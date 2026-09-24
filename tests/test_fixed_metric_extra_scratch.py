import pytest
import torch
from reports.toy100.fixed_metric_extra_scratch import FixedMetricExtraRecorder


def test_bilinear_update_uses_first_metric_once():
    x=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    y=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64))
    ox=torch.optim.Adam([x],lr=.1,betas=(0.,.9),eps=0.)
    oy=torch.optim.Adam([y],lr=.1,betas=(0.,.9),eps=0.)
    recorder=FixedMetricExtraRecorder()
    for phase in recorder.phases(0,oy,ox,{}):
        oy.zero_grad();(-x.detach()*y).sum().backward();recorder.step(oy,torch.optim.Adam.step)
        ox.zero_grad();(x*y.detach()).sum().backward();recorder.step(ox,torch.optim.Adam.step)
    assert x.item()==pytest.approx(.895)
    assert y.item()==pytest.approx(2.09)
    assert ox.state[x]['step']==1 and oy.state[y]['step']==1
    assert recorder.rng_replay_verified==1


def test_rng_replays_global_host_and_policy_streams():
    class Policy:pass
    x=torch.nn.Parameter(torch.tensor([1.]));y=torch.nn.Parameter(torch.tensor([2.]))
    ox=torch.optim.Adam([x],lr=.01,betas=(0.,.99));oy=torch.optim.Adam([y],lr=.01,betas=(0.,.99))
    host=torch.Generator().manual_seed(4);p=Policy();p.input_stream=torch.Generator().manual_seed(5);p.output_stream=None
    recorder=FixedMetricExtraRecorder();draws=[]
    for phase in recorder.phases(0,oy,ox,dict(stream=host,noise_policy=p)):
        draws.append((torch.rand(3),torch.rand(3,generator=host),torch.rand(3,generator=p.input_stream)))
        y.grad=-x.detach().clone();recorder.step(oy,torch.optim.Adam.step)
        x.grad=y.detach().clone();recorder.step(ox,torch.optim.Adam.step)
    assert all(torch.equal(a,b) for a,b in zip(*draws))


def test_zero_game_field_stays_zero_without_zero_centered_penalty():
    x=torch.nn.Parameter(torch.tensor([0.]));y=torch.nn.Parameter(torch.tensor([0.]))
    ox=torch.optim.Adam([x],lr=.1,betas=(0.,.99));oy=torch.optim.Adam([y],lr=.1,betas=(0.,.99))
    recorder=FixedMetricExtraRecorder()
    for phase in recorder.phases(0,oy,ox,{}):
        y.grad=-x.detach().clone();recorder.step(oy,torch.optim.Adam.step)
        x.grad=y.detach().clone();recorder.step(ox,torch.optim.Adam.step)
    assert x.item()==0 and y.item()==0


def test_delayed_activation_preserves_original_host_and_rng():
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.fixed_metric_extra_scratch import fixed_metric_extra
    torch.set_num_threads(1)
    def run():
        policy=NoisePolicy(.029,.5,.1,1200)
        result=mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),noise_policy=policy,diagnostics=True)
        return result,policy.receipt(),torch.get_rng_state(),policy.input_stream.get_state()
    original=run()
    with fixed_metric_extra(start_step=1000):
        wrapped=run()
    assert original[:2]==wrapped[:2]
    assert all(torch.equal(a,b) for a,b in zip(original[2:],wrapped[2:]))
