import torch
from reports.toy100.output_rmsprop_scratch import lift_output_step, OutputRMSProp


def test_damped_lift_matches_independent_primal_solve_and_rests_without_signal():
    j=torch.tensor([[1., 2., -.5], [0., 1., 3.]], dtype=torch.float64)
    p=torch.tensor([.1, .3, .7], dtype=torch.float64)
    y=torch.tensor([.02, -.01], dtype=torch.float64)
    ridge=1e-3*(j@torch.diag(p)@j.T).diagonal().mean()
    expected=torch.linalg.solve(j.T@j+ridge*torch.diag(1/p),j.T@y)
    assert torch.allclose(lift_output_step(j,p,y),expected,atol=1e-12)
    assert torch.equal(lift_output_step(j,p,torch.zeros_like(y)),torch.zeros_like(p))


def test_output_moments_aggregate_duplicate_draws_and_advance_once():
    model=torch.nn.Linear(2,1,bias=False).double()
    with torch.no_grad(): model.weight.copy_(torch.tensor([[.4,-.2]]))
    z=torch.nn.Parameter(torch.tensor([[.2,.3],[-.1,.5]],dtype=torch.float64))
    opt=torch.optim.Adam([dict(params=list(model.parameters()),_comparison_prior=False),
                          dict(params=[z],_comparison_prior=True)],lr=.00425,betas=(0.,.999))
    controller=OutputRMSProp();controller.register(model)
    model(z[[0,0,1]]).square().mean().backward()
    rng=torch.get_rng_state().clone()
    controller.step(opt,torch.optim.Adam.step)
    assert controller.count.tolist()==[1,1]
    assert all(int(s['step'])==1 for s in opt.state.values())
    assert torch.equal(torch.get_rng_state(),rng)
    assert torch.isfinite(model.weight).all() and torch.isfinite(z).all()
    assert controller.receipt['updates'][-1]['fit_trials']<=13
    controller.close()
