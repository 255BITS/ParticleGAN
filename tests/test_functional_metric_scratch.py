import torch
from reports.toy100.functional_metric_scratch import metric_correction, FunctionalMetric


def test_woodbury_matches_independent_primal_solve_and_preserves_zero():
    j=torch.tensor([[1.,2.,-.5],[0.,1.,3.]],dtype=torch.float64)
    p=torch.tensor([.1,.3,.7],dtype=torch.float64)
    g=torch.tensor([-.2,.5,.8],dtype=torch.float64)
    ridge=.04
    actual=metric_correction(-p*g,p,j,ridge)
    expected=torch.linalg.solve(torch.diag(1/p)+j.T@j/ridge,-g)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(metric_correction(torch.zeros(3,dtype=torch.float64),p,j,ridge),torch.zeros_like(g))


def test_duplicate_rows_are_exact_and_functional_gain_is_bounded():
    j=torch.tensor([[1.,2.],[1.,2.],[3.,-1.]],dtype=torch.float64)
    jw=torch.tensor([[2.**.5,2.*2.**.5],[3.,-1.]],dtype=torch.float64)
    p=torch.tensor([.4,.7],dtype=torch.float64)
    delta=torch.tensor([.2,-.3],dtype=torch.float64)
    assert torch.allclose(metric_correction(delta,p,j,.2),metric_correction(delta,p,jw,.2),atol=1e-12)


def test_linear_generator_matches_primal_solution_and_prior_keeps_adam():
    model=torch.nn.Linear(2,1,bias=False).double()
    z=torch.nn.Parameter(torch.tensor([[.2,.3]],dtype=torch.float64))
    opt=torch.optim.Adam([dict(params=list(model.parameters()),_comparison_prior=False),
                          dict(params=[z],_comparison_prior=True)],lr=.1,betas=(0.,.9))
    before=model.weight.detach().clone()
    controller=FunctionalMetric(output_step=.029);controller.register(model)
    loss=model(z).sum(); loss.backward()
    g=model.weight.grad.detach().flatten().clone(); zgrad=z.grad.detach().clone()
    p=.1/(g.abs()+1e-8)
    expected=torch.linalg.solve(torch.diag(1/p)+z.detach().T@z.detach()/.029,-g)
    rng=torch.get_rng_state().clone()
    controller.step(opt,torch.optim.Adam.step)
    assert torch.allclose(model.weight-before,expected.reshape(1,2),atol=1e-12)
    assert torch.allclose(z.detach(),torch.tensor([[.2,.3]],dtype=torch.float64)-.1*zgrad/(zgrad.abs()+1e-8))
    assert int(opt.state[model.weight]['step'])==1
    assert torch.equal(torch.get_rng_state(),rng)
    controller.close()
