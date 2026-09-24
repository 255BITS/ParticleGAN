import math

import torch

from reports.toy100.convex_readout_value import ReadoutProblem, fit_readout


def problem(delta):
    dimension = delta.shape[1]
    jac = torch.eye(dimension, dtype=torch.float64).expand(2*len(delta), dimension, dimension).clone()
    return ReadoutProblem(delta, jac)


def test_readout_hessian_is_psd_and_bias_is_an_exact_gauge():
    p = problem(torch.tensor([[1., -.5], [-.8, 1.], [.1, -.3]], dtype=torch.float64))
    w = torch.tensor([2., -.7], dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(p.loss, w)
    assert torch.linalg.eigvalsh(hessian).min() >= -1e-12
    real = torch.tensor([[.2, -.1], [.7, .4], [.8, .5]], dtype=torch.float64)
    fake = real+p.delta
    bias = torch.tensor(13., dtype=torch.float64, requires_grad=True)
    scores = (fake@w+bias)-(real@w+bias)
    loss = torch.nn.functional.softplus(scores).mean()
    assert torch.autograd.grad(loss, bias)[0] == 0
    assert torch.allclose(scores, p.delta@w, atol=2e-15, rtol=0)


def test_global_gap_bound_contains_known_optimum_and_solver_rests():
    p = problem(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float64))
    w = torch.tensor([1.8, -1.4], dtype=torch.float64, requires_grad=True)
    value = p.loss(w)
    gradient = torch.autograd.grad(value, w)[0]
    bound = p.certificate(w.detach(), gradient, float(value.detach()), math.log(2.))
    assert bound['lower'] <= math.log(2.) <= bound['upper']
    final, receipt = fit_readout(p, w.detach())
    assert receipt['status'] == 'CERTIFIED_GAP'
    assert receipt['certificate']['gap'] <= 1e-7
    assert final.norm() < 1e-3
    zero, rest = fit_readout(p, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(zero, torch.zeros_like(zero))
    assert rest['closure_calls'] == 1 and rest['certificate']['gap'] < 1e-15


def test_profiled_value_improves_toward_matched_mean_with_verified_bounds():
    real = torch.tensor([-1., 1.], dtype=torch.float64)
    old = problem((.5-real).unsqueeze(1))
    w, base = fit_readout(old, torch.zeros(1, dtype=torch.float64))
    theta = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    generator_value = -torch.nn.functional.softplus((theta-real)*w[0]).mean()
    gradient = torch.autograd.grad(generator_value, theta)[0]
    assert gradient > 0  # fixed metric descent reduces the mean discrepancy
    new = problem((.4-real).unsqueeze(1))
    _, trial = fit_readout(new, w)
    assert base['status'] == trial['status'] == 'CERTIFIED_GAP'
    # V=log2-min D loss, so the new D-loss LOWER bound must exceed
    # the old evaluated UPPER bound for a certified V decrease.
    assert trial['certificate']['lower'] > base['certificate']['upper']


def test_rank_deficiency_never_receives_a_false_global_certificate():
    delta = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    p = ReadoutProblem(delta, torch.zeros(4, 2, 2, dtype=torch.float64))
    certificate = p.certificate(torch.zeros(2), torch.zeros(2), math.log(2.), math.log(2.))
    assert not certificate['available']


def test_disjoint_bounds_verify_value_decrease_without_exact_inner_minima():
    real = torch.tensor([-1., 1.], dtype=torch.float64)
    bounds = []
    for theta in (.5, .4):
        p = problem((theta-real).unsqueeze(1))
        optimum, _ = fit_readout(p, torch.zeros(1, dtype=torch.float64))
        approximate = (optimum+.001).requires_grad_(True)
        loss = p.loss(approximate)
        gradient = torch.autograd.grad(loss, approximate)[0]
        bound = p.certificate(approximate.detach(), gradient, float(loss.detach()), float(loss.detach()))
        assert bound['gap'] > 1e-7  # deliberately misses the solver target
        bounds.append(bound)
    assert bounds[1]['lower'] > bounds[0]['upper']


def test_v2_changes_only_the_declared_certificate_gate_sites():
    from reports.toy100.pr84_convex_profiled_value1530_v2 import transformed_main
    _, source = transformed_main()  # includes five exact source-shape assertions
    compile(source, '<convex-value-gate-test>', 'exec')
    assert "base_fit['certificate']['available'] and fd_pass" in source
    assert "verified = bound is not None and bound > required + 1e-12" in source
