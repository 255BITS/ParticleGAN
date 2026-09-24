import pytest
import torch

from reports.toy100.joint_output_pullback import fit_output_targets


def model():
    generator = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        generator.weight.copy_(torch.eye(2))
    prior = torch.nn.Parameter(torch.eye(2))
    return generator, prior


def test_joint_nonlinear_map_lands_and_preserves_rng_and_gradients():
    g, z = model()
    g.weight.grad = torch.full_like(g.weight, 3.)
    z.grad = torch.full_like(z, 4.)
    before_rng = torch.get_rng_state().clone()
    target = torch.tensor([[1.1, 0.], [0., .9]])
    result = fit_output_targets(g, z, target)
    assert result['status'] == 'CONVERGED'
    assert result['final_max_row_error'] <= result['absolute_threshold']
    assert all(row['after'] < row['before'] for row in result['records'])
    assert not torch.equal(g.weight, torch.eye(2))
    assert not torch.equal(z, torch.eye(2))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(g.weight.grad, torch.full_like(g.weight, 3.))
    assert torch.equal(z.grad, torch.full_like(z, 4.))


def test_exact_match_rests_without_factorization(monkeypatch):
    g, z = model()
    before = [g.weight.detach().clone(), z.detach().clone()]
    monkeypatch.setattr(torch.linalg, 'svd', lambda *a, **kw: pytest.fail('zero residual factorized'))
    result = fit_output_targets(g, z, g(z).detach())
    assert result['status'] == 'CONVERGED'
    assert result['records'] == []
    assert torch.equal(g.weight, before[0]) and torch.equal(z, before[1])


def test_exception_restores_both_parameter_owners(monkeypatch):
    g, z = model()
    before = [g.weight.detach().clone(), z.detach().clone()]
    original = torch.linalg.svd
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert not torch.equal(g.weight, before[0])
            assert not torch.equal(z, before[1])
            raise RuntimeError('injected numerical failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, 'svd', fail_second)
    with pytest.raises(RuntimeError, match='injected numerical failure'):
        fit_output_targets(g, z, torch.tensor([[1.1, 0.], [0., .9]]))
    assert torch.equal(g.weight, before[0]) and torch.equal(z, before[1])


def test_zero_jacobian_unreachable_target_rests():
    g, z = model()
    with torch.no_grad():
        g.weight.zero_()
        z.zero_()
    result = fit_output_targets(g, z, torch.ones_like(z))
    assert result['status'] == 'NO_ACCEPTABLE_STEP'
    assert result['records'][0]['rank'] == 0
    assert result['initial_squared_error'] == result['final_squared_error']
    assert torch.equal(g.weight, torch.zeros_like(g.weight))
    assert torch.equal(z, torch.zeros_like(z))
