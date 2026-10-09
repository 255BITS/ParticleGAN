"""Analytic geometry, ownership and checkpoint tests; no training evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particlegan import get_recipe
from particlegan.optim.constraint_geometry import (
    ConstraintGeometryOptimizer, constraint_geometry_backward, project_nonascent,
)


def test_intersecting_and_opposed_halfspaces():
    d = torch.tensor([2., 1.], dtype=torch.float64)
    a = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(project_nonascent(d, a), d.new_tensor([0., 1.]))
    a = d.new_tensor([[1., 0.], [1., 1.]])
    out = project_nonascent(d, a)
    torch.testing.assert_close(out, d.new_tensor([0., 0.]))
    assert bool((a @ out <= 1e-10).all())
    assert torch.equal(project_nonascent(-d, a), -d)


def test_post_normalization_protection_preserves_unsampled_rows_and_pending_state():
    table = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    vector = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = ConstraintGeometryOptimizer([dict(params=[table], role="prior"),
                                       dict(params=[vector], role="generator")], lr=.1, smoothing=.001)
    opt.set_sampled_rows(table, torch.tensor([0, 2]))
    protected = table.sum() + vector.sum()
    constraint_geometry_backward(-protected, opt, (protected,))
    pending = deepcopy(opt.state_dict())
    restored = ConstraintGeometryOptimizer([dict(params=[table], role="prior"),
                                            dict(params=[vector], role="generator")], lr=.1, smoothing=.001)
    restored.load_state_dict(pending)
    restored.step()
    assert torch.equal(table[1], torch.zeros(2, dtype=torch.float64))
    assert float((table.sum() + vector.sum()).detach()) <= 1e-12
    assert restored.constraint_geometry_stats["projected_steps"] == 1
    assert restored.constraint_geometry_stats["max_derivative_before"] > .1
    assert restored.constraint_geometry_stats["max_derivative_after"] <= 1e-12
    with pytest.raises(ValueError, match="explicit protected"):
        restored.step()
    bad = deepcopy(restored.state_dict())
    bad["constraint_geometry"]["mode"] = "other"
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(bad)


def test_disabled_backward_and_exact_winner_resolution():
    p = torch.nn.Parameter(torch.tensor([.3, .4], dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=.1)
    loss = p.square().sum()
    constraint_geometry_backward(loss, opt, (loss,))
    torch.testing.assert_close(p.grad, p.detach() * 2)
    root = Path(__file__).resolve().parents[1]
    original = json.loads(next((root / "configs/forge/configurations").glob("bcap-dualnorm--5b1ef*.json")).read_text())
    control = json.loads((root / "configs/forge/ideas/constraint_geometry-control-v1.json").read_text())
    recipe = get_recipe(control["recipe_preset"], **control["recipe_overrides"])
    from dataclasses import asdict
    resolved = json.loads(json.dumps(asdict(recipe)))
    resolved.pop("constraint_geometry_mode")
    expected = dict(original["resolved_configuration_recipe"])
    # Public task binding supplies prior implementation/resources in both arms.
    for key in ("prior_kind", "sigma_rel", "standardize", "total_steps", "batch_size", "num_particles", "z_dim"):
        resolved.pop(key, None)
        expected.pop(key, None)
    assert resolved == expected


def test_recipe_rejects_incompatible_update():
    with pytest.raises(ValueError, match="zero-momentum"):
        get_recipe("bcap", optimizer_momentum=.5, constraint_geometry_mode="nonascent")


def test_inactive_projection_preserves_already_applied_step_bitwise():
    from particlegan.optim.dualnorm import NormalizedOptimizer
    old = torch.tensor([.1234567], dtype=torch.float32)
    plain = torch.nn.Parameter(old.clone())
    guarded = torch.nn.Parameter(old.clone())
    options = dict(lr=.12345676, smoothing=.001)
    baseline = NormalizedOptimizer([plain], **options)
    candidate = ConstraintGeometryOptimizer([guarded], **options)
    plain.sum().backward()
    baseline.step()
    loss = guarded.sum()
    constraint_geometry_backward(loss, candidate, (loss,))
    candidate.step()
    # The fixture deliberately crosses zero. Recomposition destroys a real
    # low-order result, so ordinary allclose would miss this regression.
    assert not torch.equal(old + (plain.detach() - old), plain.detach())
    assert torch.equal(guarded.detach(), plain.detach())
    assert candidate.constraint_geometry_stats['projected_steps'] == 0
    legacy = deepcopy(candidate.state_dict())
    legacy['constraint_geometry']['schema'] = 1
    with pytest.raises(ValueError, match='checkpoint'):
        candidate.load_state_dict(legacy)


def _assert_bitwise_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_bitwise_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_bitwise_state(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['nonascent', 'direction_blend', 'strict_progress'])
def test_extra_backward_and_inactive_checkpoint_resume_parity(dtype, device, mode):
    """Software fixture: public initialization, sampled ownership, no training gate."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from particlegan.init import deterministic_orthogonal_
    from particlegan.optim.dualnorm import NormalizedOptimizer
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    from particlegan.optim.direction_blend import DirectionBlendOptimizer
    import random
    import numpy as np

    with torch.random.fork_rng(devices=[]):
        network = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))
    deterministic_orthogonal_(network, seed=0)
    network = network.to(device=device, dtype=dtype)
    networks = [network, deepcopy(network)]
    tables = [torch.nn.Parameter(torch.arange(10, device=device, dtype=dtype).reshape(5, 2) / 10)
              for _ in range(2)]
    parameters = [list(net.parameters()) + [table] for net, table in zip(networks, tables)]
    options = dict(lr=.012, smoothing=.001, convolution='per_offset')
    groups = [dict(params=list(networks[0].parameters()), role='generator'),
              dict(params=[tables[0]], role='prior', lr=.030)]
    plain = NormalizedOptimizer(groups, **options)
    optimizer_type = {'nonascent': ConstraintGeometryOptimizer, 'direction_blend': DirectionBlendOptimizer,
                      'strict_progress': StrictProgressOptimizer}[mode]
    guarded = optimizer_type([
        dict(params=list(networks[1].parameters()), role='generator'),
        dict(params=[tables[1]], role='prior', lr=.030)], **options)
    rows = torch.tensor([0, 2, 2, 4], device=device)

    def rng_state():
        return dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()),
                    cpu=torch.get_rng_state().clone(),
                    cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if device == 'cuda' else [])

    def assert_rng(a, b):
        left, right = a.pop('numpy'), b.pop('numpy')
        assert np.array_equal(left[1], right[1])
        _assert_bitwise_state((left[0], *left[2:]), (right[0], *right[2:]))
        _assert_bitwise_state(a, b)

    # Two optimizer steps straddle a checkpoint round trip. No random samples,
    # alternate seed, task grading, or private trainer are involved.
    for index in range(2):
        for opt, table in zip((plain, guarded), tables):
            opt.zero_grad(set_to_none=False)
            opt.set_sampled_rows(table, rows)
        losses = [net(table[rows]).square().mean() for net, table in zip(networks, tables)]
        before_grad = [None if p.grad is None else p.grad.clone() for p in parameters[1]]
        before_rng = rng_state()
        guarded.bind_protected_losses((losses[1],), protected_evaluator=lambda:
                                     (networks[1](tables[1][rows]).square().mean(),))
        assert_rng(before_rng, rng_state())
        for p, old_grad in zip(parameters[1], before_grad):
            if old_grad is None:
                assert p.grad is None
            else:
                assert torch.equal(p.grad, old_grad)
        for loss in losses:
            loss.backward()
        for left, right in zip(*parameters):
            assert torch.equal(left.grad, right.grad)
        before_rng = rng_state()
        plain.step()
        guarded.step()
        assert_rng(before_rng, rng_state())
        assert guarded.constraint_geometry_stats['projected_steps'] == 0
        for left, right in zip(*parameters):
            assert torch.equal(left, right)
        saved = deepcopy(guarded.state_dict())
        base_saved = deepcopy(saved)
        base_saved.pop('constraint_geometry')
        base_saved.pop('strict_progress', None)
        base_saved.pop('direction_blend', None)
        _assert_bitwise_state(plain.state_dict(), base_saved)
        if index == 0:
            guarded.load_state_dict(saved)
            plain.load_state_dict(deepcopy(plain.state_dict()))
            _assert_bitwise_state(guarded.state_dict(), saved)


def test_strict_progress_realizes_descent_and_backtracks_nonlinear_overshoot():
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    p = torch.nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    opt = StrictProgressOptimizer([p], lr=1., smoothing=.001)
    protected = p.square().sum()
    old = float(protected.detach())
    constraint_geometry_backward(-p.sum(), opt, (protected,),
                                 protected_evaluator=lambda: (p.square().sum(),))
    opt.step()
    assert float(p.square().sum().detach()) < old
    assert opt.strict_progress_stats['accepted_steps'] == 1
    assert opt.strict_progress_stats['backtracks'] == 2
    assert opt.strict_progress_stats['min_accepted_scale'] == .25
    assert opt.strict_progress_stats['max_accepted_armijo_violation'] == 0
    assert opt.constraint_geometry_stats['steps'] == 1
    assert opt.strict_progress_stats['max_retained_norm_ratio'] <= 1


def test_strict_progress_opposition_and_pending_checkpoint_contract():
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    opt = StrictProgressOptimizer([p], lr=.1, smoothing=.001)
    evaluator = lambda: (p.sum(), -p.sum())
    constraint_geometry_backward(-p.sum(), opt, evaluator(), protected_evaluator=evaluator)
    state = deepcopy(opt.state_dict())
    restored = StrictProgressOptimizer([p], lr=.1, smoothing=.001)
    restored.load_state_dict(state)
    with pytest.raises(ValueError, match='evaluator'):
        restored.step()
    assert torch.equal(p, torch.tensor([1.], dtype=torch.float64))
    restored.bind_protected_evaluator(evaluator)
    restored.step()
    assert torch.equal(p, torch.tensor([1.], dtype=torch.float64))
    assert restored.strict_progress_stats['pareto_stalls'] == 1
    assert restored.strict_progress_stats['rejected_steps'] == 1
    assert restored.strict_progress_stats['probes'] == 0
    assert restored.constraint_geometry_stats['steps'] == 1
    bad = deepcopy(restored.state_dict())
    bad['strict_progress']['stats']['accepted_steps'] = -1
    with pytest.raises(ValueError, match='counter'):
        restored.load_state_dict(bad)


def test_strict_progress_inactive_cancellation_and_callback_rng_fail_closed():
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    from particlegan.optim.dualnorm import NormalizedOptimizer
    p = torch.nn.Parameter(torch.tensor([.1234567]))
    plain = torch.nn.Parameter(p.detach().clone())
    options = dict(lr=.12345676, smoothing=.001)
    opt = StrictProgressOptimizer([p], **options)
    baseline = NormalizedOptimizer([plain], **options)
    calls = []
    def evaluator():
        calls.append(True)
        return (p.sum(),)
    loss = p.sum()
    constraint_geometry_backward(loss, opt, (loss,), protected_evaluator=evaluator)
    plain.sum().backward()
    baseline.step()
    opt.step()
    assert torch.equal(p, plain) and not calls
    opt.zero_grad()
    loss = p.sum()
    def stochastic_evaluator():
        return (p.sum() + torch.rand(()),)
    constraint_geometry_backward(-loss, opt, (loss,), protected_evaluator=stochastic_evaluator)
    before, rng = p.detach().clone(), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='ambient RNG'):
        opt.step()
    assert torch.equal(p, before) and torch.equal(rng, torch.get_rng_state())


def test_strict_progress_public_recipe_and_deterministic_replay_restriction():
    from particlegan.optim.strict_progress import StrictProgressOptimizer
    from particlegan import GANTrainer
    recipe = get_recipe('bcap', optimizer_family='dualnorm', constraint_geometry_mode='strict_progress')
    module = torch.nn.Linear(2, 1)
    assert isinstance(recipe.make_generator_optimizer(module), StrictProgressOptimizer)
    with pytest.raises(ValueError, match='deterministic replay'):
        GANTrainer(get_recipe('bcap', optimizer_family='dualnorm', constraint_geometry_mode='strict_progress',
                              standardize=False, output_noise_std=.1), module, torch.nn.Linear(1, 1))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_direction_blend_full_scale_can_overshoot_without_evaluator(device, dtype):
    from particlegan.optim.direction_blend import DirectionBlendOptimizer
    from particlegan.optim.strict_progress import common_descent
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    p = torch.nn.Parameter(torch.tensor([.1], device=device, dtype=dtype))
    opt = DirectionBlendOptimizer([p], lr=1., smoothing=.001)
    protected = p.square().sum()
    old = p.detach().clone()
    constraint_geometry_backward(-p.sum(), opt, (protected,),
        protected_evaluator=lambda: pytest.fail('direction-only arm called evaluator'))
    normals = opt._protected.clone()
    pending = deepcopy(opt.state_dict())
    opt.load_state_dict(pending)
    opt.step()
    from particlegan.optim.dualnorm import NormalizedOptimizer
    base_p = torch.nn.Parameter(old.clone())
    base = NormalizedOptimizer([base_p], lr=1., smoothing=.001)
    (-base_p.sum()).backward()
    base.step()
    expected, possible = common_descent(base_p.detach()-old, normals)
    assert possible and torch.equal(p.detach(), old+expected)
    assert float(p.square().sum().detach()) > float(protected.detach())
    assert float(normals.double() @ (p.detach()-old).double()) < 0
    assert opt.direction_blend_stats['blended_steps'] == 1
    opt.load_state_dict(deepcopy(opt.state_dict()))
    bad = deepcopy(opt.state_dict())
    bad['direction_blend']['stats']['conflict_steps'] = -1
    with pytest.raises(ValueError, match='counter'):
        opt.load_state_dict(bad)


def test_direction_blend_public_recipe():
    from particlegan.optim.direction_blend import DirectionBlendOptimizer
    recipe = get_recipe('bcap', optimizer_family='dualnorm', constraint_geometry_mode='direction_blend')
    opt = recipe.make_generator_optimizer(torch.nn.Linear(2, 2))
    assert isinstance(opt, DirectionBlendOptimizer)
