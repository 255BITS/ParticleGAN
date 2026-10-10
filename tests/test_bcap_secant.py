"""Software checks of secant algebra, ownership and public checkpoint replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.secant import SecantOptimizer
from experiments.forge.techniques import validate_same_technique
from experiments.forge.api import task_formulation_context
from benchmarks.toy_audit.api_images import WordFixture


ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_actual_normalized_length_curvature_and_recovery():
    p = nn.Parameter(torch.tensor([10.], dtype=torch.float64))
    opt = get_recipe('bcap', optimizer_secant_mode='bounded', optimizer_smoothing=0., lr=1.).make_generator_optimizer([p])
    assert isinstance(opt, SecantOptimizer)
    p.grad = torch.tensor([10.], dtype=p.dtype)
    opt.step()
    before = p.clone()
    p.grad = torch.tensor([1.], dtype=p.dtype)
    opt.step()
    # s=1, y=9, g=1; curvature fraction=1/18 falls below bounded floor.
    torch.testing.assert_close(before - p, torch.tensor([1/16], dtype=p.dtype), atol=1e-8, rtol=1e-8)
    assert opt.secant_stats['floor_hits'] == 1
    assert opt.secant_stats['history_uses'] == 1
    eta, theta = opt.state[p]['secant_eta'].clone(), opt.state[p]['secant_theta'].clone()
    before = p.clone()
    opt.step()  # zero y: only the slow-growth condition restricts recovery.
    expected = eta * (1 + theta).sqrt()
    torch.testing.assert_close(before - p, expected.expand_as(p), atol=1e-8, rtol=1e-8)
    assert opt.state[p]['secant_clock'] == 3


def test_nonfloor_secant_fraction_and_normalized_direction():
    p = nn.Parameter(torch.tensor([10., 0.], dtype=torch.float64))
    opt = get_recipe('bcap', optimizer_secant_mode='bounded', optimizer_smoothing=0., lr=1.).make_generator_optimizer([p])
    p.grad = torch.tensor([2., 0.], dtype=p.dtype); opt.step()
    before = p.clone()
    p.grad = torch.tensor([1., 0.], dtype=p.dtype); opt.step()
    torch.testing.assert_close(before - p, torch.tensor([.5, 0.], dtype=p.dtype), atol=1e-8, rtol=1e-8)
    assert opt.secant_stats['floor_hits'] == 0


def test_owned_row_history_and_exact_sparse_resume():
    recipe = get_recipe('bcap', optimizer_secant_mode='bounded', lr=.2)
    def build():
        p = nn.Parameter(torch.arange(6., dtype=torch.float64).reshape(3, 2))
        return p, recipe.make_generator_optimizer([p], latent_table=p)
    p, opt = build()
    p.grad = torch.ones_like(p); opt.set_sampled_rows(p, torch.tensor([0, 0])); opt.step()
    frozen = p[1:].clone()
    first = deepcopy(opt.state[p])
    p.grad.fill_(20); opt.set_sampled_rows(p, torch.tensor([2])); opt.step()
    equal(first['secant_x'][0], opt.state[p]['secant_x'][0])
    assert torch.equal(opt.state[p]['secant_clock'], torch.tensor([1, 0, 1]))
    assert torch.equal(p[1], frozen[0])
    saved, model = deepcopy(opt.state_dict()), p.clone()
    restored, resumed = build(); restored.data.copy_(model); resumed.load_state_dict(saved)
    for x, o in ((p, opt), (restored, resumed)):
        x.grad = torch.full_like(x, .1); o.set_sampled_rows(x, torch.tensor([0, 2])); o.step()
    equal(p, restored); equal(opt.state_dict(), resumed.state_dict())


def test_disabled_old_packets_and_structural_boundary():
    p = nn.Parameter(torch.zeros(2, 3)); recipe = get_recipe('bcap')
    opt = recipe.make_generator_optimizer([p])
    assert type(opt) is NormalizedOptimizer
    assert 'optimizer_secant_mode' not in recipe.to_dict()
    assert 'secant' not in opt.state_dict()
    opt.load_state_dict(deepcopy(opt.state_dict()))
    with pytest.raises(ValueError, match='mechanisms'):
        validate_same_technique(recipe, recipe.replace(optimizer_secant_mode='bounded'))


@pytest.mark.parametrize('delta', [dict(optimizer_secant_mode='bad'), dict(optimizer_momentum=.5),
    dict(optimizer_family='adam', optimizer_smoothing=0., optimizer_convolution='none'),
    dict(constraint_geometry_mode='direction_blend'), dict(critic_step_mode='finite_cap')])
def test_unsupported_compositions_rejected(delta):
    with pytest.raises(ValueError):
        get_recipe('bcap', **{**dict(optimizer_secant_mode='bounded'), **delta})


def test_nonfinite_input_rejected_before_mutation_and_bad_history():
    p = nn.Parameter(torch.ones(2)); opt = get_recipe('bcap', optimizer_secant_mode='bounded').make_generator_optimizer([p])
    saved = deepcopy(opt.state_dict()); p.grad = torch.full_like(p, float('nan'))
    with pytest.raises(ValueError, match='finite'):
        opt.step()
    equal(saved, opt.state_dict()); assert torch.equal(p, torch.ones(2))
    p.grad = torch.ones_like(p); opt.step(); saved = deepcopy(opt.state_dict())
    saved['state'][0]['secant_clock'].fill_(-1)
    with pytest.raises(ValueError, match='negative'):
        opt.load_state_dict(saved)


def test_zero_proposal_and_no_change_history_remain_finite():
    p = nn.Parameter(torch.ones(2))
    opt = get_recipe('bcap', optimizer_secant_mode='bounded').make_generator_optimizer([p])
    for _ in range(3):
        p.grad = torch.zeros_like(p); opt.step()
    assert torch.equal(p, torch.ones(2))
    assert opt.secant_stats['zero_proposals'] == 3
    assert opt.secant_stats['proposal_length_sum'] == opt.secant_stats['applied_length_sum'] == 0
    assert opt.secant_stats['history_uses'] == 2
    opt.validate_state_dict(opt.state_dict())


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_public_joint_word_exact_resume_and_streams(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    candidate = json.loads((ROOT / 'configs/forge/ideas/bcap-three-phase-incumbent-v1.json').read_text())
    candidate['recipe_overrides']['optimizer_secant_mode'] = 'bounded'
    task = json.loads((ROOT / 'configs/forge/tasks/five_word_joint_smoke.json').read_text())
    def build():
        context = task_formulation_context(candidate, task, {'seed': 0}, device=device, root=ROOT)
        fixture = WordFixture(device=device, seed=0, recipe_name=None, max_steps=20001, components=context)
        return context, fixture
    try:
        context, fixture = build()
        for _ in range(3): fixture.step()
        saved, named = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
        suffix = [fixture.step() for _ in range(2)]
        restored_context, restored = build()
        restored.policy.load_state_dict(saved['api_state'])
        restored.data_generator.set_state(saved['data_generator'])
        restored.restore_component_transport(saved.get('component_transport'))
        restored_context.streams.load_state_dict(named)
        equal(suffix, [restored.step() for _ in range(2)])
        equal(fixture.state_dict(), restored.state_dict())
        equal(context.streams.state_dict(), restored_context.streams.state_dict())
        assert fixture.opt_g.secant_stats['steps'] == 5
    finally:
        torch.set_num_threads(previous)
