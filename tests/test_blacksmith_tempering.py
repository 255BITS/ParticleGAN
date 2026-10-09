"""Evidence gain algebra, public factories and checkpointed sparse observation law."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.techniques import technique_signature


@pytest.mark.parametrize('device', ['cpu', 'cuda:1'])
def test_current_direction_with_conflicting_evidence(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    p = nn.Parameter(torch.zeros(2, dtype=torch.float64, device=device))
    opt = NormalizedOptimizer([p], lr=1., tempering=.95)
    p.grad = p.new_tensor([1., 0.]); opt.step()
    torch.testing.assert_close(p, p.new_tensor([-1., 0.]), atol=1e-8, rtol=0)
    p.grad = p.new_tensor([-1., 0.]); opt.step()
    # Equal opposing observations have ratio ((1-beta)/(1+beta))^2.
    gain = ((1 - .95) / (1 + .95)) ** 2
    torch.testing.assert_close(p, p.new_tensor([-1 + gain, 0.]), atol=1e-8, rtol=0)
    assert opt.state[p]['temper_count'].item() == 2
    before = p.clone(); p.grad = None; opt.step()
    assert torch.equal(p, before) and opt.state[p]['temper_count'].item() == 2
    p.grad = torch.zeros_like(p); opt.step()
    assert torch.equal(p, before) and opt.state[p]['temper_count'].item() == 3


def test_coherent_strikes_and_recovery_do_not_anneal_by_elapsed_time():
    p = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = NormalizedOptimizer([p], lr=1., tempering=.95)
    for _ in range(60):
        p.grad = p.new_tensor([1., 0.]); opt.step()
    torch.testing.assert_close(p[0], p.new_tensor(-60.), atol=1e-6, rtol=0)
    for _ in range(60):
        p.grad = p.new_tensor([-1., 0.]); opt.step()
    s = opt.state[p]
    gain = s['temper_mean'].square().sum() / (s['temper_square'].sum() * (1 - .95 ** 120))
    assert gain > .8


def test_row_observation_history_and_exact_checkpoint_replay():
    recipe = get_recipe('bcap', optimizer_tempering=.95)
    p = nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    opt = recipe.make_generator_optimizer([p], latent_table=p)
    p.grad = torch.ones_like(p); opt.set_sampled_rows(p, torch.tensor([0, 2])); opt.step()
    original = deepcopy(opt.state_dict())
    assert opt.state[p]['temper_count'].tolist() == [[1], [0], [1], [0]]
    assert torch.equal(p[[1, 3]], torch.zeros(2, 2, dtype=p.dtype))
    q = nn.Parameter(p.detach().clone()); resumed = recipe.make_generator_optimizer([q], latent_table=q)
    resumed.load_state_dict(original)
    assert resumed.state[q]['temper_count'].dtype == torch.long
    for table, optimizer in [(p, opt), (q, resumed)]:
        table.grad = -torch.ones_like(table)
        optimizer.set_sampled_rows(table, torch.tensor([2, 3])); optimizer.step()
    assert torch.equal(p, q)
    assert torch.equal(opt.state[p]['temper_count'], resumed.state[q]['temper_count'])
    assert opt.state[p]['temper_count'].tolist() == [[1], [0], [2], [1]]
    bad = deepcopy(original); next(iter(bad['state'].values()))['temper_count'][0] = -1
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad)
    mismatch = deepcopy(original); mismatch['dualnorm']['tempering']['decay'] = .9
    with pytest.raises(ValueError, match='tempering'):
        resumed.load_state_dict(mismatch)


def test_convolution_tempering_uses_same_scalar_for_all_offsets():
    recipe = get_recipe('bcap', optimizer_tempering=.95)
    model = nn.Conv2d(2, 2, 3, bias=False).double()
    baseline = get_recipe('bcap').make_generator_optimizer(model)
    tempered = recipe.make_generator_optimizer(deepcopy(model))
    p, q = baseline.param_groups[0]['params'][0], tempered.param_groups[0]['params'][0]
    gradient = torch.arange(1, p.numel()+1, dtype=p.dtype).reshape_as(p)
    before = p.clone()
    p.grad = gradient.clone(); q.grad = gradient.clone()
    baseline.step(); tempered.step()
    torch.testing.assert_close(p, q, atol=1e-12, rtol=0)
    displacement = before-p
    q.grad = -gradient; tempered.step()
    gain = ((1-.95)/(1+.95))**2
    torch.testing.assert_close(q, p+gain*displacement, atol=1e-12, rtol=0)


def test_disabled_packets_and_structural_binding():
    base = get_recipe('bcap'); candidate = get_recipe('bcap', optimizer_tempering=.95)
    assert 'optimizer_tempering' not in base.to_dict()
    assert recipe_field_owner('optimizer_tempering') == 'technique'
    assert technique_signature(base) != technique_signature(candidate)
    for decay in [-.1, 1., float('nan'), True]:
        with pytest.raises(ValueError):
            get_recipe('bcap', optimizer_tempering=decay)
    with pytest.raises(ValueError):
        get_recipe('bcap', optimizer_tempering=.95, optimizer_momentum=.5)
