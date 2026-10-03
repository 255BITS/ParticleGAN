"""Exercise the scoped global schedule bridge and frozen recipe boundary."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import MoGParticlePrior, get_recipe, learning_rate_scale
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
from benchmarks.transfer_suite import shared_schedule_search as search
from benchmarks.transfer_suite.vector_tasks import fixed_policy


PLAN = Path('reports/transfer_suite/unadjusted/runs/shared-schedule-search/screen_plan.json')


def test_only_global_schedule_fields_extend_shared_recipe():
    plan = json.loads(PLAN.read_text())
    jobs, recipes, profile = search.prepare(plan)
    assert [j['spec']['name'] for j in jobs] == ['vector_unequal_mass']
    assert len(recipes) == 4 and 'vector_unequal_mass' in profile
    for _, recipe in recipes:
        assert (recipe.lr, recipe.d_lr_mult, recipe.prior_lr_mult, recipe.reg_coeff) == (.00425, 1., 2., 6.)
    plan['candidates'][0]['overrides']['batch_size'] = 1
    with pytest.raises(ValueError):
        search.prepare(plan)


def test_mixed_ae_and_direct_prior_groups_receive_same_global_schedule():
    torch.set_num_threads(1)
    recipe = get_recipe(lr=.00425, d_lr_mult=1., prior_lr_mult=2.,
                        lr_anneal_start=.3, lr_floor=.01).replace(name='schedule_check')
    applied = []
    with search.configured_schedule(recipe) as receipt, optimizer_defaults(recipe, applied):
        prior = MoGParticlePrior(8, 2, sigma=.025)
        g, d = nn.Linear(2, 2), nn.Linear(2, 1)
        opt_g = torch.optim.Adam([{'params': list(g.parameters()) + list(prior.parameters()),
                                   'lr': .123, 'betas': (.5, .8)}], lr=.4, betas=(.6, .7))
        opt_d = torch.optim.Adam(d.parameters(), lr=.321, betas=(.5, .8))
        direct = nn.Parameter(torch.zeros(8, 2))
        opt_p = torch.optim.Adam([direct], lr=.123)
        controller = evaluate.FixedControl(fixed_policy('cosine'), 100)
        for step in (0, 40, 80):
            controller.step(opt_g, step, 'g')
            controller.step(opt_d, step, 'd')
            controller.step(opt_p, step, bridge.optimizer_role(opt_p, dict(opt_p=opt_p)))
            scale = learning_rate_scale(step, 100, .3, .01)
            assert [group['lr'] for group in opt_g.param_groups] == [recipe.lr*scale, recipe.lr*2*scale]
            assert opt_d.param_groups[0]['lr'] == recipe.lr*scale
            assert opt_p.param_groups[0]['lr'] == recipe.lr*2*scale
        assert [(group['role'], group['parameters']) for group in applied] == [
            ('g', 6), ('prior', 16), ('d', 3), ('prior', 16)]
        assert all(action['multiplier'] == learning_rate_scale(action['step'], 100, .3, .01)
                   for action in controller.trace)
    assert receipt['bridge_calls'] == 9
    assert receipt['bridge_total_steps'] == [100]
    assert bridge.learning_rate_scale is learning_rate_scale
