"""Catch ignored optimizer options and changed test conditions in the comparison."""
from benchmarks.locked_shared.recorded_recipes import GAN_V1, GAN_V2
import pytest
import torch
from torch import nn

from particlegan import MoGParticlePrior, get_recipe, learning_rate_scale
from benchmarks import learned_lr_evaluation as bridge
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite.compare_defaults import effective_spec, optimizer_defaults, plan
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.vector_tasks import fixed_policy


@pytest.mark.parametrize('recipe', [GAN_V2, GAN_V1])
def test_recipe_replaces_mixed_group_rates_and_explicit_ae_betas(recipe):
    applied = []
    with optimizer_defaults(recipe, applied):
        prior = MoGParticlePrior(8, 2)
        g, d = nn.Linear(2, 2), nn.Linear(2, 1)
        opt_g = torch.optim.Adam([{'params': list(g.parameters()) + list(prior.parameters()),
                                   'lr': .123, 'betas': (.5, .8)}], lr=.4, betas=(.6, .7))
        opt_d = torch.optim.Adam(d.parameters(), lr=.321, betas=(.5, .8))
        controller = evaluate.FixedControl(fixed_policy(), 100)
        for step in (0, 80):
            controller.step(opt_g, step, 'g')
            controller.step(opt_d, step, 'd')
            scale = learning_rate_scale(step, 100, .6, .05)
            assert [group['lr'] for group in opt_g.param_groups] == [recipe.lr * scale, recipe.lr * recipe.prior_lr_mult * scale]
            assert opt_d.param_groups[0]['lr'] == recipe.lr * recipe.d_lr_mult * scale
            assert all(group['betas'] == recipe.betas for opt in (opt_g, opt_d) for group in opt.param_groups)
        assert [(r['role'], r['parameters']) for r in applied] == [('g', 6), ('prior', 16), ('d', 3)]


@pytest.mark.parametrize('recipe', [GAN_V2, GAN_V1])
def test_direct_particle_optimizer_uses_prior_rate(recipe):
    applied = []
    with optimizer_defaults(recipe, applied):
        points = nn.Parameter(torch.zeros(8, 2))
        opt_p = torch.optim.Adam([points], lr=.123)
        role = bridge.optimizer_role(opt_p, dict(opt_p=opt_p))
        controller = evaluate.FixedControl(fixed_policy(), 100)
        controller.step(opt_p, 0, role)
        assert opt_p.param_groups[0]['lr'] == recipe.lr * recipe.prior_lr_mult
        assert applied[0]['role'] == 'prior'


def test_both_defaults_keep_all_nineteen_targets_architectures_and_resources():
    jobs = plan()
    assert len(jobs) == 19
    assert sum(j['spec']['runner'] == 'legacy' for j in jobs) == 9
    for job in jobs:
        old, new = [effective_spec(job['spec'], recipe) for recipe in (GAN_V1, GAN_V2)]
        if old['runner'] == 'legacy':
            assert old == new
        else:
            a, b = axes(old, old['runner']), axes(new, new['runner'])
            for key in ('target', 'architecture', 'resources'):
                assert a[key] == b[key]
