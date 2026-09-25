"""The image experiment consumes public primitives without changing study settings."""
from pathlib import Path

import pytest
import torch

from experiments.config import read_config
from experiments.train_cifar_ddgan import DEFAULTS, training_recipe, resolve_config, validate
from particlegan import GANLoss, GradientPenalty, ParticleRegularizer, learning_rate_scale
from particlegan.diffusion import DrawSource


CONFIGS = sorted((Path(__file__).resolve().parents[1] / 'configs/cifar_ddgan').rglob('*.yaml'))


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: str(p.relative_to(p.parents[1])))
def test_existing_cifar_configs_resolve_into_recipe(path):
    cfg = resolve_config(read_config(path))
    validate(cfg)
    recipe = training_recipe(cfg)
    assert recipe.total_steps == cfg['steps']
    assert recipe.num_classes == cfg['classes']
    assert recipe.ucd_weight == cfg['ucd_lambda']
    assert recipe.ema_decay == cfg['ema']
    assert recipe.betas == (cfg['beta1'], .999)
    for key in ('lr', 'd_lr_mult', 'prior_lr_mult', 'prior_reg', 'z_dim', 'num_particles',
                'loss_type', 'gan_mode', 'reg_coeff', 'reg_kappa',
                'reg_every', 'reg_method', 'lr_anneal_start', 'lr_floor'):
        assert getattr(recipe, key) == cfg[key]
    assert isinstance(recipe.make_loss(), GANLoss)
    assert isinstance(recipe.make_gradient_penalty(), GradientPenalty)
    assert isinstance(recipe.make_prior_regularizer(), ParticleRegularizer)


@pytest.mark.parametrize('kind', ['learned', 'fixed', 'gaussian'])
def test_cifar_optimizer_factory_matches_historical_groups_and_updates(kind):
    import copy
    cfg = {**DEFAULTS, 'z_dim': 3, 'num_particles': 8, 'prior': kind, 'prior_reg': .3}
    recipe = training_recipe(cfg)
    g, d = torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)
    d.bias.requires_grad_(False)
    prior = DrawSource(kind, 8, 3, 101, 'cpu')
    old_g, old_d, old_prior = copy.deepcopy((g, d, prior))
    groups = [{'params': list(old_g.parameters()), 'lr': cfg['lr']}]
    if kind == 'learned':
        groups.append({'params': list(old_prior.parameters()), 'lr': cfg['lr'] * cfg['prior_lr_mult']})
    old_og = torch.optim.Adam(groups, betas=(cfg['beta1'], .999), fused=False)
    old_od = torch.optim.Adam((p for p in old_d.parameters() if p.requires_grad),
                              lr=cfg['lr'] * cfg['d_lr_mult'], betas=(cfg['beta1'], .999), fused=False)
    og, od = recipe.make_optimizers(g, d, prior, fused=False)
    assert og.state_dict()['param_groups'] == old_og.state_dict()['param_groups']
    assert od.state_dict()['param_groups'] == old_od.state_dict()['param_groups']
    for current, old in zip((g, d, prior), (old_g, old_d, old_prior)):
        for p, q in zip(current.parameters(), old.parameters()):
            if p.requires_grad:
                p.grad = torch.ones_like(p)
                q.grad = torch.ones_like(q)
    for opt in (og, od, old_og, old_od):
        opt.step()
    for current, old in zip((g, d, prior), (old_g, old_d, old_prior)):
        for p, q in zip(current.parameters(), old.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    z = torch.randn(6, 3, requires_grad=True)
    expected = cfg['prior_reg'] * ParticleRegularizer()(z)
    actual = recipe.make_prior_regularizer()(z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, z)[0], torch.autograd.grad(expected, z)[0], rtol=0, atol=0)


def test_cifar_lr_preserves_completed_update_convention():
    import math
    cfg = {**DEFAULTS, 'lr_floor': .05}
    for step in (1, 6000, 6001, 9999, 10000):
        frac = max(0, (step - 1 - cfg['lr_anneal_start'] * cfg['steps']) /
                   ((1 - cfg['lr_anneal_start']) * cfg['steps']))
        expected = cfg['lr_floor'] + (1 - cfg['lr_floor']) * .5 * (1 + math.cos(math.pi * frac))
        assert learning_rate_scale(step - 1, cfg['steps'], cfg['lr_anneal_start'], cfg['lr_floor']) == expected


def test_implicit_condition_cache_respects_backbone_and_explicit_overrides():
    pixel = resolve_config({'d_backbone': 'pixel'})
    assert pixel['cache_condition'] is False
    validate(pixel)
    assert resolve_config({})['cache_condition'] is True
    with pytest.raises(ValueError, match='condition cache'):
        validate(resolve_config({'d_backbone': 'pixel', 'cache_condition': True}))


def test_runner_resolves_historical_pixel_config_before_canonical_capture():
    from experiments.run_grid import load_config
    source = Path(__file__).resolve().parents[1] / 'configs/cifar_ddgan/smoke/width32.yaml'
    cfg = load_config(str(source), DEFAULTS)
    assert cfg['cache_condition'] is False
    assert resolve_config(cfg) == cfg
    validate(cfg)
