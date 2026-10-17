"""Check profile boundaries and actual global optimizer application."""
from copy import deepcopy

import pytest
import torch

from particlegan import get_recipe
from benchmarks.transfer_suite import shared_discriminator_search as canonical
from benchmarks.transfer_suite import shared_profile_search as profile
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.shared_pointnorm_research import ARCHITECTURES
from particlegan import learning_rate_scale
from reports.transfer_suite.unadjusted.build import validate_schedule
from benchmarks.gan_v3 import gan_v3_recipe, legacy_dict


def declaration():
    return dict(candidates=[dict(name='profile_check', overrides=dict(lr=.002))],
                discriminators={'vector_unequal_mass': deepcopy(ARCHITECTURES[0])})


def test_rejects_per_host_recipe_fields():
    declared = declaration()
    declared['task_overrides'] = {'vector_unequal_mass': {'lr': .001}}
    with pytest.raises(ValueError):
        profile.prepare(declared)


def test_rejects_generator_or_nonvector_profile():
    declared = declaration()
    declared['discriminators']['mode_hold'] = declared['discriminators'].pop('vector_unequal_mass')
    with pytest.raises(ValueError):
        profile.prepare(declared)


def test_rejects_unreviewed_implementation():
    declared = declaration()
    declared['discriminators']['vector_unequal_mass']['implementation'] = 'unknown_module'
    with pytest.raises(KeyError):
        profile.prepare(declared)


def test_profile_applies_global_recipe_and_restores_runner():
    torch.set_num_threads(1)
    job = deepcopy(next(j for j in plan() if j['spec']['name'] == 'vector_unequal_mass'))
    # Short integration fixture only; this is not a scored benchmark episode.
    job['spec']['steps'] = 24
    recipe = gan_v3_recipe(lr=.002, d_lr_mult=.75, prior_lr_mult=3., betas=(.1, .95),
                        reg_coeff=4.).replace(name='profile_check')
    original_recipe, original_constructor = canonical.recipe, canonical.constructor
    result = profile.episode(job, recipe, ARCHITECTURES[0])
    assert not result['result'].get('error')
    assert len(result['result']['observations']) == 24
    assert result['recipe'] == legacy_dict(recipe)
    assert result['original_spec'] == job['spec']
    assert {r['role'] for r in result['applied']} == {'g', 'd', 'prior'}
    for group in result['applied']:
        assert group['lr'] == recipe.lr * {'g': 1., 'd': .75, 'prior': 3.}[group['role']]
        assert tuple(group['betas']) == recipe.betas
    assert canonical.recipe is original_recipe
    assert canonical.constructor is original_constructor


def test_schedule_receipts_must_match_declared_global_recipe():
    recipe = gan_v3_recipe(lr_anneal_start=.3, lr_floor=.01)
    payload = dict(spec={'steps': 1200}, result={'actions': [
        dict(step=1000, role='g', multiplier=learning_rate_scale(1000, 1200, .3, .01))]})
    payload['schedule'] = dict(kind='cosine', hold=.3, floor=.01, bridge_calls=2400)
    validate_schedule(payload, recipe)
    payload['schedule']['hold'] = .6
    with pytest.raises(AssertionError, match='Schedule declaration differs from recipe'):
        validate_schedule(payload, recipe)
    payload['schedule']['hold'] = .3
    payload['result']['actions'][0]['multiplier'] = learning_rate_scale(1000, 1200, .6, .05)
    with pytest.raises(AssertionError, match='Schedule action contradicts recipe'):
        validate_schedule(payload, recipe)
