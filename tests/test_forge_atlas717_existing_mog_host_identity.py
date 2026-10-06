"""ROOT-paid metadata-only host identity checks; no owner or scientific draw."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

from experiments.forge import api, atlas_existing_mog as bridge, hostprofiles
from experiments.forge.planning import candidate_revision_for


def fixture(*, bcap=False):
    candidate = dict(id='bcap' if bcap else bridge.CANDIDATE_ID,
        recipe_preset='bcap' if bcap else 'atlas', recipe_overrides={},
        prior=dict(kind='mog', sigma=.025, standardize=False, learnable=True),
        extensions={}, initializer='deterministic_orthogonal', execution_path='public_trainer')
    if not bcap:
        candidate['claim_contract'] = dict(experimental_track=bridge.TRACK_ID)
    context = api.FormulationContext(recipe_preset=candidate['recipe_preset'],
        prior=candidate['prior'], candidate_id=None if bcap else bridge.CANDIDATE_ID)
    candidate['resolved_recipe'] = asdict(context.recipe)
    source = dict(digest='a' * 64)
    return dict(candidate=candidate, source=source, view=dict(id=bridge.VIEW_ID),
                candidate_revision=candidate_revision_for(source['digest'], candidate))


class Atlas717ExistingMoGHostIdentity(unittest.TestCase):
    def test_host_identity_uses_exact_existing_track_metadata_and_full_recipe(self):
        fixed = fixture()
        observed = []
        original = api.FormulationContext
        def context(**kwargs):
            value = original(**kwargs)
            observed.append((kwargs, value))
            return value
        with patch.object(api, 'FormulationContext', side_effect=context):
            hostprofiles._validate_candidate_identity(fixed)
        self.assertEqual(len(observed), 1)
        kwargs, actual = observed[0]
        self.assertEqual(kwargs['candidate_id'], bridge.CANDIDATE_ID)
        self.assertEqual(asdict(actual.recipe), fixed['candidate']['resolved_recipe'])
        self.assertTrue(actual.recipe.particle_birth_death)
        self.assertTrue(actual.recipe.row_evidence_gate)
        self.assertEqual(actual.prior_config['kind'], 'mog')
        self.assertIs(actual.prior_config['standardize'], False)
        self.assertIsNone(actual._trainer)
        self.assertIsNone(actual._policy)
        self.assertIsNone(actual._host_prior)
        for field, mutate in [
            ('view', lambda r: r['view'].update(id='different_view')),
            ('track', lambda r: r['candidate']['claim_contract'].update(experimental_track='atlas717_noisy025')),
            ('recipe', lambda r: r['candidate']['resolved_recipe'].update(lr=.0085)),
            ('revision', lambda r: r.update(candidate_revision='b' * 64)),
        ]:
            value = deepcopy(fixed); mutate(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                hostprofiles._validate_candidate_identity(value)

    def test_bcap_retains_original_generic_identity(self):
        fixed = fixture(bcap=True)
        observed = []
        original = api.FormulationContext
        def context(**kwargs):
            value = original(**kwargs)
            observed.append((kwargs, value))
            return value
        with patch.object(api, 'FormulationContext', side_effect=context):
            hostprofiles._validate_candidate_identity(fixed)
        self.assertEqual(len(observed), 1)
        kwargs, actual = observed[0]
        self.assertIsNone(kwargs['candidate_id'])
        self.assertEqual(asdict(actual.recipe), fixed['candidate']['resolved_recipe'])
        self.assertEqual(actual.recipe.reg_arm, 'b_cap')
        self.assertEqual(actual.recipe.reg_coeff, 1.)
        self.assertFalse(actual.recipe.particle_birth_death)
        self.assertIsNone(actual._trainer)
        self.assertIsNone(actual._policy)


if __name__ == '__main__':
    unittest.main()
