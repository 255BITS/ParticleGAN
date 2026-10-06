"""ROOT-paid metadata regression source; no owner, update, sample or grade."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.forge import atlas_existing_mog as bridge, promotion
from experiments.forge.api import FormulationContext

ROOT = Path(__file__).resolve().parents[1]
PRIOR = dict(kind='mog', sigma=.025, standardize=False, learnable=True)


def candidate():
    value = json.loads((ROOT / 'configs/forge/ideas' / (bridge.CANDIDATE_ID + '.json')).read_text())
    value['prior'] = deepcopy(PRIOR)
    return value


def request(declaration=None, view=None):
    declaration = candidate() if declaration is None else declaration
    # Reference manifest only: no public trainer, prior, owner, optimizer or draw.
    context = FormulationContext(recipe_preset=declaration['recipe_preset'],
        recipe_overrides=declaration.get('recipe_overrides', {}),
        prior=declaration.get('prior'), seed=0,
        candidate_id=bridge.CANDIDATE_ID if declaration['id'] == bridge.CANDIDATE_ID else None)
    protocol = dict(id='screening', seed=0)
    rng = context.streams.manifest()
    return dict(candidate=declaration, view=dict(id=bridge.VIEW_ID if view is None else view),
        protocol=protocol, rng=rng, tasks={},
        jobs=[dict(science=dict(seed=0, protocol=deepcopy(protocol), rng=deepcopy(rng)))])


class Atlas717ExistingMoGScreening(unittest.TestCase):
    def test_screening_forwards_only_the_existing_exact_track_context(self):
        fixed = request()
        observed = []
        def context(**kwargs):
            actual = FormulationContext(**kwargs)
            observed.append((kwargs, actual))
            return actual
        with patch.object(promotion, 'FormulationContext', side_effect=context):
            promotion.validate_screening_submission(fixed)
        self.assertEqual(len(observed), 1)
        kwargs, actual = observed[0]
        self.assertEqual(kwargs['candidate_id'], bridge.CANDIDATE_ID)
        self.assertEqual(actual.prior_config['kind'], 'mog')
        self.assertIs(actual.prior_config['standardize'], False)
        self.assertEqual(actual.prior_config['sigma'], .025)
        self.assertTrue(actual.recipe.particle_birth_death)
        self.assertTrue(actual.recipe.row_evidence_gate)
        self.assertIsNone(actual._trainer)
        self.assertIsNone(actual._policy)
        self.assertIsNone(actual._host_prior)
        self.assertEqual(fixed['protocol']['seed'], 0)

    def test_wrong_track_view_override_and_coupled_mog_remain_refused(self):
        base = request()
        changes = [
            ('candidate', lambda r: r['candidate'].update(id='atlas')),
            ('view', lambda r: r['view'].update(id='different_view')),
            ('track', lambda r: r['candidate']['claim_contract'].update(experimental_track='atlas717_noisy025')),
            ('override', lambda r: r['candidate'].update(recipe_overrides={'lr': .0085})),
            ('coupled', lambda r: r['candidate']['prior'].update(standardize=True)),
            ('seed', lambda r: r['protocol'].update(seed=1)),
        ]
        for name, mutate in changes:
            value = deepcopy(base); mutate(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                promotion.validate_screening_submission(value)

    def test_other_family_keeps_original_context_identity(self):
        fixed = request(dict(id='bcap', recipe_preset='bcap', recipe_overrides={}, prior=deepcopy(PRIOR)))
        observed = []
        def context(**kwargs):
            actual = FormulationContext(**kwargs)
            observed.append((kwargs, actual))
            return actual
        with patch.object(promotion, 'FormulationContext', side_effect=context):
            promotion.validate_screening_submission(fixed)
        self.assertEqual(len(observed), 1)
        kwargs, actual = observed[0]
        self.assertIsNone(kwargs['candidate_id'])
        self.assertFalse(actual.recipe.particle_birth_death)
        self.assertEqual(actual.recipe.reg_arm, 'b_cap')
        self.assertEqual(actual.recipe.reg_coeff, 1.)
        self.assertIsNone(actual._trainer)
        self.assertIsNone(actual._policy)


if __name__ == '__main__':
    unittest.main()
