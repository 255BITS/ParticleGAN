"""Focused metadata controls. Authored only; no numerical/model trial."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from experiments.forge import atlas_existing_mog as owner
from experiments.forge.atlas_existing_mog_ae_probe import ADAPTER_SHA256

ROOT = Path(__file__).resolve().parents[1]


def candidate():
    return dict(id=owner.CANDIDATE_ID, recipe_preset='atlas', recipe_overrides={},
        claim_contract=dict(experimental_track=owner.TRACK_ID, sampling_law='task_declared',
                            schedule='schedule_free', scoring_weights='live'))


def task(name):
    return json.loads((ROOT / 'configs/forge/tasks' / (name + '.json')).read_text())


def prior():
    return dict(sigma=0.02500000037252903, sigma_dtype='torch.float32')


def reaction(step=1000, backend='knn'):
    fake = dict(kind='mog', code_path='particlegan.particle_prior.MoGParticlePrior',
        sigma=prior()['sigma'], sigma_dtype='torch.float32', sigma_units='raw_latent_coordinates',
        standardize=False, row_weights='uniform', same_table_parameter=True,
        sampler='original_public_sample_then_existing_dv12_and_output_noise', stream='private_birth_death')
    return dict(schema='forge_atlas791_mog_nearest_positive_v1', candidate_id=owner.CANDIDATE_ID,
        completed_steps=step, core_pins={name: deepcopy(owner.CORE_PINS[name]) for name in
            ('particlegan/birth_death.py', 'particlegan/policy.py')},
        birth_death_type='particlegan.birth_death.ParticleBirthDeath', actual_backend=backend,
        reaction_prior_alias=True, reaction_table_alias=True, fake_pool_prior=fake,
        state_config_fake_pool_prior=deepcopy(fake), sampling_calls_added=0)


class Atlas791NearestPositiveMetadataTests(unittest.TestCase):
    def test_only_corrected_candidate_and_hashed_marker(self):
        self.assertTrue(owner.supports_candidate(candidate()))
        for identifier in ('atlas-existing-mog-tier1-717-v1', 'atlas-noisy025-tier1-717-v1', 'atlas-existing-mog-kernel760-v1', 'atlas'):
            wrong = candidate(); wrong['id'] = identifier
            self.assertFalse(owner.supports_candidate(wrong))
        wrong = candidate(); wrong['claim_contract']['experimental_track'] = 'atlas717_existing_mog'
        self.assertFalse(owner.supports_candidate(wrong))

    def test_all_original_six_payloads_caps_and_true_blockers(self):
        self.assertEqual(set(owner.PARENTS), {'gaussian1d_acquisition', 'two_pole', 'unused_token_hold',
            'ae_gan_hold', 'ring16_acquisition', 'five_word_joint_acquisition'})
        self.assertEqual(sum(owner.ALLOWANCES.values()), 2220)
        self.assertEqual(set(owner.SUPPORTED), {'gaussian1d_acquisition', 'two_pole', 'ae_gan_hold', 'ring16_acquisition'})
        self.assertEqual(set(owner.BLOCKED), {'unused_token_hold', 'five_word_joint_acquisition'})
        for name in owner.PARENTS:
            with self.subTest(task=name):
                self.assertEqual(owner.validate(task(name))['task_id'], name)
                raw = (ROOT / 'configs/forge/tasks' / (name + '.json')).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), owner.PARENTS[name]['raw_sha256'])
                self.assertEqual(task(name)['resources']['timeout_seconds'], owner.ALLOWANCES[name])

    def test_full79_recipe_and_original_prior_bindings(self):
        for name, fields in owner.EXPECTED_RECIPES.items():
            with self.subTest(task=name):
                self.assertEqual(len(fields), 79)
                self.assertIsNone(fields['total_steps'])
                self.assertIs(fields['standardize'], False)
                self.assertEqual(fields['loss'], 'relativistic')
                self.assertEqual(fields['prior_kind'], 'particles' if name == 'two_pole' else 'mog')
        changed = task('gaussian1d_acquisition'); changed['execution']['prior']['sigma'] = .03
        with self.assertRaises(ValueError): owner.validate(changed)

    def test_exact_float32_public_kernel_receipt_is_metadata_only(self):
        good = reaction()
        before = deepcopy(good)
        self.assertIsNone(owner.validate_reaction_kernel_receipt(good, prior(), completed_steps=1000))
        self.assertEqual(good, before)
        self.assertEqual(good['sampling_calls_added'], 0)

    def test_feature_cells_and_missing_kernel_are_refused(self):
        for wrong in (reaction(backend='feature_cells'), {k: v for k, v in reaction().items() if k != 'fake_pool_prior'}):
            with self.subTest(receipt=wrong):
                with self.assertRaises(ValueError):
                    owner.validate_reaction_kernel_receipt(wrong, prior(), completed_steps=1000)

    def test_center_only_or_foreign_table_and_stream_are_refused(self):
        for field, value in (('sigma', 0.0), ('sigma', .03), ('same_table_parameter', False),
                             ('sampler', 'table_centres'), ('stream', 'noise/prior/gaussian')):
            wrong = reaction(); wrong['fake_pool_prior'][field] = value
            wrong['state_config_fake_pool_prior'] = deepcopy(wrong['fake_pool_prior'])
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    owner.validate_reaction_kernel_receipt(wrong, prior(), completed_steps=1000)
        for field in ('reaction_prior_alias', 'reaction_table_alias'):
            wrong = reaction(); wrong[field] = False
            with self.assertRaises(ValueError):
                owner.validate_reaction_kernel_receipt(wrong, prior(), completed_steps=1000)

    def test_original_core_checkpoint_or_clock_cannot_be_rebound(self):
        for mutator in ('core', 'config', 'clock', 'candidate'):
            wrong = reaction()
            if mutator == 'core': wrong['core_pins']['particlegan/birth_death.py']['sha256'] = 'b14c50c611a8cf188347e391739fca50a5171200eb6be90e4e3d1509d64e47ed'
            if mutator == 'config': wrong['state_config_fake_pool_prior'].pop('sampler')
            if mutator == 'clock': wrong['completed_steps'] = True
            if mutator == 'candidate': wrong['candidate_id'] = 'atlas-existing-mog-tier1-717-v1'
            with self.subTest(mutator=mutator):
                with self.assertRaises(ValueError):
                    owner.validate_reaction_kernel_receipt(wrong, prior(), completed_steps=1000)

    def test_pending_backend_is_only_zero_clock_construction(self):
        self.assertIsNone(owner.validate_reaction_kernel_receipt(reaction(0, 'pending'), prior(), completed_steps=0))
        with self.assertRaises(ValueError):
            owner.validate_reaction_kernel_receipt(reaction(1, 'pending'), prior(), completed_steps=1)

    def test_owned_probe_tracks_current_adapter_bytes(self):
        actual = hashlib.sha256((ROOT / 'experiments/forge/atlas_existing_mog_ae.py').read_bytes()).hexdigest()
        self.assertEqual(ADAPTER_SHA256, actual)

    def test_saved_direct_gradient_covers_real_and_particle_rows(self):
        from experiments.forge.atlas717_existing_mog_media import _validate_particle_records
        observations = [dict(step=i, synthetic_display_fixture=True) for i in range(1, 25)]
        records = [dict(step=row['step'], metrics=deepcopy(row),
                        particles=[[0.0]] * 12, target=[[0.0]] * 12,
                        critic_gradient=[[0.0]] * 24) for row in observations]
        self.assertIsNone(_validate_particle_records(observations, records))
        for key, count in (('critic_gradient', 12), ('particles', 24), ('target', 24)):
            wrong = deepcopy(records); wrong[0][key] = [[0.0]] * count
            with self.subTest(key=key, count=count):
                with self.assertRaises(ValueError):
                    _validate_particle_records(observations, wrong)

    def test_ready_workflow_and_catalog_sources_are_explicit(self):
        study = json.loads((ROOT / 'configs/forge/studies/atlas-existing-mog-nearest-positive791-study-v1.json').read_text())
        self.assertIn(study['status'], ('draft', 'ready'))
        self.assertEqual(study['candidate'], owner.CANDIDATE_ID)
        self.assertEqual(study['scope']['view'], owner.VIEW_ID)
        paths = set(owner.supporting_source_paths(task('two_pole'), candidate(), root=ROOT))
        self.assertIn('configs/forge/legacy-ideas-v1.json', paths)
        self.assertIn('configs/forge/defaults.json', paths)
        self.assertIn('configs/forge/ideas/ka2.json', paths)
        self.assertIn('configs/forge/ideas/atlas-existing-mog-nearest-positive791-v1.json', paths)
        self.assertIn('configs/forge/studies/atlas-existing-mog-nearest-positive791-study-v1.json', paths)
        for name in owner.PARENTS:
            self.assertIn('configs/forge/tasks/' + name + '.json', paths)

    def test_previous_kernel_source_and_identity_cannot_supply_new_radius_credit(self):
        for step in (0, 1000):
            for field in ('core', 'schema', 'candidate'):
                wrong = reaction(step=step, backend='pending' if step == 0 else 'knn')
                if field == 'core':
                    wrong['core_pins']['particlegan/birth_death.py'] = dict(
                        sha256='5db0af4bbb12c9c5369c59585b914eb7eff41a40995352bc75bef1046e728a85', bytes=47722)
                if field == 'schema': wrong['schema'] = 'forge_atlas760_mog_reaction_kernel_v1'
                if field == 'candidate': wrong['candidate_id'] = 'atlas-existing-mog-kernel760-v1'
                with self.subTest(step=step, field=field):
                    with self.assertRaises(ValueError):
                        owner.validate_reaction_kernel_receipt(wrong, prior(), completed_steps=step)

    def test_pinned_birth_source_and_other_fixed_controls(self):
        raw = (ROOT / 'particlegan/birth_death.py').read_bytes()
        self.assertEqual(owner.CORE_PINS['particlegan/birth_death.py'], dict(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        self.assertEqual(owner.CORE_PINS['particlegan/birth_death.py']['sha256'],
            '7482b29f0065be446a6b5e087a4b55d0961c8a6f2602cf9919ca502edca78c91')
        self.assertEqual(owner.CORE_PINS['particlegan/policy.py']['sha256'],
            '7a6ac1410260d720f58b6903310443f3f8dc3a283c1f3b5abd921965835a4967')
        for mutator in ('kernel', 'network', 'data', 'gate', 'horizon'):
            changed = task('gaussian1d_acquisition')
            if mutator == 'kernel': changed['execution']['prior']['kind'] = 'noisy_particles'
            if mutator == 'network': changed['execution']['host_definition']['z_dim'] += 1
            if mutator == 'data': changed['execution']['host_definition']['means'][0][0] += .1
            if mutator == 'gate': changed['evaluation']['thresholds'][0][2] = 4095
            if mutator == 'horizon': changed['execution']['steps'] += 1
            with self.subTest(field=mutator):
                with self.assertRaises(ValueError): owner.validate(changed)

    def test_media_and_cards_require_the_new_frozen_scope(self):
        from experiments.forge.atlas717_existing_mog_media import EXPECTED_VIEW, EXPECTED_CLAIM_CONTRACT, OWNER_PINS
        view = json.loads((ROOT / 'configs/forge/views/atlas_existing_mog_nearest_positive791_v1.json').read_text())
        self.assertEqual(EXPECTED_VIEW, view)
        self.assertEqual(EXPECTED_CLAIM_CONTRACT, candidate()['claim_contract'])
        for name, expected in OWNER_PINS.items():
            with self.subTest(module=name):
                self.assertEqual(expected, hashlib.sha256((ROOT / 'experiments/forge' / (name + '.py')).read_bytes()).hexdigest())
        self.assertIs(view['reporting']['family_totals'], False)



if __name__ == '__main__':
    unittest.main()
