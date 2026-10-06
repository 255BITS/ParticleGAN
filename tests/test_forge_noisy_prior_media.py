"""Synthetic saved-media contract tests, authored without execution.

These metadata/tensor fixtures are not observations or scientific receipts.
ROOT owns paid execution; no model, sampler, optimizer or grader is called.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import torch

from experiments.forge import noisy_prior_tier1 as noisy
from experiments.forge import tier1_media as media

ROOT = Path(__file__).resolve().parents[1]


class NoisySavedMediaTest(unittest.TestCase):
    def fixture(self, parent):
        task = noisy.make_variant(json.loads((ROOT / 'configs/forge/tasks' / (parent + '.json')).read_text()))
        steps = task['execution']['steps']
        clocks = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
        kernel = dict(kind='noisy_particle_cloud',
            code_path='particlegan.noisy_particle_prior.NoisyParticlePrior',
            sigma=.02500000037252903 if noisy.PARENTS[parent]['sigma'] else 0.,
            standardize=False, learned_width=False, sigma_units='raw_latent_coordinates',
            row_weights='uniform', zero_sigma_consumes_no_kernel_rng=True)
        evidence = dict(
            observations=[dict(step=step) for step in clocks],
            sampling_law=task['evaluation']['sampling_law'], scoring_weights='live',
            policy_controls=dict(cohort=task['task_cohort'], completed_steps=steps,
                implementation_observed=True, requested_owners_bound=True,
                lifecycle=dict(complete=True)),
            policy_observations=[dict(completed_steps=step, observed=True,
                table_alias_preserved=True, task_id=task['id'], parent_task_id=parent,
                policy_owner='particlegan.UpdatePolicy', weights='live',
                sampling_law=task['evaluation']['sampling_law'],
                eval_output_noise=task['evaluation']['eval_output_noise'],
                sampling_calls_added_by_receipt=0, prior=deepcopy(kernel)) for step in clocks],
            policy_purity=[dict(completed_steps=step, pure=True,
                before_sha256='f' * 64, after_sha256='f' * 64) for step in clocks])
        if parent in {'gaussian1d_acquisition', 'ring16_acquisition'}:
            evidence.update(host=dict(definition=deepcopy(task['execution']['host_definition'])),
                saved_observer_outputs=dict(path='observed-samples.pt', kind='scored_vector_samples_v1',
                    sha256='1' * 64, bytes=1234, observation_count=24,
                    optimizer_updates_added=0, sampling_draws_added=0))
        elif parent == 'two_pole':
            evidence['saved_particle_observations'] = [dict(step=step,
                particles=[[0.] for _ in range(12)],
                target=[[-1. if i % 2 else 1.] for i in range(12)],
                metrics=deepcopy(row)) for step, row in zip(clocks, evidence['observations'])]
        return task, evidence

    def test_ordinary_render_path_has_no_new_noisy_requirements(self):
        self.assertIsNone(media._noisy_media_binding({'id': 'two_pole'}, {}))

    def test_exact_prior_scope_and_original_live_observer_labels(self):
        for parent in ('gaussian1d_acquisition', 'ring16_acquisition', 'two_pole'):
            with self.subTest(parent=parent):
                task, evidence = self.fixture(parent)
                bound = media._noisy_media_binding(task, evidence)
                self.assertEqual(bound['cohort'], noisy.COHORT)
                self.assertEqual(bound['parent_task_id'], parent)
                self.assertEqual(bound['sigma'], noisy.PARENTS[parent]['sigma'])
                self.assertEqual(bound['weights'], 'live')
                self.assertEqual(bound['eval_output_noise'], task['evaluation']['eval_output_noise'])
                self.assertEqual(bound['latent_prior_sampled'], parent != 'two_pole')
                self.assertIs(bound['ordinary_parent_credit'], False)

    def test_scored_descriptor_rejects_extra_work_missing_reads_or_changed_target(self):
        task, base = self.fixture('ring16_acquisition')
        changes = {
            'foreign_path': lambda e: e['saved_observer_outputs'].__setitem__('path', '../foreign.pt'),
            'wrong_kind': lambda e: e['saved_observer_outputs'].__setitem__('kind', 'new_eval_samples'),
            'missing_saved_read': lambda e: e['saved_observer_outputs'].__setitem__('observation_count', 23),
            'alias_read_count': lambda e: e['saved_observer_outputs'].__setitem__('observation_count', 24.),
            'extra_draw': lambda e: e['saved_observer_outputs'].__setitem__('sampling_draws_added', 1),
            'extra_update': lambda e: e['saved_observer_outputs'].__setitem__('optimizer_updates_added', 1),
            'false_zero': lambda e: e['saved_observer_outputs'].__setitem__('sampling_draws_added', False),
            'missing_hash': lambda e: e['saved_observer_outputs'].__setitem__('sha256', ''),
            'empty_bytes': lambda e: e['saved_observer_outputs'].__setitem__('bytes', 0),
            'target_changed': lambda e: e['host']['definition']['means'][0].__setitem__(0, 10.),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                evidence = deepcopy(base); change(evidence)
                with self.assertRaises(ValueError):
                    media._noisy_media_binding(task, evidence)

    def test_owner_law_and_all24_numeric_clocks_precede_media(self):
        task, base = self.fixture('gaussian1d_acquisition')
        changes = {
            'no_owner': lambda e: e.pop('policy_controls'),
            'selected_weights': lambda e: e['policy_observations'][0].__setitem__('weights', 'averaged'),
            'extra_output_noise': lambda e: e['policy_observations'][0].__setitem__('eval_output_noise', 'noisy'),
            'state_changed': lambda e: e['policy_purity'][0].__setitem__('after_sha256', '0' * 64),
            'missing_numeric_read': lambda e: e['observations'].pop(),
            'numeric_clock_changed': lambda e: e['observations'][-1].__setitem__('step', 999),
            'float_clock_alias': lambda e: e['observations'][-1].__setitem__('step', 1000.),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                evidence = deepcopy(base); change(evidence)
                with self.assertRaises(ValueError):
                    media._noisy_media_binding(task, evidence)

    def test_original24_scored4096_tensor_join_and_shape_refusals(self):
        task, evidence = self.fixture('ring16_acquisition')
        # Zero fake software tensors consume no RNG and are never graded.
        saved = torch.zeros(4096, 2)
        records = [dict(step=row['step'], samples=saved) for row in evidence['observations']]
        media._validate_noisy_scored_records(task, evidence['observations'], records)
        for label, change in {
            'missing_read': lambda rows: rows.pop(),
            'wrong_clock': lambda rows: rows[-1].__setitem__('step', 399),
            'integer_alias': lambda rows: rows[-1].__setitem__('step', 400.),
            'wrong_count': lambda rows: rows[-1].__setitem__('samples', torch.zeros(4095, 2)),
            'wrong_dimensions': lambda rows: rows[-1].__setitem__('samples', torch.zeros(4096, 1)),
            'integer_tensor': lambda rows: rows[-1].__setitem__('samples', torch.zeros(4096, 2, dtype=torch.int64)),
        }.items():
            with self.subTest(label=label):
                rows = [dict(row) for row in records]; change(rows)
                with self.assertRaises(ValueError):
                    media._validate_noisy_scored_records(task, evidence['observations'], rows)

    def test_two_pole_requires_retained_coordinates_and_keeps_original_frame_subset(self):
        task, evidence = self.fixture('two_pole')
        evidence.pop('saved_particle_observations')
        with self.assertRaises(ValueError):
            media._noisy_media_binding(task, evidence)
        indices = media._indices(24)
        self.assertEqual(len(indices), 9)
        self.assertEqual((indices[0], indices[-1]), (0, 23))

    def ae_arrays(self):
        # Original1024 decoder/target cardinality, twelve latent rows and two
        # anchors, represented by fake finite software values, never graded.
        clocks = sorted({math.ceil(i * 250 / 24) for i in range(1, 25)})
        metrics = dict(recon_mse=1., hold=1.)
        observations = [dict(step=step, **metrics) for step in clocks]
        arrays = {key: [[0., 0.] for _ in range(count)] for key, count in
            (('generated', 1024), ('reconstructed', 1024), ('target', 1024), ('prior', 12), ('anchors', 2))}
        records = [dict(step=step, metrics=deepcopy(metrics), **arrays) for step in clocks]
        return observations, records

    def test_ae_retained_original_target_and_two_decoder_outputs_are_metadata_only(self):
        observations, records = self.ae_arrays()
        media._validate_noisy_ae_records(observations, records)
        self.assertEqual(len(records), 24)
        self.assertEqual(records[-1]['step'], 250)
        self.assertEqual(len(records[-1]['target']), 1024)
        self.assertEqual(len(records[-1]['prior']), 12)

    def test_ae_lost_read_changed_metric_shape_or_nonfinite_arrays_refuse(self):
        observations, base = self.ae_arrays()
        changes = {
            'missing_read': lambda rows: rows.pop(),
            'wrong_clock': lambda rows: rows[-1].__setitem__('step', 249),
            'float_clock': lambda rows: rows[-1].__setitem__('step', 250.),
            'changed_metric': lambda rows: rows[-1]['metrics'].__setitem__('hold', .1),
            'missing_generated': lambda rows: rows[-1].pop('generated'),
            'wrong_pair_count': lambda rows: rows[-1]['reconstructed'].pop(),
            'wrong_latent_rows': lambda rows: rows[-1]['prior'].pop(),
            'wrong_anchors': lambda rows: rows[-1]['anchors'].pop(),
            'nonfinite': lambda rows: rows[-1]['target'][0].__setitem__(0, float('nan')),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                records = deepcopy(base); change(records)
                with self.assertRaises(ValueError):
                    media._validate_noisy_ae_records(observations, records)


if __name__ == '__main__':
    unittest.main()
