"""Focused software checks for saved-observation media; authored, not run here."""
from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experiments.forge import atlas717_existing_mog_media as media
from experiments.forge.contracts import read_json, stable_hash


ROOT = Path(__file__).resolve().parents[1]


def request_fixture():
    return dict(candidate=read_json(ROOT / 'configs/forge/ideas/atlas-existing-mog-tier1-717-v1.json'),
        view=deepcopy(media.EXPECTED_VIEW), through_tier=1,
        tasks={item['task']: read_json(ROOT / 'configs/forge/tasks' / (item['task'] + '.json'))
               for item in media.EXPECTED_VIEW['assignments']},
        candidate_revision='0' * 64, protocol=read_json(ROOT / 'configs/forge/protocols/screening.json'),
        source=dict(commit='1' * 40, digest='2' * 64, snapshot_path='/INERT/private/source'), runtime={})


def clocks(steps):
    return [math.ceil(i * steps / 24) for i in range(1, 25)]


class Atlas717ExistingMoGMedia(unittest.TestCase):
    def test_exact_track_scope_and_original_thresholds(self):
        request = request_fixture()
        media._validate_request_scope(request)
        for kind in ('candidate', 'track', 'view', 'gate'):
            changed = deepcopy(request)
            if kind == 'candidate':
                changed['candidate']['id'] = 'atlas-noisy025-tier1-717-v1'
            elif kind == 'track':
                changed['candidate']['claim_contract']['experimental_track'] = 'atlas717_noisy025'
            elif kind == 'view':
                changed['view']['reporting']['family_totals'] = True
            else:
                changed['tasks']['gaussian1d_acquisition']['evaluation']['thresholds'][0][2] = 4095
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                media._validate_request_scope(changed)

    def test_original_scored_tensor_schedule_shape_and_finiteness(self):
        task = request_fixture()['tasks']['gaussian1d_acquisition']
        measured = [dict(step=step) for step in clocks(task['execution']['steps'])]
        values = torch.zeros(4096, 1)
        records = [dict(step=row['step'], samples=values) for row in measured]
        media._validate_scored_records(task, measured, records)
        changed = list(records)
        changed[-1] = dict(step=measured[-1]['step'], samples=torch.zeros(4095, 1))
        with self.assertRaises(ValueError):
            media._validate_scored_records(task, measured, changed)
        changed[-1] = dict(step=measured[-1]['step'], samples=values.clone())
        changed[-1]['samples'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            media._validate_scored_records(task, measured, changed)
        changed[-1] = dict(step=measured[-1]['step'] - 1, samples=values)
        with self.assertRaises(ValueError):
            media._validate_scored_records(task, measured, changed)

    def test_original_direct_metrics_and_gradient_shapes(self):
        measured = [dict(step=step, travel=0.4, critic_gradient_median=0.5) for step in clocks(80)]
        records = [dict(step=row['step'], metrics=deepcopy(row), particles=[[0.4]] * 12,
                        target=[[-1.]] * 6 + [[1.]] * 6, critic_gradient=[[0.5]] * 12)
                   for row in measured]
        media._validate_particle_records(measured, records)
        changed = deepcopy(records)
        changed[-1]['metrics']['travel'] = 0.8
        with self.assertRaises(ValueError):
            media._validate_particle_records(measured, changed)
        changed = deepcopy(records)
        changed[-1]['critic_gradient'] = [[0.5]] * 11
        with self.assertRaises(ValueError):
            media._validate_particle_records(measured, changed)

    def test_original_ae_decoder_outputs_and_metrics(self):
        measured = [dict(step=step, hold=0.2, reconstruction_mse=0.01) for step in clocks(250)]
        arrays = dict(generated=[[0., 0.]] * 1024, reconstructed=[[0., 0.]] * 1024,
                      target=[[0., 0.]] * 1024, prior=[[0., 0.]] * 12, anchors=[[-1., 0.], [1., 0.]])
        records = [dict(step=row['step'], metrics={k: v for k, v in row.items() if k != 'step'},
                        **arrays) for row in measured]
        media._validate_ae_records(measured, records)
        changed = deepcopy(records)
        changed[-1]['metrics']['hold'] = 0.3
        with self.assertRaises(ValueError):
            media._validate_ae_records(measured, changed)
        changed = deepcopy(records)
        changed[-1]['reconstructed'] = [[0., 0.]] * 1023
        with self.assertRaises(ValueError):
            media._validate_ae_records(measured, changed)

    def test_collected_terminal_grade_source_and_attempt_join(self):
        request = request_fixture()
        local = Path('/INERT/atlas717-original-collected-attempt')
        worker = dict(attempt='INERT_ATTEMPT', token='INERT_TOKEN', directory=str(local))
        job = dict(task_id='two_pole', task_ids=['two_pole'])
        raw_result = dict(evidence={'observations': []})
        raw = dict(attempt_status='completed', token=worker['token'], result=raw_result,
                   grading=dict(raw_hash=stable_hash(raw_result), source_digest=request['source']['digest'],
                                grades={'two_pole': {'gate_status': 'FAIL'}}))
        result = dict(candidate_revision=request['candidate_revision'], attempt_id=worker['attempt'], raw=raw,
                      task_results=[dict(task_id='two_pole', gate_status='FAIL', raw_status='completed',
                                         evidence={'INERT': True})])
        envelope = dict(request=request, job=job, worker=worker)

        def check(record):
            certificate = dict(result_hash=stable_hash(record), source=request['source'],
                               runtime=request['runtime'], local_artifact_root=str(local))
            files = {'request.json': envelope, 'result.json': record, 'evidence.json': certificate}
            with patch.object(media, '_validate_request_scope'), \
                 patch.object(media, 'read_json', side_effect=lambda path: files[Path(path).name]), \
                 patch.object(media, 'render', return_value={'INERT_RENDER_NOT_EXECUTED': True}):
                return media.export_attempt('/INERT/durable-attempt', '/INERT/media')

        self.assertEqual(len(check(result)), 1)
        for kind in ('token', 'source', 'attempt', 'status', 'grade'):
            changed = deepcopy(result)
            if kind == 'token':
                changed['raw']['token'] = 'FOREIGN'
            elif kind == 'source':
                changed['raw']['grading']['source_digest'] = 'f' * 64
            elif kind == 'attempt':
                changed['attempt_id'] = 'FOREIGN'
            elif kind == 'status':
                changed['raw']['attempt_status'] = 'error'
            else:
                changed['task_results'][0]['gate_status'] = 'PASS'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check(changed)

    def test_nine_original_frames_include_the_actual_terminal_read(self):
        self.assertEqual(media._indices(24), [0, 3, 6, 9, 12, 14, 17, 20, 23])


if __name__ == '__main__':
    unittest.main()
