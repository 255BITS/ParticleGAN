"""ROOT-paid metadata regressions; AUTHORED_NOT_RUN and no target execution.

Run only after the proposed exact two modules are independently reviewed and
authorized for installation. No Queue, model factory, sampler or training call.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = None
CANDIDATE = 'atlas-original-two-pole-passive889-v1'
VIEW = 'atlas_original_two_pole_passive889_v1'
STUDY = 'atlas-original-two-pole-passive889-study-v1'
PINS = {
    'experiments/forge/promotion.py': ('2e5ae306e0b33851792511e2f572fcc37471af27c20b1525ec312cb74c46f857', 33703),
    'experiments/forge/hostprofiles.py': ('d7b55cb24ea31264ad6b096b20dc0a28556f843f30a8422624c889d0b997e727', 15459),
    'experiments/forge/api.py': ('3494c26774544e28a8e950f79e0c1abdf387d6455f0ed068f5f2516522dfa0a9', 50840),
    'experiments/forge/atlas889_two_pole_owner.py': ('428b0aa4a7fc272f8eac39961ebf3b742e513b603d402d95d910b73ff0924856', 58663),
    'configs/forge/tasks/two_pole.json': ('55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5', 3099),
}
CHECKS = (
    'exact889_metadata_identity_forwarded',
    'wrong_view_rejected',
    'unrelated_candidate_rejected',
    'unsupported_override_rejected',
    'screening_seed_overrides_rejected',
    'original_direct_task_law_unchanged',
    'resolved_recipe_echo_rejected',
    'candidate_revision_echo_rejected',
)


class MetadataSubmissionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.forge import api, atlas889_two_pole_owner, hostprofiles, planning, promotion
        cls.api, cls.hostprofiles, cls.promotion = api, hostprofiles, promotion
        for relative, expected in PINS.items():
            raw = (ROOT / relative).read_bytes()
            if (hashlib.sha256(raw).hexdigest(), len(raw)) != expected:
                raise ValueError('reviewed metadata/control Source differs: ' + relative)
        for module in (api, atlas889_two_pole_owner, hostprofiles, planning, promotion):
            expected = ROOT / 'experiments/forge' / (module.__name__.rsplit('.', 1)[1] + '.py')
            if Path(module.__file__).resolve() != expected:
                raise ValueError('metadata controls loaded a foreign module: ' + module.__name__)
        cls.request = planning.resolve_idea(ROOT, CANDIDATE, study=STUDY, freeze_source=False)
        if (cls.request['preflight_blockers'] or cls.request['study_review']['status'] != 'READY'
                or set(cls.request['tasks']) != {'two_pole'}
                or cls.request['requires_independent_grading'] is not True):
            raise ValueError('real maintained ready original-task metadata is required')

    def _validators(self):
        return ((self.promotion, 'FormulationContext', self.promotion.validate_screening_submission),
                (self.api, 'FormulationContext', self.hostprofiles._validate_candidate_identity))

    def test_exact889_metadata_identity_forwarded(self):
        original = self.api.FormulationContext
        for module, name, validate in self._validators():
            with self.subTest(validator=validate.__name__):
                request = deepcopy(self.request)
                before = deepcopy(request)
                with patch.object(module, name, wraps=original) as constructor:
                    validate(request)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(constructor.call_args.kwargs['candidate_id'], CANDIDATE)
                self.assertEqual(constructor.call_args.kwargs['prior'], before['candidate']['prior'])
                self.assertEqual(request, before)

    def test_wrong_view_rejected(self):
        from experiments.forge.atlas_existing_mog import VIEW_ID as original791_view
        for view_id in ('inert-unrelated-view', original791_view):
            for _, _, validate in self._validators():
                with self.subTest(validator=validate.__name__, view=view_id):
                    request = deepcopy(self.request)
                    request['view']['id'] = view_id
                    with self.assertRaises(ValueError):
                        validate(request)

    def test_unrelated_candidate_rejected(self):
        for _, _, validate in self._validators():
            with self.subTest(validator=validate.__name__):
                request = deepcopy(self.request)
                request['candidate']['id'] = 'inert-unrelated-candidate'
                with self.assertRaises(ValueError):
                    validate(request)

    def test_unsupported_override_rejected(self):
        for _, _, validate in self._validators():
            with self.subTest(validator=validate.__name__):
                request = deepcopy(self.request)
                request['candidate']['recipe_overrides'] = {'lr': .01}
                with self.assertRaises(ValueError):
                    validate(request)

    def test_screening_seed_overrides_rejected(self):
        request = deepcopy(self.request)
        request['protocol']['seed'] = 1
        with self.assertRaises(ValueError):
            self.promotion.validate_screening_submission(request)
        request = deepcopy(self.request)
        request['tasks']['two_pole']['execution']['seed'] = 1
        with self.assertRaises(ValueError):
            self.promotion.validate_screening_submission(request)

    def test_original_direct_task_law_unchanged(self):
        task = deepcopy(self.request['tasks']['two_pole'])
        context = self.api.task_formulation_context(self.request['candidate'], task,
            self.request['protocol'], root=ROOT)
        recipe = json.dumps(asdict(context.recipe), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(recipe).hexdigest(),
            'd5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95')
        self.assertEqual(context.prior_config['kind'], 'particle_cloud')
        self.assertEqual(context.prior_config['sigma'], 0.)
        self.assertIsNone(context._trainer)
        self.assertIsNone(context._policy)
        self.assertEqual(task['execution']['steps'], 80)
        self.assertEqual(task['evaluation']['observations'], 24)
        self.assertEqual(task['evaluation']['minimum_stable_checks'], 5)
        changed = deepcopy(task)
        changed['execution']['steps'] = 81
        with self.assertRaises(ValueError):
            self.api.task_formulation_context(self.request['candidate'], changed,
                self.request['protocol'], root=ROOT)

    def test_resolved_recipe_echo_rejected(self):
        request = deepcopy(self.request)
        request['candidate']['resolved_recipe']['lr'] = .01
        with self.assertRaisesRegex(ValueError, 'resolved_recipe differs'):
            self.hostprofiles._validate_candidate_identity(request)

    def test_candidate_revision_echo_rejected(self):
        request = deepcopy(self.request)
        request['candidate_revision'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'scientific identity differs'):
            self.hostprofiles._validate_candidate_identity(request)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--controls-json', required=True)
    args = parser.parse_args()
    global ROOT
    ROOT = Path(args.root).resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetadataSubmissionControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun == len(CHECKS)
    # This optional record is software-only; it is never the M11 scientific
    # admission proof, nor a claim that these external fixtures are Source.files.
    Path(args.controls_json).write_text(json.dumps(dict(
        schema='forge_atlas889_metadata_submission_control_result_v1',
        status='PASS' if passed else 'FAIL',
        checks=dict.fromkeys(CHECKS, 'PASS') if passed else {},
        software_only=True, original_target_run=False), sort_keys=True, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
