"""Two metadata-only schema3 regression tests; authored NOT_RUN by Source author."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from experiments.forge.contracts import validate_idea


class CorrectedIdeaContractTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).resolve().parents[1] /
            'configs/forge/ideas/atlas-existing-mog-nearest-positive791-v1.json').read_text())

    def test_corrected_schema3_idea_validates_without_mutation(self):
        idea=self.fixture();before=deepcopy(idea)
        self.assertIsNone(validate_idea(idea))
        self.assertEqual(idea,before)

    def test_old_source_files_field_remains_refused(self):
        idea=self.fixture();idea['source_files']=['particlegan/birth_death.py']
        with self.assertRaisesRegex(ValueError,'unsupported fields: source_files'):
            validate_idea(idea)
