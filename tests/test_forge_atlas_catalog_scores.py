"""Read-only publication controls for the original Atlas/catalog score cohorts."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "atlas_catalog_scores", ROOT / "experiments/forge/atlas_catalog_scores.py")
scores = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scores)


class AtlasCatalogScoreTests(unittest.TestCase):
    def test_original_atlas_and_catalog_scores_render_without_regrading(self):
        text = "\n".join(scores.atlas_catalog_score_lines(
            ROOT, ROOT / "reports/forge/technique-inventory.md"))
        self.assertIn("0 PASS / 1 FAIL / 2 INCOMPLETE / 4 BLOCKED across 7 questions", text)
        self.assertIn("7 PASS / 4 FAIL across 11 cases", text)
        self.assertIn("| `img_blobs4` | FAIL; full horizon incomplete | 0/12 | 300/600 |", text)
        self.assertIn("CDF KS 0.029021; diagnostic only, checkpoint export ERROR", text)
        self.assertIn("component covariance error 1.812179; diagnostic only", text)
        self.assertIn("2/4 state-digest comparisons match", text)
        self.assertIn("| `rms_cap` | PASS | 8/24 | 0.460720 | 0.405106 |", text)
        self.assertIn("285 distinct attempt IDs: 142 PASS / 136 FAIL / 7 INCOMPLETE", text)
        self.assertIn("Full G Frobenius; `vector_overlap`; seed 0 | FAIL → PASS | 14 → 15 | 3 → 5", text)
        self.assertIn("Full G Frobenius; `vector_overlap`; seed 1 | FAIL → PASS | 14 → 16 | 3 → 5", text)
        self.assertIn("`trajectory`; seed 0 | FAIL → PASS | 0 → 18 | 0 → 18 | identity MSE 0.021320 → 0.000101", text)
        self.assertIn("covariance error 0.230126 → 0.402998", text)
        self.assertIn("source `d378734f40b09ce223a389e8f54a9783ec6a0c75`", text)
        self.assertIn("byte-pinned score receipts](research-catalog/leaderboard-scores.json)", text)
        self.assertIn(scores.ATLAS_COMMIT, text)
        self.assertIn(scores.CATALOG_COMMIT, text)

    def test_rendering_does_not_change_any_source_bytes(self):
        before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in scores.SCORE_PINS}
        scores.atlas_catalog_score_lines(ROOT, ROOT / "reports/forge/technique-inventory.md")
        after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in scores.SCORE_PINS}
        self.assertEqual(before, after)
        self.assertEqual(after, scores.SCORE_PINS)

    def test_absent_publications_emit_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(scores.atlas_catalog_score_lines(Path(directory), Path(directory) / "board.md"), [])

    def test_changed_source_bytes_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / scores.ATLAS / "results.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"parent_qualification_credit": true}')
            with self.assertRaisesRegex(ValueError, "preserved score evidence changed"):
                scores.atlas_score_lines(root)

    def test_partial_publication_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / scores.ATLAS / "results.json"
            path.parent.mkdir(parents=True)
            path.write_bytes((ROOT / scores.ATLAS / "results.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "missing preserved score evidence"):
                scores.atlas_score_lines(root)

    def test_catalog_deduplication_and_qualification_are_checked(self):
        original = scores._read
        for change in ("duplicate", "qualification"):
            def altered(root, relative):
                value = deepcopy(original(root, relative))
                if change == "duplicate" and relative.endswith("/attempts.json"):
                    value[1]["attempt_id"] = value[0]["attempt_id"]
                if change == "qualification" and relative.endswith("/leaderboard-scores.json"):
                    value["qualification_input"] = True
                return value
            with self.subTest(change=change), patch.object(scores, "_read", altered):
                with self.assertRaisesRegex(ValueError, "catalog score scope"):
                    scores.catalog_score_lines(ROOT, ROOT / "board.md")

    def test_compact_score_receipt_matches_original_catalog_and_archive_pins(self):
        receipt = json.loads((ROOT / scores.CATALOG / "leaderboard-scores.json").read_text())
        studies = {s["study_id"]: s for s in json.loads((ROOT / scores.CATALOG / "studies.json").read_text())}
        pins = json.loads((ROOT / scores.CATALOG / "archive-pins.json").read_text())
        self.assertEqual(len(receipt["comparisons"]), 5)
        self.assertEqual(receipt["new_training_attempts"], 0)
        self.assertEqual(receipt["catalog_source_sha256"], scores.SCORE_PINS[f"{scores.CATALOG}/studies.json"])
        self.assertEqual(receipt["archive_pins_sha256"], scores.SCORE_PINS[f"{scores.CATALOG}/archive-pins.json"])
        for row in receipt["comparisons"]:
            study = studies[row["study_id"]]
            original = next(c for c in study["comparisons"] if (c["task"], c["seed"]) == (row["task"], row["seed"]))
            self.assertTrue(original["matched_contract_verified"])
            for role in ("baseline", "candidate"):
                arm = row[role]
                self.assertEqual(arm["metrics"], original[role]["metrics"])
                self.assertEqual(arm["artifact"], original[role]["artifact"])
                self.assertEqual(arm["sha256"], pins[arm["artifact"]]["sha256"])
                self.assertEqual(arm["status"], original[role]["status"])
                self.assertEqual(arm["observations"], 24)
                self.assertLessEqual(arm["passing_suffix"], arm["passing_observations"])


if __name__ == "__main__":
    unittest.main()
