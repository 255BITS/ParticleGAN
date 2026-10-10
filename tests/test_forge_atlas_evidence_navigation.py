"""Model-free controls for Atlas evidence navigation; no scientific module imports."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


def source_namespace():
    maintained = Path(__file__).resolve().parents[1] / "experiments/forge/family_reports.py"
    private = Path(__file__).with_name("family_reports.py")
    path = private if private.is_file() else maintained
    tree = ast.parse(path.read_text())
    functions = {"_atlas_evidence_navigation", "_full_original_atlas_status", "cell", "number", "link", "_count"}
    assignments = {"ATLAS_EVIDENCE_REPORTS", "FULL_ORIGINAL_ATLAS_CONFIG_SHA256", "COMPLETE"}
    selected = [node for node in tree.body if
                isinstance(node, ast.FunctionDef) and node.name in functions or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets)]
    namespace = {"Counter": Counter, "deepcopy": deepcopy, "os": os, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fixtures():
    source = {"origin_commit": "9da0e927a0a4242ee813ceb0340e9fee70ccdc40",
              "execution_digest": "00c19ef01f68e3222888a9b0edc6fcc5f635a384b982260c4e005832c1ed09ae"}
    native = {"schema": "pg_atlas_restoration_public_report_v1",
              "claims": {key: False for key in ("qualification_credit", "original_common26_credit", "default_adoption", "speed_ranking")},
              "native_cases": [{"task_id": task, "candidate_id": "atlas-native-original-representation-selected-seed0-v1",
                                "recipe_sha256": "152ee607b4a2b18d99e9f440985b920363a481a14236ae12409747e9c283738a",
                                "source": deepcopy(source), "grade": {"status": "PASS", "full_protocol_complete": True}}
                               for task in ("grid100", "rotated100", "staggered100")]}
    ae = {"schema": "pg_ae633_sourceguard_public_checkpoint_v1", "candidate_id": "atlas-full-original-ae-sourceguard-repair-v1",
          "task_id": "ae_gan_hold", "recipe_sha256": "a26eb000fbb7616c90265a4bf31c5126e75f86a4cbd04912d71dc0292167e250",
          "source": {"origin_commit": "36718f9ecbffe6a217914de7a9c22cff1349079e",
                     "digest": "2f71661f3407a3af3b166493fa0cb1f6730e8ac4689c2b37a8c8a9e424c5c233"},
          "complete_protocol": True, "completed_updates": 250, "observation_count": 24, "status": "PASS",
          "accepted_evidence": {"grade": {"status": "PASS", "gate_status": "PASS"}},
          "qualification_credit": False, "default_adoption": False, "speed_ranking": False}
    common = {"schema": "pg_common26_full_original_diagnostic_checkpoint_v1",
              "config_sha256": "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4",
              "campaign_status": "HALTED_INCOMPLETE", "halt_required": True,
              "counts": {"PASS": 5, "FAIL": 10, "INVALID": 1, "BLOCKED": 9, "NOT_RUN": 1},
              "current_source": {"origin_commit": "7f7037a54edc54cfb8f7466f6a98aee56e17b6db", "digest": "primary-source"},
              "retained_first_case_source": {"origin_commit": "c6dbe8f6cdbe5c4df1c25ca67d807f075939e99d", "digest": "first-source"},
              **{key: False for key in ("qualification_credit", "default_adoption", "speed_ranking", "same_source_whole26_claim", "measured_whole26_ranking")}}
    return {"live_clean_common26": common, "native_restoration": native, "ae_sourceguard": ae}


class VirtualFile:
    def __init__(self, root, name):
        self.root, self.name = root, name
    def is_file(self):
        return self.name in self.root.raw
    def stat(self):
        return SimpleNamespace(st_size=len(self.root.raw[self.name]))


class VirtualRoot:
    def __init__(self, raw):
        self.raw = raw
    def __truediv__(self, path):
        return VirtualFile(self, Path(path).as_posix())


def bound_fixture(namespace, data=None):
    data = fixtures() if data is None else data
    raw, pins, reads = {}, [], []
    for scope, relative, original_sha, original_bytes, schema in namespace["ATLAS_EVIDENCE_REPORTS"]:
        if scope not in data:
            pins.append((scope, relative, original_sha, original_bytes, schema))
            continue
        value = json.dumps(data[scope], sort_keys=True).encode()
        raw[relative] = value
        pins.append((scope, relative, hashlib.sha256(value).hexdigest(), len(value), schema))
    namespace["ATLAS_EVIDENCE_REPORTS"] = tuple(pins)
    namespace["file_hash"] = lambda path: hashlib.sha256(raw[path.name]).hexdigest()
    def load(path):
        reads.append(path.as_posix())
        return json.loads(raw[path.as_posix()])
    return VirtualRoot(raw), load, raw, reads


class AtlasEvidenceNavigationTests(unittest.TestCase):
    def setUp(self):
        self.ns = source_namespace()
    def project(self, data=None):
        root, load, raw, reads = bound_fixture(self.ns, data)
        return self.ns["_atlas_evidence_navigation"](root, load)
    def test_distinct_cohorts_keep_original_halt_and_no_selected_credit(self):
        nav = self.project()
        self.assertFalse(nav["qualification_input"])
        self.assertFalse(nav["selected_row_credit"])
        self.assertEqual([c["scope_id"] for c in nav["cohorts"]], ["live_clean_common26", "native_restoration", "ae_sourceguard"])
        common, native, ae = nav["cohorts"]
        self.assertEqual(common["status"], "HALTED_INCOMPLETE")
        self.assertTrue(common["halt_required"])
        self.assertEqual(common["counts"], fixtures()["live_clean_common26"]["counts"])
        self.assertNotEqual(native["source"], ae["source"])
        self.assertEqual(len(native["task_statuses"]), 3)
        self.assertEqual(len(ae["task_statuses"]), 1)
        self.assertNotIn("passed", nav)
        self.assertNotIn("required", nav)
    def test_missing_later_reports_retains_archived_first_case_only(self):
        self.assertIsNone(self.project({}))
        context = {"first_case": {"status": "FAIL", "metric_receipts": []}, "readout": "first/README.md", "public_result": "first/results.json"}
        lines = self.ns["_full_original_atlas_status"](Path("/synthetic"), Path("/synthetic/reports/page.md"), {"full_original_atlas_common26": context})
        text = "\n".join(lines)
        self.assertIn("Archived first", text)
        for stale in ("remaining 25", "completed 1/26", "pending adapter"):
            self.assertNotIn(stale, text)
        self.assertEqual(self.ns["_full_original_atlas_status"](Path("/synthetic"), Path("/synthetic/reports/page.md"), {}), [])
    def test_changed_byte_pin_is_refused_before_json_load(self):
        root, load, raw, reads = bound_fixture(self.ns)
        path = self.ns["ATLAS_EVIDENCE_REPORTS"][0][1]
        raw[path] = raw[path].replace(b'5', b'6', 1)
        with self.assertRaisesRegex(ValueError, "public report changed"):
            self.ns["_atlas_evidence_navigation"](root, load)
        self.assertEqual(reads, [])
    def test_changed_size_is_refused_before_json_load(self):
        root, load, raw, reads = bound_fixture(self.ns)
        raw[self.ns["ATLAS_EVIDENCE_REPORTS"][0][1]] += b' '
        with self.assertRaisesRegex(ValueError, "public report changed"):
            self.ns["_atlas_evidence_navigation"](root, load)
        self.assertEqual(reads, [])
    def test_wrong_native_identity_and_duplicate_task_refuse(self):
        for field, value in (("candidate_id", "atlas"), ("recipe_sha256", "0" * 64), ("task_id", "grid100")):
            with self.subTest(field=field):
                self.ns = source_namespace()
                data = fixtures()
                data["native_restoration"]["native_cases"][1][field] = value
                with self.assertRaises(ValueError):
                    self.project(data)
    def test_native_wrong_source_refuses_even_under_new_fixture_pin(self):
        data = fixtures()
        data["native_restoration"]["native_cases"][0]["source"]["execution_digest"] = "0" * 64
        with self.assertRaises(ValueError):
            self.project(data)
    def test_ae_incomplete_or_conflicting_accepted_grade_refuses(self):
        for field, value in (("complete_protocol", False), ("observation_count", 1), ("status", "FAIL")):
            with self.subTest(field=field):
                self.ns = source_namespace()
                data = fixtures()
                data["ae_sourceguard"][field] = value
                with self.assertRaises(ValueError):
                    self.project(data)
    def test_qualification_claim_and_foreign_schema_refuse(self):
        for field, value in (("qualification_credit", True), ("schema", "forge_tier1_scoped_evidence_v1")):
            with self.subTest(field=field):
                self.ns = source_namespace()
                data = fixtures()
                data["ae_sourceguard"][field] = value
                with self.assertRaises(ValueError):
                    self.project(data)
    def test_optional_next_steps_link_has_no_scientific_cell_projection(self):
        nav = self.project()
        nav["next_steps"] = {"readout": "reports/forge/atlas-inventory-next-steps-20261005.md", "sha256": "0" * 64}
        original = deepcopy(nav)
        text = "\n".join(self.ns["_full_original_atlas_status"](Path("/synthetic"), Path("/synthetic/reports/page.md"), {"atlas_evidence_navigation": nav}))
        self.assertIn("Atlas inventory gaps and bounded next steps", text)
        self.assertIn("HALTED_INCOMPLETE", text)
        self.assertEqual(nav, original)
        self.assertNotIn("passed", nav)
    def test_navigation_does_not_change_scientific_counts(self):
        tasks = [{"task_id": "grid100", "status": "BLOCKED", "current_contract": "matches",
                  "execution_recorded": False}]
        retained = deepcopy(tasks)
        count = deepcopy(self.ns["_count"](tasks))
        data = fixtures(); before = deepcopy(data)
        self.project(data)
        self.assertEqual(tasks, retained)
        self.assertEqual(self.ns["_count"](tasks), count)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
