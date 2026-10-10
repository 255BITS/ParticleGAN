"""Saved-metadata tests only: no model construction, sampling or training."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import family_row_pin, scientific_row_hash
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint
from reports.forge.prepare_inventory_family_pins import ROOT, _validate_refresh_sources, propose


def certify(report):
    report.setdefault("provenance", {}).pop("input_digest", None)
    report["provenance"]["input_digest"] = stable_hash(report)
    return report


class SavedPinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / "configs/forge", self.root / "configs/forge")
        atomic_json(self.root / "configs/forge/trainer-families.json", {
            "schema_version": 1,
            "families": [{"id": "family-a", "label": "A", "candidates": ["base-a", "selected-a"],
                          "canonical_candidate": "base-a"},
                         {"id": "family-b", "label": "B", "candidates": ["base-b"],
                          "canonical_candidate": "base-b"}]})
        self.view = read_json(self.root / "configs/forge/views/discriminator_stability.json")
        self.required = sorted(item["task"] for item in self.view["assignments"]
                               if item["importance"] == "required" and item["qualification_tier"] == 1)
        self.denominator = [sum(item["importance"] == "required" and item["qualification_tier"] == tier
                                for item in self.view["assignments"]) for tier in (1, 2, 3)]
        self.declarations = {name: {"id": name, "trainer_family": family} for name, family in
                             (("base-a", "family-a"), ("selected-a", "family-a"), ("base-b", "family-b"))}
        self.contracts = {}
        self.task_digests = {}
        for item in self.view["assignments"]:
            task = read_json(self.root / "configs/forge/tasks" / (item["task"] + ".json"))
            contract = {"execution_sha256": task_execution_fingerprint(task),
                        "evaluation_sha256": task_evaluation_fingerprint(task),
                        "timeout_seconds": task["resources"]["timeout_seconds"]}
            digest = stable_hash(contract)
            self.contracts[digest] = contract
            self.task_digests[item["task"]] = digest
        old_view = {**self.view, "revision": self.view["revision"] - 1}
        self.manifest = {"view": self.view["id"], "view_revision": old_view["revision"],
                         "policy_fingerprint": stable_hash(old_view),
                         "tier_requirements": {str(tier): [item["task"] for item in self.view["assignments"]
                             if item["importance"] == "required" and item["qualification_tier"] == tier]
                             for tier in (1, 2, 3)}}
        self.old_row = self.row("selected-a", "family-a", ["PASS"] * len(self.required), source="8" * 64)
        self.old_board = certify({**self.manifest, "rows": [self.old_row]})
        self.old_selection = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
                              "view": self.view["id"], "policy_fingerprint": self.manifest["policy_fingerprint"],
                              "selections": [family_row_pin(self.old_row, selection_kind="current_measurement",
                                  reason="Original policy measurement.", measurement_views=[self.view["id"]])],
                              "historical_selections": []}
        self.round = {"view": self.view["id"], "view_revision": self.view["revision"], "execution_backend": "cuda",
                      "candidate_ids": list(self.declarations), "configuration_ids": ["selected-a"],
                      "required_denominator_by_tier": self.denominator, "selection_rule": "Preserve pre-run choices.",
                      "idea_card_hashes": {name: stable_hash(card) for name, card in self.declarations.items() if name != "selected-a"},
                      "configuration_card_hashes": {"selected-a": stable_hash(self.declarations["selected-a"])}}
        self.staged = certify({"publication_scope": "frozen_source", "execution_backend": "cuda",
                               "frozen_source": {"commit": "a" * 40, "source_digests": ["9" * 64]},
                               "view": self.view["id"], "view_revision": self.view["revision"],
                               "policy_fingerprint": stable_hash(self.view),
                               "tier_requirements": self.manifest["tier_requirements"], "task_contracts": self.contracts,
                               "rows": [self.row("base-a", "family-a", ["PASS"] * len(self.required)),
                                        self.row("selected-a", "family-a", ["FAIL"] + ["PASS"] * (len(self.required) - 1)),
                                        self.row("base-b", "family-b", ["BLOCKED"] + ["PASS"] * (len(self.required) - 1))]})

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, name, family, statuses, *, source="9" * 64):
        observed = dict(zip(self.required, statuses))
        return {"candidate_id": name, "candidate_revision": stable_hash(self.declarations[name]),
                "cohort": stable_hash({"name": name, "source": source}), "trainer_family": family,
                "runtime_cohort": {"execution_backend": "cuda", "model": "metadata-fixture"},
                "bindings": {"source_digest": source, "recipe_sha256": stable_hash(name),
                             "protocol_sha256": "p", "rng_sha256": "r", "task_keys_sha256": "t",
                             "task_contracts": deepcopy(self.task_digests),
                             "recorded_source_origin_commits": ["a" * 40]},
                "attempt_ids": [name + "-attempt"], "qualified_tier": 1 if all(s == "PASS" for s in statuses) else 0,
                "tasks": [{"task_id": item["task"], "status": observed.get(item["task"], "UNKNOWN")}
                          for item in self.view["assignments"] if item["importance"] == "required"],
                "tiers": {str(tier): {"passed": sum(item["qualification_tier"] == tier and observed.get(item["task"]) == "PASS"
                                                    for item in self.view["assignments"] if item["importance"] == "required"),
                                      "total": self.denominator[tier - 1]} for tier in (1, 2, 3)}}

    def call(self, **options):
        return propose(self.root, self.staged, self.old_board, self.old_selection, self.round,
                       self.manifest, [self.old_row], self.declarations, **options)

    def same_view(self):
        self.manifest.update(view_revision=self.view["revision"], policy_fingerprint=stable_hash(self.view),
                             cohorts=[{"source_commit": "8" * 40}])
        self.old_board.update({key: deepcopy(self.manifest[key]) for key in
                               ("view", "view_revision", "policy_fingerprint", "tier_requirements")})
        certify(self.old_board)
        self.old_selection["policy_fingerprint"] = self.manifest["policy_fingerprint"]

    def test_preserves_preselected_failure_instead_of_better_canonical_and_archives_exact_old_row(self):
        before = deepcopy((self.staged, self.old_board, self.old_selection, self.round, self.manifest))
        card, audit = self.call()
        pins = {pin["trainer_family"]: pin for pin in card["selections"]}
        self.assertEqual(pins["family-a"]["candidate_id"], "selected-a")
        self.assertEqual(pins["family-a"]["selection_kind"], "current_measurement")
        self.assertEqual(pins["family-b"]["selection_kind"], "historical_incumbent")
        self.assertEqual(card["historical_selections"][0]["scientific_row_sha256"], scientific_row_hash(self.old_row))
        self.assertNotIn("measurement_views", card["historical_selections"][0])
        self.assertEqual(audit["archived_pins"][0]["original_qualified_tier"], 1)
        self.assertFalse(audit["publication_performed"])
        self.assertEqual(before, (self.staged, self.old_board, self.old_selection, self.round, self.manifest))

    def test_no_task_pooling(self):
        for row in self.staged["rows"][:2]:
            first = row["candidate_id"] == "base-a"
            for index, task in enumerate(row["tasks"]):
                if task["task_id"] in self.required:
                    task["status"] = "PASS" if (index % 2 == 0) == first else "FAIL"
            row["qualified_tier"] = 0
            row["tiers"]["1"]["passed"] = sum(task["status"] == "PASS" for task in row["tasks"]
                                                  if task["task_id"] in self.required)
        certify(self.staged)
        card, audit = self.call()
        pin = next(pin for pin in card["selections"] if pin["trainer_family"] == "family-a")
        selected = next(row for row in self.staged["rows"] if row["candidate_id"] == "selected-a")
        self.assertEqual(pin["scientific_row_sha256"], scientific_row_hash(selected))
        decision = next(item for item in audit["decisions"] if item["trainer_family"] == "family-a")
        self.assertIn("FAIL", decision["required_tier1_statuses"].values())

    def test_unresolved_intended_choice_is_visible_unavailable_and_never_replaced(self):
        self.staged["rows"] = [row for row in self.staged["rows"] if row["candidate_id"] != "selected-a"]
        self.staged["unresolved_configuration_rows"] = [{"candidate_id": "selected-a", "blockers": ["immutable refusal"]}]
        certify(self.staged)
        card, audit = self.call()
        self.assertFalse(any(pin["trainer_family"] == "family-a" for pin in card["selections"]))
        self.assertEqual(audit["unavailable_selections"][0]["candidate_id"], "selected-a")
        self.assertEqual(audit["unavailable_selections"][0]["unresolved_blockers"], ["immutable refusal"])

    def test_duplicate_runtime_cannot_be_selected_silently(self):
        duplicate = deepcopy(self.staged["rows"][1])
        duplicate["runtime_cohort"]["model"] = "other-gpu"
        self.staged["rows"].append(duplicate)
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "multiple exact-source rows"):
            self.call()

    def test_current_contract_mismatch_refuses_pin(self):
        digest = self.task_digests[self.required[0]]
        self.staged["task_contracts"][digest]["timeout_seconds"] += 1
        contract = self.staged["task_contracts"].pop(digest)
        new_digest = stable_hash(contract)
        self.staged["task_contracts"][new_digest] = contract
        for row in self.staged["rows"]:
            row["bindings"]["task_contracts"][self.required[0]] = new_digest
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "current execution, evaluation and budget"):
            self.call()

    def test_tampered_stage_or_old_pin_refused(self):
        self.staged["rows"][1]["qualified_tier"] = 7
        with self.assertRaisesRegex(ValueError, "input digest mismatch"):
            self.call()
        certify(self.staged)
        self.old_selection["selections"][0]["scientific_row_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "old pin needs one exact verified row"):
            self.call()

    def test_preflight_source_without_any_measured_rows_is_refused(self):
        for row in self.staged["rows"]:
            row["attempt_ids"] = []
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "no measured rows"):
            self.call()

    def test_measured_commit_must_come_from_receipts_and_old_history_must_be_unique(self):
        self.staged["frozen_source"]["commit"] = "b" * 40
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "declared executed source commit"):
            self.call()
        self.staged["frozen_source"]["commit"] = "a" * 40
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "old pin needs one exact verified row"):
            propose(self.root, self.staged, self.old_board, self.old_selection, self.round,
                    self.manifest, [self.old_row, deepcopy(self.old_row)], self.declarations)

    def test_configured_standard_is_explicit_and_grants_no_default_adoption(self):
        self.staged["rows"][1] = self.row("selected-a", "family-a", ["PASS"] * len(self.required))
        certify(self.staged)
        card, audit = self.call(configured_standard=True)
        pin = next(pin for pin in card["selections"] if pin["trainer_family"] == "family-a")
        self.assertEqual(pin["selection_kind"], "configured_standard")
        self.assertFalse(card["default_adoption"])
        self.assertFalse(audit["default_adoption"])

    def test_same_view_source_refresh_is_explicit_and_preserves_pre_run_failed_choice(self):
        self.same_view()
        before = deepcopy((self.staged, self.old_board, self.old_selection, self.round, self.manifest))
        with self.assertRaisesRegex(ValueError, "later view revision"):
            self.call()
        card, audit = self.call(refresh_source=True)
        selected = next(pin for pin in card["selections"] if pin["trainer_family"] == "family-a")
        self.assertEqual(selected["candidate_id"], "selected-a")
        self.assertEqual(selected["scientific_row_sha256"], scientific_row_hash(self.staged["rows"][1]))
        self.assertEqual(card["historical_selections"][0]["scientific_row_sha256"], scientific_row_hash(self.old_row))
        self.assertEqual(audit["archived_pins"][0]["original_qualified_tier"], 1)
        self.assertEqual(audit["preparation_mode"], "same_view_source_refresh")
        self.assertEqual(audit["evidence_manifest_unchanged_sha256"], stable_hash(self.manifest))
        self.assertEqual(card["policy_fingerprint"], self.old_selection["policy_fingerprint"])
        self.assertFalse(audit["publication_performed"])
        self.assertFalse(audit["qualification_reuse"])
        self.assertEqual(before, (self.staged, self.old_board, self.old_selection, self.round, self.manifest))

    def test_source_refresh_refuses_policy_changes_and_already_registered_identity(self):
        with self.assertRaisesRegex(ValueError, "identical registered view policy"):
            self.call(refresh_source=True)
        self.same_view()
        self.staged["tier_requirements"] = deepcopy(self.staged["tier_requirements"])
        self.staged["tier_requirements"]["1"] = list(reversed(self.staged["tier_requirements"]["1"]))
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "identical registered view policy"):
            self.call(refresh_source=True)
        self.staged["tier_requirements"] = deepcopy(self.manifest["tier_requirements"])
        certify(self.staged)
        self.manifest["cohorts"][0]["source_commit"] = self.staged["frozen_source"]["commit"]
        with self.assertRaisesRegex(ValueError, "new executed frozen source identity"):
            self.call(refresh_source=True)
        self.manifest["cohorts"][0]["source_commit"] = "8" * 40
        self.staged["frozen_source"]["source_digests"] = [self.old_row["bindings"]["source_digest"]]
        certify(self.staged)
        with self.assertRaisesRegex(ValueError, "new executed frozen source identity"):
            self.call(refresh_source=True)

    def test_source_refresh_verifies_original_origin_and_snapshot_bytes_without_loading(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "fixture.py").write_text("fixture = 1\n")
        files = {"fixture.py": file_hash(snapshot / "fixture.py")}
        source = {"schema_version": 1, "origin_commit": "a" * 40, "files": files, "digest": stable_hash(files)}
        atomic_json(snapshot / "forge-source.json", source)
        row = deepcopy(self.staged["rows"][0])
        row["bindings"]["source_digest"] = source["digest"]
        staged = {"frozen_source": {"commit": source["origin_commit"], "source_digests": [source["digest"]]}, "rows": [row]}
        attempt = row["attempt_ids"][0]
        request_path = self.root / "reports/forge/attempts" / attempt / "request.json"
        request = {"candidate": {"id": row["candidate_id"]}, "candidate_revision": row["candidate_revision"],
                   "source": {**source, "snapshot_path": str(snapshot)}}
        atomic_json(request_path, {"request": request})
        proof_path = self.root / "reports/forge/technique-receipts" / (attempt + ".json")
        proof = {"provenance": {"source_origin_commit": source["origin_commit"], "source_digest": source["digest"]}}
        atomic_json(proof_path, proof)
        verified = _validate_refresh_sources(self.root, staged)
        self.assertEqual(len(verified), 1)
        self.assertTrue(verified[0]["source_bytes_verified"])
        self.assertEqual(verified[0]["manifest_sha256"], file_hash(snapshot / "forge-source.json"))
        proof["provenance"]["source_origin_commit"] = "b" * 40
        atomic_json(proof_path, proof)
        with self.assertRaisesRegex(ValueError, "exact staged receipt/source identity"):
            _validate_refresh_sources(self.root, staged)
        proof["provenance"]["source_origin_commit"] = source["origin_commit"]
        atomic_json(proof_path, proof)
        staged["frozen_source"]["source_digests"].append("0" * 64)
        with self.assertRaisesRegex(ValueError, "manifest without a measured original source"):
            _validate_refresh_sources(self.root, staged)
        staged["frozen_source"]["source_digests"].pop()
        (snapshot / "fixture.py").write_text("fixture = 2\n")
        with self.assertRaisesRegex(ValueError, "source snapshot was changed"):
            _validate_refresh_sources(self.root, staged)


if __name__ == "__main__":
    unittest.main()
