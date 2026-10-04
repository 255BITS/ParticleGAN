"""Standalone synthetic controls: standard library, no scientific module imports.

Run alone with --noconftest and plugin autoload disabled for a bounded reporting
check. The maintained renderer is parsed, never imported. Only its pure display
functions execute against private dictionaries and temporary fake declarations.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments/forge/common26_comparison.py"
RENDERER_PATH = ROOT / "reports/forge/regenerate_technique_inventory.py"
spec = importlib.util.spec_from_file_location("private_pure_common26_contract", CONTRACT_PATH)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def load_display(source):
    """Compile only named pure functions; replace the exact lazy pure import."""
    tree = ast.parse(source)
    names = {"_load_common26_display_choices", "_common26_display_projection", "_common26_link",
             "_common26_label", "_common26_reference_card", "_common26_evidence_links",
             "_common26_current_markdown", "_current_markdown", "_current_tier_cell",
             "_ordinary_representation", "_separate_baseline_links", "_shared_score_intro"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = deepcopy(node)
            if node.name == "_common26_display_projection":
                lazy = node.body[1]
                if (not isinstance(lazy, ast.ImportFrom)
                        or lazy.module != "experiments.forge.common26_comparison"
                        or [(item.name, item.asname) for item in lazy.names] != [("project_common26", None)]):
                    raise AssertionError("unexpected pure comparison dependency")
                node.body.pop(1)
            nodes.append(node)
    scope = {"Path": Path, "Counter": Counter, "deepcopy": deepcopy, "hashlib": hashlib,
             "json": json, "os": os, "shlex": shlex, "project_common26": contract.project_common26,
             "CURRENT_PREFIX": Path("reports/forge/technique-inventory"),
             "EVIDENCE_MANIFEST": Path("reports/forge/technique-evidence/manifest.json"),
             "COMMON26_DISPLAY_AUDIT": Path("configs/forge/selections/common26-display-audit-v1.json")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "private-display-functions", "exec"), scope)
    return scope


display = load_display(RENDERER_PATH.read_text())


def fixture():
    return {"view": "discriminator_stability", "view_revision": 3,
            "tier_requirements": {tier: list(tasks) for tier, tasks in contract.TASKS_BY_TIER.items()},
            "rows": [{"trainer_family": family, "candidate_id": "canonical-" + family,
                      "configuration_id": "canonical-" + family, "candidate_revision": "1" * 64,
                      "bindings": {"source_digest": "2" * 64, "recipe_sha256": "3" * 64},
                      "technique": label, "qualified_tier": 3,
                      "tiers": {tier: {"passed": len(tasks), "total": len(tasks)}
                                for tier, tasks in contract.TASKS_BY_TIER.items()},
                      "tasks": [{"task_id": task, "status": "PASS"}
                                for tasks in contract.TASKS_BY_TIER.values() for task in tasks],
                      "selection": {"selection_kind": "historical_incumbent", "qualified": True},
                      "runtime_cohort": {"execution_backend": "cuda"},
                      "cost": {"wall_seconds": 17.0}}
                     for family, label in contract.FAMILIES],
            "evidence_sources": {}, "provenance": {"input_digest": "synthetic-only"}}


def choice(family):
    value = {"candidate_id": "canonical-" + family, "configuration_id": "canonical-" + family,
             "candidate_revision": "1" * 64, "declaration_sha256": "4" * 64,
             "recipe_sha256": "3" * 64, "source_digest": "2" * 64}
    if family == "atlas":
        value["base_config"] = deepcopy(contract.ATLAS_CONFIG)
    return value


def references():
    return {"original_pr223_atlas": {"counts": {"PASS": 19},
                    "readout": "reports/forge/old-original/README.md",
                    "fresh_retest": {"readout": "reports/forge/stopped/README.md", "counts": {"PASS": 16}},
                    "native3_continuation": {"readout": "reports/forge/invalid/README.md"},
                    "native3_repaired_continuation": {"readout": "reports/forge/repaired/README.md"}},
            "atlas_unblocking_progress": {
                "baseline": {"readout": "reports/forge/c6/README.md", "counts": {"PASS": 7, "FAIL": 11}},
                "adaptations": [{"family": "atlas_conditional", "readout": "reports/forge/conditional/README.md",
                                  "passed": 4, "required": 26}],
                "word": {"readout": "reports/forge/word-context/README.md", "status": "INVALID"}},
            "word_half_base_diagnostic": {"readout": "reports/forge/half-base/README.md", "passed": 0},
            "baseline_debugging": {"readout": "reports/forge/debug/README.md"},
            "completed_api_studies": {"rows": [{"id": "archive", "readout": "reports/forge/archive/README.md",
                                                 "label": "19/19 winner", "required_cells": 19}]}}


class FreshCommon26DisplayControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="forge-common26-private-fake-")
        self.root = Path(self.temporary.name)
        self.path = self.root / "reports/forge/technique-inventory.md"
        self.result = fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def render(self, result=None):
        return display["_current_markdown"](result or self.result, self.root, self.path)

    def write_audit(self, choices, **extra):
        path = self.root / display["COMMON26_DISPLAY_AUDIT"]
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema": "pg_common26_display_identity_audit_v1", "chosen_configurations": choices, **extra}
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return path

    def test_historical_all_pass_rows_and_linked_scopes_never_fill_fresh_cells(self):
        self.result.update(references())
        before = deepcopy(self.result)
        text = self.render()
        table = [line for line in text.splitlines() if line.startswith("|")]
        self.assertEqual(len(table), 14)
        self.assertEqual(table[0], "| Model/configuration | Representation | Fresh common-26 score/status |")
        self.assertEqual(text.count("NOT_RUN (26 required)"), 12)
        self.assertEqual(text.count("| --- | --- | --- |"), 1)
        for forbidden in ("19/19", "7/26", "4/26", "0/26", "**PASS", "Recorded tier"):
            self.assertNotIn(forbidden, text)
        leading, tail = text.split("## Qualification and scope\n", 1)
        for readout in ("old-original", "stopped", "invalid", "repaired", "c6", "conditional",
                        "word-context", "half-base", "debug", "archive"):
            self.assertIn(f"]({readout}/README.md)", leading)
            self.assertNotIn(f"]({readout}/README.md)", tail)
        self.assertEqual(self.result, before)
        projection = display["_common26_display_projection"](self.result, self.root)
        self.assertTrue(all(row["fresh_common"] == contract.UNSCORED for row in projection["rows"]))

    def test_null_identity_shows_reference_and_exact_requested_atlas_without_alias(self):
        text = self.render()
        atlas = next(line for line in text.splitlines() if line.startswith("| Full Atlas ·"))
        self.assertIn("configs/100gaussians/atlas.json@a3ee5c67ac65", atlas)
        self.assertIn("Configuration freeze pending", atlas)
        self.assertIn("Particles (declared)", atlas)
        self.assertIn("Eligibility: BLOCKED", atlas)
        self.assertNotIn("canonical-atlas", atlas)
        self.assertIn("canonical reference: canonical-r1r2", text)
        self.assertIn("configs/forge/ideas/canonical-atlas.json", text)
        self.assertIn("Differing canonical Atlas reference", text)
        self.assertEqual(sum(line.startswith("| Full Atlas ·") for line in text.splitlines()), 1)

    def test_wrong_roster_view_revision_or_order_is_refused(self):
        changes = [lambda value: value["rows"].pop(),
                   lambda value: value["rows"].append(deepcopy(value["rows"][0])),
                   lambda value: value["rows"][0].update(trainer_family="unregistered"),
                   lambda value: value.update(view="quality_coverage"),
                   lambda value: value.update(view_revision=2),
                   lambda value: value.update(view_revision=True),
                   lambda value: value["tier_requirements"]["1"].reverse()]
        for change in changes:
            value = deepcopy(self.result)
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                display["_common26_display_projection"](value, self.root)

    def test_current_campaign_name_or_flags_cannot_enable_numeric_acceptance(self):
        self.result.update(CURRENT=True, fresh=True, campaign_id="brand-new", created_at="2099-01-01")
        self.result["rows"][0].update(current=True, fresh=True, accepted=True)
        projection = display["_common26_display_projection"](self.result, self.root)
        self.assertEqual(projection["numerical_acceptance"], "UNIMPLEMENTED")
        self.assertTrue(all(row["fresh_common"]["passed"] is None for row in projection["rows"]))
        self.result["rows"][0]["fresh_common"] = {**contract.UNSCORED, "passed": 26, "status": "PASS"}
        with self.assertRaises(ValueError):
            self.render()

    def test_cached_numbers_are_refused_and_cached_identities_not_used(self):
        cached = display["_common26_display_projection"](self.result, self.root)
        cached["rows"][0]["chosen_configuration"] = {"candidate_id": "forged-cached-name"}
        cached["rows"][0]["representation"] = {"kind": "Particles", "basis": "DECLARED"}
        self.result["common26_display"] = cached
        text = self.render()
        self.assertNotIn("forged-cached-name", text)
        self.assertIn("| UNKNOWN | NOT_RUN", text)
        cached["rows"][0]["fresh_common"]["passed"] = 1
        with self.assertRaises(ValueError):
            self.render()

    def test_manifest_hash_is_injected_without_self_reference_and_grants_no_score(self):
        path = self.write_audit({"r1r2": choice("r1r2")})
        raw = path.read_bytes()
        choices = display["_load_common26_display_choices"](self.root)
        self.assertEqual(choices["r1r2"]["audit_reference"], {
            "path": display["COMMON26_DISPLAY_AUDIT"].as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        self.assertNotIn("audit_reference", json.loads(raw)["chosen_configurations"]["r1r2"])
        projected = display["_common26_display_projection"](self.result, self.root)
        self.assertIsNone(projected["rows"][0]["fresh_common"]["passed"])
        self.assertEqual(projected["rows"][0]["representation"]["kind"], "UNKNOWN")
        self.assertIn("common26-display-audit-v1.json", self.render())

    def test_malformed_partial_self_pinned_or_scoring_manifest_is_refused(self):
        values = [({"r1r2": {**choice("r1r2"), "candidate_revision": None}}, {}),
                  ({"r1r2": {**choice("r1r2"), "source_digest": None}}, {}),
                  ({"r1r2": {**choice("r1r2"), "audit_reference": {}}}, {}),
                  ({"r1r2": {**choice("r1r2"), "passed": 26}}, {}),
                  ({"unknown-family": choice("r1r2")}, {}),
                  ({"atlas": {**choice("atlas"), "base_config": {"path": "configs/forge/ideas/atlas.json"}}}, {}),
                  ({}, {"freshness_approved": True})]
        for choices, extras in values:
            self.write_audit(choices, **extras)
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                self.render()
        path = self.write_audit({})
        path.write_text('{"schema":"pg_common26_display_identity_audit_v1","schema":"duplicate","chosen_configurations":{}}')
        with self.assertRaises(ValueError):
            self.render()

    def test_only_exact_content_bound_prior_declarations_label_representation(self):
        self.write_audit({"r1r2": choice("r1r2")})
        priors = [{"prior": {"kind": "mog"}}, {"prior": {"kind": "particles"}}]
        pins = [contract._digest(value) for value in priors]
        self.result["task_contracts"] = dict(zip(pins, priors))
        tasks = [task for tier in contract.TASKS_BY_TIER.values() for task in tier]
        row = self.result["rows"][0]
        row["bindings"]["task_contracts"] = {task: pins[index % 2] for index, task in enumerate(tasks)}
        projection = display["_common26_display_projection"](self.result, self.root)
        self.assertEqual(projection["rows"][0]["representation"], {"kind": "MoG / Particles", "basis": "DECLARED"})
        self.assertIsNone(projection["rows"][0]["fresh_common"]["passed"])
        row["bindings"]["source_digest"] = "f" * 64
        self.assertEqual(display["_common26_display_projection"](self.result, self.root)["rows"][0]["representation"]["kind"], "UNKNOWN")
        row["bindings"]["source_digest"] = "2" * 64
        self.result["task_contracts"][pins[0]]["prior"]["kind"] = "particles"
        self.assertEqual(display["_common26_display_projection"](self.result, self.root)["rows"][0]["representation"]["kind"], "UNKNOWN")

    def test_unsafe_source_scoped_link_is_refused(self):
        for path in ("../outside.md", "/absolute.md", "https://foreign", "reports\\outside.md"):
            self.result["original_pr223_atlas"] = {"readout": path}
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render()

    def test_standalone_api_verdict_and_metrics_remain_only_linked_evidence(self):
        score = {"trainer_family": "k3p", "case": {"title": "Gaussian histogram question"},
                 "run": {"verdict": "PASS", "final_metrics": {"cdf_ks": 0.000001}},
                 "readout": "reports/toy_audit/api_contract/separate/README.md",
                 "gif": "reports/toy_audit/api_contract/separate/goal.gif"}
        self.result["standalone_api_scores"] = [score]
        self.result["declared_view"] = {"revision": 4, "added_required_tasks": {"1": ["extra-question"]}}
        self.result["publication_refresh"] = {"scientific_rows_preserved": True}
        before = deepcopy(self.result)
        text = self.render()
        leading = text.split("## Qualification and scope\n", 1)[0]
        self.assertIn("Standalone API evidence: k3p · Gaussian histogram question", leading)
        self.assertIn("actual-training GIF", leading)
        self.assertIn("toy_audit/api_contract/separate/README.md", leading)
        self.assertIn("#declared_view", leading)
        self.assertIn("--refresh-publication", text)
        self.assertNotIn("1/1", text)
        self.assertNotIn("cdf_ks", text)
        self.assertNotIn("0.000001", text)
        self.assertEqual(text.count("NOT_RUN (26 required)"), 12)
        self.assertEqual(self.result, before)

    def test_nonordinary_scopes_do_not_invoke_comparison(self):
        for view, recorded in (("quality_coverage", None),
                               ("discriminator_stability", "configs/forge/view-history/v2.json")):
            value = deepcopy(self.result)
            value.update(view=view, view_revision=2)
            if recorded:
                value["recorded_policy"] = recorded
            def forbidden(*args, **kwargs):
                self.fail("other-view rendering invoked comparison")
            saved = display["_common26_display_projection"]
            display["_common26_display_projection"] = forbidden
            try:
                original = display["_current_markdown"](value, self.root, self.path)
                # Only the existing callbacks/data apply to the preserved branch.
                value["atlas_unblocking_progress"] = references()["atlas_unblocking_progress"]
                value["word_half_base_diagnostic"] = references()["word_half_base_diagnostic"]
                value["original_pr223_atlas"] = references()["original_pr223_atlas"]
                self.assertEqual(display["_current_markdown"](value, self.root, self.path), original)
            finally:
                display["_common26_display_projection"] = saved

    def test_shared_index_reuses_leading_table_links_and_preserves_accounting_scope(self):
        path = self.root / "reports/forge/shared-score-index-20261003/README.md"
        path.parent.mkdir(parents=True)
        path.write_text("private archived-boundary fixture")
        self.result.update(references())
        self.result["word_half_base_diagnostic"].update(
            accounting={"inclusive_charged_seconds": 21, "gpu0_charged_seconds": 8, "gpu1_charged_seconds": 13},
            cost={"paid_seconds": 7})
        calls = []
        display["_shared_score_archive"] = lambda root: calls.append(root)
        actual_path, content = display["_shared_score_intro"](self.root, self.result)
        expected = display["_current_markdown"](self.result, self.root, path).split("## Qualification and scope\n", 1)[0]
        expected = expected.replace("# Current model/configuration scores", "# Shared ParticleGAN score index", 1)
        self.assertTrue(content.startswith(expected))
        self.assertEqual(actual_path, path)
        self.assertEqual(content.count("NOT_RUN (26 required)"), 12)
        self.assertIn("21 / 10500 seconds", content)
        self.assertEqual(calls, [self.root])


if __name__ == "__main__":
    unittest.main()
