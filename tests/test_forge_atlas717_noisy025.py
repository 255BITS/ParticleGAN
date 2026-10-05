"""Software-only declarations/cache tests; no scientific evidence is produced."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.forge import atlas_noisy025_tier1 as metadata
from experiments.forge import atlas_noisy025_adapters as helpers
from experiments.forge import planning, preflight, views
from experiments.forge.contracts import stable_hash

ROOT = Path(__file__).resolve().parents[1]


class Atlas717MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parents = {name: json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())
                       for name in metadata.PARENTS}

    def request(self):
        tasks = {metadata.task_id(name): (deepcopy(task) if name in metadata.DIRECT_CONTROLS
                                         else metadata.make_variant(task))
                 for name, task in self.parents.items()}
        return {"candidate": {"id": metadata.CANDIDATE_ID, "claim_contract": deepcopy(metadata.CLAIM_CONTRACT)},
                "view": metadata.branch_view(), "through_tier": 1, "tasks": tasks,
                "protocol": {"seed": 0}, "source": {"snapshot_path": "/inert-atlas717-snapshot",
                   "files": {preflight.MODULE: "0" * 64}}, "jobs": [{"compatibility_key": "synthetic-unchanged"}]}

    def test_four_sampled_variants_restore_every_original_field(self):
        for name in metadata.SAMPLED_PARENTS:
            with self.subTest(parent=name):
                original = deepcopy(self.parents[name])
                variant = metadata.make_variant(original)
                bound = metadata.validate(variant, root=ROOT)
                self.assertEqual(bound["sigma"], .025)
                self.assertFalse(bound["ordinary_parent_credit"])
                restored = helpers._parent(variant)
                self.assertEqual(restored, original)
                self.assertEqual(self.parents[name], original)
        word = metadata.make_variant(self.parents["five_word_joint_acquisition"])
        self.assertEqual(word["prior_substitution_parent"]["parent_sigma"], 0.)
        self.assertEqual(word["execution"]["prior"]["sigma"], .025)

    def test_original_direct_controls_are_not_noisy_wrappers(self):
        for name in metadata.DIRECT_CONTROLS:
            with self.subTest(control=name):
                original = self.parents[name]
                self.assertEqual(metadata.task_id(name), name)
                self.assertFalse(metadata.is_noisy_task(original))
                self.assertFalse(metadata.validate_control(original, root=ROOT)["latent_prior_sampled"])
                with self.assertRaises(ValueError):
                    metadata.make_variant(original)

    def test_no_other_scientific_axis_can_change(self):
        baseline = metadata.make_variant(self.parents["gaussian1d_acquisition"])
        mutations = [lambda t: t["execution"]["prior"].update(sigma=.03),
                     lambda t: t["execution"].update(steps=t["execution"]["steps"] - 1),
                     lambda t: t["evaluation"]["thresholds"].append(["extra_gate", ">=", 0]),
                     lambda t: t["resources"].update(timeout_seconds=1500),
                     lambda t: t.update(requires_capabilities=[])]
        for mutate in mutations:
            with self.subTest(mutation=mutate.__code__.co_firstlineno):
                task = deepcopy(baseline)
                mutate(task)
                with self.assertRaises(ValueError):
                    metadata.validate(task)

    def test_word_n5_host_and_conditional_law_cannot_be_replaced(self):
        task = metadata.make_variant(self.parents["five_word_joint_acquisition"])
        self.assertEqual(helpers._parent(task), self.parents["five_word_joint_acquisition"])
        candidate = {"id": metadata.CANDIDATE_ID, "recipe_preset": "atlas", "recipe_overrides": {}}
        self.assertIn("N5", " ".join(helpers.blockers(task, candidate)))
        task["execution"]["new_population"] = 11
        with self.assertRaises(ValueError):
            metadata.validate(task)

    def test_full_six_denominator_and_original_caps(self):
        request = self.request()
        metadata.validate_request_scope(request)
        self.assertEqual(len(request["view"]["assignments"]), 6)
        self.assertEqual(sum(t["resources"]["timeout_seconds"] for t in request["tasks"].values()), 2220)
        self.assertTrue(all(a["importance"] == "required" for a in request["view"]["assignments"]))
        self.assertFalse(request["view"]["reporting"]["family_totals"])
        del request["tasks"]["unused_token_hold"]
        with self.assertRaises(ValueError):
            metadata.validate_request_scope(request)

    def test_hashed_track_marker_separates_identical_direct_control_jobs(self):
        candidate = {"id": metadata.CANDIDATE_ID, "recipe_preset": "atlas", "recipe_overrides": {},
                     "claim_contract": deepcopy(metadata.CLAIM_CONTRACT), "resolved_recipe": {"inert_reference": True},
                     "prior": {"kind": "particle_cloud", "sigma": 0., "standardize": False}}
        other = deepcopy(candidate)
        other["id"] = "atlas-existing-mog-tier1-717-v1"
        # The planner omits nominal IDs; the already-hashed claim prevents reuse.
        self.assertEqual(planning.candidate_revision_for("0" * 64, candidate),
                         planning.candidate_revision_for("0" * 64, other))
        other["claim_contract"]["experimental_track"] = "atlas717_existing_mog"
        revisions = [planning.candidate_revision_for("0" * 64, c) for c in (candidate, other)]
        self.assertNotEqual(*revisions)
        for name in metadata.DIRECT_CONTROLS:
            science = {"execution": views.task_execution_fingerprint(self.parents[name]),
                       "evaluation": views.task_evaluation_fingerprint(self.parents[name])}
            keys = [stable_hash({**science, "candidate_revision": revision}) for revision in revisions]
            self.assertNotEqual(*keys)

    def test_missing_or_foreign_track_marker_is_not_our_request(self):
        for marker in (None, "atlas717_existing_mog"):
            request = self.request()
            request["candidate"]["claim_contract"]["experimental_track"] = marker
            with self.assertRaises(ValueError):
                metadata.validate_request_scope(request)

    def test_task_map_cannot_swap_valid_sampled_owners(self):
        request = self.request()
        a, b = [metadata.task_id(name) for name in ("gaussian1d_acquisition", "ring16_acquisition")]
        request["tasks"][a], request["tasks"][b] = request["tasks"][b], request["tasks"][a]
        with self.assertRaises(ValueError):
            metadata.validate_request_scope(request)

    def test_copied_study_closure_includes_declarations_and_complete_catalog(self):
        paths = set(metadata.request_source_paths(ROOT, self.request()["candidate"]))
        for relative in ("configs/forge/defaults.json", "configs/forge/legacy-ideas-v1.json",
                         "configs/forge/ideas/ka2.json", "configs/forge/ideas/" + metadata.CANDIDATE_ID + ".json",
                         str(metadata.STUDY_PATH), str(metadata.PRIOR_EVIDENCE_PATH)):
            self.assertIn(relative, paths)
        catalog = {str(path.relative_to(ROOT)) for path in (ROOT / "configs/forge/tasks").glob("*.json")}
        catalog.update(str(path.relative_to(ROOT)) for path in (ROOT / "configs/forge/task-variants").rglob("*.json"))
        self.assertTrue(catalog <= paths)
        self.assertEqual(metadata.request_source_paths(ROOT, {"id": "unrelated-candidate"}), ())
        with self.assertRaises(ValueError):
            metadata.request_source_paths(ROOT, {"id": metadata.CANDIDATE_ID, "claim_contract": {}})

    def test_admin_cache_is_excluded_but_malformed_admin_is_rejected(self):
        task = metadata.make_variant(self.parents["ae_gan_hold"])
        task.update(preflight_blockers=["Noisy AE metadata must use its frozen owned module"],
                    field_ownership={"synthetic_cached": True})
        metadata.validate(task)
        task["preflight_blockers"] = "invalid cache"
        with self.assertRaises(ValueError):
            metadata.validate(task)

    def test_preflight_recomputes_cache_without_self_lock_or_science_change(self):
        request = self.request()
        for task in request["tasks"].values():
            task.update(preflight_blockers=["Noisy AE metadata must use its frozen owned module"],
                        field_ownership={"stale": True})
        before = deepcopy(request)
        def checked(task, candidate, protocol, *, root, tasks):
            self.assertNotIn("preflight_blockers", task)
            self.assertNotIn("field_ownership", task)
            task["field_ownership"] = {"recomputed": True}
            return ["genuine fixed-host blocker"] if task["id"] == "unused_token_hold" else []
        with patch("experiments.forge.sources.verify_snapshot") as verified, patch.object(Path, "is_file", return_value=True), \
             patch.object(preflight, "task_preflight", side_effect=checked):
            preflight.recheck_request(request)
        verified.assert_called_once()
        for name, task in request["tasks"].items():
            self.assertEqual(metadata._scientific(task), metadata._scientific(before["tasks"][name]))
            self.assertEqual(task["field_ownership"], {"recomputed": True})
        self.assertEqual(request["jobs"], before["jobs"])
        self.assertEqual(request["source"], before["source"])
        self.assertEqual(request["tasks"]["unused_token_hold"]["preflight_blockers"], ["genuine fixed-host blocker"])

    def test_namespace_source_or_mid_recheck_error_never_clears_original_cache(self):
        for failure in ("source", "recheck"):
            with self.subTest(failure=failure):
                request = self.request()
                for task in request["tasks"].values():
                    task["preflight_blockers"] = ["cached owned-module blocker"]
                before = deepcopy(request)
                calls = []
                def checked(*args, **kwargs):
                    calls.append(True)
                    if len(calls) == 2:
                        raise ValueError("foreign owned-module source")
                    return []
                with patch.object(Path, "is_file", return_value=True), \
                     patch("experiments.forge.sources.verify_snapshot", side_effect=ValueError("source drift") if failure == "source" else None), \
                     patch.object(preflight, "task_preflight", side_effect=checked):
                    with self.assertRaises(ValueError):
                        preflight.recheck_request(request)
                self.assertEqual(request, before)

    def test_other_cohorts_keep_the_original_cache_path(self):
        request = self.request()
        request["candidate"]["id"] = "unrelated-existing-candidate"
        request["view"]["id"] = "unrelated-existing-view"
        for task in request["tasks"].values():
            task["preflight_blockers"] = ["original cached value"]
        def original(task, *args, **kwargs):
            self.assertEqual(task["preflight_blockers"], ["original cached value"])
            return ["unchanged original check"]
        with patch.object(Path, "is_file", return_value=True), patch("experiments.forge.sources.verify_snapshot"), \
             patch.object(preflight, "task_preflight", side_effect=original):
            preflight.recheck_request(request)


class OriginalScalarGraderTests(unittest.TestCase):
    def test_final_scalar_pass_is_not_five_check_stability(self):
        task = json.loads((ROOT / "configs/forge/tasks/gaussian1d_acquisition.json").read_text())
        good = {"sample_count": 4096, "finite_fraction": 1., "mean_error_sigma": .01, "std_ratio": 1., "cdf_ks": .01}
        bad = {**good, "cdf_ks": .2}
        steps = task["execution"]["steps"]
        import math
        points = [{"step": math.ceil(i * steps / 24), **(good if i == 24 else bad)} for i in range(1, 25)]
        self.assertEqual(views._transfer(task, {"observations": points, "live": good})["status"], "FAIL")
        missing_count = [{key: value for key, value in point.items() if key != "sample_count"}
                         for point in points]
        self.assertEqual(views._transfer(task, {"observations": missing_count, "live": good})["status"], "INVALID")
        for point in points[-5:]:
            point.update(good)
        self.assertEqual(views._transfer(task, {"observations": points, "live": good})["status"], "PASS")
        self.assertEqual(views._transfer(task, {"observations": points[1:], "live": good})["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
