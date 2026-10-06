"""Declaration-only tests; synthetic variants do not represent measured runs.

No model/owner/grader is constructed. ROOT may run this Source under its paid
metadata phase after the new parser and declaration module are installed.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.forge import noisy_prior_tier1 as noisy

ROOT = Path(__file__).resolve().parents[1]


class NoisyPriorTier1DeclarationsTest(unittest.TestCase):
    def parent(self, name):
        return json.loads((ROOT / "configs/forge/tasks" / (name + ".json")).read_text())

    def originals(self):
        return {name: self.parent(name) for name in noisy.PARENTS}

    def fixture(self, directory):
        root = Path(directory)
        parent_dir = root / "configs/forge/tasks"
        parent_dir.mkdir(parents=True)
        for name in noisy.PARENTS:
            (parent_dir / (name + ".json")).write_bytes(
                (ROOT / "configs/forge/tasks" / (name + ".json")).read_bytes())
        protocol = root / noisy.PROTOCOL_PATH
        protocol.parent.mkdir(parents=True)
        protocol.write_bytes((ROOT / noisy.PROTOCOL_PATH).read_bytes())
        return root

    def test_exact_six_keep_every_other_parent_field(self):
        originals = self.originals()
        self.assertEqual(len(originals), 6)
        for name, parent in originals.items():
            with self.subTest(parent=name):
                before = deepcopy(parent)
                task = noisy.make_variant(parent)
                info = noisy.validate(task)
                self.assertFalse(info["ordinary_parent_credit"])
                self.assertFalse(info["owner_compatibility_proven"])
                restored = deepcopy(task)
                restored.pop("task_cohort")
                restored.pop("prior_substitution_parent")
                restored["id"] = name
                restored["execution"]["prior"]["kind"] = parent["execution"]["prior"]["kind"]
                self.assertEqual(restored, parent)
                self.assertEqual(parent, before)
                self.assertEqual(task["evaluation"], parent["evaluation"])
                self.assertEqual(task["resources"], parent["resources"])
        self.assertEqual(sum(x["resources"]["timeout_seconds"] for x in originals.values()), 2220)

    def test_scientific_mutations_refuse(self):
        # Each edit changes a different protected field; no alternate observation
        # or numerical-threshold convention can enter under this cohort label.
        original = noisy.make_variant(self.parent("gaussian1d_acquisition"))
        changes = {
            "sigma": lambda t: t["execution"]["prior"].__setitem__("sigma", 0.03),
            "standardize": lambda t: t["execution"]["prior"].__setitem__("standardize", True),
            "learnable": lambda t: t["execution"]["prior"].__setitem__("learnable", False),
            "zero_alias": lambda t: t["execution"]["prior"].__setitem__("standardize", 0),
            "horizon": lambda t: t["execution"].__setitem__("steps", 80),
            "network": lambda t: t["execution"]["host_definition"].__setitem__("hidden", 64),
            "observer": lambda t: t["evaluation"].__setitem__("scoring_weights", "averaged"),
            "sampling": lambda t: t["evaluation"].__setitem__("sampling_law", "served_noisy"),
            "gate": lambda t: t["evaluation"]["thresholds"][2].__setitem__(2, 999),
            "reads": lambda t: t["evaluation"].__setitem__("observations", 1),
            "cap": lambda t: t["resources"].__setitem__("timeout_seconds", 1500),
            "source": lambda t: t["evaluation"]["sources"].__setitem__("benchmarks/transfer_suite/protocol.py", "0" * 64),
            "seed": lambda t: t.__setitem__("seed", 1),
        }
        for label, change in changes.items():
            with self.subTest(field=label):
                task = deepcopy(original)
                change(task)
                with self.assertRaises(ValueError):
                    noisy.validate(task)

    def test_only_two_administrative_fields_are_excluded(self):
        task = noisy.make_variant(self.parent("two_pole"))
        task["field_ownership"] = {"fixture": "synthetic administrative receipt"}
        task["preflight_blockers"] = ["fixture owner unsupported"]
        noisy.validate(task)
        for key in ("sources", "evaluation_contract", "execution_contract", "runtime"):
            with self.subTest(key=key):
                changed = deepcopy(task)
                changed[key] = {"fixture": True}
                with self.assertRaises(ValueError):
                    noisy.validate(changed)

    def test_identity_and_false_alias_cannot_grant_parent_credit(self):
        task = noisy.make_variant(self.parent("two_pole"))
        for label, change in {
            "parent_id": lambda t: t.__setitem__("id", "two_pole"),
            "other_cohort": lambda t: t.__setitem__("task_cohort", "tier1_policy_selected_cloud_v1"),
            "old_kind": lambda t: t["execution"]["prior"].__setitem__("kind", "particle_cloud"),
            "wrong_pin": lambda t: t["prior_substitution_parent"].__setitem__("json_sha256", "0" * 64),
            "credit": lambda t: t["prior_substitution_parent"].__setitem__("ordinary_parent_credit", True),
            "false_zero": lambda t: t["prior_substitution_parent"].__setitem__("qualification_reuse", 0),
        }.items():
            with self.subTest(label=label):
                changed = deepcopy(task)
                change(changed)
                with self.assertRaises(ValueError):
                    noisy.validate(changed)

    def test_present_cohort_cannot_drop_blocked_hosts_or_add_tasks(self):
        with TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            parents = self.originals()
            self.assertEqual(noisy.load_variants(root, parents), {})
            noisy.write_declarations(root)
            self.assertEqual(len(noisy.load_variants(root, parents)), 6)
            removed = root / noisy.VARIANT_DIRECTORY / ("unused_token_hold" + noisy.SUFFIX + ".json")
            raw = removed.read_bytes()
            removed.unlink()
            with self.assertRaises(ValueError):
                noisy.load_variants(root, parents)
            removed.write_bytes(raw)
            (root / noisy.VARIANT_DIRECTORY / "unrequested.json").write_text("{}")
            with self.assertRaises(ValueError):
                noisy.load_variants(root, parents)

    def test_parent_and_stream_raw_drift_is_a_preflight_failure(self):
        with TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            task = noisy.make_variant(self.parent("two_pole"))
            noisy.validate(task, root=root)
            protocol = root / noisy.PROTOCOL_PATH
            protocol.write_bytes(protocol.read_bytes() + b" ")
            # Declaration loading is still valid; physical preflight refuses.
            noisy.validate(task)
            with self.assertRaises(ValueError):
                noisy.validate(task, root=root)
        with TemporaryDirectory() as temporary:
            root = self.fixture(temporary)
            task = noisy.make_variant(self.parent("two_pole"))
            parent = root / "configs/forge/tasks/two_pole.json"
            parent.write_bytes(parent.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                noisy.validate(task, root=root)

    def test_branch_required_denominator_and_ordinary_policy_are_separate(self):
        view = noisy.branch_view()
        self.assertEqual(view["evidence_scope"], "prior_substitution_variant")
        self.assertIs(view["reporting"]["family_totals"], False)
        self.assertEqual(view["calibration"]["status"], "provisional")
        assignments = view["assignments"]
        self.assertEqual([x["task"] for x in assignments], [name + noisy.SUFFIX for name in noisy.PARENTS])
        self.assertTrue(all(x["qualification_tier"] == 1 and x["importance"] == "required" for x in assignments))


if __name__ == "__main__":
    unittest.main()
