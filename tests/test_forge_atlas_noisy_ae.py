"""Synthetic metadata/ownership tests only; no AE model or numerical evidence.

These tests are AUTHORED_NOT_RUN by M. They import inert definitions and use
ordinary metadata fakes; ROOT owns any future paid execution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.forge import atlas_noisy_ae as ae
from experiments.forge import noisy_prior_tier1 as noisy

ROOT = Path(__file__).resolve().parents[1]


class NoisyAEGuardsTest(unittest.TestCase):
    def task(self):
        return noisy.make_variant(json.loads((ROOT / "configs/forge/tasks/ae_gan_hold.json").read_text()))

    def fixture(self):
        # No score or sampled cloud is fabricated. Only owner-control metadata
        # is tested here; grade_result still needs all original metric values.
        files = deepcopy(ae.SOURCE_PINS)
        source = Path(ae.__file__).resolve()
        files["experiments/forge/atlas_noisy_ae.py"] = dict(
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)
        contract = dict(files=files, task_id=ae.TASK_ID, parent_task_id="ae_gan_hold",
            external_max_steps=250, actual_prior=deepcopy(ae.ACTUAL_PRIOR),
            observation=dict(observations=deepcopy(ae.CLOCKS),
                             selected_or_averaged_sampling=False, evaluation_DV12=False))
        recipe = deepcopy(ae.RESOLVED_RECIPE)
        hooks = {name: 250 for name in ("begin_step", "before_critic_backward", "after_critic_step",
            "before_generator_backward", "after_generator_backward", "after_generator_step", "finish_step")}
        receipt = dict(schema="forge_atlas_noisy_ae_evidence_v1", task_id=ae.TASK_ID,
            actual_prior=deepcopy(ae.ACTUAL_PRIOR),
            public_prior_type="particlegan.noisy_particle_prior.NoisyParticlePrior",
            table_shape=[12, 2], same_location_parameter=True, table_optimizer_alias=True,
            initial_completed_steps=0, initial_optimizer_state_entries={"generator": 0, "discriminator": 0},
            completed_steps=250, full_recipe=recipe, full_recipe_sha256=ae.digest(recipe),
            source_contract=contract, source_contract_sha256=ae.digest(contract),
            prior_kernel=dict(kind="noisy_particle_cloud", sigma=.02500000037252903, standardize=False,
                learned_width=False, row_weights="uniform",
                code_path="particlegan.noisy_particle_prior.NoisyParticlePrior"),
            ordered_lifecycle_calls=hooks)
        return dict(noisy_prior686=receipt, scoring_weights="live",
            sampling_law="generated_and_reconstructed_prior_with_scheduled_output_noise",
            eval_output_noise="public_recipe_schedule", observations=[{"step": x} for x in ae.CLOCKS],
            measurement_purity=[dict(step=x, pure=True, evaluation_DV12=False, unintended_rng_deviations=0)
                                for x in [0, *ae.CLOCKS]],
            guards=dict(all_finite=True, public_policy_state_finite=True, hooks_exercised=True,
                unintended_rng_deviations=0,
                optimizer_updates={role:250 for role in ("generator", "encoder", "prior", "discriminator", "noise")}))

    def rejected(self, evidence):
        self.assertEqual(ae.validate_evidence(self.task(), evidence)["status"], "INVALID")

    def test_current_metadata_wire_preserves_exact_original_task(self):
        task = self.task()
        self.assertEqual(ae.blockers(task, {"recipe_preset":"atlas", "recipe_overrides":{}}), [])
        self.assertEqual(len(ae.RESOLVED_RECIPE), 79)
        self.assertEqual(ae.RESOLVED_RECIPE["name"], "atlas")
        self.assertIsNone(ae.RESOLVED_RECIPE["total_steps"])
        self.assertEqual(ae.RESOLVED_RECIPE["encoder_mode"], "none")
        self.assertEqual(ae.RESOLVED_RECIPE["prior_kind"], "noisy_particles")
        self.assertIs(ae.RESOLVED_RECIPE["standardize"], False)
        self.assertEqual(ae.TASK_BINDINGS["num_particles"], 12)
        self.assertEqual(ae.TASK_BINDINGS["z_dim"], 2)
        self.assertEqual(ae.TASK_BINDINGS["batch_size"], 64)
        self.assertEqual(task["resources"]["timeout_seconds"], 300)
        self.assertEqual(task["execution"]["steps"], 250)
        self.assertEqual(task["evaluation"]["observations"], 24)
        self.assertEqual(task["evaluation"]["minimum_stable_checks"], 5)

    def test_actual_float32_width_metadata_and_changed_width(self):
        evidence = self.fixture()
        self.assertEqual(evidence["noisy_prior686"]["prior_kernel"]["sigma"], .02500000037252903)
        self.assertIsNone(ae.validate_evidence(self.task(), evidence))
        evidence["noisy_prior686"]["prior_kernel"]["sigma"] = .03
        self.rejected(evidence)

    def test_metadata_acceptance_is_no_numerical_grade(self):
        fixture = self.fixture()
        self.assertIsNone(ae.validate_evidence(self.task(), fixture))
        self.assertFalse(any("recon_mse" in row or "hold" in row for row in fixture["observations"]))

    def test_prior_source_recipe_alias_drift_refuses_even_rehashed(self):
        for label, change in {
            "sigma": lambda e: e["noisy_prior686"]["prior_kernel"].__setitem__("sigma", .04),
            "std": lambda e: e["noisy_prior686"]["prior_kernel"].__setitem__("standardize", True),
            "actual_type": lambda e: e["noisy_prior686"].__setitem__("public_prior_type", "MoGParticlePrior"),
            "table_copy": lambda e: e["noisy_prior686"].__setitem__("same_location_parameter", False),
            "table_opt": lambda e: e["noisy_prior686"].__setitem__("table_optimizer_alias", False),
            "recipe": lambda e: e["noisy_prior686"]["full_recipe"].__setitem__("encoder_mode", "ae"),
            "source": lambda e: e["noisy_prior686"]["source_contract"]["files"]["particlegan/recipes.py"].__setitem__("sha256", "0"*64),
            "stale_loss_source": lambda e: e["noisy_prior686"]["source_contract"]["files"]["particlegan/gan_loss.py"].__setitem__("sha256", "1c1019dfe71c583e32a05df0d2f794f9fff6d9ee1ea0332f57f6379ae70cf6b7"),
        }.items():
            with self.subTest(label=label):
                e = self.fixture(); change(e)
                r = e["noisy_prior686"]
                r["source_contract_sha256"] = ae.digest(r["source_contract"])
                r["full_recipe_sha256"] = ae.digest(r["full_recipe"])
                self.rejected(e)

    def test_original_observation_and_seven_hooks_are_mandatory(self):
        for label, change in {
            "selected": lambda e: e.__setitem__("scoring_weights", "averaged"),
            "new_observer": lambda e: e.__setitem__("sampling_law", "served_noisy"),
            "new_noise": lambda e: e.__setitem__("eval_output_noise", "clean"),
            "missing_read": lambda e: e["observations"].pop(),
            "initial_purity": lambda e: e["measurement_purity"].pop(0),
            "dv12_eval": lambda e: e["measurement_purity"][2].__setitem__("evaluation_DV12", True),
            "drift": lambda e: e["measurement_purity"][2].__setitem__("pure", False),
            "encoder_clock": lambda e: e["guards"]["optimizer_updates"].__setitem__("encoder", 249),
            "hook": lambda e: e["noisy_prior686"]["ordered_lifecycle_calls"].__setitem__("after_generator_step", 249),
            "nonfinite": lambda e: e["guards"].__setitem__("public_policy_state_finite", False),
            "float_alias": lambda e: e["observations"][-1].__setitem__("step", 250.0),
            "zero_alias": lambda e: e["noisy_prior686"].__setitem__("initial_completed_steps", False),
        }.items():
            with self.subTest(label=label):
                e = self.fixture(); change(e); self.rejected(e)

    def test_tuning_and_different_host_cannot_enter_ae_bridge(self):
        task = self.task()
        for candidate in ({"recipe_preset":"atlas", "recipe_overrides":{"lr":.01}},
                {"recipe_preset":"atlas", "extensions":{"encoder":{}}},
                {"recipe_preset":"bcap"}, {"recipe_preset":"atlas", "implementation":"custom"}):
            self.assertTrue(ae.blockers(task, candidate))
        task["execution"]["host"] = "mode_hold"
        self.assertTrue(ae.blockers(task, {"recipe_preset":"atlas"}))

    def test_one_factory_and_witnesses_precede_any_owner_use(self):
        # A Python object and metadata-only reader replace the real admitted
        # factory; no initializer/Torch/optimizer/forward can be invoked here.
        owner = object.__new__(ae.AEOwner)
        owner.restored = False
        calls, writes = [], []
        def factory():
            calls.append("factory"); return owner
        def reader(value):
            self.assertIs(value, owner); calls.append("reader")
            return {"completed_steps":0, "fixture":"not a measured owner"}
        fresh = ae._OrdinaryFreshOwner(lambda: calls.append("source"), {"fixture":"admission"}, Path("/fixture"))
        with patch.object(ae, "owner_initial_receipt", reader), patch("experiments.forge.contracts.atomic_json", side_effect=lambda path,value: writes.append((path.name,value))), patch("builtins.print"):
            self.assertIs(fresh.construct(factory, reader), owner)
            self.assertEqual([name for name,value in writes], ["INITIALIZATION.json", "MODEL_STARTED.json"])
            self.assertEqual(writes[-1][1]["completed_steps"], 0)
            with self.assertRaises(ValueError): fresh.construct(factory, reader)
        self.assertEqual(calls.count("factory"), 1)


if __name__ == "__main__":
    unittest.main()
