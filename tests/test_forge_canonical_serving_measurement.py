"""Inert serving control-flow checks; no torch, array, model or evaluator imports."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).parents[1]
namespace = {"__name__": "pure_serving_measurement_source"}
MODULE_BYTES = (ROOT / "experiments/forge/canonical_serving_measurement.py").read_bytes()
exec(compile(MODULE_BYTES, "<private-serving-source>", "exec", dont_inherit=True), namespace)
measure = namespace["measure_served_samples"]
contract = namespace["measurement_contract"]
TRAINING_SOURCE_PATH = ROOT / "particlegan/training.py"
TRAINING_SOURCE_BYTES = TRAINING_SOURCE_PATH.read_bytes()
POLICY_SOURCE_BYTES = (ROOT / "particlegan/policy.py").read_bytes()
tree = ast.parse(TRAINING_SOURCE_BYTES)
trainer_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GANTrainer")
served_method = next(n for n in trainer_class.body if isinstance(n, ast.FunctionDef) and n.name == "served_model")
delegation = {}
exec(compile(ast.Module(body=[served_method], type_ignores=[]), "<exact-public-delegation-only>", "exec", dont_inherit=True), delegation)


class Stream:
    def __init__(self):
        self.draws = 0


class Served:
    def __init__(self, policy):
        self.source = "averaged" if policy.averaging_eligible else "fast"
        self.output_sigma = policy.sigma
        self.completed_steps = policy.completed_steps
        self.row_semantics = "independent"
        self.routing = self.encoder = self.router = self.generation = None
        self.generator, self.critic, self.prior, self.table = object(), object(), object(), object()
        self.policy = policy
        self.sample_calls = []

    def sample(self, n, *, generator, output_noise):
        self.sample_calls.append((n, generator, output_noise))
        generator.draws += n
        if self.policy.mutation:
            self.policy.weights["G"] += 1
        if self.policy.failure:
            raise LookupError("SYNTHETIC sampling failure")
        return SimpleNamespace(shape=(n + self.policy.shape_offset, 2), served_marker=self.source)


class Policy:
    _STREAMS = ("latent_generator", "penalty_generator", "noise_generator", "eval_generator")

    def __init__(self):
        self.recipe = SimpleNamespace(model="gan", conditioning="scalar", encoder_mode="none", output_noise_mode="learned")
        self.row_semantics, self._phase, self.completed_steps = "independent", "ready", 7000
        self.G, self.D, self.prior, self.table = object(), object(), object(), object()
        self.encoder = self.router = self.generation = None
        for name in self._STREAMS:
            setattr(self, name, Stream())
        self.weights = {"G": 4, "D": 5, "prior": 6, "average": 7}
        self.optimizer_steps = {"G": 7000, "D": 7000}
        self.averaging_eligible, self.sigma = False, 0.029
        self.mutation, self.failure, self.shape_offset = False, False, 0
        self.select_calls, self.factory_received = 0, []
        self.served = None
        self.snapshot_override = None

    def served_model(self, *, generation_factory=None):
        self.select_calls += 1
        self.factory_received.append(generation_factory)
        self.served = Served(self)
        if self.snapshot_override:
            self.snapshot_override(self.served)
        return self.served


class Trainer:
    _STREAMS = (*Policy._STREAMS, "model_generator", "input_noise_generator", "prior_noise_generator")
    # Exact public delegation AST from the pinned source, never a model class.
    served_model = delegation["served_model"]

    def __init__(self):
        self.policy = Policy()
        for name in Policy._STREAMS:
            setattr(self, name, getattr(self.policy, name))
        for name in set(self._STREAMS) - set(Policy._STREAMS):
            setattr(self, name, Stream())


def fingerprint(owner):
    policy = getattr(owner, "policy", owner)
    body = {"clock": policy.completed_steps, "phase": policy._phase,
            "weights": policy.weights, "optimizers": policy.optimizer_steps,
            "average_eligible": policy.averaging_eligible, "sigma": policy.sigma,
            "streams": {name: getattr(owner, name).draws for name in owner._STREAMS}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class Controls(unittest.TestCase):
    def call(self, owner, rng=None, **kwargs):
        return measure(owner, 10, dedicated_eval_rng=rng or Stream(),
                       state_fingerprint=fingerprint, expected_step=7000, **kwargs)

    def test_live_owner_alias_names_are_derived_from_public_source(self):
        policy = next(n for n in ast.parse(POLICY_SOURCE_BYTES).body
                      if isinstance(n, ast.ClassDef) and n.name == "UpdatePolicy")
        constructor = next(n for n in policy.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        owned_assignments = []
        for node in ast.walk(constructor):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
                for target, value in zip(node.targets[0].elts, node.value.elts):
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" and isinstance(value, ast.Name):
                        owned_assignments.append((target.attr, value.id))
        self.assertIn(("G", "generator"), owned_assignments)
        self.assertIn(("D", "critic"), owned_assignments)
        self.assertIn(("prior", "prior"), owned_assignments)
        self.assertFalse(hasattr(Policy(), "generator"))
        self.assertFalse(hasattr(Policy(), "critic"))

    def test_actual_public_delegation_and_one_selected_noisy_draw(self):
        owner, rng = Trainer(), Stream()
        before = fingerprint(owner)
        samples, receipt = self.call(owner, rng)
        self.assertEqual(owner.policy.select_calls, 1)
        self.assertEqual(owner.policy.factory_received, [None])
        self.assertEqual(owner.policy.served.sample_calls, [(10, rng, True)])
        self.assertEqual(samples.shape, (10, 2))
        self.assertEqual(receipt["selected_source"], samples.served_marker)
        self.assertEqual(receipt["output_sigma"], 0.029)
        self.assertEqual(rng.draws, 10)
        self.assertEqual(fingerprint(owner), before)
        self.assertIsNone(receipt["numeric_grade"])
        self.assertFalse(receipt["full_protocol_credit"])

    def test_intrinsic_fast_and_averaged_selection_never_metric_choice(self):
        for eligible, wanted in ((False, "fast"), (True, "averaged")):
            owner = Trainer(); owner.policy.averaging_eligible = eligible
            samples, receipt = self.call(owner)
            self.assertEqual(receipt["selected_source"], wanted)
            self.assertEqual(samples.served_marker, wanted)
            self.assertEqual(owner.policy.select_calls, 1)
            self.assertEqual(len(owner.policy.served.sample_calls), 1)

    def test_caller_owned_public_policy_also_uses_its_selector(self):
        policy = Policy()
        _, receipt = self.call(policy)
        self.assertEqual(receipt["selected_source"], "fast")
        self.assertEqual(policy.select_calls, 1)

    def test_rng_alias_to_every_actual_owner_stream_refused_before_selection(self):
        for name in Trainer._STREAMS:
            owner = Trainer()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.call(owner, getattr(owner, name))
            self.assertEqual(owner.policy.select_calls, 0)

    def test_caller_data_rng_alias_refused(self):
        owner, data = Trainer(), Stream()
        with self.assertRaises(ValueError):
            self.call(owner, data, extra_owner_rngs=(data,))
        self.assertEqual(owner.policy.select_calls, 0)

    def test_enumeration_or_context_law_cannot_be_silently_substituted(self):
        for law in ("conditional_context", "routed_context", "enumerated_image_centers"):
            owner = Trainer()
            with self.subTest(law=law), self.assertRaises(ValueError):
                self.call(owner, sampling_law=law)
            self.assertEqual(owner.policy.select_calls, 0)

    def test_independent_marker_cannot_admit_encoder_router_or_live_callback(self):
        for name in ("encoder", "router", "generation"):
            owner = Trainer(); setattr(owner.policy, name, object())
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.call(owner)
            self.assertEqual(owner.policy.select_calls, 0)

    def test_active_update_wrong_clock_and_float_alias_refused(self):
        for phase, step in (("critic", 7000), ("ready", 6999), ("ready", 7000.0)):
            owner = Trainer(); owner.policy._phase, owner.policy.completed_steps = phase, step
            with self.subTest(phase=phase, step=step), self.assertRaises(ValueError):
                self.call(owner)
            self.assertEqual(owner.policy.select_calls, 0)

    def test_live_table_or_model_alias_refused_before_sampling(self):
        for role in ("table", "generator", "critic", "prior"):
            owner = Trainer()
            owner.policy.snapshot_override = lambda served, role=role: setattr(served, role, getattr(owner.policy, {"generator": "G", "critic": "D"}.get(role, role)))
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.call(owner)
            self.assertEqual(owner.policy.served.sample_calls, [])

    def test_nonindependent_snapshot_and_wrong_selected_clock_refused(self):
        for name, value in (("row_semantics", "conditional"), ("routing", object()), ("completed_steps", 6999), ("source", "best_score")):
            owner = Trainer(); owner.policy.snapshot_override = lambda served, name=name, value=value: setattr(served, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.call(owner)
            self.assertEqual(owner.policy.served.sample_calls, [])

    def test_invalid_sigma_refused_but_zero_is_actual_policy_law(self):
        for sigma in (float("nan"), float("inf"), -0.1, True):
            owner = Trainer(); owner.policy.sigma = sigma
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                self.call(owner)
            self.assertEqual(owner.policy.served.sample_calls, [])
        owner = Trainer(); owner.policy.sigma = 0.0
        _, receipt = self.call(owner)
        self.assertTrue(receipt["output_noise"])
        self.assertEqual(receipt["output_sigma"], 0.0)

    def test_mutated_semantic_state_is_rejected_after_draw(self):
        owner = Trainer(); owner.policy.mutation = True
        with self.assertRaisesRegex(RuntimeError, "changed semantic"):
            self.call(owner)
        self.assertEqual(len(owner.policy.served.sample_calls), 1)

    def test_sampling_failure_is_propagated_after_purity_check_no_retry(self):
        owner = Trainer(); owner.policy.failure = True
        with self.assertRaisesRegex(LookupError, "SYNTHETIC"):
            self.call(owner)
        self.assertEqual(owner.policy.select_calls, 1)
        self.assertEqual(len(owner.policy.served.sample_calls), 1)

    def test_wrong_shape_does_not_trigger_another_draw(self):
        owner = Trainer(); owner.policy.shape_offset = -1
        with self.assertRaisesRegex(ValueError, "batch shape"):
            self.call(owner)
        self.assertEqual(len(owner.policy.served.sample_calls), 1)

    def test_contract_has_one_sampling_law_no_numeric_promotions(self):
        declared = contract()
        self.assertTrue(declared["output_noise"])
        self.assertFalse(declared["full_protocol_credit_from_one_observation"])
        self.assertIn("enumerated_image_centers", declared["unsupported_sampling_laws"])
        calls = [node.func.attr for node in ast.walk(ast.parse(MODULE_BYTES))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertEqual(calls.count("served_model"), 1)
        self.assertEqual(calls.count("sample"), 1)
        self.assertFalse(set(calls) & {"step", "backward", "load_state_dict", "generate", "routed_forward"})


if __name__ == "__main__":
    unittest.main()
