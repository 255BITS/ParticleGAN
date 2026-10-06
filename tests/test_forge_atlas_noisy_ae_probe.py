"""Metadata-only fake fences. Authored Source; M has not run these tests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.forge import atlas_noisy_ae_probe as probe


class NoisyAEInitializerProbeTest(unittest.TestCase):
    def test_ordinary_tasks_map_and_cpu_threads_reach_only_metadata_source_boundary(self):
        from experiments.forge import atlas_noisy_ae as adapter
        root = Path(probe.__file__).resolve().parents[2]
        task = {"id": probe.TASK_ID, "resources": {"cpu_threads": 1, "gpus": 0}}
        class AtSourceBoundary(Exception): pass
        # The valid ordinary shape reaches its first metadata boundary. The
        # list shape must refuse before that boundary; no Torch import occurs.
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), patch.object(
                adapter, "source_guard", side_effect=AtSourceBoundary) as boundary:
            with self.assertRaises(ValueError):
                probe.copied_initializer_preflight({"tasks": [task]}, task, root)
            boundary.assert_not_called()
            with self.assertRaises(AtSourceBoundary):
                probe.copied_initializer_preflight({"tasks": {probe.TASK_ID: task}}, task, root)
            boundary.assert_called_once()

    def fakes(self):
        class Model:
            def __init__(self): pass
            def _call_impl(self): pass
        class Optimizer:
            def __init__(self): pass
            def step(self): pass
        class Tensor: pass
        class Context:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __call__(self, function):
                def wrapped(*args, **kwargs):
                    with self: return function(*args, **kwargs)
                return wrapped
        torch = SimpleNamespace(nn=SimpleNamespace(Module=Model),
            optim=SimpleNamespace(Optimizer=Optimizer), Tensor=Tensor, no_grad=Context,
            rand=lambda: None, tensor=lambda: None)
        numpy = SimpleNamespace(random=SimpleNamespace(random=lambda: None))
        return torch, numpy

    def test_wrapper_metadata_creation_is_allowed_but_context_and_target_are_refused(self):
        torch, numpy = self.fakes()
        called = []
        with probe._fences(torch, numpy) as (counts, guard):
            wrapper = torch.no_grad()(lambda: called.append(True))
            self.assertTrue(callable(wrapper))
            self.assertFalse(called)
            self.assertFalse(any(counts.values()))
            with self.assertRaises(ValueError): wrapper()
            self.assertEqual(counts["context_entries"], 1)
            self.assertFalse(called)

    def test_constructors_forward_optimizer_rng_and_array_calls_are_refused(self):
        torch, numpy = self.fakes()
        with probe._fences(torch, numpy) as (counts, guard):
            for kind in ("recipe_constructors", "policy_constructors", "owner_constructors"):
                class Owned:
                    def __init__(self): pass
                guard(Owned, "__init__", kind)
                with self.subTest(kind=kind), self.assertRaises(ValueError): Owned()
                self.assertEqual(counts[kind], 1)
            for function in (torch.nn.Module, lambda: torch.nn.Module._call_impl(None),
                             torch.optim.Optimizer, torch.rand, numpy.random.random, torch.tensor):
                with self.subTest(function=function), self.assertRaises(ValueError): function()
            self.assertEqual(counts["model_constructors"], 1)
            self.assertEqual(counts["forwards"], 1)
            self.assertEqual(counts["updates"], 1)
            self.assertEqual(counts["evaluation_draws"], 2)
            self.assertEqual(counts["array_constructors"], 1)

    def test_exact_guard_call_is_profiled_and_no_factory_or_initializer_is_called(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "copied_initializer_preflight")
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        call = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_loaded_source")
        self.assertEqual([arg.id for arg in call.args[:2]], ["root", "value"])
        self.assertEqual(call.args[2].value, "particlegan/init.py")
        self.assertTrue(any(k.arg == "no_grad" and isinstance(k.value, ast.Attribute)
                            and k.value.attr == "no_grad" for k in call.keywords))
        forbidden = {"Recipe", "UpdatePolicy", "AEOwner", "deterministic_orthogonal_", "value", "unwrapped"}
        self.assertFalse(any(isinstance(node.func, ast.Name) and node.func.id in forbidden for node in calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "setprofile"
                            and node.lineno < call.lineno for node in calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "setprofile"
                            and node.lineno > call.lineno for node in calls))


if __name__ == "__main__":
    unittest.main()
