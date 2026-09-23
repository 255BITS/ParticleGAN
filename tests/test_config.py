"""TOML/YAML and recipe migration parity, without running experiments."""
import inspect
import importlib.util
from pathlib import Path
import tempfile
import unittest

from experiments.config import read_config, recipe_defaults
from experiments.run_grid import code_provenance, load_config, trainer_defaults

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_toml_yaml_match_and_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            toml = Path(temp) / "config.toml"
            yaml = Path(temp) / "config.yaml"
            toml.write_text('out_dir = "runs/custom"\nlr = 0.0003\nalpha_bar = [1.0, 0.5, 0.01]\n')
            yaml.write_text('out_dir: runs/custom\nlr: 0.0003\nalpha_bar: [1.0, 0.5, 0.01]\n')
            self.assertEqual(read_config(toml), read_config(yaml))
            defaults = dict(out_dir="unused", lr=.0006, alpha_bar=[], batch_size=256)
            self.assertEqual(load_config(toml, defaults), load_config(yaml, defaults))
            self.assertEqual(load_config(toml, defaults)["batch_size"], 256)
            toml.write_text('out_dir = "runs/custom"\nmisspelled_lr = 1\n')
            with self.assertRaisesRegex(ValueError, "unknown config keys"):
                load_config(toml, defaults)
            yaml.write_text('- not\n- a mapping\n')
            with self.assertRaisesRegex(ValueError, "mapping"):
                read_config(yaml)

    def test_comparison_collects_toml_and_verifies_original_bytes(self):
        import hashlib
        import json
        from experiments.compare_priors import collect
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            config = out / "input.toml"
            config.write_text('prior = "particles"\nseed = 1\nout_dir = ' + json.dumps(str(out)) + '\n')
            expected = read_config(config)
            (out / "summary.json").write_text(json.dumps({
                "config": expected, "final": {"modes": 100}, "collapse_events": []}))
            manifest = {"runs": [{**expected, "config": str(config),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}]}
            self.assertEqual(collect(manifest)[0]["final"], {"modes": 100})
            config.write_text(config.read_text() + "# Edited bytes\n")
            with self.assertRaisesRegex(ValueError, "Config changed"):
                collect(manifest)

    def test_primary_defaults_and_shipped_toml_match(self):
        from particlegan import get_recipe
        for trainer, name in (("train_100gaussians", "100gaussians"),
                              ("train_denoising", "denoising")):
            defaults = trainer_defaults(str(ROOT / "experiments" / f"{trainer}.py"))
            supplied = read_config(ROOT / "configs" / name / "default.toml")
            # TOML has no null literal; optional None defaults stay omitted.
            self.assertEqual(set(defaults) - set(supplied),
                             {key for key, value in defaults.items() if value is None})
            self.assertEqual(defaults, load_config(ROOT / "configs" / name / "default.toml", defaults))
            recipe = get_recipe()
            self.assertEqual(defaults["lr"], recipe.lr)
            self.assertEqual(defaults["batch_size"], recipe.batch_size)
            self.assertEqual(defaults["z_dim"], recipe.z_dim)
            self.assertEqual(defaults["num_particles"], recipe.num_particles)
        historical = read_config(ROOT / "configs/denoising/ddgan_ucd.yaml")
        self.assertEqual(load_config(ROOT / "configs/denoising/ddgan_ucd.yaml", defaults),
                         {**defaults, **historical})
        self.assertEqual(historical["lr"], .0006)  # Explicit old run configuration.

    def test_example_signature_matches_recipe(self):
        spec = importlib.util.spec_from_file_location("example100", ROOT / "examples/100gaussians.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parameters = inspect.signature(module.train).parameters
        for key, value in recipe_defaults("100gaussians").items():
            self.assertEqual(parameters[key].default, value, key)

    def test_runner_rejects_arbitrary_dynamic_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = Path(temp) / "trainer.py"
            trainer.write_text('DEFAULTS = dict(out_dir="run")\n')
            with self.assertRaisesRegex(ValueError, "literal mapping"):
                trainer_defaults(str(trainer))
            trainer.write_text('DEFAULTS = {**untrusted_helper("denoising")}\n')
            with self.assertRaisesRegex(ValueError, "literal mapping"):
                trainer_defaults(str(trainer))

    def test_provenance_includes_core_and_config(self):
        import sys
        provenance = code_provenance(str(ROOT / "experiments/train_denoising.py"), sys.executable)
        self.assertIn("experiments/config.py", provenance["sources"])
        self.assertIn("particlegan/recipes.py", provenance["sources"])
        self.assertIn("particlegan/diffusion.py", provenance["sources"])


if __name__ == "__main__":
    unittest.main()
