"""Runner/pipeline regression tests: subprocess stubs only, no Torch or GPU.

Run: python -m unittest discover -s tests -p 'test_run_grid.py'
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_grid.py"
GENERATOR = ROOT / "experiments/gen_sparse_configs.py"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = self.root / "calls.jsonl"
        self.trainer = self.root / "trainer.py"
        self.trainer.write_text('''import argparse, json, pathlib, time, yaml
DEFAULTS = {"out_dir": "unused", "coeff": 0.2, "seed": 1, "fail": False,
            "missing_summary": False, "bad_config": False, "empty_final": False, "delay": 0}
parser = argparse.ArgumentParser()
parser.add_argument("--config")
cfg = {**DEFAULTS, **yaml.safe_load(pathlib.Path(parser.parse_args().config).read_text())}
with pathlib.Path(CALLS).open("a") as f:
    f.write(json.dumps(cfg) + "\\n")
time.sleep(cfg["delay"])
out = pathlib.Path(cfg["out_dir"])
out.mkdir(parents=True, exist_ok=True)
if not cfg["missing_summary"]:
    recorded = {**cfg, "coeff": -1} if cfg["bad_config"] else cfg
    (out / "summary.json").write_text(json.dumps({"config": recorded,
        "final": {} if cfg["empty_final"] else {"score": 1}}))
raise SystemExit(9 if cfg["fail"] else 0)
'''.replace("CALLS", repr(str(self.calls))))

    def config(self, name="run", **overrides):
        path = self.root / f"{name}.yaml"
        cfg = {"out_dir": str(self.root / "runs" / name), **overrides}
        path.write_text(yaml.safe_dump(cfg))
        return path, Path(cfg["out_dir"])

    def command(self, pattern="*.yaml", *args):
        return [sys.executable, str(RUNNER), "--configs", str(self.root / pattern),
                "--trainer", str(self.trainer), "--python", sys.executable,
                "--gpus", "0", "--workers_per_gpu", "2", *args]

    def run_grid(self, pattern="*.yaml", *args):
        return subprocess.run(self.command(pattern, *args), cwd=self.root,
                              capture_output=True, text=True, timeout=15)

    def call_count(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def test_reuse_requires_current_complete_config_and_source(self):
        path, out = self.config()
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertTrue((out / "run_grid_complete.json").is_file())
        # Adding an explicitly equal default is the same effective config.
        self.config(coeff=0.2)
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 1)
        self.config(coeff=0.7)
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 2)
        self.assertEqual(json.loads((out / "summary.json").read_text())["config"]["coeff"], 0.7)
        # Unspecified defaults and dirty code changes both invalidate reuse.
        self.trainer.write_text(self.trainer.read_text().replace('"seed": 1', '"seed": 2'))
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 3)
        self.trainer.write_text(self.trainer.read_text() + "\n# changed implementation\n")
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 4)

    def test_legacy_and_tampered_summaries_are_rerun_and_preserved(self):
        _, out = self.config()
        out.mkdir(parents=True)
        (out / "summary.json").write_text('{"config": {}, "final": {"score": 42}}')
        (out / "old_samples.npy").write_text("keep this")
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 1)
        archives = list((out.parent / ".run_grid_history" / out.name).iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "old_samples.npy").read_text(), "keep this")
        self.assertFalse((out / "old_samples.npy").exists())
        saved = json.loads((out / "summary.json").read_text())
        saved["final"]["score"] = 999
        (out / "summary.json").write_text(json.dumps(saved))
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 2)

    def test_failed_force_does_not_leave_reusable_or_analyzable_success(self):
        _, out = self.config()
        self.assertEqual(self.run_grid().returncode, 0)
        self.config(fail=True)
        result = self.run_grid("*.yaml", "--force")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((out / "summary.json").exists())
        self.assertFalse((out / "run_grid_complete.json").exists())
        self.assertTrue((out / "summary.failed.json").exists())
        self.assertTrue((self.root / "results/failures.txt").is_file())
        self.config()
        self.assertEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 3)

    def test_incomplete_or_mismatched_success_is_a_failure(self):
        for name, knob in (("missing", "missing_summary"), ("bad", "bad_config"), ("empty", "empty_final")):
            with self.subTest(knob=knob):
                _, out = self.config(name, **{knob: True})
                self.assertNotEqual(self.run_grid(f"{name}.yaml").returncode, 0)
                self.assertFalse((out / "summary.json").exists())
                self.assertFalse((out / "run_grid_complete.json").exists())

    def test_failures_propagate_while_unrelated_runs_finish(self):
        self.config("fail", fail=True)
        _, good = self.config("good")
        self.assertNotEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 2)
        self.assertTrue((good / "run_grid_complete.json").exists())

    def test_invalid_config_aborts_preflight_without_starting_any_child(self):
        self.config("good")
        bad = self.root / "bad.yaml"
        for value in ("[not: a mapping]", "{bad YAML", "unknown: true\n"):
            with self.subTest(value=value):
                bad.write_text(value)
                self.assertNotEqual(self.run_grid().returncode, 0)
                self.assertEqual(self.call_count(), 0)

    def test_nonpositive_workers_are_rejected_even_in_dry_run(self):
        self.config()
        for workers in ("0", "-1"):
            result = self.run_grid("*.yaml", "--workers_per_gpu", workers, "--dry_run")
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.call_count(), 0)

    def test_duplicate_alias_and_nested_outputs_are_rejected(self):
        _, first = self.config("first")
        self.config("second", out_dir=str(first.parent / "unused" / ".." / first.name))
        self.assertNotEqual(self.run_grid().returncode, 0)
        self.config("second", out_dir=str(first / "nested"))
        self.assertNotEqual(self.run_grid().returncode, 0)
        self.assertEqual(self.call_count(), 0)

    def test_sources_and_unrecognized_nonempty_directories_are_protected(self):
        self.config(out_dir=str(ROOT / "lib"))
        result = self.run_grid()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input/source", result.stderr)
        self.assertTrue((ROOT / "lib/particle_prior.py").is_file())
        _, out = self.config()
        out.mkdir(parents=True)
        (out / "important.txt").write_text("preserve me")
        result = self.run_grid()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without run artifacts", result.stdout)
        self.assertEqual((out / "important.txt").read_text(), "preserve me")
        self.assertEqual(self.call_count(), 0)

    def test_interpreter_provenance_resolves_bare_names_through_path(self):
        _, out = self.config()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        executable = bin_dir / "test-python"
        executable.write_text(f"#!{sys.executable}\nimport os, sys\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
        executable.chmod(0o755)
        cmd = self.command()
        cmd[cmd.index("--python") + 1] = executable.name
        env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "run_grid_complete.json").read_text())
        self.assertEqual(manifest["provenance"]["python"], str(executable.resolve()))

    def test_independent_runners_do_not_share_an_active_output(self):
        self.config(delay=1)
        first = subprocess.Popen(self.command(), cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not self.calls.exists() and first.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            second = self.run_grid()
            self.assertNotEqual(second.returncode, 0)
            stdout, stderr = first.communicate(timeout=10)
            self.assertEqual(first.returncode, 0, stdout + stderr)
            self.assertEqual(self.call_count(), 1)
        finally:
            if first.poll() is None:
                first.kill()
                first.communicate()

    def test_manifest_only_runs_its_listed_configs(self):
        included, _ = self.config("included")
        self.config("stale", fail=True)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps([str(included)]))
        cmd = self.command()
        cmd[2:4] = ["--config_manifest", str(manifest)]
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.call_count(), 1)


class SparsePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def generate(self, stage, *args):
        return subprocess.run([sys.executable, str(GENERATOR), "--stage", stage, *args],
                              cwd=self.root, capture_output=True, text=True, timeout=10)

    def test_base_applies_last_in_every_stage_and_variants_are_isolated(self):
        for stage in ("smoke", "recipe", "ucd", "sparse", "discrete", "champion", "fewshot"):
            with self.subTest(stage=stage):
                result = self.generate(stage, "--base", "coeff=0.37,emb_dim=3,total_steps=7")
                self.assertEqual(result.returncode, 0, result.stderr)
                cfg_dir = next((self.root / "configs/sparse").glob(f"{stage}__*"))
                listed = json.loads((cfg_dir / "manifest.json").read_text())
                for path in listed:
                    cfg = yaml.safe_load((self.root / path).read_text())
                    self.assertEqual((cfg["coeff"], cfg["emb_dim"], cfg["total_steps"]), (0.37, 3, 7))
                    self.assertEqual(Path(cfg["out_dir"]).parent.name, cfg_dir.name)
        self.assertEqual(self.generate("sparse").returncode, 0)
        default_dir = self.root / "configs/sparse/sparse"
        self.assertTrue(default_dir.is_dir())
        self.assertEqual(self.generate("sparse", "--seeds", "9").returncode, 0)
        dirs = list((self.root / "configs/sparse").glob("sparse*"))
        self.assertEqual(len(dirs), 3)

    def test_invalid_generator_inputs_are_errors(self):
        for args in (("--seeds", ""), ("--seeds", "1,1"), ("--base", "seed=2"),
                     ("--base", "bad"), ("--total_steps", "0")):
            self.assertNotEqual(self.generate("sparse", *args).returncode, 0)

    def pipeline_fixture(self):
        experiments = self.root / "experiments"
        experiments.mkdir()
        shutil.copy(ROOT / "experiments/sparse_pipeline.sh", experiments)
        wrapper = self.root / "stub python"
        wrapper.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path("calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\\n")
stage = args[args.index("--stage") + 1] if "--stage" in args else ""
if args[0].endswith("gen_sparse_configs.py"):
    assert stage in {{"recipe", "ucd", "sparse", "discrete", "champion", "fewshot", "smoke"}}
    if "--print_stage" in args:
        print(stage + "__variant")
if os.environ.get("FAIL_AT") and args[0].endswith(os.environ["FAIL_AT"]):
    raise SystemExit(7)
''')
        wrapper.chmod(0o755)
        env = {**os.environ, "PY": str(wrapper)}
        return experiments / "sparse_pipeline.sh", env

    def test_real_shell_pipeline_runs_all_valid_stages_and_uses_resolved_names(self):
        script, env = self.pipeline_fixture()
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        analysis = [call for call in calls if call[0].endswith("analyze_sparse.py")]
        self.assertEqual([call[-1] for call in analysis],
                         [f"{stage}__variant" for stage in ("recipe", "ucd", "sparse", "discrete", "champion", "fewshot")])
        grids = [call for call in calls if call[0].endswith("run_grid.py")]
        self.assertTrue(all("--config_manifest" in call for call in grids))

    def test_real_shell_pipeline_stops_on_a_failed_stage(self):
        script, env = self.pipeline_fixture()
        env["FAIL_AT"] = "run_grid.py"
        result = subprocess.run(["bash", str(script), "recipe", "sparse"], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertFalse(any(call[0].endswith("analyze_sparse.py") for call in calls))
        self.assertFalse(any("sparse" in call[1:] for call in calls))


class ArchiveAnalysisTests(unittest.TestCase):
    def test_archives_are_excluded_from_both_current_run_loaders(self):
        sys.path.insert(0, str(ROOT))
        try:
            from experiments import analyze, analyze_sparse
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            stage = runs_root / "stage"
            (stage / ".run_grid_history" / "old_attempt").mkdir(parents=True)
            (stage / "current_incomplete").mkdir()
            runs, skipped = analyze.load_runs(stage)
            self.assertEqual(runs, [])
            self.assertEqual(skipped["no_summary"], 1)
            with mock.patch.object(analyze_sparse, "RUNS_ROOT", runs_root):
                runs, _, missing = analyze_sparse.load_stage("stage")
            self.assertFalse(runs)
            self.assertEqual(missing, 1)


if __name__ == "__main__":
    unittest.main()
