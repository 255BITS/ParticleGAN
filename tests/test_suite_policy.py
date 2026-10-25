"""Policies of the test suite itself: GPU opt-in, CPU thread budget and CI scope."""
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest
import torch
import yaml

from tests.conftest import thread_budget

ROOT = Path(__file__).resolve().parents[1]
CUDA_SKIP = re.compile(r"not\s+torch\.cuda\.is_available\(\)")


def test_every_cuda_only_test_file_declares_the_cuda_marker():
    """A CUDA skip without the marker runs silently on a workstation GPU."""
    unmarked = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if (CUDA_SKIP.search(source) or "RUN_CUDA_IMAGE_TESTS" in source) and "pytest.mark.cuda" not in source:
            unmarked.append(path.relative_to(ROOT).as_posix())
    assert not unmarked


def test_cuda_marker_is_declared_and_deselected_by_default():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert any(marker.startswith("cuda:") for marker in config["markers"])
    assert "-m 'not cuda'" in config["addopts"]


def test_default_collection_excludes_cuda_tests_and_opt_in_selects_them(tmp_path):
    marked = ROOT / "tests/test_e22_noise_floor.py"
    common = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(marked)]
    default = subprocess.run(common, cwd=ROOT, capture_output=True, text=True, timeout=300)
    opted = subprocess.run(common + ["-m", "cuda"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert default.returncode == 0 and opted.returncode == 0, default.stdout + opted.stdout
    default_ids = [line for line in default.stdout.splitlines() if "::" in line]
    opted_ids = [line for line in opted.stdout.splitlines() if "::" in line]
    assert default_ids and opted_ids
    assert not any("cuda" in line for line in default_ids)
    assert all("cuda" in line for line in opted_ids)


@pytest.mark.parametrize("environ,expected", [
    ({}, None), ({"OMP_NUM_THREADS": "3"}, 3), ({"MKL_NUM_THREADS": "2"}, 2),
    ({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "2"}, 2), ({"OMP_NUM_THREADS": "0"}, None),
    ({"OMP_NUM_THREADS": "many"}, None),
])
def test_thread_budget_respects_omp_and_mkl(environ, expected):
    assert thread_budget(environ) == expected


def test_session_pins_torch_threads_to_the_declared_budget(_pinned_torch_threads):
    budget = thread_budget()
    if budget is not None:
        assert _pinned_torch_threads == budget
    assert torch.get_num_threads() == _pinned_torch_threads


def test_a_test_that_changes_threads_does_not_leak_into_the_next_one_part1():
    torch.set_num_threads(max(1, torch.get_num_threads()) + 3)


def test_a_test_that_changes_threads_does_not_leak_into_the_next_one_part2(_pinned_torch_threads):
    assert torch.get_num_threads() == _pinned_torch_threads


def test_minimum_supported_python_ci_job_runs_pytest():
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    job = workflow["jobs"]["cpu-tests"]
    assert {"python": "3.11", "shard": 0} in job["strategy"]["matrix"]["include"]
    runs = [step for step in job["steps"] if "scripts/ci_pytest.py run" in step.get("run", "")]
    conditions = {step.get("if") for step in runs}
    assert "matrix.python == '3.11'" in conditions and "matrix.python == '3.12'" in conditions
    # Only the complete 3.12 shard set feeds the release coverage receipt check.
    uploads = [step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact")]
    assert all("3.12" in step.get("if", "") for step in uploads)
