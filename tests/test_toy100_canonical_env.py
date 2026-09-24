"""The continuous-learning gate refuses to score without the canonical env."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args, env):
    return subprocess.run(
        [PYTHON, "-m", "benchmarks.toy100.canonical_env", *args],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )


def test_scoring_refuses_without_the_shell_env():
    result = _run(["--check"], os.environ.copy())
    assert result.returncode == 2
    assert "REFUSING TO SCORE" in result.stderr
    assert "scripts/toy100_env.sh" in result.stderr


def test_scoring_refuses_a_partial_pin():
    env = os.environ.copy()
    env.update({
        "TOY100_CANONICAL_ENV": "1",
        "TOY100_SEED": "0",
        "MKL_CBWR": "AVX2",
        "TOY100_CBWR_MODE": "AVX2",
        "TOY100_CBWR_REASON": "incomplete",
        "PYTHONHASHSEED": "0",
    })
    result = _run(["--check"], env)
    assert result.returncode == 2
    assert "REFUSING TO SCORE" in result.stderr
    assert "ATEN_CPU_CAPABILITY" in result.stderr


def test_shell_entry_activates_and_check_prints_a_receipt():
    result = subprocess.run(
        ["bash", "-c", "source scripts/toy100_env.sh && python3 -m benchmarks.toy100.canonical_env --check"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["torch"]
    assert receipt["cpu"]["vendor"]
    assert receipt["cpu"]["model"]
    assert receipt["mkl_version"]
    assert receipt["cbwr_getter"] == "mkl_serv_cbwr_get(-1)"
    assert receipt["cbwr_effective"] == receipt["cbwr_env"]
    assert receipt["cbwr_env"] in ("AVX2,STRICT", "COMPATIBLE")
    assert receipt["isa"]["ATEN_CPU_CAPABILITY"] == "avx2"
    assert receipt["isa"]["MKL_ENABLE_INSTRUCTIONS"] == "AVX2"
    assert receipt["isa"]["ONEDNN_MAX_CPU_ISA"] == "AVX2"
    assert receipt["isa"]["DNNL_MAX_CPU_ISA"] == "AVX2"
    assert receipt["threads"]["torch"] == 1
    assert receipt["threads"]["omp_env"] == "1"
    assert receipt["threads"]["mkl_env"] == "1"
    assert receipt["threads"]["mkl"] == 1
    assert receipt["deterministic"] is True
    assert receipt["seed"] == 0
    assert "AVX2,STRICT" in result.stderr or "COMPATIBLE" in result.stderr


def test_gate_script_refuses_before_creating_output(tmp_path):
    out = tmp_path / "score"
    result = subprocess.run(
        [PYTHON, "reports/toy100/gan_followup_probe.py",
         "--phase", "stay", "--method", "holdw15", "--output", str(out)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "REFUSING TO SCORE" in result.stderr
    assert not out.exists()
