"""Shared fixtures for the ParticleGAN end-to-end suite.

Every e2e test exercises production validators on the repository's real
configuration. There is deliberately no test-local copy of a contract here: a
mirror can only agree with itself and so can never catch a defect.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to ParticleGAN repository root."""
    return ROOT


@pytest.fixture
def run_cli():
    """Execute a CLI command within the repository root environment."""
    def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    return _run
