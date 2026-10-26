"""Storage discipline of what Git tracks, and the Forge declaration check.

These read the Git index (``git ls-files``), never ``git status``: stray local
files, runs or checkpoints in a developer checkout are allowed and must not
fail the suite. Only content that would be committed is checked.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

BULK_SUFFIXES = (".pt", ".pth", ".npz", ".jsonl", ".h5")
MAX_TRACKED_MB = 25


@pytest.fixture(scope="module")
def tracked(repo_root) -> list[str]:
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo_root), capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"not a Git checkout: {error}")
    return [name for name in done.stdout.decode().split("\0") if name]


def test_public_api_reports_track_no_bulk_tensors_or_streams(tracked):
    bulk = [name for name in tracked
            if name.startswith("reports/toy_audit/api_contract/") and name.endswith(BULK_SUFFIXES)]
    assert not bulk


def test_public_api_receipts_are_compact(repo_root, tracked):
    receipts = [name for name in tracked if name.startswith("reports/") and Path(name).name == "receipt.json"]
    oversized = [name for name in receipts if (repo_root / name).stat().st_size > 10 * 1024 * 1024]
    assert not oversized


def test_no_tracked_file_exceeds_the_size_cap(repo_root, tracked):
    cap = MAX_TRACKED_MB * 1024 * 1024
    oversized = [name for name in tracked if (repo_root / name).is_file() and (repo_root / name).stat().st_size > cap]
    assert not oversized


def test_runs_and_editor_artifacts_are_never_tracked(tracked):
    assert not [name for name in tracked if name.startswith("runs/")]
    assert not [name for name in tracked if name.endswith((".swp", ".swo", "~"))]


def test_forge_validate_accepts_the_committed_declarations(run_cli):
    res = run_cli([sys.executable, "-m", "experiments.forge", "validate"], timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
