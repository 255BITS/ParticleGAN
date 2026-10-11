"""CUDA smoke for the four repaired hosts and the grid100 replay checks.

Exits 2 when this process has no CUDA device. On an A6000:

    python -m benchmarks.locked_shared.cuda_host_smoke
"""

from __future__ import annotations

import sys


def main() -> int:
    import os
    from pathlib import Path

    import pytest
    import torch

    if not torch.cuda.is_available():
        print("CUDA is required for this smoke", file=sys.stderr)
        return 2
    os.chdir(Path(__file__).resolve().parents[2])
    return int(pytest.main(["-q", "tests/test_legacy_host_cuda_harness.py"]))


if __name__ == "__main__":
    raise SystemExit(main())
