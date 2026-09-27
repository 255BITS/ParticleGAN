#!/usr/bin/env python
"""Fresh-Gaussian control: the 100-Gaussian problem with N(0, I) latents.

Identical to ``100gaussians.py --prior fresh_gaussian`` (same recipe, networks
and metrics); only the default prior differs. For the finite frozen-table
control, pass ``--prior frozen_gaussian``.
"""

from importlib import import_module
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_benchmark = import_module("examples.100gaussians")
Gaussians100 = _benchmark.Gaussians100


if __name__ == "__main__":
    raise SystemExit(_benchmark.cli(default_prior="fresh_gaussian"))
