#!/usr/bin/env python
"""Fresh-Gaussian control using exactly the particle example's training recipe.

All options are shared with 100gaussians.py. Only the default prior and output
folder differ. For the finite frozen-table control, pass --prior frozen_gaussian.
"""

from functools import partial
from importlib import import_module
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_benchmark = import_module("examples.100gaussians")
train = partial(_benchmark.train, prior_kind="fresh_gaussian",
                out_dir="100gaussians_no_particles_samples")


def main():
    _benchmark.main(default_prior="fresh_gaussian",
                    default_out_dir="100gaussians_no_particles_samples")


if __name__ == "__main__":
    main()
