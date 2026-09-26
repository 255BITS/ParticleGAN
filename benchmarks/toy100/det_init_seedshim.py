"""Shift every torch seed by ``K3P_SEED_OFFSET`` and run the remaining argv.

Init values do not read this offset. Only ``manual_seed`` streams move, which
is the sample and noise randomness. Offset 0 is not routed through this shim.
"""
from __future__ import annotations

import os
import runpy
import sys

import torch

OFFSET = int(os.environ.get("K3P_SEED_OFFSET", "0"))
module = None
if sys.argv[1] == "-m":
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
else:
    script = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = sys.argv[1:]
if OFFSET:
    def shift(seed):
        return (int(seed) + OFFSET) % (2 ** 63)

    def manual_seed(seed):
        return torch.default_generator.manual_seed(shift(seed))

    torch.manual_seed = torch.random.manual_seed = manual_seed
    _Base = torch.Generator

    class Generator(_Base):
        def manual_seed(self, seed):
            return super().manual_seed(shift(seed))

    torch.Generator = Generator
if module is not None:
    runpy.run_module(module, run_name="__main__")
else:
    runpy.run_path(script, run_name="__main__")
