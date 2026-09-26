"""Run a K3P driver with an optional sample-seed offset.

``K3P_SEED_OFFSET`` is added to every ``manual_seed`` the driver calls. Offset 0
leaves the declared seeds alone. The offset does not enter the initializer:
family E weights are a function of shape and parameter order only.
"""
import os
import runpy
import sys

OFF = int(os.environ.get("K3P_SEED_OFFSET", "0"))
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))

import torch

if OFF:
    def shift(seed):
        return (int(seed) + OFF) % (2 ** 63)

    def manual_seed(seed):
        return torch.default_generator.manual_seed(shift(seed))

    torch.manual_seed = torch.random.manual_seed = manual_seed
    base = torch.Generator

    class Generator(base):
        def manual_seed(self, seed):
            return super().manual_seed(shift(seed))

    torch.Generator = Generator

runpy.run_path(script, run_name="__main__")
