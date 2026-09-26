"""Run a module with every torch seed shifted by a constant offset.

Usage (from the repo root):

    python benchmarks/toy100/seed_offset_run.py OFFSET module args...

Offset 0 is a no-op. A deterministic init ignores this shift. Sample and
noise seeds move. ``torch.manual_seed`` and ``torch.Generator.manual_seed``
both pick up the offset.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
offset = int(sys.argv[1])
module = sys.argv[2]
sys.argv = [module, *sys.argv[3:]]

import torch

_manual = torch.manual_seed
_cuda = torch.cuda.manual_seed
_cuda_all = torch.cuda.manual_seed_all


def manual_seed(seed):
    return _manual(int(seed) + offset)


def cuda_manual_seed(seed):
    return _cuda(int(seed) + offset)


def cuda_manual_seed_all(seed):
    return _cuda_all(int(seed) + offset)


torch.manual_seed = manual_seed
torch.random.manual_seed = manual_seed
torch.cuda.manual_seed = cuda_manual_seed
torch.cuda.manual_seed_all = cuda_manual_seed_all

_Generator = torch.Generator


class _OffsetGenerator(_Generator):
    def manual_seed(self, seed):
        return super().manual_seed(int(seed) + offset)


torch.Generator = _OffsetGenerator

import runpy
runpy.run_module(module, run_name="__main__")
