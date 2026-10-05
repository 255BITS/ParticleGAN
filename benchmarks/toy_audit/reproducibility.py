"""Shared defaults for prospective public-API comparisons.

Archived experiments retain their pinned source and original seed. Component
seeds select independent streams within this one protocol, never seed trials.
"""
from contextlib import contextmanager
from functools import wraps
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 0
VERSION = "toy-comparison-v2"


@contextmanager
def construction_rng(seed, device):
    """Seed constructor fixtures without consuming the caller's random state."""
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    device = torch.device(device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def reproducible_execution(function):
    """Fix the entire run, including hosts that consume ambient training RNGs."""
    @wraps(function)
    def execute(*args, **kwargs):
        settings = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with construction_rng(kwargs.get("seed", DEFAULT_SEED), kwargs.get("device", "cpu")):
                return function(*args, **kwargs)
        finally:
            torch.set_num_threads(settings[0])
            torch.use_deterministic_algorithms(settings[1], warn_only=settings[2])
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = settings[3:5]
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = settings[5:7]
            if workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return execute
