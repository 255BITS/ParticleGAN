"""Compose the measured truncation rule with public serialized autograd."""
from contextlib import nullcontext
from copy import deepcopy

import torch

from .ring16_spectral_truncation import polar as truncated_polar


def polar(matrix, generator=None):
    if torch.autograd.is_multithreading_enabled():
        raise RuntimeError("combined polar update must execute inside serialized trainer scope")
    return truncated_polar(matrix, generator)


def trainer_candidate(candidate, arm):
    if arm != "every_step":
        raise ValueError("combined study supports continuous execution only")
    result = deepcopy(candidate)
    result["extensions"] = {**result.get("extensions", {}), "serial_backward": True}
    return result


def step_context(arm, update):
    # Public GANTrainer.serial_backward owns the complete update scope.
    return nullcontext()
