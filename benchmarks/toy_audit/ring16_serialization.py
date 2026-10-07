"""Execution-only hooks for the shared fresh-live Ring16 diagnostic runner."""
from contextlib import nullcontext
from copy import deepcopy

import torch


def trainer_candidate(candidate, arm):
    result = deepcopy(candidate)
    if arm == "every_step":
        result["extensions"] = {**result.get("extensions", {}), "serial_backward": True}
    return result


def step_context(arm, update):
    # The continuous arm uses the public checkpointed trainer flag. The
    # boundary-only arm retains False and explicitly scopes its one intervention.
    return (torch.autograd.set_multithreading_enabled(False)
            if arm == "boundary_only" and update == 401 else nullcontext())
