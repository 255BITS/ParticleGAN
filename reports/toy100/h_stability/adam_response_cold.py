"""Adam response cold screens using frozen runners and existing state capture.

Only observation plumbing is added here. Training variants are declared in JSON;
the pinned H sources, host resources, scoring and budgets remain unchanged.
"""
from contextlib import contextmanager
from copy import deepcopy
import inspect
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "reports/toy100")]
import numpy as np
import torch
import selected_h_remaining
import critic_signal
from adam_response import response_policy
from benchmarks.locked_shared import two_pole
from benchmarks.locked_shared.hosts import (unipolar, mid_scale_identity, cover_leftover,
                                           residual_student, ae_gan_hold, unused_token_hold)

screen = selected_h_remaining.screen
critic_signal.signal_policy = response_policy
original_capture = screen.capture
original_sources = screen.source_hashes
HOSTS = {"two_pole": (two_pole, 80), "unipolar": (unipolar, 400),
         "mid_scale_identity": (mid_scale_identity, mid_scale_identity.GATE_STEPS),
         "cover_leftover": (cover_leftover, cover_leftover.GATE_STEPS),
         "residual_student": (residual_student, residual_student.PROTOCOL['steps']),
         "ae_gan_hold": (ae_gan_hold, ae_gan_hold.STEPS),
         "unused_token_hold": (unused_token_hold, unused_token_hold.STEPS)}


@contextmanager
def capture(directory, task):
    if task not in HOSTS:
        with original_capture(directory, task):
            yield
        return
    module, budget = HOSTS[task]
    original_checkpoint = module.checkpoint

    def checkpoint(step, measure):
        original_checkpoint(step, measure)
        if step != budget:
            return
        values = inspect.currentframe().f_back.f_locals
        payload = dict(task=task, step=step, torch_rng=torch.get_rng_state(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       models={}, optimizers={}, tensors={}, streams={})
        for name, value in values.items():
            if isinstance(value, torch.nn.Module):
                payload["models"][name] = deepcopy(value.state_dict())
            elif isinstance(value, torch.optim.Optimizer):
                payload["optimizers"][name] = deepcopy(value.state_dict())
            elif isinstance(value, torch.Tensor):
                payload["tensors"][name] = value.detach().clone()
            elif isinstance(value, torch.Generator):
                payload["streams"][name] = value.get_state()
        policy = values.get("noise_policy")
        if policy is not None:
            payload["noise_policy"] = deepcopy(policy.__dict__)
        if "ema" in values:
            payload["ema"] = deepcopy(values["ema"].__dict__)
        with torch.no_grad():
            if task == "two_pole":
                payload["samples"] = values["particles"].detach().clone()
            elif task == "unipolar":
                payload["sample_scales"] = [0., 1.]
                payload["samples"] = torch.stack([values["student"].delta(s) for s in (0., 1.)])
            elif task == "mid_scale_identity":
                scales = module._eval_scales(values["arm"])
                payload["sample_scales"] = list(scales)
                payload["samples"] = torch.stack([values["student"].state(s) for s in scales])
            elif task == "cover_leftover":
                payload["sample_scales"] = [-1., 1.]
                payload["samples"] = torch.stack([values["residual"].delta(s) for s in (-1., 1.)])
        torch.save(payload, directory / "final-state.pt")

    with patch.object(module, "checkpoint", checkpoint):
        yield


def sources():
    result = original_sources()
    for path in (Path(__file__), Path(__file__).with_name("adam_response.py")):
        name = str(path.resolve().relative_to(ROOT))
        result[name] = screen.sha(ROOT / name)
    return result


screen.capture = capture
screen.source_hashes = sources

if __name__ == "__main__":
    screen.main()
