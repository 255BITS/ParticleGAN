"""Exact historical RP1 eager diagnostic setup; not part of the public API.

Run once immediately after constructor, before any state receipt/update.
"""
import torch


def apply_historical_eager_state(trainer):
    for opt in (trainer.opt_g, trainer.opt_d):
        for group in opt.param_groups:
            for p in group["params"]:
                opt.state[p].update(step=torch.zeros((), dtype=torch.float32, device=p.device),
                                    exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
